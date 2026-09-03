from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import re
import shutil
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.build_league_dataset import clean_column_names
from src.data.build_league_dataset import parse_match_date
from src.data.build_league_dataset import read_raw_csv
from src.experiments.ah_settlement_engine_audit import no_vig_overround
from src.experiments.ah_settlement_engine_audit import settle_side


RAW_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")
REPORT_ROOT = Path("outputs/reports")

ORGANIZATION_REPORT_PATH = REPORT_ROOT / "full_raw_dump_organization_report.md"
MANIFEST_PATH = REPORT_ROOT / "raw_csv_organization_manifest.csv"
BACKFILL_REPORT_PATH = REPORT_ROOT / "full_historical_ah_backfill.md"
COVERAGE_PATH = REPORT_ROOT / "full_historical_ah_coverage_by_league_season.csv"
RAW_VS_PROCESSED_PATH = REPORT_ROOT / "full_historical_ah_raw_vs_processed.csv"
SETTLEMENT_PATH = REPORT_ROOT / "full_historical_ah_settlement_compatibility.csv"
INCLUSION_PATH = REPORT_ROOT / "full_historical_ah_recommended_inclusion_set.csv"

REQUIRED_MATCH_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
CANONICAL_AH = ["AHh", "AvgAHH", "AvgAHA"]
LEGACY_AH = ["BbAHh", "BbAvAHH", "BbAvAHA"]
LEGACY_MAX = ["BbMxAHH", "BbMxAHA"]
CLOSING_AH = ["AHCh", "AvgCAHH", "AvgCAHA"]
TARGET_PRIORITY = [
    "E0",
    "D1",
    "I1",
    "SP1",
    "F1",
    "P1",
    "N1",
    "B1",
    "T1",
    "G1",
    "E1",
    "E2",
    "E3",
    "SC0",
    "D2",
    "I2",
    "SP2",
    "F2",
]


@dataclass(frozen=True)
class Discovery:
    path: Path
    file_size: int
    modified_time: float
    file_hash: str
    rows: int
    columns: list[str]
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    div_value: str
    inferred_league: str
    league_inference_method: str
    league_ambiguous: bool
    season: str
    season_start_year: int | None
    season_end_year: int | None
    season_inference_method: str
    season_ambiguous: bool
    has_required_match_columns: bool
    has_canonical_ah: bool
    has_legacy_ah: bool
    has_legacy_max: bool
    read_status: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_all(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not set(columns).issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    converted = frame[columns].apply(pd.to_numeric, errors="coerce")
    return converted.notna().all(axis=1)


def normalize_league_token(token: object) -> str:
    return str(token).strip().upper().replace("-", "")


def compact_season_from_years(start_year: int, end_year: int | None = None) -> str:
    if end_year is None:
        end_year = start_year + 1
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def season_years_from_compact(token: str) -> tuple[int, int] | None:
    if not re.fullmatch(r"\d{4}", token):
        return None
    start_short = int(token[:2])
    end_short = int(token[2:])
    start_century = 2000 if start_short <= 60 else 1900
    end_century = 2000 if end_short <= 60 else 1900
    start = start_century + start_short
    end = end_century + end_short
    if end <= start:
        end += 100
    if end - start != 1:
        return None
    return start, end


def season_token_from_text(text: str) -> tuple[str, int, int] | None:
    normalized = text.replace("\\", "/")
    match = re.search(r"(?<!\d)((?:19|20)\d{2})[_-]((?:19|20)\d{2})(?!\d)", normalized)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        if end - start == 1:
            return compact_season_from_years(start, end), start, end
    match = re.search(r"(?<!\d)([0-9]{4})(?!\d)", normalized)
    if match:
        years = season_years_from_compact(match.group(1))
        if years is not None:
            start, end = years
            return match.group(1), start, end
    return None


def infer_season_from_dates(first_date: pd.Timestamp | None, last_date: pd.Timestamp | None) -> tuple[str, int, int] | None:
    if first_date is None or pd.isna(first_date):
        return None
    start = int(first_date.year if int(first_date.month) >= 6 else first_date.year - 1)
    end = start + 1
    if last_date is not None and pd.notna(last_date):
        last_year = int(last_date.year)
        if last_year not in {start, end} and abs(last_year - end) > 1:
            return None
    return compact_season_from_years(start, end), start, end


def league_from_path(path: Path) -> tuple[str, str, bool]:
    tokens = [path.stem, *[part for part in path.parts if part not in {"data", "raw", "seasons"}]]
    known = sorted(TARGET_PRIORITY, key=len, reverse=True)
    for token in tokens:
        upper = normalize_league_token(token)
        for league in known:
            if re.search(rf"(^|[^A-Z0-9]){re.escape(league)}([^A-Z0-9]|$)", upper):
                return league, "filename_or_parent", False
    candidates = []
    for token in tokens:
        candidates.extend(re.findall(r"(?<![A-Z0-9])([A-Z]{1,3}\d{1,2})(?![A-Z0-9])", normalize_league_token(token)))
    candidates = sorted(set(candidates), key=lambda item: (-len(item), item))
    if len(candidates) == 1:
        return candidates[0], "filename_or_parent", False
    if len(candidates) > 1:
        return ";".join(candidates), "filename_or_parent", True
    return "", "missing", True


def infer_league(frame: pd.DataFrame, path: Path) -> tuple[str, str, bool, str]:
    if "Div" in frame.columns:
        values = sorted({normalize_league_token(value) for value in frame["Div"].dropna().unique() if str(value).strip()})
        if len(values) == 1:
            return values[0], "Div", False, values[0]
        if len(values) > 1:
            fallback, method, ambiguous = league_from_path(path)
            return fallback or ";".join(values), "Div_inconsistent", True, ";".join(values)
    fallback, method, ambiguous = league_from_path(path)
    return fallback, method, ambiguous, ""


def discover_one(path: Path) -> Discovery:
    stat = path.stat()
    file_hash = sha256_file(path)
    try:
        frame, _, _, _ = read_raw_csv(path)
        frame = clean_column_names(frame)
        columns = list(frame.columns)
        if "Date" in frame.columns:
            frame = parse_match_date(frame)
            dates = frame["Date"].dropna()
            first_date = dates.min() if len(dates) else None
            last_date = dates.max() if len(dates) else None
        else:
            first_date = None
            last_date = None
        league, league_method, league_ambiguous, div_value = infer_league(frame, path)
        file_season = season_token_from_text(path.name)
        parent_season = season_token_from_text("/".join(path.parts[:-1]))
        date_season = infer_season_from_dates(first_date, last_date)
        season_source = file_season or date_season or parent_season
        season_method = "filename" if file_season else ("date_range" if date_season else ("parent_folder" if parent_season else "missing"))
        if season_source is None:
            season, start_year, end_year, season_ambiguous = "", None, None, True
        else:
            season, start_year, end_year = season_source
            season_ambiguous = False
        return Discovery(
            path=path,
            file_size=int(stat.st_size),
            modified_time=float(stat.st_mtime),
            file_hash=file_hash,
            rows=int(len(frame)),
            columns=columns,
            first_date=first_date,
            last_date=last_date,
            div_value=div_value,
            inferred_league=league,
            league_inference_method=league_method,
            league_ambiguous=league_ambiguous,
            season=season,
            season_start_year=start_year,
            season_end_year=end_year,
            season_inference_method=season_method,
            season_ambiguous=season_ambiguous,
            has_required_match_columns=set(REQUIRED_MATCH_COLUMNS).issubset(frame.columns),
            has_canonical_ah=set(CANONICAL_AH).issubset(frame.columns),
            has_legacy_ah=set(LEGACY_AH).issubset(frame.columns),
            has_legacy_max=set(LEGACY_MAX).issubset(frame.columns),
            read_status="ok",
        )
    except Exception as exc:
        league, league_method, league_ambiguous = league_from_path(path)
        season_source = season_token_from_text(path.name) or season_token_from_text("/".join(path.parts[:-1]))
        return Discovery(
            path=path,
            file_size=int(stat.st_size),
            modified_time=float(stat.st_mtime),
            file_hash=file_hash,
            rows=0,
            columns=[],
            first_date=None,
            last_date=None,
            div_value="",
            inferred_league=league,
            league_inference_method=league_method,
            league_ambiguous=league_ambiguous,
            season=season_source[0] if season_source else "",
            season_start_year=season_source[1] if season_source else None,
            season_end_year=season_source[2] if season_source else None,
            season_inference_method="filename_or_parent_after_read_failure" if season_source else "missing",
            season_ambiguous=season_source is None,
            has_required_match_columns=False,
            has_canonical_ah=False,
            has_legacy_ah=False,
            has_legacy_max=False,
            read_status=f"read_failed:{type(exc).__name__}:{exc}",
        )


def discovery_to_row(item: Discovery) -> dict[str, object]:
    return {
        "original_path": str(item.path),
        "file_size": item.file_size,
        "modified_time": item.modified_time,
        "hash": item.file_hash,
        "rows": item.rows,
        "columns_count": len(item.columns),
        "columns": ";".join(item.columns),
        "first_valid_date": "" if item.first_date is None or pd.isna(item.first_date) else item.first_date.date().isoformat(),
        "last_valid_date": "" if item.last_date is None or pd.isna(item.last_date) else item.last_date.date().isoformat(),
        "div_value": item.div_value,
        "league": item.inferred_league,
        "league_inference_method": item.league_inference_method,
        "league_ambiguous": item.league_ambiguous,
        "season": item.season,
        "season_start_year": item.season_start_year,
        "season_end_year": item.season_end_year,
        "season_inference_method": item.season_inference_method,
        "season_ambiguous": item.season_ambiguous,
        "has_required_match_columns": item.has_required_match_columns,
        "has_canonical_ah": item.has_canonical_ah,
        "has_legacy_ah": item.has_legacy_ah,
        "has_legacy_max": item.has_legacy_max,
        "read_status": item.read_status,
    }


def next_variant_path(base_path: Path) -> Path:
    index = 1
    while True:
        candidate = base_path.with_name(f"{base_path.stem}__variant_{index}{base_path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def organize_files(discoveries: list[Discovery]) -> pd.DataFrame:
    rows = []
    for item in discoveries:
        base = discovery_to_row(item)
        safe = (
            item.read_status == "ok"
            and not item.league_ambiguous
            and not item.season_ambiguous
            and bool(item.inferred_league)
            and bool(item.season)
            and item.has_required_match_columns
        )
        if not safe:
            notes = []
            if item.read_status != "ok":
                notes.append(item.read_status)
            if item.league_ambiguous or not item.inferred_league:
                notes.append("ambiguous_league")
            if item.season_ambiguous or not item.season:
                notes.append("ambiguous_season")
            if not item.has_required_match_columns:
                notes.append("missing_required_match_columns")
            rows.append(
                {
                    **base,
                    "canonical_path": "",
                    "action": "skipped_ambiguous_or_incomplete",
                    "selected_for_processing": False,
                    "notes": ";".join(notes),
                }
            )
            continue
        canonical = RAW_ROOT / item.inferred_league / "seasons" / f"{item.inferred_league}_{item.season}.csv"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        try:
            same_path = item.path.resolve() == canonical.resolve()
        except FileNotFoundError:
            same_path = False
        if canonical.exists():
            existing_hash = sha256_file(canonical)
            if same_path:
                action = "already_canonical"
                target = canonical
                notes = "source_is_destination"
            elif existing_hash == item.file_hash:
                action = "duplicate_identical"
                target = canonical
                notes = "destination_hash_matches"
            else:
                target = next_variant_path(canonical)
                shutil.copy2(item.path, target)
                action = "copied_variant"
                notes = "destination_existed_with_different_hash"
        else:
            shutil.copy2(item.path, canonical)
            target = canonical
            action = "copied"
            notes = "copied_to_canonical_structure"
        rows.append(
            {
                **base,
                "canonical_path": str(target),
                "action": action,
                "selected_for_processing": False,
                "notes": notes,
            }
        )
    manifest = pd.DataFrame(rows)
    if len(manifest):
        selectable = manifest[
            manifest["canonical_path"].astype(str).ne("")
            & manifest["league_ambiguous"].eq(False)
            & manifest["season_ambiguous"].eq(False)
            & manifest["has_required_match_columns"].eq(True)
            & manifest["read_status"].eq("ok")
        ].copy()
        selectable["_has_any_ah"] = selectable["has_canonical_ah"].astype(bool) | selectable["has_legacy_ah"].astype(bool)
        selectable["_mtime"] = pd.to_numeric(selectable["modified_time"], errors="coerce").fillna(0)
        selectable = selectable.sort_values(
            ["league", "season", "_has_any_ah", "rows", "has_required_match_columns", "_mtime"],
            ascending=[True, True, False, False, False, False],
        )
        selected_index = selectable.groupby(["league", "season"], dropna=False).head(1).index
        manifest.loc[selected_index, "selected_for_processing"] = True
        manifest = manifest.drop(columns=[column for column in manifest.columns if column.startswith("_")], errors="ignore")
    return manifest


def canonicalize_ah(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mappings = {
        "AHh": "BbAHh",
        "AvgAHH": "BbAvAHH",
        "AvgAHA": "BbAvAHA",
        "MaxAHH": "BbMxAHH",
        "MaxAHA": "BbMxAHA",
    }
    for canonical, legacy in mappings.items():
        if canonical not in out.columns and legacy in out.columns:
            out[canonical] = out[legacy]
        elif canonical in out.columns and legacy in out.columns:
            current = pd.to_numeric(out[canonical], errors="coerce")
            legacy_values = pd.to_numeric(out[legacy], errors="coerce")
            out[canonical] = current.where(current.notna(), legacy_values)
    return out


def ah_source_schema(frame: pd.DataFrame) -> str:
    canonical_rows = numeric_all(frame, CANONICAL_AH)
    legacy_rows = numeric_all(frame, LEGACY_AH)
    if canonical_rows.any():
        return "canonical"
    if legacy_rows.any():
        return "legacy"
    return "missing"


def process_selected_files(manifest: pd.DataFrame) -> pd.DataFrame:
    selected = manifest[manifest["selected_for_processing"].eq(True)].copy()
    process_rows = []
    for league, group in selected.groupby("league"):
        frames = []
        for _, row in group.sort_values("season").iterrows():
            source = Path(row["canonical_path"])
            try:
                frame, _, read_mode, skipped = read_raw_csv(source)
                frame = clean_column_names(frame)
                frame = parse_match_date(frame)
                before = len(frame)
                frame = frame.dropna(subset=REQUIRED_MATCH_COLUMNS).copy()
                schema_before = ah_source_schema(frame)
                legacy_recoverable = numeric_all(frame, LEGACY_AH)
                canonical_before = numeric_all(frame, CANONICAL_AH)
                frame = canonicalize_ah(frame)
                canonical_after = numeric_all(frame, CANONICAL_AH)
                frame["league"] = league
                frame["season_start_year"] = int(row["season_start_year"])
                frame["season_end_year"] = int(row["season_end_year"])
                frame["source_raw_file"] = str(source)
                frame["ah_source_schema"] = schema_before
                frames.append(frame)
                process_rows.append(
                    {
                        "league": league,
                        "season": row["season"],
                        "season_start_year": int(row["season_start_year"]),
                        "season_end_year": int(row["season_end_year"]),
                        "source_raw_file": str(source),
                        "input_rows": int(before),
                        "processed_rows": int(len(frame)),
                        "required_rows_removed": int(before - len(frame)),
                        "read_mode": read_mode,
                        "skipped_bad_lines": int(skipped),
                        "ah_source_schema": schema_before,
                        "canonical_ah_rows_before_mapping": int(canonical_before.sum()),
                        "legacy_ah_recoverable_rows": int((~canonical_before & legacy_recoverable).sum()),
                        "canonical_ah_rows_after_mapping": int(canonical_after.sum()),
                        "status": "processed",
                    }
                )
            except Exception as exc:
                process_rows.append(
                    {
                        "league": league,
                        "season": row["season"],
                        "season_start_year": row["season_start_year"],
                        "season_end_year": row["season_end_year"],
                        "source_raw_file": str(source),
                        "input_rows": 0,
                        "processed_rows": 0,
                        "required_rows_removed": 0,
                        "read_mode": "",
                        "skipped_bad_lines": 0,
                        "ah_source_schema": "missing",
                        "canonical_ah_rows_before_mapping": 0,
                        "legacy_ah_recoverable_rows": 0,
                        "canonical_ah_rows_after_mapping": 0,
                        "status": f"failed:{type(exc).__name__}:{exc}",
                    }
                )
        if frames:
            output = pd.concat(frames, ignore_index=True, sort=False)
            output = output.sort_values(["Date", "HomeTeam", "AwayTeam"]).drop_duplicates(["Date", "HomeTeam", "AwayTeam"], keep="first").reset_index(drop=True)
            out_path = PROCESSED_ROOT / league / f"{league}_matches.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(out_path, index=False)
    return pd.DataFrame(process_rows)


def load_processed_files() -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(PROCESSED_ROOT.glob("*/*_matches.csv")):
        league = path.parent.name
        frame = pd.read_csv(path, low_memory=False)
        if "Date" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        for column in ["season_start_year", "season_end_year", "FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames[league] = frame
    return frames


def coverage_audit(frames: dict[str, pd.DataFrame], process_log: pd.DataFrame) -> pd.DataFrame:
    process_lookup = {}
    if len(process_log):
        for _, row in process_log.iterrows():
            process_lookup[(row["league"], int(row["season_start_year"]))] = row
    rows = []
    for league, frame in frames.items():
        if "season_start_year" not in frame.columns:
            continue
        for season_start, group in frame.groupby("season_start_year", dropna=True):
            season_start = int(season_start)
            proc = process_lookup.get((league, season_start))
            scores = group[["FTHG", "FTAG"]].notna().all(axis=1) if {"FTHG", "FTAG"}.issubset(group.columns) else pd.Series(False, index=group.index)
            ah_rows = numeric_all(group, CANONICAL_AH)
            adjusted = group["FTHG"] - group["FTAG"] + group["AHh"] if {"FTHG", "FTAG", "AHh"}.issubset(group.columns) else pd.Series(np.nan, index=group.index)
            one_x_two = numeric_all(group, ["AvgH", "AvgD", "AvgA"])
            ftr = group["FTR"].notna() if "FTR" in group.columns else pd.Series(False, index=group.index)
            closing = numeric_all(group, CLOSING_AH) if set(CLOSING_AH).issubset(group.columns) else pd.Series(False, index=group.index)
            rows.append(
                {
                    "league": league,
                    "season": compact_season_from_years(season_start, season_start + 1),
                    "season_start_year": season_start,
                    "season_end_year": season_start + 1,
                    "matches": int(len(group)),
                    "rows_with_valid_scores": int(scores.sum()),
                    "one_x_two_odds_coverage": float(one_x_two.mean()) if len(group) else 0.0,
                    "ah_odds_coverage": float(ah_rows.mean()) if len(group) else 0.0,
                    "usable_ah_target_rows": int((ah_rows & scores & adjusted.notna() & adjusted.ne(0)).sum()),
                    "usable_1x2_target_rows": int((one_x_two & ftr).sum()),
                    "big_home_favourite_ah_rows": int((ah_rows & group["AHh"].le(-1.0)).sum()) if "AHh" in group.columns else 0,
                    "canonical_ah_rows": int(ah_rows.sum()),
                    "legacy_ah_recovered_rows": int(proc["legacy_ah_recoverable_rows"]) if proc is not None and "legacy_ah_recoverable_rows" in proc else 0,
                    "missing_ah_rows": int((~ah_rows).sum()),
                    "closing_ah_coverage_diagnostic_only": float(closing.mean()) if len(group) and set(CLOSING_AH).issubset(group.columns) else 0.0,
                    "source_raw_files": ";".join(sorted(group["source_raw_file"].dropna().astype(str).unique().tolist())) if "source_raw_file" in group.columns else "",
                }
            )
    return pd.DataFrame(rows).sort_values(["league", "season_start_year"]).reset_index(drop=True)


def split_handicap(handicap: float) -> tuple[float, ...]:
    scaled = float(handicap) * 4.0
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-8:
        return (float(handicap),)
    if rounded % 2 == 0:
        return (float(handicap),)
    lower = math.floor(float(handicap) * 2.0) / 2.0
    upper = math.ceil(float(handicap) * 2.0) / 2.0
    return (lower, upper)


def settlement_audit(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    labels = {"full_win", "half_win", "push", "half_loss", "full_loss"}
    rows = []
    for league, frame in frames.items():
        if not {"season_start_year", "FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA"}.issubset(frame.columns):
            continue
        for season_start, group in frame.groupby("season_start_year", dropna=True):
            group = group.copy()
            margin = group["FTHG"] - group["FTAG"]
            home = [settle_side(m, h, o) for m, h, o in zip(margin, group["AHh"], group["AvgAHH"])]
            away = [settle_side(-m, -h if pd.notna(h) else np.nan, o) for m, h, o in zip(margin, group["AHh"], group["AvgAHA"])]
            group["home_label"] = [item.label for item in home]
            group["away_label"] = [item.label for item in away]
            group["home_profit"] = [item.profit for item in home]
            group["away_profit"] = [item.profit for item in away]
            valid = group["home_label"].isin(labels) & group["away_label"].isin(labels)
            adjusted = margin + group["AHh"]
            binary_previous = np.where(adjusted > 0, 1.0, np.where(adjusted < 0, 0.0, np.nan))
            settlement_cover = np.where(group["home_label"].isin(["full_win", "half_win"]), 1.0, np.where(group["home_label"].isin(["half_loss", "full_loss"]), 0.0, np.nan))
            comparable = valid & pd.Series(binary_previous, index=group.index).notna() & pd.Series(settlement_cover, index=group.index).notna()
            mismatches = int((pd.Series(binary_previous, index=group.index)[comparable].to_numpy() != pd.Series(settlement_cover, index=group.index)[comparable].to_numpy()).sum())
            ah_valid = numeric_all(group, CANONICAL_AH) & group[["FTHG", "FTAG"]].notna().all(axis=1)
            half_line = group["AHh"].apply(lambda value: pd.notna(value) and abs(float(value) * 2 - round(float(value) * 2)) < 1e-8)
            quarter_line = group["AHh"].apply(lambda value: pd.notna(value) and abs(float(value) * 4 - round(float(value) * 4)) < 1e-8 and round(float(value) * 4) % 2 != 0)
            overround = no_vig_overround(group["AvgAHH"], group["AvgAHA"])
            valid_group = group[valid].copy()
            rows.append(
                {
                    "league": league,
                    "season": compact_season_from_years(int(season_start), int(season_start) + 1),
                    "season_start_year": int(season_start),
                    "season_end_year": int(season_start) + 1,
                    "matches": int(len(group)),
                    "valid_ah_settlement_rows": int(valid.sum()),
                    "impossible_settlement_rows": int((group["home_label"].eq("impossible") | group["away_label"].eq("impossible")).sum()),
                    "binary_target_mismatch_rows": mismatches,
                    "settlement_compatible": bool(mismatches == 0 and int((group["home_label"].eq("impossible") | group["away_label"].eq("impossible")).sum()) == 0),
                    "push_rate": float(group["home_label"].eq("push").mean()) if len(group) else 0.0,
                    "half_line_rate": float((half_line & ah_valid).sum() / ah_valid.sum()) if ah_valid.sum() else 0.0,
                    "quarter_line_rate": float((quarter_line & ah_valid).sum() / ah_valid.sum()) if ah_valid.sum() else 0.0,
                    "partial_home_cover_rate": float(group["home_label"].isin(["half_win"]).sum() / valid.sum()) if valid.sum() else 0.0,
                    "partial_home_no_cover_rate": float(group["home_label"].isin(["half_loss"]).sum() / valid.sum()) if valid.sum() else 0.0,
                    "average_home_ah_profit_using_AvgAHH": float(valid_group["home_profit"].mean()) if len(valid_group) else np.nan,
                    "average_away_ah_profit_using_AvgAHA": float(valid_group["away_profit"].mean()) if len(valid_group) else np.nan,
                    "bookmaker_overround_estimate": float(overround[valid].mean()) if valid.sum() else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["league", "season_start_year"]).reset_index(drop=True)


def raw_vs_processed(manifest: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    raw = (
        manifest[manifest["read_status"].eq("ok")]
        .groupby(["league", "season"], dropna=False)
        .agg(
            local_raw_files=("original_path", "count"),
            organized_files=("canonical_path", lambda values: int(pd.Series(values).astype(str).ne("").sum())),
            selected_files=("selected_for_processing", "sum"),
            raw_rows=("rows", "max"),
            raw_has_canonical_ah=("has_canonical_ah", "max"),
            raw_has_legacy_ah=("has_legacy_ah", "max"),
            ambiguous_files=("league_ambiguous", "sum"),
        )
        .reset_index()
    )
    return raw.merge(
        coverage[
            [
                "league",
                "season",
                "matches",
                "canonical_ah_rows",
                "legacy_ah_recovered_rows",
                "missing_ah_rows",
                "ah_odds_coverage",
            ]
        ],
        on=["league", "season"],
        how="left",
    ).sort_values(["league", "season"]).reset_index(drop=True)


def inclusion_set(coverage: pd.DataFrame, settlement: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ambiguous_leagues = set(manifest[manifest["league_ambiguous"].eq(True)]["league"].dropna().astype(str).tolist())
    for league, group in coverage.groupby("league"):
        settle = settlement[settlement["league"].eq(league)]
        usable_seasons = int((group["usable_ah_target_rows"] > 0).sum())
        usable_rows = int(group["usable_ah_target_rows"].sum())
        matches = int(group["matches"].sum())
        ah_rows = int(group["canonical_ah_rows"].sum())
        compatible = bool(len(settle) > 0 and settle["settlement_compatible"].all() and int(settle["binary_target_mismatch_rows"].sum()) == 0)
        major_ambiguity = league in ambiguous_leagues
        if major_ambiguity:
            classification = "ambiguous_raw_files"
        elif matches < 100:
            classification = "insufficient_data"
        elif ah_rows == 0:
            classification = "no_ah_odds"
        elif usable_seasons >= 3 and usable_rows >= 500 and compatible:
            first = int(group[group["usable_ah_target_rows"] > 0]["season_start_year"].min())
            classification = "ah_ready_full_historical" if first <= 2010 else "ah_ready_recent_only"
        elif usable_rows > 0 and compatible:
            classification = "ah_partial"
        else:
            classification = "reject_for_now"
        rows.append(
            {
                "league": league,
                "classification": classification,
                "seasons": int(group["season"].nunique()),
                "usable_ah_seasons": usable_seasons,
                "matches": matches,
                "usable_ah_rows": usable_rows,
                "canonical_ah_rows": ah_rows,
                "legacy_ah_recovered_rows": int(group["legacy_ah_recovered_rows"].sum()),
                "settlement_compatible": compatible,
                "binary_target_mismatch_rows": int(settle["binary_target_mismatch_rows"].sum()) if len(settle) else 0,
                "ambiguous_raw_files": major_ambiguity,
            }
        )
    return pd.DataFrame(rows).sort_values(["classification", "league"]).reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def classify_run(manifest: pd.DataFrame, process_log: pd.DataFrame, coverage: pd.DataFrame) -> str:
    if manifest.empty or int(manifest["selected_for_processing"].sum()) == 0:
        return "organization_failed"
    ambiguous = int((manifest["action"].eq("skipped_ambiguous_or_incomplete")).sum())
    failures = int(process_log["status"].astype(str).str.startswith("failed").sum()) if len(process_log) else 0
    selected = int(manifest["selected_for_processing"].sum())
    processed = int(process_log["status"].eq("processed").sum()) if len(process_log) else 0
    remaining_no_ah = int((coverage["canonical_ah_rows"].eq(0) & coverage["legacy_ah_recovered_rows"].gt(0)).sum()) if len(coverage) else 0
    if failures or processed < selected:
        return "backfill_partial"
    if remaining_no_ah:
        return "backfill_partial"
    if ambiguous:
        return "backfill_completed_with_ambiguous_files"
    return "backfill_completed"


def write_reports(
    manifest: pd.DataFrame,
    process_log: pd.DataFrame,
    coverage: pd.DataFrame,
    raw_processed: pd.DataFrame,
    settlement: pd.DataFrame,
    inclusion: pd.DataFrame,
    classification: str,
) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    action_summary = manifest["action"].value_counts(dropna=False).rename_axis("action").reset_index(name="files") if len(manifest) else pd.DataFrame()
    league_summary = (
        coverage.groupby("league")
        .agg(
            seasons=("season", "nunique"),
            matches=("matches", "sum"),
            canonical_ah_rows=("canonical_ah_rows", "sum"),
            legacy_ah_recovered_rows=("legacy_ah_recovered_rows", "sum"),
            missing_ah_rows=("missing_ah_rows", "sum"),
        )
        .reset_index()
        if len(coverage)
        else pd.DataFrame()
    )
    lines = [
        "# Full Raw Dump Organization Report",
        "",
        f"Final classification: `{classification}`",
        "",
        "Scope: local football-data CSV files under `data/raw`. No predictive models, betting strategies, value searches, threshold optimization, Transfermarkt data, player features, lineups, external APIs, scraping, deletion, or confirmed edge claims were used.",
        "",
        "Raw CSV contents were preserved unchanged. Safe files were copied into `data/raw/<LEAGUE>/seasons/<LEAGUE>_<SEASON>.csv`; conflicting files were retained as `__variant_N` files.",
        "",
        "## Organization Actions",
        "",
        markdown_table(action_summary, ["action", "files"], max_rows=40),
        "",
        "## Processed League Summary",
        "",
        markdown_table(league_summary, ["league", "seasons", "matches", "canonical_ah_rows", "legacy_ah_recovered_rows", "missing_ah_rows"], max_rows=80),
        "",
        "## Recommended Inclusion",
        "",
        markdown_table(inclusion, ["league", "classification", "usable_ah_seasons", "usable_ah_rows", "settlement_compatible", "ambiguous_raw_files"], max_rows=100),
        "",
        "## Output Files",
        "",
        f"- `{MANIFEST_PATH}`",
        f"- `{BACKFILL_REPORT_PATH}`",
        f"- `{COVERAGE_PATH}`",
        f"- `{RAW_VS_PROCESSED_PATH}`",
        f"- `{SETTLEMENT_PATH}`",
        f"- `{INCLUSION_PATH}`",
        "",
    ]
    ORGANIZATION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    backfill_lines = [
        "# Full Historical AH Backfill",
        "",
        f"Final classification: `{classification}`",
        "",
        "Legacy AH mappings applied only when canonical values were missing: `BbAHh -> AHh`, `BbAvAHH -> AvgAHH`, `BbAvAHA -> AvgAHA`, `BbMxAHH -> MaxAHH`, `BbMxAHA -> MaxAHA`.",
        "",
        "## Processing Log Summary",
        "",
        markdown_table(process_log.groupby(["league", "status"]).size().reset_index(name="season_files"), ["league", "status", "season_files"], max_rows=120) if len(process_log) else "_No processing rows._",
        "",
        "## AH Coverage By League",
        "",
        markdown_table(league_summary, ["league", "seasons", "matches", "canonical_ah_rows", "legacy_ah_recovered_rows", "missing_ah_rows"], max_rows=80),
        "",
        "No confirmed edge is claimed. No modeling was run.",
        "",
    ]
    BACKFILL_REPORT_PATH.write_text("\n".join(backfill_lines), encoding="utf-8")


def main() -> None:
    initial_files = sorted(RAW_ROOT.rglob("*.csv"))
    discoveries = [discover_one(path) for path in initial_files]
    manifest = organize_files(discoveries)
    process_log = process_selected_files(manifest)
    processed_frames = load_processed_files()
    coverage = coverage_audit(processed_frames, process_log)
    settlement = settlement_audit(processed_frames)
    raw_processed = raw_vs_processed(manifest, coverage)
    inclusion = inclusion_set(coverage, settlement, manifest)
    classification = classify_run(manifest, process_log, coverage)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_columns = [
        "original_path",
        "canonical_path",
        "action",
        "league",
        "season",
        "hash",
        "rows",
        "columns_count",
        "has_required_match_columns",
        "has_canonical_ah",
        "has_legacy_ah",
        "selected_for_processing",
        "notes",
    ]
    extra_columns = [column for column in manifest.columns if column not in manifest_columns]
    manifest[manifest_columns + extra_columns].to_csv(MANIFEST_PATH, index=False)
    coverage.to_csv(COVERAGE_PATH, index=False)
    raw_processed.to_csv(RAW_VS_PROCESSED_PATH, index=False)
    settlement.to_csv(SETTLEMENT_PATH, index=False)
    inclusion.to_csv(INCLUSION_PATH, index=False)
    write_reports(manifest, process_log, coverage, raw_processed, settlement, inclusion, classification)
    print(
        {
            "discovered_csv_files": len(initial_files),
            "manifest_rows": len(manifest),
            "selected_files": int(manifest["selected_for_processing"].sum()) if len(manifest) else 0,
            "processed_season_files": int(process_log["status"].eq("processed").sum()) if len(process_log) else 0,
            "processed_leagues": len(processed_frames),
            "coverage_rows": len(coverage),
            "settlement_rows": len(settlement),
            "classification": classification,
        }
    )


if __name__ == "__main__":
    main()
