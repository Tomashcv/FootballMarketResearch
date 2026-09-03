"""Read-only V4 Phase 1 market-data inventory and contract audit."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import re
import warnings
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .market_contract import ColumnContract, classify_column


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "outputs/reports/v4_price_discovery"
FD_ARCHIVE = ROOT / "data/raw_external/football_data_full_archive/football_data_full_archive.zip"
BTB_ARCHIVE = ROOT / "data/raw_external/beat_the_bookie_1x2/beat_the_bookie_1x2_archive.zip"

SCOPE_LEAGUES = ("E0", "SP1", "D1", "I1", "F1", "B1", "G1", "N1", "P1", "SC0", "T1")
EXCLUDED_LEAGUES = ("E1", "E2", "E3")
FOOTIQO_LEAGUES = {
    "premier_league_sample": "E0",
    "spain_laliga": "SP1",
    "germany_bundesliga": "D1",
    "italy_serie_a": "I1",
    "france_ligue_1": "F1",
}
BTB_LEAGUES = {
    "England: Premier League": "E0",
    "Spain: Primera Division": "SP1",
    "Germany: Bundesliga": "D1",
    "Italy: Serie A": "I1",
    "France: Ligue 1": "F1",
    "Belgium: First Division A": "B1",
    "Netherlands: Eredivisie": "N1",
    "Portugal: Primeira Liga": "P1",
    "Scotland: Premiership": "SC0",
    "Turkey: Super Lig": "T1",
    "Greece: Super League": "G1",
}

INVENTORY_COLUMNS = [
    "source", "source_file", "archive_member", "canonical_source_copy", "league", "season",
    "column", "market", "role", "selection", "bookmaker", "line", "timing_classification",
    "timing_evidence_id", "feature_policy", "match_rows", "non_null_values", "valid_odds_values",
    "invalid_odds_le_1", "valid_odds_coverage", "min_value", "max_value", "schema_signature",
]

ANOMALY_COLUMNS = [
    "anomaly_type", "severity", "source", "source_file", "league", "season", "column",
    "affected_rows", "details",
]


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip().lstrip("\ufeff").replace("ï»¿", "") for c in frame.columns]
    return frame.loc[:, [bool(c) and not c.startswith("Unnamed:") for c in frame.columns]]


def parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    iso = text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", na=False)
    output = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        output.loc[iso] = pd.to_datetime(text.loc[iso], errors="coerce", yearfirst=True)
        output.loc[~iso] = pd.to_datetime(text.loc[~iso], errors="coerce", dayfirst=True)
    return output


def season_label(date: pd.Timestamp) -> str:
    if pd.isna(date):
        return "unknown"
    start = int(date.year if date.month >= 7 else date.year - 1)
    return f"{start}/{start + 1}"


def normalize_team(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def schema_signature(columns: Iterable[str]) -> str:
    value = "|".join(str(c) for c in columns)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preferred_raw_file(path: Path, league: str) -> bool:
    return bool(re.fullmatch(fr"{re.escape(league)}_\d{{4}}\.csv", path.name)) and path.parent.name == "seasons"


def season_from_raw_filename(path: Path, league: str) -> str | None:
    name = path.stem
    short = re.fullmatch(fr"{re.escape(league)}_(\d{{2}})(\d{{2}})(?:__variant_\d+)?", name)
    if short:
        start2, end2 = map(int, short.groups())
        if (start2 + 1) % 100 != end2:
            return None
        start = 1900 + start2 if start2 >= 90 else 2000 + start2
        return f"{start}/{start + 1}"
    long = re.fullmatch(fr"{re.escape(league)}_(\d{{4}})_(\d{{4}})", name)
    if long and int(long.group(2)) == int(long.group(1)) + 1:
        return f"{long.group(1)}/{long.group(2)}"
    end_year = re.fullmatch(fr"{re.escape(league)}_(20\d{{2}})", name)
    if end_year:
        end = int(end_year.group(1))
        return f"{end - 1}/{end}"
    return None


def read_csv_resilient(source: Path | bytes) -> tuple[pd.DataFrame, int]:
    """Read legacy CSVs while explicitly counting rows wider/shorter than the header."""

    try:
        if isinstance(source, Path):
            return clean_columns(pd.read_csv(source, low_memory=False, encoding_errors="replace")), 0
        return clean_columns(pd.read_csv(io.BytesIO(source), low_memory=False, encoding_errors="replace")), 0
    except pd.errors.ParserError:
        text = source.read_text(encoding="utf-8-sig", errors="replace") if isinstance(source, Path) else source.decode("utf-8-sig", "replace")
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
        rows: list[list[str]] = []
        malformed = 0
        for row in reader:
            if len(row) != len(header):
                malformed += 1
            rows.append((row + [""] * len(header))[: len(header)])
        return clean_columns(pd.DataFrame(rows, columns=header)), malformed


def archive_season(member: str) -> str:
    match = re.search(r"/(\d{2})_(\d{2})/", member)
    if not match:
        return "unknown"
    start2 = int(match.group(1))
    start = 1900 + start2 if start2 >= 90 else 2000 + start2
    return f"{start}/{start + 1}"


def inventory_for_group(
    frame: pd.DataFrame,
    source: str,
    source_file: str,
    member: str,
    league: str,
    season: str,
    canonical: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []
    signature = schema_signature(frame.columns)
    for column in frame.columns:
        contract = classify_column(source, column, member)
        if contract is None:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        non_null = int(numeric.notna().sum())
        valid = int((numeric > 1).sum()) if contract.role == "odds" else 0
        invalid = int((numeric.notna() & (numeric <= 1)).sum()) if contract.role == "odds" else 0
        rows.append(
            {
                "source": source,
                "source_file": source_file,
                "archive_member": member,
                "canonical_source_copy": canonical,
                "league": league,
                "season": season,
                "column": column,
                **asdict(contract),
                "match_rows": len(frame),
                "non_null_values": non_null,
                "valid_odds_values": valid,
                "invalid_odds_le_1": invalid,
                "valid_odds_coverage": valid / len(frame) if len(frame) and contract.role == "odds" else np.nan,
                "min_value": float(numeric.min()) if non_null else np.nan,
                "max_value": float(numeric.max()) if non_null else np.nan,
                "schema_signature": signature,
            }
        )
        if invalid:
            anomalies.append(anomaly("odds_le_1", "error", source, source_file, league, season, column, invalid, "Numeric odds at or below 1 are invalid and excluded from coverage."))
        if contract.role == "line" and contract.market == "Asian Handicap":
            bad = numeric.notna() & ((numeric.abs() > 10) | ((numeric * 4).round().sub(numeric * 4).abs() > 1e-8))
            if bad.any():
                anomalies.append(anomaly("impossible_handicap_line", "error", source, source_file, league, season, column, int(bad.sum()), "AH line outside +/-10 or not on a quarter-goal increment."))
        if contract.market == "Over/Under" and contract.line is not None and (contract.line <= 0 or contract.line > 10 or abs(contract.line * 4 - round(contract.line * 4)) > 1e-8):
            anomalies.append(anomaly("impossible_total_line", "error", source, source_file, league, season, column, len(frame), f"Encoded total line {contract.line} is outside the accepted positive <=10 quarter-goal grid."))
    return rows, anomalies


def anomaly(kind: str, severity: str, source: str, source_file: str, league: str = "", season: str = "", column: str = "", affected_rows: int = 0, details: str = "") -> dict[str, object]:
    return {
        "anomaly_type": kind,
        "severity": severity,
        "source": source,
        "source_file": source_file,
        "league": league,
        "season": season,
        "column": column,
        "affected_rows": affected_rows,
        "details": details,
    }


def fixture_anomalies(frame: pd.DataFrame, source: str, source_file: str, league: str, season: str) -> list[dict[str, object]]:
    if not {"Date", "HomeTeam", "AwayTeam"}.issubset(frame.columns):
        return []
    dates = parse_dates(frame["Date"])
    keys = dates.dt.strftime("%Y-%m-%d").fillna("") + "|" + frame["HomeTeam"].map(normalize_team) + "|" + frame["AwayTeam"].map(normalize_team)
    count = int(keys.duplicated(keep=False).sum())
    if not count:
        return []
    return [anomaly("duplicated_logical_fixture_within_file", "error", source, source_file, league, season, affected_rows=count, details="Duplicate date/home/away logical keys within one physical source file.")]


def scan_football_data_raw() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, int], list[dict[str, object]]]:
    inventory: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []
    canonical_groups: list[dict[str, object]] = []
    fixture_rows: list[dict[str, object]] = []
    stats = {"files": 0, "rows": 0}
    hashes: dict[str, list[str]] = defaultdict(list)
    for path in sorted((ROOT / "data/raw").rglob("*.csv")):
        path_text = str(path.relative_to(ROOT))
        path_parts = set(path.parts)
        candidate = next((league for league in SCOPE_LEAGUES if league in path_parts or path.name.startswith(league)), None)
        if candidate is None or any(excluded in path_parts or path.name.startswith(excluded) for excluded in EXCLUDED_LEAGUES):
            continue
        try:
            frame, malformed = read_csv_resilient(path)
        except Exception as exc:
            anomalies.append(anomaly("source_read_error", "error", "football_data", path_text, details=f"{type(exc).__name__}: {exc}"))
            continue
        if malformed:
            anomalies.append(anomaly("malformed_legacy_csv_rows", "error", "football_data", path_text, affected_rows=malformed, details="Rows did not match header width; header-width fields were retained and these rows remain anomalous."))
        stats["files"] += 1
        stats["rows"] += len(frame)
        hashes[file_sha256(path)].append(path_text)
        if "Div" not in frame.columns:
            anomalies.append(anomaly("missing_div_column", "error", "football_data", path_text, details="Cannot verify league scope."))
            continue
        div = frame["Div"].astype("string").str.strip().str.replace("ï»¿", "", regex=False)
        dates = parse_dates(frame["Date"]) if "Date" in frame else pd.Series(pd.NaT, index=frame.index)
        filename_season = season_from_raw_filename(path, candidate)
        seasons = pd.Series(filename_season, index=frame.index) if filename_season else dates.map(season_label)
        for league in SCOPE_LEAGUES:
            league_mask = div.eq(league)
            if not league_mask.any():
                continue
            for season in sorted(seasons[league_mask].unique()):
                group = frame.loc[league_mask & seasons.eq(season)].copy()
                canonical = preferred_raw_file(path, league)
                inv, bad = inventory_for_group(group, "football_data", path_text, "", league, season, canonical)
                inventory.extend(inv)
                anomalies.extend(bad)
                anomalies.extend(fixture_anomalies(group, "football_data", path_text, league, season))
                if canonical:
                    canonical_groups.append({"source": "football_data", "source_file": path_text, "league": league, "season": season, "frame": group})
                if {"Date", "HomeTeam", "AwayTeam"}.issubset(group.columns):
                    for idx in group.index:
                        fixture_rows.append(
                            {
                                "league": league,
                                "date": dates.loc[idx].strftime("%Y-%m-%d") if pd.notna(dates.loc[idx]) else "",
                                "home": normalize_team(group.loc[idx, "HomeTeam"]),
                                "away": normalize_team(group.loc[idx, "AwayTeam"]),
                                "home_goals": pd.to_numeric(group.loc[idx, "FTHG"], errors="coerce") if "FTHG" in group else np.nan,
                                "away_goals": pd.to_numeric(group.loc[idx, "FTAG"], errors="coerce") if "FTAG" in group else np.nan,
                                "source_file": path_text,
                            }
                        )
    for digest, paths in hashes.items():
        if len(paths) > 1:
            anomalies.append(anomaly("duplicate_source_copy", "warning", "football_data", "|".join(paths), affected_rows=len(paths), details=f"Byte-identical source copies; sha256={digest}."))
    return inventory, anomalies, canonical_groups, stats, fixture_rows


def scan_football_data_archive(existing_seasons: set[tuple[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    inventory: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []
    canonical_groups: list[dict[str, object]] = []
    stats = {"files": 0, "rows": 0}
    with zipfile.ZipFile(FD_ARCHIVE) as archive:
        for member in sorted(archive.namelist()):
            match = re.fullmatch(r"main/main/\d{2}_\d{2}/([A-Z0-9]+)\.csv", member)
            if not match or match.group(1) not in SCOPE_LEAGUES:
                continue
            league = match.group(1)
            season = archive_season(member)
            source_file = str(FD_ARCHIVE.relative_to(ROOT))
            try:
                frame, malformed = read_csv_resilient(archive.read(member))
            except Exception as exc:
                anomalies.append(anomaly("archive_member_read_error", "error", "football_data", source_file, league, season, affected_rows=0, details=f"{member}: {type(exc).__name__}: {exc}"))
                continue
            if malformed:
                anomalies.append(anomaly("malformed_legacy_csv_rows", "error", "football_data", f"{source_file}::{member}", league, season, affected_rows=malformed, details="Rows did not match header width; header-width fields were retained and these rows remain anomalous."))
            if "Div" in frame:
                frame = frame[frame["Div"].astype("string").str.strip().eq(league)].copy()
            stats["files"] += 1
            stats["rows"] += len(frame)
            canonical = (league, season) not in existing_seasons
            inv, bad = inventory_for_group(frame, "football_data", source_file, member, league, season, canonical)
            inventory.extend(inv)
            anomalies.extend(bad)
            anomalies.extend(fixture_anomalies(frame, "football_data", f"{source_file}::{member}", league, season))
            if canonical:
                canonical_groups.append({"source": "football_data", "source_file": f"{source_file}::{member}", "league": league, "season": season, "frame": frame})
    return inventory, anomalies, canonical_groups, stats


def scan_footiqo() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    inventory: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []
    stats = {"files": 0, "rows": 0}
    for path in sorted((ROOT / "data/raw_external/footiqo_manual/leagues").glob("*/*Odds*.csv")):
        source_file = str(path.relative_to(ROOT))
        frame = clean_columns(pd.read_csv(path, low_memory=False, encoding_errors="replace"))
        slug = next((key for key in FOOTIQO_LEAGUES if key in str(path.parent)), "")
        league = FOOTIQO_LEAGUES.get(slug, "")
        if not league:
            anomalies.append(anomaly("unmapped_footiqo_league", "error", "footiqo", source_file))
            continue
        dates = parse_dates(frame["matchDate"])
        seasons = frame["Season"].astype("string").str.strip() if "Season" in frame else dates.map(season_label)
        stats["files"] += 1
        stats["rows"] += len(frame)
        for season in sorted(seasons.unique()):
            group = frame[seasons.eq(season)].copy()
            inv, bad = inventory_for_group(group, "footiqo", source_file, "", league, season, True)
            inventory.extend(inv)
            anomalies.extend(bad)
            if "id" in group and group["id"].duplicated().any():
                anomalies.append(anomaly("duplicated_logical_fixture_within_file", "error", "footiqo", source_file, league, season, affected_rows=int(group["id"].duplicated(keep=False).sum()), details="Duplicate Footiqo source match ids."))
            groups.append({"source": "footiqo", "source_file": source_file, "league": league, "season": season, "frame": group})
    return inventory, anomalies, groups, stats


def _btb_match_map(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member) as raw, gzip.GzipFile(fileobj=raw) as zipped:
        frame = pd.read_csv(zipped, low_memory=False, encoding_errors="replace")
    frame.columns = [str(c).strip() for c in frame.columns]
    frame["league_code"] = frame["league"].astype("string").str.strip().map(BTB_LEAGUES)
    dates = parse_dates(frame["match_datetime"])
    frame["season"] = dates.map(season_label)
    return frame[["match_id", "league_code", "season"]]


def scan_beat_the_bookie() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    inventory: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []
    stats = {"files": 0, "rows": 0}
    source_file = str(BTB_ARCHIVE.relative_to(ROOT))
    with zipfile.ZipFile(BTB_ARCHIVE) as archive:
        # The compact closing_odds table can be audited by league-season.  Its
        # filename is deliberately not accepted as timing proof.
        with archive.open("closing_odds.csv.gz") as raw, gzip.GzipFile(fileobj=raw) as zipped:
            closing = clean_columns(pd.read_csv(zipped, low_memory=False, encoding_errors="replace"))
        closing["league_code"] = closing["league"].map(BTB_LEAGUES)
        dates = parse_dates(closing["match_date"])
        closing["season"] = dates.map(season_label)
        scoped = closing[closing["league_code"].isin(SCOPE_LEAGUES)].copy()
        stats["files"] += 1
        stats["rows"] += len(closing)
        for (league, season), group in scoped.groupby(["league_code", "season"], sort=True):
            inv, bad = inventory_for_group(group, "beat_the_bookie", source_file, "closing_odds.csv.gz", str(league), str(season), True)
            inventory.extend(inv)
            anomalies.extend(bad)
            groups.append({"source": "beat_the_bookie", "source_file": f"{source_file}::closing_odds.csv.gz", "league": league, "season": season, "frame": group})

        # Series columns are individually inventoried, but the local metadata
        # does not define bN identities or the 0..71 observation timestamps.
        # We therefore report schema presence and leave valid coverage unresolved
        # instead of manufacturing a decision timestamp.
        for member, match_member in [
            ("odds_series.csv.gz", "odds_series_matches.csv.gz"),
            ("odds_series_b.csv.gz", "odds_series_b_matches.csv.gz"),
        ]:
            mapping = _btb_match_map(archive, match_member)
            scope_mapping = mapping[mapping["league_code"].isin(SCOPE_LEAGUES)]
            with archive.open(member) as raw, gzip.GzipFile(fileobj=raw) as zipped:
                header = next(csv.reader([zipped.readline().decode("utf-8", "replace")]))
            source_rows = len(mapping)
            scope_rows = len(scope_mapping)
            stats["files"] += 2
            stats["rows"] += source_rows + len(mapping)
            signature = schema_signature(header)
            for column in header:
                contract = classify_column("beat_the_bookie", column, member)
                if contract is None:
                    continue
                inventory.append(
                    {
                        "source": "beat_the_bookie", "source_file": source_file, "archive_member": member,
                        "canonical_source_copy": True, "league": "MULTI_SCOPE", "season": "MULTI_SCOPE",
                        "column": column, **asdict(contract), "match_rows": scope_rows,
                        "non_null_values": np.nan, "valid_odds_values": np.nan, "invalid_odds_le_1": np.nan,
                        "valid_odds_coverage": np.nan, "min_value": np.nan, "max_value": np.nan,
                        "schema_signature": signature,
                    }
                )
            anomalies.append(anomaly("encoded_series_semantics_unresolved", "blocker", "beat_the_bookie", f"{source_file}::{member}", affected_rows=scope_rows, details="All 6,912 encoded 1X2 series columns are inventoried, but bN bookmaker identities and index 0..71 timestamp direction/frequency are undocumented locally; per-league-season valid coverage is prohibited until resolved."))
    return inventory, anomalies, groups, stats


def valid(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") > 1


def group_contracts(source: str, frame: pd.DataFrame, member: str = "") -> dict[str, ColumnContract]:
    return {column: contract for column in frame.columns if (contract := classify_column(source, column, member)) is not None}


def complete_market_masks(source: str, frame: pd.DataFrame, member: str = "") -> dict[tuple[str, str, str], pd.Series]:
    contracts = group_contracts(source, frame, member)
    output: dict[tuple[str, str, str], pd.Series] = {}
    for timing in {contract.timing_classification for contract in contracts.values()}:
        if timing == "not_applicable":
            continue
        for market in ("1X2", "Over/Under", "Asian Handicap"):
            relevant = {c: x for c, x in contracts.items() if x.market == market and x.role == "odds" and x.timing_classification == timing}
            if not relevant:
                continue
            if market == "1X2":
                keys = {(x.bookmaker, x.line) for x in relevant.values()}
                selections = {"home", "draw", "away"}
            else:
                keys = {(x.bookmaker, x.line) for x in relevant.values()}
                selections = {"over", "under"} if market == "Over/Under" else {"home", "away"}
            masks = []
            for bookmaker, line in keys:
                columns = {x.selection: c for c, x in relevant.items() if x.bookmaker == bookmaker and x.line == line}
                if not selections.issubset(columns):
                    continue
                mask = pd.Series(True, index=frame.index)
                for selection in selections:
                    mask &= valid(frame[columns[selection]])
                if market == "Asian Handicap":
                    line_candidates = [
                        c for c, x in contracts.items()
                        if x.market == market and x.role == "line" and (
                            (timing == "verified_closing" and c == "AHCh")
                            or (timing != "verified_closing" and c != "AHCh")
                        )
                    ]
                    if line_candidates:
                        line_ok = pd.concat([pd.to_numeric(frame[c], errors="coerce").notna() for c in line_candidates], axis=1).any(axis=1)
                        mask &= line_ok
                    else:
                        mask &= False
                masks.append(mask)
            if masks:
                output[(market, timing, "any_bookmaker")] = pd.concat(masks, axis=1).any(axis=1)
    return output


def coverage_row(group: dict[str, object]) -> dict[str, object]:
    source = str(group["source"])
    frame = group["frame"]
    assert isinstance(frame, pd.DataFrame)
    member = "closing_odds.csv.gz" if source == "beat_the_bookie" else ""
    masks = complete_market_masks(source, frame, member)
    n = len(frame)

    def union(market: str | None = None, timings: set[str] | None = None) -> pd.Series:
        selected = [mask for (m, t, _), mask in masks.items() if (market is None or m == market) and (timings is None or t in timings)]
        return pd.concat(selected, axis=1).any(axis=1) if selected else pd.Series(False, index=frame.index)

    contracts = group_contracts(source, frame, member)
    ah_lines = [c for c, x in contracts.items() if x.market == "Asian Handicap" and x.role == "line"]
    bookmakers = set()
    aggregate_labels = {"Market average", "Market maximum", "BetBrain average", "BetBrain maximum", "market average", "market maximum", "unknown"}
    for column, contract in contracts.items():
        if contract.role == "odds" and contract.bookmaker not in aggregate_labels and valid(frame[column]).any():
            bookmakers.add(contract.bookmaker)
    dates_col = "Date" if "Date" in frame else "matchDate" if "matchDate" in frame else "match_date" if "match_date" in frame else None
    dates = parse_dates(frame[dates_col]) if dates_col else pd.Series(pd.NaT, index=frame.index)
    one = union("1X2")
    ah = union("Asian Handicap")
    ou = union("Over/Under")
    opening = union(timings={"verified_opening"})
    closing = union(timings={"verified_closing"})
    closing_1x2 = union("1X2", {"verified_closing"})
    closing_ah = union("Asian Handicap", {"verified_closing"})
    closing_ou = union("Over/Under", {"verified_closing"})
    unknown = union(timings={"current_snapshot_unknown_time", "timing_unknown"})
    ah_line_mask = pd.concat([pd.to_numeric(frame[c], errors="coerce").notna() for c in ah_lines], axis=1).any(axis=1) if ah_lines else pd.Series(False, index=frame.index)
    return {
        "source": source,
        "source_file": group["source_file"],
        "league": group["league"],
        "season": group["season"],
        "match_rows": n,
        "date_min": dates.min().date().isoformat() if dates.notna().any() else "",
        "date_max": dates.max().date().isoformat() if dates.notna().any() else "",
        "one_x_two_coverage": float(one.mean()) if n else 0.0,
        "ah_line_coverage": float(ah_line_mask.mean()) if n else 0.0,
        "ah_odds_coverage": float(ah.mean()) if n else 0.0,
        "ou_line_coverage": float(ou.mean()) if n else 0.0,
        "ou_odds_coverage": float(ou.mean()) if n else 0.0,
        "bookmaker_count": len(bookmakers),
        "opening_coverage": float(opening.mean()) if n else 0.0,
        "closing_coverage": float(closing.mean()) if n else 0.0,
        "closing_one_x_two_coverage": float(closing_1x2.mean()) if n else 0.0,
        "closing_ah_coverage": float(closing_ah.mean()) if n else 0.0,
        "closing_ou_coverage": float(closing_ou.mean()) if n else 0.0,
        "closing_cross_market_coverage": float((closing_1x2 & closing_ah & closing_ou).mean()) if n else 0.0,
        "timing_unknown_coverage": float(unknown.mean()) if n else 0.0,
        "closing_without_opening_rows": int((closing & ~opening).sum()),
        "opening_without_closing_rows": int((opening & ~closing).sum()),
        "schema_anomalies": "",
    }


def cross_file_anomalies(inventory: pd.DataFrame, fixture_rows: list[dict[str, object]], coverage: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    if fixture_rows:
        fixtures = pd.DataFrame(fixture_rows)
        fixtures = fixtures[fixtures["date"].ne("")]
        for key, group in fixtures.groupby(["league", "date", "home", "away"], sort=False):
            scores = group[["home_goals", "away_goals"]].dropna().drop_duplicates()
            if len(scores) > 1:
                output.append(anomaly("inconsistent_score_rows", "error", "football_data", "|".join(sorted(group["source_file"].unique())[:20]), key[0], affected_rows=len(group), details=f"Logical fixture {key[1]} {key[2]} vs {key[3]} has {len(scores)} distinct full-time scores."))

    meaning = inventory.groupby(["source", "column"])[["market", "role"]].apply(lambda x: len(x.drop_duplicates()))
    for (source, column), count in meaning.items():
        if count > 1:
            output.append(anomaly("column_meaning_changed_between_seasons", "blocker", source, "multiple", column=column, affected_rows=int(count), details="The same raw column name mapped to more than one market/role contract."))

    fd = inventory[(inventory["source"].eq("football_data")) & inventory["canonical_source_copy"].astype(bool) & inventory["role"].eq("odds")]
    for league, league_frame in fd.groupby("league"):
        seasons = sorted(league_frame["season"].dropna().unique())
        prior: set[str] | None = None
        for season in seasons:
            current = set(league_frame.loc[league_frame["season"].eq(season), "bookmaker"].dropna())
            if prior is not None and current != prior:
                appeared = sorted(current - prior)
                disappeared = sorted(prior - current)
                output.append(anomaly("bookmaker_columns_appeared_or_disappeared", "info", "football_data", "canonical_source_copy", league, season, affected_rows=len(appeared) + len(disappeared), details=f"appeared={appeared}; disappeared={disappeared}"))
            prior = current

    for row in coverage.itertuples(index=False):
        if row.closing_without_opening_rows:
            output.append(anomaly("fixtures_with_closing_but_no_verified_opening", "blocker", row.source, row.source_file, row.league, row.season, affected_rows=int(row.closing_without_opening_rows), details="Verified closing prices exist, but no locally evidenced verified-opening price exists for the same fixture."))
        if row.opening_without_closing_rows:
            output.append(anomaly("fixtures_with_verified_opening_but_no_closing", "warning", row.source, row.source_file, row.league, row.season, affected_rows=int(row.opening_without_closing_rows), details="Verified opening exists without verified closing."))
    return output


def build_contract(inventory: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source", "column", "market", "role", "selection", "bookmaker", "line",
        "timing_classification", "timing_evidence_id", "feature_policy",
    ]
    contract = inventory[columns].drop_duplicates().sort_values(columns[:2] + ["market", "timing_classification"], kind="stable")
    contract["same_decision_timestamp_comparable"] = contract["timing_classification"].isin({"verified_opening", "verified_closing"})
    contract["comparison_policy"] = np.where(
        contract["timing_classification"].eq("verified_closing"),
        "closing-to-closing only within the same raw fixture row and documented provider schema",
        np.where(
            contract["timing_classification"].eq("verified_opening"),
            "opening-to-opening only within the same documented snapshot",
            "prohibited: decision timestamp is not verified",
        ),
    )
    return contract


def safe_closing_windows(coverage: pd.DataFrame, market: str) -> list[str]:
    metric = {"1X2": "closing_one_x_two_coverage", "Asian Handicap": "closing_ah_coverage", "Over/Under": "closing_ou_coverage"}[market]
    rows = coverage[(coverage["source"].eq("football_data")) & coverage[metric].gt(0)].copy()
    output: list[str] = []
    for league, group in rows.groupby("league", sort=True):
        group = group.drop_duplicates("season").sort_values("season")
        seasons = group["season"].tolist()
        output.append(f"{league}: {seasons[0]} through {seasons[-1]}, all included seasons listed in the coverage CSV (minimum {metric}={group[metric].min():.1%})")
    return output


def write_reports(inventory: pd.DataFrame, contract: pd.DataFrame, coverage: pd.DataFrame, anomalies: pd.DataFrame, stats: dict[str, dict[str, int]]) -> str:
    verified_open = inventory[(inventory["role"].eq("odds")) & inventory["timing_classification"].eq("verified_opening")]
    verified_close = inventory[(inventory["role"].eq("odds")) & inventory["timing_classification"].eq("verified_closing") & inventory["valid_odds_values"].fillna(0).gt(0)]
    answers = {
        "verified opening 1X2 odds": not verified_open[verified_open["market"].eq("1X2")].empty,
        "verified closing 1X2 odds": not verified_close[verified_close["market"].eq("1X2")].empty,
        "opening and closing for same bookmaker": False,
        "opening AH data": not verified_open[verified_open["market"].eq("Asian Handicap")].empty,
        "closing AH data": not verified_close[verified_close["market"].eq("Asian Handicap")].empty,
        "opening O/U data": not verified_open[verified_open["market"].eq("Over/Under")].empty,
        "closing O/U data": not verified_close[verified_close["market"].eq("Over/Under")].empty,
    }
    if all(answers.values()):
        decision = "v4_phase1_cross_market_contract_ready_research_only"
    elif answers["verified opening 1X2 odds"] and answers["verified closing 1X2 odds"]:
        decision = "v4_phase1_clv_contract_ready_research_only"
    elif any(answers.values()):
        decision = "v4_phase1_partial_market_timing_contract"
    else:
        decision = "v4_phase1_blocked_no_verified_market_timing"

    inspected_files = sum(x["files"] for x in stats.values())
    inspected_rows = sum(x["rows"] for x in stats.values())
    close_1x2 = safe_closing_windows(coverage, "1X2")
    close_ah = safe_closing_windows(coverage, "Asian Handicap")
    close_ou = safe_closing_windows(coverage, "Over/Under")
    evidence = """## Timing evidence

- `fd_archive_notes_closing_definition`: the embedded football-data archive `notes.txt` explicitly defines closing odds as the last odds before the match starts and says they use an added `C` after the bookmaker/Max/Avg abbreviation. This verifies recognized `C` price columns as closing.
- `fd_archive_notes_scheduled_collection_not_opening`: the same notes say ordinary weekend odds were collected Friday afternoons and midweek odds Tuesday afternoons. That is evidence for a scheduled pre-match snapshot, but not evidence of first-available/opening prices; these columns are `current_snapshot_unknown_time`.
- `footiqo_project_docs_timing_unknown`: `docs/README_PIPELINE.md` and `docs/SUPER_CSV_CONTRACT.md` explicitly state Footiqo odds timing is unknown.
- `btb_local_metadata_no_series_timestamp_semantics`: the Beat-the-Bookie metadata lists line movement/CLV as potential uses but does not define encoded bookmaker identities or the direction/frequency of indices 0..71. The `closing_odds.csv.gz` filename alone is not timing proof.
"""
    def yn(value: bool) -> str:
        return "Yes" if value else "No"

    report = f"""# V4 Phase 1 Market Data Contract Audit

Decision: **{decision}**

Phase 1 only. No predictive model was built or trained, no betting profit was calculated, no raw file was modified, and the frozen V3/V3 Next and paper pipelines were not changed. No confirmed edge is claimed.

## Direct answers

1. **Do we have verified opening 1X2 odds?** {yn(answers['verified opening 1X2 odds'])}. Football-data non-C odds are documented scheduled snapshots, not verified openings; Footiqo and Beat-the-Bookie series timing is unresolved.
2. **Do we have verified closing 1X2 odds?** {yn(answers['verified closing 1X2 odds'])}. Recognized football-data C variants have explicit provider documentation.
3. **Do we have opening and closing values for the same bookmaker?** No. The apparent non-C/C pairs must not be called opening/closing pairs because the non-C side is not verified opening.
4. **Do we have opening AH data?** {yn(answers['opening AH data'])}.
5. **Do we have closing AH data?** {yn(answers['closing AH data'])}. Football-data C AH prices/lines are closing-labelled by explicit schema evidence.
6. **Do we have opening O/U data?** {yn(answers['opening O/U data'])}.
7. **Do we have closing O/U data?** {yn(answers['closing O/U data'])}. Football-data C total prices are closing-labelled by explicit schema evidence.
8. **Can a valid CLV target be built?** No. A leakage-safe CLV target requires a verified decision-time/opening observation paired to a verified later/closing observation for the same bookmaker, selection, line, and fixture. The first endpoint is absent.
9. **Can cross-market features be built at a common decision time?** No for decision-time features. Football-data closing 1X2/AH/O-U may be compared only as same-row closing diagnostics/labels where complete; closing data cannot be used as pre-match decision features. Unknown-timing Footiqo/Beat data cannot be mixed with it.
10. **What exact years and leagues are safe?** No league-year is safe for verified-opening features or CLV. The following are safe only for research-only verified-closing labels/diagnostics, subject to the reported row coverage and anomalies.

## Verified-closing availability by market

### 1X2

""" + ("\n".join(f"- {x}" for x in close_1x2) if close_1x2 else "- None") + """

### Asian Handicap

""" + ("\n".join(f"- {x}" for x in close_ah) if close_ah else "- None") + """

### Over/Under

""" + ("\n".join(f"- {x}" for x in close_ou) if close_ou else "- None") + f"""

{evidence}

## Feature and label policy

- **Allowed as opening-time features:** none of the audited raw odds columns. There are no `verified_opening` odds.
- **Closing labels/diagnostics only:** recognized football-data `C` 1X2, AH, and O/U prices with complete same-row market pairs/triples. Never use them in selection or prediction features.
- **Prohibited until timing is resolved:** all football-data non-C scheduled snapshot odds, all Footiqo odds, and all Beat-the-Bookie series/aggregate odds.
- **Same timestamp comparisons:** only recognized complete football-data closing markets on the same raw fixture row; this supports market-structure diagnostics, not decision-time cross-market features.
- **AH lines:** `AHCh` is a verified-closing structural line; non-C AH line fields are scheduled/unknown-time snapshots and remain prohibited as decision-time features.
- **Not applicable:** bookmaker-count columns are counts, not standalone price observations.

## Audit scope and accounting

- Physical/archive members inspected: {inspected_files}
- Rows inspected or metadata-mapped: {inspected_rows}
- Odds inventory rows: {len(inventory)}
- Contract rows: {len(contract)}
- League-season coverage rows: {len(coverage)}
- Schema anomaly rows: {len(anomalies)}
- Included leagues: {', '.join(SCOPE_LEAGUES)}
- Explicitly excluded leagues: {', '.join(EXCLUDED_LEAGUES)}

The inventory includes every recognized market column in scoped raw football-data CSVs and archive members, every Footiqo odds column, every compact Beat-the-Bookie aggregate column, and all 13,824 encoded Beat-the-Bookie series price columns. Encoded series valid coverage is intentionally blank because local timestamp/bookmaker semantics are unresolved.

## Anomaly interpretation

`v4_market_schema_anomalies.csv` records byte-identical source copies, within-file duplicate fixtures, score conflicts, invalid odds, impossible lines, schema/bookmaker transitions, unresolved encoded series semantics, and closing-without-verified-opening counts. Duplicate storage is not treated as independent market coverage. Coverage uses one preferred raw football-data season file, falling back to the archive only when the preferred raw season is absent.
"""
    (OUT_DIR / "v4_data_contract_audit.md").write_text(report, encoding="utf-8")
    decision_report = f"""# V4 Phase 1 Decision

**{decision}**

Verified football-data closing 1X2, Asian Handicap, and Over/Under observations exist for reported league-seasons. No verified opening odds exist. Therefore a valid opening-to-closing CLV label and common decision-time cross-market feature set cannot yet be built.

No model was built or trained, no profit was calculated, no frozen V3/V3 Next or paper-pipeline code was changed, and no confirmed edge is claimed. Stop after Phase 1.
"""
    (OUT_DIR / "v4_phase1_decision.md").write_text(decision_report, encoding="utf-8")
    return decision


def run() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_inv, raw_bad, raw_groups, raw_stats, fixture_rows = scan_football_data_raw()
    existing = {(str(group["league"]), str(group["season"])) for group in raw_groups}
    archive_inv, archive_bad, archive_groups, archive_stats = scan_football_data_archive(existing)
    footiqo_inv, footiqo_bad, footiqo_groups, footiqo_stats = scan_footiqo()
    btb_inv, btb_bad, btb_groups, btb_stats = scan_beat_the_bookie()
    inventory = pd.DataFrame(raw_inv + archive_inv + footiqo_inv + btb_inv, columns=INVENTORY_COLUMNS)
    inventory = inventory.sort_values(["source", "league", "season", "source_file", "archive_member", "column"], kind="stable").reset_index(drop=True)
    groups = raw_groups + archive_groups + footiqo_groups + btb_groups
    coverage = pd.DataFrame([coverage_row(group) for group in groups]).sort_values(["source", "league", "season"], kind="stable").reset_index(drop=True)
    bad = raw_bad + archive_bad + footiqo_bad + btb_bad
    bad.extend(cross_file_anomalies(inventory, fixture_rows, coverage))
    anomalies = pd.DataFrame(bad, columns=ANOMALY_COLUMNS).sort_values(["severity", "anomaly_type", "source", "league", "season"], kind="stable").reset_index(drop=True)
    counts = anomalies.groupby(["source", "source_file", "league", "season"])["anomaly_type"].agg(lambda x: ";".join(sorted(set(x)))).to_dict()
    coverage["schema_anomalies"] = [counts.get((r.source, r.source_file, r.league, r.season), "") for r in coverage.itertuples(index=False)]
    contract = build_contract(inventory)

    inventory.to_csv(OUT_DIR / "v4_odds_column_inventory.csv", index=False)
    contract.to_csv(OUT_DIR / "v4_market_data_contract.csv", index=False)
    coverage.to_csv(OUT_DIR / "v4_market_coverage_by_league_season.csv", index=False)
    anomalies.to_csv(OUT_DIR / "v4_market_schema_anomalies.csv", index=False)
    stats = {"football_data_raw": raw_stats, "football_data_archive": archive_stats, "footiqo": footiqo_stats, "beat_the_bookie": btb_stats}
    decision = write_reports(inventory, contract, coverage, anomalies, stats)
    return {
        "decision": decision,
        "files_inspected": sum(x["files"] for x in stats.values()),
        "rows_inspected": sum(x["rows"] for x in stats.values()),
        "inventory_rows": len(inventory),
        "coverage_rows": len(coverage),
        "anomaly_rows": len(anomalies),
    }
