from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.build_league_dataset import clean_required_rows
from src.data.build_league_dataset import extract_season_from_filename
from src.data.build_league_dataset import infer_season_from_dates
from src.data.build_league_dataset import parse_match_date
from src.data.build_league_dataset import read_raw_csv


LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "E1", "E2", "E3", "SC0"]
SEASON_STARTS = list(range(2004, 2026))
REQUESTED_AH_COLUMNS = ["AHh", "AvgAHH", "AvgAHA", "BbAH", "BbAHh", "MaxAHH", "MaxAHA"]
LEGACY_AH_COLUMNS = ["BbAvAHH", "BbAvAHA", "BbMxAHH", "BbMxAHA"]
CORE_AH_COLUMNS = ["AHh", "AvgAHH", "AvgAHA"]
MATCH_KEY = ["Date", "HomeTeam", "AwayTeam"]

REPORT_PATH = Path("outputs/reports/historical_ah_backfill_audit.md")
COVERAGE_PATH = Path("outputs/reports/historical_ah_raw_vs_processed_coverage.csv")
ACTIONS_PATH = Path("outputs/reports/historical_ah_reprocess_actions.csv")
COMPATIBILITY_PATH = Path("outputs/reports/historical_ah_settlement_compatibility.csv")


def processed_path(league: str) -> Path:
    return Path("data/processed") / league / f"{league}_matches.csv"


def raw_files_for_league(league: str) -> list[Path]:
    root = Path("data/raw") / league
    files = []
    files.extend(sorted((root / "seasons").glob("*.csv")))
    files.extend(sorted(root.glob("*.csv")))
    return sorted(set(files))


def expected_url(league: str, season_start: int) -> str:
    season_code = f"{str(season_start)[-2:]}{str(season_start + 1)[-2:]}"
    return f"https://www.football-data.co.uk/mmz4281/{season_code}/{league}.csv"


def numeric_non_null(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not set(columns).issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    converted = frame[columns].apply(pd.to_numeric, errors="coerce")
    return converted.notna().all(axis=1)


def read_match_frame(path: Path) -> pd.DataFrame:
    frame, _, _, _ = read_raw_csv(path)
    frame = parse_match_date(frame)
    return frame


def season_from_raw(path: Path, frame: pd.DataFrame) -> tuple[int | None, int | None]:
    start, end = extract_season_from_filename(path)
    if start is not None and end is not None:
        return start, end
    return infer_season_from_dates(frame)


def raw_true_ah_mask(frame: pd.DataFrame) -> pd.Series:
    canonical = numeric_non_null(frame, ["AHh", "AvgAHH", "AvgAHA"])
    legacy = numeric_non_null(frame, ["BbAHh", "BbAvAHH", "BbAvAHA"])
    return canonical | legacy


def raw_file_inventory() -> pd.DataFrame:
    rows = []
    for league in LEAGUES:
        for path in raw_files_for_league(league):
            try:
                frame = read_match_frame(path)
                start, end = season_from_raw(path, frame)
                requested_present = [column for column in REQUESTED_AH_COLUMNS if column in frame.columns]
                legacy_present = [column for column in LEGACY_AH_COLUMNS if column in frame.columns]
                rows.append(
                    {
                        "league": league,
                        "season_start_year": start,
                        "season_end_year": end,
                        "raw_path": str(path),
                        "raw_rows": int(len(frame)),
                        "requested_ah_columns_present": ";".join(requested_present),
                        "legacy_ah_columns_present": ";".join(legacy_present),
                        "raw_canonical_ah_rows": int(numeric_non_null(frame, CORE_AH_COLUMNS).sum()),
                        "raw_legacy_ah_rows": int(numeric_non_null(frame, ["BbAHh", "BbAvAHH", "BbAvAHA"]).sum()),
                        "raw_true_ah_rows": int(raw_true_ah_mask(frame).sum()),
                        "read_status": "ok",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "league": league,
                        "season_start_year": pd.NA,
                        "season_end_year": pd.NA,
                        "raw_path": str(path),
                        "raw_rows": 0,
                        "requested_ah_columns_present": "",
                        "legacy_ah_columns_present": "",
                        "raw_canonical_ah_rows": 0,
                        "raw_legacy_ah_rows": 0,
                        "raw_true_ah_rows": 0,
                        "read_status": f"read_failed:{type(exc).__name__}:{exc}",
                    }
                )
    return pd.DataFrame(rows)


def processed_inventory() -> pd.DataFrame:
    rows = []
    for league in LEAGUES:
        path = processed_path(league)
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        if "season_start_year" not in frame.columns:
            continue
        requested_present = [column for column in REQUESTED_AH_COLUMNS if column in frame.columns]
        for season_start, group in frame.groupby("season_start_year", dropna=True):
            rows.append(
                {
                    "league": league,
                    "season_start_year": int(season_start),
                    "processed_path": str(path),
                    "processed_rows": int(len(group)),
                    "processed_requested_ah_columns_present": ";".join(requested_present),
                    "processed_core_ah_rows": int(numeric_non_null(group, CORE_AH_COLUMNS).sum()),
                }
            )
    return pd.DataFrame(rows)


def aggregate_coverage(raw_inventory: pd.DataFrame, processed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for league in LEAGUES:
        for season_start in SEASON_STARTS:
            raw = raw_inventory[
                raw_inventory["league"].eq(league)
                & pd.to_numeric(raw_inventory["season_start_year"], errors="coerce").eq(season_start)
            ].copy()
            proc = processed[
                processed["league"].eq(league)
                & pd.to_numeric(processed["season_start_year"], errors="coerce").eq(season_start)
            ].copy()
            raw_exists = len(raw) > 0
            raw_true_rows = int(pd.to_numeric(raw.get("raw_true_ah_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if raw_exists else 0
            raw_canonical_rows = int(pd.to_numeric(raw.get("raw_canonical_ah_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if raw_exists else 0
            raw_legacy_rows = int(pd.to_numeric(raw.get("raw_legacy_ah_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if raw_exists else 0
            processed_core_rows = int(pd.to_numeric(proc.get("processed_core_ah_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(proc) else 0
            raw_best = raw.sort_values(["raw_true_ah_rows", "raw_canonical_ah_rows", "raw_rows"], ascending=False).head(1)
            rows.append(
                {
                    "league": league,
                    "season_start_year": season_start,
                    "season_label": f"{season_start}/{str(season_start + 1)[-2:]}",
                    "local_raw_exists": raw_exists,
                    "raw_paths": ";".join(raw["raw_path"].astype(str).tolist()) if raw_exists else "",
                    "best_raw_path": raw_best["raw_path"].iloc[0] if len(raw_best) else "",
                    "expected_url_if_missing": "" if raw_exists else expected_url(league, season_start),
                    "requested_ah_columns_present": ";".join(sorted(set(";".join(raw["requested_ah_columns_present"].dropna().astype(str)).split(";")) - {""})) if raw_exists else "",
                    "legacy_ah_columns_present": ";".join(sorted(set(";".join(raw["legacy_ah_columns_present"].dropna().astype(str)).split(";")) - {""})) if raw_exists else "",
                    "raw_canonical_ah_rows": raw_canonical_rows,
                    "raw_legacy_ah_rows": raw_legacy_rows,
                    "raw_true_ah_rows": raw_true_rows,
                    "raw_true_ah_present": raw_true_rows > 0,
                    "processed_exists_for_season": len(proc) > 0,
                    "processed_rows": int(pd.to_numeric(proc.get("processed_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(proc) else 0,
                    "processed_core_ah_rows": processed_core_rows,
                    "processed_ah_present": processed_core_rows > 0,
                    "raw_has_ah_but_processed_lost_it": raw_true_rows > 0 and processed_core_rows == 0,
                    "audit_note": "missing_local_raw_report_url_only" if not raw_exists else ("parser_backfill_needed" if raw_true_rows > 0 and processed_core_rows == 0 else "no_loss_detected"),
                }
            )
    return pd.DataFrame(rows)


def canonicalize_ah_columns(frame: pd.DataFrame) -> pd.DataFrame:
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
            canonical_values = pd.to_numeric(out[canonical], errors="coerce")
            legacy_values = pd.to_numeric(out[legacy], errors="coerce")
            out[canonical] = canonical_values.where(canonical_values.notna(), legacy_values)
    return out


def build_recovered_season_frame(league: str, season_start: int, raw_path: Path) -> pd.DataFrame:
    frame = read_match_frame(raw_path)
    frame = canonicalize_ah_columns(frame)
    frame, _ = clean_required_rows(frame)
    frame["league"] = league
    frame["season_start_year"] = season_start
    frame["season_end_year"] = season_start + 1
    frame["source_file"] = raw_path.name
    return frame


def reprocess_lost_seasons(coverage: pd.DataFrame) -> pd.DataFrame:
    action_rows = []
    lost = coverage[coverage["raw_has_ah_but_processed_lost_it"]].copy()
    for league, group in lost.groupby("league"):
        path = processed_path(league)
        existing = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
        recovered_frames = []
        for _, row in group.iterrows():
            season_start = int(row["season_start_year"])
            raw_path = Path(row["best_raw_path"])
            try:
                recovered = build_recovered_season_frame(league, season_start, raw_path)
                core_rows = int(numeric_non_null(recovered, CORE_AH_COLUMNS).sum())
                if core_rows == 0:
                    status = "skipped_no_canonical_ah_after_mapping"
                else:
                    recovered_frames.append(recovered)
                    status = "queued_for_processed_merge"
                action_rows.append(
                    {
                        "league": league,
                        "season_start_year": season_start,
                        "season_end_year": season_start + 1,
                        "raw_path": str(raw_path),
                        "processed_path": str(path),
                        "action": "replace_or_append_season",
                        "recovered_rows": int(len(recovered)),
                        "recovered_core_ah_rows": core_rows,
                        "status": status,
                    }
                )
            except Exception as exc:
                action_rows.append(
                    {
                        "league": league,
                        "season_start_year": season_start,
                        "season_end_year": season_start + 1,
                        "raw_path": str(raw_path),
                        "processed_path": str(path),
                        "action": "replace_or_append_season",
                        "recovered_rows": 0,
                        "recovered_core_ah_rows": 0,
                        "status": f"failed:{type(exc).__name__}:{exc}",
                    }
                )
        if not recovered_frames:
            continue
        recovered_all = pd.concat(recovered_frames, ignore_index=True, sort=False)
        recovered_seasons = set(recovered_all["season_start_year"].dropna().astype(int).tolist())
        if len(existing):
            existing = existing[~pd.to_numeric(existing["season_start_year"], errors="coerce").isin(recovered_seasons)].copy()
            output = pd.concat([existing, recovered_all], ignore_index=True, sort=False)
        else:
            output = recovered_all
        if "Date" in output.columns:
            output["Date"] = pd.to_datetime(output["Date"], errors="coerce")
        output = output.sort_values(["Date", "HomeTeam", "AwayTeam"]).drop_duplicates(MATCH_KEY, keep="first").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)
        for action in action_rows:
            if action["league"] == league and action["status"] == "queued_for_processed_merge":
                action["status"] = "completed"
    return pd.DataFrame(action_rows)


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


def settle_label(team_margin: float, handicap: float, odds: float) -> str:
    if pd.isna(team_margin) or pd.isna(handicap) or pd.isna(odds) or float(odds) <= 1.0:
        return "invalid"
    outcomes = []
    for part in split_handicap(float(handicap)):
        adjusted = float(team_margin) + part
        outcomes.append(1 if adjusted > 0 else (0 if adjusted == 0 else -1))
    wins = outcomes.count(1)
    pushes = outcomes.count(0)
    losses = outcomes.count(-1)
    if wins == len(outcomes):
        return "full_win"
    if wins and pushes and not losses:
        return "half_win"
    if pushes == len(outcomes):
        return "push"
    if losses and pushes and not wins:
        return "half_loss"
    if losses == len(outcomes):
        return "full_loss"
    return "impossible"


def settlement_compatibility(actions: pd.DataFrame) -> pd.DataFrame:
    completed = actions[actions["status"].eq("completed")].copy() if len(actions) else pd.DataFrame()
    rows = []
    labels = {"full_win", "half_win", "push", "half_loss", "full_loss"}
    for _, action in completed.iterrows():
        league = action["league"]
        season_start = int(action["season_start_year"])
        frame = pd.read_csv(processed_path(league), low_memory=False)
        frame = frame[pd.to_numeric(frame["season_start_year"], errors="coerce").eq(season_start)].copy()
        for column in ["FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        margin = frame["FTHG"] - frame["FTAG"]
        home_labels = [settle_label(m, h, o) for m, h, o in zip(margin, frame["AHh"], frame["AvgAHH"])]
        away_labels = [settle_label(-m, -h if pd.notna(h) else np.nan, o) for m, h, o in zip(margin, frame["AHh"], frame["AvgAHA"])]
        frame["home_label"] = home_labels
        frame["away_label"] = away_labels
        valid = frame["home_label"].isin(labels) & frame["away_label"].isin(labels)
        binary_previous = np.where(margin + frame["AHh"] > 0, 1.0, np.where(margin + frame["AHh"] < 0, 0.0, np.nan))
        settlement_cover = np.where(frame["home_label"].isin(["full_win", "half_win"]), 1.0, np.where(frame["home_label"].isin(["half_loss", "full_loss"]), 0.0, np.nan))
        comparable = valid & pd.Series(binary_previous).notna() & pd.Series(settlement_cover).notna()
        mismatches = int((pd.Series(binary_previous)[comparable].to_numpy() != pd.Series(settlement_cover)[comparable].to_numpy()).sum())
        rows.append(
            {
                "league": league,
                "season_start_year": season_start,
                "season_end_year": season_start + 1,
                "processed_path": str(processed_path(league)),
                "matches": int(len(frame)),
                "valid_ah_settlement_rows": int(valid.sum()),
                "missing_ah_rows": int((~numeric_non_null(frame, CORE_AH_COLUMNS)).sum()),
                "impossible_settlement_rows": int((frame["home_label"].eq("impossible") | frame["away_label"].eq("impossible")).sum()),
                "push_rows_excluded_from_binary_target": int(frame["home_label"].eq("push").sum()),
                "binary_target_mismatch_rows": mismatches,
                "binary_target_compatible": bool(mismatches == 0),
                "status": "compatible" if mismatches == 0 and int(valid.sum()) > 0 else "review_needed",
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    return view.to_markdown(index=False)


def write_report(coverage: pd.DataFrame, actions: pd.DataFrame, compatibility: pd.DataFrame, classification: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lost_column = "pre_backfill_raw_has_ah_but_processed_lost_it"
    if lost_column not in coverage.columns:
        lost_column = "raw_has_ah_but_processed_lost_it"
    lost = coverage[coverage[lost_column]].copy()
    missing_raw = coverage[~coverage["local_raw_exists"]].copy()
    raw_summary = (
        coverage.groupby("league")
        .agg(
            local_raw_seasons=("local_raw_exists", "sum"),
            raw_true_ah_seasons=("raw_true_ah_present", "sum"),
            processed_ah_seasons=("processed_ah_present", "sum"),
            recovered_loss_seasons=(lost_column, "sum"),
            remaining_lost_ah_seasons=("raw_has_ah_but_processed_lost_it", "sum"),
        )
        .reset_index()
    )
    lines = [
        "# Historical AH Backfill Audit",
        "",
        f"Final classification: `{classification}`",
        "",
        "Scope: local football-data CSV raw files and direct football-data CSV URL reporting only. No models, betting strategies, value searches, threshold optimization, Transfermarkt data, dynamic scraping, or confirmed edge claims were used. Raw files were read only and left unchanged.",
        "",
        "True AH odds were counted when a row had either canonical `AHh`/`AvgAHH`/`AvgAHA` values or legacy football-data `BbAHh`/`BbAvAHH`/`BbAvAHA` values. Legacy `BbMxAHH`/`BbMxAHA` were also mapped to canonical max columns during targeted processed-file backfill.",
        "",
        "## League Summary",
        "",
        markdown_table(raw_summary, ["league", "local_raw_seasons", "raw_true_ah_seasons", "processed_ah_seasons", "recovered_loss_seasons", "remaining_lost_ah_seasons"], max_rows=20),
        "",
        "## Backfill Actions",
        "",
        markdown_table(actions, ["league", "season_start_year", "season_end_year", "action", "recovered_rows", "recovered_core_ah_rows", "status"], max_rows=80),
        "",
        "## Settlement Compatibility For Recovered Seasons",
        "",
        markdown_table(compatibility, ["league", "season_start_year", "season_end_year", "matches", "valid_ah_settlement_rows", "missing_ah_rows", "impossible_settlement_rows", "binary_target_mismatch_rows", "status"], max_rows=80),
        "",
        "## Seasons Where Raw AH Was Lost In Processed Files Before Backfill",
        "",
        markdown_table(lost, ["league", "season_label", "best_raw_path", "raw_canonical_ah_rows", "raw_legacy_ah_rows", "pre_backfill_processed_core_ah_rows", "processed_core_ah_rows", "recovered_by_backfill", "audit_note"], max_rows=120),
        "",
        "## Missing Local Raw Files",
        "",
        "Missing local raw rows are listed in `historical_ah_raw_vs_processed_coverage.csv` with `expected_url_if_missing`; no downloads were performed.",
        "",
        markdown_table(missing_raw, ["league", "season_label", "expected_url_if_missing"], max_rows=40),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def classify(coverage: pd.DataFrame, actions: pd.DataFrame, compatibility: pd.DataFrame) -> str:
    lost_count = int(coverage["raw_has_ah_but_processed_lost_it"].sum())
    completed_count = int(actions["status"].eq("completed").sum()) if len(actions) else 0
    if lost_count == 0:
        return "no_backfill_needed"
    if completed_count == 0:
        return "backfill_possible"
    if completed_count == lost_count:
        parser_bug = int(coverage[coverage["raw_has_ah_but_processed_lost_it"]]["raw_legacy_ah_rows"].gt(0).sum()) > 0
        return "parser_bug_found" if parser_bug else "backfill_completed"
    return "backfill_possible"


def main() -> None:
    raw_inventory = raw_file_inventory()
    before_processed = processed_inventory()
    initial_coverage = aggregate_coverage(raw_inventory, before_processed)
    actions = reprocess_lost_seasons(initial_coverage)
    after_processed = processed_inventory()
    final_coverage = aggregate_coverage(raw_inventory, after_processed)
    before_flags = initial_coverage[
        [
            "league",
            "season_start_year",
            "processed_core_ah_rows",
            "processed_ah_present",
            "raw_has_ah_but_processed_lost_it",
        ]
    ].rename(
        columns={
            "processed_core_ah_rows": "pre_backfill_processed_core_ah_rows",
            "processed_ah_present": "pre_backfill_processed_ah_present",
            "raw_has_ah_but_processed_lost_it": "pre_backfill_raw_has_ah_but_processed_lost_it",
        }
    )
    final_coverage = final_coverage.merge(before_flags, on=["league", "season_start_year"], how="left")
    if len(actions):
        completed = actions[actions["status"].eq("completed")][["league", "season_start_year"]].copy()
        completed["recovered_by_backfill"] = True
        final_coverage = final_coverage.merge(completed, on=["league", "season_start_year"], how="left")
        final_coverage["recovered_by_backfill"] = final_coverage["recovered_by_backfill"].fillna(False)
    else:
        final_coverage["recovered_by_backfill"] = False
    compatibility = settlement_compatibility(actions)
    classification = classify(initial_coverage, actions, compatibility)

    COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_coverage.to_csv(COVERAGE_PATH, index=False)
    actions.to_csv(ACTIONS_PATH, index=False)
    compatibility.to_csv(COMPATIBILITY_PATH, index=False)
    write_report(final_coverage, actions, compatibility, classification)
    print(
        {
            "raw_inventory_files": int(len(raw_inventory)),
            "coverage_rows": int(len(final_coverage)),
            "lost_seasons_initial": int(initial_coverage["raw_has_ah_but_processed_lost_it"].sum()),
            "actions": int(len(actions)),
            "completed_actions": int(actions["status"].eq("completed").sum()) if len(actions) else 0,
            "classification": classification,
        }
    )


if __name__ == "__main__":
    main()
