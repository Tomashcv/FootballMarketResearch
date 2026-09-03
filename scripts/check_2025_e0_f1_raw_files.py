from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "reports" / "v3_2025_completion"
OUT_PATH = OUT_DIR / "e0_f1_raw_file_check.csv"

TARGETS = {
    "E0": ROOT / "data" / "raw" / "E0" / "seasons" / "E0_2526.csv",
    "F1": ROOT / "data" / "raw" / "F1" / "seasons" / "F1_2526.csv",
}

REQUIRED_BASE_COLUMNS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_SETS = {
    "B365": ["B365H", "B365D", "B365A"],
    "Avg": ["AvgH", "AvgD", "AvgA"],
    "Plain": ["H", "D", "A"],
}
MIN_DATE = pd.Timestamp("2025-07-01")
MAX_DATE = pd.Timestamp("2026-06-30")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)
    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(values, errors="coerce", dayfirst=False)
    return parsed


def result_matches_scores(df: pd.DataFrame) -> pd.Series:
    home_goals = pd.to_numeric(df["FTHG"], errors="coerce")
    away_goals = pd.to_numeric(df["FTAG"], errors="coerce")
    expected = pd.Series(pd.NA, index=df.index, dtype="object")
    expected.loc[home_goals > away_goals] = "H"
    expected.loc[home_goals == away_goals] = "D"
    expected.loc[home_goals < away_goals] = "A"
    return df["FTR"].astype(str).str.upper().eq(expected)


def valid_odds_rows(df: pd.DataFrame) -> tuple[str | None, int]:
    for source, columns in ODDS_SETS.items():
        if all(col in df.columns for col in columns):
            odds = df[columns].apply(pd.to_numeric, errors="coerce")
            valid = odds.notna().all(axis=1) & (odds > 1.0).all(axis=1)
            return source, int(valid.sum())
    return None, 0


def check_file(league: str, path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "league": league,
        "target_path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "rows": 0,
        "schema_ok": False,
        "odds_source_available": None,
        "date_min": None,
        "date_max": None,
        "date_range_2025_26_ok": False,
        "valid_results": False,
        "valid_1x2_odds": False,
        "duplicate_logical_fixtures": None,
        "raw_file_modified": False,
        "status": "missing",
        "details": "file missing; stop before V3 validation rerun",
    }
    if not path.exists():
        return row

    before_hash = sha256(path)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        after_hash = sha256(path)
        row.update(
            {
                "raw_file_modified": before_hash != after_hash,
                "status": "fail",
                "details": f"could not read CSV: {exc}",
            }
        )
        return row

    after_hash = sha256(path)
    missing_columns = [col for col in REQUIRED_BASE_COLUMNS if col not in df.columns]
    odds_source, valid_odds_count = valid_odds_rows(df)
    dates = parse_dates(df["Date"]) if "Date" in df.columns else pd.Series(pd.NaT, index=df.index)

    row_count = len(df)
    date_range_ok = bool(row_count > 0 and dates.notna().all() and dates.min() >= MIN_DATE and dates.max() <= MAX_DATE)
    scores_numeric = (
        pd.to_numeric(df["FTHG"], errors="coerce").notna().all()
        and pd.to_numeric(df["FTAG"], errors="coerce").notna().all()
        if {"FTHG", "FTAG"}.issubset(df.columns)
        else False
    )
    ftr_valid = df["FTR"].astype(str).str.upper().isin(["H", "D", "A"]).all() if "FTR" in df.columns else False
    results_valid = bool(row_count > 0 and scores_numeric and ftr_valid and result_matches_scores(df).all()) if not missing_columns else False
    odds_valid = bool(row_count > 0 and odds_source is not None and valid_odds_count == row_count)

    duplicate_count = None
    if {"Div", "HomeTeam", "AwayTeam"}.issubset(df.columns):
        logical = pd.DataFrame(
            {
                "Div": df["Div"].astype(str).str.strip(),
                "season_start_year": 2025,
                "HomeTeam": df["HomeTeam"].astype(str).str.strip().str.lower(),
                "AwayTeam": df["AwayTeam"].astype(str).str.strip().str.lower(),
            }
        )
        duplicate_count = int(logical.duplicated(keep=False).sum())

    schema_ok = not missing_columns and odds_source is not None
    checks_ok = bool(schema_ok and row_count > 0 and date_range_ok and results_valid and odds_valid and duplicate_count == 0)

    details = []
    if missing_columns:
        details.append(f"missing required columns: {', '.join(missing_columns)}")
    if odds_source is None:
        details.append("missing supported 1X2 odds columns")
    if row_count <= 0:
        details.append("no rows")
    if not date_range_ok:
        details.append("date range is not fully inside 2025-07-01 to 2026-06-30")
    if not results_valid:
        details.append("invalid or inconsistent results")
    if not odds_valid:
        details.append("invalid 1X2 odds")
    if duplicate_count not in (0, None):
        details.append(f"duplicate logical fixture rows: {duplicate_count}")
    if before_hash != after_hash:
        details.append("raw file hash changed during read")

    row.update(
        {
            "rows": row_count,
            "schema_ok": schema_ok,
            "odds_source_available": odds_source,
            "date_min": dates.min().date().isoformat() if dates.notna().any() else None,
            "date_max": dates.max().date().isoformat() if dates.notna().any() else None,
            "date_range_2025_26_ok": date_range_ok,
            "valid_results": results_valid,
            "valid_1x2_odds": odds_valid,
            "duplicate_logical_fixtures": duplicate_count,
            "raw_file_modified": before_hash != after_hash,
            "status": "pass" if checks_ok else "fail",
            "details": "; ".join(details) if details else "ready for frozen V3 validation rerun",
        }
    )
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks = pd.DataFrame([check_file(league, path) for league, path in TARGETS.items()])
    checks.to_csv(OUT_PATH, index=False)

    if not checks["exists"].all():
        missing = checks.loc[~checks["exists"], "target_path"].tolist()
        print("Missing required raw files; stop before V3 validation rerun:")
        for path in missing:
            print(f"- {path}")
        return

    if not checks["status"].eq("pass").all():
        print("Raw files found but validation failed; stop before V3 validation rerun.")
        print(checks[["league", "status", "details"]].to_string(index=False))
        return

    print("E0/F1 raw files passed acquisition checks.")
    print("Next step: rerun the frozen V3 2025 validation pipeline without changing rules, thresholds, features, or training window.")


if __name__ == "__main__":
    main()
