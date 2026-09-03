from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from zipfile import ZipFile
import sys

import numpy as np
import pandas as pd


V2_MATRIX = Path("data/processed/features/football_feature_matrix_v2_transfermarkt_partial.csv")
CLUBELO_ZIP = Path("data/raw_external/clubelo_manual/clubelo_archive.zip")
MAPPING_CSV = Path("outputs/reports/clubelo_team_mapping_candidates.csv")
FEATURE_POLICY_CSV = Path("outputs/reports/clubelo_candidate_feature_policy.csv")
LOCKED_ROW_PREDICTIONS = Path("outputs/reports/feature_matrix_v2_tm_1x2_locked_row_predictions.csv")
LOCKED_SELECTED_BETS = Path("outputs/reports/feature_matrix_v2_tm_1x2_locked_selected_bets.csv")

OUT_MATRIX = Path("data/processed/features/football_feature_matrix_v3_clubelo_partial.csv")
REPORT_DIR = Path("outputs/reports")
BUILD_REPORT_MD = REPORT_DIR / "feature_matrix_v3_clubelo_build_report.md"
COVERAGE_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_coverage_by_league_season.csv"
STALENESS_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_staleness_audit.csv"
LOCKED_COVERAGE_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_locked_candidate_coverage.csv"
DICT_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_feature_dictionary_delta.csv"
LEAKAGE_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_leakage_checks.csv"
MAPPING_STATUS_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_mapping_status.csv"

LOWER_ENGLISH = {"E1", "E2", "E3"}
TOP_DIVISIONS = {"E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "SC0"}

CLUBELO_MODEL_COLUMNS = [
    "clubelo_home_rating",
    "clubelo_away_rating",
    "clubelo_diff",
    "clubelo_abs_diff",
    "clubelo_home_minus_internal_elo",
    "clubelo_away_minus_internal_elo",
    "clubelo_diff_minus_internal_elo_diff",
    "clubelo_staleness_home",
    "clubelo_staleness_away",
    "clubelo_missing_home",
    "clubelo_missing_away",
    "clubelo_missing_both",
    "clubelo_home_rating_date",
    "clubelo_away_rating_date",
    "clubelo_home_club_mapped_flag",
    "clubelo_away_club_mapped_flag",
    "clubelo_both_teams_mapped_flag",
    "clubelo_both_ratings_available_flag",
]


def md_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return view.to_markdown(index=False)


def status(ok: bool) -> str:
    return "pass" if ok else "fail"


def read_ratings() -> pd.DataFrame:
    with ZipFile(CLUBELO_ZIP) as zf:
        with zf.open("EloRatings.csv") as handle:
            ratings = pd.read_csv(handle, low_memory=False)
    ratings["date_parsed"] = pd.to_datetime(ratings["date"], errors="coerce")
    ratings["elo"] = pd.to_numeric(ratings["elo"], errors="coerce")
    return ratings


def accepted_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(MAPPING_CSV)
    accepted = mapping[
        mapping["mapping_status"].isin(["accepted_exact_normalized", "accepted_high_confidence_fuzzy"])
        & mapping["accepted_clubelo_club"].notna()
        & mapping["candidate_club"].eq(mapping["accepted_clubelo_club"])
    ][["league", "football_team", "accepted_clubelo_club", "mapping_status", "candidate_score"]].drop_duplicates()
    accepted = accepted.sort_values(["league", "football_team", "candidate_score"], ascending=[True, True, False]).drop_duplicates(["league", "football_team"], keep="first")
    return accepted


def build_rating_index(ratings: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    clean = ratings.dropna(subset=["club", "date_parsed", "elo"]).copy().sort_values(["club", "date_parsed"])
    out = {}
    for club, group in clean.groupby("club"):
        out[str(club)] = (
            group["date_parsed"].to_numpy(dtype="datetime64[ns]"),
            group["elo"].to_numpy(dtype=float),
        )
    return out


def latest_before(index: dict[str, tuple[np.ndarray, np.ndarray]], club: object, match_date: pd.Timestamp) -> tuple[float, object, float]:
    if pd.isna(club) or str(club) not in index or pd.isna(match_date):
        return np.nan, pd.NaT, np.nan
    dates, values = index[str(club)]
    pos = bisect_left(dates, np.datetime64(match_date)) - 1
    if pos < 0:
        return np.nan, pd.NaT, np.nan
    rating_date = pd.Timestamp(dates[pos])
    return float(values[pos]), rating_date.date().isoformat(), float((match_date - rating_date).days)


def join_mappings(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    home = mapping.rename(
        columns={
            "football_team": "home_team",
            "accepted_clubelo_club": "_clubelo_home_club_audit",
            "mapping_status": "_clubelo_home_mapping_status",
            "candidate_score": "_clubelo_home_mapping_score",
        }
    )[["league", "home_team", "_clubelo_home_club_audit", "_clubelo_home_mapping_status", "_clubelo_home_mapping_score"]]
    away = mapping.rename(
        columns={
            "football_team": "away_team",
            "accepted_clubelo_club": "_clubelo_away_club_audit",
            "mapping_status": "_clubelo_away_mapping_status",
            "candidate_score": "_clubelo_away_mapping_score",
        }
    )[["league", "away_team", "_clubelo_away_club_audit", "_clubelo_away_mapping_status", "_clubelo_away_mapping_score"]]
    out = df[["match_id", "match_date", "league", "season_start_year", "season_end_year", "home_team", "away_team"]].copy()
    out = out.merge(home, on=["league", "home_team"], how="left", validate="many_to_one")
    out = out.merge(away, on=["league", "away_team"], how="left", validate="many_to_one")
    return out


def build_clubelo_columns(df: pd.DataFrame, ratings: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = join_mappings(df, mapping)
    index = build_rating_index(ratings)
    home_rating = []
    away_rating = []
    home_date = []
    away_date = []
    home_stale = []
    away_stale = []
    for match_date_raw, home_club, away_club in zip(
        joined["match_date"],
        joined["_clubelo_home_club_audit"],
        joined["_clubelo_away_club_audit"],
    ):
        match_date = pd.Timestamp(match_date_raw)
        hr, hd, hs = latest_before(index, home_club, match_date)
        ar, ad, aas = latest_before(index, away_club, match_date)
        home_rating.append(hr)
        away_rating.append(ar)
        home_date.append(hd)
        away_date.append(ad)
        home_stale.append(hs)
        away_stale.append(aas)
    features = pd.DataFrame(index=df.index)
    features["clubelo_home_rating"] = home_rating
    features["clubelo_away_rating"] = away_rating
    features["clubelo_diff"] = features["clubelo_home_rating"] - features["clubelo_away_rating"]
    features["clubelo_abs_diff"] = features["clubelo_diff"].abs()
    features["clubelo_home_minus_internal_elo"] = features["clubelo_home_rating"] - pd.to_numeric(df["home_elo"], errors="coerce")
    features["clubelo_away_minus_internal_elo"] = features["clubelo_away_rating"] - pd.to_numeric(df["away_elo"], errors="coerce")
    features["clubelo_diff_minus_internal_elo_diff"] = features["clubelo_diff"] - pd.to_numeric(df["elo_diff"], errors="coerce")
    features["clubelo_staleness_home"] = home_stale
    features["clubelo_staleness_away"] = away_stale
    features["clubelo_missing_home"] = features["clubelo_home_rating"].isna()
    features["clubelo_missing_away"] = features["clubelo_away_rating"].isna()
    features["clubelo_missing_both"] = features["clubelo_missing_home"] | features["clubelo_missing_away"]
    features["clubelo_home_rating_date"] = home_date
    features["clubelo_away_rating_date"] = away_date
    features["clubelo_home_club_mapped_flag"] = joined["_clubelo_home_club_audit"].notna().to_numpy()
    features["clubelo_away_club_mapped_flag"] = joined["_clubelo_away_club_audit"].notna().to_numpy()
    features["clubelo_both_teams_mapped_flag"] = features["clubelo_home_club_mapped_flag"] & features["clubelo_away_club_mapped_flag"]
    features["clubelo_both_ratings_available_flag"] = features["clubelo_home_rating"].notna() & features["clubelo_away_rating"].notna()
    audit = pd.concat([joined.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    return features, audit


def segment_summary(frame: pd.DataFrame, segment: str, group_col: str | None = None) -> list[dict[str, object]]:
    rows = []
    groups = [(segment, frame)] if group_col is None else [(str(k), g) for k, g in frame.groupby(group_col, dropna=False)]
    for group, g in groups:
        staleness = pd.concat(
            [
                g.loc[g["clubelo_home_rating"].notna(), "clubelo_staleness_home"],
                g.loc[g["clubelo_away_rating"].notna(), "clubelo_staleness_away"],
            ]
        )
        rows.append(
            {
                "segment": segment,
                "group": group,
                "rows": int(len(g)),
                "home_mapped": int(g["clubelo_home_club_mapped_flag"].sum()),
                "away_mapped": int(g["clubelo_away_club_mapped_flag"].sum()),
                "both_mapped": int(g["clubelo_both_teams_mapped_flag"].sum()),
                "home_rating_available": int(g["clubelo_home_rating"].notna().sum()),
                "away_rating_available": int(g["clubelo_away_rating"].notna().sum()),
                "both_ratings_available": int(g["clubelo_both_ratings_available_flag"].sum()),
                "both_rating_coverage": float(g["clubelo_both_ratings_available_flag"].mean()) if len(g) else np.nan,
                "mean_staleness_days": float(staleness.mean()) if len(staleness) else np.nan,
                "median_staleness_days": float(staleness.median()) if len(staleness) else np.nan,
                "max_staleness_days": float(staleness.max()) if len(staleness) else np.nan,
            }
        )
    return rows


def coverage_reports(v3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    rows.extend(segment_summary(v3, "all_rows"))
    rows.extend(segment_summary(v3[v3["season_start_year"].between(2014, 2026)], "modern_2014_2026"))
    rows.extend(segment_summary(v3[v3["league"].isin(TOP_DIVISIONS)], "top_divisions_only"))
    rows.extend(segment_summary(v3[~v3["league"].isin(LOWER_ENGLISH)], "excluding_E1_E2_E3"))
    rows.extend(segment_summary(v3, "by_league", "league"))
    rows.extend(segment_summary(v3, "by_season_start_year", "season_start_year"))
    coverage = pd.DataFrame(rows)
    coverage.to_csv(COVERAGE_CSV, index=False)

    stale_rows = []
    stale_rows.extend(segment_summary(v3, "staleness_by_season_start_year", "season_start_year"))
    season_2025 = v3[v3["season_start_year"].eq(2025)]
    season_end_2026 = v3[v3["season_end_year"].eq(2026)]
    calendar_2026 = v3[pd.to_datetime(v3["match_date"], errors="coerce").dt.year.eq(2026)]
    stale_rows.extend(segment_summary(season_2025, "season_start_year_2025"))
    stale_rows.extend(segment_summary(season_end_2026, "season_end_year_2026"))
    stale_rows.extend(segment_summary(calendar_2026, "calendar_year_2026"))
    staleness = pd.DataFrame(stale_rows)
    staleness.to_csv(STALENESS_CSV, index=False)

    locked_rows = []
    if LOCKED_ROW_PREDICTIONS.exists():
        pred_ids = set(pd.read_csv(LOCKED_ROW_PREDICTIONS, usecols=["match_id"])["match_id"])
        locked_rows.extend(segment_summary(v3[v3["match_id"].isin(pred_ids)], "locked_v2_prediction_row_universe"))
    if LOCKED_SELECTED_BETS.exists():
        bet_ids = set(pd.read_csv(LOCKED_SELECTED_BETS, usecols=["match_id"])["match_id"])
        locked_rows.extend(segment_summary(v3[v3["match_id"].isin(bet_ids)], "locked_v2_selected_bets"))
    locked = pd.DataFrame(locked_rows)
    locked.to_csv(LOCKED_COVERAGE_CSV, index=False)
    return coverage, staleness, locked


def feature_dictionary() -> pd.DataFrame:
    definitions = {
        "clubelo_home_rating": ("float", "Latest ClubElo rating strictly before match_date for mapped home club.", True, "low if strict-before join is preserved", "Missing if no accepted mapping or no prior rating."),
        "clubelo_away_rating": ("float", "Latest ClubElo rating strictly before match_date for mapped away club.", True, "low if strict-before join is preserved", "Missing if no accepted mapping or no prior rating."),
        "clubelo_diff": ("float", "clubelo_home_rating - clubelo_away_rating.", True, "low", "Derived only from safe ratings."),
        "clubelo_abs_diff": ("float", "Absolute value of clubelo_diff.", True, "low", "Derived only from safe ratings."),
        "clubelo_home_minus_internal_elo": ("float", "clubelo_home_rating - existing internal home_elo.", True, "low", "Existing internal Elo is already in v2."),
        "clubelo_away_minus_internal_elo": ("float", "clubelo_away_rating - existing internal away_elo.", True, "low", "Existing internal Elo is already in v2."),
        "clubelo_diff_minus_internal_elo_diff": ("float", "clubelo_diff - existing internal elo_diff.", True, "low", "Derived only from safe ratings and v2 internal Elo."),
        "clubelo_staleness_home": ("float", "Days between match_date and home rating date.", True, "low", "Minimum should be >= 1."),
        "clubelo_staleness_away": ("float", "Days between match_date and away rating date.", True, "low", "Minimum should be >= 1."),
        "clubelo_missing_home": ("bool", "Home ClubElo rating missing after date-safe lookup.", True, "low", "Explicit missingness flag; no fabricated rating."),
        "clubelo_missing_away": ("bool", "Away ClubElo rating missing after date-safe lookup.", True, "low", "Explicit missingness flag; no fabricated rating."),
        "clubelo_missing_both": ("bool", "Either home or away ClubElo rating missing.", True, "low", "Explicit missingness flag; no fabricated rating."),
        "clubelo_home_rating_date": ("date/string", "Date of joined home rating.", False, "medium if used directly", "Audit/debug column; not a model feature."),
        "clubelo_away_rating_date": ("date/string", "Date of joined away rating.", False, "medium if used directly", "Audit/debug column; not a model feature."),
        "clubelo_home_club_mapped_flag": ("bool", "Accepted ClubElo mapping exists for home team.", True, "low", "No club name is stored in model matrix."),
        "clubelo_away_club_mapped_flag": ("bool", "Accepted ClubElo mapping exists for away team.", True, "low", "No club name is stored in model matrix."),
        "clubelo_both_teams_mapped_flag": ("bool", "Accepted ClubElo mappings exist for both teams.", True, "low", "No club names are stored in model matrix."),
        "clubelo_both_ratings_available_flag": ("bool", "Date-safe ClubElo ratings exist for both teams.", True, "low", "No fabricated ratings."),
    }
    rows = [
        {
            "column": col,
            "type": definitions[col][0],
            "definition": definitions[col][1],
            "allowed_as_model_feature": definitions[col][2],
            "leakage_risk": definitions[col][3],
            "notes": definitions[col][4],
        }
        for col in CLUBELO_MODEL_COLUMNS
    ]
    out = pd.DataFrame(rows)
    out.to_csv(DICT_CSV, index=False)
    return out


def leakage_checks(v2: pd.DataFrame, v3: pd.DataFrame, audit: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    match_dates = pd.to_datetime(v2["match_date"], errors="coerce")
    home_dates = pd.to_datetime(features["clubelo_home_rating_date"], errors="coerce")
    away_dates = pd.to_datetime(features["clubelo_away_rating_date"], errors="coerce")
    home_rating_exists = features["clubelo_home_rating"].notna()
    away_rating_exists = features["clubelo_away_rating"].notna()
    date_safe_home = (home_dates[home_rating_exists] < match_dates[home_rating_exists]).all()
    date_safe_away = (away_dates[away_rating_exists] < match_dates[away_rating_exists]).all()
    same_day = int(((home_dates == match_dates) & home_rating_exists).sum() + ((away_dates == match_dates) & away_rating_exists).sum())
    future = int(((home_dates > match_dates) & home_rating_exists).sum() + ((away_dates > match_dates) & away_rating_exists).sum())
    added_cols = [c for c in v3.columns if c not in v2.columns]
    forbidden_patterns = ["FTHome", "FTAway", "FTResult", "HomeShots", "AwayShots", "HomeTarget", "AwayTarget", "HTHome", "HTAway", "HTResult"]
    forbidden_added = [c for c in added_cols if any(pat.lower() in c.lower() for pat in forbidden_patterns)]
    name_cols = [
        c
        for c in added_cols
        if c not in {"clubelo_home_rating_date", "clubelo_away_rating_date"}
        and pd.api.types.is_object_dtype(v3[c])
        and any(tok in c.lower() for tok in ["team", "club", "name"])
    ]
    fabricated_home = int((features["clubelo_home_rating"].notna() & ~features["clubelo_home_club_mapped_flag"]).sum())
    fabricated_away = int((features["clubelo_away_rating"].notna() & ~features["clubelo_away_club_mapped_flag"]).sum())
    unchanged_cols = [c for c in v2.columns if c in v3.columns]
    unchanged = v2[unchanged_cols].equals(v3[unchanged_cols])
    rows = [
        ("all_home_rating_dates_strictly_before_match_date", date_safe_home, int(home_rating_exists.sum()), ""),
        ("all_away_rating_dates_strictly_before_match_date", date_safe_away, int(away_rating_exists.sum()), ""),
        ("no_same_day_rating_joins", same_day == 0, same_day, ""),
        ("no_future_rating_joins", future == 0, future, ""),
        ("no_clubelo_matches_result_score_postmatch_columns_added", len(forbidden_added) == 0, len(forbidden_added), "|".join(forbidden_added)),
        ("no_team_or_club_name_string_columns_added_as_model_features", len(name_cols) == 0, len(name_cols), "|".join(name_cols)),
        ("no_fabricated_home_ratings", fabricated_home == 0, fabricated_home, ""),
        ("no_fabricated_away_ratings", fabricated_away == 0, fabricated_away, ""),
        ("v2_row_count_preserved", len(v2) == len(v3), len(v3), f"v2={len(v2)} v3={len(v3)}"),
        ("v2_column_values_unchanged", unchanged, len(unchanged_cols), ""),
        ("only_allowed_clubelo_columns_added", set(added_cols) == set(CLUBELO_MODEL_COLUMNS), len(added_cols), "|".join(sorted(set(added_cols) - set(CLUBELO_MODEL_COLUMNS)))),
    ]
    out = pd.DataFrame([{"check": name, "status": status(bool(ok)), "count": int(count), "detail": detail} for name, ok, count, detail in rows])
    out.to_csv(LEAKAGE_CSV, index=False)
    return out


def mapping_status(audit: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "match_id",
        "match_date",
        "league",
        "season_start_year",
        "home_team",
        "away_team",
        "_clubelo_home_club_audit",
        "_clubelo_away_club_audit",
        "_clubelo_home_mapping_status",
        "_clubelo_away_mapping_status",
        "_clubelo_home_mapping_score",
        "_clubelo_away_mapping_score",
        "clubelo_home_rating",
        "clubelo_away_rating",
        "clubelo_home_rating_date",
        "clubelo_away_rating_date",
        "clubelo_staleness_home",
        "clubelo_staleness_away",
        "clubelo_both_ratings_available_flag",
    ]
    out = audit[cols].copy()
    out.to_csv(MAPPING_STATUS_CSV, index=False)
    return out


def decide(coverage: pd.DataFrame, locked: pd.DataFrame, checks: pd.DataFrame) -> str:
    if checks["status"].ne("pass").any():
        return "feature_matrix_v3_build_failed"
    top = coverage[(coverage["segment"].eq("top_divisions_only")) & (coverage["group"].eq("top_divisions_only"))]
    locked_pred = locked[locked["segment"].eq("locked_v2_prediction_row_universe")]
    locked_bets = locked[locked["segment"].eq("locked_v2_selected_bets")]
    top_cov = float(top["both_rating_coverage"].iloc[0]) if len(top) else 0.0
    locked_pred_cov = float(locked_pred["both_rating_coverage"].iloc[0]) if len(locked_pred) else np.nan
    locked_bets_cov = float(locked_bets["both_rating_coverage"].iloc[0]) if len(locked_bets) else np.nan
    if top_cov >= 0.97 and (pd.isna(locked_pred_cov) or locked_pred_cov >= 0.97) and (pd.isna(locked_bets_cov) or locked_bets_cov >= 0.97):
        return "feature_matrix_v3_build_ready_good"
    if top_cov >= 0.90 and (pd.isna(locked_bets_cov) or locked_bets_cov >= 0.90):
        return "feature_matrix_v3_build_ready_partial"
    return "feature_matrix_v3_build_failed"


def write_report(decision: str, coverage: pd.DataFrame, staleness: pd.DataFrame, locked: pd.DataFrame, checks: pd.DataFrame, v2_rows: int, v3_rows: int) -> None:
    key_cov = coverage[coverage["segment"].isin(["all_rows", "modern_2014_2026", "top_divisions_only", "excluding_E1_E2_E3"])]
    lines = [
        "# Feature Matrix V3 ClubElo Partial Build Report",
        "",
        f"Decision: `{decision}`",
        "",
        "No predictive models, value searches, threshold optimization, or locked v2 candidate changes were run. No confirmed edge is claimed.",
        "",
        "## Build Summary",
        f"- Input v2 rows: {v2_rows}",
        f"- Output v3 rows: {v3_rows}",
        f"- Added ClubElo columns: {len(CLUBELO_MODEL_COLUMNS)}",
        f"- Output matrix: `{OUT_MATRIX}`",
        "",
        "## Key Coverage",
        md_table(key_cov[["segment", "rows", "both_ratings_available", "both_rating_coverage", "median_staleness_days", "max_staleness_days"]]),
        "",
        "## Locked Candidate Coverage",
        md_table(locked[["segment", "rows", "both_ratings_available", "both_rating_coverage", "median_staleness_days", "max_staleness_days"]]),
        "",
        "## 2025/26 Staleness",
        md_table(staleness[staleness["segment"].isin(["season_start_year_2025", "season_end_year_2026", "calendar_year_2026"])][["segment", "rows", "both_ratings_available", "both_rating_coverage", "mean_staleness_days", "median_staleness_days", "max_staleness_days"]]),
        "",
        "## Leakage Checks",
        md_table(checks, 80),
        "",
    ]
    BUILD_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    v2 = pd.read_csv(V2_MATRIX, low_memory=False)
    v2["match_date"] = pd.to_datetime(v2["match_date"], errors="coerce")
    ratings = read_ratings()
    mapping = accepted_mapping()
    features, audit = build_clubelo_columns(v2, ratings, mapping)
    v3 = pd.concat([v2.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    coverage, staleness, locked = coverage_reports(v3)
    dictionary = feature_dictionary()
    checks = leakage_checks(v2.reset_index(drop=True), v3, audit, features)
    mapping_status(audit)
    decision = decide(coverage, locked, checks)
    v3.to_csv(OUT_MATRIX, index=False)
    write_report(decision, coverage, staleness, locked, checks, len(v2), len(v3))

    key = coverage[coverage["segment"].isin(["modern_2014_2026", "top_divisions_only", "excluding_E1_E2_E3"])]
    locked_bets = locked[locked["segment"].eq("locked_v2_selected_bets")]
    print(
        {
            "decision": decision,
            "v2_rows": len(v2),
            "v3_rows": len(v3),
            "added_columns": len(CLUBELO_MODEL_COLUMNS),
            "coverage": {r.segment: round(float(r.both_rating_coverage), 6) for r in key.itertuples(index=False)},
            "locked_selected_bets_coverage": round(float(locked_bets["both_rating_coverage"].iloc[0]), 6) if len(locked_bets) else None,
            "failed_checks": int(checks["status"].ne("pass").sum()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
