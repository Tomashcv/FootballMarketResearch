from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from zipfile import ZipFile
import hashlib
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from difflib import SequenceMatcher


V3_MATRIX = Path("data/processed/features/football_feature_matrix_v3_clubelo_partial.csv")
UNDERSTAT_ZIP = Path("data/raw_external/understat_manual/understat_archive.zip")
LOCKED_ROW_PREDICTIONS = Path("outputs/reports/feature_matrix_v2_tm_1x2_locked_row_predictions.csv")
LOCKED_V3_SELECTED_BETS = Path("outputs/reports/feature_matrix_v3_clubelo_locked_selected_bets.csv")

OUT_MATRIX = Path("data/processed/features/football_feature_matrix_v4_1_understat_partial.csv")
REPORT_DIR = Path("outputs/reports")

BUILD_REPORT_MD = REPORT_DIR / "feature_matrix_v4_1_understat_build_report.md"
MAPPING_CSV = REPORT_DIR / "understat_team_mapping_candidates.csv"
DATE_SAFETY_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_date_safety_audit.csv"
COVERAGE_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_coverage_by_league_season.csv"
MISSINGNESS_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_missingness.csv"
DICT_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_feature_dictionary_delta.csv"
LEAKAGE_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_leakage_checks.csv"
SCOPE_MD = REPORT_DIR / "feature_matrix_v4_1_understat_recommended_model_scope.md"

LEAGUE_TO_UNDERSTAT = {
    "E0": "EPL",
    "D1": "Bundesliga",
    "SP1": "La liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}
TOP5 = set(LEAGUE_TO_UNDERSTAT)
WINDOWS = [5, 10, 20]
ROLL_METRICS = {
    "xg_for": "xG",
    "xg_against": "xGA",
    "npxg_for": "npxG",
    "npxg_against": "npxGA",
    "xg_diff": "_xg_diff",
    "npxg_diff": "npxGD",
    "ppda": "ppda",
    "ppda_allowed": "ppda_allowed",
    "deep": "deep",
    "deep_allowed": "deep_allowed",
    "xpts": "xpts",
    "points": "pts",
    "goal_diff": "_goal_diff",
}
LEGAL_WORDS = {
    "afc",
    "as",
    "athletic",
    "calcio",
    "cf",
    "club",
    "de",
    "fc",
    "football",
    "futbol",
    "futebol",
    "real",
    "sc",
    "sport",
    "sporting",
    "the",
    "u",
    "ud",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [tok for tok in text.split() if tok not in LEGAL_WORDS]
    return " ".join(tokens)


def score_name(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    token_left = " ".join(sorted(left.split()))
    token_right = " ".join(sorted(right.split()))
    return max(SequenceMatcher(None, left, right).ratio(), SequenceMatcher(None, token_left, token_right).ratio())


def read_understat() -> pd.DataFrame:
    with ZipFile(UNDERSTAT_ZIP) as zf:
        with zf.open("game_stats.csv") as handle:
            game = pd.read_csv(handle, low_memory=False)
    game["understat_date"] = pd.to_datetime(game["date"], errors="coerce")
    for col in ["xG", "xGA", "npxG", "npxGA", "ppda", "ppda_allowed", "deep", "deep_allowed", "xpts", "pts", "scored", "missed", "npxGD"]:
        game[col] = pd.to_numeric(game[col], errors="coerce")
    game["_xg_diff"] = game["xG"] - game["xGA"]
    game["_goal_diff"] = game["scored"] - game["missed"]
    return game


def understat_schema_info(game: pd.DataFrame) -> dict[str, object]:
    numeric_issues = {}
    for col in ["xG", "xGA", "npxG", "npxGA", "ppda", "ppda_allowed", "deep", "deep_allowed", "xpts", "pts", "scored", "missed", "npxGD"]:
        numeric_issues[col] = int(pd.to_numeric(game[col], errors="coerce").isna().sum() - game[col].isna().sum())
    return {
        "columns": list(game.drop(columns=["understat_date", "_xg_diff", "_goal_diff"]).columns),
        "rows": int(len(game)),
        "date_min": game["understat_date"].min(),
        "date_max": game["understat_date"].max(),
        "leagues": sorted(game["league"].dropna().unique()),
        "seasons": sorted(game["season"].dropna().unique()),
        "teams": int(game["club_name"].nunique()),
        "duplicate_team_date_home_away": int(game.duplicated(["league", "season", "club_name", "home_away", "date"]).sum()),
        "missing_values": {c: int(v) for c, v in game.drop(columns=["understat_date", "_xg_diff", "_goal_diff"]).isna().sum().items() if int(v) > 0},
        "numeric_parse_issues": {k: v for k, v in numeric_issues.items() if v > 0},
    }


def build_mapping(v3: pd.DataFrame, game: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    under_clubs = game[["league", "club_name"]].drop_duplicates().copy()
    under_clubs["club_norm"] = under_clubs["club_name"].map(normalize_name)
    club_by_league = {k: g.reset_index(drop=True) for k, g in under_clubs.groupby("league")}
    teams = pd.concat(
        [
            v3[v3["league"].isin(TOP5)][["league", "home_team"]].rename(columns={"home_team": "team"}),
            v3[v3["league"].isin(TOP5)][["league", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    teams["understat_league"] = teams["league"].map(LEAGUE_TO_UNDERSTAT)
    teams["team_norm"] = teams["team"].map(normalize_name)
    rows = []
    accepted = []
    for team in teams.sort_values(["league", "team"]).itertuples(index=False):
        candidates = club_by_league.get(team.understat_league, pd.DataFrame(columns=["league", "club_name", "club_norm"]))
        exact = candidates[candidates["club_norm"].eq(team.team_norm)]
        if len(exact) == 1:
            status = "accepted_exact_normalized"
            mapped = str(exact["club_name"].iloc[0])
            top = exact.assign(score=1.0).head(1)
        else:
            scored = candidates.copy()
            scored["score"] = scored["club_norm"].map(lambda x: score_name(team.team_norm, x))
            top = scored.sort_values(["score", "club_name"], ascending=[False, True]).head(5)
            top_score = float(top["score"].iloc[0]) if len(top) else 0.0
            second = float(top["score"].iloc[1]) if len(top) > 1 else 0.0
            if len(exact) > 1:
                status = "ambiguous_exact_normalized"
                mapped = ""
            elif top_score >= 0.94 and top_score - second >= 0.03:
                status = "accepted_high_confidence_fuzzy"
                mapped = str(top["club_name"].iloc[0])
            elif top_score >= 0.82:
                status = "fuzzy_candidates_review_required"
                mapped = ""
            else:
                status = "unmatched"
                mapped = ""
        if mapped:
            accepted.append(
                {
                    "league": team.league,
                    "football_team": team.team,
                    "understat_league": team.understat_league,
                    "understat_club_name": mapped,
                    "mapping_status": status,
                }
            )
        if len(top) == 0:
            rows.append(
                {
                    "league": team.league,
                    "football_team": team.team,
                    "understat_league": team.understat_league,
                    "team_norm": team.team_norm,
                    "mapping_status": status,
                    "accepted_understat_club_name": mapped,
                    "candidate_rank": np.nan,
                    "candidate_club_name": "",
                    "candidate_score": np.nan,
                }
            )
        else:
            for rank, cand in enumerate(top.itertuples(index=False), start=1):
                rows.append(
                    {
                        "league": team.league,
                        "football_team": team.team,
                        "understat_league": team.understat_league,
                        "team_norm": team.team_norm,
                        "mapping_status": status,
                        "accepted_understat_club_name": mapped,
                        "candidate_rank": rank,
                        "candidate_club_name": cand.club_name,
                        "candidate_score": float(cand.score),
                    }
                )
    mapping = pd.DataFrame(rows)
    accepted_map = pd.DataFrame(accepted).drop_duplicates(["league", "football_team"])
    mapping.to_csv(MAPPING_CSV, index=False)
    return mapping, accepted_map


def attach_mapping(v3: pd.DataFrame, accepted: pd.DataFrame) -> pd.DataFrame:
    home = accepted.rename(
        columns={
            "football_team": "home_team",
            "understat_league": "_understat_home_league",
            "understat_club_name": "_understat_home_club",
            "mapping_status": "_understat_home_mapping_status",
        }
    )[["league", "home_team", "_understat_home_league", "_understat_home_club", "_understat_home_mapping_status"]]
    away = accepted.rename(
        columns={
            "football_team": "away_team",
            "understat_league": "_understat_away_league",
            "understat_club_name": "_understat_away_club",
            "mapping_status": "_understat_away_mapping_status",
        }
    )[["league", "away_team", "_understat_away_league", "_understat_away_club", "_understat_away_mapping_status"]]
    audit = v3[["match_id", "match_date", "league", "season_start_year", "season_end_year", "home_team", "away_team"]].copy()
    audit = audit.merge(home, on=["league", "home_team"], how="left", validate="many_to_one")
    audit = audit.merge(away, on=["league", "away_team"], how="left", validate="many_to_one")
    audit["_understat_home_mapped"] = audit["_understat_home_club"].notna()
    audit["_understat_away_mapped"] = audit["_understat_away_club"].notna()
    return audit


def build_history_index(game: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    cols = ["understat_date"] + list(ROLL_METRICS.values())
    out = {}
    for (league, club), g in game.sort_values("understat_date").groupby(["league", "club_name"]):
        out[(str(league), str(club))] = g[cols].reset_index(drop=True)
    return out


def prior_features(history: pd.DataFrame | None, match_date: pd.Timestamp) -> tuple[dict[str, float], int, float, pd.Timestamp | pd.NaT]:
    vals: dict[str, float] = {}
    if history is None or history.empty or pd.isna(match_date):
        for window in WINDOWS:
            for metric in ROLL_METRICS:
                vals[f"understat_{metric}_roll{window}"] = np.nan
        return vals, 0, np.nan, pd.NaT
    dates = history["understat_date"].to_numpy(dtype="datetime64[ns]")
    pos = bisect_left(dates, np.datetime64(match_date))
    prior = history.iloc[:pos]
    count = int(len(prior))
    latest = pd.NaT if count == 0 else prior["understat_date"].iloc[-1]
    days = np.nan if count == 0 else float((match_date - latest).total_seconds() / 86400.0)
    for window in WINDOWS:
        tail = prior.tail(window)
        for metric, source in ROLL_METRICS.items():
            vals[f"understat_{metric}_roll{window}"] = float(tail[source].mean()) if len(tail) else np.nan
    return vals, count, days, latest


def build_features(v3: pd.DataFrame, audit: pd.DataFrame, game: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = build_history_index(game)
    feature_rows = []
    safety_rows = []
    for row in audit.to_dict("records"):
        match_date = pd.Timestamp(row["match_date"])
        home_hist = index.get((str(row["_understat_home_league"]), str(row["_understat_home_club"])))
        away_hist = index.get((str(row["_understat_away_league"]), str(row["_understat_away_club"])))
        home_vals, home_count, home_days, home_latest = prior_features(home_hist, match_date)
        away_vals, away_count, away_days, away_latest = prior_features(away_hist, match_date)
        out = {}
        for key in home_vals:
            out[f"home_{key}"] = home_vals[key]
            out[f"away_{key}"] = away_vals[key]
            out[f"home_minus_away_{key}"] = home_vals[key] - away_vals[key] if pd.notna(home_vals[key]) and pd.notna(away_vals[key]) else np.nan
        out["understat_home_history_count"] = home_count
        out["understat_away_history_count"] = away_count
        out["understat_home_latest_days_ago"] = home_days
        out["understat_away_latest_days_ago"] = away_days
        out["understat_home_available_flag"] = home_count > 0
        out["understat_away_available_flag"] = away_count > 0
        out["understat_both_available_flag"] = home_count > 0 and away_count > 0
        feature_rows.append(out)
        safety_rows.append(
            {
                "match_id": row["match_id"],
                "match_date": match_date,
                "league": row["league"],
                "season_start_year": row["season_start_year"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_mapped": bool(row["_understat_home_mapped"]),
                "away_mapped": bool(row["_understat_away_mapped"]),
                "home_prior_history_count": home_count,
                "away_prior_history_count": away_count,
                "both_prior_history_available": home_count > 0 and away_count > 0,
                "home_latest_understat_date": home_latest,
                "away_latest_understat_date": away_latest,
                "home_latest_days_ago": home_days,
                "away_latest_days_ago": away_days,
            }
        )
    features = pd.DataFrame(feature_rows, index=v3.index)
    safety = pd.DataFrame(safety_rows)
    return features, safety


def segment_summary(frame: pd.DataFrame, segment: str, group_col: str | None = None) -> list[dict[str, object]]:
    rows = []
    groups = [(segment, frame)] if group_col is None else [(str(k), g) for k, g in frame.groupby(group_col, dropna=False)]
    for group, g in groups:
        rows.append(
            {
                "segment": segment,
                "group": group,
                "rows": int(len(g)),
                "home_available": int(g["understat_home_available_flag"].sum()) if len(g) else 0,
                "away_available": int(g["understat_away_available_flag"].sum()) if len(g) else 0,
                "both_available": int(g["understat_both_available_flag"].sum()) if len(g) else 0,
                "both_available_rate": float(g["understat_both_available_flag"].mean()) if len(g) else np.nan,
                "home_history_count_median": float(g["understat_home_history_count"].median()) if len(g) else np.nan,
                "away_history_count_median": float(g["understat_away_history_count"].median()) if len(g) else np.nan,
                "latest_days_ago_median": float(pd.concat([g["understat_home_latest_days_ago"], g["understat_away_latest_days_ago"]]).median()) if len(g) else np.nan,
                "latest_days_ago_max": float(pd.concat([g["understat_home_latest_days_ago"], g["understat_away_latest_days_ago"]]).max()) if len(g) else np.nan,
            }
        )
    return rows


def coverage_reports(v4: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rows.extend(segment_summary(v4, "all_rows"))
    rows.extend(segment_summary(v4[v4["league"].isin(TOP5)], "top5_only"))
    rows.extend(segment_summary(v4[v4["league"].isin(TOP5) & v4["season_start_year"].between(2020, 2025)], "top5_test_2020_2025"))
    rows.extend(segment_summary(v4[v4["season_start_year"].eq(2024)], "season_start_year_2024"))
    rows.extend(segment_summary(v4[v4["season_start_year"].eq(2025)], "season_start_year_2025"))
    rows.extend(segment_summary(v4[pd.to_datetime(v4["match_date"]).gt(pd.Timestamp("2024-09-29"))], "after_understat_max_date_2024_09_29"))
    rows.extend(segment_summary(v4, "by_league", "league"))
    rows.extend(segment_summary(v4, "by_season_start_year", "season_start_year"))
    if LOCKED_ROW_PREDICTIONS.exists():
        ids = set(pd.read_csv(LOCKED_ROW_PREDICTIONS, usecols=["match_id"])["match_id"])
        rows.extend(segment_summary(v4[v4["match_id"].isin(ids)], "locked_v3_prediction_row_universe"))
    if LOCKED_V3_SELECTED_BETS.exists():
        ids = set(pd.read_csv(LOCKED_V3_SELECTED_BETS, usecols=["match_id"])["match_id"])
        rows.extend(segment_summary(v4[v4["match_id"].isin(ids)], "locked_v3_selected_bets"))
    coverage = pd.DataFrame(rows)
    coverage.to_csv(COVERAGE_CSV, index=False)
    missing_rows = []
    for col in features.columns:
        missing_rows.append(
            {
                "column": col,
                "missing": int(features[col].isna().sum()),
                "missing_rate": float(features[col].isna().mean()),
                "non_null": int(features[col].notna().sum()),
            }
        )
    missing = pd.DataFrame(missing_rows)
    missing.to_csv(MISSINGNESS_CSV, index=False)
    return coverage, missing


def mapping_coverage(mapping: pd.DataFrame, v3: pd.DataFrame) -> pd.DataFrame:
    teams = mapping.drop_duplicates(["league", "football_team"])
    status_counts = teams.groupby(["league", "mapping_status"]).size().rename("teams").reset_index()
    # Add fixture coverage rows using accepted statuses only.
    accepted_status = {"accepted_exact_normalized", "accepted_high_confidence_fuzzy"}
    return status_counts


def feature_dictionary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in features.columns:
        allowed = not col.endswith("_date") and not any(tok in col.lower() for tok in ["club", "team", "result", "scored", "missed"])
        if col.startswith("home_minus_away_"):
            definition = "Home minus away difference of rolling past-only Understat feature."
        elif col.startswith("home_"):
            definition = "Home team rolling past-only Understat feature."
        elif col.startswith("away_"):
            definition = "Away team rolling past-only Understat feature."
        else:
            definition = "Understat availability or staleness audit feature."
        rows.append(
            {
                "column": col,
                "type": str(features[col].dtype),
                "definition": definition,
                "allowed_as_model_feature": bool(allowed),
                "leakage_risk": "low_if_strict_past_only" if allowed else "audit_or_excluded",
                "notes": "Computed only from Understat rows with understat_date < match_date; missing remains missing.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DICT_CSV, index=False)
    return out


def leakage_checks(v3: pd.DataFrame, v4: pd.DataFrame, features: pd.DataFrame, safety: pd.DataFrame, raw_before: tuple[int, int, str], raw_after: tuple[int, int, str]) -> pd.DataFrame:
    added = [c for c in v4.columns if c not in v3.columns]
    original_bad = []
    for col in v3.columns:
        left = v3[col]
        right = v4[col]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            ok = np.allclose(pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce"), equal_nan=True, rtol=1e-12, atol=1e-12)
        else:
            ok = left.astype("string").fillna("<NA>").equals(right.astype("string").fillna("<NA>"))
        if not ok:
            original_bad.append(col)
    match_date = pd.to_datetime(safety["match_date"])
    home_latest = pd.to_datetime(safety["home_latest_understat_date"], errors="coerce")
    away_latest = pd.to_datetime(safety["away_latest_understat_date"], errors="coerce")
    home_exists = home_latest.notna()
    away_exists = away_latest.notna()
    same_day = int(((home_latest.dt.normalize() == match_date.dt.normalize()) & home_exists).sum() + ((away_latest.dt.normalize() == match_date.dt.normalize()) & away_exists).sum())
    future = int(((home_latest >= match_date) & home_exists).sum() + ((away_latest >= match_date) & away_exists).sum())
    bad_direct = [c for c in added if re.search(r"(^|_)result($|_)|scored|missed|current", c, re.I)]
    bad_names = [c for c in added if pd.api.types.is_object_dtype(v4[c]) and re.search(r"team|club|name", c, re.I)]
    rows = [
        ("all_contributing_understat_rows_strictly_before_match_date", future == 0, int(home_exists.sum() + away_exists.sum()), ""),
        ("no_same_day_understat_joins", same_day == 0, same_day, ""),
        ("no_future_understat_joins", future == 0, future, ""),
        ("no_understat_result_score_current_match_direct_features", len(bad_direct) == 0, len(bad_direct), "|".join(bad_direct)),
        ("no_team_name_string_columns_added_as_model_features", len(bad_names) == 0, len(bad_names), "|".join(bad_names)),
        ("v3_row_count_preserved", len(v3) == len(v4), len(v4), f"v3={len(v3)} v4={len(v4)}"),
        ("v3_columns_unchanged", len(original_bad) == 0, len(original_bad), "|".join(original_bad[:20])),
        ("raw_zip_unchanged", raw_before == raw_after, 0, f"before={raw_before} after={raw_after}"),
    ]
    out = pd.DataFrame([{"check": n, "status": "pass" if ok else "fail", "count": int(count), "detail": detail} for n, ok, count, detail in rows])
    out.to_csv(LEAKAGE_CSV, index=False)
    return out


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return view.to_markdown(index=False)


def write_reports(decision: str, schema: dict[str, object], mapping: pd.DataFrame, coverage: pd.DataFrame, missing: pd.DataFrame, checks: pd.DataFrame, added_cols: int) -> None:
    key_cov = coverage[coverage["segment"].isin(["all_rows", "top5_only", "top5_test_2020_2025", "season_start_year_2024", "season_start_year_2025", "after_understat_max_date_2024_09_29", "locked_v3_prediction_row_universe"])]
    mapping_counts = mapping.drop_duplicates(["league", "football_team"]).groupby(["league", "mapping_status"]).size().rename("teams").reset_index()
    BUILD_REPORT_MD.write_text(
        "\n".join(
            [
                "# Feature Matrix V4.1 Understat Partial Build Audit",
                "",
                f"Decision: `{decision}`",
                "",
                "No predictive models, value searches, threshold optimization, FBref data, or locked v3 candidate changes were run. No confirmed edge is claimed.",
                "",
                "## Understat Schema",
                f"- Rows: {schema['rows']}",
                f"- Columns: {', '.join(schema['columns'])}",
                f"- Date range: {schema['date_min']} to {schema['date_max']}",
                f"- Leagues: {', '.join(schema['leagues'])}",
                f"- Seasons: {min(schema['seasons'])} to {max(schema['seasons'])}",
                f"- Team count: {schema['teams']}",
                f"- Duplicate team/date/home_away rows: {schema['duplicate_team_date_home_away']}",
                f"- Missing values: {schema['missing_values'] or 'none'}",
                f"- Numeric parse issues: {schema['numeric_parse_issues'] or 'none'}",
                "",
                "## Build Summary",
                f"- Output matrix: `{OUT_MATRIX}`",
                f"- Added Understat columns: {added_cols}",
                "- Current-match Understat values were not joined directly; all rolling features use only rows with `understat_date < match_date`.",
                "",
                "## Mapping Summary",
                md_table(mapping_counts, 40),
                "",
                "## Key Coverage",
                md_table(key_cov[["segment", "rows", "both_available", "both_available_rate", "home_history_count_median", "away_history_count_median", "latest_days_ago_median", "latest_days_ago_max"]], 20),
                "",
                "## Leakage Checks",
                md_table(checks, 50),
                "",
            ]
        ),
        encoding="utf-8",
    )
    SCOPE_MD.write_text(
        "\n".join(
            [
                "# V4.1 Understat Recommended Model Scope",
                "",
                f"Decision: `{decision}`",
                "",
                "Recommended safe scope: top-five Understat-mapped leagues only (`E0`, `D1`, `SP1`, `I1`, `F1`) with explicit missing flags and staleness/history-count controls.",
                "",
                "Important constraints:",
                "- Use only rolling past-only Understat features computed from `understat_date < match_date`.",
                "- Do not use RFPL until a target football-data league mapping exists.",
                "- Treat 2024/25 and 2025/26 carefully because Understat source data ends on 2024-09-29; stale/missing flags are required.",
                "- Do not use current-match xG, result, scored, or missed as direct pre-match features.",
                "- No confirmed edge is claimed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    raw_before = (UNDERSTAT_ZIP.stat().st_size, int(UNDERSTAT_ZIP.stat().st_mtime), file_sha256(UNDERSTAT_ZIP))
    v3 = pd.read_csv(V3_MATRIX, low_memory=False)
    v3["match_date"] = pd.to_datetime(v3["match_date"], errors="coerce")
    game = read_understat()
    schema = understat_schema_info(game)
    mapping, accepted = build_mapping(v3, game)
    audit = attach_mapping(v3, accepted)
    features, safety = build_features(v3, audit, game)
    v4 = pd.concat([v3.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    safety.to_csv(DATE_SAFETY_CSV, index=False)
    coverage, missing = coverage_reports(v4, features)
    feature_dictionary(features)
    raw_after = (UNDERSTAT_ZIP.stat().st_size, int(UNDERSTAT_ZIP.stat().st_mtime), file_sha256(UNDERSTAT_ZIP))
    checks = leakage_checks(v3.reset_index(drop=True), v4, features, safety, raw_before, raw_after)
    if checks["status"].ne("pass").any():
        decision = "v4_1_understat_build_failed"
    else:
        top = coverage[coverage["segment"].eq("top5_only")].iloc[0]
        test = coverage[coverage["segment"].eq("top5_test_2020_2025")].iloc[0]
        if float(top["both_available_rate"]) >= 0.95 and float(test["both_available_rate"]) >= 0.90:
            decision = "v4_1_understat_feature_build_ready_good"
        elif float(top["both_available_rate"]) >= 0.75:
            decision = "v4_1_understat_feature_build_ready_partial"
        else:
            decision = "v4_1_understat_mapping_ready_only"
    v4.to_csv(OUT_MATRIX, index=False)
    write_reports(decision, schema, mapping, coverage, missing, checks, len(features.columns))
    key = coverage[coverage["segment"].isin(["top5_only", "top5_test_2020_2025", "season_start_year_2025", "locked_v3_prediction_row_universe"])]
    print(
        {
            "decision": decision,
            "v3_rows": len(v3),
            "v4_rows": len(v4),
            "added_columns": len(features.columns),
            "coverage": {r.segment: round(float(r.both_available_rate), 6) for r in key.itertuples(index=False)},
            "failed_checks": int(checks["status"].ne("pass").sum()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
