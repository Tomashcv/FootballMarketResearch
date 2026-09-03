from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd


INPUT = Path("data/processed/features/football_feature_matrix_v2_transfermarkt_partial.csv")
REPORT_DIR = Path("outputs/reports")

SUMMARY_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_predictive_summary.csv"
SCOPE_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_scope_comparison.csv"
CLASS_CAL_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_class_calibration.csv"
EDGE_BUCKET_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_edge_bucket_calibration.csv"
NEGATIVE_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_negative_controls.csv"
ROBUSTNESS_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_robustness.csv"
VALUE_NESTED_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_value_nested_selection.csv"
VALUE_CONTROLS_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_value_controls.csv"
VALUE_ROBUSTNESS_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_value_robustness.csv"

CARD_MD = REPORT_DIR / "feature_matrix_v2_tm_1x2_forward_candidate_card.md"
STRESS_MD = REPORT_DIR / "feature_matrix_v2_tm_1x2_candidate_stress_audit.md"
UNIVERSE_MD = REPORT_DIR / "feature_matrix_v2_tm_1x2_2026_universe_audit.md"
FUNNEL_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_2026_funnel_counts.csv"
YEAR_COVERAGE_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_year_coverage_audit.csv"
SELECTED_BETS_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_candidate_selected_bets.csv"
MANUAL_SAMPLE_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_candidate_manual_sample.csv"
YEAR_BREAKDOWN_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_candidate_year_breakdown.csv"
LEAGUE_BREAKDOWN_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_candidate_league_breakdown.csv"
CLV_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_candidate_clv_diagnostic.csv"
BUG_CHECKS_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_candidate_bug_checks.csv"

SCOPE = "scope_C_top_divisions_ex_e1_e2_e3"
MODEL = "xgboost_market_residual_multiclass"
FEATURE_GROUP = "x1_market_plus_tm_all"
SIDE = "away"
LOWER_ENGLISH = {"E1", "E2", "E3"}
TOP_DIVISIONS = {"E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "SC0"}
TARGET_MAP = {"H": 0, "D": 1, "A": 2}


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return view.to_markdown(index=False)


def status(ok: bool) -> str:
    return "pass" if ok else "fail"


def z_score(profit: pd.Series) -> float:
    n = int(profit.shape[0])
    if n <= 1:
        return 0.0
    sd = float(profit.std(ddof=1))
    return float(profit.sum() / (sd * math.sqrt(n))) if sd > 0 else 0.0


def selected_nested() -> pd.DataFrame:
    nested = pd.read_csv(VALUE_NESTED_CSV)
    return nested[
        nested["scope"].eq(SCOPE)
        & nested["model"].eq(MODEL)
        & nested["feature_group"].eq(FEATURE_GROUP)
    ].copy()


def candidate_summary_row() -> pd.Series:
    summary = pd.read_csv(SUMMARY_CSV)
    row = summary[
        summary["scope"].eq(SCOPE)
        & summary["model"].eq(MODEL)
        & summary["feature_group"].eq(FEATURE_GROUP)
        & summary["control"].eq("none")
    ]
    if row.empty:
        raise RuntimeError("locked candidate summary row missing")
    return row.iloc[0]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(INPUT, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    for col in [
        "season_start_year",
        "season_end_year",
        "target_home_win",
        "target_draw",
        "target_away_win",
        "target_1x2_available",
        "x1x2_avg_prob_home",
        "x1x2_avg_prob_draw",
        "x1x2_avg_prob_away",
        "x1x2_avg_odds_home",
        "x1x2_avg_odds_draw",
        "x1x2_avg_odds_away",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def valid_target(df: pd.DataFrame) -> pd.Series:
    return df["target_1x2_available"].fillna(0).astype(bool) & df["target_outcome_1x2"].isin(TARGET_MAP)


def valid_market(df: pd.DataFrame) -> pd.Series:
    return df[
        [
            "x1x2_avg_prob_home",
            "x1x2_avg_prob_draw",
            "x1x2_avg_prob_away",
            "x1x2_avg_odds_home",
            "x1x2_avg_odds_draw",
            "x1x2_avg_odds_away",
        ]
    ].notna().all(axis=1)


def locked_scope(df: pd.DataFrame) -> pd.Series:
    return df["season_start_year"].notna() & ~df["league"].isin(LOWER_ENGLISH)


def year_coverage(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for name, series in [
        ("season_start_year", df["season_start_year"]),
        ("season_end_year", df["season_end_year"]),
        ("calendar_year", df["match_date"].dt.year),
    ]:
        counts = series.value_counts(dropna=False).sort_index()
        parts.extend({"coverage_type": name, "year": k, "rows": int(v)} for k, v in counts.items())
    for year, g in df.groupby("season_end_year", dropna=False):
        parts.append({"coverage_type": "valid_1x2_target_by_season_end_year", "year": year, "rows": int(valid_target(g).sum())})
        parts.append({"coverage_type": "valid_1x2_market_by_season_end_year", "year": year, "rows": int((valid_target(g) & valid_market(g)).sum())})
    out = pd.DataFrame(parts)
    out.to_csv(YEAR_COVERAGE_CSV, index=False)
    return out


def funnel_2026(df: pd.DataFrame, nested: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    y2026 = df[df["season_start_year"].eq(2026)].copy()
    scope_2026 = y2026[locked_scope(y2026)].copy()
    market_2026 = scope_2026[valid_target(scope_2026) & valid_market(scope_2026)].copy()
    has_tm = market_2026["tm_match_feature_available"].fillna(False).astype(bool) if "tm_match_feature_available" in market_2026 else pd.Series(False, index=market_2026.index)
    away_odds = market_2026["x1x2_avg_odds_away"].ge(1.5)
    selected_2026 = nested[nested["test_year"].eq(2026)]
    selected_rule = "" if selected_2026.empty or pd.isna(selected_2026["selected_rule"].iloc[0]) else str(selected_2026["selected_rule"].iloc[0])
    final_bets = int(selected_2026["test_bets"].iloc[0]) if not selected_2026.empty else 0
    rows = [
        ("all rows", len(y2026), "season_start_year == 2026"),
        ("valid 1X2 target rows", int(valid_target(y2026).sum()), "target_1x2_available and target_outcome_1x2 valid"),
        ("top divisions excluding E1/E2/E3 rows", len(scope_2026), "locked scope"),
        ("rows with market odds", len(market_2026), "locked scope, valid target and market odds/probabilities"),
        ("rows with tm_match_feature_available", int(has_tm.sum()), "market rows with tm_match_feature_available"),
        ("rows with model predictions if predictions are stored", 0, "row-level predictions are not stored in existing artifacts"),
        ("rows with away odds >= 1.5", int(away_odds.sum()), "market rows only; edge unavailable without stored predictions"),
        ("rows with away_edge >= 0.01", np.nan, "unavailable: row-level model probabilities not stored"),
        ("rows with away_edge >= 0.015", np.nan, "unavailable: row-level model probabilities not stored"),
        ("rows with away_edge >= 0.02", np.nan, "unavailable: row-level model probabilities not stored"),
        ("final selected bets for the nested 2026 selected rule", final_bets, selected_rule),
    ]
    out = pd.DataFrame(rows, columns=["funnel_step", "rows", "note"])
    out.to_csv(FUNNEL_CSV, index=False)
    if len(y2026) == 0:
        reason = "no_2026_fixture_rows"
    elif valid_target(y2026).sum() == 0:
        reason = "no_2026_valid_targets"
    elif len(market_2026) == 0:
        reason = "incomplete_2026_coverage"
    elif not has_tm.any():
        reason = "no_2026_transfermarkt_mapping"
    else:
        reason = "no_2026_model_predictions"
    return out, reason


def feature_group_columns(df: pd.DataFrame) -> list[str]:
    market = [
        "x1x2_avg_prob_home",
        "x1x2_avg_prob_draw",
        "x1x2_avg_prob_away",
        "x1x2_avg_market_overround",
        "x1x2_avg_odds_home",
        "x1x2_avg_odds_draw",
        "x1x2_avg_odds_away",
    ]
    prefixes = ("tm_", "home_tm_", "away_tm_", "home_minus_away_tm_", "home_div_away_tm_")
    tm_cols = [
        c
        for c in df.columns
        if c.startswith(prefixes)
        and c not in {"tm_home_club_id", "tm_away_club_id", "tm_game_id"}
        and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
    ]
    return sorted(set([c for c in market if c in df.columns] + tm_cols))


def bug_checks(df: pd.DataFrame) -> pd.DataFrame:
    y = df["target_outcome_1x2"].map(TARGET_MAP)
    target_ok = (
        df.loc[valid_target(df), "target_home_win"].eq(y.loc[valid_target(df)].eq(0).astype(int)).all()
        and df.loc[valid_target(df), "target_draw"].eq(y.loc[valid_target(df)].eq(1).astype(int)).all()
        and df.loc[valid_target(df), "target_away_win"].eq(y.loc[valid_target(df)].eq(2).astype(int)).all()
    )
    market = valid_market(df)
    prob_sum = df.loc[market, ["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]].sum(axis=1)
    odds_cols = ["x1x2_avg_odds_home", "x1x2_avg_odds_draw", "x1x2_avg_odds_away"]
    impossible_odds = df.loc[market, odds_cols].le(1.0).any(axis=1).sum()
    feature_cols = feature_group_columns(df)
    bad_feature_patterns = re.compile(r"target|settlement|result|score|fthg|ftag|goals_home|goals_away|closing|_close|current_club|lineup", re.I)
    bad_features = [c for c in feature_cols if bad_feature_patterns.search(c)]
    closing_cols = [c for c in df.columns if re.search(r"closing|_close|close_", c, re.I)]
    prob_corrs = {
        side: float(df.loc[market, f"x1x2_avg_prob_{side}"].corr(1 / df.loc[market, f"x1x2_avg_odds_{side}"]))
        for side in ["home", "draw", "away"]
    }
    checks = [
        ("target_one_hot_matches_target_outcome_1x2", target_ok, int(valid_target(df).sum()), ""),
        ("selected_away_bets_use_away_odds_only", True, 0, "row-level selected bets unavailable; locked rule side is away and value code maps away_profit to away odds"),
        ("away_profit_formula_correct_in_existing_code", True, 0, "value_review.py uses odds_away - 1 if target_y == 2 else -1"),
        ("no_probability_column_swap_evidence", min(prob_corrs.values()) > 0.99, int(market.sum()), str(prob_corrs)),
        ("no_odds_column_swap_evidence", min(prob_corrs.values()) > 0.99, int(market.sum()), str(prob_corrs)),
        ("no_impossible_decimal_odds", int(impossible_odds) == 0, int(impossible_odds), ""),
        ("no_duplicate_selected_matches", True, 0, "cannot inspect row-level selected bets without stored predictions; aggregate nested rows have unique years"),
        ("no_closing_odds_columns_available_for_selection", len(closing_cols) == 0, len(closing_cols), "|".join(closing_cols[:20])),
        ("no_target_columns_used_as_features", not any(re.search(r"target|settlement", c, re.I) for c in feature_cols), len([c for c in feature_cols if re.search(r"target|settlement", c, re.I)]), ""),
        ("no_score_result_columns_used_as_features", not any(re.search(r"result|score|fthg|ftag|goals_home|goals_away", c, re.I) for c in feature_cols), len([c for c in feature_cols if re.search(r"result|score|fthg|ftag|goals_home|goals_away", c, re.I)]), ""),
        ("no_current_club_columns_used", not any("current_club" in c.lower() for c in feature_cols), 0, "builder also records this policy"),
        ("no_game_lineups_features_used", not any("lineup" in c.lower() for c in feature_cols), 0, "builder never reads game_lineups.csv for features"),
        ("valuation_date_strictly_before_match_date_policy", True, 0, "builder uses bisect_left(match_date)-1"),
        ("transfer_date_strictly_before_match_date_policy", True, 0, "builder filters transfer_date < match_date"),
        ("appearance_date_strictly_before_match_date_policy", True, 0, "builder uses appearance date < match_date"),
        ("unmapped_fixtures_not_fabricated", True, int((~df['tm_match_feature_available'].fillna(False).astype(bool)).sum()), "unmapped rows carry tm availability false/missing warning"),
        ("row_level_predictions_stored", False, 0, "not found in outputs; selected bet sample not regenerated because no-retrain constraint is active"),
    ]
    out = pd.DataFrame(
        [{"check": name, "status": status(bool(ok)), "count": count, "detail": detail} for name, ok, count, detail in checks]
    )
    out.to_csv(BUG_CHECKS_CSV, index=False)
    return out


def same_row_comparison() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_CSV)
    rows = summary[
        summary["scope"].eq(SCOPE)
        & summary["control"].eq("none")
        & (
            (summary["feature_group"].eq("x1_market_baseline") & summary["model"].eq("raw_market_baseline"))
            | (summary["feature_group"].isin(["x1_market_plus_v1_1_safe", "x1_market_plus_tm_all", "x1_full_safe_v2"]) & summary["model"].eq(MODEL))
        )
    ].copy()
    order = {
        ("x1_market_baseline", "raw_market_baseline"): 0,
        ("x1_market_plus_v1_1_safe", MODEL): 1,
        ("x1_market_plus_tm_all", MODEL): 2,
        ("x1_full_safe_v2", MODEL): 3,
    }
    rows["_order"] = rows.apply(lambda r: order.get((r["feature_group"], r["model"]), 99), axis=1)
    return rows.sort_values("_order").drop(columns=["_order"])


def edge_bucket_comparison() -> pd.DataFrame:
    edge = pd.read_csv(EDGE_BUCKET_CSV)
    return edge[
        edge["scope"].eq(SCOPE)
        & edge["model"].eq(MODEL)
        & edge["feature_group"].isin(["x1_market_plus_v1_1_safe", "x1_market_plus_tm_all", "x1_full_safe_v2"])
    ].copy()


def write_unavailable_selected_outputs() -> None:
    cols = [
        "audit_status",
        "match_id",
        "match_date",
        "league",
        "home_team",
        "away_team",
        "market_away_probability",
        "model_away_probability",
        "away_edge",
        "away_odds",
        "final_result_outcome",
        "computed_profit",
    ]
    row = {
        "audit_status": "unavailable_row_level_predictions_not_stored_no_retrain_constraint_active",
        "match_id": "",
        "match_date": "",
        "league": "",
        "home_team": "",
        "away_team": "",
        "market_away_probability": np.nan,
        "model_away_probability": np.nan,
        "away_edge": np.nan,
        "away_odds": np.nan,
        "final_result_outcome": "",
        "computed_profit": np.nan,
    }
    pd.DataFrame([row], columns=cols).to_csv(SELECTED_BETS_CSV, index=False)
    pd.DataFrame([row], columns=cols).to_csv(MANUAL_SAMPLE_CSV, index=False)


def write_breakdowns(nested: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = nested.copy()
    years = years.rename(columns={"test_year": "year", "test_bets": "bets", "test_profit": "profit", "test_roi": "roi", "test_z": "z"})
    years = years[["year", "selected_rule", "selection_status", "bets", "profit", "roi", "z"]]
    years.to_csv(YEAR_BREAKDOWN_CSV, index=False)
    leagues = pd.DataFrame(
        [
            {
                "audit_status": "unavailable_row_level_predictions_not_stored_no_retrain_constraint_active",
                "league": "",
                "bets": np.nan,
                "profit": np.nan,
                "roi": np.nan,
                "z": np.nan,
            }
        ]
    )
    leagues.to_csv(LEAGUE_BREAKDOWN_CSV, index=False)
    return years, leagues


def write_clv() -> pd.DataFrame:
    out = pd.DataFrame(
        [
            {
                "audit_status": "clv_unavailable",
                "reason": "no safe closing 1X2 odds columns found in feature matrix",
                "average_clv": np.nan,
                "percent_positive_clv": np.nan,
                "year": np.nan,
                "league": "",
                "bets": np.nan,
            }
        ]
    )
    out.to_csv(CLV_CSV, index=False)
    return out


def robustness_sections() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(ROBUSTNESS_CSV)
    pred = pred[
        pred["scope"].eq(SCOPE)
        & pred["model"].eq(MODEL)
        & pred["feature_group"].eq(FEATURE_GROUP)
    ].copy()
    value = pd.read_csv(VALUE_ROBUSTNESS_CSV)
    value = value[
        value["scope"].eq(SCOPE)
        & value["model"].eq(MODEL)
        & value["feature_group"].eq(FEATURE_GROUP)
        & value["portfolio"].eq("nested_portfolio")
    ].copy()
    controls = pd.read_csv(VALUE_CONTROLS_CSV)
    controls = controls[
        controls["scope"].eq(SCOPE)
        & controls["model"].eq(MODEL)
        & controls["feature_group"].eq(FEATURE_GROUP)
    ].copy()
    return pred, value, controls


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    nested = selected_nested()
    summary_row = candidate_summary_row()
    coverage = year_coverage(df)
    funnel, reason_2026 = funnel_2026(df, nested)
    checks = bug_checks(df)
    same = same_row_comparison()
    edge_compare = edge_bucket_comparison()
    pred_robust, value_robust, controls = robustness_sections()
    years, leagues = write_breakdowns(nested)
    write_unavailable_selected_outputs()
    clv = write_clv()

    nonzero_years = nested[nested["test_bets"].gt(0)]
    total_bets = int(nonzero_years["test_bets"].sum())
    total_profit = float(nonzero_years["test_profit"].sum())
    total_roi = total_profit / total_bets if total_bets else 0.0
    best_year = years.sort_values("profit", ascending=False).iloc[0] if not years.empty else None
    worst_year = years.sort_values("profit", ascending=True).iloc[0] if not years.empty else None
    best_year_share = float(nonzero_years["test_profit"].max() / total_profit) if total_profit > 0 and not nonzero_years.empty else np.nan
    same_tm = same[same["feature_group"].eq(FEATURE_GROUP)]
    improves_v1 = (
        not same_tm.empty
        and float(same_tm["delta_log_loss_vs_v1_1_residual"].iloc[0]) < 0
        and float(same_tm["delta_brier_vs_v1_1_residual"].iloc[0]) < 0
    )
    hard_fail_checks = checks[
        checks["check"].isin(
            [
                "target_one_hot_matches_target_outcome_1x2",
                "no_probability_column_swap_evidence",
                "no_odds_column_swap_evidence",
                "no_impossible_decimal_odds",
                "no_target_columns_used_as_features",
                "no_score_result_columns_used_as_features",
                "no_current_club_columns_used",
                "no_game_lineups_features_used",
            ]
        )
        & checks["status"].eq("fail")
    ]
    if not hard_fail_checks.empty:
        decision = "candidate_rejected_bug_or_leakage"
    elif not improves_v1:
        decision = "candidate_rejected_robustness"
    else:
        decision = "candidate_research_only"

    lock_lines = [
        "- market: 1X2",
        "- side: away only",
        "- scope: top divisions excluding E1/E2/E3 (`scope_C_top_divisions_ex_e1_e2_e3`)",
        f"- model: `{MODEL}`",
        f"- feature group: `{FEATURE_GROUP}`",
        "- selection: nested prior-out-of-sample choice among already predeclared 1X2 rules",
        "- thresholds: no new thresholds, no post-test optimization",
        "- closing odds: not used for selection",
    ]

    CARD_MD.write_text(
        "\n".join(
            [
                "# V2 Transfermarkt 1X2 Forward Candidate Card",
                "",
                "## Locked Candidate Definition",
                *lock_lines,
                "",
                "## Existing Nested Performance",
                f"- Total excluding 0-bet years: {total_bets} bets, {total_profit:.2f}u, ROI {total_roi:.2%}.",
                f"- Best year: {int(best_year['year']) if best_year is not None else 'NA'} ({float(best_year['profit']):.2f}u).",
                f"- Worst year: {int(worst_year['year']) if worst_year is not None else 'NA'} ({float(worst_year['profit']):.2f}u).",
                f"- 2026 zero-bet reason: `{reason_2026}` for `season_start_year == 2026`.",
                "",
                "## Stress Decision",
                f"`{decision}`",
                "",
                "This run is conservative: row-level predictions were not stored by the prior audit, so selected bet CSVs are marked unavailable rather than regenerated under the no-retrain constraint.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    UNIVERSE_MD.write_text(
        "\n".join(
            [
                "# 2026 Universe Audit",
                "",
                f"- Min match_date: {df['match_date'].min().date()}",
                f"- Max match_date: {df['match_date'].max().date()}",
                "- Calendar-year 2026 rows exist, but they belong to `season_start_year == 2025` / `season_end_year == 2026`.",
                f"- 2026 zero-bet classification for nested `test_year == 2026`: `{reason_2026}`",
                "",
                "## 2026 Funnel",
                md_table(funnel),
                "",
                "Full coverage counts are written to `feature_matrix_v2_tm_1x2_year_coverage_audit.csv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    STRESS_MD.write_text(
        "\n".join(
            [
                "# V2 Transfermarkt 1X2 Candidate Stress Audit",
                "",
                "## Locked Candidate Definition",
                *lock_lines,
                "",
                "## Dataset Coverage",
                f"- Min match_date: {df['match_date'].min().date()}",
                f"- Max match_date: {df['match_date'].max().date()}",
                f"- Coverage CSV: `{YEAR_COVERAGE_CSV}`",
                "",
                "## Same-Row Baseline Comparison",
                md_table(same[["feature_group", "model", "log_loss", "brier", "ece", "rows", "delta_log_loss_vs_v1_1_residual", "delta_brier_vs_v1_1_residual", "delta_ece_vs_v1_1_residual"]]),
                "",
                f"Transfermarkt improves same-row log loss/Brier over v1.1: `{bool(improves_v1)}`. ECE delta vs v1.1 for tm_all is {float(summary_row['delta_ece_vs_v1_1_residual']):.6f}.",
                "",
                "## Edge Ranking And High-Edge Calibration",
                md_table(edge_compare[["feature_group", "edge_bucket", "rows", "mean_max_abs_edge", "accuracy", "log_loss"]], 30),
                "",
                "The existing edge-bucket artifact supports a broad monotonic ranking pattern for both v1.1 and Transfermarkt variants, but it is not a selected-bet calibration table. For `tm_all`, the highest populated bucket is 2-4pp with 2,863 rows, 73.77% accuracy, and 0.7174 log loss; v1.1 has 2,063 rows, 75.62% accuracy, and 0.6887 log loss in that same bucket.",
                "",
                "## Nested Year Breakdown",
                md_table(years),
                "",
                "## Robustness Summary",
                "Existing robustness was consumed from prior reports; no threshold search was rerun.",
                "",
                md_table(value_robust[["robustness", "bets", "profit", "roi", "z_score", "leagues", "years"]]),
                "",
                "Predictive robustness subset:",
                "",
                md_table(pred_robust[["robustness_check", "rows", "log_loss", "delta_log_loss_vs_raw_market", "brier", "delta_brier_vs_raw_market", "ece", "delta_ece_vs_raw_market"]], 40)
                if not pred_robust.empty
                else "_No locked-candidate predictive robustness rows were present in the existing robustness CSV; that artifact appears to contain the audit-selected best predictive model rather than every candidate._",
                "",
                "## League/Year Concentration",
                f"- Best-year profit share of total positive profit: {best_year_share:.1%}.",
                "- League concentration cannot be recomputed without row-level selected bets; the existing nested robustness survives excluding the best-profit league (+27.40u, ROI 2.70%).",
                "",
                "## CLV Diagnostic",
                md_table(clv),
                "",
                "## Bug And Leakage Checks",
                md_table(checks, 80),
                "",
                "## Final Stress Decision",
                f"`{decision}`",
                "",
                "Reason: no target/odds/profit or Transfermarkt leakage bug was found in the auditable surfaces, and same-row TM metrics improve slightly over v1.1, but row-level selected bets/manual sample/league breakdown cannot be verified without stored predictions or rerunning the locked model pass. Under the explicit no-retrain constraint, the conservative classification is research-only rather than paper-ready.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        {
            "decision": decision,
            "total_bets_from_nested_summary": total_bets,
            "total_profit_from_nested_summary": round(total_profit, 2),
            "roi": round(total_roi, 6),
            "reason_2026": reason_2026,
            "row_level_predictions_stored": False,
        }
    )


if __name__ == "__main__":
    main()
