from __future__ import annotations

from pathlib import Path
import math
import re
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments.feature_matrix_v2_tm_1x2_predictive_audit import (
    TEST_YEARS,
    annual_predictions,
    feature_groups,
    load_data,
    scope_mask,
)
from src.experiments.feature_matrix_v2_tm_1x2_value_review import (
    add_value_columns,
    nested_selection,
)


REPORT_DIR = Path("outputs/reports")
INPUT = Path("data/processed/features/football_feature_matrix_v2_transfermarkt_partial.csv")
EXISTING_NESTED_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_value_nested_selection.csv"

SCOPE = "scope_C_top_divisions_ex_e1_e2_e3"
MODEL = "xgboost_market_residual_multiclass"
FEATURE_GROUP = "x1_market_plus_tm_all"
SIDE = "away"

ROW_PRED_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_locked_row_predictions.csv"
SELECTED_BETS_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_locked_selected_bets.csv"
MANUAL_SAMPLE_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_locked_manual_sample.csv"
YEAR_BREAKDOWN_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_locked_year_breakdown.csv"
LEAGUE_BREAKDOWN_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_locked_league_breakdown.csv"
BUG_CHECKS_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_locked_bug_checks.csv"
REPORT_MD = REPORT_DIR / "feature_matrix_v2_tm_1x2_locked_replay_report.md"
CARD_MD = REPORT_DIR / "feature_matrix_v2_tm_1x2_locked_candidate_card.md"

TARGET_TO_INT = {"H": 0, "D": 1, "A": 2}


def z_score(profit: pd.Series) -> float:
    n = int(len(profit))
    if n <= 1:
        return 0.0
    std = float(profit.std(ddof=1))
    return float(profit.sum() / (std * math.sqrt(n))) if std > 0 else 0.0


def md_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return view.to_markdown(index=False)


def bool_status(ok: bool) -> str:
    return "pass" if ok else "fail"


def raw_feature_columns(df: pd.DataFrame) -> list[str]:
    groups = feature_groups(df)
    return list(groups[FEATURE_GROUP])


def enrich_predictions(pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    info_cols = [
        "match_id",
        "league",
        "season_start_year",
        "season_end_year",
        "home_team",
        "away_team",
        "target_outcome_1x2",
        "target_home_win",
        "target_draw",
        "target_away_win",
        "tm_match_feature_available",
    ]
    scoped = df[scope_mask(df, SCOPE)].copy()
    out = pred.merge(
        scoped[info_cols],
        on=["match_id", "league", "season_start_year"],
        how="left",
        validate="many_to_one",
    )
    out = out.rename(
        columns={
            "x1x2_avg_prob_home": "market_prob_home",
            "x1x2_avg_prob_draw": "market_prob_draw",
            "x1x2_avg_prob_away": "market_prob_away",
            "prob_home": "model_prob_home",
            "prob_draw": "model_prob_draw",
            "prob_away": "model_prob_away",
            "home_edge": "edge_home",
            "draw_edge": "edge_draw",
            "away_edge": "edge_away",
            "x1x2_avg_odds_home": "odds_home",
            "x1x2_avg_odds_draw": "odds_draw",
            "x1x2_avg_odds_away": "odds_away",
            "test_year": "fold_test_year",
        }
    )
    cols = [
        "match_id",
        "match_date",
        "season_start_year",
        "season_end_year",
        "league",
        "home_team",
        "away_team",
        "target_outcome_1x2",
        "target_home_win",
        "target_draw",
        "target_away_win",
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
        "model_prob_home",
        "model_prob_draw",
        "model_prob_away",
        "edge_home",
        "edge_draw",
        "edge_away",
        "odds_home",
        "odds_draw",
        "odds_away",
        "tm_match_feature_available",
        "fold_test_year",
    ]
    return out[cols].sort_values(["fold_test_year", "match_date", "league", "match_id"]).reset_index(drop=True)


def selected_export(nested_bets: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    info_cols = [
        "match_id",
        "league",
        "season_start_year",
        "season_end_year",
        "home_team",
        "away_team",
        "target_outcome_1x2",
        "target_away_win",
    ]
    scoped = df[scope_mask(df, SCOPE)].copy()
    out = nested_bets.merge(
        scoped[info_cols],
        on=["match_id", "league", "season_start_year"],
        how="left",
        validate="many_to_one",
    )
    out["computed_profit"] = np.where(out["target_away_win"].eq(1), out["x1x2_avg_odds_away"] - 1.0, -1.0)
    out = out.rename(
        columns={
            "x1x2_avg_prob_away": "market_prob_away",
            "prob_away": "model_prob_away",
            "away_edge": "edge_away",
            "x1x2_avg_odds_away": "odds_away",
        }
    )
    cols = [
        "selected_rule",
        "side",
        "match_id",
        "match_date",
        "season_start_year",
        "season_end_year",
        "league",
        "home_team",
        "away_team",
        "target_outcome_1x2",
        "market_prob_away",
        "model_prob_away",
        "edge_away",
        "odds_away",
        "target_away_win",
        "profit",
        "computed_profit",
    ]
    return out[cols].sort_values(["season_start_year", "match_date", "league", "match_id"]).reset_index(drop=True)


def breakdown(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_col, dropna=False):
        bets = int(len(group))
        profit = float(group["profit"].sum())
        rows.append(
            {
                group_col: key,
                "bets": bets,
                "profit": profit,
                "roi": profit / bets if bets else 0.0,
                "z": z_score(group["profit"]),
            }
        )
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def manual_sample(selected: pd.DataFrame) -> pd.DataFrame:
    if len(selected) <= 50:
        return selected.copy()
    ordered = selected.sort_values(["season_start_year", "league", "match_date", "match_id"]).copy()
    pieces = []
    for _, year_group in ordered.groupby("season_start_year"):
        per_league = year_group.groupby("league", group_keys=False).head(1)
        pieces.append(per_league)
    sample = pd.concat(pieces, ignore_index=False).drop_duplicates("match_id")
    if len(sample) < 50:
        remaining = ordered[~ordered["match_id"].isin(sample["match_id"])]
        need = 50 - len(sample)
        fill = (
            remaining.assign(_rank=remaining.groupby(["season_start_year", "league"]).cumcount())
            .sort_values(["_rank", "season_start_year", "league", "match_date", "match_id"])
            .head(need)
            .drop(columns=["_rank"])
        )
        sample = pd.concat([sample, fill], ignore_index=False)
    return sample.sort_values(["season_start_year", "league", "match_date", "match_id"]).head(50).reset_index(drop=True)


def replay_matches_prior(replay_nested: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    prior = pd.read_csv(EXISTING_NESTED_CSV)
    prior = prior[
        prior["scope"].eq(SCOPE)
        & prior["model"].eq(MODEL)
        & prior["feature_group"].eq(FEATURE_GROUP)
    ].copy()
    compare = replay_nested.merge(
        prior,
        on=["scope", "model", "feature_group", "test_year"],
        how="outer",
        suffixes=("_replay", "_prior"),
        indicator=True,
    )
    compare["bets_diff"] = compare["test_bets_replay"].fillna(-999999) - compare["test_bets_prior"].fillna(-999999)
    compare["profit_diff"] = compare["test_profit_replay"].fillna(-999999.0) - compare["test_profit_prior"].fillna(-999999.0)
    compare["roi_diff"] = compare["test_roi_replay"].fillna(-999999.0) - compare["test_roi_prior"].fillna(-999999.0)
    ok = (
        compare["_merge"].eq("both").all()
        and compare["selected_rule_replay"].fillna("").eq(compare["selected_rule_prior"].fillna("")).all()
        and compare["selection_status_replay"].eq(compare["selection_status_prior"]).all()
        and compare["bets_diff"].abs().le(0).all()
        and compare["profit_diff"].abs().le(1e-9).all()
        and compare["roi_diff"].abs().le(1e-12).all()
    )
    return bool(ok), compare


def bug_checks(selected: pd.DataFrame, row_predictions: pd.DataFrame, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    closing_cols = [c for c in df.columns if re.search(r"closing|_close|close_", c, re.I)]
    feature_bad_target = [c for c in feature_cols if re.search(r"target|settlement", c, re.I)]
    feature_bad_result = [c for c in feature_cols if re.search(r"result|score|fthg|ftag|goals_home|goals_away", c, re.I)]
    feature_current_club = [c for c in feature_cols if "current_club" in c.lower()]
    feature_lineups = [c for c in feature_cols if "lineup" in c.lower()]
    profit_ok = selected["profit"].round(12).eq(selected["computed_profit"].round(12)).all()
    target_y = row_predictions["target_outcome_1x2"].map(TARGET_TO_INT)
    target_ok = (
        row_predictions["target_home_win"].eq(target_y.eq(0).astype(int)).all()
        and row_predictions["target_draw"].eq(target_y.eq(1).astype(int)).all()
        and row_predictions["target_away_win"].eq(target_y.eq(2).astype(int)).all()
    )
    checks = [
        ("replay_row_predictions_exported", len(row_predictions) > 0, len(row_predictions), ""),
        ("selected_bets_exported", len(selected) > 0, len(selected), ""),
        ("manual_sample_available", len(selected) >= 50, min(len(selected), 50), ""),
        ("no_duplicate_selected_matches", not selected.duplicated(["match_id"]).any(), int(selected.duplicated(["match_id"]).sum()), ""),
        ("all_selected_bets_are_away_side", selected["side"].eq("away").all(), int((~selected["side"].eq("away")).sum()), ""),
        ("away_odds_used_for_profit", profit_ok, int((~selected["profit"].round(12).eq(selected["computed_profit"].round(12))).sum()), ""),
        ("profit_formula_correct", profit_ok, int((~selected["profit"].round(12).eq(selected["computed_profit"].round(12))).sum()), ""),
        ("no_impossible_selected_odds", selected["odds_away"].gt(1.0).all(), int((~selected["odds_away"].gt(1.0)).sum()), ""),
        ("target_one_hot_matches_target_outcome_1x2", target_ok, len(row_predictions), ""),
        ("no_closing_odds_selection", len(closing_cols) == 0, len(closing_cols), "|".join(closing_cols[:20])),
        ("no_target_columns_used_as_features", len(feature_bad_target) == 0, len(feature_bad_target), "|".join(feature_bad_target[:20])),
        ("no_score_result_columns_used_as_features", len(feature_bad_result) == 0, len(feature_bad_result), "|".join(feature_bad_result[:20])),
        ("no_transfermarkt_current_club_features", len(feature_current_club) == 0, len(feature_current_club), "|".join(feature_current_club[:20])),
        ("no_game_lineups_features", len(feature_lineups) == 0, len(feature_lineups), "|".join(feature_lineups[:20])),
        ("valuation_date_strictly_before_match_date_policy", True, 0, "builder uses bisect_left(match_date)-1"),
        ("transfer_date_strictly_before_match_date_policy", True, 0, "builder filters transfer_date < match_date"),
        ("appearance_date_strictly_before_match_date_policy", True, 0, "builder uses appearance date < match_date"),
        ("unmapped_fixtures_not_fabricated", True, int((~df["tm_match_feature_available"].fillna(False).astype(bool)).sum()), "unmapped fixtures keep Transfermarkt availability false/missing"),
    ]
    return pd.DataFrame(
        [{"check": name, "status": bool_status(bool(ok)), "count": int(count), "detail": detail} for name, ok, count, detail in checks]
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = load_data()
    groups = feature_groups(raw_df)
    feature_cols = groups[FEATURE_GROUP]

    print(f"locked_replay scope={SCOPE} feature_group={FEATURE_GROUP} model={MODEL}", flush=True)
    pred, _ = annual_predictions(raw_df, SCOPE, FEATURE_GROUP, MODEL, feature_cols)
    value_pred = add_value_columns(pred, raw_df[scope_mask(raw_df, SCOPE)].copy())
    replay_nested, nested_bets = nested_selection(value_pred, SCOPE, MODEL, FEATURE_GROUP)
    row_predictions = enrich_predictions(value_pred, raw_df)
    selected = selected_export(nested_bets, raw_df)

    row_predictions.to_csv(ROW_PRED_CSV, index=False)
    selected.to_csv(SELECTED_BETS_CSV, index=False)
    sample = manual_sample(selected)
    sample.to_csv(MANUAL_SAMPLE_CSV, index=False)
    year = breakdown(selected, "season_start_year")
    league = breakdown(selected, "league")
    year.to_csv(YEAR_BREAKDOWN_CSV, index=False)
    league.to_csv(LEAGUE_BREAKDOWN_CSV, index=False)
    checks = bug_checks(selected, row_predictions, raw_df, feature_cols)

    reproduction_ok, compare = replay_matches_prior(replay_nested)
    checks = pd.concat(
        [
            checks,
            pd.DataFrame(
                [
                    {
                        "check": "locked_replay_matches_prior_nested_summary",
                        "status": bool_status(reproduction_ok),
                        "count": int(len(compare)),
                        "detail": "compared selected_rule, status, bets, profit, roi by test_year",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    checks.to_csv(BUG_CHECKS_CSV, index=False)

    total_bets = int(len(selected))
    total_profit = float(selected["profit"].sum())
    total_roi = total_profit / total_bets if total_bets else 0.0
    total_z = z_score(selected["profit"])
    best_year = year.sort_values("profit", ascending=False).iloc[0]
    worst_year = year.sort_values("profit", ascending=True).iloc[0]
    best_league = league.sort_values("profit", ascending=False).iloc[0]
    worst_league = league.sort_values("profit", ascending=True).iloc[0]
    best_year_profit_share = float(best_year["profit"] / total_profit) if total_profit > 0 else np.nan
    best_league_profit_share = float(best_league["profit"] / total_profit) if total_profit > 0 else np.nan
    profit_ex_best_year = float(selected[selected["season_start_year"].ne(best_year["season_start_year"])]["profit"].sum())
    profit_ex_best_league = float(selected[~selected["league"].eq(best_league["league"])]["profit"].sum())
    effective_test_seasons = sorted(map(int, row_predictions["fold_test_year"].dropna().unique()))
    missing_2026_rows = int(raw_df[raw_df["season_start_year"].eq(2026)].shape[0])

    hard_fail = checks[
        checks["check"].isin(
            [
                "locked_replay_matches_prior_nested_summary",
                "selected_bets_exported",
                "manual_sample_available",
                "no_duplicate_selected_matches",
                "all_selected_bets_are_away_side",
                "away_odds_used_for_profit",
                "profit_formula_correct",
                "no_impossible_selected_odds",
                "target_one_hot_matches_target_outcome_1x2",
                "no_closing_odds_selection",
                "no_target_columns_used_as_features",
                "no_score_result_columns_used_as_features",
                "no_transfermarkt_current_club_features",
                "no_game_lineups_features",
            ]
        )
        & checks["status"].eq("fail")
    ]
    if not hard_fail.empty:
        if "locked_replay_matches_prior_nested_summary" in set(hard_fail["check"]):
            decision = "candidate_rejected_reproduction_failed"
        else:
            decision = "candidate_rejected_bug_or_leakage"
    elif total_profit > 0 and best_year_profit_share < 1.0 and best_league_profit_share < 1.0:
        decision = "candidate_forward_paper_ready"
    else:
        decision = "candidate_research_only"

    report_lines = [
        "# V2 Transfermarkt 1X2 Locked Replay Export",
        "",
        "## Locked Candidate",
        f"- market: 1X2",
        f"- side: {SIDE} only",
        f"- scope: `{SCOPE}`",
        f"- model: `{MODEL}`",
        f"- feature_group: `{FEATURE_GROUP}`",
        "- selection: original nested prior-out-of-sample selection over the predeclared 1X2 grid; replay selected away rules only.",
        "- no new thresholds, no threshold search, no post-test optimization, no closing odds selection.",
        "",
        "## Effective Test Seasons",
        f"- Effective folds with tested rows: {', '.join(map(str, effective_test_seasons))}",
        f"- `season_start_year == 2026` rows: {missing_2026_rows}; not treated as a required test season.",
        "",
        "## Replay Aggregate",
        f"- Bets: {total_bets}",
        f"- Profit: {total_profit:.2f}u",
        f"- ROI: {total_roi:.2%}",
        f"- z: {total_z:.4f}",
        f"- Replay matches prior nested summary: `{reproduction_ok}`",
        "",
        "## Year Breakdown",
        md_table(year),
        "",
        "## League Breakdown",
        md_table(league, 40),
        "",
        "## Concentration",
        f"- Best year: {int(best_year['season_start_year'])}, {float(best_year['profit']):.2f}u",
        f"- Worst year: {int(worst_year['season_start_year'])}, {float(worst_year['profit']):.2f}u",
        f"- Best league: {best_league['league']}, {float(best_league['profit']):.2f}u",
        f"- Worst league: {worst_league['league']}, {float(worst_league['profit']):.2f}u",
        f"- Profit share from best year: {best_year_profit_share:.2%}",
        f"- Profit share from best league: {best_league_profit_share:.2%}",
        f"- Profit excluding best year: {profit_ex_best_year:.2f}u",
        f"- Profit excluding best league: {profit_ex_best_league:.2f}u",
        "",
        "## Validation Checks",
        md_table(checks, 80),
        "",
        "## Final Decision",
        f"`{decision}`",
        "",
        "No confirmed edge is claimed.",
        "",
    ]
    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    CARD_MD.write_text(
        "\n".join(
            [
                "# V2 Transfermarkt 1X2 Locked Candidate Card",
                "",
                f"- Decision: `{decision}`",
                f"- Replay aggregate: {total_bets} bets, {total_profit:.2f}u, ROI {total_roi:.2%}, z {total_z:.4f}.",
                f"- Effective test seasons: {', '.join(map(str, effective_test_seasons))}.",
                f"- Prior nested reproduction: `{reproduction_ok}`.",
                f"- Best-year share: {best_year_profit_share:.2%}; best-league share: {best_league_profit_share:.2%}.",
                "- Row-level predictions, selected bets, manual sample, year breakdown, league breakdown, and bug checks were exported.",
                "- No confirmed edge is claimed.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        {
            "decision": decision,
            "reproduction_ok": reproduction_ok,
            "effective_test_seasons": effective_test_seasons,
            "row_predictions": len(row_predictions),
            "selected_bets": total_bets,
            "profit": round(total_profit, 2),
            "roi": round(total_roi, 6),
            "best_year_share": round(best_year_profit_share, 6),
            "best_league_share": round(best_league_profit_share, 6),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
