from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.experiments.feature_matrix_v2_tm_1x2_predictive_audit import (  # noqa: E402
    CLASS_TO_INT,
    MARKET_BASELINE,
    TEST_YEARS,
    brier_multi,
    ece_multi,
    feature_groups,
    model_predict,
    normalize_probs,
)


INPUT = Path(
    "data/processed/super_csvs/research_ready_plus/football_data_extended/"
    "super_1x2_football_data_top5_extended_full_features_research_v1_deduped_review_fixed.csv"
)
OLD_V2_MATRIX = Path("data/processed/features/football_feature_matrix_v2_transfermarkt_partial.csv")
OLD_VALUE_REPLAY = Path("outputs/reports/feature_matrix_v3_clubelo_locked_rule_value_replay.csv")
OLD_YEAR = Path("outputs/reports/feature_matrix_v3_clubelo_locked_rule_year_breakdown.csv")
OLD_LEAGUE = Path("outputs/reports/feature_matrix_v3_clubelo_locked_rule_league_breakdown.csv")
OLD_SELECTED = Path("outputs/reports/feature_matrix_v2_tm_1x2_locked_selected_bets.csv")
OUT_DIR = Path("outputs/reports/v3_reproduction")

SCOPE = "scope_C_top_divisions_ex_e1_e2_e3"
MODEL = "xgboost_market_residual_multiclass"
OLD_FEATURE_GROUP = "v3_tm_plus_clubelo_core_staleness"
OLD_BASE_GROUP = "x1_market_plus_tm_all"
LOCKED_RULE_SCHEDULE = {
    2021: ("away_edge_0.01_odds_1.5", 0.01, 1.5),
    2022: ("away_edge_0.01_odds_1.5", 0.01, 1.5),
    2023: ("away_edge_0.015_odds_1.5", 0.015, 1.5),
    2024: ("away_edge_0.015_odds_1.5", 0.015, 1.5),
    2025: ("away_edge_0.015_odds_1.5", 0.015, 1.5),
}

CLUBELO_CORE = [
    "clubelo_home_rating",
    "clubelo_away_rating",
    "clubelo_diff",
    "clubelo_abs_diff",
    "clubelo_missing_home",
    "clubelo_missing_away",
    "clubelo_missing_both",
]
CLUBELO_STALENESS = [
    "clubelo_staleness_home",
    "clubelo_staleness_away",
    "clubelo_diff_minus_internal_elo_diff",
]

LEAGUE_MAP = {
    "england_premier_league": "E0",
    "spain_laliga": "SP1",
    "germany_bundesliga": "D1",
    "italy_serie_a": "I1",
    "france_ligue_1": "F1",
}

DIRECT_MAP = {
    "x1x2_avg_prob_home": "x1_home_no_vig_prob",
    "x1x2_avg_prob_draw": "x1_draw_no_vig_prob",
    "x1x2_avg_prob_away": "x1_away_no_vig_prob",
    "x1x2_avg_market_overround": "x1_overround",
    "x1x2_avg_odds_home": "x1_home_odds",
    "x1x2_avg_odds_draw": "x1_draw_odds",
    "x1x2_avg_odds_away": "x1_away_odds",
    "home_tm_squad_value_total_prior365": "home_tm_total_market_value",
    "away_tm_squad_value_total_prior365": "away_tm_total_market_value",
    "home_minus_away_tm_squad_value_total_prior365": "tm_total_market_value_diff",
    "home_tm_squad_value_mean_prior365": "home_tm_avg_market_value",
    "away_tm_squad_value_mean_prior365": "away_tm_avg_market_value",
    "home_tm_squad_value_median_prior365": "home_tm_median_market_value",
    "away_tm_squad_value_median_prior365": "away_tm_median_market_value",
    "home_tm_squad_value_top11_prior365": "home_tm_top11_market_value",
    "away_tm_squad_value_top11_prior365": "away_tm_top11_market_value",
    "home_minus_away_tm_squad_value_top11_prior365": "tm_top11_market_value_diff",
    "home_tm_squad_valued_player_count_prior365": "home_tm_players_with_value_count",
    "away_tm_squad_valued_player_count_prior365": "away_tm_players_with_value_count",
    "home_minus_away_tm_squad_valued_player_count_prior365": "tm_players_with_value_count_diff",
    "home_tm_arrivals_count_365d": "home_tm_incoming_transfer_count_365d",
    "away_tm_arrivals_count_365d": "away_tm_incoming_transfer_count_365d",
    "home_minus_away_tm_arrivals_count_365d": None,
    "home_tm_departures_count_365d": "home_tm_outgoing_transfer_count_365d",
    "away_tm_departures_count_365d": "away_tm_outgoing_transfer_count_365d",
    "home_minus_away_tm_departures_count_365d": None,
    "home_tm_avg_valuation_staleness_days_prior365": "home_tm_latest_valuation_days_ago",
    "away_tm_avg_valuation_staleness_days_prior365": "away_tm_latest_valuation_days_ago",
    "tm_has_valuation_data_home": "home_tm_value_found_flag",
    "tm_has_valuation_data_away": "away_tm_value_found_flag",
    "tm_home_feature_available": "home_tm_value_found_flag",
    "tm_away_feature_available": "away_tm_value_found_flag",
    "tm_match_feature_available": "tm_both_value_found_flag",
    "tm_fixture_mapped": "transfermarkt_available",
    "clubelo_home_rating": "home_clubelo_rating",
    "clubelo_away_rating": "away_clubelo_rating",
    "clubelo_diff": "clubelo_diff_home_minus_away",
    "clubelo_staleness_home": "home_clubelo_days_stale",
    "clubelo_staleness_away": "away_clubelo_days_stale",
}


def z_score(profit: pd.Series) -> float:
    if len(profit) <= 1:
        return 0.0
    sd = float(profit.std(ddof=1))
    return float(profit.sum() / (sd * math.sqrt(len(profit)))) if sd > 0 else 0.0


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = profit.cumsum()
    return float((curve - curve.cummax()).min())


def load_old_feature_cols() -> list[str]:
    old_sample = pd.read_csv(OLD_V2_MATRIX, nrows=1000, low_memory=False)
    tm_cols = feature_groups(old_sample)[OLD_BASE_GROUP]
    return sorted(set(tm_cols + CLUBELO_CORE + CLUBELO_STALENESS))


def result_to_old(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {"h", "home", "home_win", "1"}:
        return "H"
    if s in {"d", "draw", "x"}:
        return "D"
    if s in {"a", "away", "away_win", "2"}:
        return "A"
    return str(value).strip().upper() if str(value).strip().upper() in CLASS_TO_INT else None


def build_adapter(clean: pd.DataFrame, old_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame(index=clean.index)
    out["match_id"] = clean["extended_canonical_match_id"].fillna(clean["canonical_match_id"]).astype(str)
    out["match_date"] = pd.to_datetime(clean["match_date"], errors="coerce")
    out["league"] = clean["competition_slug"].map(LEAGUE_MAP)
    out["season_start_year"] = pd.to_numeric(clean["season_start_year"], errors="coerce").astype("Int64")
    out["season_end_year"] = out["season_start_year"].astype("Int64") + 1
    out["home_team"] = clean.get("home_team_raw", clean.get("home_team_normalized", "")).astype(str)
    out["away_team"] = clean.get("away_team_raw", clean.get("away_team_normalized", "")).astype(str)
    out["target_outcome_1x2"] = clean["result_1x2"].map(result_to_old)
    out["target_y"] = out["target_outcome_1x2"].map(CLASS_TO_INT)
    out["target_1x2_available"] = out["target_y"].notna()
    out["source_file_cleaned"] = clean.get("source_file", pd.Series("", index=clean.index)).astype(str)
    out["x1_odds_source_cleaned"] = clean.get("x1_odds_source", pd.Series("", index=clean.index)).astype(str)

    mapping_rows = []
    for old_col in old_cols:
        src = DIRECT_MAP.get(old_col)
        if src and src in clean.columns:
            out[old_col] = pd.to_numeric(clean[src], errors="coerce")
            status = "mapped_direct"
        elif old_col == "clubelo_abs_diff":
            out[old_col] = pd.to_numeric(clean["clubelo_diff_home_minus_away"], errors="coerce").abs()
            src = "abs(clubelo_diff_home_minus_away)"
            status = "derived_direct"
        elif old_col == "clubelo_missing_home":
            out[old_col] = ~pd.to_numeric(clean["home_clubelo_found_flag"], errors="coerce").fillna(0).astype(bool)
            src = "not home_clubelo_found_flag"
            status = "derived_direct"
        elif old_col == "clubelo_missing_away":
            out[old_col] = ~pd.to_numeric(clean["away_clubelo_found_flag"], errors="coerce").fillna(0).astype(bool)
            src = "not away_clubelo_found_flag"
            status = "derived_direct"
        elif old_col == "clubelo_missing_both":
            out[old_col] = ~pd.to_numeric(clean["clubelo_both_found_flag"], errors="coerce").fillna(0).astype(bool)
            src = "not clubelo_both_found_flag"
            status = "derived_direct"
        elif old_col == "home_minus_away_tm_arrivals_count_365d":
            out[old_col] = pd.to_numeric(clean["home_tm_incoming_transfer_count_365d"], errors="coerce") - pd.to_numeric(
                clean["away_tm_incoming_transfer_count_365d"], errors="coerce"
            )
            src = "home_tm_incoming_transfer_count_365d-away_tm_incoming_transfer_count_365d"
            status = "derived_direct"
        elif old_col == "home_minus_away_tm_departures_count_365d":
            out[old_col] = pd.to_numeric(clean["home_tm_outgoing_transfer_count_365d"], errors="coerce") - pd.to_numeric(
                clean["away_tm_outgoing_transfer_count_365d"], errors="coerce"
            )
            src = "home_tm_outgoing_transfer_count_365d-away_tm_outgoing_transfer_count_365d"
            status = "derived_direct"
        elif old_col == "home_minus_away_tm_avg_valuation_staleness_days_prior365":
            out[old_col] = pd.to_numeric(clean["home_tm_latest_valuation_days_ago"], errors="coerce") - pd.to_numeric(
                clean["away_tm_latest_valuation_days_ago"], errors="coerce"
            )
            src = "home_tm_latest_valuation_days_ago-away_tm_latest_valuation_days_ago"
            status = "derived_direct"
        else:
            out[old_col] = np.nan
            src = ""
            status = "missing_in_cleaned_dataset"
        mapping_rows.append({"old_feature": old_col, "cleaned_source": src or "", "mapping_status": status})

    # Old loader required these market fields even if not in the feature group list.
    for old_col, src in DIRECT_MAP.items():
        if old_col not in out.columns and src and src in clean.columns:
            out[old_col] = pd.to_numeric(clean[src], errors="coerce")

    valid = (
        out["target_1x2_available"].fillna(False)
        & out["league"].notna()
        & out[["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]].notna().all(axis=1)
        & out[["x1x2_avg_odds_home", "x1x2_avg_odds_draw", "x1x2_avg_odds_away"]].notna().all(axis=1)
        & out[["x1x2_avg_odds_home", "x1x2_avg_odds_draw", "x1x2_avg_odds_away"]].gt(1).all(axis=1)
    )
    out = out[valid].copy()
    out["target_y"] = out["target_y"].astype(int)
    return out.sort_values(["match_date", "match_id"]).reset_index(drop=True), pd.DataFrame(mapping_rows)


def annual_predictions(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    preds = []
    rng = np.random.default_rng(20260701)
    for year in TEST_YEARS:
        train = df[df["season_start_year"].astype(int).lt(year)].copy()
        test = df[df["season_start_year"].astype(int).eq(year)].copy()
        if len(train) < 500 or len(test) == 0:
            continue
        prob = model_predict(MODEL, train, test, cols, rng)
        pred = test[
            [
                "match_id",
                "match_date",
                "league",
                "season_start_year",
                "season_end_year",
                "home_team",
                "away_team",
                "target_y",
                "target_outcome_1x2",
                "x1x2_avg_prob_home",
                "x1x2_avg_prob_draw",
                "x1x2_avg_prob_away",
                "x1x2_avg_odds_home",
                "x1x2_avg_odds_draw",
                "x1x2_avg_odds_away",
                "source_file_cleaned",
                "x1_odds_source_cleaned",
            ]
        ].copy()
        pred[["prob_home", "prob_draw", "prob_away"]] = prob
        pred["fold_test_year"] = year
        preds.append(pred)
    return pd.concat(preds, ignore_index=True, sort=False) if preds else pd.DataFrame()


def add_value_columns(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for side, cls in [("home", 0), ("draw", 1), ("away", 2)]:
        out[f"{side}_edge"] = out[f"prob_{side}"] - out[f"x1x2_avg_prob_{side}"]
        out[f"{side}_profit"] = np.where(out["target_y"].eq(cls), out[f"x1x2_avg_odds_{side}"] - 1.0, -1.0)
    return out


def select_locked(value_pred: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for year, (rule, edge, min_odds) in LOCKED_RULE_SCHEDULE.items():
        current = value_pred[value_pred["season_start_year"].astype(int).eq(year)].copy()
        selected = current[current["away_edge"].ge(edge) & current["x1x2_avg_odds_away"].ge(min_odds)].copy()
        selected["selected_rule"] = rule
        selected["side"] = "away"
        selected["profit"] = selected["away_profit"]
        parts.append(selected)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def summary(selected: pd.DataFrame) -> dict[str, object]:
    bets = int(len(selected))
    profit = float(selected["profit"].sum()) if bets else 0.0
    return {
        "feature_group": OLD_FEATURE_GROUP,
        "dataset": "cleaned_extended_review_fixed",
        "bets": bets,
        "profit": profit,
        "roi": profit / bets if bets else 0.0,
        "z": z_score(selected["profit"]) if bets else 0.0,
        "max_drawdown": max_drawdown(selected["profit"]) if bets else 0.0,
        "years": int(selected["season_start_year"].nunique()) if bets else 0,
        "leagues": int(selected["league"].nunique()) if bets else 0,
        "best_year": selected.groupby("season_start_year")["profit"].sum().idxmax() if bets else "",
        "best_league": selected.groupby("league")["profit"].sum().idxmax() if bets else "",
    }


def breakdown(selected: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in selected.groupby(by, dropna=False):
        rows.append(
            {
                by: key,
                "bets": int(len(g)),
                "profit": float(g["profit"].sum()),
                "roi": float(g["profit"].mean()) if len(g) else 0.0,
                "z": z_score(g["profit"]),
            }
        )
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True) if rows else pd.DataFrame(columns=[by, "bets", "profit", "roi", "z"])


def merge_old_breakdown(clean_breakdown: pd.DataFrame, by: str) -> pd.DataFrame:
    old_path = OLD_YEAR if by == "season_start_year" else OLD_LEAGUE
    if not old_path.exists() or clean_breakdown.empty:
        return clean_breakdown
    old = pd.read_csv(old_path)
    old = old[old["feature_group"].eq(OLD_FEATURE_GROUP)].copy()
    if old.empty:
        return clean_breakdown
    old = old.rename(columns={"bets": "old_bets", "profit": "old_profit", "roi": "old_roi", "z": "old_z"})
    out = clean_breakdown.merge(old[[by, "old_bets", "old_profit", "old_roi", "old_z"]], on=by, how="outer")
    out["clean_minus_old_bets"] = out["bets"].fillna(0) - out["old_bets"].fillna(0)
    out["clean_minus_old_profit"] = out["profit"].fillna(0) - out["old_profit"].fillna(0)
    return out.sort_values(by).reset_index(drop=True)


def metric_rows(value_pred: pd.DataFrame) -> pd.DataFrame:
    if value_pred.empty:
        return pd.DataFrame()
    y = value_pred["target_y"].to_numpy(dtype=int)
    prob = normalize_probs(value_pred[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float))
    market = normalize_probs(value_pred[["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]].to_numpy(dtype=float))
    return pd.DataFrame(
        [
            {
                "feature_group": OLD_FEATURE_GROUP,
                "rows": int(len(value_pred)),
                "log_loss": float(log_loss(y, prob, labels=[0, 1, 2])),
                "market_log_loss": float(log_loss(y, market, labels=[0, 1, 2])),
                "delta_log_loss_vs_market": float(log_loss(y, prob, labels=[0, 1, 2]) - log_loss(y, market, labels=[0, 1, 2])),
                "brier": brier_multi(y, prob),
                "market_brier": brier_multi(y, market),
                "delta_brier_vs_market": brier_multi(y, prob) - brier_multi(y, market),
                "ece": ece_multi(y, prob),
                "market_ece": ece_multi(y, market),
            }
        ]
    )


def old_reference() -> pd.DataFrame:
    if not OLD_VALUE_REPLAY.exists():
        return pd.DataFrame()
    old = pd.read_csv(OLD_VALUE_REPLAY)
    return old[old["feature_group"].eq(OLD_FEATURE_GROUP)].copy()


def write_reports(clean: pd.DataFrame, adapter: pd.DataFrame, mapping: pd.DataFrame, pred: pd.DataFrame, selected: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old_ref = old_reference()
    old_cols = set(mapping["old_feature"])
    mapped = mapping[mapping["mapping_status"].ne("missing_in_cleaned_dataset")]
    missing = mapping[mapping["mapping_status"].eq("missing_in_cleaned_dataset")]
    used_cols = [c for c in mapping["old_feature"] if c in adapter.columns and adapter[c].notna().any()]

    config_rows = [
        {"record_type": "config", "key": "scope", "value": SCOPE},
        {"record_type": "config", "key": "model", "value": MODEL},
        {"record_type": "config", "key": "old_feature_group", "value": OLD_FEATURE_GROUP},
        {"record_type": "config", "key": "base_old_feature_group", "value": OLD_BASE_GROUP},
        {"record_type": "config", "key": "locked_rule_schedule", "value": str(LOCKED_RULE_SCHEDULE)},
        {"record_type": "config", "key": "old_feature_count", "value": str(len(old_cols))},
        {"record_type": "config", "key": "mapped_or_derived_feature_count", "value": str(len(mapped))},
        {"record_type": "config", "key": "usable_non_null_feature_count", "value": str(len(used_cols))},
        {"record_type": "config", "key": "missing_old_feature_count", "value": str(len(missing))},
        {"record_type": "config", "key": "class_order", "value": "H,D,A"},
    ]
    config_df = pd.DataFrame(config_rows)
    feature_df = mapping.copy()
    feature_df.insert(0, "record_type", "feature_mapping")
    feature_df.insert(1, "key", "")
    feature_df.insert(2, "value", "")
    pd.concat([config_df, feature_df], ignore_index=True, sort=False).to_csv(OUT_DIR / "v3_old_config_extracted.csv", index=False)

    clean_sum = pd.DataFrame([summary(selected)])
    if not old_ref.empty:
        for col in ["bets", "profit", "roi", "z"]:
            clean_sum[f"old_{col}"] = old_ref.iloc[0][col]
            clean_sum[f"clean_minus_old_{col}"] = clean_sum[col].iloc[0] - old_ref.iloc[0][col]
    clean_sum["prediction_rows"] = len(pred)
    clean_sum["clean_input_rows"] = len(clean)
    clean_sum["adapter_valid_rows"] = len(adapter)
    clean_sum["old_required_features"] = len(old_cols)
    clean_sum["mapped_or_derived_features"] = len(mapped)
    clean_sum["missing_old_features"] = len(missing)
    clean_sum["exact_feature_contract_reconstructed"] = len(missing) == 0
    clean_sum.to_csv(OUT_DIR / "v3_clean_reproduction_summary.csv", index=False)

    season_breakdown = merge_old_breakdown(breakdown(selected, "season_start_year"), "season_start_year")
    league_breakdown = merge_old_breakdown(breakdown(selected, "league"), "league")
    season_breakdown.to_csv(OUT_DIR / "v3_by_season.csv", index=False)
    league_breakdown.to_csv(OUT_DIR / "v3_by_league.csv", index=False)

    old_sel = pd.read_csv(OLD_SELECTED) if OLD_SELECTED.exists() else pd.DataFrame()
    overlap = pd.DataFrame(
        [
            {
                "old_selected_file_available": OLD_SELECTED.exists(),
                "old_selected_rows": int(len(old_sel)) if not old_sel.empty else 0,
                "clean_selected_rows": int(len(selected)),
                "exact_match_id_overlap_rows": 0,
                "overlap_method": "not_comparable_old_match_ids_vs_extended_canonical_ids",
                "note": "Old selected bets use legacy match_id values; cleaned extended rows use deterministic extended canonical IDs.",
            }
        ]
    )
    overlap.to_csv(OUT_DIR / "v3_overlap_audit.csv", index=False)

    leakage = pd.DataFrame(
        [
            {"check": "raw_files_modified", "status": "pass", "details": "script reads processed inputs only"},
            {"check": "extra_sources_joined", "status": "pass", "details": "no joins performed; used supplied cleaned full-feature dataset"},
            {"check": "new_rule_search", "status": "pass", "details": "used fixed old locked away schedule only"},
            {"check": "threshold_optimization", "status": "pass", "details": "no thresholds selected or tuned"},
            {"check": "forbidden_identity_features_used", "status": "pass", "details": "match/team/source identifiers excluded from feature list"},
            {"check": "raw_odds_as_features", "status": "review", "details": "old V3 feature group includes market odds/probabilities by design"},
            {"check": "exact_old_feature_contract_reconstructed", "status": "pass" if len(missing) == 0 else "fail", "details": f"{len(missing)} old required features unavailable in cleaned dataset"},
            {"check": "classification", "status": "pass", "details": "research_only; no confirmed edge"},
        ]
    )
    leakage.to_csv(OUT_DIR / "v3_leakage_checks.csv", index=False)

    metric = metric_rows(pred)
    lines = [
        "# V3 1X2 Reproduction on Cleaned Extended Football-Data",
        "",
        "This is a reproduction/validation audit only. No new rules, thresholds, filters, sources, or confirmed-edge claims were introduced.",
        "",
        f"Decision: `{decision_label(clean_sum.iloc[0], missing)}`",
        "",
        "## Old Config Located",
        "",
        f"- Scope: `{SCOPE}`",
        f"- Model: `{MODEL}`",
        f"- Feature group: `{OLD_FEATURE_GROUP}`",
        f"- Side/rule: away only, locked year schedule `{LOCKED_RULE_SCHEDULE}`",
        "- Settlement: 1X2 away win unit stake profit, using actual away odds.",
        "",
        "## Feature Contract",
        "",
        f"- Old required features: {len(old_cols)}",
        f"- Mapped or directly derived in cleaned dataset: {len(mapped)}",
        f"- Missing in cleaned dataset: {len(missing)}",
        "",
        "The cleaned extended file has a newer, smaller Transfermarkt schema than the old V2/V3 feature matrix. Missing old features were not approximated; they were set unavailable and dropped by the old model routine only if all-null.",
        "",
        "## Old Reference",
        old_ref.to_markdown(index=False) if not old_ref.empty else "_Old V3 reference row not found._",
        "",
        "## Cleaned Reproduction Summary",
        clean_sum.to_markdown(index=False),
        "",
        "## Cleaned Season Breakdown",
        season_breakdown.to_markdown(index=False) if not season_breakdown.empty else "_No selected bets._",
        "",
        "## Cleaned League Breakdown",
        league_breakdown.to_markdown(index=False) if not league_breakdown.empty else "_No selected bets._",
        "",
        "## Side And Odds Source",
        "",
        "- Side: away only, matching the old locked candidate.",
        f"- Odds source counts: `{selected['x1_odds_source_cleaned'].value_counts(dropna=False).to_dict() if len(selected) else {}}`",
        "- Duplicate-row inflation check: cleaned reproduction used the review-fixed deduplicated input and selected one row per cleaned match ID.",
        "",
        "## Predictive Metrics",
        metric.to_markdown(index=False) if not metric.empty else "_No predictions generated._",
        "",
        "## Missing Feature Warning",
        "Because the exact old feature contract was not fully reconstructable on the cleaned dataset, this cannot be treated as an exact survival of the old V3 candidate.",
    ]
    (OUT_DIR / "v3_reproduction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    decision = [
        "# V3 Reproduction Decision",
        "",
        f"Decision: `{decision_label(clean_sum.iloc[0], missing)}`",
        "",
        "The exact old V3 configuration was found, but the cleaned extended dataset does not contain the full old Transfermarkt feature contract. The compatibility replay is therefore diagnostic only.",
        "",
        "No confirmed edge is claimed.",
    ]
    (OUT_DIR / "v3_reproduction_decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")


def decision_label(row: pd.Series, missing: pd.DataFrame) -> str:
    if len(missing) > 0:
        return "v3_reproduction_rejected_on_clean_data"
    if row["bets"] <= 0 or row["profit"] <= 0 or row["roi"] <= 0:
        return "v3_reproduction_rejected_on_clean_data"
    if row["z"] >= 1.0 and row["years"] >= 4 and row["leagues"] >= 3:
        return "v3_reproduction_survives_as_research_only"
    return "v3_reproduction_rejected_on_clean_data"


def main() -> None:
    clean = pd.read_csv(INPUT, low_memory=False)
    old_cols = load_old_feature_cols()
    adapter, mapping = build_adapter(clean, old_cols)
    pred = annual_predictions(adapter, old_cols)
    value_pred = add_value_columns(pred)
    selected = select_locked(value_pred)
    write_reports(clean, adapter, mapping, value_pred, selected)


if __name__ == "__main__":
    main()
