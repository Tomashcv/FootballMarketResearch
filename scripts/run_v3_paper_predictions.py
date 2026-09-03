from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format.*")
warnings.filterwarnings("ignore", message="Parsing dates.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import build_football_data_full_scope_1x2 as fd  # noqa: E402
from scripts.run_v3_2025_validation import build_external_exact  # noqa: E402
from scripts.run_v3_exact_reproduction import EXACT_INPUT, old_feature_cols  # noqa: E402
from src.experiments.feature_matrix_v2_tm_1x2_predictive_audit import model_predict, normalize_probs  # noqa: E402
from src.features.internal_elo_features import add_internal_elo_features  # noqa: E402
from src.paper_trading.v3_pipeline import (  # noqa: E402
    FEATURE_GROUP,
    LATEST_CANDIDATE_PICKS,
    LATEST_ROW_PREDICTIONS,
    LATEST_SKIPPED_PICKS,
    LATEST_WARNINGS,
    MODEL,
    RULE_NAME,
    SCOPE,
    add_internal_elo_with_history,
    build_paper_market_dataset,
    build_prediction_adapter,
    build_snapshot_manifest,
    ensure_dirs,
    leakage_check_failed,
    leakage_checks,
    select_candidate_picks_after_quality_gate,
    select_current_raw_norm,
    utc_stamp,
)


def make_predictions(hist_exact: pd.DataFrame, validation: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    hist_adapter = build_prediction_adapter(hist_exact, feature_cols, require_target=True)
    test_adapter = build_prediction_adapter(validation, feature_cols, require_target=False)
    if test_adapter.empty:
        return pd.DataFrame()
    paper_year = int(pd.to_numeric(test_adapter["season_start_year"], errors="coerce").max())
    scoped_train = hist_adapter[
        hist_adapter["season_start_year"].notna()
        & ~hist_adapter["league"].isin({"E1", "E2", "E3"})
        & hist_adapter["season_start_year"].astype(int).lt(paper_year)
    ].copy()
    scoped_test = test_adapter[
        test_adapter["season_start_year"].notna()
        & ~test_adapter["league"].isin({"E1", "E2", "E3"})
        & test_adapter["season_start_year"].astype(int).eq(paper_year)
    ].copy()
    if len(scoped_train) < 500 or scoped_test.empty:
        return pd.DataFrame()
    prob = model_predict(MODEL, scoped_train, scoped_test, feature_cols, np.random.default_rng(20260701))
    pred = scoped_test[
        [
            "match_id",
            "full_scope_match_id",
            "canonical_match_id",
            "logical_match_key",
            "source_file",
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
            "x1_odds_source",
            "clubelo_both_found_flag",
            "tm_both_value_found_flag",
            "tm_match_feature_available",
            "classification",
        ]
    ].copy()
    pred[["model_home_prob", "model_draw_prob", "model_away_prob"]] = normalize_probs(prob)
    pred = pred.rename(
        columns={
            "x1x2_avg_prob_home": "market_home_prob",
            "x1x2_avg_prob_draw": "market_draw_prob",
            "x1x2_avg_prob_away": "market_away_prob",
            "x1x2_avg_odds_away": "selected_odds",
        }
    )
    pred["away_edge"] = pred["model_away_prob"] - pred["market_away_prob"]
    pred["scope"] = SCOPE
    pred["model"] = MODEL
    pred["feature_group"] = FEATURE_GROUP
    pred["rule_name"] = RULE_NAME
    pred["selected_side"] = "away"
    pred["prediction_label"] = "research_only_paper"
    return pred


def main() -> None:
    ensure_dirs()
    run_id = f"v3paper_{utc_stamp()}"
    warnings: list[str] = []
    fd.DATA_ROOTS = [fd.ROOT / "data/raw"]
    inventory, norm_all = fd.discover_and_normalize()
    current_norm, current_warnings = select_current_raw_norm(norm_all)
    warnings.extend(current_warnings)
    validation_market, _marketed, skipped_no_odds, market_warnings = build_paper_market_dataset(fd, current_norm)
    warnings.extend(market_warnings)
    manifest = build_snapshot_manifest(run_id, current_norm, validation_market, warnings)
    if validation_market.empty:
        pd.DataFrame().to_csv(LATEST_ROW_PREDICTIONS, index=False)
        pd.DataFrame().to_csv(LATEST_CANDIDATE_PICKS, index=False)
        skipped_no_odds.to_csv(LATEST_SKIPPED_PICKS, index=False)
        pd.DataFrame({"warning": warnings or ["No current paper prediction rows."]}).to_csv(LATEST_WARNINGS, index=False)
        print("v3_paper_pipeline_blocked_missing_current_data")
        return
    hist_exact = pd.read_csv(EXACT_INPUT, low_memory=False)
    validation = build_external_exact(validation_market)
    validation = add_internal_elo_with_history(hist_exact, validation, add_internal_elo_features)
    feature_cols = old_feature_cols()
    pred = make_predictions(hist_exact, validation, feature_cols)
    checks = leakage_checks(validation, pred)
    if leakage_check_failed(checks):
        warnings.extend(checks.loc[checks["status"].eq("fail"), "details"].astype(str).tolist())
        checks.to_csv(LATEST_WARNINGS, index=False)
        pd.DataFrame().to_csv(LATEST_ROW_PREDICTIONS, index=False)
        pd.DataFrame().to_csv(LATEST_CANDIDATE_PICKS, index=False)
        skipped_no_odds.to_csv(LATEST_SKIPPED_PICKS, index=False)
        print("v3_paper_pipeline_blocked_data_quality")
        return
    picks, skipped_rule, checks = select_candidate_picks_after_quality_gate(validation, pred, run_id, str(manifest["run_id"]))
    pred.to_csv(LATEST_ROW_PREDICTIONS, index=False)
    picks.to_csv(LATEST_CANDIDATE_PICKS, index=False)
    skipped = pd.concat([skipped_no_odds, skipped_rule], ignore_index=True, sort=False)
    skipped.to_csv(LATEST_SKIPPED_PICKS, index=False)
    pd.DataFrame({"warning": warnings or ["none"]}).to_csv(LATEST_WARNINGS, index=False)
    print("v3_paper_pipeline_built_research_only")
    print(f"run_id={run_id} predictions={len(pred)} candidate_picks={len(picks)}")


if __name__ == "__main__":
    main()
