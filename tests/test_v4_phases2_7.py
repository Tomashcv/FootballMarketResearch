import json

import numpy as np
import pandas as pd

from src.v4.data.market_panel import deterministic_match_id, validate_panel
from src.v4.data.phase1b_audit import ah_price_shift, no_vig_probabilities, ou_price_shift, price_clv
from src.v4.models.dynamic_scoreline import ScoreCandidate, run_candidate
from src.v4.reporting.phase7 import haircut_odds, reproducible_cluster_bootstrap, settle_1x2
from src.v4.validation.nested import assert_fold_isolation, temporal_folds


def test_deterministic_fixture_identity_and_team_normalization():
    a = deterministic_match_id("E0", "2024-08-17", "Man United", "Fulham")
    b = deterministic_match_id("E0", "2024-08-17", "Man-United", "Fulham")
    assert a == b
    assert a != deterministic_match_id("E0", "2024-08-18", "Man United", "Fulham")


def test_panel_validation_rejects_duplicate_identity_and_feature_leakage():
    panel = pd.DataFrame({
        "id__canonical_match_id": [1, 1], "id__weekday": ["Saturday", "Sunday"],
        "quality__safe_snapshot_timing": [True, True], "label_close__ah_same_line": [False, False],
        "label_close__ah_same_line_prob_shift_home": [np.nan, np.nan],
        "label_close__ah_same_line_prob_shift_away": [np.nan, np.nan],
        "feature_snapshot__ah_home_line": [-1.25, 0.25],
        "feature_snapshot__ou25_total_line": [2.5, 2.5], "label_close__ou25_total_line": [2.5, 2.5],
    })
    contract = {"approved_feature_columns": ["feature_snapshot__ah_home_line", "label_close__bad"]}
    checks = validate_panel(panel, contract).set_index("check")
    assert checks.loc["unique_canonical_match_id", "status"] == "fail"
    assert checks.loc["no_closing_features", "status"] == "fail"


def test_no_vig_and_clv_sign_conventions():
    p = no_vig_probabilities([2.0, 4.0, 4.0])
    assert np.allclose(p, [0.5, 0.25, 0.25])
    assert price_clv(2.20, 2.00) == pytest_approx(0.10)
    assert price_clv(1.80, 2.00) == pytest_approx(-0.10)


def pytest_approx(value):
    import pytest
    return pytest.approx(value)


def test_ah_and_ou_different_lines_are_not_price_compared():
    assert np.isnan(ah_price_shift(-1.25, -1.0, [1.9, 1.9], [2.0, 1.8])).all()
    assert np.isnan(ou_price_shift(2.5, 2.75, [1.9, 1.9], [2.0, 1.8])).all()
    assert np.isfinite(ah_price_shift(-1.25, -1.25, [1.9, 1.9], [2.0, 1.8])).all()


def test_dynamic_engine_predicts_before_same_match_update():
    matches = pd.DataFrame([
        {"canonical_id": 1, "match_date": "2020-01-01", "league": "E0", "season": "2020", "home": "A", "away": "B", "home_goals": 5, "away_goals": 0, "ftr": "H", "ah_line": 0.0},
        {"canonical_id": 2, "match_date": "2020-01-08", "league": "E0", "season": "2020", "home": "A", "away": "B", "home_goals": 0, "away_goals": 0, "ftr": "D", "ah_line": 0.0},
    ])
    matches.match_date = pd.to_datetime(matches.match_date)
    out = run_candidate(matches, ScoreCandidate("test", 0.0, 10.0, 0.0))
    assert out.loc[0, "feature_history__home_prior_matches"] == 0
    assert out.loc[1, "feature_history__home_prior_matches"] == 1
    assert out.diagnostic__prediction_before_update.all()


def test_nested_fold_order_and_isolation():
    folds = temporal_folds([2012, 2013, 2014, 2015, 2016])
    assert [f["test_season"] for f in folds] == [2015, 2016]
    for fold in folds:
        assert_fold_isolation(fold)
        assert fold["test_season"] not in fold["train_seasons"]
        assert fold["calibration_season"] != fold["test_season"]


def test_settlement_and_haircut_math():
    assert settle_1x2("away", "A", 2.5) == 1.5
    assert settle_1x2("home", "A", 2.5) == -1.0
    assert haircut_odds(2.0, 0.01) == 1.98
    assert settle_1x2("home", "H", haircut_odds(2.0, 0.01)) == pytest_approx(0.98)


def test_cluster_bootstrap_is_reproducible():
    bets = pd.DataFrame({"season": [1, 1, 2], "league": ["A", "B", "A"], "profit": [1.0, -1.0, 2.0]})
    a = reproducible_cluster_bootstrap(bets, ["season", "league"], iterations=10, seed=7)
    b = reproducible_cluster_bootstrap(bets, ["season", "league"], iterations=10, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_feature_group_artifact_has_namespace_and_no_close_or_result():
    groups = json.loads(open("data/processed/v4/v4_model_feature_groups_v1.json").read())
    flat = [c for columns in groups.values() for c in columns]
    assert all(c.startswith(("feature_snapshot__", "feature_history__", "feature_external__")) for c in flat)
    assert not any(c.startswith(("label_close__", "result__")) for c in flat)
