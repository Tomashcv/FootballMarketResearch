import numpy as np
import torch

from src.experiments import e0_away_ah_deep_cross_wide_deep_falsification_review as review
from tests.test_e0_team_sequence_model_review import sequence_sample


def test_current_network_outputs_one_logit_per_row_for_all_locked_models():
    x = torch.randn(4, 7)

    for model_type in [
        "wide_linear_residual",
        "small_deep_mlp",
        "deep_cross_network",
        "wide_deep_combined",
    ]:
        model = review.CurrentTabularNetwork(7, model_type)
        output = model(x)
        assert output.shape == (4,)


def test_random_feature_noise_is_deterministic_and_shape_preserving():
    x = np.ones((5, 3), dtype=np.float32)

    first = review._noise_like(x, 123)
    second = review._noise_like(x, 123)

    assert first.shape == x.shape
    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, x)


def test_leakage_audit_current_features_exclude_closing_columns():
    data = sequence_sample()
    data["AvgCAHA"] = 1.91
    audit = review.leakage_audit(data)

    assert audit["passed"].all()
    feature_detail = audit[audit["check"].eq("closing_absent_current_features")]["detail"].iloc[0]
    assert "AvgCAHA" not in feature_detail


def test_selected_bets_for_probability_uses_market_residual_scores():
    data = sequence_sample()
    probability = np.full(len(data), 0.75)
    selected = {"selected_threshold": -1.0, "selected_score_threshold": 0.01}

    bets = review.selected_bets_for_probability(data, probability, selected, "unit", "wide_linear_residual", "ensemble")

    assert "model_score" in bets.columns
    assert (bets["model_score"] > 0.0).all()
