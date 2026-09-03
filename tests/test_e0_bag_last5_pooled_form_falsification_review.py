import numpy as np
import pandas as pd

from src.experiments import e0_away_ah_bag_last5_pooled_form_falsification_review as pooled
from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced
from tests.test_e0_team_sequence_model_review import sequence_sample


def test_pooled_features_use_only_prior_matches():
    data = sequence_sample()
    features = pooled.build_pooled_last5_features(data)

    assert data.loc[4, "FTHG"] == 99
    assert features.loc[4, "home_l5_mean_goals_for"] != 99
    assert np.isclose(features.loc[4, "home_l5_mean_goals_for"], (0 + 1 + 0 + 1) / 4)


def test_pooled_features_exclude_sequence_odds_and_ah_columns():
    names = pooled.pooled_feature_names()

    for column in pooled.EXCLUDED_SEQUENCE_COLUMNS:
        assert all(column not in name for name in names)


def test_dataframe_with_pooled_features_has_expected_width():
    data = sequence_sample()
    output, names = pooled.dataframe_with_pooled_features(data)

    assert len(names) == len(pooled.POOLED_SOURCE_COLUMNS) * 3
    assert set(names).issubset(output.columns)


def test_randomized_pooled_features_preserve_shape_and_columns():
    data, names = pooled.dataframe_with_pooled_features(sequence_sample())
    randomized = pooled._randomize_pooled_columns(data, names, np.random.default_rng(7))

    assert randomized.shape == data.shape
    assert list(randomized.columns) == list(data.columns)


def test_leakage_audit_passes_for_pooled_last5():
    data, names = pooled.dataframe_with_pooled_features(sequence_sample())
    audit = pooled.leakage_audit(data, names)

    assert audit["passed"].all()


def test_closing_columns_absent_from_pooled_and_current_features():
    data, names = pooled.dataframe_with_pooled_features(sequence_sample())
    data["AvgCAHA"] = 1.91

    numeric, categorical = pooled.current_columns(data, True, names)

    assert "AvgCAHA" not in numeric + categorical
    assert all("CAH" not in column for column in names)
    assert advanced.TARGET_COLUMN not in names
