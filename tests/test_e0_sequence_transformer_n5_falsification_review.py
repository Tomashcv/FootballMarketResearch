import pandas as pd

from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced
from src.experiments import e0_away_ah_sequence_transformer_n5_falsification_review as falsification
from src.experiments import e0_away_ah_sequence_transformer_n5_no_seq_odds_falsification_review as no_seq
from tests.test_e0_team_sequence_model_review import sequence_sample


def test_without_team_current_cats_removes_team_features():
    data = sequence_sample()
    config = falsification.VariantConfig(
        "without_team_current_cats_ablation",
        numeric_columns=tuple(advanced.NUMERIC_FEATURE_COLUMNS),
        categorical_columns=(),
    )

    numeric, categorical = falsification.current_feature_columns(data, config)

    assert "HomeTeam" not in numeric + categorical
    assert "AwayTeam" not in numeric + categorical


def test_no_sequence_odds_ablation_removes_ah_and_odds_features():
    data = sequence_sample()
    config = falsification.VariantConfig(
        "no_sequence_odds_ah_ablation",
        sequence_columns=tuple(
            column
            for column in falsification.sequence_review.SEQUENCE_FEATURE_COLUMNS
            if column not in falsification.SEQUENCE_ODDS_COLUMNS
        ),
    )
    home, away = falsification.build_sequence_arrays(data, config)

    assert home.shape[-1] == len(config.sequence_columns)
    assert away.shape[-1] == len(config.sequence_columns)
    for column in falsification.SEQUENCE_ODDS_COLUMNS:
        assert column not in config.sequence_columns


def test_leakage_audit_passes_for_strict_prior_sequences():
    data = sequence_sample()
    audit = falsification.leakage_audit(data, falsification.VariantConfig("locked_ensemble"))

    assert audit["passed"].all()


def test_no_seq_odds_locked_columns_exclude_sequence_ah_and_odds():
    for column in falsification.SEQUENCE_ODDS_COLUMNS:
        assert column not in no_seq.NO_SEQ_ODDS_COLUMNS


def test_random_sequence_rows_controls_keep_shape_and_prior_dates():
    data = sequence_sample()
    config = no_seq.no_seq_config("random_sequence_rows_same_team_negative_control", random_sequence_rows="same_team")
    home, away = falsification.build_sequence_arrays(data, config)
    audit = falsification.leakage_audit(data, config)

    assert home.shape == (len(data), falsification.LOCKED_SEQUENCE_LENGTH, len(no_seq.NO_SEQ_ODDS_COLUMNS))
    assert away.shape == home.shape
    assert audit["passed"].all()
