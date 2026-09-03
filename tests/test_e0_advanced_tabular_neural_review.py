import pandas as pd

from src.experiments.e0_away_ah_advanced_tabular_neural_review import CATEGORICAL_FEATURE_COLUMNS
from src.experiments.e0_away_ah_advanced_tabular_neural_review import TARGET_COLUMN
from src.experiments.e0_away_ah_advanced_tabular_neural_review import available_feature_columns
from src.experiments.e0_away_ah_advanced_tabular_neural_review import fit_preprocessor
from src.experiments.e0_away_ah_advanced_tabular_neural_review import make_temporal_splits
from src.experiments.e0_away_ah_advanced_tabular_neural_review import run_model_nested


def synthetic_rows():
    rows = []
    for season in [2021, 2022, 2023, 2024]:
        for index in range(16):
            rows.append(
                {
                    "Date": pd.Timestamp(f"{season - 1}-09-01") + pd.Timedelta(days=index),
                    "HomeTeam": ["A", "B", "C", "FutureHome"][index % 4],
                    "AwayTeam": ["X", "Y", "Z", "FutureAway"][index % 4],
                    "season_end_year": season,
                    "ah_line": -1.0 - 0.25 * (index % 5),
                    "home_ah_odds": 1.85 + 0.01 * (index % 4),
                    "away_ah_odds": 1.86 + 0.02 * (index % 5),
                    "home_market_probability": 0.52,
                    "away_market_probability": 0.48,
                    "overround": 1.04,
                    "profit": 0.9 if (index + season) % 3 == 0 else -1.0,
                    TARGET_COLUMN: 1 if (index + season) % 3 == 0 else 0,
                    "clv_probability_pp": 0.1 if index % 2 == 0 else -0.1,
                }
            )
    return pd.DataFrame(rows)


def test_temporal_splits_do_not_use_future_seasons():
    splits = make_temporal_splits([2021, 2022, 2023, 2024])

    assert [(split.train_years, split.validation_year, split.test_year) for split in splits] == [
        ((2021,), 2022, 2023),
        ((2021, 2022), 2023, 2024),
    ]
    assert all(max(split.train_years) < split.validation_year < split.test_year for split in splits)


def test_preprocessor_fits_categories_only_on_training_past():
    data = synthetic_rows()
    train = data[data["season_end_year"].eq(2021)].copy()
    train = train[~train["HomeTeam"].eq("FutureHome")].copy()
    train = train[~train["AwayTeam"].eq("FutureAway")].copy()

    preprocessor, numeric, categorical = fit_preprocessor(train)
    encoder = preprocessor.named_transformers_["categorical"]
    home_categories = set(encoder.categories_[categorical.index("HomeTeam")])
    away_categories = set(encoder.categories_[categorical.index("AwayTeam")])

    assert "FutureHome" not in home_categories
    assert "FutureAway" not in away_categories
    assert TARGET_COLUMN not in numeric + categorical


def test_closing_odds_absent_from_feature_matrix():
    data = synthetic_rows()
    data["AvgCAHA"] = 1.90
    data["AHCh"] = -1.25

    numeric, categorical = available_feature_columns(data)

    assert all("AvgC" not in column and "AHCh" not in column for column in numeric + categorical)
    assert set(categorical) == set(CATEGORICAL_FEATURE_COLUMNS)


def test_model_nested_output_is_deterministic_with_fixed_seed():
    data = synthetic_rows()

    first_by_year, first_bets, first_metrics = run_model_nested(data, "logistic_test", "logistic", "binary_cover", 7)
    second_by_year, second_bets, second_metrics = run_model_nested(data, "logistic_test", "logistic", "binary_cover", 7)

    pd.testing.assert_frame_equal(first_by_year.reset_index(drop=True), second_by_year.reset_index(drop=True))
    pd.testing.assert_frame_equal(first_bets.reset_index(drop=True), second_bets.reset_index(drop=True))
    pd.testing.assert_frame_equal(first_metrics.reset_index(drop=True), second_metrics.reset_index(drop=True))


def test_neural_training_handles_tiny_validation_seasons_safely():
    data = synthetic_rows().groupby("season_end_year").head(3).reset_index(drop=True)

    by_year, bets, metrics = run_model_nested(data, "neural_tiny", "neural_mlp", "market_residual", 11)

    assert set(by_year["test_year"]) == {2023, 2024}
    assert "selected_filter" in by_year.columns
    assert isinstance(bets, pd.DataFrame)
    assert set(metrics["test_year"]) == {2023, 2024}
