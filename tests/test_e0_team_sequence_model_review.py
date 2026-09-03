import pandas as pd

from src.experiments import e0_away_ah_team_sequence_model_review as seq
from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced


def sequence_sample():
    rows = []
    teams = [("A", "B"), ("A", "C"), ("B", "A"), ("C", "A"), ("A", "D"), ("D", "A")]
    for i, (home, away) in enumerate(teams):
        rows.append(
            {
                "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": 99 if i == 4 else i % 3,
                "FTAG": 88 if i == 4 else (i + 1) % 3,
                "season_end_year": 2024 if i < 4 else 2025,
                "ah_line": -1.0 - 0.25 * (i % 3),
                "home_ah_odds": 1.9,
                "away_ah_odds": 1.95,
                "home_market_probability": 0.52,
                "away_market_probability": 0.48,
                "overround": 1.04,
                "home_rest_days": 7 + i,
                "away_rest_days": 6 + i,
                "home_internal_elo_pre": 1500 + i,
                "away_internal_elo_pre": 1490 - i,
                "travel_distance_km": 100 + i,
                "weather_temperature_c": 10 + i,
                "weather_precipitation_mm": 0.1 * i,
                "weather_wind_speed_kph": 15 + i,
                "profit": 0.9 if i % 2 == 0 else -1.0,
                advanced.TARGET_COLUMN: 1 if i % 2 == 0 else 0,
                "clv_probability_pp": 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_sequences_use_only_matches_before_current_date():
    data = sequence_sample()
    home, _ = seq.build_sequence_arrays(data, 5)
    current = data.iloc[4]
    home_sequence = home[4]

    assert current["HomeTeam"] == "A"
    assert home_sequence[-1, seq.SEQUENCE_FEATURE_COLUMNS.index("goals_for")] == data.loc[3, "FTAG"]
    assert home_sequence[-2, seq.SEQUENCE_FEATURE_COLUMNS.index("goals_for")] == data.loc[2, "FTAG"]


def test_current_match_not_included_in_own_sequence():
    data = sequence_sample()
    home, away = seq.build_sequence_arrays(data, 5)

    current_home_goals = data.loc[4, "FTHG"]
    current_away_goals = data.loc[4, "FTAG"]

    assert current_home_goals not in home[4, :, seq.SEQUENCE_FEATURE_COLUMNS.index("goals_for")]
    assert current_away_goals not in away[4, :, seq.SEQUENCE_FEATURE_COLUMNS.index("goals_for")]


def test_same_day_matches_are_not_history():
    data = sequence_sample()
    data.loc[1, "Date"] = data.loc[0, "Date"]
    home, _ = seq.build_sequence_arrays(data, 5)

    assert home[1].sum() == 0.0


def test_sequence_scaler_fitted_only_on_train_rows():
    data = sequence_sample()
    home, away = seq.build_sequence_arrays(data, 3)
    train_index = data[data["season_end_year"].eq(2024)].index.to_numpy()
    scaler = seq.fit_sequence_scaler(home[train_index], away[train_index])

    assert scaler.mean_[seq.SEQUENCE_FEATURE_COLUMNS.index("weather_temperature_c")] < 20


def test_closing_odds_absent_from_current_features():
    data = sequence_sample()
    data["AvgCAHA"] = 1.91

    numeric, categorical = advanced.available_feature_columns(data)

    assert "AvgCAHA" not in numeric + categorical


def test_sequence_model_deterministic_with_fixed_seed():
    data = sequence_sample()
    by_year_1, bets_1, metrics_1 = seq.run_sequence_nested(data, "gru", "binary_cover", 5, 11)
    by_year_2, bets_2, metrics_2 = seq.run_sequence_nested(data, "gru", "binary_cover", 5, 11)

    pd.testing.assert_frame_equal(by_year_1.reset_index(drop=True), by_year_2.reset_index(drop=True))
    pd.testing.assert_frame_equal(bets_1.reset_index(drop=True), bets_2.reset_index(drop=True))
    pd.testing.assert_frame_equal(metrics_1.reset_index(drop=True), metrics_2.reset_index(drop=True))
