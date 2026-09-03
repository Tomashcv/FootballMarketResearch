import math

import pandas as pd

from src.features.contextual_features import add_market_disagreement_features
from src.features.internal_elo_features import add_internal_elo_features
from src.features.internal_elo_features import add_internal_elo_market_disagreement_features
from src.features.internal_elo_features import expected_score


def test_first_match_uses_default_elo():
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "FTHG": [1],
            "FTAG": [0],
        }
    )
    output = add_internal_elo_features(matches)
    assert output.loc[0, "home_internal_elo_pre"] == 1500
    assert output.loc[0, "away_internal_elo_pre"] == 1500
    assert output.loc[0, "internal_elo_home_win_prob"] == expected_score(1560, 1500)


def test_ratings_update_only_after_match():
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "HomeTeam": ["A", "A"],
            "AwayTeam": ["B", "B"],
            "FTHG": [1, 0],
            "FTAG": [0, 0],
        }
    )
    output = add_internal_elo_features(matches, home_advantage_elo=0)
    assert output.loc[0, "home_internal_elo_pre"] == 1500
    assert output.loc[0, "away_internal_elo_pre"] == 1500
    assert output.loc[1, "home_internal_elo_pre"] == 1510
    assert output.loc[1, "away_internal_elo_pre"] == 1490


def test_no_future_leakage_with_input_out_of_order():
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-01"],
            "HomeTeam": ["A", "A"],
            "AwayTeam": ["B", "B"],
            "FTHG": [0, 1],
            "FTAG": [0, 0],
        }
    )
    output = add_internal_elo_features(matches, home_advantage_elo=0)
    later_input_row = output.loc[0]
    earlier_input_row = output.loc[1]
    assert earlier_input_row["home_internal_elo_pre"] == 1500
    assert later_input_row["home_internal_elo_pre"] == 1510


def test_chronological_ordering_is_deterministic_with_same_date_time():
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "Time": ["15:00", "15:00", "15:00"],
            "HomeTeam": ["B", "A", "A"],
            "AwayTeam": ["C", "B", "B"],
            "FTHG": [1, 1, 0],
            "FTAG": [0, 0, 0],
        }
    )
    first = add_internal_elo_features(matches, home_advantage_elo=0)
    second = add_internal_elo_features(matches.sample(frac=1, random_state=7).reset_index(drop=True), home_advantage_elo=0)

    first_keyed = first.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    second_keyed = second.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    pd.testing.assert_series_equal(
        first_keyed["home_internal_elo_pre"],
        second_keyed["home_internal_elo_pre"],
        check_names=False,
    )


def test_missing_new_team_starts_at_default_elo():
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "HomeTeam": ["A", "Promoted"],
            "AwayTeam": ["B", "A"],
            "FTHG": [1, 0],
            "FTAG": [0, 1],
        }
    )
    output = add_internal_elo_features(matches)
    assert output.loc[1, "home_internal_elo_pre"] == 1500


def test_market_disagreement_uses_main_odds_and_internal_elo():
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "FTHG": [1],
            "FTAG": [0],
            "AvgH": [2.0],
            "AvgD": [3.0],
            "AvgA": [4.0],
        }
    )
    with_market = add_market_disagreement_features(matches)
    output = add_internal_elo_market_disagreement_features(with_market)
    expected_home_market = output.loc[0, "avg_1x2_AvgH_no_vig_probability"]
    expected_elo = output.loc[0, "internal_elo_home_win_prob"]
    assert math.isclose(output.loc[0, "market_home_prob_minus_internal_elo_prob"], expected_home_market - expected_elo)
    assert "market_away_prob_minus_internal_elo_prob" in output.columns
