import argparse
from collections import defaultdict

import pandas as pd

from src.common.paths import get_market_dataset_path
from src.common.paths import get_market_features_path


DEFAULT_AVERAGE_GOALS = 2.70
DEFAULT_TEAM_GOALS = 1.35
DEFAULT_DRAW_RATE = 0.25
DEFAULT_UNDER_25_RATE = 0.50
DEFAULT_LOW_GOALS_RATE = 0.35
DEFAULT_BOTH_SCORE_RATE = 0.50
DEFAULT_DAYS_SINCE_LAST_MATCH = 14


def average_last_matches(team_history, key, window, default_value):
    if len(team_history) == 0:
        return default_value

    recent_matches = team_history[-window:]

    if len(recent_matches) == 0:
        return default_value

    values = []

    for match in recent_matches:
        values.append(float(match[key]))

    return sum(values) / len(values)


def build_history_features(team_history, window, prefix, defaults):
    features = {}

    features[f"{prefix}_goals_for_last_{window}"] = average_last_matches(
        team_history,
        "goals_for",
        window,
        defaults["team_goals"]
    )

    features[f"{prefix}_goals_against_last_{window}"] = average_last_matches(
        team_history,
        "goals_against",
        window,
        defaults["team_goals"]
    )

    features[f"{prefix}_total_goals_last_{window}"] = average_last_matches(
        team_history,
        "total_goals",
        window,
        defaults["average_goals"]
    )

    features[f"{prefix}_draw_rate_last_{window}"] = average_last_matches(
        team_history,
        "draw",
        window,
        defaults["draw_rate"]
    )

    features[f"{prefix}_under_25_rate_last_{window}"] = average_last_matches(
        team_history,
        "under_25",
        window,
        defaults["under_25_rate"]
    )

    features[f"{prefix}_low_goals_rate_last_{window}"] = average_last_matches(
        team_history,
        "low_goals",
        window,
        defaults["low_goals_rate"]
    )

    features[f"{prefix}_both_score_rate_last_{window}"] = average_last_matches(
        team_history,
        "both_score",
        window,
        defaults["both_score_rate"]
    )

    features[f"{prefix}_goal_difference_abs_last_{window}"] = average_last_matches(
        team_history,
        "goal_difference_abs",
        window,
        1.0
    )

    features[f"{prefix}_goal_difference_signed_last_{window}"] = average_last_matches(
        team_history,
        "goal_difference_signed",
        window,
        0.0
    )

    return features


def add_diff_and_balance_features(features, left_prefix, right_prefix, output_prefix, window):
    base_names = [
        "goals_for",
        "goals_against",
        "total_goals",
        "draw_rate",
        "under_25_rate",
        "low_goals_rate",
        "both_score_rate",
        "goal_difference_abs",
        "goal_difference_signed",
    ]

    for base_name in base_names:
        left_value = features[f"{left_prefix}_{base_name}_last_{window}"]
        right_value = features[f"{right_prefix}_{base_name}_last_{window}"]

        features[f"{output_prefix}_{base_name}_last_{window}_diff"] = left_value - right_value
        features[f"{output_prefix}_{base_name}_last_{window}_abs_diff"] = abs(left_value - right_value)
        features[f"{output_prefix}_{base_name}_last_{window}_average"] = (left_value + right_value) / 2.0


def add_expected_match_features(features, window, league_average_goals):
    expected_home_goals = (
        features[f"home_home_goals_for_last_{window}"]
        + features[f"away_away_goals_against_last_{window}"]
    ) / 2.0

    expected_away_goals = (
        features[f"away_away_goals_for_last_{window}"]
        + features[f"home_home_goals_against_last_{window}"]
    ) / 2.0

    expected_total_goals = expected_home_goals + expected_away_goals
    expected_goal_gap = abs(expected_home_goals - expected_away_goals)

    features[f"expected_home_goals_simple_{window}"] = expected_home_goals
    features[f"expected_away_goals_simple_{window}"] = expected_away_goals
    features[f"expected_total_goals_simple_{window}"] = expected_total_goals
    features[f"expected_goal_gap_simple_{window}"] = expected_goal_gap

    features[f"expected_total_goals_simple_{window}_minus_league_avg"] = (
        expected_total_goals - league_average_goals
    )

    features[f"expected_total_goals_simple_{window}_minus_2_5"] = (
        expected_total_goals - 2.5
    )

    features[f"combined_context_draw_rate_{window}"] = (
        features[f"home_home_draw_rate_last_{window}"]
        + features[f"away_away_draw_rate_last_{window}"]
    ) / 2.0

    features[f"combined_context_under_25_rate_{window}"] = (
        features[f"home_home_under_25_rate_last_{window}"]
        + features[f"away_away_under_25_rate_last_{window}"]
    ) / 2.0

    features[f"combined_context_low_goals_rate_{window}"] = (
        features[f"home_home_low_goals_rate_last_{window}"]
        + features[f"away_away_low_goals_rate_last_{window}"]
    ) / 2.0

    features[f"combined_context_goal_gap_abs_{window}"] = (
        features[f"home_home_goal_difference_abs_last_{window}"]
        + features[f"away_away_goal_difference_abs_last_{window}"]
    ) / 2.0


def build_features_for_match(
    row,
    overall_histories,
    home_histories,
    away_histories,
    last_match_dates,
    current_date,
    defaults
):
    home_team = row["HomeTeam"]
    away_team = row["AwayTeam"]

    features = {}

    for window in [5, 10]:
        home_overall_features = build_history_features(
            overall_histories[home_team],
            window,
            "home_overall",
            defaults
        )

        away_overall_features = build_history_features(
            overall_histories[away_team],
            window,
            "away_overall",
            defaults
        )

        home_home_features = build_history_features(
            home_histories[home_team],
            window,
            "home_home",
            defaults
        )

        away_away_features = build_history_features(
            away_histories[away_team],
            window,
            "away_away",
            defaults
        )

        features.update(home_overall_features)
        features.update(away_overall_features)
        features.update(home_home_features)
        features.update(away_away_features)

        add_diff_and_balance_features(
            features,
            "home_overall",
            "away_overall",
            "overall",
            window
        )

        add_diff_and_balance_features(
            features,
            "home_home",
            "away_away",
            "home_away_context",
            window
        )

        add_expected_match_features(
            features,
            window,
            defaults["average_goals"]
        )

    features["home_matches_played_before"] = len(overall_histories[home_team])
    features["away_matches_played_before"] = len(overall_histories[away_team])
    features["home_home_matches_played_before"] = len(home_histories[home_team])
    features["away_away_matches_played_before"] = len(away_histories[away_team])

    if home_team in last_match_dates:
        home_days_since = (current_date - last_match_dates[home_team]).days
    else:
        home_days_since = DEFAULT_DAYS_SINCE_LAST_MATCH

    if away_team in last_match_dates:
        away_days_since = (current_date - last_match_dates[away_team]).days
    else:
        away_days_since = DEFAULT_DAYS_SINCE_LAST_MATCH

    features["home_days_since_last_match"] = home_days_since
    features["away_days_since_last_match"] = away_days_since
    features["days_since_last_match_diff"] = home_days_since - away_days_since

    market_home_probability = float(row["market_probability_home"])
    market_draw_probability = float(row["market_probability_draw"])
    market_away_probability = float(row["market_probability_away"])
    market_not_draw_probability = float(row["market_probability_not_draw"])

    home_away_gap = abs(market_home_probability - market_away_probability)
    favorite_probability = max(market_home_probability, market_away_probability)
    underdog_probability = min(market_home_probability, market_away_probability)

    features["market_home_probability"] = market_home_probability
    features["market_draw_probability"] = market_draw_probability
    features["market_away_probability"] = market_away_probability
    features["market_not_draw_probability"] = market_not_draw_probability

    features["market_home_minus_away_probability"] = (
        market_home_probability - market_away_probability
    )

    features["market_home_away_probability_gap"] = home_away_gap
    features["market_favorite_probability"] = favorite_probability
    features["market_underdog_probability"] = underdog_probability

    features["market_draw_minus_league_draw_rate"] = (
        market_draw_probability - defaults["draw_rate"]
    )

    features["market_draw_minus_0_25"] = (
        market_draw_probability - 0.25
    )

    features["market_draw_to_favorite_ratio"] = (
        market_draw_probability / max(favorite_probability, 0.000001)
    )

    features["market_draw_to_home_away_gap_ratio"] = (
        market_draw_probability / max(home_away_gap, 0.01)
    )

    return features


def update_histories_after_date(
    date_group,
    overall_histories,
    home_histories,
    away_histories,
    last_match_dates,
    current_date
):
    for _, row in date_group.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_goals = float(row["FTHG"])
        away_goals = float(row["FTAG"])
        total_goals = home_goals + away_goals

        is_draw = 0
        if str(row["FTR"]) == "D":
            is_draw = 1

        under_25 = 0
        if total_goals < 2.5:
            under_25 = 1

        low_goals = 0
        if total_goals <= 2.0:
            low_goals = 1

        both_score = 0
        if home_goals > 0 and away_goals > 0:
            both_score = 1

        goal_difference_abs = abs(home_goals - away_goals)

        home_record = {
            "goals_for": home_goals,
            "goals_against": away_goals,
            "total_goals": total_goals,
            "draw": is_draw,
            "under_25": under_25,
            "low_goals": low_goals,
            "both_score": both_score,
            "goal_difference_abs": goal_difference_abs,
            "goal_difference_signed": home_goals - away_goals,
        }

        away_record = {
            "goals_for": away_goals,
            "goals_against": home_goals,
            "total_goals": total_goals,
            "draw": is_draw,
            "under_25": under_25,
            "low_goals": low_goals,
            "both_score": both_score,
            "goal_difference_abs": goal_difference_abs,
            "goal_difference_signed": away_goals - home_goals,
        }

        overall_histories[home_team].append(home_record)
        overall_histories[away_team].append(away_record)

        home_histories[home_team].append(home_record)
        away_histories[away_team].append(away_record)

        last_match_dates[home_team] = current_date
        last_match_dates[away_team] = current_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    args = parser.parse_args()

    league_code = args.league.upper()
    market_name = "draw_vs_not_draw"

    input_path = get_market_dataset_path(league_code, market_name)
    output_path = get_market_features_path(league_code, market_name)

    dataframe = pd.read_csv(input_path, low_memory=False)

    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    dataframe = dataframe.dropna(subset=["Date"]).copy()

    dataframe["total_goals"] = dataframe["FTHG"].astype(float) + dataframe["FTAG"].astype(float)

    dataframe = dataframe.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    dataframe["match_id"] = dataframe.index

    overall_histories = defaultdict(list)
    home_histories = defaultdict(list)
    away_histories = defaultdict(list)
    last_match_dates = {}

    total_goals_so_far = 0.0
    total_matches_so_far = 0
    total_draws_so_far = 0
    total_under_25_so_far = 0
    total_low_goals_so_far = 0
    total_both_score_so_far = 0

    feature_rows = []

    grouped_by_date = dataframe.groupby("Date", sort=True)

    for current_date, date_group in grouped_by_date:
        if total_matches_so_far > 0:
            league_avg_goals_so_far = total_goals_so_far / total_matches_so_far
            league_draw_rate_so_far = total_draws_so_far / total_matches_so_far
            league_under_25_rate_so_far = total_under_25_so_far / total_matches_so_far
            league_low_goals_rate_so_far = total_low_goals_so_far / total_matches_so_far
            league_both_score_rate_so_far = total_both_score_so_far / total_matches_so_far
            default_team_goals = league_avg_goals_so_far / 2.0
        else:
            league_avg_goals_so_far = DEFAULT_AVERAGE_GOALS
            league_draw_rate_so_far = DEFAULT_DRAW_RATE
            league_under_25_rate_so_far = DEFAULT_UNDER_25_RATE
            league_low_goals_rate_so_far = DEFAULT_LOW_GOALS_RATE
            league_both_score_rate_so_far = DEFAULT_BOTH_SCORE_RATE
            default_team_goals = DEFAULT_TEAM_GOALS

        defaults = {
            "average_goals": league_avg_goals_so_far,
            "team_goals": default_team_goals,
            "draw_rate": league_draw_rate_so_far,
            "under_25_rate": league_under_25_rate_so_far,
            "low_goals_rate": league_low_goals_rate_so_far,
            "both_score_rate": league_both_score_rate_so_far,
        }

        for _, row in date_group.iterrows():
            features = build_features_for_match(
                row,
                overall_histories,
                home_histories,
                away_histories,
                last_match_dates,
                current_date,
                defaults
            )

            features["match_id"] = int(row["match_id"])
            features["league_avg_goals_so_far"] = league_avg_goals_so_far
            features["league_draw_rate_so_far"] = league_draw_rate_so_far
            features["league_under_25_rate_so_far"] = league_under_25_rate_so_far
            features["league_low_goals_rate_so_far"] = league_low_goals_rate_so_far
            features["league_both_score_rate_so_far"] = league_both_score_rate_so_far
            features["matches_played_in_league_so_far"] = total_matches_so_far

            feature_rows.append(features)

        update_histories_after_date(
            date_group,
            overall_histories,
            home_histories,
            away_histories,
            last_match_dates,
            current_date
        )

        for _, row in date_group.iterrows():
            total_goals = float(row["total_goals"])

            total_goals_so_far += total_goals
            total_matches_so_far += 1
            total_draws_so_far += int(row["target_draw"])

            if total_goals < 2.5:
                total_under_25_so_far += 1

            if total_goals <= 2.0:
                total_low_goals_so_far += 1

            if float(row["FTHG"]) > 0 and float(row["FTAG"]) > 0:
                total_both_score_so_far += 1

    features_dataframe = pd.DataFrame(feature_rows)

    final_dataframe = dataframe.merge(features_dataframe, on="match_id", how="left")
    final_dataframe = final_dataframe.drop(columns=["match_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_dataframe.to_csv(output_path, index=False)

    feature_columns = [column for column in final_dataframe.columns if column not in dataframe.columns]

    print("")
    print("Draw features criadas sem leakage:")
    print(output_path)
    print("Jogos:", len(final_dataframe))
    print("Features novas:", len(feature_columns))
    print("")
    print("Exemplos de features:")
    for column in feature_columns[:50]:
        print("-", column)


if __name__ == "__main__":
    main()
