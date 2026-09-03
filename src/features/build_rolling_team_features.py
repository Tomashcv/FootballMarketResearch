import argparse
from collections import defaultdict

import pandas as pd

from src.common.paths import get_market_dataset_path
from src.common.paths import get_market_features_path


DEFAULT_AVERAGE_GOALS = 2.70
DEFAULT_TEAM_GOALS = 1.35
DEFAULT_OVER_25_RATE = 0.50
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


def build_basic_history_features(team_history, window, prefix, defaults):
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

    features[f"{prefix}_over_25_rate_last_{window}"] = average_last_matches(
        team_history,
        "over_25",
        window,
        defaults["over_25_rate"]
    )

    return features


def add_difference_features(features, left_prefix, right_prefix, output_prefix, window):
    features[f"{output_prefix}_goals_for_last_{window}_diff"] = (
        features[f"{left_prefix}_goals_for_last_{window}"]
        - features[f"{right_prefix}_goals_for_last_{window}"]
    )

    features[f"{output_prefix}_goals_against_last_{window}_diff"] = (
        features[f"{left_prefix}_goals_against_last_{window}"]
        - features[f"{right_prefix}_goals_against_last_{window}"]
    )

    features[f"{output_prefix}_total_goals_last_{window}_diff"] = (
        features[f"{left_prefix}_total_goals_last_{window}"]
        - features[f"{right_prefix}_total_goals_last_{window}"]
    )

    features[f"{output_prefix}_over_25_rate_last_{window}_diff"] = (
        features[f"{left_prefix}_over_25_rate_last_{window}"]
        - features[f"{right_prefix}_over_25_rate_last_{window}"]
    )


def add_matchup_expected_goal_features(features, window, league_avg_goals):
    expected_home_goals = (
        features[f"home_home_goals_for_last_{window}"]
        + features[f"away_away_goals_against_last_{window}"]
    ) / 2.0

    expected_away_goals = (
        features[f"away_away_goals_for_last_{window}"]
        + features[f"home_home_goals_against_last_{window}"]
    ) / 2.0

    expected_total_goals = expected_home_goals + expected_away_goals

    features[f"expected_home_goals_simple_{window}"] = expected_home_goals
    features[f"expected_away_goals_simple_{window}"] = expected_away_goals
    features[f"expected_total_goals_simple_{window}"] = expected_total_goals

    features[f"expected_total_goals_simple_{window}_minus_2_5"] = (
        expected_total_goals - 2.5
    )

    features[f"expected_total_goals_simple_{window}_minus_league_avg"] = (
        expected_total_goals - league_avg_goals
    )

    features[f"home_attack_vs_away_defense_{window}"] = (
        features[f"home_home_goals_for_last_{window}"]
        - features[f"away_away_goals_against_last_{window}"]
    )

    features[f"away_attack_vs_home_defense_{window}"] = (
        features[f"away_away_goals_for_last_{window}"]
        - features[f"home_home_goals_against_last_{window}"]
    )

    features[f"combined_home_away_over_rate_{window}"] = (
        features[f"home_home_over_25_rate_last_{window}"]
        + features[f"away_away_over_25_rate_last_{window}"]
    ) / 2.0

    features[f"combined_home_away_total_goals_{window}"] = (
        features[f"home_home_total_goals_last_{window}"]
        + features[f"away_away_total_goals_last_{window}"]
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
        home_overall_features = build_basic_history_features(
            overall_histories[home_team],
            window,
            "home_overall",
            defaults
        )

        away_overall_features = build_basic_history_features(
            overall_histories[away_team],
            window,
            "away_overall",
            defaults
        )

        home_home_features = build_basic_history_features(
            home_histories[home_team],
            window,
            "home_home",
            defaults
        )

        away_away_features = build_basic_history_features(
            away_histories[away_team],
            window,
            "away_away",
            defaults
        )

        features.update(home_overall_features)
        features.update(away_overall_features)
        features.update(home_home_features)
        features.update(away_away_features)

        add_difference_features(
            features,
            "home_overall",
            "away_overall",
            "overall",
            window
        )

        add_difference_features(
            features,
            "home_home",
            "away_away",
            "home_away_context",
            window
        )

        add_matchup_expected_goal_features(
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

    features["market_over_probability_minus_league_over_rate"] = (
        float(row["market_probability_over"]) - defaults["over_25_rate"]
    )

    features["market_over_probability_minus_0_5"] = (
        float(row["market_probability_over"]) - 0.5
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
        total_goals = float(row["total_goals"])
        over_25 = int(row["target_over_25"])

        home_record = {
            "goals_for": home_goals,
            "goals_against": away_goals,
            "total_goals": total_goals,
            "over_25": over_25,
        }

        away_record = {
            "goals_for": away_goals,
            "goals_against": home_goals,
            "total_goals": total_goals,
            "over_25": over_25,
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
    parser.add_argument("--market", required=True)
    args = parser.parse_args()

    league_code = args.league.upper()
    market_name = args.market

    input_path = get_market_dataset_path(league_code, market_name)
    output_path = get_market_features_path(league_code, market_name)

    dataframe = pd.read_csv(input_path, low_memory=False)

    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    dataframe = dataframe.dropna(subset=["Date"]).copy()

    dataframe = dataframe.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    dataframe["match_id"] = dataframe.index

    overall_histories = defaultdict(list)
    home_histories = defaultdict(list)
    away_histories = defaultdict(list)
    last_match_dates = {}

    total_goals_so_far = 0.0
    total_matches_so_far = 0
    total_overs_so_far = 0

    feature_rows = []

    grouped_by_date = dataframe.groupby("Date", sort=True)

    for current_date, date_group in grouped_by_date:
        if total_matches_so_far > 0:
            league_avg_goals_so_far = total_goals_so_far / total_matches_so_far
            league_over_25_rate_so_far = total_overs_so_far / total_matches_so_far
            default_team_goals = league_avg_goals_so_far / 2.0
        else:
            league_avg_goals_so_far = DEFAULT_AVERAGE_GOALS
            league_over_25_rate_so_far = DEFAULT_OVER_25_RATE
            default_team_goals = DEFAULT_TEAM_GOALS

        defaults = {
            "average_goals": league_avg_goals_so_far,
            "team_goals": default_team_goals,
            "over_25_rate": league_over_25_rate_so_far,
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
            features["league_over_25_rate_so_far"] = league_over_25_rate_so_far
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
            total_goals_so_far += float(row["total_goals"])
            total_matches_so_far += 1
            total_overs_so_far += int(row["target_over_25"])

    features_dataframe = pd.DataFrame(feature_rows)

    final_dataframe = dataframe.merge(features_dataframe, on="match_id", how="left")
    final_dataframe = final_dataframe.drop(columns=["match_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_dataframe.to_csv(output_path, index=False)

    feature_columns = [column for column in final_dataframe.columns if column not in dataframe.columns]

    print("")
    print("Rolling features V2 criadas sem leakage:")
    print(output_path)
    print("Jogos:", len(final_dataframe))
    print("Features novas:", len(feature_columns))
    print("")
    print("Exemplos de features:")
    for column in feature_columns[:40]:
        print("-", column)


if __name__ == "__main__":
    main()
