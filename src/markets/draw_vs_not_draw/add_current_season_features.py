import argparse
from collections import defaultdict

import pandas as pd

from src.common.paths import get_market_features_path


DEFAULT_POINTS_PER_GAME = 1.0
DEFAULT_GOAL_DIFF_PER_GAME = 0.0
DEFAULT_TEAM_GOALS_PER_GAME = 1.35
DEFAULT_DRAW_RATE = 0.25


def empty_team_state():
    return {
        "matches": 0,
        "points": 0.0,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "goal_diff": 0.0,
        "draws": 0.0,
    }


def per_game(state, key, default_value):
    matches = int(state["matches"])

    if matches == 0:
        return default_value

    return float(state[key]) / matches


def build_state_features(home_state, away_state, home_home_state, away_away_state):
    features = {}

    home_matches = int(home_state["matches"])
    away_matches = int(away_state["matches"])

    features["home_current_season_matches_before"] = home_matches
    features["away_current_season_matches_before"] = away_matches
    features["min_team_matches_this_season_before"] = min(home_matches, away_matches)

    features["home_current_season_points_per_game_before"] = per_game(
        home_state,
        "points",
        DEFAULT_POINTS_PER_GAME
    )

    features["away_current_season_points_per_game_before"] = per_game(
        away_state,
        "points",
        DEFAULT_POINTS_PER_GAME
    )

    features["home_current_season_goal_diff_per_game_before"] = per_game(
        home_state,
        "goal_diff",
        DEFAULT_GOAL_DIFF_PER_GAME
    )

    features["away_current_season_goal_diff_per_game_before"] = per_game(
        away_state,
        "goal_diff",
        DEFAULT_GOAL_DIFF_PER_GAME
    )

    features["home_current_season_goals_for_per_game_before"] = per_game(
        home_state,
        "goals_for",
        DEFAULT_TEAM_GOALS_PER_GAME
    )

    features["away_current_season_goals_for_per_game_before"] = per_game(
        away_state,
        "goals_for",
        DEFAULT_TEAM_GOALS_PER_GAME
    )

    features["home_current_season_goals_against_per_game_before"] = per_game(
        home_state,
        "goals_against",
        DEFAULT_TEAM_GOALS_PER_GAME
    )

    features["away_current_season_goals_against_per_game_before"] = per_game(
        away_state,
        "goals_against",
        DEFAULT_TEAM_GOALS_PER_GAME
    )

    features["home_current_season_draw_rate_before"] = per_game(
        home_state,
        "draws",
        DEFAULT_DRAW_RATE
    )

    features["away_current_season_draw_rate_before"] = per_game(
        away_state,
        "draws",
        DEFAULT_DRAW_RATE
    )

    features["home_home_current_season_matches_before"] = int(home_home_state["matches"])
    features["away_away_current_season_matches_before"] = int(away_away_state["matches"])

    features["home_home_current_season_points_per_game_before"] = per_game(
        home_home_state,
        "points",
        DEFAULT_POINTS_PER_GAME
    )

    features["away_away_current_season_points_per_game_before"] = per_game(
        away_away_state,
        "points",
        DEFAULT_POINTS_PER_GAME
    )

    features["home_home_current_season_goal_diff_per_game_before"] = per_game(
        home_home_state,
        "goal_diff",
        DEFAULT_GOAL_DIFF_PER_GAME
    )

    features["away_away_current_season_goal_diff_per_game_before"] = per_game(
        away_away_state,
        "goal_diff",
        DEFAULT_GOAL_DIFF_PER_GAME
    )

    features["home_home_current_season_draw_rate_before"] = per_game(
        home_home_state,
        "draws",
        DEFAULT_DRAW_RATE
    )

    features["away_away_current_season_draw_rate_before"] = per_game(
        away_away_state,
        "draws",
        DEFAULT_DRAW_RATE
    )

    features["current_season_points_per_game_diff"] = (
        features["home_current_season_points_per_game_before"]
        - features["away_current_season_points_per_game_before"]
    )

    features["current_season_points_per_game_abs_diff"] = abs(
        features["current_season_points_per_game_diff"]
    )

    features["current_season_goal_diff_per_game_diff"] = (
        features["home_current_season_goal_diff_per_game_before"]
        - features["away_current_season_goal_diff_per_game_before"]
    )

    features["current_season_goal_diff_per_game_abs_diff"] = abs(
        features["current_season_goal_diff_per_game_diff"]
    )

    features["current_season_goals_for_per_game_abs_diff"] = abs(
        features["home_current_season_goals_for_per_game_before"]
        - features["away_current_season_goals_for_per_game_before"]
    )

    features["current_season_goals_against_per_game_abs_diff"] = abs(
        features["home_current_season_goals_against_per_game_before"]
        - features["away_current_season_goals_against_per_game_before"]
    )

    features["current_season_draw_rate_average"] = (
        features["home_current_season_draw_rate_before"]
        + features["away_current_season_draw_rate_before"]
    ) / 2.0

    features["current_season_draw_rate_abs_diff"] = abs(
        features["home_current_season_draw_rate_before"]
        - features["away_current_season_draw_rate_before"]
    )

    features["home_away_context_current_season_points_per_game_abs_diff"] = abs(
        features["home_home_current_season_points_per_game_before"]
        - features["away_away_current_season_points_per_game_before"]
    )

    features["home_away_context_current_season_goal_diff_per_game_abs_diff"] = abs(
        features["home_home_current_season_goal_diff_per_game_before"]
        - features["away_away_current_season_goal_diff_per_game_before"]
    )

    features["home_away_context_current_season_draw_rate_average"] = (
        features["home_home_current_season_draw_rate_before"]
        + features["away_away_current_season_draw_rate_before"]
    ) / 2.0

    features["home_away_context_current_season_draw_rate_abs_diff"] = abs(
        features["home_home_current_season_draw_rate_before"]
        - features["away_away_current_season_draw_rate_before"]
    )

    return features


def update_state(state, goals_for, goals_against, points, is_draw):
    state["matches"] += 1
    state["points"] += float(points)
    state["goals_for"] += float(goals_for)
    state["goals_against"] += float(goals_against)
    state["goal_diff"] += float(goals_for) - float(goals_against)

    if is_draw:
        state["draws"] += 1.0


def update_states_after_date(date_group, overall_states, home_states, away_states):
    for _, row in date_group.iterrows():
        season = int(row["season_end_year"])
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_goals = float(row["FTHG"])
        away_goals = float(row["FTAG"])

        is_draw = home_goals == away_goals

        if home_goals > away_goals:
            home_points = 3.0
            away_points = 0.0
        elif home_goals < away_goals:
            home_points = 0.0
            away_points = 3.0
        else:
            home_points = 1.0
            away_points = 1.0

        home_key = (season, home_team)
        away_key = (season, away_team)

        update_state(
            overall_states[home_key],
            home_goals,
            away_goals,
            home_points,
            is_draw
        )

        update_state(
            overall_states[away_key],
            away_goals,
            home_goals,
            away_points,
            is_draw
        )

        update_state(
            home_states[home_key],
            home_goals,
            away_goals,
            home_points,
            is_draw
        )

        update_state(
            away_states[away_key],
            away_goals,
            home_goals,
            away_points,
            is_draw
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    args = parser.parse_args()

    league_code = args.league.upper()
    market_name = "draw_vs_not_draw"

    input_path = get_market_features_path(league_code, market_name)

    dataframe = pd.read_csv(input_path, low_memory=False)

    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    dataframe = dataframe.dropna(subset=["Date"]).copy()

    dataframe = dataframe.sort_values(["season_end_year", "Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    dataframe["current_season_feature_id"] = dataframe.index

    overall_states = defaultdict(empty_team_state)
    home_states = defaultdict(empty_team_state)
    away_states = defaultdict(empty_team_state)

    feature_rows = []

    grouped = dataframe.groupby(["season_end_year", "Date"], sort=True)

    for _, date_group in grouped:
        for _, row in date_group.iterrows():
            season = int(row["season_end_year"])
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]

            home_key = (season, home_team)
            away_key = (season, away_team)

            features = build_state_features(
                overall_states[home_key],
                overall_states[away_key],
                home_states[home_key],
                away_states[away_key]
            )

            features["current_season_feature_id"] = int(row["current_season_feature_id"])

            feature_rows.append(features)

        update_states_after_date(
            date_group,
            overall_states,
            home_states,
            away_states
        )

    features_dataframe = pd.DataFrame(feature_rows)

    columns_to_remove = []

    for column in dataframe.columns:
        if column.startswith("home_current_season_"):
            columns_to_remove.append(column)
        elif column.startswith("away_current_season_"):
            columns_to_remove.append(column)
        elif column.startswith("home_home_current_season_"):
            columns_to_remove.append(column)
        elif column.startswith("away_away_current_season_"):
            columns_to_remove.append(column)
        elif column.startswith("current_season_") and column != "current_season_feature_id":
            columns_to_remove.append(column)
        elif column.startswith("home_away_context_current_season_"):
            columns_to_remove.append(column)
        elif column == "min_team_matches_this_season_before":
            columns_to_remove.append(column)

    dataframe = dataframe.drop(columns=columns_to_remove, errors="ignore")

    final_dataframe = dataframe.merge(
        features_dataframe,
        on="current_season_feature_id",
        how="left"
    )

    final_dataframe = final_dataframe.drop(columns=["current_season_feature_id"])

    final_dataframe.to_csv(input_path, index=False)

    print("")
    print("Current-season draw features adicionadas:")
    print(input_path)
    print("Jogos:", len(final_dataframe))
    print("Features current-season adicionadas:", len(features_dataframe.columns) - 1)

    print("")
    print("Exemplos:")
    for column in features_dataframe.columns:
        if column != "current_season_feature_id":
            print("-", column)


if __name__ == "__main__":
    main()
