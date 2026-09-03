from collections import defaultdict

import numpy as np
import pandas as pd


DEFAULT_REST_DAYS = 14


def _count_recent_matches(match_dates, current_date, days):
    cutoff = current_date - pd.Timedelta(days=days)
    return sum(1 for match_date in match_dates if cutoff <= match_date < current_date)


def add_schedule_features(matches):
    dataframe = matches.copy()
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    dataframe = dataframe.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    histories = defaultdict(list)
    feature_rows = []

    for current_date, date_group in dataframe.groupby("Date", sort=True):
        for index, row in date_group.iterrows():
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]
            home_history = histories[home_team]
            away_history = histories[away_team]

            home_rest = (current_date - home_history[-1]).days if home_history else DEFAULT_REST_DAYS
            away_rest = (current_date - away_history[-1]).days if away_history else DEFAULT_REST_DAYS

            feature_rows.append(
                {
                    "row_index": index,
                    "home_rest_days": home_rest,
                    "away_rest_days": away_rest,
                    "rest_days_diff": home_rest - away_rest,
                    "home_short_rest_3d": home_rest <= 3,
                    "away_short_rest_3d": away_rest <= 3,
                    "home_short_rest_4d": home_rest <= 4,
                    "away_short_rest_4d": away_rest <= 4,
                    "home_matches_last_7d": _count_recent_matches(home_history, current_date, 7),
                    "away_matches_last_7d": _count_recent_matches(away_history, current_date, 7),
                    "home_matches_last_14d": _count_recent_matches(home_history, current_date, 14),
                    "away_matches_last_14d": _count_recent_matches(away_history, current_date, 14),
                    "home_matches_last_21d": _count_recent_matches(home_history, current_date, 21),
                    "away_matches_last_21d": _count_recent_matches(away_history, current_date, 21),
                    "home_matches_played_before": len(home_history),
                    "away_matches_played_before": len(away_history),
                }
            )

        for _, row in date_group.iterrows():
            histories[row["HomeTeam"]].append(current_date)
            histories[row["AwayTeam"]].append(current_date)

    features = pd.DataFrame(feature_rows).set_index("row_index")
    output = dataframe.join(features, how="left")
    output["matches_last_7d_diff"] = output["home_matches_last_7d"] - output["away_matches_last_7d"]
    output["matches_last_14d_diff"] = output["home_matches_last_14d"] - output["away_matches_last_14d"]
    output["matches_last_21d_diff"] = output["home_matches_last_21d"] - output["away_matches_last_21d"]
    return output


def add_early_season_features(matches):
    dataframe = matches.copy()
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    dataframe = dataframe.sort_values(["season_end_year", "Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    season_counts = defaultdict(int)
    rows = []

    for _, row in dataframe.iterrows():
        season = int(row["season_end_year"])
        home_key = (season, row["HomeTeam"])
        away_key = (season, row["AwayTeam"])
        home_before = season_counts[home_key]
        away_before = season_counts[away_key]

        rows.append(
            {
                "home_season_matches_before": home_before,
                "away_season_matches_before": away_before,
                "min_team_season_matches_before": min(home_before, away_before),
                "max_team_season_matches_before": max(home_before, away_before),
                "home_first_5_season_matches": home_before < 5,
                "away_first_5_season_matches": away_before < 5,
                "both_teams_first_5_season_matches": home_before < 5 and away_before < 5,
                "either_team_first_5_season_matches": home_before < 5 or away_before < 5,
                "home_first_10_season_matches": home_before < 10,
                "away_first_10_season_matches": away_before < 10,
                "both_teams_first_10_season_matches": home_before < 10 and away_before < 10,
                "season_match_count_diff": home_before - away_before,
            }
        )

        season_counts[home_key] += 1
        season_counts[away_key] += 1

    features = pd.DataFrame(rows)
    return pd.concat([dataframe, features], axis=1)


def add_approx_new_to_league_features(matches):
    dataframe = matches.copy()
    dataframe["season_end_year"] = pd.to_numeric(dataframe["season_end_year"], errors="coerce").astype("Int64")
    first_seen = {}
    for _, row in dataframe.sort_values(["season_end_year", "Date"]).iterrows():
        season = int(row["season_end_year"])
        first_seen.setdefault(row["HomeTeam"], season)
        first_seen.setdefault(row["AwayTeam"], season)

    output = dataframe.copy()
    output["home_first_seen_league_season"] = output["HomeTeam"].map(first_seen)
    output["away_first_seen_league_season"] = output["AwayTeam"].map(first_seen)
    output["home_new_to_league"] = output["season_end_year"].astype(int) == output["home_first_seen_league_season"].astype(int)
    output["away_new_to_league"] = output["season_end_year"].astype(int) == output["away_first_seen_league_season"].astype(int)
    output["either_new_to_league"] = output["home_new_to_league"] | output["away_new_to_league"]
    return output
