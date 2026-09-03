import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import OUTPUTS_DIR


INTERNAL_ELO_COLUMNS = [
    "home_internal_elo_pre",
    "away_internal_elo_pre",
    "internal_elo_diff_home_minus_away",
    "internal_elo_home_win_prob",
]


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def match_scores(home_goals, away_goals):
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None, None
    if float(home_goals) > float(away_goals):
        return 1.0, 0.0
    if float(home_goals) < float(away_goals):
        return 0.0, 1.0
    return 0.5, 0.5


def sort_matches_for_elo(matches):
    dataframe = matches.copy()
    dataframe["_original_index"] = dataframe.index
    dataframe["_elo_date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    if "Time" in dataframe.columns:
        time_text = dataframe["Time"].fillna("00:00").astype(str)
        dataframe["_elo_time"] = pd.to_timedelta(time_text + ":00", errors="coerce")
    else:
        dataframe["_elo_time"] = pd.to_timedelta("00:00:00")
    dataframe["_elo_time"] = dataframe["_elo_time"].fillna(pd.to_timedelta("00:00:00"))
    return dataframe.sort_values(["_elo_date", "_elo_time", "HomeTeam", "AwayTeam", "_original_index"], kind="mergesort")


def add_internal_elo_features(
    matches,
    starting_elo=1500.0,
    k_factor=20.0,
    home_advantage_elo=60.0,
):
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Internal Elo requires columns: {sorted(missing)}")

    output = matches.copy()
    for column in INTERNAL_ELO_COLUMNS:
        output[column] = np.nan

    ratings = {}
    ordered = sort_matches_for_elo(output)

    for index, row in ordered.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        if pd.isna(home_team) or pd.isna(away_team):
            continue

        home_rating = ratings.get(home_team, float(starting_elo))
        away_rating = ratings.get(away_team, float(starting_elo))
        home_expected = expected_score(home_rating + float(home_advantage_elo), away_rating)
        away_expected = 1.0 - home_expected

        output.at[index, "home_internal_elo_pre"] = home_rating
        output.at[index, "away_internal_elo_pre"] = away_rating
        output.at[index, "internal_elo_diff_home_minus_away"] = home_rating - away_rating
        output.at[index, "internal_elo_home_win_prob"] = home_expected

        home_score, away_score = match_scores(row["FTHG"], row["FTAG"])
        if home_score is None:
            continue

        ratings[home_team] = home_rating + float(k_factor) * (home_score - home_expected)
        ratings[away_team] = away_rating + float(k_factor) * (away_score - away_expected)

    return output


def add_internal_elo_market_disagreement_features(matches):
    dataframe = matches.copy()
    if "internal_elo_home_win_prob" not in dataframe.columns:
        dataframe = add_internal_elo_features(dataframe)

    if "avg_1x2_AvgH_no_vig_probability" in dataframe.columns:
        dataframe["market_home_prob_minus_internal_elo_prob"] = (
            dataframe["avg_1x2_AvgH_no_vig_probability"] - dataframe["internal_elo_home_win_prob"]
        )

    if "avg_1x2_AvgA_no_vig_probability" in dataframe.columns:
        dataframe["internal_elo_away_win_prob"] = 1.0 - dataframe["internal_elo_home_win_prob"]
        dataframe["market_away_prob_minus_internal_elo_prob"] = (
            dataframe["avg_1x2_AvgA_no_vig_probability"] - dataframe["internal_elo_away_win_prob"]
        )

    return dataframe


def build_internal_elo_coverage(matches_by_league):
    rows = []
    for league, matches in matches_by_league.items():
        dataframe = add_internal_elo_features(matches)
        complete = dataframe[INTERNAL_ELO_COLUMNS].notna().all(axis=1)
        rows.append(
            {
                "league": league,
                "matches": int(len(dataframe)),
                "covered_matches": int(complete.sum()),
                "coverage_rate": float(complete.mean()) if len(dataframe) else 0.0,
                "teams": int(len(set(dataframe["HomeTeam"].dropna()) | set(dataframe["AwayTeam"].dropna()))),
            }
        )
    return pd.DataFrame(rows).sort_values("league").reset_index(drop=True)


def write_internal_elo_coverage_report(matches_by_league, path=OUTPUTS_DIR / "reports" / "internal_elo_coverage.csv"):
    coverage = build_internal_elo_coverage(matches_by_league)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_path, index=False)
    return coverage
