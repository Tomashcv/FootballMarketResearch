import argparse

import pandas as pd

from src.common.odds_utils import decimal_to_implied_probability
from src.common.paths import get_league_matches_path
from src.common.paths import get_market_dataset_path


HOME_ODDS_CANDIDATES = [
    "AvgH",
    "BbAvH",
    "B365H",
    "PSH",
    "PH",
    "MaxH",
    "BbMxH",
]

DRAW_ODDS_CANDIDATES = [
    "AvgD",
    "BbAvD",
    "B365D",
    "PSD",
    "PD",
    "MaxD",
    "BbMxD",
]

AWAY_ODDS_CANDIDATES = [
    "AvgA",
    "BbAvA",
    "B365A",
    "PSA",
    "PA",
    "MaxA",
    "BbMxA",
]


def existing_columns(dataframe, candidates):
    columns = []

    for column in candidates:
        if column in dataframe.columns:
            columns.append(column)

    return columns


def coalesce_numeric_columns(dataframe, columns):
    result = pd.Series([None] * len(dataframe), index=dataframe.index, dtype="float64")

    for column in columns:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        result = result.fillna(values)

    return result


def build_market_probabilities(row):
    home_raw = decimal_to_implied_probability(row["home_odds"])
    draw_raw = decimal_to_implied_probability(row["draw_odds"])
    away_raw = decimal_to_implied_probability(row["away_odds"])

    if home_raw is None or draw_raw is None or away_raw is None:
        return pd.Series({
            "market_probability_home": None,
            "market_probability_draw": None,
            "market_probability_away": None,
            "market_probability_not_draw": None,
        })

    total = home_raw + draw_raw + away_raw

    if total <= 0:
        return pd.Series({
            "market_probability_home": None,
            "market_probability_draw": None,
            "market_probability_away": None,
            "market_probability_not_draw": None,
        })

    home_probability = home_raw / total
    draw_probability = draw_raw / total
    away_probability = away_raw / total
    not_draw_probability = home_probability + away_probability

    return pd.Series({
        "market_probability_home": home_probability,
        "market_probability_draw": draw_probability,
        "market_probability_away": away_probability,
        "market_probability_not_draw": not_draw_probability,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    args = parser.parse_args()

    league_code = args.league.upper()
    market_name = "draw_vs_not_draw"

    input_path = get_league_matches_path(league_code)
    output_path = get_market_dataset_path(league_code, market_name)

    dataframe = pd.read_csv(input_path, low_memory=False)
    dataframe = dataframe.copy()

    home_columns = existing_columns(dataframe, HOME_ODDS_CANDIDATES)
    draw_columns = existing_columns(dataframe, DRAW_ODDS_CANDIDATES)
    away_columns = existing_columns(dataframe, AWAY_ODDS_CANDIDATES)

    if len(home_columns) == 0:
        raise ValueError("Não encontrei odds Home.")

    if len(draw_columns) == 0:
        raise ValueError("Não encontrei odds Draw.")

    if len(away_columns) == 0:
        raise ValueError("Não encontrei odds Away.")

    print("Colunas Home usadas por prioridade:")
    for column in home_columns:
        print("-", column)

    print("Colunas Draw usadas por prioridade:")
    for column in draw_columns:
        print("-", column)

    print("Colunas Away usadas por prioridade:")
    for column in away_columns:
        print("-", column)

    dataframe["home_odds"] = coalesce_numeric_columns(dataframe, home_columns)
    dataframe["draw_odds"] = coalesce_numeric_columns(dataframe, draw_columns)
    dataframe["away_odds"] = coalesce_numeric_columns(dataframe, away_columns)

    dataframe["target_draw"] = 0
    dataframe.loc[dataframe["FTR"] == "D", "target_draw"] = 1

    dataframe["target_not_draw"] = 1 - dataframe["target_draw"]

    before_drop = len(dataframe)

    dataframe = dataframe.dropna(subset=["home_odds", "draw_odds", "away_odds"]).copy()
    dataframe = dataframe[dataframe["home_odds"] > 1.0].copy()
    dataframe = dataframe[dataframe["draw_odds"] > 1.0].copy()
    dataframe = dataframe[dataframe["away_odds"] > 1.0].copy()

    after_drop = len(dataframe)

    probabilities = dataframe.apply(build_market_probabilities, axis=1)
    dataframe = pd.concat([dataframe, probabilities], axis=1)

    dataframe = dataframe.dropna(
        subset=[
            "market_probability_home",
            "market_probability_draw",
            "market_probability_away",
            "market_probability_not_draw",
        ]
    ).copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    print("")
    print("Dataset Draw vs Not Draw criado:")
    print(output_path)
    print("Jogos originais:", before_drop)
    print("Jogos com odds 1X2:", after_drop)
    print("Coverage:", round(after_drop / before_drop, 4))
    print("Draw rate:", round(float(dataframe["target_draw"].mean()), 4))
    print("Odd média Draw:", round(float(dataframe["draw_odds"].mean()), 4))
    print("Prob mercado média Draw:", round(float(dataframe["market_probability_draw"].mean()), 4))
    print("Prob mercado média Not Draw:", round(float(dataframe["market_probability_not_draw"].mean()), 4))


if __name__ == "__main__":
    main()
