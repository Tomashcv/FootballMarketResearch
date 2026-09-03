import argparse

import pandas as pd

from src.common.odds_utils import decimal_to_implied_probability
from src.common.odds_utils import normalize_two_way_probabilities
from src.common.paths import get_league_matches_path
from src.common.paths import get_market_dataset_path


# Preferimos odds médias/consenso para modelar o mercado.
# Não usamos Max/BbMx como primeira opção porque isso é "best price",
# não necessariamente a probabilidade média do mercado.
OVER_ODDS_CANDIDATES = [
    "Avg>2.5",
    "BbAv>2.5",
    "B365>2.5",
    "P>2.5",
    "Max>2.5",
    "BbMx>2.5",
]

UNDER_ODDS_CANDIDATES = [
    "Avg<2.5",
    "BbAv<2.5",
    "B365<2.5",
    "P<2.5",
    "Max<2.5",
    "BbMx<2.5",
]


def existing_columns(dataframe, candidates):
    columns = []

    for column in candidates:
        if column in dataframe.columns:
            columns.append(column)

    return columns


def coalesce_numeric_columns(dataframe, columns):
    if len(columns) == 0:
        return pd.Series([None] * len(dataframe), index=dataframe.index)

    result = pd.Series([None] * len(dataframe), index=dataframe.index, dtype="float64")

    for column in columns:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        result = result.fillna(values)

    return result


def build_market_probabilities(row):
    over_probability_raw = decimal_to_implied_probability(row["over_25_odds"])
    under_probability_raw = decimal_to_implied_probability(row["under_25_odds"])

    over_probability, under_probability = normalize_two_way_probabilities(
        over_probability_raw,
        under_probability_raw
    )

    return pd.Series({
        "market_probability_over": over_probability,
        "market_probability_under": under_probability,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    args = parser.parse_args()

    league_code = args.league.upper()
    market_name = "over_under_25"

    input_path = get_league_matches_path(league_code)
    output_path = get_market_dataset_path(league_code, market_name)

    dataframe = pd.read_csv(input_path, low_memory=False)
    dataframe = dataframe.copy()

    over_columns = existing_columns(dataframe, OVER_ODDS_CANDIDATES)
    under_columns = existing_columns(dataframe, UNDER_ODDS_CANDIDATES)

    if len(over_columns) == 0:
        print("Colunas disponíveis:")
        print(list(dataframe.columns))
        raise ValueError("Não encontrei nenhuma coluna de odds Over 2.5.")

    if len(under_columns) == 0:
        print("Colunas disponíveis:")
        print(list(dataframe.columns))
        raise ValueError("Não encontrei nenhuma coluna de odds Under 2.5.")

    print("Colunas Over disponíveis usadas por prioridade:")
    for column in over_columns:
        print("-", column)

    print("Colunas Under disponíveis usadas por prioridade:")
    for column in under_columns:
        print("-", column)

    dataframe["total_goals"] = dataframe["FTHG"].astype(float) + dataframe["FTAG"].astype(float)

    dataframe["target_over_25"] = 0
    dataframe.loc[dataframe["total_goals"] > 2.5, "target_over_25"] = 1

    dataframe["over_25_odds"] = coalesce_numeric_columns(dataframe, over_columns)
    dataframe["under_25_odds"] = coalesce_numeric_columns(dataframe, under_columns)

    before_drop = len(dataframe)

    dataframe = dataframe.dropna(subset=["over_25_odds", "under_25_odds"]).copy()
    dataframe = dataframe[dataframe["over_25_odds"] > 1.0].copy()
    dataframe = dataframe[dataframe["under_25_odds"] > 1.0].copy()

    after_drop = len(dataframe)

    probabilities = dataframe.apply(build_market_probabilities, axis=1)
    dataframe = pd.concat([dataframe, probabilities], axis=1)

    dataframe = dataframe.dropna(subset=["market_probability_over", "market_probability_under"]).copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    print("")
    print("Dataset Over/Under 2.5 criado:")
    print(output_path)
    print("Jogos originais:", before_drop)
    print("Jogos com odds O/U 2.5:", after_drop)
    print("Coverage:", round(after_drop / before_drop, 4))
    print("Over rate:", round(float(dataframe["target_over_25"].mean()), 4))
    print("Odd média Over:", round(float(dataframe["over_25_odds"].mean()), 4))
    print("Odd média Under:", round(float(dataframe["under_25_odds"].mean()), 4))
    print("Prob mercado média Over:", round(float(dataframe["market_probability_over"].mean()), 4))
    print("Prob mercado média Under:", round(float(dataframe["market_probability_under"].mean()), 4))


if __name__ == "__main__":
    main()
