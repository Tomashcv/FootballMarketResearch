import numpy as np
import pandas as pd

from src.features.schedule_features import add_approx_new_to_league_features
from src.features.schedule_features import add_early_season_features
from src.features.schedule_features import add_schedule_features
from src.features.internal_elo_features import add_internal_elo_market_disagreement_features
from src.features.travel_features import build_travel_features
from src.features.weather_features import add_weather_features
from src.features.weather_features import add_weather_shock_features


CLOSING_MARKERS = ("AHCh", "CAH", "AvgC", "MaxC", "B365C", "PC", "PSCH", "PSCD", "PSCA")


def is_closing_column(column):
    return any(marker in column for marker in CLOSING_MARKERS)


def assert_no_closing_columns(columns):
    closing = [column for column in columns if is_closing_column(str(column))]
    if closing:
        raise ValueError(f"Closing odds columns are not bet-time-safe features: {closing}")


def normalized_probabilities_from_odds(dataframe, odds_columns, prefix):
    output = dataframe.copy()
    assert_no_closing_columns(odds_columns)
    raw_columns = []
    for column in odds_columns:
        raw_column = f"{prefix}_{column}_raw_implied".replace(">", "over_").replace("<", "under_").replace(".", "_")
        output[raw_column] = 1.0 / pd.to_numeric(output[column], errors="coerce")
        raw_columns.append(raw_column)
    total = output[raw_columns].sum(axis=1)
    for column, raw_column in zip(odds_columns, raw_columns):
        probability_column = f"{prefix}_{column}_no_vig_probability".replace(">", "over_").replace("<", "under_").replace(".", "_")
        output[probability_column] = output[raw_column] / total
    output[f"{prefix}_overround"] = total
    return output


def add_market_disagreement_features(matches):
    dataframe = matches.copy()

    if {"AvgH", "AvgD", "AvgA"}.issubset(dataframe.columns):
        dataframe = normalized_probabilities_from_odds(dataframe, ["AvgH", "AvgD", "AvgA"], "avg_1x2")
    if {"MaxH", "MaxD", "MaxA"}.issubset(dataframe.columns):
        dataframe = normalized_probabilities_from_odds(dataframe, ["MaxH", "MaxD", "MaxA"], "max_1x2")
    if {"B365H", "B365D", "B365A"}.issubset(dataframe.columns):
        dataframe = normalized_probabilities_from_odds(dataframe, ["B365H", "B365D", "B365A"], "b365_1x2")
    if {"PSH", "PSD", "PSA"}.issubset(dataframe.columns):
        dataframe = normalized_probabilities_from_odds(dataframe, ["PSH", "PSD", "PSA"], "ps_1x2")

    for side in ["H", "D", "A"]:
        avg_col = f"avg_1x2_Avg{side}_no_vig_probability"
        max_col = f"max_1x2_Max{side}_no_vig_probability"
        b365_col = f"b365_1x2_B365{side}_no_vig_probability"
        ps_col = f"ps_1x2_PS{side}_no_vig_probability"
        if avg_col in dataframe.columns and max_col in dataframe.columns:
            dataframe[f"market_max_minus_avg_{side.lower()}_prob"] = dataframe[max_col] - dataframe[avg_col]
        if avg_col in dataframe.columns and b365_col in dataframe.columns:
            dataframe[f"market_b365_minus_avg_{side.lower()}_prob"] = dataframe[b365_col] - dataframe[avg_col]
        if avg_col in dataframe.columns and ps_col in dataframe.columns:
            dataframe[f"market_ps_minus_avg_{side.lower()}_prob"] = dataframe[ps_col] - dataframe[avg_col]

    if {"Avg>2.5", "Avg<2.5"}.issubset(dataframe.columns):
        dataframe = normalized_probabilities_from_odds(dataframe, ["Avg>2.5", "Avg<2.5"], "avg_ou25")
    if {"Max>2.5", "Max<2.5"}.issubset(dataframe.columns):
        dataframe = normalized_probabilities_from_odds(dataframe, ["Max>2.5", "Max<2.5"], "max_ou25")

    if {"AvgAHH", "AvgAHA"}.issubset(dataframe.columns):
        dataframe = normalized_probabilities_from_odds(dataframe, ["AvgAHH", "AvgAHA"], "avg_ah")
    if {"MaxAHH", "MaxAHA"}.issubset(dataframe.columns):
        dataframe = normalized_probabilities_from_odds(dataframe, ["MaxAHH", "MaxAHA"], "max_ah")
    if {"avg_ah_AvgAHH_no_vig_probability", "max_ah_MaxAHH_no_vig_probability"}.issubset(dataframe.columns):
        dataframe["market_max_minus_avg_ah_home_prob"] = (
            dataframe["max_ah_MaxAHH_no_vig_probability"] - dataframe["avg_ah_AvgAHH_no_vig_probability"]
        )
        dataframe["market_max_minus_avg_ah_away_prob"] = (
            dataframe["max_ah_MaxAHA_no_vig_probability"] - dataframe["avg_ah_AvgAHA_no_vig_probability"]
        )

    return dataframe


def add_elo_market_disagreement_features(matches, elo):
    required = {"Date", "team", "elo"}
    missing = required - set(elo.columns)
    if missing:
        raise ValueError(f"Elo reference missing columns: {sorted(missing)}")

    dataframe = matches.copy()
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    elo_frame = elo.copy()
    elo_frame["Date"] = pd.to_datetime(elo_frame["Date"], errors="coerce")
    elo_frame = elo_frame.sort_values(["team", "Date"])

    def latest_elo(team, date):
        team_rows = elo_frame[(elo_frame["team"] == team) & (elo_frame["Date"] < date)]
        if len(team_rows) == 0:
            return np.nan
        return float(team_rows.iloc[-1]["elo"])

    dataframe["home_elo"] = dataframe.apply(lambda row: latest_elo(row["HomeTeam"], row["Date"]), axis=1)
    dataframe["away_elo"] = dataframe.apply(lambda row: latest_elo(row["AwayTeam"], row["Date"]), axis=1)
    dataframe["elo_diff"] = dataframe["home_elo"] - dataframe["away_elo"]
    dataframe["elo_home_probability_simple"] = 1.0 / (1.0 + 10.0 ** (-dataframe["elo_diff"] / 400.0))

    if "avg_1x2_AvgH_no_vig_probability" in dataframe.columns:
        dataframe["elo_minus_market_home_probability"] = (
            dataframe["elo_home_probability_simple"] - dataframe["avg_1x2_AvgH_no_vig_probability"]
        )
    return dataframe


def build_contextual_features(matches, coordinates=None, weather=None, climate_normals=None, elo=None):
    dataframe = matches.copy()
    dataframe = add_schedule_features(dataframe)
    dataframe = add_early_season_features(dataframe)
    dataframe = add_approx_new_to_league_features(dataframe)
    dataframe = add_market_disagreement_features(dataframe)
    if {"FTHG", "FTAG"}.issubset(dataframe.columns):
        dataframe = add_internal_elo_market_disagreement_features(dataframe)

    if coordinates is not None:
        dataframe = build_travel_features(dataframe, coordinates)
    if weather is not None:
        dataframe = add_weather_features(dataframe, weather)
    if climate_normals is not None:
        dataframe = add_weather_shock_features(dataframe, climate_normals)
    if elo is not None:
        dataframe = add_elo_market_disagreement_features(dataframe, elo)

    return dataframe


def build_data_availability_summary(matches_by_league):
    rows = []
    for league, dataframe in matches_by_league.items():
        columns = set(dataframe.columns)
        rows.append(
            {
                "league": league,
                "matches": len(dataframe),
                "has_time": "Time" in columns,
                "has_main_1x2_odds": {"AvgH", "AvgD", "AvgA"}.issubset(columns),
                "has_main_ou25_odds": {"Avg>2.5", "Avg<2.5"}.issubset(columns),
                "has_main_ah_odds": {"AHh", "AvgAHH", "AvgAHA"}.issubset(columns),
                "has_closing_odds": any(is_closing_column(column) for column in columns),
                "has_local_stadium_coordinates": False,
                "has_local_weather": False,
                "has_local_clubelo": False,
                "schedule_features_available": True,
                "early_season_features_available": True,
                "approx_new_to_league_available": True,
                "market_disagreement_available": True,
            }
        )
    return pd.DataFrame(rows).sort_values("league").reset_index(drop=True)
