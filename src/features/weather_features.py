import numpy as np
import pandas as pd


MATCH_KEY_COLUMNS = ["Date", "HomeTeam", "AwayTeam"]
WEATHER_VALUE_COLUMNS = [
    "temperature_c",
    "precipitation_mm",
    "wind_speed_kph",
    "humidity_pct",
    "pressure_hpa",
    "snow_depth_cm",
]


def normalize_match_date(dataframe):
    output = dataframe.copy()
    output["Date"] = pd.to_datetime(output["Date"], errors="coerce").dt.normalize()
    return output


def add_weather_features(matches, weather):
    """Join pre-fetched weather by match key.

    The function intentionally does not fetch weather. The caller must supply
    weather observed or forecasted at the appropriate pre-kickoff timestamp.
    """
    missing = set(MATCH_KEY_COLUMNS) - set(weather.columns)
    if missing:
        raise ValueError(f"Weather reference missing columns: {sorted(missing)}")

    dataframe = normalize_match_date(matches)
    weather_frame = normalize_match_date(weather)

    keep_columns = MATCH_KEY_COLUMNS + [c for c in WEATHER_VALUE_COLUMNS + ["weather_code"] if c in weather_frame.columns]
    weather_frame = weather_frame[keep_columns].drop_duplicates(MATCH_KEY_COLUMNS, keep="last")
    weather_frame = weather_frame.add_prefix("weather_").rename(
        columns={
            "weather_Date": "Date",
            "weather_HomeTeam": "HomeTeam",
            "weather_AwayTeam": "AwayTeam",
        }
    )

    output = dataframe.merge(weather_frame, on=MATCH_KEY_COLUMNS, how="left")
    output["has_weather"] = output[[c for c in output.columns if c.startswith("weather_")]].notna().any(axis=1)
    return output


def add_weather_shock_features(matches_with_weather, away_climate_normals):
    """Compare match weather with the away team's normal monthly climate."""
    required = {"team", "month"}
    missing = required - set(away_climate_normals.columns)
    if missing:
        raise ValueError(f"Climate normal reference missing columns: {sorted(missing)}")

    dataframe = matches_with_weather.copy()
    dataframe["match_month"] = pd.to_datetime(dataframe["Date"], errors="coerce").dt.month

    normals = away_climate_normals.copy()
    normals = normals.rename(
        columns={
            "avg_temp_c": "temperature_c",
            "avg_wind_kph": "wind_speed_kph",
            "avg_precip_mm": "precipitation_mm",
        }
    )
    normals["team"] = normals["team"].astype(str)
    normals["month"] = pd.to_numeric(normals["month"], errors="coerce").astype("Int64")
    normals = normals.add_prefix("away_normal_").rename(
        columns={
            "away_normal_team": "AwayTeam",
            "away_normal_month": "match_month",
        }
    )

    output = dataframe.merge(normals, on=["AwayTeam", "match_month"], how="left")

    pairs = [
        ("weather_temperature_c", "away_normal_temperature_c", "away_temperature_shock_c"),
        ("weather_precipitation_mm", "away_normal_precipitation_mm", "away_precipitation_shock_mm"),
        ("weather_wind_speed_kph", "away_normal_wind_speed_kph", "away_wind_speed_shock_kph"),
        ("weather_humidity_pct", "away_normal_humidity_pct", "away_humidity_shock_pct"),
    ]
    for observed_col, normal_col, output_col in pairs:
        if observed_col in output.columns and normal_col in output.columns:
            output[output_col] = pd.to_numeric(output[observed_col], errors="coerce") - pd.to_numeric(
                output[normal_col], errors="coerce"
            )
        else:
            output[output_col] = np.nan

    shock_columns = [column for column in output.columns if column.startswith("away_") and column.endswith("_shock_c")]
    shock_columns += [column for column in output.columns if "_shock_" in column]
    output["has_weather_normals"] = output[[c for c in output.columns if c.startswith("away_normal_")]].notna().any(axis=1)
    output["has_weather_shock"] = output[shock_columns].notna().any(axis=1) if shock_columns else False
    return output
