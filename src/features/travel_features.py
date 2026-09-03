import math
import os
import re
import unicodedata

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088
REQUIRED_COORDINATE_COLUMNS = {"team", "latitude", "longitude"}


COORDINATE_ALIASES = {
    "Team": "team",
    "FDCOUK": "team_alias",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Country": "country",
    "Stadium": "stadium",
}

OVERRIDE_ALIASES = {
    "team": "team",
    "stadium": "stadium",
    "city": "city",
    "country": "country",
    "latitude": "latitude",
    "longitude": "longitude",
}

KNOWN_TEAM_ALIASES = {
    "ac milan": "milan",
    "hellas verona": "verona",
    "inter milan": "inter",
    "internazionale": "inter",
    "spal 2013": "spal",
}


def normalize_team_name(name):
    value = str(name).strip().lower()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.replace("&", " and ")
    value = re.sub(r"\b(fc|afc|cf|cd|sc|ac|as|ss|sd|ud|rc|club|the)\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def canonical_team_key(name):
    key = normalize_team_name(name)
    return KNOWN_TEAM_ALIASES.get(key, key)


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Return great-circle distance in kilometers."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan

    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def validate_coordinate_reference(coordinates):
    coordinates = coordinates.rename(columns={k: v for k, v in COORDINATE_ALIASES.items() if k in coordinates.columns})
    missing = REQUIRED_COORDINATE_COLUMNS - set(coordinates.columns)
    if missing:
        raise ValueError(f"Coordinate reference missing columns: {sorted(missing)}")


def normalize_coordinate_reference(coordinates):
    reference = coordinates.rename(columns={k: v for k, v in COORDINATE_ALIASES.items() if k in coordinates.columns}).copy()
    validate_coordinate_reference(reference)
    reference["team"] = reference["team"].astype(str).str.strip()
    reference["latitude"] = pd.to_numeric(reference["latitude"], errors="coerce")
    reference["longitude"] = pd.to_numeric(reference["longitude"], errors="coerce")

    optional_defaults = {
        "stadium": "",
        "country": "",
        "surface": "",
        "timezone": "",
        "is_island": False,
        "remote_region": False,
        "source": "",
        "confidence": "",
        "match_source": "coordinate_reference",
    }
    for column, default in optional_defaults.items():
        if column not in reference.columns:
            reference[column] = default

    alias_rows = []
    if "team_alias" in reference.columns:
        for _, row in reference.dropna(subset=["team_alias"]).iterrows():
            alias_row = row.copy()
            alias_row["team"] = str(row["team_alias"]).strip()
            alias_rows.append(alias_row)
    if alias_rows:
        reference = pd.concat([reference, pd.DataFrame(alias_rows)], ignore_index=True)

    reference["team_normalized"] = reference["team"].map(canonical_team_key)
    reference = reference.dropna(subset=["latitude", "longitude"])
    reference = reference[(reference["latitude"] != 0) | (reference["longitude"] != 0)].copy()
    reference = reference.drop_duplicates("team", keep="first")
    return reference.reset_index(drop=True)


def load_manual_overrides(path):
    overrides = pd.read_csv(path)
    overrides = overrides.rename(columns={k: v for k, v in OVERRIDE_ALIASES.items() if k in overrides.columns}).copy()
    validate_coordinate_reference(overrides)
    overrides["team"] = overrides["team"].astype(str).str.strip()
    overrides["latitude"] = pd.to_numeric(overrides["latitude"], errors="coerce")
    overrides["longitude"] = pd.to_numeric(overrides["longitude"], errors="coerce")
    if "confidence" in overrides.columns:
        overrides = overrides[overrides["confidence"].astype(str).str.lower().eq("high")].copy()
    overrides["match_source"] = "manual_override"
    return normalize_coordinate_reference(overrides)


def combine_coordinate_sources(coordinates=None, manual_overrides=None):
    frames = []
    if coordinates is not None:
        frames.append(normalize_coordinate_reference(coordinates))
    if manual_overrides is not None:
        if isinstance(manual_overrides, (str, bytes, os.PathLike)):
            frames.append(load_manual_overrides(manual_overrides))
        else:
            overrides = manual_overrides.copy()
            if "match_source" not in overrides.columns:
                overrides["match_source"] = "manual_override"
            frames.append(normalize_coordinate_reference(overrides))
    if not frames:
        return pd.DataFrame(columns=["team", "latitude", "longitude", "team_normalized"])

    combined = pd.concat(frames, ignore_index=True)
    source_rank = []
    for index in range(len(combined)):
        source_rank.append(index)
    combined["_source_rank"] = source_rank
    combined = combined.drop_duplicates("team", keep="last")
    combined = combined.drop_duplicates("team_normalized", keep="last")
    return combined.drop(columns=["_source_rank"]).reset_index(drop=True)


def match_team_coordinates(teams, coordinate_reference):
    reference = normalize_coordinate_reference(coordinate_reference)
    exact = reference.set_index("team")
    normalized = reference.drop_duplicates("team_normalized").set_index("team_normalized")
    rows = []
    for team in teams:
        clean_team = str(team).strip()
        method = "unmatched"
        matched_team = ""
        row = None
        if clean_team in exact.index:
            row = exact.loc[clean_team]
            method = "exact"
            matched_team = clean_team
        else:
            key = canonical_team_key(clean_team)
            if key in normalized.index:
                row = normalized.loc[key]
                method = "normalized"
                matched_team = row["team"]
        if row is None:
            rows.append({"team": clean_team, "matched": False, "match_method": method})
        else:
            rows.append(
                {
                    "team": clean_team,
                    "matched": True,
                    "match_method": method,
                    "matched_team": matched_team,
                    "stadium": row.get("stadium", ""),
                    "country": row.get("country", ""),
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "source": row.get("source", ""),
                    "confidence": row.get("confidence", ""),
                    "match_source": row.get("match_source", "coordinate_reference"),
                }
            )
    return pd.DataFrame(rows)


def add_team_coordinate_features(matches, coordinates, manual_overrides=None):
    reference = combine_coordinate_sources(coordinates, manual_overrides)
    dataframe = matches.copy()
    dataframe["HomeTeam"] = dataframe["HomeTeam"].astype(str).str.strip()
    dataframe["AwayTeam"] = dataframe["AwayTeam"].astype(str).str.strip()
    dataframe["home_team_normalized"] = dataframe["HomeTeam"].map(canonical_team_key)
    dataframe["away_team_normalized"] = dataframe["AwayTeam"].map(canonical_team_key)

    home_reference = reference.add_prefix("home_").rename(columns={"home_team_normalized": "home_team_normalized"})
    away_reference = reference.add_prefix("away_").rename(columns={"away_team_normalized": "away_team_normalized"})

    dataframe = dataframe.merge(home_reference, on="home_team_normalized", how="left")
    dataframe = dataframe.merge(away_reference, on="away_team_normalized", how="left")

    dataframe["has_home_coordinates"] = dataframe["home_latitude"].notna() & dataframe["home_longitude"].notna()
    dataframe["has_away_coordinates"] = dataframe["away_latitude"].notna() & dataframe["away_longitude"].notna()
    dataframe["has_both_coordinates"] = dataframe["has_home_coordinates"] & dataframe["has_away_coordinates"]

    if "home_country" in dataframe.columns and "away_country" in dataframe.columns:
        dataframe["same_country_trip"] = (
            dataframe["home_country"].astype(str).str.lower()
            == dataframe["away_country"].astype(str).str.lower()
        )

    return dataframe


def add_travel_features(matches_with_coordinates):
    dataframe = matches_with_coordinates.copy()
    required = ["home_latitude", "home_longitude", "away_latitude", "away_longitude"]
    for column in required:
        if column not in dataframe.columns:
            dataframe[column] = np.nan

    dataframe["travel_distance_km"] = dataframe.apply(
        lambda row: haversine_distance_km(
            row["away_latitude"],
            row["away_longitude"],
            row["home_latitude"],
            row["home_longitude"],
        ),
        axis=1,
    )
    dataframe["travel_distance_log1p"] = np.log1p(dataframe["travel_distance_km"])
    dataframe["long_trip_300km"] = dataframe["travel_distance_km"] >= 300.0
    dataframe["long_trip_600km"] = dataframe["travel_distance_km"] >= 600.0
    dataframe["very_long_trip_1000km"] = dataframe["travel_distance_km"] >= 1000.0

    away_island = dataframe.get("away_is_island", False)
    home_island = dataframe.get("home_is_island", False)
    dataframe["island_trip"] = pd.Series(away_island, index=dataframe.index).fillna(False).astype(bool) | pd.Series(
        home_island, index=dataframe.index
    ).fillna(False).astype(bool)

    away_remote = dataframe.get("away_remote_region", False)
    home_remote = dataframe.get("home_remote_region", False)
    dataframe["remote_region_trip"] = pd.Series(away_remote, index=dataframe.index).fillna(False).astype(bool) | pd.Series(
        home_remote, index=dataframe.index
    ).fillna(False).astype(bool)

    return dataframe


def build_travel_features(matches, coordinates, manual_overrides=None):
    dataframe = add_team_coordinate_features(matches, coordinates, manual_overrides)
    return add_travel_features(dataframe)
