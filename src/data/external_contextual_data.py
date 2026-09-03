import io
import json
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from src.common.io_utils import safe_slug
from src.common.paths import DATA_DIR
from src.common.paths import OUTPUTS_DIR
from src.features.travel_features import build_travel_features


WEATHER_CACHE_DIR = DATA_DIR / "external" / "weather" / "open_meteo_cache"
WEATHER_RANGE_CACHE_DIR = DATA_DIR / "external" / "weather" / "open_meteo_range_cache"
WEATHER_TABLE_PATH = DATA_DIR / "external" / "weather" / "historical_match_weather.csv"
CLIMATE_NORMALS_PATH = DATA_DIR / "external" / "weather" / "monthly_climate_normals.csv"
CLUBELO_CACHE_DIR = DATA_DIR / "external" / "clubelo" / "team_cache"
CLUBELO_TABLE_PATH = DATA_DIR / "external" / "clubelo" / "prematch_elo.csv"
FAILED_REQUESTS_PATH = OUTPUTS_DIR / "reports" / "external_contextual_failed_requests.csv"
PITCH_SURFACE_PATH = DATA_DIR / "manual" / "stadium_surface_overrides.csv"
PITCH_SURFACE_UNKNOWN_REPORT_PATH = OUTPUTS_DIR / "reports" / "pitch_surface_unknown_teams.md"
ALLOWED_PITCH_SURFACES = {"grass", "artificial", "hybrid", "unknown"}

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CLUBELO_TEAM_URL = "https://api.clubelo.com/{team}"
CLUBELO_USER_AGENT = "FootballV2-ruflo-contextual-data/1.0"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

MATCH_KEY_COLUMNS = ["Date", "HomeTeam", "AwayTeam"]


def normalize_date(value):
    return pd.to_datetime(value, errors="coerce").normalize()


def weather_cache_path(cache_dir, team, date):
    date_value = normalize_date(date).strftime("%Y-%m-%d")
    return Path(cache_dir) / f"{date_value}_{safe_slug(team)}.json"


def weather_range_cache_path(cache_dir, stadium, start_date, end_date):
    start_value = normalize_date(start_date).strftime("%Y-%m-%d")
    end_value = normalize_date(end_date).strftime("%Y-%m-%d")
    return Path(cache_dir) / f"{safe_slug(stadium)}_{start_value}_{end_value}.json"


def parse_open_meteo_daily(payload):
    daily = payload.get("daily", {})
    if not daily or not daily.get("time"):
        return {}

    def first(name):
        values = daily.get(name, [])
        return values[0] if values else np.nan

    return {
        "temperature_c": first("temperature_2m_mean"),
        "precipitation_mm": first("precipitation_sum"),
        "wind_speed_kph": first("wind_speed_10m_max"),
        "weather_code": first("weather_code"),
    }


def parse_open_meteo_daily_series(payload):
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    rows = []
    for index, date in enumerate(dates):
        def at(name):
            values = daily.get(name, [])
            return values[index] if index < len(values) else np.nan

        rows.append(
            {
                "Date": normalize_date(date),
                "temperature_c": at("temperature_2m_mean"),
                "precipitation_mm": at("precipitation_sum"),
                "wind_speed_kph": at("wind_speed_10m_max"),
                "weather_code": at("weather_code"),
            }
        )
    return pd.DataFrame(rows)


def fetch_open_meteo_daily_weather(latitude, longitude, date, session=None, timeout=30):
    session = session or requests
    date_value = normalize_date(date).strftime("%Y-%m-%d")
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": date_value,
        "end_date": date_value,
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,weather_code",
        "timezone": "UTC",
    }
    response = session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_open_meteo_daily_weather_range(latitude, longitude, start_date, end_date, session=None, timeout=30):
    session = session or requests
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": normalize_date(start_date).strftime("%Y-%m-%d"),
        "end_date": normalize_date(end_date).strftime("%Y-%m-%d"),
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,weather_code",
        "timezone": "UTC",
    }
    response = session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def cache_identifier(row):
    stadium = row.get("home_stadium", "")
    if pd.notna(stadium) and str(stadium).strip():
        return str(stadium).strip()
    return str(row["HomeTeam"]).strip()


def cached_open_meteo_weather(
    team,
    latitude,
    longitude,
    date,
    cache_dir=WEATHER_CACHE_DIR,
    fetch_missing=True,
    session=None,
    force_refresh=False,
    sleep_seconds=0.0,
):
    path = weather_cache_path(cache_dir, team, date)
    if path.exists() and not force_refresh:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return parse_open_meteo_daily(payload), True

    if not fetch_missing:
        return {}, False

    if sleep_seconds:
        time.sleep(float(sleep_seconds))
    payload = fetch_open_meteo_daily_weather(latitude, longitude, date, session=session)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return parse_open_meteo_daily(payload), False


def cached_open_meteo_weather_range(
    stadium,
    latitude,
    longitude,
    start_date,
    end_date,
    cache_dir=WEATHER_RANGE_CACHE_DIR,
    fetch_missing=True,
    session=None,
    force_refresh=False,
    sleep_seconds=0.0,
):
    path = weather_range_cache_path(cache_dir, stadium, start_date, end_date)
    if path.exists() and not force_refresh:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return parse_open_meteo_daily_series(payload), True

    if not fetch_missing:
        return pd.DataFrame(columns=["Date", "temperature_c", "precipitation_mm", "wind_speed_kph", "weather_code"]), False

    if sleep_seconds:
        time.sleep(float(sleep_seconds))
    payload = fetch_open_meteo_daily_weather_range(latitude, longitude, start_date, end_date, session=session)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return parse_open_meteo_daily_series(payload), False


def build_historical_weather_table(
    matches,
    coordinates,
    manual_overrides=None,
    cache_dir=WEATHER_CACHE_DIR,
    fetch_missing=True,
    session=None,
    force_refresh=False,
    sleep_seconds=0.0,
):
    with_coordinates = build_travel_features(matches, coordinates, manual_overrides)
    fetched_by_request = {}
    rows = []
    for _, row in with_coordinates.iterrows():
        date = normalize_date(row["Date"])
        if pd.isna(date):
            continue

        base = {
            "Date": date,
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "team": row["HomeTeam"],
            "stadium": row.get("home_stadium", ""),
            "latitude": row.get("home_latitude", np.nan),
            "longitude": row.get("home_longitude", np.nan),
            "weather_cache_hit": False,
            "weather_fetch_attempted": False,
        }

        if not bool(row.get("has_home_coordinates", False)):
            rows.append(base)
            continue

        identifier = cache_identifier(row)
        request_key = (
            date.strftime("%Y-%m-%d"),
            identifier,
            round(float(row["home_latitude"]), 6),
            round(float(row["home_longitude"]), 6),
        )
        if request_key in fetched_by_request:
            values, cache_hit = fetched_by_request[request_key]
        else:
            values, cache_hit = cached_open_meteo_weather(
                identifier,
                row["home_latitude"],
                row["home_longitude"],
                date,
                cache_dir=cache_dir,
                fetch_missing=fetch_missing,
                session=session,
                force_refresh=force_refresh,
                sleep_seconds=sleep_seconds,
            )
            fetched_by_request[request_key] = (values, cache_hit)
        base.update(values)
        base["weather_cache_hit"] = cache_hit
        base["weather_fetch_attempted"] = not cache_hit and fetch_missing
        rows.append(base)

    output = pd.DataFrame(rows)
    if len(output) == 0:
        return pd.DataFrame(
            columns=MATCH_KEY_COLUMNS
            + [
                "team",
                "stadium",
                "latitude",
                "longitude",
                "temperature_c",
                "precipitation_mm",
                "wind_speed_kph",
                "weather_code",
                "weather_cache_hit",
                "weather_fetch_attempted",
            ]
        )
    for column in ["temperature_c", "precipitation_mm", "wind_speed_kph", "weather_code"]:
        if column not in output.columns:
            output[column] = np.nan
    return output.drop_duplicates(MATCH_KEY_COLUMNS, keep="last").reset_index(drop=True)


def empty_weather_table():
    return pd.DataFrame(
        columns=MATCH_KEY_COLUMNS
        + [
            "team",
            "stadium",
            "latitude",
            "longitude",
            "temperature_c",
            "precipitation_mm",
            "wind_speed_kph",
            "weather_code",
            "weather_cache_hit",
            "weather_range_cache_hit",
            "weather_fetch_attempted",
        ]
    )


def build_historical_weather_table_range(
    matches,
    coordinates,
    manual_overrides=None,
    range_cache_dir=WEATHER_RANGE_CACHE_DIR,
    match_cache_dir=WEATHER_CACHE_DIR,
    fetch_missing=True,
    session=None,
    force_refresh=False,
    sleep_seconds=0.0,
):
    with_coordinates = build_travel_features(matches, coordinates, manual_overrides)
    if len(with_coordinates) == 0:
        return empty_weather_table()

    with_coordinates["weather_date"] = pd.to_datetime(with_coordinates["Date"], errors="coerce").dt.normalize()
    rows = []
    series_by_request = {}

    coordinate_rows = with_coordinates[with_coordinates.get("has_home_coordinates", False).astype(bool)].copy()
    if len(coordinate_rows) > 0:
        coordinate_rows["cache_identifier"] = coordinate_rows.apply(cache_identifier, axis=1)
        coordinate_rows["range_key"] = coordinate_rows.apply(
            lambda row: (
                row["cache_identifier"],
                round(float(row["home_latitude"]), 6),
                round(float(row["home_longitude"]), 6),
            ),
            axis=1,
        )
        for range_key, group in coordinate_rows.dropna(subset=["weather_date"]).groupby("range_key", sort=False):
            first = group.iloc[0]
            start_date = group["weather_date"].min()
            end_date = group["weather_date"].max()
            weather_series, range_cache_hit = cached_open_meteo_weather_range(
                first["cache_identifier"],
                first["home_latitude"],
                first["home_longitude"],
                start_date,
                end_date,
                cache_dir=range_cache_dir,
                fetch_missing=fetch_missing,
                session=session,
                force_refresh=force_refresh,
                sleep_seconds=sleep_seconds,
            )
            if len(weather_series) > 0:
                weather_series = weather_series.copy()
                weather_series["Date"] = pd.to_datetime(weather_series["Date"], errors="coerce").dt.normalize()
                weather_series = weather_series.drop_duplicates("Date", keep="last").set_index("Date")
            series_by_request[range_key] = (weather_series, range_cache_hit)

    for _, row in with_coordinates.iterrows():
        date = normalize_date(row["Date"])
        if pd.isna(date):
            continue

        base = {
            "Date": date,
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "team": row["HomeTeam"],
            "stadium": row.get("home_stadium", ""),
            "latitude": row.get("home_latitude", np.nan),
            "longitude": row.get("home_longitude", np.nan),
            "weather_cache_hit": False,
            "weather_range_cache_hit": False,
            "weather_fetch_attempted": False,
        }
        if not bool(row.get("has_home_coordinates", False)):
            rows.append(base)
            continue

        identifier = cache_identifier(row)
        range_key = (identifier, round(float(row["home_latitude"]), 6), round(float(row["home_longitude"]), 6))
        weather_series, range_cache_hit = series_by_request.get(
            range_key,
            (pd.DataFrame(columns=["temperature_c", "precipitation_mm", "wind_speed_kph", "weather_code"]), False),
        )
        if len(weather_series) > 0 and date in weather_series.index:
            values = weather_series.loc[date][["temperature_c", "precipitation_mm", "wind_speed_kph", "weather_code"]].to_dict()
            base.update(values)
            base["weather_range_cache_hit"] = range_cache_hit
            base["weather_fetch_attempted"] = not range_cache_hit and fetch_missing
        else:
            values, match_cache_hit = cached_open_meteo_weather(
                identifier,
                row["home_latitude"],
                row["home_longitude"],
                date,
                cache_dir=match_cache_dir,
                fetch_missing=False,
            )
            base.update(values)
            base["weather_cache_hit"] = match_cache_hit
            base["weather_fetch_attempted"] = False
        rows.append(base)

    output = pd.DataFrame(rows)
    if len(output) == 0:
        return empty_weather_table()
    for column in ["temperature_c", "precipitation_mm", "wind_speed_kph", "weather_code"]:
        if column not in output.columns:
            output[column] = np.nan
    return output.drop_duplicates(MATCH_KEY_COLUMNS, keep="last").reset_index(drop=True)


def build_monthly_climate_normals(weather):
    if len(weather) == 0:
        return pd.DataFrame(columns=["team", "month", "avg_temp_c", "avg_wind_kph", "avg_precip_mm"])
    dataframe = weather.copy()
    for column in ["temperature_c", "precipitation_mm", "wind_speed_kph"]:
        if column not in dataframe.columns:
            dataframe[column] = np.nan
    dataframe = dataframe.dropna(subset=["temperature_c", "precipitation_mm", "wind_speed_kph"], how="all")
    if len(dataframe) == 0:
        return pd.DataFrame(columns=["team", "month", "avg_temp_c", "avg_wind_kph", "avg_precip_mm", "samples"])
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    dataframe["month"] = dataframe["Date"].dt.month
    grouped = (
        dataframe.dropna(subset=["team", "month"])
        .groupby(["team", "month"], as_index=False)
        .agg(
            avg_temp_c=("temperature_c", "mean"),
            avg_wind_kph=("wind_speed_kph", "mean"),
            avg_precip_mm=("precipitation_mm", "mean"),
            samples=("Date", "count"),
        )
    )
    return grouped.sort_values(["team", "month"]).reset_index(drop=True)


def clubelo_cache_path(cache_dir, team):
    return Path(cache_dir) / f"{safe_slug(team)}.csv"


def clubelo_url(team):
    return CLUBELO_TEAM_URL.format(team=quote(str(team).strip()))


def is_retryable_clubelo_error(error):
    if isinstance(
        error,
        (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ),
    ):
        return True
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code in RETRYABLE_STATUS_CODES


def fetch_clubelo_team_history(
    team,
    session=None,
    timeout=30,
    max_retries=3,
    retry_backoff_seconds=5.0,
):
    session = session or requests
    url = clubelo_url(team)
    headers = {"User-Agent": CLUBELO_USER_AGENT}
    attempts = max(1, int(max_retries) + 1)
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=timeout, headers=headers)
            status_code = getattr(response, "status_code", None)
            if status_code in RETRYABLE_STATUS_CODES:
                error = requests.exceptions.HTTPError(f"HTTP {status_code} for {url}", response=response)
                raise error
            response.raise_for_status()
            return response.text
        except Exception as error:
            last_error = error
            if not is_retryable_clubelo_error(error) or attempt == attempts:
                raise
            if retry_backoff_seconds:
                time.sleep(float(retry_backoff_seconds) * attempt)

    raise last_error


def normalize_clubelo_history(csv_text, requested_team):
    try:
        dataframe = pd.read_csv(io.StringIO(csv_text))
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["Date", "team", "elo"])
    if len(dataframe) == 0:
        return pd.DataFrame(columns=["Date", "team", "elo"])

    columns = {column.lower(): column for column in dataframe.columns}
    date_col = columns.get("from") or columns.get("date")
    elo_col = columns.get("elo")
    club_col = columns.get("club") or columns.get("team")
    if date_col is None or elo_col is None:
        return pd.DataFrame(columns=["Date", "team", "elo"])

    output = pd.DataFrame(
        {
            "Date": pd.to_datetime(dataframe[date_col], errors="coerce").dt.normalize(),
            "team": dataframe[club_col].astype(str) if club_col else str(requested_team),
            "elo": pd.to_numeric(dataframe[elo_col], errors="coerce"),
        }
    )
    output = output.dropna(subset=["Date", "elo"]).copy()
    output["requested_team"] = str(requested_team)
    return output.sort_values(["team", "Date"]).reset_index(drop=True)


def cached_clubelo_history(
    team,
    cache_dir=CLUBELO_CACHE_DIR,
    fetch_missing=True,
    session=None,
    force_refresh=False,
    sleep_seconds=0.0,
    max_retries=3,
    retry_backoff_seconds=5.0,
):
    path = clubelo_cache_path(cache_dir, team)
    if path.exists() and not force_refresh:
        return pd.read_csv(path, parse_dates=["Date"]), True
    if not fetch_missing:
        return pd.DataFrame(columns=["Date", "team", "elo", "requested_team"]), False

    if sleep_seconds:
        time.sleep(float(sleep_seconds))
    csv_text = fetch_clubelo_team_history(
        team,
        session=session,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    history = normalize_clubelo_history(csv_text, team)
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, index=False)
    return history, False


def failed_request_row(team, error):
    return {
        "provider": "clubelo",
        "team": str(team),
        "url": clubelo_url(team),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def write_failed_requests(failures, path=FAILED_REQUESTS_PATH):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["provider", "team", "url", "error_type", "error"]
    dataframe = pd.DataFrame(failures, columns=columns)
    dataframe.to_csv(output_path, index=False)
    return dataframe


def build_clubelo_history_for_matches(
    matches,
    cache_dir=CLUBELO_CACHE_DIR,
    fetch_missing=True,
    session=None,
    force_refresh=False,
    sleep_seconds=0.0,
    limit_teams=None,
    max_retries=3,
    retry_backoff_seconds=5.0,
    fail_fast=False,
    failed_requests_path=FAILED_REQUESTS_PATH,
):
    teams = sorted(set(matches["HomeTeam"].dropna().astype(str)) | set(matches["AwayTeam"].dropna().astype(str)))
    if limit_teams is not None:
        teams = teams[: int(limit_teams)]
    frames = []
    failures = []
    for team in teams:
        try:
            history, _ = cached_clubelo_history(
                team,
                cache_dir=cache_dir,
                fetch_missing=fetch_missing,
                session=session,
                force_refresh=force_refresh,
                sleep_seconds=sleep_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        except Exception as error:
            if fail_fast:
                raise
            failures.append(failed_request_row(team, error))
            continue
        if len(history) == 0:
            continue
        history = history.copy()
        history["requested_team"] = team
        frames.append(history)
    write_failed_requests(failures, failed_requests_path)
    if not frames:
        return pd.DataFrame(columns=["Date", "team", "elo", "requested_team"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["requested_team", "Date"], keep="last")


def latest_pre_match_elo(team, match_date, elo_history):
    date = normalize_date(match_date)
    team_history = elo_history[
        (elo_history["requested_team"].astype(str) == str(team)) & (pd.to_datetime(elo_history["Date"]) < date)
    ]
    if len(team_history) == 0:
        return np.nan
    return float(team_history.sort_values("Date").iloc[-1]["elo"])


def build_prematch_elo_table(matches, elo_history):
    dataframe = matches[MATCH_KEY_COLUMNS].copy()
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce").dt.normalize()
    history = elo_history.copy()
    if len(history) == 0:
        dataframe["home_elo"] = np.nan
        dataframe["away_elo"] = np.nan
        dataframe["has_home_elo"] = False
        dataframe["has_away_elo"] = False
        dataframe["has_both_elo"] = False
        return dataframe
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce").dt.normalize()
    dataframe["home_elo"] = dataframe.apply(lambda row: latest_pre_match_elo(row["HomeTeam"], row["Date"], history), axis=1)
    dataframe["away_elo"] = dataframe.apply(lambda row: latest_pre_match_elo(row["AwayTeam"], row["Date"], history), axis=1)
    dataframe["has_home_elo"] = dataframe["home_elo"].notna()
    dataframe["has_away_elo"] = dataframe["away_elo"].notna()
    dataframe["has_both_elo"] = dataframe["has_home_elo"] & dataframe["has_away_elo"]
    return dataframe


def load_pitch_surface_table(path=PITCH_SURFACE_PATH):
    if not Path(path).exists():
        return pd.DataFrame(columns=["team", "stadium", "pitch_surface", "source_note"])
    return validate_pitch_surface_table(pd.read_csv(path))


def validate_pitch_surface_table(surfaces):
    required = {"team", "pitch_surface"}
    missing = required - set(surfaces.columns)
    if missing:
        raise ValueError(f"Pitch surface table missing columns: {sorted(missing)}")

    dataframe = surfaces.copy()
    dataframe["pitch_surface"] = dataframe["pitch_surface"].fillna("unknown").astype(str).str.strip().str.lower()
    invalid = sorted(set(dataframe["pitch_surface"]) - ALLOWED_PITCH_SURFACES)
    if invalid:
        allowed = ", ".join(sorted(ALLOWED_PITCH_SURFACES))
        raise ValueError(f"Invalid pitch_surface values: {invalid}. Allowed values: {allowed}")
    return dataframe


def add_pitch_surface_features(matches, surfaces):
    dataframe = matches.copy()
    surface_frame = validate_pitch_surface_table(surfaces)
    if "stadium" not in surface_frame.columns:
        surface_frame["stadium"] = ""
    if "source_note" not in surface_frame.columns:
        surface_frame["source_note"] = ""
    surface_frame["team"] = surface_frame["team"].astype(str).str.strip()
    surface_frame = surface_frame.drop_duplicates("team", keep="last")
    home = surface_frame.add_prefix("home_").rename(columns={"home_team": "HomeTeam"})
    away = surface_frame.add_prefix("away_").rename(columns={"away_team": "AwayTeam"})
    output = dataframe.merge(home, on="HomeTeam", how="left").merge(away, on="AwayTeam", how="left")
    home_surface = output["home_pitch_surface"].astype(str).str.strip().str.lower()
    away_surface = output["away_pitch_surface"].astype(str).str.strip().str.lower()
    output["has_home_pitch_surface"] = output["home_pitch_surface"].notna() & ~home_surface.isin(["", "nan", "unknown"])
    output["has_away_pitch_surface"] = output["away_pitch_surface"].notna() & ~away_surface.isin(["", "nan", "unknown"])
    output["same_pitch_surface"] = (
        output["has_home_pitch_surface"]
        & output["has_away_pitch_surface"]
        & (output["home_pitch_surface"] == output["away_pitch_surface"])
    )
    return output


def build_pitch_surface_unknown_teams(matches, surfaces):
    teams = []
    for side_column in ["HomeTeam", "AwayTeam"]:
        frame = matches[["league", side_column]].rename(columns={side_column: "team"}).copy()
        teams.append(frame)
    team_leagues = pd.concat(teams, ignore_index=True).dropna(subset=["league", "team"]).drop_duplicates()

    surface_frame = validate_pitch_surface_table(surfaces)
    surface_frame["team"] = surface_frame["team"].astype(str).str.strip()
    surface_frame = surface_frame.drop_duplicates("team", keep="last")
    output = team_leagues.merge(surface_frame[["team", "stadium", "pitch_surface", "source_note"]], on="team", how="left")
    output["pitch_surface"] = output["pitch_surface"].fillna("unknown")
    output["unknown_reason"] = output["source_note"].fillna("missing from manual surface table")
    return output[output["pitch_surface"].eq("unknown")].sort_values(["league", "team"]).reset_index(drop=True)


def write_pitch_surface_unknown_report(matches, surfaces, path=PITCH_SURFACE_UNKNOWN_REPORT_PATH):
    unknown = build_pitch_surface_unknown_teams(matches, surfaces)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Pitch Surface Unknown Teams",
        "",
        "Teams listed here either have `pitch_surface = unknown` in `data/manual/stadium_surface_overrides.csv` or are missing from that manual table.",
        "",
    ]
    if len(unknown) == 0:
        lines.append("No unknown pitch surfaces found.")
    else:
        for league, group in unknown.groupby("league", sort=True):
            lines.extend([f"## {league}", "", "| Team | Stadium | Reason |", "| --- | --- | --- |"])
            for _, row in group.iterrows():
                stadium = "" if pd.isna(row.get("stadium")) else str(row.get("stadium"))
                reason = "" if pd.isna(row.get("unknown_reason")) else str(row.get("unknown_reason"))
                lines.append(f"| {row['team']} | {stadium} | {reason} |")
            lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return unknown


def coverage_row(name, dataframe, covered_mask):
    total = int(len(dataframe))
    covered = int(covered_mask.sum()) if total else 0
    return {"scope": name, "rows": total, "covered_rows": covered, "coverage_rate": covered / total if total else 0.0}


def write_coverage_reports(matches, weather=None, normals=None, elo=None, surfaces=None, report_dir=OUTPUTS_DIR / "reports"):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if weather is None:
        weather = pd.DataFrame(columns=MATCH_KEY_COLUMNS + ["temperature_c", "wind_speed_kph", "precipitation_mm"])
    weather_joined = matches[MATCH_KEY_COLUMNS].copy()
    if len(weather) > 0:
        weather_joined["Date"] = pd.to_datetime(weather_joined["Date"], errors="coerce").dt.normalize()
        weather_frame = weather.copy()
        weather_frame["Date"] = pd.to_datetime(weather_frame["Date"], errors="coerce").dt.normalize()
        weather_joined = weather_joined.merge(
            weather_frame[MATCH_KEY_COLUMNS + [c for c in ["temperature_c", "wind_speed_kph", "precipitation_mm"] if c in weather_frame.columns]],
            on=MATCH_KEY_COLUMNS,
            how="left",
        )
    if "temperature_c" in weather_joined.columns:
        weather_mask = weather_joined["temperature_c"].notna()
    else:
        weather_mask = pd.Series(False, index=weather_joined.index)
    pd.DataFrame([coverage_row("match_weather", weather_joined, weather_mask)]).to_csv(
        report_dir / "weather_coverage.csv", index=False
    )

    teams = sorted(set(matches["HomeTeam"].dropna().astype(str)) | set(matches["AwayTeam"].dropna().astype(str)))
    if normals is None:
        normals = pd.DataFrame(columns=["team", "month", "avg_temp_c", "avg_wind_kph", "avg_precip_mm"])
    normal_pairs = pd.MultiIndex.from_product([teams, range(1, 13)], names=["team", "month"]).to_frame(index=False)
    normal_joined = normal_pairs.merge(normals, on=["team", "month"], how="left")
    if "avg_temp_c" in normal_joined.columns:
        normals_mask = normal_joined["avg_temp_c"].notna()
    else:
        normals_mask = pd.Series(False, index=normal_joined.index)
    pd.DataFrame([coverage_row("team_month_climate_normals", normal_joined, normals_mask)]).to_csv(
        report_dir / "climate_normals_coverage.csv", index=False
    )

    if elo is None:
        elo = pd.DataFrame(columns=MATCH_KEY_COLUMNS + ["home_elo", "away_elo"])
    elo_mask = pd.Series(False, index=matches.index)
    if len(elo) > 0 and {"home_elo", "away_elo"}.issubset(elo.columns):
        elo_mask = elo["home_elo"].notna() & elo["away_elo"].notna()
    pd.DataFrame([coverage_row("prematch_clubelo", matches, elo_mask)]).to_csv(report_dir / "clubelo_coverage.csv", index=False)

    if surfaces is None:
        surfaces = pd.DataFrame(columns=["team", "pitch_surface"])
    surface_joined = add_pitch_surface_features(matches[MATCH_KEY_COLUMNS], surfaces)
    surface_mask = surface_joined["has_home_pitch_surface"] & surface_joined["has_away_pitch_surface"]
    pd.DataFrame([coverage_row("pitch_surface", surface_joined, surface_mask)]).to_csv(
        report_dir / "pitch_surface_coverage.csv", index=False
    )
