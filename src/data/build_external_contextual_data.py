import argparse
from pathlib import Path

import pandas as pd

from src.common.paths import get_league_matches_path
from src.data.external_contextual_data import CLIMATE_NORMALS_PATH
from src.data.external_contextual_data import CLUBELO_CACHE_DIR
from src.data.external_contextual_data import CLUBELO_TABLE_PATH
from src.data.external_contextual_data import PITCH_SURFACE_PATH
from src.data.external_contextual_data import WEATHER_CACHE_DIR
from src.data.external_contextual_data import WEATHER_RANGE_CACHE_DIR
from src.data.external_contextual_data import WEATHER_TABLE_PATH
from src.data.external_contextual_data import build_clubelo_history_for_matches
from src.data.external_contextual_data import build_historical_weather_table
from src.data.external_contextual_data import build_historical_weather_table_range
from src.data.external_contextual_data import build_monthly_climate_normals
from src.data.external_contextual_data import build_prematch_elo_table
from src.data.external_contextual_data import cache_identifier
from src.data.external_contextual_data import clubelo_cache_path
from src.data.external_contextual_data import load_pitch_surface_table
from src.data.external_contextual_data import normalize_date
from src.data.external_contextual_data import weather_cache_path
from src.data.external_contextual_data import weather_range_cache_path
from src.data.external_contextual_data import write_coverage_reports
from src.data.external_contextual_data import write_pitch_surface_unknown_report
from src.features.travel_features import build_travel_features


DEFAULT_LEAGUES = ["E0", "E1", "E2", "E3", "SP1", "I1", "D1", "F1"]


def load_matches(leagues, start_season=None, end_season=None, limit_matches=None):
    frames = []
    for league in leagues:
        path = get_league_matches_path(league)
        if not path.exists():
            continue
        dataframe = pd.read_csv(path, low_memory=False)
        dataframe["league"] = league
        frames.append(dataframe)
    if not frames:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "league"])
    matches = pd.concat(frames, ignore_index=True)
    if "season_end_year" in matches.columns:
        matches["season_end_year"] = pd.to_numeric(matches["season_end_year"], errors="coerce")
        if start_season is not None:
            matches = matches[matches["season_end_year"] >= int(start_season)].copy()
        if end_season is not None:
            matches = matches[matches["season_end_year"] <= int(end_season)].copy()
    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
    matches = matches.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    if limit_matches is not None:
        matches = matches.head(int(limit_matches)).copy()
    return matches


def load_coordinates():
    stadium_path = Path("data/external/stadiums/stadiums_with_gps_coordinates.csv")
    overrides_path = Path("data/manual/team_stadium_overrides.csv")
    coordinates = pd.read_csv(stadium_path) if stadium_path.exists() else pd.DataFrame(columns=["team", "latitude", "longitude"])
    overrides = pd.read_csv(overrides_path) if overrides_path.exists() else None
    return coordinates, overrides


def maybe_load_csv(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


def selected_season_label(matches, start_season=None, end_season=None):
    if start_season is not None or end_season is not None:
        return f"{start_season or 'min'}-{end_season or 'max'}"
    if "season_end_year" not in matches.columns or len(matches) == 0:
        return "none"
    seasons = pd.to_numeric(matches["season_end_year"], errors="coerce").dropna().astype(int)
    if len(seasons) == 0:
        return "none"
    return f"{seasons.min()}-{seasons.max()}"


def build_weather_request_plan(
    matches,
    coordinates,
    overrides=None,
    cache_dir=WEATHER_CACHE_DIR,
    only_missing=True,
    force_refresh=False,
):
    if len(matches) == 0:
        return pd.DataFrame(
            columns=["Date", "team", "stadium", "latitude", "longitude", "cache_path", "is_cached", "should_fetch"]
        )
    with_coordinates = build_travel_features(matches, coordinates, overrides)
    rows = []
    for _, row in with_coordinates.iterrows():
        if not bool(row.get("has_home_coordinates", False)):
            continue
        date = normalize_date(row["Date"])
        if pd.isna(date):
            continue
        identifier = cache_identifier(row)
        rows.append(
            {
                "Date": date,
                "team": row["HomeTeam"],
                "stadium": row.get("home_stadium", ""),
                "cache_identifier": identifier,
                "latitude": row["home_latitude"],
                "longitude": row["home_longitude"],
                "request_key": (
                    date.strftime("%Y-%m-%d"),
                    identifier,
                    round(float(row["home_latitude"]), 6),
                    round(float(row["home_longitude"]), 6),
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["Date", "team", "stadium", "latitude", "longitude", "cache_path", "is_cached", "should_fetch"]
        )
    plan = pd.DataFrame(rows).drop_duplicates("request_key", keep="first").reset_index(drop=True)
    plan["cache_path"] = plan.apply(lambda row: weather_cache_path(cache_dir, row["cache_identifier"], row["Date"]), axis=1)
    plan["is_cached"] = plan["cache_path"].map(lambda path: Path(path).exists())
    plan["should_fetch"] = (~plan["is_cached"]) | bool(force_refresh)
    if only_missing and not force_refresh:
        plan["should_fetch"] = ~plan["is_cached"]
    return plan


def build_weather_range_request_plan(
    matches,
    coordinates,
    overrides=None,
    cache_dir=WEATHER_RANGE_CACHE_DIR,
    only_missing=True,
    force_refresh=False,
):
    if len(matches) == 0:
        return pd.DataFrame(
            columns=[
                "start_date",
                "end_date",
                "team",
                "stadium",
                "latitude",
                "longitude",
                "cache_path",
                "is_cached",
                "should_fetch",
            ]
        )
    with_coordinates = build_travel_features(matches, coordinates, overrides)
    rows = []
    for _, row in with_coordinates.iterrows():
        if not bool(row.get("has_home_coordinates", False)):
            continue
        date = normalize_date(row["Date"])
        if pd.isna(date):
            continue
        identifier = cache_identifier(row)
        rows.append(
            {
                "Date": date,
                "team": row["HomeTeam"],
                "stadium": row.get("home_stadium", ""),
                "cache_identifier": identifier,
                "latitude": row["home_latitude"],
                "longitude": row["home_longitude"],
                "range_key": (
                    identifier,
                    round(float(row["home_latitude"]), 6),
                    round(float(row["home_longitude"]), 6),
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["start_date", "end_date", "team", "stadium", "latitude", "longitude", "cache_path", "is_cached", "should_fetch"]
        )
    base = pd.DataFrame(rows)
    plan_rows = []
    for _, group in base.groupby("range_key", sort=False):
        first = group.iloc[0]
        start_date = group["Date"].min()
        end_date = group["Date"].max()
        plan_rows.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "team": first["team"],
                "stadium": first["stadium"],
                "cache_identifier": first["cache_identifier"],
                "latitude": first["latitude"],
                "longitude": first["longitude"],
            }
        )
    plan = pd.DataFrame(plan_rows)
    plan["cache_path"] = plan.apply(
        lambda row: weather_range_cache_path(cache_dir, row["cache_identifier"], row["start_date"], row["end_date"]),
        axis=1,
    )
    plan["is_cached"] = plan["cache_path"].map(lambda path: Path(path).exists())
    plan["should_fetch"] = (~plan["is_cached"]) | bool(force_refresh)
    if only_missing and not force_refresh:
        plan["should_fetch"] = ~plan["is_cached"]
    return plan


def build_clubelo_request_plan(matches, cache_dir=CLUBELO_CACHE_DIR, limit_teams=None, only_missing=True, force_refresh=False):
    teams = sorted(set(matches.get("HomeTeam", pd.Series(dtype=str)).dropna().astype(str)) | set(matches.get("AwayTeam", pd.Series(dtype=str)).dropna().astype(str)))
    if limit_teams is not None:
        teams = teams[: int(limit_teams)]
    plan = pd.DataFrame({"team": teams})
    if len(plan) == 0:
        return pd.DataFrame(columns=["team", "cache_path", "is_cached", "should_fetch"])
    plan["cache_path"] = plan["team"].map(lambda team: clubelo_cache_path(cache_dir, team))
    plan["is_cached"] = plan["cache_path"].map(lambda path: Path(path).exists())
    plan["should_fetch"] = (~plan["is_cached"]) | bool(force_refresh)
    if only_missing and not force_refresh:
        plan["should_fetch"] = ~plan["is_cached"]
    return plan


def print_fetch_plan(args, matches, weather_plan, clubelo_plan, match_weather_plan=None, range_weather_plan=None):
    print("External contextual data plan")
    print(f"selected_leagues: {','.join([league.upper() for league in args.leagues])}")
    print(f"selected_seasons: {selected_season_label(matches, args.start_season, args.end_season)}")
    print(f"matches: {len(matches)}")
    print(f"weather_mode: {args.weather_mode}")
    print(f"unique_weather_requests: {len(weather_plan)}")
    if match_weather_plan is not None and range_weather_plan is not None:
        match_count = len(match_weather_plan)
        range_count = len(range_weather_plan)
        reduction = match_count - range_count
        reduction_rate = reduction / match_count if match_count else 0.0
        print(f"match_mode_weather_requests: {match_count}")
        print(f"range_mode_weather_requests: {range_count}")
        print(f"weather_request_reduction: {reduction}")
        print(f"weather_request_reduction_rate: {reduction_rate:.4f}")
    print(f"clubelo_team_history_requests: {len(clubelo_plan)}")
    print(f"weather_already_cached: {int(weather_plan['is_cached'].sum()) if len(weather_plan) else 0}")
    print(f"weather_missing: {int((~weather_plan['is_cached']).sum()) if len(weather_plan) else 0}")
    print(f"clubelo_already_cached: {int(clubelo_plan['is_cached'].sum()) if len(clubelo_plan) else 0}")
    print(f"clubelo_missing: {int((~clubelo_plan['is_cached']).sum()) if len(clubelo_plan) else 0}")
    if args.dry_run:
        weather_fetches = weather_plan[weather_plan["should_fetch"]]
        clubelo_fetches = clubelo_plan[clubelo_plan["should_fetch"]]
        print("dry_run: true")
        print("would_fetch_weather:")
        weather_columns = ["team", "stadium", "latitude", "longitude"]
        if args.weather_mode == "range":
            weather_columns = ["start_date", "end_date"] + weather_columns
        else:
            weather_columns = ["Date"] + weather_columns
        print(weather_fetches[weather_columns].head(20).to_string(index=False))
        print("would_fetch_clubelo:")
        print(clubelo_fetches[["team"]].head(50).to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    parser.add_argument("--fetch-weather", action="store_true", help="Call Open-Meteo for missing weather cache entries.")
    parser.add_argument("--fetch-clubelo", action="store_true", help="Call ClubElo for missing team history cache entries.")
    parser.add_argument("--weather-mode", choices=["match", "range"], default="range")
    parser.add_argument("--start-season", type=int)
    parser.add_argument("--end-season", type=int)
    parser.add_argument("--limit-matches", type=int)
    parser.add_argument("--limit-teams", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--force-refresh", action="store_true", help="Overwrite existing provider cache entries.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    matches = load_matches(
        [league.upper() for league in args.leagues],
        start_season=args.start_season,
        end_season=args.end_season,
        limit_matches=args.limit_matches,
    )
    coordinates, overrides = load_coordinates()
    match_weather_plan = build_weather_request_plan(
        matches,
        coordinates,
        overrides,
        only_missing=args.only_missing,
        force_refresh=args.force_refresh,
    )
    range_weather_plan = build_weather_range_request_plan(
        matches,
        coordinates,
        overrides,
        only_missing=args.only_missing,
        force_refresh=args.force_refresh,
    )
    weather_plan = range_weather_plan if args.weather_mode == "range" else match_weather_plan
    clubelo_plan = build_clubelo_request_plan(
        matches,
        limit_teams=args.limit_teams,
        only_missing=args.only_missing,
        force_refresh=args.force_refresh,
    )
    print_fetch_plan(
        args,
        matches,
        weather_plan,
        clubelo_plan,
        match_weather_plan=match_weather_plan,
        range_weather_plan=range_weather_plan,
    )

    if args.dry_run:
        return

    if args.weather_mode == "range":
        weather = build_historical_weather_table_range(
            matches,
            coordinates,
            manual_overrides=overrides,
            fetch_missing=args.fetch_weather,
            force_refresh=args.force_refresh,
            sleep_seconds=args.sleep_seconds,
        )
    else:
        weather = build_historical_weather_table(
            matches,
            coordinates,
            manual_overrides=overrides,
            fetch_missing=args.fetch_weather,
            force_refresh=args.force_refresh,
            sleep_seconds=args.sleep_seconds,
        )
    WEATHER_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(WEATHER_TABLE_PATH, index=False)

    normals = build_monthly_climate_normals(weather)
    CLIMATE_NORMALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    normals.to_csv(CLIMATE_NORMALS_PATH, index=False)

    if args.fetch_clubelo:
        elo_history = build_clubelo_history_for_matches(
            matches,
            fetch_missing=True,
            force_refresh=args.force_refresh,
            sleep_seconds=args.sleep_seconds,
            limit_teams=args.limit_teams,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            fail_fast=args.fail_fast,
        )
        prematch_elo = build_prematch_elo_table(matches, elo_history)
        CLUBELO_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        prematch_elo.to_csv(CLUBELO_TABLE_PATH, index=False)
    else:
        prematch_elo = maybe_load_csv(CLUBELO_TABLE_PATH)
        if prematch_elo is None:
            prematch_elo = build_prematch_elo_table(matches, pd.DataFrame(columns=["Date", "team", "elo", "requested_team"]))
            CLUBELO_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
            prematch_elo.to_csv(CLUBELO_TABLE_PATH, index=False)

    surfaces = load_pitch_surface_table(PITCH_SURFACE_PATH)
    write_coverage_reports(matches, weather=weather, normals=normals, elo=prematch_elo, surfaces=surfaces)
    write_pitch_surface_unknown_report(matches, surfaces)

    print(WEATHER_TABLE_PATH)
    print(CLIMATE_NORMALS_PATH)
    print(CLUBELO_TABLE_PATH)
    print(PITCH_SURFACE_PATH)


if __name__ == "__main__":
    main()
