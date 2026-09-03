import json

import pandas as pd
import requests

from src.data import build_external_contextual_data as builder
from src.data.external_contextual_data import add_pitch_surface_features
from src.data.external_contextual_data import build_clubelo_history_for_matches
from src.data.external_contextual_data import cached_clubelo_history
from src.data.external_contextual_data import build_historical_weather_table
from src.data.external_contextual_data import build_historical_weather_table_range
from src.data.external_contextual_data import build_monthly_climate_normals
from src.data.external_contextual_data import build_prematch_elo_table
from src.data.external_contextual_data import build_pitch_surface_unknown_teams
from src.data.external_contextual_data import cached_open_meteo_weather
from src.data.external_contextual_data import fetch_clubelo_team_history
from src.data.external_contextual_data import latest_pre_match_elo
from src.data.external_contextual_data import validate_pitch_surface_table
from src.data.external_contextual_data import write_pitch_surface_unknown_report
from src.features.contextual_features import add_elo_market_disagreement_features


class FakeWeatherResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeWeatherSession:
    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, timeout=30):
        self.calls += 1
        return FakeWeatherResponse(
            {
                "daily": {
                    "time": [params["start_date"]],
                    "temperature_2m_mean": [12.5],
                    "precipitation_sum": [1.2],
                    "wind_speed_10m_max": [22.0],
                    "weather_code": [3],
                }
            }
        )


class FakeClubEloResponse:
    status_code = 200
    text = "From,Club,Elo\n2024-01-01,A,1500\n"

    def raise_for_status(self):
        return None


class FakeClubEloSession:
    def __init__(self):
        self.calls = 0

    def get(self, url, timeout=30, headers=None):
        self.calls += 1
        self.last_headers = headers
        return FakeClubEloResponse()


class SequenceClubEloSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.urls = []
        self.headers = []

    def get(self, url, timeout=30, headers=None):
        self.calls += 1
        self.urls.append(url)
        self.headers.append(headers)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def sample_matches():
    return pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "HomeTeam": ["A", "B"],
            "AwayTeam": ["B", "A"],
            "AvgH": [2.0, 2.0],
        }
    )


def test_open_meteo_cache_prevents_repeated_api_calls(tmp_path):
    session = FakeWeatherSession()
    first, first_hit = cached_open_meteo_weather("A", 51.5, -0.1, "2024-01-02", tmp_path, True, session)
    second, second_hit = cached_open_meteo_weather("A", 51.5, -0.1, "2024-01-02", tmp_path, True, session)

    assert first["temperature_c"] == 12.5
    assert second["wind_speed_kph"] == 22.0
    assert not first_hit
    assert second_hit
    assert session.calls == 1


def test_open_meteo_force_refresh_allows_refetch(tmp_path):
    session = FakeWeatherSession()
    cached_open_meteo_weather("A", 51.5, -0.1, "2024-01-02", tmp_path, True, session)
    cached_open_meteo_weather("A", 51.5, -0.1, "2024-01-02", tmp_path, True, session, force_refresh=True)
    assert session.calls == 2


def test_clubelo_cached_records_are_skipped(tmp_path):
    session = FakeClubEloSession()
    cached_clubelo_history("A", tmp_path, True, session)
    cached_clubelo_history("A", tmp_path, True, session)
    assert session.calls == 1


def test_clubelo_force_refresh_allows_refetch(tmp_path):
    session = FakeClubEloSession()
    cached_clubelo_history("A", tmp_path, True, session)
    cached_clubelo_history("A", tmp_path, True, session, force_refresh=True)
    assert session.calls == 2


def test_clubelo_timeout_then_success_retries_with_user_agent():
    session = SequenceClubEloSession([requests.exceptions.ConnectTimeout("timeout"), FakeClubEloResponse()])
    text = fetch_clubelo_team_history("A", session=session, max_retries=1, retry_backoff_seconds=0)
    assert "Elo" in text
    assert session.calls == 2
    assert session.headers[0]["User-Agent"].startswith("FootballV2-ruflo-contextual-data")


def test_clubelo_repeated_timeout_writes_failed_request(tmp_path):
    session = SequenceClubEloSession(
        [requests.exceptions.ConnectTimeout("timeout"), requests.exceptions.ConnectTimeout("timeout")]
    )
    matches = pd.DataFrame({"HomeTeam": ["A"], "AwayTeam": ["A"]})
    history = build_clubelo_history_for_matches(
        matches,
        cache_dir=tmp_path / "cache",
        session=session,
        max_retries=1,
        retry_backoff_seconds=0,
        failed_requests_path=tmp_path / "failed.csv",
    )
    failures = pd.read_csv(tmp_path / "failed.csv")
    assert len(history) == 0
    assert failures.loc[0, "team"] == "A"
    assert failures.loc[0, "error_type"] == "ConnectTimeout"


def test_clubelo_fail_fast_raises(tmp_path):
    session = SequenceClubEloSession([requests.exceptions.ReadTimeout("timeout")])
    matches = pd.DataFrame({"HomeTeam": ["A"], "AwayTeam": ["A"]})
    try:
        build_clubelo_history_for_matches(
            matches,
            cache_dir=tmp_path / "cache",
            session=session,
            max_retries=0,
            retry_backoff_seconds=0,
            fail_fast=True,
            failed_requests_path=tmp_path / "failed.csv",
        )
    except requests.exceptions.ReadTimeout:
        pass
    else:
        raise AssertionError("Expected fail-fast to raise")


def test_clubelo_cached_team_does_not_retry(tmp_path):
    cache_path = tmp_path / "a.csv"
    cache_path.write_text("Date,team,elo,requested_team\n2024-01-01,A,1500,A\n", encoding="utf-8")
    session = SequenceClubEloSession([requests.exceptions.ConnectTimeout("timeout")])
    history, cache_hit = cached_clubelo_history("A", tmp_path, True, session, max_retries=3, retry_backoff_seconds=0)
    assert cache_hit
    assert len(history) == 1
    assert session.calls == 0


def test_clubelo_failed_team_does_not_block_other_teams(tmp_path):
    session = SequenceClubEloSession(
        [
            requests.exceptions.ConnectTimeout("timeout"),
            requests.exceptions.ConnectTimeout("timeout"),
            FakeClubEloResponse(),
        ]
    )
    matches = pd.DataFrame({"HomeTeam": ["A"], "AwayTeam": ["B"]})
    history = build_clubelo_history_for_matches(
        matches,
        cache_dir=tmp_path / "cache",
        session=session,
        max_retries=1,
        retry_backoff_seconds=0,
        failed_requests_path=tmp_path / "failed.csv",
    )
    failures = pd.read_csv(tmp_path / "failed.csv")
    assert failures["team"].tolist() == ["A"]
    assert history["requested_team"].tolist() == ["B"]


def test_cached_weather_missing_data_safe_without_fetch(tmp_path):
    values, cache_hit = cached_open_meteo_weather("Missing", 51.5, -0.1, "2024-01-02", tmp_path, False)
    assert values == {}
    assert not cache_hit


def test_weather_ingestion_joins_by_match_date_and_home_team(tmp_path):
    cache_payload = {
        "daily": {
            "time": ["2024-01-02"],
            "temperature_2m_mean": [7.0],
            "precipitation_sum": [0.0],
            "wind_speed_10m_max": [15.0],
            "weather_code": [1],
        }
    }
    cache_file = tmp_path / "2024-01-02_a.json"
    cache_file.write_text(json.dumps(cache_payload), encoding="utf-8")
    matches = sample_matches()
    coordinates = pd.DataFrame({"team": ["A", "B"], "latitude": [51.5, 52.5], "longitude": [-0.1, -1.1]})

    output = build_historical_weather_table(matches, coordinates, cache_dir=tmp_path, fetch_missing=False)

    first = output[output["HomeTeam"] == "A"].iloc[0]
    second = output[output["HomeTeam"] == "B"].iloc[0]
    assert first["temperature_c"] == 7.0
    assert pd.isna(second.get("temperature_c"))


def test_weather_requests_are_deduplicated_by_stadium_date(tmp_path):
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-02"],
            "HomeTeam": ["A", "C"],
            "AwayTeam": ["B", "D"],
        }
    )
    coordinates = pd.DataFrame(
        {
            "Team": ["A", "C"],
            "Stadium": ["Shared Park", "Shared Park"],
            "Latitude": [51.5, 51.5],
            "Longitude": [-0.1, -0.1],
        }
    )
    plan = builder.build_weather_request_plan(matches, coordinates, cache_dir=tmp_path)
    assert len(plan) == 1


def test_weather_range_plan_reduces_requests_by_stadium_range(tmp_path):
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "HomeTeam": ["A", "A", "B"],
            "AwayTeam": ["B", "C", "A"],
        }
    )
    coordinates = pd.DataFrame(
        {
            "Team": ["A", "B"],
            "Stadium": ["A Park", "B Park"],
            "Latitude": [51.5, 52.5],
            "Longitude": [-0.1, -1.1],
        }
    )
    match_plan = builder.build_weather_request_plan(matches, coordinates, cache_dir=tmp_path / "match")
    range_plan = builder.build_weather_range_request_plan(matches, coordinates, cache_dir=tmp_path / "range")
    assert len(match_plan) == 3
    assert len(range_plan) == 2
    a_row = range_plan[range_plan["stadium"] == "A Park"].iloc[0]
    assert str(a_row["start_date"].date()) == "2024-01-02"
    assert str(a_row["end_date"].date()) == "2024-01-03"


def test_weather_range_cache_derives_match_level_weather(tmp_path):
    range_payload = {
        "daily": {
            "time": ["2024-01-02", "2024-01-03"],
            "temperature_2m_mean": [7.0, 8.0],
            "precipitation_sum": [0.0, 1.0],
            "wind_speed_10m_max": [15.0, 16.0],
            "weather_code": [1, 2],
        }
    }
    (tmp_path / "range" / "a-park_2024-01-02_2024-01-03.json").parent.mkdir(parents=True)
    (tmp_path / "range" / "a-park_2024-01-02_2024-01-03.json").write_text(json.dumps(range_payload), encoding="utf-8")
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "HomeTeam": ["A", "A"],
            "AwayTeam": ["B", "C"],
        }
    )
    coordinates = pd.DataFrame({"Team": ["A"], "Stadium": ["A Park"], "Latitude": [51.5], "Longitude": [-0.1]})
    output = build_historical_weather_table_range(
        matches,
        coordinates,
        range_cache_dir=tmp_path / "range",
        match_cache_dir=tmp_path / "match",
        fetch_missing=False,
    )
    assert output["temperature_c"].tolist() == [7.0, 8.0]
    assert output["weather_range_cache_hit"].tolist() == [True, True]


def test_weather_range_mode_falls_back_to_existing_match_cache(tmp_path):
    match_payload = {
        "daily": {
            "time": ["2024-01-02"],
            "temperature_2m_mean": [9.0],
            "precipitation_sum": [0.5],
            "wind_speed_10m_max": [12.0],
            "weather_code": [3],
        }
    }
    (tmp_path / "match").mkdir()
    (tmp_path / "match" / "2024-01-02_a-park.json").write_text(json.dumps(match_payload), encoding="utf-8")
    matches = pd.DataFrame({"Date": ["2024-01-02"], "HomeTeam": ["A"], "AwayTeam": ["B"]})
    coordinates = pd.DataFrame({"Team": ["A"], "Stadium": ["A Park"], "Latitude": [51.5], "Longitude": [-0.1]})
    output = build_historical_weather_table_range(
        matches,
        coordinates,
        range_cache_dir=tmp_path / "range",
        match_cache_dir=tmp_path / "match",
        fetch_missing=False,
    )
    assert output.loc[0, "temperature_c"] == 9.0
    assert output.loc[0, "weather_cache_hit"]


def test_clubelo_requests_are_deduplicated_by_team(tmp_path):
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "HomeTeam": ["A", "A"],
            "AwayTeam": ["B", "B"],
        }
    )
    plan = builder.build_clubelo_request_plan(matches, cache_dir=tmp_path)
    assert plan["team"].tolist() == ["A", "B"]


def test_builder_limit_teams_works(tmp_path):
    matches = pd.DataFrame({"HomeTeam": ["C", "A"], "AwayTeam": ["B", "D"]})
    plan = builder.build_clubelo_request_plan(matches, cache_dir=tmp_path, limit_teams=2)
    assert len(plan) == 2
    assert plan["team"].tolist() == ["A", "B"]


def test_builder_limit_matches_works(tmp_path, monkeypatch):
    path = tmp_path / "E0_matches.csv"
    pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "HomeTeam": ["C", "A", "B"],
            "AwayTeam": ["D", "B", "C"],
            "season_end_year": [2025, 2025, 2025],
        }
    ).to_csv(path, index=False)
    monkeypatch.setattr(builder, "get_league_matches_path", lambda league: path)
    matches = builder.load_matches(["E0"], limit_matches=2)
    assert len(matches) == 2
    assert matches["HomeTeam"].tolist() == ["A", "B"]


def test_builder_cached_records_are_skipped_in_plan(tmp_path):
    cache_payload = {
        "daily": {
            "time": ["2024-01-02"],
            "temperature_2m_mean": [7.0],
            "precipitation_sum": [0.0],
            "wind_speed_10m_max": [15.0],
            "weather_code": [1],
        }
    }
    (tmp_path / "2024-01-02_a-park.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    matches = pd.DataFrame({"Date": ["2024-01-02"], "HomeTeam": ["A"], "AwayTeam": ["B"]})
    coordinates = pd.DataFrame({"Team": ["A"], "Stadium": ["A Park"], "Latitude": [51.5], "Longitude": [-0.1]})
    plan = builder.build_weather_request_plan(matches, coordinates, cache_dir=tmp_path)
    assert plan.loc[0, "is_cached"]
    assert not plan.loc[0, "should_fetch"]


def test_builder_force_refresh_marks_cached_records_for_fetch(tmp_path):
    matches = pd.DataFrame({"HomeTeam": ["A"], "AwayTeam": ["B"]})
    cache_path = builder.clubelo_cache_path(tmp_path, "A")
    cache_path.write_text("Date,team,elo,requested_team\n2024-01-01,A,1500,A\n", encoding="utf-8")
    plan = builder.build_clubelo_request_plan(matches, cache_dir=tmp_path, force_refresh=True)
    a_row = plan[plan["team"] == "A"].iloc[0]
    assert a_row["is_cached"]
    assert a_row["should_fetch"]


def test_builder_dry_run_makes_zero_external_calls(monkeypatch):
    matches = pd.DataFrame({"Date": ["2024-01-02"], "HomeTeam": ["A"], "AwayTeam": ["B"], "season_end_year": [2025]})
    coordinates = pd.DataFrame({"Team": ["A"], "Latitude": [51.5], "Longitude": [-0.1]})
    monkeypatch.setattr(builder, "load_matches", lambda *args, **kwargs: matches)
    monkeypatch.setattr(builder, "load_coordinates", lambda: (coordinates, None))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry-run must not build/fetch provider data")

    monkeypatch.setattr(builder, "build_historical_weather_table", fail_if_called)
    monkeypatch.setattr(builder, "build_clubelo_history_for_matches", fail_if_called)
    builder.main(["--leagues", "E0", "--fetch-weather", "--fetch-clubelo", "--dry-run"])


def test_builder_dry_run_prints_range_request_reduction(monkeypatch, capsys):
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "HomeTeam": ["A", "A"],
            "AwayTeam": ["B", "C"],
            "season_end_year": [2025, 2025],
        }
    )
    coordinates = pd.DataFrame({"Team": ["A"], "Stadium": ["A Park"], "Latitude": [51.5], "Longitude": [-0.1]})
    monkeypatch.setattr(builder, "load_matches", lambda *args, **kwargs: matches)
    monkeypatch.setattr(builder, "load_coordinates", lambda: (coordinates, None))
    builder.main(["--leagues", "E0", "--dry-run"])
    output = capsys.readouterr().out
    assert "weather_mode: range" in output
    assert "match_mode_weather_requests: 2" in output
    assert "range_mode_weather_requests: 1" in output
    assert "weather_request_reduction: 1" in output


def test_climate_normals_use_requested_columns():
    weather = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2025-01-10"],
            "team": ["A", "A"],
            "temperature_c": [10.0, 14.0],
            "wind_speed_kph": [20.0, 30.0],
            "precipitation_mm": [1.0, 3.0],
        }
    )
    normals = build_monthly_climate_normals(weather)
    row = normals.iloc[0]
    assert row["team"] == "A"
    assert row["month"] == 1
    assert row["avg_temp_c"] == 12.0
    assert row["avg_wind_kph"] == 25.0
    assert row["avg_precip_mm"] == 2.0


def test_latest_pre_match_elo_uses_strictly_prior_date_only():
    history = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "requested_team": ["A", "A", "A"],
            "team": ["A", "A", "A"],
            "elo": [1500, 1600, 1700],
        }
    )
    assert latest_pre_match_elo("A", "2024-01-02", history) == 1500


def test_prematch_elo_table_handles_missing_data_safely():
    matches = sample_matches()
    output = build_prematch_elo_table(matches, pd.DataFrame(columns=["Date", "team", "elo", "requested_team"]))
    assert output["home_elo"].isna().all()
    assert not output["has_both_elo"].any()


def test_contextual_elo_join_does_not_use_same_day_rating():
    matches = pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "AvgH": [2.0],
            "AvgD": [3.0],
            "AvgA": [4.0],
        }
    )
    elo = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "team": ["A", "A"],
            "elo": [1500, 1700],
        }
    )
    output = add_elo_market_disagreement_features(matches, elo)
    assert output.loc[0, "home_elo"] == 1500
    assert pd.isna(output.loc[0, "away_elo"])


def test_pitch_surface_joins_via_manual_table():
    matches = sample_matches()
    surfaces = pd.DataFrame(
        {
            "team": ["A", "B"],
            "stadium": ["A Park", "B Park"],
            "pitch_surface": ["grass", "hybrid"],
            "source_note": ["manual", "manual"],
        }
    )
    output = add_pitch_surface_features(matches, surfaces)
    assert output.loc[0, "home_pitch_surface"] == "grass"
    assert output.loc[0, "away_pitch_surface"] == "hybrid"
    assert output.loc[0, "has_home_pitch_surface"]
    assert not output.loc[0, "same_pitch_surface"]


def test_pitch_surface_unknown_does_not_count_as_known():
    matches = sample_matches()
    surfaces = pd.DataFrame(
        {
            "team": ["A", "B"],
            "stadium": ["A Park", "B Park"],
            "pitch_surface": ["unknown", "grass"],
            "source_note": ["manual", "manual"],
        }
    )
    output = add_pitch_surface_features(matches, surfaces)
    assert not output.loc[0, "has_home_pitch_surface"]
    assert output.loc[0, "has_away_pitch_surface"]


def test_pitch_surface_allowed_values_validation_accepts_known_values():
    surfaces = pd.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "pitch_surface": ["grass", "artificial", "hybrid", "unknown"],
        }
    )
    output = validate_pitch_surface_table(surfaces)
    assert output["pitch_surface"].tolist() == ["grass", "artificial", "hybrid", "unknown"]


def test_pitch_surface_allowed_values_validation_rejects_invalid_value():
    surfaces = pd.DataFrame({"team": ["A"], "pitch_surface": ["mud"]})
    try:
        validate_pitch_surface_table(surfaces)
    except ValueError as error:
        assert "Invalid pitch_surface values" in str(error)
        assert "grass" in str(error)
    else:
        raise AssertionError("Expected invalid pitch surface rejection")


def test_pitch_surface_unknown_report_groups_by_league_and_missing_manual_rows(tmp_path):
    matches = pd.DataFrame(
        {
            "league": ["E0", "E0", "E1"],
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "HomeTeam": ["A", "B", "C"],
            "AwayTeam": ["B", "D", "A"],
        }
    )
    surfaces = pd.DataFrame(
        {
            "team": ["A", "B"],
            "stadium": ["A Park", "B Park"],
            "pitch_surface": ["unknown", "grass"],
            "source_note": ["needs verification", "manual"],
        }
    )
    unknown = build_pitch_surface_unknown_teams(matches, surfaces)
    assert unknown[["league", "team"]].values.tolist() == [["E0", "A"], ["E0", "D"], ["E1", "A"], ["E1", "C"]]

    report_path = tmp_path / "pitch_surface_unknown_teams.md"
    write_pitch_surface_unknown_report(matches, surfaces, report_path)
    report = report_path.read_text(encoding="utf-8")
    assert "## E0" in report
    assert "| A | A Park | needs verification |" in report
    assert "| D |  | missing from manual surface table |" in report
