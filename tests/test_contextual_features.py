import math

import pandas as pd

from src.features.contextual_features import add_market_disagreement_features
from src.features.contextual_features import assert_no_closing_columns
from src.features.contextual_features import build_contextual_features
from src.features.schedule_features import add_schedule_features
from src.features.travel_features import build_travel_features
from src.features.travel_features import combine_coordinate_sources
from src.features.travel_features import haversine_distance_km
from src.features.travel_features import match_team_coordinates
from src.features.weather_features import add_weather_features
from src.features.weather_features import add_weather_shock_features


def sample_matches():
    return pd.DataFrame(
        {
            "Date": ["2024-08-01", "2024-08-01", "2024-08-05", "2024-08-10"],
            "HomeTeam": ["A", "C", "A", "B"],
            "AwayTeam": ["B", "D", "C", "A"],
            "season_end_year": [2025, 2025, 2025, 2025],
            "AvgH": [2.0, 2.2, 1.9, 2.4],
            "AvgD": [3.2, 3.1, 3.3, 3.0],
            "AvgA": [4.0, 3.8, 4.1, 3.2],
            "MaxH": [2.1, 2.3, 2.0, 2.5],
            "MaxD": [3.3, 3.2, 3.4, 3.1],
            "MaxA": [4.2, 4.0, 4.3, 3.4],
            "Avg>2.5": [1.9, 2.0, 1.8, 2.1],
            "Avg<2.5": [1.95, 1.9, 2.0, 1.8],
            "AHh": [-0.5, 0.25, -1.0, 0.0],
            "AvgAHH": [1.9, 1.95, 1.85, 1.92],
            "AvgAHA": [1.95, 1.9, 2.0, 1.94],
        }
    )


def test_haversine_distance_london_to_manchester_roughly():
    distance = haversine_distance_km(51.5074, -0.1278, 53.4808, -2.2426)
    assert 260 <= distance <= 270


def test_schedule_features_do_not_use_same_day_matches():
    matches = sample_matches()
    output = add_schedule_features(matches)
    same_day = output[output["Date"] == pd.Timestamp("2024-08-01")]
    assert same_day["home_rest_days"].tolist() == [14, 14]
    assert same_day["away_rest_days"].tolist() == [14, 14]

    later = output[output["Date"] == pd.Timestamp("2024-08-10")].iloc[0]
    assert later["away_rest_days"] == 5


def test_schedule_features_are_deterministic():
    matches = sample_matches()
    first = add_schedule_features(matches)
    second = add_schedule_features(matches)
    pd.testing.assert_frame_equal(first, second)


def test_closing_columns_rejected():
    try:
        assert_no_closing_columns(["AvgH", "AvgCH"])
    except ValueError as error:
        assert "Closing odds" in str(error)
    else:
        raise AssertionError("Expected closing column rejection")


def test_market_disagreement_uses_main_odds():
    output = add_market_disagreement_features(sample_matches())
    assert "avg_1x2_AvgH_no_vig_probability" in output.columns
    assert "market_max_minus_avg_h_prob" in output.columns
    assert math.isfinite(output.loc[0, "avg_1x2_AvgH_no_vig_probability"])


def test_travel_features_with_reference_table():
    coordinates = pd.DataFrame(
        {
            "team": ["A", "B"],
            "latitude": [51.5074, 53.4808],
            "longitude": [-0.1278, -2.2426],
            "country": ["England", "England"],
        }
    )
    matches = pd.DataFrame({"Date": ["2024-01-01"], "HomeTeam": ["A"], "AwayTeam": ["B"]})
    output = build_travel_features(matches, coordinates)
    assert output.loc[0, "has_both_coordinates"]
    assert 260 <= output.loc[0, "travel_distance_km"] <= 270


def test_travel_features_accept_common_coordinate_aliases():
    coordinates = pd.DataFrame(
        {
            "Team": ["A", "B"],
            "Latitude": [51.5074, 53.4808],
            "Longitude": [-0.1278, -2.2426],
            "Country": ["England", "England"],
        }
    )
    matches = pd.DataFrame({"Date": ["2024-01-01"], "HomeTeam": ["A"], "AwayTeam": ["B"]})
    output = build_travel_features(matches, coordinates)
    assert output.loc[0, "has_both_coordinates"]


def test_travel_features_match_fd_name_alias_column():
    coordinates = pd.DataFrame(
        {
            "Team": ["Manchester United FC", "Liverpool FC"],
            "FDCOUK": ["Man United", "Liverpool"],
            "Latitude": [53.4631, 53.4308],
            "Longitude": [-2.2913, -2.9608],
            "Country": ["England", "England"],
        }
    )
    matches = pd.DataFrame({"Date": ["2024-01-01"], "HomeTeam": ["Man United"], "AwayTeam": ["Liverpool"]})
    output = build_travel_features(matches, coordinates)
    assert output.loc[0, "has_both_coordinates"]


def test_travel_features_match_manual_override():
    coordinates = pd.DataFrame(columns=["team", "latitude", "longitude"])
    overrides = pd.DataFrame(
        {
            "team": ["Inter"],
            "stadium": ["Giuseppe Meazza"],
            "country": ["Italy"],
            "latitude": [45.4781],
            "longitude": [9.1240],
            "confidence": ["high"],
        }
    )
    matches = pd.DataFrame({"Date": ["2024-01-01"], "HomeTeam": ["Inter"], "AwayTeam": ["Inter"]})
    output = build_travel_features(matches, coordinates, overrides)
    assert output.loc[0, "has_both_coordinates"]
    assert output.loc[0, "home_stadium"] == "Giuseppe Meazza"


def test_match_team_coordinates_reports_unmatched_teams():
    coordinates = pd.DataFrame({"team": ["A"], "latitude": [51.5], "longitude": [-0.1]})
    report = match_team_coordinates(["A", "Missing FC"], coordinates)
    assert report.loc[report["team"] == "A", "matched"].item()
    missing = report.loc[report["team"] == "Missing FC"].iloc[0]
    assert not missing["matched"]
    assert missing["match_method"] == "unmatched"


def test_i1_known_alias_matches_manual_override():
    overrides = pd.DataFrame(
        {
            "team": ["Inter"],
            "stadium": ["Giuseppe Meazza"],
            "country": ["Italy"],
            "latitude": [45.4781],
            "longitude": [9.1240],
            "confidence": ["high"],
            "match_source": ["manual_override"],
        }
    )
    report = match_team_coordinates(["Internazionale"], overrides)
    row = report.iloc[0]
    assert row["matched"]
    assert row["matched_team"] == "Inter"
    assert row["match_source"] == "manual_override"


def test_e0_missing_weather_teams_match_manual_overrides():
    coordinates = pd.read_csv("tests/fixtures/stadiums_with_gps_coordinates.csv")
    overrides = pd.read_csv("tests/fixtures/team_stadium_overrides.csv")
    reference = combine_coordinate_sources(coordinates, overrides)
    teams = ["Bournemouth", "Brentford", "Huddersfield", "Luton", "Sheffield United"]
    report = match_team_coordinates(teams, reference)
    assert report["matched"].all()
    assert set(report["match_source"]) == {"manual_override"}

    brentford = report[report["team"] == "Brentford"].iloc[0]
    assert brentford["stadium"] == "Gtech Community Stadium"


def test_sp1_2020_2025_missing_stadium_teams_match_manual_overrides():
    coordinates = pd.read_csv("tests/fixtures/stadiums_with_gps_coordinates.csv")
    overrides = pd.read_csv("tests/fixtures/team_stadium_overrides.csv")
    reference = combine_coordinate_sources(coordinates, overrides)
    teams = ["Alaves", "Cadiz", "Eibar", "Girona", "Huesca", "Las Palmas", "Leganes"]

    report = match_team_coordinates(teams, reference)

    assert report["matched"].all()
    assert set(report["match_source"]) == {"manual_override"}
    assert report.set_index("team").loc["Las Palmas", "stadium"] == "Estadio de Gran Canaria"


def test_sp1_historical_manual_overrides_include_absent_known_misses():
    overrides = pd.read_csv("tests/fixtures/team_stadium_overrides.csv")
    teams = {"Cordoba", "Hercules"}
    rows = overrides[overrides["team"].isin(teams)]

    assert set(rows["team"]) == teams
    assert set(rows["confidence"]) == {"high"}
    assert set(rows["source"]) == {"data/external/stadiums/world_soccer_stadiums.json"}


def test_d1_2020_2025_missing_stadium_teams_match_manual_overrides():
    coordinates = pd.read_csv("tests/fixtures/stadiums_with_gps_coordinates.csv")
    overrides = pd.read_csv("tests/fixtures/team_stadium_overrides.csv")
    reference = combine_coordinate_sources(coordinates, overrides)
    teams = ["Bielefeld", "Bochum", "Darmstadt", "Heidenheim", "Holstein Kiel", "RB Leipzig", "Union Berlin"]

    report = match_team_coordinates(teams, reference)

    assert report["matched"].all()
    assert set(report["match_source"]) == {"manual_override"}
    assert report.set_index("team").loc["RB Leipzig", "stadium"] == "Red Bull Arena"
    assert report.set_index("team").loc["Union Berlin", "stadium"] == "An der Alten Försterei"


def test_d1_historical_manual_overrides_include_absent_known_misses():
    overrides = pd.read_csv("tests/fixtures/team_stadium_overrides.csv")
    teams = {"Ingolstadt", "Paderborn"}
    rows = overrides[overrides["team"].isin(teams)]

    assert set(rows["team"]) == teams
    assert set(rows["confidence"]) == {"high"}
    assert set(rows["source"]) == {"data/external/stadiums/world_soccer_stadiums.json"}


def test_f1_2020_2025_missing_stadium_teams_match_manual_overrides():
    coordinates = pd.read_csv("tests/fixtures/stadiums_with_gps_coordinates.csv")
    overrides = pd.read_csv("tests/fixtures/team_stadium_overrides.csv")
    reference = combine_coordinate_sources(coordinates, overrides)
    teams = ["Angers", "Clermont", "Le Havre", "Lens", "Metz", "Nimes", "Strasbourg"]

    report = match_team_coordinates(teams, reference)

    assert report["matched"].all()
    assert set(report["match_source"]) == {"manual_override"}
    assert report.set_index("team").loc["Le Havre", "stadium"] == "Stade Océane"
    assert report.set_index("team").loc["Lens", "stadium"] == "Stade Bollaert-Delelis"


def test_f1_historical_manual_overrides_include_absent_known_misses_but_not_arles():
    overrides = pd.read_csv("tests/fixtures/team_stadium_overrides.csv")
    teams = {"Ajaccio GFCO", "Amiens"}
    rows = overrides[overrides["team"].isin(teams)]

    assert set(rows["team"]) == teams
    assert set(rows["confidence"]) == {"high"}
    assert set(rows["source"]) == {"data/external/stadiums/world_soccer_stadiums.json"}
    assert "Arles" not in set(overrides["team"])


def test_weather_join_and_shock_are_match_key_based():
    matches = pd.DataFrame({"Date": ["2024-01-01"], "HomeTeam": ["A"], "AwayTeam": ["B"]})
    weather = pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "temperature_c": [5.0],
            "wind_speed_kph": [30.0],
        }
    )
    normals = pd.DataFrame({"team": ["B"], "month": [1], "temperature_c": [10.0], "wind_speed_kph": [12.0]})
    with_weather = add_weather_features(matches, weather)
    output = add_weather_shock_features(with_weather, normals)
    assert output.loc[0, "has_weather"]
    assert output.loc[0, "away_temperature_shock_c"] == -5.0
    assert output.loc[0, "away_wind_speed_shock_kph"] == 18.0


def test_contextual_builder_low_risk_without_external_tables():
    output = build_contextual_features(sample_matches())
    assert "home_rest_days" in output.columns
    assert "both_teams_first_5_season_matches" in output.columns
    assert "avg_1x2_AvgH_no_vig_probability" in output.columns
