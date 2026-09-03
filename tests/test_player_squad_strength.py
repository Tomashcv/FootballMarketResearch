from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features.player_squad_strength import discover_player_data_files
from src.features.player_squad_strength import add_transfermarkt_window_features
from src.features.player_squad_strength import inspect_player_file
from src.features.player_squad_strength import latest_squad_features
from src.features.player_squad_strength import load_time_safe_observations
from src.features.player_squad_strength import normalize_name
from src.features.player_squad_strength import transfermarkt_window_features


def test_player_file_discovery_and_time_safety(tmp_path: Path) -> None:
    source = tmp_path / "sofifa_player_ratings.csv"
    pd.DataFrame(
        {
            "snapshot_date": ["2020-08-01", "2020-08-01"],
            "club_name": ["Arsenal", "Arsenal"],
            "player_name": ["A Player", "B Player"],
            "overall": [80, 78],
        }
    ).to_csv(source, index=False)

    discovered = discover_player_data_files([tmp_path])
    audit = pd.DataFrame([inspect_player_file(path) for path in discovered])

    assert discovered == [source]
    assert bool(audit.loc[0, "time_safe_candidate"])
    observations = load_time_safe_observations(audit)
    assert len(observations) == 2
    assert set(observations["club_key"]) == {"arsenal"}


def test_undated_player_file_is_not_time_safe(tmp_path: Path) -> None:
    source = tmp_path / "transfermarkt_market_value.csv"
    pd.DataFrame(
        {
            "club_name": ["Arsenal"],
            "player_name": ["A Player"],
            "market_value": [10_000_000],
        }
    ).to_csv(source, index=False)

    audit = inspect_player_file(source)

    assert not audit["time_safe_candidate"]
    assert "missing dated" in audit["time_safety_reason"]


def test_latest_squad_features_use_only_before_match_date() -> None:
    observations = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2020-08-01", "2020-08-01", "2020-09-15"]),
            "club_name": ["Arsenal", "Arsenal", "Arsenal"],
            "club_key": ["arsenal", "arsenal", "arsenal"],
            "player_name": ["A", "B", "C"],
            "player_key": ["a", "b", "c"],
            "position_group": ["ATT", "MID", "ATT"],
            "age": [24, 26, 22],
            "market_value": [10_000_000, 20_000_000, 100_000_000],
            "overall": [80, 82, 95],
            "potential": [84, 83, 97],
        }
    )

    before_later_snapshot = latest_squad_features(observations, normalize_name("Arsenal"), pd.Timestamp("2020-09-01"))
    after_later_snapshot = latest_squad_features(observations, normalize_name("Arsenal"), pd.Timestamp("2020-10-01"))

    assert before_later_snapshot["squad_market_value_total"] == 30_000_000
    assert before_later_snapshot["fifa_overall_top5"] == 81
    assert after_later_snapshot["squad_market_value_total"] == 100_000_000


def test_transfermarkt_window_features_use_latest_player_values_before_match() -> None:
    values = pd.DataFrame(
        {
            "valuation_date": pd.to_datetime(
                ["2020-01-01", "2020-05-01", "2020-05-20", "2020-06-01", "2020-06-15"]
            ),
            "player_id": [1, 1, 2, 3, 4],
            "club_name": ["Arsenal", "Arsenal", "Arsenal", "Arsenal", "Arsenal"],
            "club_key": ["arsenal", "arsenal", "arsenal", "arsenal", "arsenal"],
            "market_value_eur": [10_000_000, 12_000_000, 8_000_000, 30_000_000, 99_000_000],
        }
    )

    features = transfermarkt_window_features(values, "Arsenal", pd.Timestamp("2020-06-15"))

    assert features["tm_value_total_180d"] == 50_000_000
    assert features["tm_value_top11_180d"] == 50_000_000
    assert features["tm_value_top5_180d"] == 50_000_000
    assert features["tm_value_median_180d"] == 12_000_000
    assert features["tm_players_count_180d"] == 3
    assert pd.isna(features["tm_value_total_365d"]) is False


def test_add_transfermarkt_window_features_uses_mapping_and_match_date() -> None:
    matches = pd.DataFrame(
        {
            "league": ["E0"],
            "Date": pd.to_datetime(["2020-06-15"]),
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
        }
    )
    mapping = pd.DataFrame(
        {
            "league": ["E0", "E0"],
            "match_team": ["Arsenal", "Chelsea"],
            "normalized_match_team": ["arsenal", "chelsea"],
            "player_data_club_name": ["Arsenal FC", "Chelsea FC"],
            "normalized_player_data_club": ["arsenal fc", "chelsea fc"],
            "valid_from": [pd.NaT, pd.NaT],
            "valid_to": [pd.NaT, pd.NaT],
            "confidence": ["reviewed", "reviewed"],
        }
    )
    values = pd.DataFrame(
        {
            "valuation_date": pd.to_datetime(["2020-05-01", "2020-05-01", "2020-06-15"]),
            "player_id": [1, 2, 3],
            "club_name": ["Arsenal FC", "Chelsea FC", "Chelsea FC"],
            "club_key": ["arsenal fc", "chelsea fc", "chelsea fc"],
            "market_value_eur": [10_000_000, 20_000_000, 200_000_000],
        }
    )

    featured = add_transfermarkt_window_features(matches, values, mapping)

    assert featured.loc[0, "home_tm_mapped_club_name"] == "Arsenal FC"
    assert featured.loc[0, "away_tm_mapped_club_name"] == "Chelsea FC"
    assert featured.loc[0, "home_tm_value_total_180d"] == 10_000_000
    assert featured.loc[0, "away_tm_value_total_180d"] == 20_000_000
    assert featured.loc[0, "home_minus_away_tm_value_total_180d"] == -10_000_000
