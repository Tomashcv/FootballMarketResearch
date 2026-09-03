from __future__ import annotations

import pandas as pd

from src.v4.data.market_contract import TIMING_CLASSES, classify_column
from src.v4.data.phase1_audit import (
    complete_market_masks,
    inventory_for_group,
    read_csv_resilient,
    season_from_raw_filename,
)


def test_football_data_non_c_prices_are_not_promoted_to_opening() -> None:
    for column in ("B365H", "AvgH", "B365AHH", "AvgAHH", "B365>2.5", "Avg<2.5"):
        contract = classify_column("football_data", column)
        assert contract is not None
        assert contract.timing_classification == "current_snapshot_unknown_time"
        assert contract.feature_policy == "prohibited_until_timing_resolved"


def test_documented_football_data_c_prices_are_verified_closing() -> None:
    expected = {
        "B365CH": "1X2",
        "AvgCA": "1X2",
        "B365CAHH": "Asian Handicap",
        "AvgCAHA": "Asian Handicap",
        "B365C>2.5": "Over/Under",
        "AvgC<2.5": "Over/Under",
    }
    for column, market in expected.items():
        contract = classify_column("football_data", column)
        assert contract is not None
        assert contract.market == market
        assert contract.timing_classification == "verified_closing"
        assert contract.feature_policy == "closing_label_or_diagnostic_only"


def test_bookmaker_counts_are_not_price_observations() -> None:
    for column in ("Bb1X2", "BbOU", "BbAH"):
        contract = classify_column("football_data", column)
        assert contract is not None
        assert contract.timing_classification == "not_applicable"
        assert contract.role == "bookmaker_count"


def test_ah_line_timing_follows_documented_snapshot_semantics() -> None:
    opening_like = classify_column("football_data", "AHh")
    closing = classify_column("football_data", "AHCh")
    assert opening_like is not None and opening_like.role == "line"
    assert opening_like.timing_classification == "current_snapshot_unknown_time"
    assert closing is not None and closing.role == "line"
    assert closing.timing_classification == "verified_closing"


def test_footiqo_and_beat_the_bookie_timing_remains_unknown() -> None:
    for column in ("H", "D", "A", "O25", "U25", "BTTSY"):
        contract = classify_column("footiqo", column)
        assert contract is not None
        assert contract.timing_classification == "timing_unknown"
    encoded = classify_column("beat_the_bookie", "home_b12_71", "odds_series.csv.gz")
    aggregate = classify_column("beat_the_bookie", "avg_odds_home_win", "closing_odds.csv.gz")
    assert encoded is not None and encoded.timing_classification == "timing_unknown"
    assert aggregate is not None and aggregate.timing_classification == "timing_unknown"


def test_all_classifications_use_the_closed_timing_vocabulary() -> None:
    columns = [
        ("football_data", "B365H", ""),
        ("football_data", "B365CH", ""),
        ("football_data", "AHh", ""),
        ("footiqo", "O25", ""),
        ("beat_the_bookie", "away_b2_10", "odds_series.csv.gz"),
    ]
    for source, column, member in columns:
        contract = classify_column(source, column, member)
        assert contract is not None
        assert contract.timing_classification in TIMING_CLASSES


def test_inventory_counts_only_odds_above_one_as_valid() -> None:
    frame = pd.DataFrame({
        "B365H": [2.0, 1.0, None],
        "B365D": [3.0, 3.0, 3.0],
        "B365A": [4.0, 4.0, 4.0],
        "FTHG": [1, 2, 0],
    })
    inventory, anomalies = inventory_for_group(frame, "football_data", "synthetic.csv", "", "E0", "2024/2025", True)
    home = next(row for row in inventory if row["column"] == "B365H")
    assert home["valid_odds_values"] == 1
    assert home["invalid_odds_le_1"] == 1
    assert home["valid_odds_coverage"] == 1 / 3
    assert any(row["anomaly_type"] == "odds_le_1" for row in anomalies)
    assert all(row["column"] != "FTHG" for row in inventory)


def test_complete_market_requires_all_selections_and_ah_line() -> None:
    frame = pd.DataFrame({
        "B365CH": [2.0, 2.0], "B365CD": [3.0, 3.0], "B365CA": [4.0, None],
        "AHCh": [-0.5, None], "B365CAHH": [1.9, 1.9], "B365CAHA": [2.0, 2.0],
    })
    masks = complete_market_masks("football_data", frame)
    assert masks[("1X2", "verified_closing", "any_bookmaker")].tolist() == [True, False]
    assert masks[("Asian Handicap", "verified_closing", "any_bookmaker")].tolist() == [True, False]


def test_season_comes_from_provider_filename_not_covid_kickoff_month(tmp_path) -> None:
    assert season_from_raw_filename(tmp_path / "E0_1920.csv", "E0") == "2019/2020"
    assert season_from_raw_filename(tmp_path / "E0_2020_2021.csv", "E0") == "2020/2021"
    assert season_from_raw_filename(tmp_path / "E0_2021.csv", "E0") == "2020/2021"


def test_malformed_legacy_csv_rows_are_retained_and_counted(tmp_path) -> None:
    path = tmp_path / "E0_0304.csv"
    path.write_text("Div,Date,B365H\nE0,01/08/03,2.0\nE0,02/08/03,3.0,EXTRA\n", encoding="utf-8")
    frame, malformed = read_csv_resilient(path)
    assert len(frame) == 2
    assert malformed == 1
    assert frame.columns.tolist() == ["Div", "Date", "B365H"]
