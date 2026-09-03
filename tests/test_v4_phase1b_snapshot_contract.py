from __future__ import annotations

import numpy as np
import pandas as pd

from src.v4.data.phase1b_audit import (
    TIMING_CLASSES_1B,
    PairSpec,
    ah_price_shift,
    build_1x2_pairs,
    fixture_timing,
    no_vig_probabilities,
    ou_price_shift,
    price_clv,
    probability_shift,
    timing_contract,
)


def test_friday_batch_precedes_saturday_and_sunday_fixtures() -> None:
    for date in (pd.Timestamp("2026-08-15"), pd.Timestamp("2026-08-16")):
        result = fixture_timing(date, time_available=False)
        assert result["eligible"] is True
        assert result["timing_classification"] == "verified_scheduled_prematch_snapshot"
        assert result["documented_collection_weekday"] == "Friday"
        assert result["full_day_separation"] is True


def test_tuesday_batch_precedes_wednesday_and_thursday_fixtures() -> None:
    for date in (pd.Timestamp("2026-08-12"), pd.Timestamp("2026-08-13")):
        result = fixture_timing(date, time_available=False)
        assert result["eligible"] is True
        assert result["timing_classification"] == "verified_scheduled_prematch_snapshot"
        assert result["documented_collection_weekday"] == "Tuesday"


def test_same_day_fixtures_are_excluded_even_when_kickoff_time_exists() -> None:
    friday = fixture_timing(pd.Timestamp("2026-08-14"), time_available=True)
    tuesday = fixture_timing(pd.Timestamp("2026-08-11"), time_available=True)
    assert friday["eligible"] is False
    assert tuesday["eligible"] is False
    assert friday["timing_classification"] == "scheduled_snapshot_ambiguous"
    assert tuesday["timing_classification"] == "scheduled_snapshot_ambiguous"
    assert "intraday_order_unproved" in friday["eligibility_reason"]
    assert "intraday_order_unproved" in tuesday["eligibility_reason"]


def test_price_clv_sign_convention() -> None:
    assert np.isclose(price_clv(2.20, 2.00), 0.10)
    assert price_clv(1.90, 2.00) < 0


def test_no_vig_probability_shift_sign_convention() -> None:
    snapshot = [2.0, 3.5, 4.0]
    closing = [2.1, 3.6, 3.5]
    shift = probability_shift(snapshot, closing)
    assert shift[2] > 0  # market moved toward away
    assert np.isclose(shift.sum(), 0.0)
    assert np.isclose(no_vig_probabilities(snapshot).sum(), 1.0)


def test_ah_different_line_prices_are_not_directly_compared() -> None:
    shift = ah_price_shift(-1.25, -1.50, [1.95, 1.95], [1.90, 2.00])
    assert np.isnan(shift).all()
    same = ah_price_shift(-1.25, -1.25, [1.95, 1.95], [1.90, 2.00])
    assert np.isfinite(same).all()


def test_ou_different_line_prices_are_not_directly_compared() -> None:
    shift = ou_price_shift(2.5, 2.75, [1.90, 2.00], [1.95, 1.95])
    assert np.isnan(shift).all()
    same = ou_price_shift(2.5, 2.5, [1.90, 2.00], [1.95, 1.95])
    assert np.isfinite(same).all()


def test_closing_fields_never_enter_snapshot_feature_classification() -> None:
    specs = [
        PairSpec("1X2", "B365", "Bet365", "bookmaker", ("B365H", "B365D", "B365A"), ("B365CH", "B365CD", "B365CA"))
    ]
    contract = timing_contract(specs)
    closing = contract[contract["column"].isin(specs[0].closing_columns)]
    assert closing["timing_classification"].eq("verified_closing").all()
    assert ~closing["closing_field_can_be_snapshot_feature"].astype(bool).any()
    assert not closing["feature_policy"].str.contains("snapshot_feature").any()
    assert set(contract["timing_classification"]).issubset(TIMING_CLASSES_1B)


def test_same_family_pair_mapping_does_not_treat_avg_as_bookmaker() -> None:
    columns = {
        "B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA",
        "AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA",
        "VCH", "VCD", "VCA", "VCCH", "VCCD", "VCCA",
    }
    pairs = {pair.family: pair for pair in build_1x2_pairs(columns)}
    assert pairs["B365"].observation_type == "bookmaker"
    assert pairs["Avg"].observation_type == "consensus_aggregate"
    assert "VC" in pairs
    assert "V" not in pairs
    assert pairs["VC"].bookmaker == "VC Bet"
