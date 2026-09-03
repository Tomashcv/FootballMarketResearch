from __future__ import annotations

import pandas as pd

from src.paper_trading.ah_e0_pipeline import (
    RULE_NAME,
    append_new_picks_to_ledger,
    config_thresholds,
    deterministic_paper_bet_id,
    empty_ledger,
    normalize_prediction_rows,
    settle_away_handicap,
)


def test_away_ah_quarter_settlement_examples() -> None:
    status, profit = settle_away_handicap(home_goals=1, away_goals=0, selected_handicap_line=1.25, selected_odds=2.0)
    assert status == "SETTLED_HALF_WIN"
    assert profit == 0.5

    status, profit = settle_away_handicap(home_goals=1, away_goals=0, selected_handicap_line=0.75, selected_odds=2.0)
    assert status == "SETTLED_HALF_LOSS"
    assert profit == -0.5

    status, profit = settle_away_handicap(home_goals=0, away_goals=1, selected_handicap_line=-0.75, selected_odds=2.0)
    assert status == "SETTLED_HALF_WIN"
    assert profit == 0.5


def sample_raw(ftr: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": "15/08/2026",
                "match_date": pd.Timestamp("2026-08-15"),
                "HomeTeam": "Home",
                "AwayTeam": "Away",
                "AHh": -1.25,
                "AvgAHA": 1.95,
                "FTHG": pd.NA,
                "FTAG": pd.NA,
                "FTR": ftr,
                "source_file": "data/raw/E0/seasons/E0_2627.csv",
                "season_start_year": 2026,
            }
        ]
    )


def test_no_duplicate_ah_ledger_entries() -> None:
    picks, _skipped = normalize_prediction_rows(sample_raw(), "run-1", "snap-1")
    ledger, new_rows = append_new_picks_to_ledger(empty_ledger(), picks, created_at_utc="2026-07-05T00:00:00Z")
    ledger, duplicate_rows = append_new_picks_to_ledger(ledger, picks, created_at_utc="2026-07-05T00:01:00Z")
    assert len(new_rows) == 1
    assert len(duplicate_rows) == 0
    assert len(ledger) == 1


def test_no_ah_pick_when_ah_odds_missing() -> None:
    raw = sample_raw()
    raw["AvgAHA"] = pd.NA
    picks, skipped = normalize_prediction_rows(raw, "run-1", "snap-1")
    assert picks.empty
    assert skipped.loc[0, "skip_status"] == "SKIPPED_NO_AH_ODDS"


def test_no_new_ah_paper_pick_when_result_already_available() -> None:
    raw = sample_raw("A")
    raw["FTHG"] = 0
    raw["FTAG"] = 1
    picks, skipped = normalize_prediction_rows(raw, "run-1", "snap-1")
    assert picks.empty
    assert skipped.loc[0, "skip_status"] == "SKIPPED_RESULT_ALREADY_AVAILABLE"


def test_ah_config_thresholds_match_frozen_values() -> None:
    thresholds = config_thresholds()
    assert thresholds["home_ah_line_max"] == -1.25
    assert thresholds["stake_units"] == 1.0


def test_ah_bet_id_uses_required_fields() -> None:
    one = deterministic_paper_bet_id("2026-08-15", "E0", "Home", "Away", "Asian Handicap", "away", 1.25, 1.95, RULE_NAME)
    two = deterministic_paper_bet_id("2026-08-15", "E0", "Home", "Away", "Asian Handicap", "away", 1.25, 1.90, RULE_NAME)
    assert one != two
