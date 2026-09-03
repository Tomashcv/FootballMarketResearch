from __future__ import annotations

import pandas as pd

from src.paper_trading.v3_pipeline import (
    RULE_NAME,
    append_new_picks_to_ledger,
    config_thresholds,
    deterministic_paper_bet_id,
    empty_ledger,
    select_candidate_picks,
    select_candidate_picks_after_quality_gate,
    settle_open_ledger,
)


def sample_pick(selected_odds: float | None = 2.25) -> pd.DataFrame:
    canonical_id = "match-1"
    bet_id = deterministic_paper_bet_id(canonical_id, "away", RULE_NAME, selected_odds, "2026-08-15")
    return pd.DataFrame(
        [
            {
                "paper_bet_id": bet_id,
                "run_id": "run-1",
                "source_snapshot_id": "snap-1",
                "canonical_match_id": canonical_id,
                "match_date": "2026-08-15",
                "league": "E0",
                "season_start_year": 2026,
                "home_team": "Home",
                "away_team": "Away",
                "selected_side": "away",
                "selected_odds": selected_odds,
                "market_home_prob": 0.45,
                "market_draw_prob": 0.25,
                "market_away_prob": 0.30,
                "model_home_prob": 0.42,
                "model_draw_prob": 0.24,
                "model_away_prob": 0.34,
                "away_edge": 0.04,
                "rule_name": RULE_NAME,
                "stake_units": 1.0,
            }
        ]
    )


def test_no_duplicate_ledger_entries() -> None:
    ledger, new_rows = append_new_picks_to_ledger(empty_ledger(), sample_pick(), created_at_utc="2026-07-05T00:00:00Z")
    ledger, duplicate_rows = append_new_picks_to_ledger(ledger, sample_pick(), created_at_utc="2026-07-05T00:01:00Z")
    assert len(new_rows) == 1
    assert len(duplicate_rows) == 0
    assert len(ledger) == 1


def test_settlement_math_correct_for_away_pick() -> None:
    ledger, _ = append_new_picks_to_ledger(empty_ledger(), sample_pick(2.25), created_at_utc="2026-07-05T00:00:00Z")
    win = settle_open_ledger(ledger, {"match-1": "A"}, settled_at_utc="2026-08-16T00:00:00Z")
    assert win.loc[0, "status"] == "SETTLED_WIN"
    assert float(win.loc[0, "profit_units"]) == 1.25

    ledger, _ = append_new_picks_to_ledger(empty_ledger(), sample_pick(2.25), created_at_utc="2026-07-05T00:00:00Z")
    loss = settle_open_ledger(ledger, {"match-1": "H"}, settled_at_utc="2026-08-16T00:00:00Z")
    assert loss.loc[0, "status"] == "SETTLED_LOSS"
    assert float(loss.loc[0, "profit_units"]) == -1.0


def test_no_picks_created_when_odds_missing() -> None:
    pred = sample_pick(None).drop(columns=["paper_bet_id", "away_edge"])
    picks, skipped = select_candidate_picks(pred, "run-1", "snap-1")
    assert picks.empty
    assert len(skipped) == 1


def test_no_new_paper_pick_when_result_already_available() -> None:
    pred = sample_pick(2.25).drop(columns=["paper_bet_id", "away_edge"])
    pred["target_outcome_1x2"] = "A"
    picks, skipped = select_candidate_picks(pred, "run-1", "snap-1")
    assert picks.empty
    assert len(skipped) == 1
    assert skipped.loc[0, "skip_status"] == "SKIPPED_RESULT_ALREADY_AVAILABLE"


def test_no_picks_created_when_feature_leakage_check_fails() -> None:
    pred = sample_pick().drop(columns=["paper_bet_id"])
    validation = pd.DataFrame(
        [
            {
                "source_file": "data/raw/E0/seasons/E0_2627.csv",
                "div": "E0",
                "classification": "research_only",
                "match_date": "2026-08-15",
                "home_clubelo_latest_date": "2026-08-15",
                "away_clubelo_latest_date": "2026-08-14",
                "home_internal_elo": 1500,
                "away_internal_elo": 1500,
                "internal_elo_diff": 0,
            }
        ]
    )
    picks, blocked, checks = select_candidate_picks_after_quality_gate(validation, pred, "run-1", "snap-1")
    assert checks["status"].eq("fail").any()
    assert picks.empty
    assert len(blocked) == 1
    assert blocked.loc[0, "skip_status"] == "BLOCKED_DATA_QUALITY"


def test_config_thresholds_match_frozen_v3_values() -> None:
    thresholds = config_thresholds()
    assert thresholds["away_edge_min"] == 0.015
    assert thresholds["away_odds_min"] == 1.5
    assert thresholds["stake_units"] == 1.0
