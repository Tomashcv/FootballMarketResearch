from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.experiments.ah_settlement_engine_audit import settle_side
from src.experiments.ah_settlement_engine_audit import split_handicap


def test_split_quarter_handicap_lines():
    assert split_handicap(-1.25) == (-1.5, -1.0)
    assert split_handicap(1.25) == (1.0, 1.5)
    assert split_handicap(-1.5) == (-1.5,)
    assert split_handicap(0.0) == (0.0,)


def test_required_home_ah_settlement_cases():
    cases = [
        (0.0, 0, "push", 0.0),
        (-0.5, 0, "full_loss", -1.0),
        (0.5, 0, "full_win", 1.0),
        (-1.0, 1, "push", 0.0),
        (-1.0, 2, "full_win", 1.0),
        (1.0, -1, "push", 0.0),
        (-0.25, 0, "half_loss", -0.5),
        (0.25, 0, "half_win", 0.5),
        (-0.75, 1, "half_win", 0.5),
        (0.75, -1, "half_loss", -0.5),
        (-1.25, 1, "half_loss", -0.5),
        (-1.25, 2, "full_win", 1.0),
        (1.25, -1, "half_win", 0.5),
        (1.25, -2, "full_loss", -1.0),
        (-1.5, 1, "full_loss", -1.0),
        (-1.5, 2, "full_win", 1.0),
    ]
    for handicap, margin, label, profit in cases:
        result = settle_side(margin, handicap, 2.0)
        assert result.label == label
        assert result.profit == profit


def test_required_away_ah_settlement_cases_are_symmetric():
    cases = [
        (0.0, 0, "push", 0.0),
        (-0.5, 0, "full_win", 1.0),
        (0.5, 0, "full_loss", -1.0),
        (-1.0, 1, "push", 0.0),
        (-1.0, 2, "full_loss", -1.0),
        (1.0, -1, "push", 0.0),
        (-0.25, 0, "half_win", 0.5),
        (0.25, 0, "half_loss", -0.5),
        (-0.75, 1, "half_loss", -0.5),
        (0.75, -1, "half_win", 0.5),
        (-1.25, 1, "half_win", 0.5),
        (-1.25, 2, "full_loss", -1.0),
        (1.25, -1, "half_loss", -0.5),
        (1.25, -2, "full_win", 1.0),
        (-1.5, 1, "full_win", 1.0),
        (-1.5, 2, "full_loss", -1.0),
    ]
    for home_handicap, home_margin, label, profit in cases:
        result = settle_side(-home_margin, -home_handicap, 2.0)
        assert result.label == label
        assert result.profit == profit
