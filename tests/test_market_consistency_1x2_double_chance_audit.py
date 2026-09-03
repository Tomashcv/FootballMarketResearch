from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.experiments.market_consistency_1x2_double_chance_audit import find_double_chance_columns


def test_1xbet_columns_are_not_double_chance_columns() -> None:
    columns = ["1XBH", "1XBD", "1XBA", "1XBCH", "1XBCD", "1XBCA"]

    found = find_double_chance_columns(columns)

    assert found == {"1x": None, "12": None, "x2": None}


def test_true_double_chance_columns_are_detected() -> None:
    columns = ["AvgH", "AvgD", "AvgA", "DC1X", "DC12", "DCX2"]

    found = find_double_chance_columns(columns)

    assert found == {"1x": "DC1X", "12": "DC12", "x2": "DCX2"}
