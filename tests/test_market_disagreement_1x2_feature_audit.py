from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.experiments.market_disagreement_1x2_feature_audit import OPEN_SOURCES
from src.experiments.market_disagreement_1x2_feature_audit import build_features
from src.experiments.market_disagreement_1x2_feature_audit import feature_columns


def sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "league": ["E0", "E0"],
            "Date": pd.to_datetime(["2024-08-01", "2024-08-02"]),
            "season_end_year": [2025, 2025],
            "HomeTeam": ["A", "C"],
            "AwayTeam": ["B", "D"],
            "FTR": ["H", "A"],
            "FTHG": [2, 0],
            "FTAG": [1, 1],
            "AvgH": [2.0, 2.5],
            "AvgD": [3.5, 3.2],
            "AvgA": [4.0, 2.9],
            "MaxH": [2.1, 2.6],
            "MaxD": [3.6, 3.3],
            "MaxA": [4.1, 3.0],
            "B365H": [1.95, 2.45],
            "B365D": [3.4, 3.1],
            "B365A": [4.2, 3.0],
            "PSH": [2.02, 2.55],
            "PSD": [3.55, 3.25],
            "PSA": [4.05, 2.85],
            "BWH": [2.0, 2.5],
            "BWD": [3.4, 3.2],
            "BWA": [4.0, 2.9],
            "IWH": [2.0, 2.5],
            "IWD": [3.5, 3.2],
            "IWA": [4.0, 2.9],
            "WHH": [2.0, 2.5],
            "WHD": [3.5, 3.2],
            "WHA": [4.0, 2.9],
            "VCH": [2.0, 2.5],
            "VCD": [3.5, 3.2],
            "VCA": [4.0, 2.9],
            "1XBH": [2.0, 2.5],
            "1XBD": [3.5, 3.2],
            "1XBA": [4.0, 2.9],
            "BFEH": [2.0, 2.5],
            "BFED": [3.5, 3.2],
            "BFEA": [4.0, 2.9],
            "AHh": [-0.5, 0.25],
            "AvgAHH": [1.9, 1.95],
            "AvgAHA": [1.95, 1.9],
            "AvgCH": [1.8, 2.2],
            "AvgCD": [3.4, 3.1],
            "AvgCA": [4.4, 3.4],
        }
    )


def test_build_features_excludes_closing_columns_from_feature_matrix() -> None:
    features, open_sources, close_sources = build_features(sample_matches())
    columns = feature_columns(features)

    assert set(open_sources) == set(OPEN_SOURCES)
    assert "avg_close" in close_sources
    assert not any("AvgC" in column or "close" in column for column in columns)


def test_ah_relevant_disagreement_features_are_created() -> None:
    features, _, _ = build_features(sample_matches())

    assert "away_1x2_market_strength_minus_ah_market_strength" in features.columns
    assert "favourite_strength_disagreement" in features.columns
    assert features["away_ah_cover_outcome"].notna().all()
