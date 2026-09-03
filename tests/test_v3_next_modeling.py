from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_football_data_full_scope_1x2 import combine_date_and_time, parse_date_series
from src.modeling.market_residual import CandidateSpec, fit_candidate
from src.modeling.probability import apply_blend_temperature, fit_blend_temperature, market_probs, probability_metrics
from src.modeling.temporal import build_nested_year_folds, split_nested_fold
from src.modeling.v3_features import add_research_derived_features, build_v3_adapter
from src.modeling.value_selection import add_value_columns, build_rule_grid, select_rule_on_validation
from src.paper_trading.v3_pipeline import select_candidate_picks


def _prob_frame(years: list[int], rows_per_year: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    idx = 0
    for year in years:
        for _ in range(rows_per_year):
            p = rng.dirichlet([2.3, 1.5, 2.0])
            y = int(rng.choice(3, p=p))
            rows.append(
                {
                    "match_id": str(idx),
                    "season_start_year": year,
                    "match_date": pd.Timestamp(year=year, month=8, day=1) + pd.Timedelta(days=idx),
                    "league": "E0" if idx % 2 else "I1",
                    "target_y": y,
                    "x1x2_avg_prob_home": p[0],
                    "x1x2_avg_prob_draw": p[1],
                    "x1x2_avg_prob_away": p[2],
                    "x1x2_avg_odds_home": 1.0 / p[0],
                    "x1x2_avg_odds_draw": 1.0 / p[1],
                    "x1x2_avg_odds_away": 1.0 / p[2],
                    "signal": float((y == 2) + rng.normal(0, 0.4)),
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def test_mixed_date_parser_is_deterministic() -> None:
    source = pd.Series(["2025-08-15", "16/08/2025", "17-08-25", "bad"])
    parsed = parse_date_series(source)
    assert parsed.iloc[0] == pd.Timestamp("2025-08-15")
    assert parsed.iloc[1] == pd.Timestamp("2025-08-16")
    assert parsed.iloc[2] == pd.Timestamp("2025-08-17")
    assert pd.isna(parsed.iloc[3])
    combined = combine_date_and_time(parsed, pd.Series(["20:00", "15:30", "", "12:00"]))
    assert combined.iloc[0] == pd.Timestamp("2025-08-15 20:00")
    assert combined.iloc[2] == pd.Timestamp("2025-08-17 00:00")


def test_blend_alpha_zero_is_market() -> None:
    market = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]])
    model = np.array([[0.7, 0.1, 0.2], [0.1, 0.2, 0.7]])
    out = apply_blend_temperature(model, market, alpha=0.0, temperature=1.0)
    np.testing.assert_allclose(out, market)


def test_calibration_grid_returns_valid_choice() -> None:
    y = np.array([0, 2, 2, 1])
    market = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5], [0.25, 0.2, 0.55], [0.3, 0.4, 0.3]])
    model = np.array([[0.6, 0.2, 0.2], [0.15, 0.2, 0.65], [0.2, 0.15, 0.65], [0.25, 0.5, 0.25]])
    fitted = fit_blend_temperature(y, model, market, [0.0, 0.5, 1.0], [0.9, 1.0, 1.1])
    assert 0.0 <= fitted.alpha <= 1.0
    assert fitted.temperature > 0
    assert np.isfinite(fitted.log_loss)


def test_nested_fold_has_four_strict_partitions() -> None:
    frame = _prob_frame(list(range(2016, 2024)), rows_per_year=10)
    folds = build_nested_year_folds(frame, [2021, 2022, 2023], min_train_rows=20)
    assert [f.test_year for f in folds] == [2021, 2022, 2023]
    train, tune, calibration, test = split_nested_fold(frame, folds[0])
    assert train["season_start_year"].max() < tune["season_start_year"].min()
    assert tune["season_start_year"].max() < calibration["season_start_year"].min()
    assert calibration["season_start_year"].max() < test["season_start_year"].min()


def test_vectorized_v3_adapter_and_derived_features() -> None:
    raw = pd.DataFrame(
        {
            "full_scope_match_id": ["1", "2"],
            "logical_match_key": ["a", "b"],
            "match_date": ["2024-08-01", "2024-08-02"],
            "div": ["E0", "I1"],
            "season_start_year": [2024, 2024],
            "home_team_raw": ["A", "C"],
            "away_team_raw": ["B", "D"],
            "result_1x2": ["H", "A"],
            "x1x2_avg_prob_home": [0.5, 0.3],
            "x1x2_avg_prob_draw": [0.25, 0.25],
            "x1x2_avg_prob_away": [0.25, 0.45],
            "x1x2_avg_market_overround": [1.05, 1.04],
            "x1x2_avg_odds_home": [2.0, 3.2],
            "x1x2_avg_odds_draw": [4.0, 4.0],
            "x1x2_avg_odds_away": [4.0, 2.2],
            "clubelo_diff": [100.0, -30.0],
            "internal_elo_diff": [80.0, -10.0],
            "clubelo_both_found_flag": [True, False],
        }
    )
    features = [
        "x1x2_avg_prob_home",
        "x1x2_avg_prob_draw",
        "x1x2_avg_prob_away",
        "x1x2_avg_market_overround",
        "x1x2_avg_odds_home",
        "x1x2_avg_odds_draw",
        "x1x2_avg_odds_away",
        "clubelo_diff",
        "internal_elo_diff",
        "clubelo_both_found_flag",
    ]
    adapted = build_v3_adapter(raw, features, require_target=True)
    derived, created = add_research_derived_features(adapted)
    assert len(adapted) == 2
    assert adapted.columns.duplicated().sum() == 0
    assert adapted["target_y"].tolist() == [0, 2]
    assert "research_market_entropy" in created
    assert derived["research_clubelo_internal_disagreement"].tolist() == [20.0, -20.0]


def test_logistic_candidate_runs_on_temporal_data() -> None:
    frame = _prob_frame(list(range(2016, 2022)), rows_per_year=25)
    train = frame[frame["season_start_year"] < 2020]
    validation = frame[frame["season_start_year"] == 2020]
    spec = CandidateSpec(
        name="ridge_test",
        family="logistic_market_plus_features",
        feature_group="test",
        params={"C": 0.1, "max_iter": 500},
        recency_half_life_years=3.0,
        league_balance_strength=0.2,
    )
    fitted, probability, metadata = fit_candidate(spec, train, validation, ["signal"])
    assert probability.shape == (len(validation), 3)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    assert metadata.feature_count == 1
    assert fitted.predict_proba(validation).shape == probability.shape


def test_rule_selection_is_predeclared_and_volume_gated() -> None:
    frame = _prob_frame([2020], rows_per_year=150)
    market = market_probs(frame)
    model = market.copy()
    model[:, 2] = np.clip(model[:, 2] + 0.04, 1e-5, 0.99)
    model[:, 0] = np.clip(model[:, 0] - 0.04, 1e-5, 0.99)
    model /= model.sum(axis=1, keepdims=True)
    value = pd.DataFrame(
        {
            "league": frame["league"],
            "target_y": frame["target_y"],
            "market_home_prob": market[:, 0],
            "market_draw_prob": market[:, 1],
            "market_away_prob": market[:, 2],
            "model_home_prob": model[:, 0],
            "model_draw_prob": model[:, 1],
            "model_away_prob": model[:, 2],
            "odds_home": frame["x1x2_avg_odds_home"],
            "odds_draw": frame["x1x2_avg_odds_draw"],
            "odds_away": frame["x1x2_avg_odds_away"],
        }
    )
    value = add_value_columns(value)
    rules = build_rule_grid(["away"], [0.02, 0.03], [1.5], [None])
    selection, table = select_rule_on_validation(value, rules, min_bets=20, require_positive_lcb=False, max_positive_league_share=1.0, minimum_positive_leagues=1)
    assert len(table) == 2
    assert selection.rule is None or selection.rule.name in set(table["name"])


def test_paper_skips_below_edge_explicitly() -> None:
    pred = pd.DataFrame(
        {
            "canonical_match_id": ["m1", "m2"],
            "match_date": ["2030-01-01", "2030-01-02"],
            "model_away_prob": [0.41, 0.50],
            "market_away_prob": [0.40, 0.40],
            "selected_odds": [2.0, 2.0],
            "target_outcome_1x2": ["", ""],
        }
    )
    picked, skipped = select_candidate_picks(pred, "run", "snap")
    assert len(picked) == 1
    assert skipped.iloc[0]["skip_status"] == "SKIPPED_BELOW_EDGE"


def test_nested_fold_purges_cross_season_date_overlap() -> None:
    rows = [
        {
            "season_start_year": 2018,
            "match_date": "2019-05-20",
            "league": "E0",
        },
        {
            "season_start_year": 2019,
            "match_date": "2019-08-10",
            "league": "E0",
        },
        {
            # Delayed 2019/20 fixture overlapping the next season.
            "season_start_year": 2019,
            "match_date": "2020-07-20",
            "league": "G1",
        },
        {
            "season_start_year": 2020,
            "match_date": "2020-07-01",
            "league": "G1",
        },
        {
            "season_start_year": 2020,
            "match_date": "2021-05-20",
            "league": "E0",
        },
        {
            "season_start_year": 2021,
            "match_date": "2021-07-01",
            "league": "G1",
        },
    ]
    frame = pd.DataFrame(rows)

    fold = build_nested_year_folds(
        frame,
        test_years=[2021],
        min_train_rows=1,
    )[0]

    train, tune, calibration, test = split_nested_fold(frame, fold)

    assert pd.to_datetime(train["match_date"]).max() < pd.to_datetime(
        tune["match_date"]
    ).min()
    assert pd.to_datetime(tune["match_date"]).max() < pd.to_datetime(
        calibration["match_date"]
    ).min()
    assert pd.to_datetime(calibration["match_date"]).max() < pd.to_datetime(
        test["match_date"]
    ).min()

    # The delayed 2019/20 match must not enter tune model selection.
    assert "2020-07-20" not in set(tune["match_date"].astype(str))
