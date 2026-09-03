from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import xgboost as xgb

from src.experiments.ah_settlement_engine_audit import LABELS, add_settlement_columns
from src.features.contextual_features import assert_no_closing_columns, build_contextual_features, is_closing_column


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

LAYER1 = ["E0", "D1", "I1", "SP1", "F1", "P1"]
LAYER2 = ["N1", "B1", "T1", "G1", "E1", "E2", "E3"]
LEAGUES = LAYER1 + LAYER2
ENGLISH_LOWER = {"E1", "E2", "E3"}
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
FEATURES = [
    "AHh",
    "AvgAHH",
    "AvgAHA",
    "avg_ah_AvgAHH_no_vig_probability",
    "avg_ah_AvgAHA_no_vig_probability",
]
TARGET_COLUMN = "target_ah_home_cover"
EDGE_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05]
SIDES = ["home", "away", "best_positive"]
MIN_ODDS = [None, 1.80, 1.85, 1.90]

REPORT_PATH = Path("outputs/reports/layer1_layer2_locked_ah_value_review.md")
PREDICTIONS_PATH = Path("outputs/reports/layer1_layer2_locked_ah_value_predictions.csv")
FIXED_RULES_PATH = Path("outputs/reports/layer1_layer2_locked_ah_value_fixed_rules.csv")
NESTED_PATH = Path("outputs/reports/layer1_layer2_locked_ah_value_nested_selection.csv")
BY_LEAGUE_PATH = Path("outputs/reports/layer1_layer2_locked_ah_value_by_league.csv")
BY_SEASON_PATH = Path("outputs/reports/layer1_layer2_locked_ah_value_by_season.csv")
CONTROLS_PATH = Path("outputs/reports/layer1_layer2_locked_ah_value_controls.csv")
ROBUSTNESS_PATH = Path("outputs/reports/layer1_layer2_locked_ah_value_robustness.csv")


@dataclass(frozen=True)
class Rule:
    side: str
    edge_threshold: float
    min_odds: float | None

    @property
    def name(self) -> str:
        odds = "no_odds_filter" if self.min_odds is None else f"odds_ge_{self.min_odds:.2f}".replace(".", "_")
        return f"{self.side}_edge_ge_{self.edge_threshold:.2f}_{odds}".replace(".", "_")


def rule_grid() -> list[Rule]:
    return [Rule(side, threshold, min_odds) for threshold in EDGE_THRESHOLDS for side in SIDES for min_odds in MIN_ODDS]


def load_contextual_matches() -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["league"] = league
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No processed match files found for locked value review")
    matches = pd.concat(frames, ignore_index=True, sort=False)
    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce").dt.normalize()
    for column in ["season_end_year", "FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA", "AvgCAHH", "AvgCAHA"]:
        if column in matches.columns:
            matches[column] = pd.to_numeric(matches[column], errors="coerce")
    contextual = build_contextual_features(matches)
    settled = add_settlement_columns(contextual)
    settled["target_ah_home_cover"] = settled["home_binary_target_previous"]
    settled["season_end_year"] = pd.to_numeric(settled["season_end_year"], errors="coerce")
    settled["row_id"] = np.arange(len(settled))
    required = ["league", "Date", "season_end_year", "FTHG", "FTAG"] + FEATURES
    settled = settled.dropna(subset=required).copy()
    settled = settled[settled["season_end_year"] >= 2020].copy()
    settled = settled[settled["valid_ah_settlement"]].copy()
    if {"AvgCAHH", "AvgCAHA"}.issubset(settled.columns):
        close_home_raw = 1.0 / pd.to_numeric(settled["AvgCAHH"], errors="coerce")
        close_away_raw = 1.0 / pd.to_numeric(settled["AvgCAHA"], errors="coerce")
        close_total = close_home_raw + close_away_raw
        settled["close_home_no_vig_probability"] = close_home_raw / close_total
        settled["close_away_no_vig_probability"] = close_away_raw / close_total
    else:
        settled["close_home_no_vig_probability"] = np.nan
        settled["close_away_no_vig_probability"] = np.nan
    return settled.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def training_frame(frame: pd.DataFrame, excluded_seasons: set[int] | None = None) -> pd.DataFrame:
    excluded_seasons = excluded_seasons or set()
    trainable = frame[~frame["season_end_year"].isin(excluded_seasons)].copy()
    return trainable.dropna(subset=[TARGET_COLUMN] + FEATURES).copy()


def fold_data(frame: pd.DataFrame, test_year: int, excluded_seasons: set[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trainable = training_frame(frame, excluded_seasons=excluded_seasons)
    validation_year = test_year - 1
    train = trainable[trainable["season_end_year"] < validation_year].copy()
    validation = trainable[trainable["season_end_year"] == validation_year].copy()
    test = frame[(frame["season_end_year"] == test_year) & ~frame["season_end_year"].isin(excluded_seasons or set())].copy()
    return train, validation, test


def fit_predict_frozen(train: pd.DataFrame, validation: pd.DataFrame, target_frame: pd.DataFrame) -> np.ndarray:
    assert_no_closing_columns(FEATURES)
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURES])
    x_validation = imputer.transform(validation[FEATURES])
    x_target = imputer.transform(target_frame[FEATURES])
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 2,
        "eta": 0.03,
        "lambda": 8.0,
        "alpha": 2.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "verbosity": 0,
    }
    model = xgb.train(
        params,
        xgb.DMatrix(x_train, label=train[TARGET_COLUMN].astype(int).to_numpy(), feature_names=FEATURES),
        num_boost_round=250,
        evals=[(xgb.DMatrix(x_validation, label=validation[TARGET_COLUMN].astype(int).to_numpy(), feature_names=FEATURES), "validation")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return np.clip(model.predict(xgb.DMatrix(x_target, feature_names=FEATURES)), 1e-6, 1 - 1e-6)


def add_prediction_columns(frame: pd.DataFrame, probabilities: np.ndarray, fold_kind: str, fold_test_year: int) -> pd.DataFrame:
    output = frame.copy()
    output["fold_kind"] = fold_kind
    output["fold_test_year"] = int(fold_test_year)
    output["raw_market_home_prob_no_vig"] = output["avg_ah_AvgAHH_no_vig_probability"].astype(float)
    output["raw_market_away_prob_no_vig"] = output["avg_ah_AvgAHA_no_vig_probability"].astype(float)
    output["model_home_cover_prob"] = probabilities
    output["model_away_cover_score"] = 1.0 - output["model_home_cover_prob"]
    output["home_edge_score"] = output["model_home_cover_prob"] - output["raw_market_home_prob_no_vig"]
    output["away_edge_score"] = output["model_away_cover_score"] - output["raw_market_away_prob_no_vig"]
    output["home_clv"] = output["close_home_no_vig_probability"] - output["raw_market_home_prob_no_vig"]
    output["away_clv"] = output["close_away_no_vig_probability"] - output["raw_market_away_prob_no_vig"]
    return output


def generate_predictions(frame: pd.DataFrame, excluded_seasons: set[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_predictions = []
    validation_predictions = []
    for test_year in TEST_YEARS:
        train, validation, test = fold_data(frame, test_year, excluded_seasons=excluded_seasons)
        if len(train) < 3000 or len(validation) < 500 or len(test) < 250:
            continue
        validation_prob = fit_predict_frozen(train, validation, validation)
        test_prob = fit_predict_frozen(train, validation, test)
        validation_predictions.append(add_prediction_columns(validation, validation_prob, "validation", test_year))
        test_predictions.append(add_prediction_columns(test, test_prob, "test", test_year))
    return (
        pd.concat(test_predictions, ignore_index=True, sort=False) if test_predictions else pd.DataFrame(),
        pd.concat(validation_predictions, ignore_index=True, sort=False) if validation_predictions else pd.DataFrame(),
    )


def selected_side_frame(predictions: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    rows = []
    if rule.side == "home":
        selected = predictions[predictions["home_edge_score"] >= rule.edge_threshold].copy()
        selected["selected_side"] = "home"
    elif rule.side == "away":
        selected = predictions[predictions["away_edge_score"] >= rule.edge_threshold].copy()
        selected["selected_side"] = "away"
    elif rule.side == "best_positive":
        selected = predictions[
            (predictions[["home_edge_score", "away_edge_score"]].max(axis=1) >= rule.edge_threshold)
        ].copy()
        selected["selected_side"] = np.where(selected["home_edge_score"] >= selected["away_edge_score"], "home", "away")
    else:
        raise ValueError(rule.side)
    if selected.empty:
        return selected
    selected["selected_edge_score"] = np.where(selected["selected_side"].eq("home"), selected["home_edge_score"], selected["away_edge_score"])
    selected["selected_odds"] = np.where(selected["selected_side"].eq("home"), selected["AvgAHH"], selected["AvgAHA"])
    selected["selected_profit"] = np.where(
        selected["selected_side"].eq("home"),
        selected["home_ah_profit_at_avg_odds"],
        selected["away_ah_profit_at_avg_odds"],
    )
    selected["selected_settlement_label"] = np.where(
        selected["selected_side"].eq("home"),
        selected["home_ah_settlement_label"],
        selected["away_ah_settlement_label"],
    )
    selected["selected_clv"] = np.where(selected["selected_side"].eq("home"), selected["home_clv"], selected["away_clv"])
    if rule.min_odds is not None:
        selected = selected[selected["selected_odds"] >= rule.min_odds].copy()
    selected["rule_name"] = rule.name
    selected["rule_side"] = rule.side
    selected["rule_edge_threshold"] = rule.edge_threshold
    selected["rule_min_odds"] = rule.min_odds if rule.min_odds is not None else np.nan
    rows.append(selected)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def z_score(profits: pd.Series) -> float:
    values = pd.to_numeric(profits, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return 0.0
    std = values.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(values.sum() / (std * np.sqrt(len(values))))


def max_drawdown(profits: pd.Series) -> float:
    values = pd.to_numeric(profits, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(np.insert(cumulative, 0, 0.0))[1:]
    return float(np.max(peak - cumulative)) if len(cumulative) else 0.0


def hhi(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    shares = series.value_counts(normalize=True)
    return float((shares * shares).sum())


def summarize_selection(selected: pd.DataFrame, scope: str, rule: Rule | None = None, control: str = "") -> dict[str, object]:
    row = {
        "scope": scope,
        "rule_name": rule.name if rule else "",
        "rule_side": rule.side if rule else "",
        "edge_threshold": rule.edge_threshold if rule else np.nan,
        "min_odds": rule.min_odds if rule and rule.min_odds is not None else np.nan,
        "control": control,
        "bets": int(len(selected)),
    }
    if selected.empty:
        row.update(
            {
                "profit": 0.0,
                "roi": 0.0,
                "z_score": 0.0,
                "max_drawdown": 0.0,
                "average_odds": np.nan,
                "average_edge_score": np.nan,
                "average_clv": np.nan,
                "clv_positive_percentage": np.nan,
                "league_hhi": np.nan,
                "home_team_hhi": np.nan,
                "away_team_hhi": np.nan,
            }
        )
        for label in LABELS:
            row[f"{label}_count"] = 0
        return row
    labels = selected["selected_settlement_label"].value_counts()
    row.update(
        {
            "profit": float(selected["selected_profit"].sum()),
            "roi": float(selected["selected_profit"].mean()),
            "z_score": z_score(selected["selected_profit"]),
            "max_drawdown": max_drawdown(selected.sort_values("Date")["selected_profit"]),
            "average_odds": float(selected["selected_odds"].mean()),
            "average_edge_score": float(selected["selected_edge_score"].mean()),
            "average_clv": float(selected["selected_clv"].mean(skipna=True)),
            "clv_positive_percentage": float((selected["selected_clv"] > 0).mean()) if selected["selected_clv"].notna().any() else np.nan,
            "league_hhi": hhi(selected["league"]),
            "home_team_hhi": hhi(selected["HomeTeam"]) if "HomeTeam" in selected.columns else np.nan,
            "away_team_hhi": hhi(selected["AwayTeam"]) if "AwayTeam" in selected.columns else np.nan,
        }
    )
    for label in LABELS:
        row[f"{label}_count"] = int(labels.get(label, 0))
    return row


def fixed_rule_review(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selections = []
    for rule in rule_grid():
        selected = selected_side_frame(predictions, rule)
        rows.append(summarize_selection(selected, "fixed_rule", rule))
        if len(selected):
            selections.append(selected)
    return pd.DataFrame(rows), pd.concat(selections, ignore_index=True, sort=False) if selections else pd.DataFrame()


def apply_rule_by_name(predictions: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    for rule in rule_grid():
        if rule.name == rule_name:
            return selected_side_frame(predictions, rule)
    raise ValueError(rule_name)


def nested_selection_review(test_predictions: pd.DataFrame, validation_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selected_tests = []
    for test_year in TEST_YEARS:
        validation = validation_predictions[validation_predictions["fold_test_year"].eq(test_year)].copy()
        test = test_predictions[test_predictions["fold_test_year"].eq(test_year)].copy()
        if validation.empty or test.empty:
            continue
        candidates = []
        for rule in rule_grid():
            selected_validation = selected_side_frame(validation, rule)
            summary = summarize_selection(selected_validation, "nested_validation", rule)
            if (
                summary["bets"] >= 200
                and summary["roi"] > 0
                and summary["z_score"] > 1.0
                and (pd.isna(summary["average_clv"]) or summary["average_clv"] > 0)
            ):
                candidates.append(summary)
        if not candidates:
            rows.append(
                {
                    "test_year": int(test_year),
                    "selected_rule": "",
                    "selection_status": "no_validation_rule_passed",
                    "validation_bets": 0,
                    "validation_profit": 0.0,
                    "validation_roi": 0.0,
                    "validation_z_score": 0.0,
                    "test_bets": 0,
                    "test_profit": 0.0,
                    "test_roi": 0.0,
                    "test_z_score": 0.0,
                }
            )
            continue
        chosen = pd.DataFrame(candidates).sort_values(["roi", "z_score", "bets"], ascending=[False, False, False]).iloc[0].to_dict()
        selected_test = apply_rule_by_name(test, str(chosen["rule_name"]))
        test_summary = summarize_selection(selected_test, "nested_test", control="nested_selected")
        selected_test["nested_test_year"] = int(test_year)
        selected_tests.append(selected_test)
        rows.append(
            {
                "test_year": int(test_year),
                "selected_rule": chosen["rule_name"],
                "selection_status": "selected",
                "validation_bets": int(chosen["bets"]),
                "validation_profit": float(chosen["profit"]),
                "validation_roi": float(chosen["roi"]),
                "validation_z_score": float(chosen["z_score"]),
                "validation_average_clv": float(chosen["average_clv"]) if pd.notna(chosen["average_clv"]) else np.nan,
                "test_bets": int(test_summary["bets"]),
                "test_profit": float(test_summary["profit"]),
                "test_roi": float(test_summary["roi"]),
                "test_z_score": float(test_summary["z_score"]),
                "test_average_clv": float(test_summary["average_clv"]) if pd.notna(test_summary["average_clv"]) else np.nan,
                "test_clv_positive_percentage": float(test_summary["clv_positive_percentage"]) if pd.notna(test_summary["clv_positive_percentage"]) else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.concat(selected_tests, ignore_index=True, sort=False) if selected_tests else pd.DataFrame()


def random_same_size(predictions: pd.DataFrame, selected: pd.DataFrame, seed: int, side: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    if selected.empty:
        return selected.copy()
    for (league, season), group in selected.groupby(["league", "season_end_year"]):
        pool = predictions[(predictions["league"].eq(league)) & (predictions["season_end_year"].eq(season))].copy()
        if side == "away":
            pool["selected_side"] = "away"
        elif side == "home":
            pool["selected_side"] = "home"
        else:
            pool["selected_side"] = rng.choice(["home", "away"], size=len(pool))
        n = min(len(group), len(pool))
        if n <= 0:
            continue
        sample = pool.sample(n=n, replace=False, random_state=int(rng.integers(0, 1_000_000))).copy()
        sample["selected_edge_score"] = np.where(sample["selected_side"].eq("home"), sample["home_edge_score"], sample["away_edge_score"])
        sample["selected_odds"] = np.where(sample["selected_side"].eq("home"), sample["AvgAHH"], sample["AvgAHA"])
        sample["selected_profit"] = np.where(sample["selected_side"].eq("home"), sample["home_ah_profit_at_avg_odds"], sample["away_ah_profit_at_avg_odds"])
        sample["selected_settlement_label"] = np.where(sample["selected_side"].eq("home"), sample["home_ah_settlement_label"], sample["away_ah_settlement_label"])
        sample["selected_clv"] = np.where(sample["selected_side"].eq("home"), sample["home_clv"], sample["away_clv"])
        pieces.append(sample)
    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()


def opposite_side(selected: pd.DataFrame) -> pd.DataFrame:
    output = selected.copy()
    if output.empty:
        return output
    output["selected_side"] = np.where(output["selected_side"].eq("home"), "away", "home")
    output["selected_edge_score"] = np.where(output["selected_side"].eq("home"), output["home_edge_score"], output["away_edge_score"])
    output["selected_odds"] = np.where(output["selected_side"].eq("home"), output["AvgAHH"], output["AvgAHA"])
    output["selected_profit"] = np.where(output["selected_side"].eq("home"), output["home_ah_profit_at_avg_odds"], output["away_ah_profit_at_avg_odds"])
    output["selected_settlement_label"] = np.where(output["selected_side"].eq("home"), output["home_ah_settlement_label"], output["away_ah_settlement_label"])
    output["selected_clv"] = np.where(output["selected_side"].eq("home"), output["home_clv"], output["away_clv"])
    return output


def always_side(predictions: pd.DataFrame, side: str) -> pd.DataFrame:
    output = predictions.copy()
    output["selected_side"] = side
    output["selected_edge_score"] = np.where(side == "home", output["home_edge_score"], output["away_edge_score"])
    output["selected_odds"] = np.where(side == "home", output["AvgAHH"], output["AvgAHA"])
    output["selected_profit"] = np.where(side == "home", output["home_ah_profit_at_avg_odds"], output["away_ah_profit_at_avg_odds"])
    output["selected_settlement_label"] = np.where(side == "home", output["home_ah_settlement_label"], output["away_ah_settlement_label"])
    output["selected_clv"] = np.where(side == "home", output["home_clv"], output["away_clv"])
    return output


def control_review(predictions: pd.DataFrame, selected: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    rows.append(summarize_selection(always_side(predictions, "home"), "control", control=f"{label}_always_home_ah"))
    rows.append(summarize_selection(always_side(predictions, "away"), "control", control=f"{label}_always_away_ah"))
    rows.append(summarize_selection(opposite_side(selected), "control", control=f"{label}_opposite_side_same_matches"))
    for draw in range(20):
        rows.append(summarize_selection(random_same_size(predictions, selected, 20260630 + draw), "control", control=f"{label}_random_same_size"))
        rows.append(summarize_selection(random_same_size(predictions, selected, 20260730 + draw, side="away"), "control", control=f"{label}_random_away_same_size"))
    rng = np.random.default_rng(42)
    shuffled = predictions.copy()
    shuffled["home_edge_score"] = rng.permutation(shuffled["home_edge_score"].to_numpy())
    shuffled["away_edge_score"] = -shuffled["home_edge_score"]
    if len(selected) and "rule_name" in selected.columns:
        rule_name = str(selected["rule_name"].mode().iloc[0])
        rows.append(summarize_selection(apply_rule_by_name(shuffled, rule_name), "control", control=f"{label}_shuffled_model_edge_scores"))
        permuted = predictions.copy()
        for _, idx in permuted.groupby(["league", "season_end_year"]).groups.items():
            values = permuted.loc[idx, "home_edge_score"].to_numpy(copy=True)
            rng.shuffle(values)
            permuted.loc[idx, "home_edge_score"] = values
            permuted.loc[idx, "away_edge_score"] = -values
        rows.append(summarize_selection(apply_rule_by_name(permuted, rule_name), "control", control=f"{label}_permuted_edge_within_league_season"))
    market = predictions.copy()
    market["home_edge_score"] = market["raw_market_home_prob_no_vig"] - 0.5
    market["away_edge_score"] = market["raw_market_away_prob_no_vig"] - 0.5
    top = market.nlargest(len(selected), ["home_edge_score"]).copy() if len(selected) else market.iloc[0:0].copy()
    if len(top):
        top["selected_side"] = "home"
        top["selected_edge_score"] = top["home_edge_score"]
        top["selected_odds"] = top["AvgAHH"]
        top["selected_profit"] = top["home_ah_profit_at_avg_odds"]
        top["selected_settlement_label"] = top["home_ah_settlement_label"]
        top["selected_clv"] = top["home_clv"]
    rows.append(summarize_selection(top, "control", control=f"{label}_no_vig_market_probability_only"))
    calibrated = top.copy()
    rows.append(summarize_selection(calibrated, "control", control=f"{label}_market_only_calibration_score"))
    random_side_top = selected.copy()
    if len(random_side_top):
        random_side_top["selected_side"] = rng.choice(["home", "away"], size=len(random_side_top))
        random_side_top = opposite_side(opposite_side(random_side_top))
    rows.append(summarize_selection(random_side_top, "control", control=f"{label}_top_edge_random_side"))
    return pd.DataFrame(rows)


def by_group(selected: pd.DataFrame, group_col: str, scope: str) -> pd.DataFrame:
    rows = []
    if selected.empty:
        return pd.DataFrame()
    for key, group in selected.groupby(group_col):
        row = summarize_selection(group, scope)
        row[group_col] = key
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_review(predictions: pd.DataFrame, selected: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    if selected.empty:
        return pd.DataFrame()
    by_season = selected.groupby("season_end_year")["selected_profit"].sum()
    by_league = selected.groupby("league")["selected_profit"].sum()
    best_season = int(by_season.idxmax())
    worst_season = int(by_season.idxmin())
    best_league = str(by_league.idxmax())
    exclusions: list[tuple[str, pd.Series]] = [
        ("exclude_best_profit_season", selected["season_end_year"].ne(best_season)),
        ("exclude_worst_profit_season", selected["season_end_year"].ne(worst_season)),
        ("exclude_best_profit_league", selected["league"].ne(best_league)),
        ("exclude_layer1", ~selected["league"].isin(LAYER1)),
        ("exclude_layer2", ~selected["league"].isin(LAYER2)),
        ("exclude_english_lower", ~selected["league"].isin(ENGLISH_LOWER)),
        ("exclude_2026", selected["season_end_year"].ne(2026)),
    ]
    exclusions.extend((f"exclude_league_{league}", selected["league"].ne(league)) for league in LEAGUES)
    for control, mask in exclusions:
        row = summarize_selection(selected[mask].copy(), "robustness", control=f"{label}_{control}")
        rows.append(row)
    return pd.DataFrame(rows)


def save_predictions(predictions: pd.DataFrame) -> None:
    keep = [
        "league",
        "season_end_year",
        "Date",
        "AHh",
        "AvgAHH",
        "AvgAHA",
        "raw_market_home_prob_no_vig",
        "raw_market_away_prob_no_vig",
        "model_home_cover_prob",
        "model_away_cover_score",
        "home_edge_score",
        "away_edge_score",
        "home_ah_profit_at_avg_odds",
        "away_ah_profit_at_avg_odds",
        "home_ah_settlement_label",
        "away_ah_settlement_label",
        "AvgCAHH",
        "AvgCAHA",
        "close_home_no_vig_probability",
        "close_away_no_vig_probability",
        "home_clv",
        "away_clv",
        "fold_test_year",
    ]
    predictions[[column for column in keep if column in predictions.columns]].to_csv(PREDICTIONS_PATH, index=False)


def classify(nested_selected: pd.DataFrame, controls: pd.DataFrame, robustness: pd.DataFrame, warnings_out: list[str]) -> str:
    if warnings_out:
        return "reject"
    if nested_selected.empty:
        return "predictive_only_no_value"
    nested_summary = summarize_selection(nested_selected, "nested_portfolio")
    if nested_summary["profit"] <= 0 or nested_summary["roi"] <= 0:
        return "predictive_only_no_value"
    if nested_summary["z_score"] < 1.5:
        return "value_research_only"
    if pd.notna(nested_summary["average_clv"]) and (
        nested_summary["average_clv"] <= 0 or nested_summary["clv_positive_percentage"] < 0.52
    ):
        return "value_research_only"
    control_agg = controls.groupby("control")["roi"].mean() if len(controls) else pd.Series(dtype=float)
    if len(control_agg) and nested_summary["roi"] <= control_agg.max():
        return "value_research_only"
    robust = robustness[robustness["control"].str.startswith("nested_", na=False)].copy()
    required = ["exclude_best_profit_season", "exclude_best_profit_league"]
    for key in required:
        subset = robust[robust["control"].str.contains(key, regex=False)]
        if len(subset) and float(subset["profit"].iloc[0]) <= 0:
            return "value_research_only"
    return "forward_paper_tracking_candidate"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 60) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def leakage_warnings() -> list[str]:
    warnings_out = []
    closing = [feature for feature in FEATURES if is_closing_column(feature)]
    if closing:
        warnings_out.append("closing feature in model features: " + ",".join(closing))
    if any("_tm_" in feature for feature in FEATURES):
        warnings_out.append("Transfermarkt feature in model features")
    if any(feature in {"HomeTeam", "AwayTeam"} for feature in FEATURES):
        warnings_out.append("team-name feature in model features")
    return warnings_out


def write_report(
    fixed_rules: pd.DataFrame,
    nested: pd.DataFrame,
    nested_selected: pd.DataFrame,
    controls: pd.DataFrame,
    robustness: pd.DataFrame,
    warnings_out: list[str],
    classification: str,
) -> None:
    fixed_best = fixed_rules.sort_values(["profit", "z_score"], ascending=[False, False]).head(20)
    nested_summary = pd.DataFrame([summarize_selection(nested_selected, "nested_portfolio")])
    control_summary = controls.groupby("control").agg(bets=("bets", "mean"), profit=("profit", "mean"), roi=("roi", "mean"), z_score=("z_score", "mean")).reset_index() if len(controls) else pd.DataFrame()
    lines = [
        "# Layer 1 + Layer 2 Locked AH Value Review",
        "",
        f"Final classification: `{classification}`",
        "",
        "Scope: historical locked value review using the frozen AH market-only model predictions as a selection score and the verified AH settlement engine for all profit/ROI calculations. No new model families, new features, Transfermarkt/player/lineup features, team-name model features, closing-odds selection features, live betting, or confirmed edge claim were used.",
        "",
        "## Leakage And Settlement Checks",
        "",
        markdown_table(pd.DataFrame({"warning": warnings_out or ["none"]}), ["warning"]),
        "",
        "## Best Fixed Rules",
        "",
        markdown_table(fixed_best, ["rule_name", "bets", "profit", "roi", "z_score", "max_drawdown", "average_odds", "average_edge_score", "average_clv", "clv_positive_percentage", "league_hhi"], max_rows=20),
        "",
        "## Nested Selection",
        "",
        markdown_table(nested, ["test_year", "selected_rule", "selection_status", "validation_bets", "validation_roi", "validation_z_score", "test_bets", "test_profit", "test_roi", "test_z_score", "test_average_clv", "test_clv_positive_percentage"], max_rows=20),
        "",
        "## Nested Portfolio Aggregate",
        "",
        markdown_table(nested_summary, ["bets", "profit", "roi", "z_score", "max_drawdown", "average_odds", "average_edge_score", "average_clv", "clv_positive_percentage", "league_hhi"], max_rows=10),
        "",
        "## Controls",
        "",
        markdown_table(control_summary.sort_values("roi", ascending=False) if len(control_summary) else control_summary, ["control", "bets", "profit", "roi", "z_score"], max_rows=80),
        "",
        "## Robustness",
        "",
        markdown_table(robustness, ["scope", "control", "bets", "profit", "roi", "z_score", "average_clv", "clv_positive_percentage"], max_rows=80),
        "",
        "No confirmed edge is claimed. This remains historical research only.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    warnings_out = leakage_warnings()
    frame = load_contextual_matches()
    predictions, validation_predictions = generate_predictions(frame)
    save_predictions(predictions)
    fixed_rules, fixed_selections = fixed_rule_review(predictions)
    nested, nested_selected = nested_selection_review(predictions, validation_predictions)
    best_rule = fixed_rules.sort_values(["profit", "z_score"], ascending=[False, False]).iloc[0]["rule_name"] if len(fixed_rules) else ""
    best_fixed_selected = apply_rule_by_name(predictions, str(best_rule)) if best_rule else pd.DataFrame()
    controls = pd.concat(
        [
            control_review(predictions, best_fixed_selected, "best_fixed"),
            control_review(predictions, nested_selected, "nested") if len(nested_selected) else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    robustness = pd.concat(
        [
            robustness_review(predictions, best_fixed_selected, "best_fixed"),
            robustness_review(predictions, nested_selected, "nested") if len(nested_selected) else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    # Re-run the nested portfolio with 2021 excluded from training/validation history.
    predictions_no_2021, validation_no_2021 = generate_predictions(frame, excluded_seasons={2021})
    nested_no_2021, selected_no_2021 = nested_selection_review(predictions_no_2021, validation_no_2021)
    if len(selected_no_2021):
        row = summarize_selection(selected_no_2021, "robustness", control="nested_exclude_2021_from_training_validation")
        robustness = pd.concat([robustness, pd.DataFrame([row])], ignore_index=True, sort=False)
    fixed_rules.to_csv(FIXED_RULES_PATH, index=False)
    nested.to_csv(NESTED_PATH, index=False)
    by_league = pd.concat(
        [
            by_group(best_fixed_selected, "league", "best_fixed_by_league"),
            by_group(nested_selected, "league", "nested_by_league") if len(nested_selected) else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    by_season = pd.concat(
        [
            by_group(best_fixed_selected, "season_end_year", "best_fixed_by_season"),
            by_group(nested_selected, "season_end_year", "nested_by_season") if len(nested_selected) else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    by_league.to_csv(BY_LEAGUE_PATH, index=False)
    by_season.to_csv(BY_SEASON_PATH, index=False)
    controls.to_csv(CONTROLS_PATH, index=False)
    robustness.to_csv(ROBUSTNESS_PATH, index=False)
    classification = classify(nested_selected, controls, robustness, warnings_out)
    write_report(fixed_rules, nested, nested_selected, controls, robustness, warnings_out, classification)
    print(
        {
            "prediction_rows": len(predictions),
            "fixed_rule_rows": len(fixed_rules),
            "nested_rows": len(nested),
            "nested_bets": int(len(nested_selected)),
            "control_rows": len(controls),
            "robustness_rows": len(robustness),
            "classification": classification,
        }
    )


if __name__ == "__main__":
    main()
