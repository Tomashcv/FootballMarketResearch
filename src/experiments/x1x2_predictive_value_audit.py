from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.metrics import expected_calibration_error


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)

LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "E1", "E2", "E3", "SC0"]
LAYER1 = {"E0", "D1", "I1", "SP1", "F1", "P1"}
LAYER2 = {"N1", "B1", "T1", "G1", "E1", "E2", "E3"}
ENGLISH_LOWER = {"E1", "E2", "E3"}
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
CLASSES = ["H", "D", "A"]
CLASS_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_CLASS = {0: "H", 1: "D", 2: "A"}

REPORT_PATH = Path("outputs/reports/x1x2_predictive_audit.md")
SUMMARY_PATH = Path("outputs/reports/x1x2_predictive_summary.csv")
CLASS_CAL_PATH = Path("outputs/reports/x1x2_class_calibration.csv")
EDGE_BUCKET_PATH = Path("outputs/reports/x1x2_edge_bucket_calibration.csv")
NEGATIVE_PATH = Path("outputs/reports/x1x2_negative_controls.csv")
ROBUSTNESS_PATH = Path("outputs/reports/x1x2_robustness.csv")
VALUE_REPORT_PATH = Path("outputs/reports/x1x2_value_review.md")
VALUE_FIXED_PATH = Path("outputs/reports/x1x2_value_fixed_rules.csv")
VALUE_NESTED_PATH = Path("outputs/reports/x1x2_value_nested_selection.csv")
VALUE_CONTROLS_PATH = Path("outputs/reports/x1x2_value_controls.csv")
VALUE_ROBUSTNESS_PATH = Path("outputs/reports/x1x2_value_robustness.csv")

MODELS = [
    "raw_market_baseline",
    "market_baseline_calibration_only",
    "multinomial_logistic_l2",
    "multinomial_logistic_elasticnet",
    "xgboost_multiclass_shallow",
    "xgboost_multiclass_depth3_regularized",
    "xgboost_market_residual_multiclass",
]

RULE_GRID = [
    (0.01, 1.50),
    (0.015, 1.50),
    (0.02, 1.50),
    (0.03, 1.50),
    (0.04, 1.50),
    (0.05, 1.50),
    (0.02, 1.80),
    (0.03, 1.80),
    (0.04, 1.80),
    (0.05, 1.80),
    (0.02, 2.00),
    (0.03, 2.00),
    (0.04, 2.00),
    (0.05, 2.00),
    (0.02, 2.50),
    (0.03, 2.50),
    (0.04, 2.50),
    (0.05, 2.50),
]


@dataclass(frozen=True)
class EvalKey:
    model: str
    feature_group: str


def normalize_rows(probabilities: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0)
    return arr / arr.sum(axis=1, keepdims=True)


def multiclass_ece(y: np.ndarray, probabilities: np.ndarray) -> float:
    p = normalize_rows(probabilities)
    confidence = p.max(axis=1)
    predicted = p.argmax(axis=1)
    correct = (predicted == y).astype(float)
    return float(expected_calibration_error(correct, confidence))


def multiclass_brier(y: np.ndarray, probabilities: np.ndarray) -> float:
    p = normalize_rows(probabilities)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def coalesce(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        if column in frame.columns:
            out = out.fillna(pd.to_numeric(frame[column], errors="coerce"))
    return out


def load_base_data() -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frame["league"] = league
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No processed league match files found.")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    for column in ["season_end_year", "FTHG", "FTAG"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["avg_home_odds"] = coalesce(data, ["AvgH", "BbAvH"])
    data["avg_draw_odds"] = coalesce(data, ["AvgD", "BbAvD"])
    data["avg_away_odds"] = coalesce(data, ["AvgA", "BbAvA"])
    data["fallback_home_odds"] = coalesce(data, ["B365H", "MaxH", "BbMxH"])
    data["fallback_draw_odds"] = coalesce(data, ["B365D", "MaxD", "BbMxD"])
    data["fallback_away_odds"] = coalesce(data, ["B365A", "MaxA", "BbMxA"])
    data["avg_odds_source"] = np.where(data[["AvgH", "AvgD", "AvgA"]].notna().all(axis=1), "Avg", np.where(data[["BbAvH", "BbAvD", "BbAvA"]].notna().all(axis=1), "BbAv", ""))
    raw_home = 1.0 / data["avg_home_odds"]
    raw_draw = 1.0 / data["avg_draw_odds"]
    raw_away = 1.0 / data["avg_away_odds"]
    data["overround"] = raw_home + raw_draw + raw_away
    data["no_vig_home_probability"] = raw_home / data["overround"]
    data["no_vig_draw_probability"] = raw_draw / data["overround"]
    data["no_vig_away_probability"] = raw_away / data["overround"]
    data["home_draw_gap"] = data["no_vig_home_probability"] - data["no_vig_draw_probability"]
    data["home_away_gap"] = data["no_vig_home_probability"] - data["no_vig_away_probability"]
    data["draw_away_gap"] = data["no_vig_draw_probability"] - data["no_vig_away_probability"]
    data["target_1x2"] = np.select(
        [data["FTHG"].gt(data["FTAG"]), data["FTHG"].eq(data["FTAG"]), data["FTHG"].lt(data["FTAG"])],
        [0.0, 1.0, 2.0],
        default=np.nan,
    )
    required = [
        "Date",
        "league",
        "season_end_year",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "target_1x2",
        "avg_home_odds",
        "avg_draw_odds",
        "avg_away_odds",
        "no_vig_home_probability",
        "no_vig_draw_probability",
        "no_vig_away_probability",
    ]
    data = data.dropna(subset=required).copy()
    data = data[data["avg_home_odds"].gt(1.0) & data["avg_draw_odds"].gt(1.0) & data["avg_away_odds"].gt(1.0)].copy()
    data["season_end_year"] = data["season_end_year"].astype(int)
    data["target_1x2"] = data["target_1x2"].astype(int)
    return data.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def add_structure_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    probs = out[["no_vig_home_probability", "no_vig_draw_probability", "no_vig_away_probability"]]
    out["favourite_probability"] = probs.max(axis=1)
    out["underdog_probability"] = probs.min(axis=1)
    out["favourite_side_code"] = probs.to_numpy().argmax(axis=1).astype(float)
    out["home_is_favourite"] = out["favourite_side_code"].eq(0.0).astype(float)
    out["away_is_favourite"] = out["favourite_side_code"].eq(2.0).astype(float)
    out["draw_probability_bucket"] = pd.cut(out["no_vig_draw_probability"], bins=[0.0, 0.22, 0.25, 0.28, 0.31, 0.35, 1.0], labels=False).astype(float)
    out["league_layer"] = np.where(out["league"].isin(LAYER1), 1.0, np.where(out["league"].isin(LAYER2), 2.0, 3.0))
    out["season_era"] = np.select(
        [
            out["season_end_year"].lt(2012),
            out["season_end_year"].between(2012, 2016),
            out["season_end_year"].between(2017, 2019),
            out["season_end_year"].ge(2020),
        ],
        [0.0, 1.0, 2.0, 3.0],
        default=np.nan,
    )
    for side_code in [0, 1, 2]:
        out[f"favourite_side_{side_code}"] = out["favourite_side_code"].eq(float(side_code)).astype(float)
    for bucket in range(6):
        out[f"draw_probability_bucket_{bucket}"] = out["draw_probability_bucket"].eq(float(bucket)).astype(float)
    for league in LEAGUES:
        out[f"league_code_{league}"] = out["league"].eq(league).astype(float)
    for layer in [1, 2, 3]:
        out[f"league_layer_{layer}"] = out["league_layer"].eq(float(layer)).astype(float)
    for era in [0, 1, 2, 3]:
        out[f"season_era_{era}"] = out["season_era"].eq(float(era)).astype(float)
    return out


def add_temporal_form_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).copy()
    out["match_order"] = np.arange(len(out))
    out["home_win_ind"] = out["target_1x2"].eq(0).astype(float)
    out["draw_ind"] = out["target_1x2"].eq(1).astype(float)
    out["away_win_ind"] = out["target_1x2"].eq(2).astype(float)
    for window in [50, 100, 200]:
        for col in ["home_win_ind", "draw_ind", "away_win_ind"]:
            out[f"league_rolling_{col}_{window}"] = out.groupby("league")[col].transform(lambda s: s.shift(1).rolling(window, min_periods=10).mean())
    rows = []
    for idx, row in out.iterrows():
        home_points = 3.0 if row["target_1x2"] == 0 else 1.0 if row["target_1x2"] == 1 else 0.0
        away_points = 3.0 if row["target_1x2"] == 2 else 1.0 if row["target_1x2"] == 1 else 0.0
        rows.append({"row_id": idx, "league": row["league"], "Date": row["Date"], "team": row["HomeTeam"], "venue": "home", "points": home_points, "win": float(row["target_1x2"] == 0), "draw": float(row["target_1x2"] == 1), "loss": float(row["target_1x2"] == 2), "gf": row["FTHG"], "ga": row["FTAG"]})
        rows.append({"row_id": idx, "league": row["league"], "Date": row["Date"], "team": row["AwayTeam"], "venue": "away", "points": away_points, "win": float(row["target_1x2"] == 2), "draw": float(row["target_1x2"] == 1), "loss": float(row["target_1x2"] == 0), "gf": row["FTAG"], "ga": row["FTHG"]})
    long = pd.DataFrame(rows).sort_values(["league", "team", "Date", "row_id"]).reset_index(drop=True)
    grouped = long.groupby(["league", "team"], sort=False)
    long["previous_match_date"] = grouped["Date"].shift(1)
    long["rest_days"] = (long["Date"] - long["previous_match_date"]).dt.days
    for window in [5, 10, 20]:
        for stat in ["points", "win", "draw", "loss", "gf", "ga"]:
            long[f"rolling_{stat}_{window}"] = grouped[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
        long[f"rolling_goal_difference_{window}"] = long[f"rolling_gf_{window}"] - long[f"rolling_ga_{window}"]
        long[f"venue_points_{window}"] = long.groupby(["league", "team", "venue"])["points"].transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
        long[f"venue_win_{window}"] = long.groupby(["league", "team", "venue"])["win"].transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
    home = long[long["venue"].eq("home")].set_index("row_id")
    away = long[long["venue"].eq("away")].set_index("row_id")
    for window in [5, 10, 20]:
        for stat in ["points", "win", "draw", "loss", "gf", "ga", "goal_difference"]:
            out[f"home_rolling_{stat}_{window}"] = home[f"rolling_{stat}_{window}"]
            out[f"away_rolling_{stat}_{window}"] = away[f"rolling_{stat}_{window}"]
            out[f"diff_rolling_{stat}_{window}"] = out[f"home_rolling_{stat}_{window}"] - out[f"away_rolling_{stat}_{window}"]
        out[f"home_team_home_only_points_{window}"] = home[f"venue_points_{window}"]
        out[f"home_team_home_only_win_{window}"] = home[f"venue_win_{window}"]
        out[f"away_team_away_only_points_{window}"] = away[f"venue_points_{window}"]
        out[f"away_team_away_only_win_{window}"] = away[f"venue_win_{window}"]
    out["home_rest_days"] = home["rest_days"]
    out["away_rest_days"] = away["rest_days"]
    out["rest_days_diff"] = out["home_rest_days"] - out["away_rest_days"]
    out["elo_not_implemented"] = 1.0
    out["row_id"] = np.arange(len(out))
    return out.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def load_data() -> pd.DataFrame:
    return add_temporal_form_features(add_structure_features(load_base_data()))


def feature_groups(data: pd.DataFrame) -> dict[str, list[str]]:
    market = [
        "avg_home_odds",
        "avg_draw_odds",
        "avg_away_odds",
        "no_vig_home_probability",
        "no_vig_draw_probability",
        "no_vig_away_probability",
        "overround",
        "home_draw_gap",
        "home_away_gap",
        "draw_away_gap",
    ]
    structure = market + [
        "favourite_probability",
        "underdog_probability",
        "home_is_favourite",
        "away_is_favourite",
    ]
    structure += [f"favourite_side_{i}" for i in [0, 1, 2]]
    structure += [f"draw_probability_bucket_{i}" for i in range(6)]
    structure += [f"league_code_{league}" for league in LEAGUES]
    structure += [f"league_layer_{i}" for i in [1, 2, 3]]
    structure += [f"season_era_{i}" for i in [0, 1, 2, 3]]
    temporal = structure.copy()
    for window in [50, 100, 200]:
        temporal += [f"league_rolling_home_win_ind_{window}", f"league_rolling_draw_ind_{window}", f"league_rolling_away_win_ind_{window}"]
    for window in [5, 10, 20]:
        for stat in ["points", "win", "draw", "loss", "gf", "ga", "goal_difference"]:
            temporal += [f"home_rolling_{stat}_{window}", f"away_rolling_{stat}_{window}", f"diff_rolling_{stat}_{window}"]
        temporal += [f"home_team_home_only_points_{window}", f"home_team_home_only_win_{window}", f"away_team_away_only_points_{window}", f"away_team_away_only_win_{window}"]
    temporal += ["home_rest_days", "away_rest_days", "rest_days_diff"]
    groups = {
        "x1_market_only": market,
        "x1_market_structure_safe": structure,
        "x1_market_plus_temporal_team_form": temporal,
    }
    return {name: [column for column in columns if column in data.columns] for name, columns in groups.items()}


def market_probabilities(frame: pd.DataFrame) -> np.ndarray:
    return normalize_rows(frame[["no_vig_home_probability", "no_vig_draw_probability", "no_vig_away_probability"]].to_numpy())


def metric_values(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    y = frame["target_1x2"].astype(int).to_numpy()
    p = normalize_rows(probabilities)
    return {"multiclass_log_loss": float(log_loss(y, p, labels=[0, 1, 2])), "brier": multiclass_brier(y, p), "ece": multiclass_ece(y, p)}


def fit_xgb_multiclass(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str], params: dict, rounds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if xgb is None:
        raise ImportError("xgboost is required for xgboost models.")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_val = imputer.transform(validation[features])
    x_test = imputer.transform(test[features])
    params = {**params, "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss", "seed": seed, "verbosity": 0, "tree_method": "hist", "nthread": 4}
    dtrain = xgb.DMatrix(x_train, label=train["target_1x2"].astype(int).to_numpy(), feature_names=features)
    dval = xgb.DMatrix(x_val, label=validation["target_1x2"].astype(int).to_numpy(), feature_names=features)
    model = xgb.train(params, dtrain, num_boost_round=rounds, evals=[(dval, "validation")], early_stopping_rounds=12, verbose_eval=False)
    return normalize_rows(model.predict(dval)), normalize_rows(model.predict(xgb.DMatrix(x_test, feature_names=features)))


def fit_residual_multiclass(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    if xgb is None:
        raise ImportError("xgboost is required for residual model.")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_val = imputer.transform(validation[features])
    x_test = imputer.transform(test[features])
    train_market = market_probabilities(train)
    val_market = market_probabilities(validation)
    test_market = market_probabilities(test)
    y_train = train["target_1x2"].astype(int).to_numpy()
    y_val = validation["target_1x2"].astype(int).to_numpy()
    y_train_oh = np.zeros_like(train_market)
    y_val_oh = np.zeros_like(val_market)
    y_train_oh[np.arange(len(y_train)), y_train] = 1.0
    y_val_oh[np.arange(len(y_val)), y_val] = 1.0
    val_res = []
    test_res = []
    for cls in [0, 1, 2]:
        params = {"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 2, "eta": 0.025, "lambda": 16.0, "alpha": 4.0, "subsample": 0.9, "colsample_bytree": 0.9, "seed": seed + cls, "verbosity": 0, "tree_method": "hist", "nthread": 4}
        dtrain = xgb.DMatrix(x_train, label=y_train_oh[:, cls] - train_market[:, cls], feature_names=features)
        dval = xgb.DMatrix(x_val, label=y_val_oh[:, cls] - val_market[:, cls], feature_names=features)
        booster = xgb.train(params, dtrain, num_boost_round=90, evals=[(dval, "validation")], early_stopping_rounds=12, verbose_eval=False)
        val_res.append(booster.predict(dval))
        test_res.append(booster.predict(xgb.DMatrix(x_test, feature_names=features)))
    return normalize_rows(val_market + np.column_stack(val_res)), normalize_rows(test_market + np.column_stack(test_res))


def fit_model(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, model: str, features: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    if model == "raw_market_baseline":
        return market_probabilities(validation), market_probabilities(test)
    if model == "market_baseline_calibration_only":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=seed, C=1.0))])
        cols = ["no_vig_home_probability", "no_vig_draw_probability", "no_vig_away_probability"]
        pipe.fit(validation[cols], validation["target_1x2"].astype(int))
        return normalize_rows(pipe.predict_proba(validation[cols])), normalize_rows(pipe.predict_proba(test[cols]))
    if model in {"multinomial_logistic_l2", "multinomial_logistic_elasticnet"}:
        penalty = "l2" if model == "multinomial_logistic_l2" else "elasticnet"
        solver = "lbfgs" if model == "multinomial_logistic_l2" else "saga"
        kwargs = {"l1_ratio": 0.2, "tol": 1e-3, "n_jobs": 4} if model == "multinomial_logistic_elasticnet" else {"tol": 1e-4}
        max_iter = 200 if model == "multinomial_logistic_elasticnet" else 1000
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=max_iter, random_state=seed, penalty=penalty, solver=solver, C=0.5, **kwargs))])
        pipe.fit(train[features], train["target_1x2"].astype(int))
        return normalize_rows(pipe.predict_proba(validation[features])), normalize_rows(pipe.predict_proba(test[features]))
    if model == "xgboost_multiclass_shallow":
        return fit_xgb_multiclass(train, validation, test, features, {"max_depth": 2, "eta": 0.035, "lambda": 12.0, "alpha": 2.0, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 20.0}, 90, seed)
    if model == "xgboost_multiclass_depth3_regularized":
        return fit_xgb_multiclass(train, validation, test, features, {"max_depth": 3, "eta": 0.03, "lambda": 18.0, "alpha": 4.0, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 25.0}, 100, seed)
    if model == "xgboost_market_residual_multiclass":
        return fit_residual_multiclass(train, validation, test, features, seed)
    raise ValueError(model)


def prediction_frame(frame: pd.DataFrame, model: str, feature_group: str, year: int, role: str, probabilities: np.ndarray) -> pd.DataFrame:
    out = frame[["row_id", "league", "season_end_year", "Date", "target_1x2", "avg_home_odds", "avg_draw_odds", "avg_away_odds", "no_vig_home_probability", "no_vig_draw_probability", "no_vig_away_probability"]].copy()
    p = normalize_rows(probabilities)
    out["model"] = model
    out["feature_group"] = feature_group
    out["test_year"] = year
    out["fold_role"] = role
    for idx, side in enumerate(["home", "draw", "away"]):
        out[f"model_{side}_probability"] = p[:, idx]
        out[f"{side}_edge"] = out[f"model_{side}_probability"] - out[f"no_vig_{side}_probability"]
    out["home_profit"] = np.where(out["target_1x2"].eq(0), out["avg_home_odds"] - 1.0, -1.0)
    out["draw_profit"] = np.where(out["target_1x2"].eq(1), out["avg_draw_odds"] - 1.0, -1.0)
    out["away_profit"] = np.where(out["target_1x2"].eq(2), out["avg_away_odds"] - 1.0, -1.0)
    return out


def run_predictions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = feature_groups(data)
    test_parts = []
    val_parts = []
    for fg, features in groups.items():
        for model in MODELS:
            for year in TEST_YEARS:
                print(f"predictive_fold model={model} feature_group={fg} test_year={year}", flush=True)
                train = data[data["season_end_year"].lt(year - 1)].copy()
                validation = data[data["season_end_year"].eq(year - 1)].copy()
                test = data[data["season_end_year"].eq(year)].copy()
                if len(train) == 0 or len(validation) == 0 or len(test) == 0 or train["target_1x2"].nunique() < 3:
                    continue
                val_p, test_p = fit_model(train, validation, test, model, features, 1000 + year)
                val_parts.append(prediction_frame(validation, model, fg, year, "validation", val_p))
                test_parts.append(prediction_frame(test, model, fg, year, "test", test_p))
    return pd.concat(test_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)


def pred_probs(frame: pd.DataFrame) -> np.ndarray:
    return frame[["model_home_probability", "model_draw_probability", "model_away_probability"]].to_numpy()


def raw_probs(frame: pd.DataFrame) -> np.ndarray:
    return frame[["no_vig_home_probability", "no_vig_draw_probability", "no_vig_away_probability"]].to_numpy()


def summarize_predictions(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pred.groupby(group_cols, dropna=False):
        m = metric_values(g, pred_probs(g))
        b = metric_values(g, raw_probs(g))
        improved = []
        for _, gy in g.groupby("test_year"):
            improved.append(metric_values(gy, pred_probs(gy))["multiclass_log_loss"] < metric_values(gy, raw_probs(gy))["multiclass_log_loss"])
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "rows": int(len(g)),
                "test_years": ";".join(map(str, sorted(g["test_year"].unique()))),
                "mean_delta_multiclass_log_loss_vs_raw_market": m["multiclass_log_loss"] - b["multiclass_log_loss"],
                "mean_delta_brier_vs_raw_market": m["brier"] - b["brier"],
                "mean_delta_ece_vs_raw_market": m["ece"] - b["ece"],
                "improved_years": int(sum(improved)),
                "model_multiclass_log_loss": m["multiclass_log_loss"],
                "market_multiclass_log_loss": b["multiclass_log_loss"],
                "model_brier": m["brier"],
                "market_brier": b["brier"],
                "model_ece": m["ece"],
                "market_ece": b["ece"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def binary_decompositions(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, fg), g in pred.groupby(["model", "feature_group"]):
        for label, cls, prob_col, market_col in [
            ("home_win_vs_not", 0, "model_home_probability", "no_vig_home_probability"),
            ("draw_vs_not", 1, "model_draw_probability", "no_vig_draw_probability"),
            ("away_win_vs_not", 2, "model_away_probability", "no_vig_away_probability"),
        ]:
            y = g["target_1x2"].eq(cls).astype(int).to_numpy()
            p = np.clip(g[prob_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
            mp = np.clip(g[market_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
            rows.append(
                {
                    "model": model,
                    "feature_group": fg,
                    "binary_decomposition": label,
                    "rows": int(len(g)),
                    "delta_log_loss_vs_raw_market": float(log_loss(y, p, labels=[0, 1]) - log_loss(y, mp, labels=[0, 1])),
                    "delta_brier_vs_raw_market": float(brier_score_loss(y, p) - brier_score_loss(y, mp)),
                    "delta_ece_vs_raw_market": float(expected_calibration_error(y, p) - expected_calibration_error(y, mp)),
                }
            )
    return pd.DataFrame(rows)


def class_calibration(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bucket_edges = np.linspace(0, 1, 11)
    for (model, fg), subset in pred.groupby(["model", "feature_group"]):
        for side, cls in [("home", 0), ("draw", 1), ("away", 2)]:
            prob_col = f"model_{side}_probability"
            target = subset["target_1x2"].eq(cls).astype(float)
            rows.append(
                {
                    "model": model,
                    "feature_group": fg,
                    "class": side,
                    "bucket": "all",
                    "rows": int(len(subset)),
                    "average_model_probability": float(subset[prob_col].mean()),
                    "realised_rate": float(target.mean()),
                    "calibration_error": float(target.mean() - subset[prob_col].mean()),
                }
            )
            tmp = subset.copy()
            tmp["bucket"] = pd.cut(tmp[prob_col], bins=bucket_edges, include_lowest=True).astype(str)
            for bucket, g in tmp.groupby("bucket"):
                if len(g) < 20:
                    continue
                t = g["target_1x2"].eq(cls).astype(float)
                rows.append(
                    {
                        "model": model,
                        "feature_group": fg,
                        "class": side,
                        "bucket": bucket,
                        "rows": int(len(g)),
                        "average_model_probability": float(g[prob_col].mean()),
                        "realised_rate": float(t.mean()),
                        "calibration_error": float(t.mean() - g[prob_col].mean()),
                    }
                )
    return pd.DataFrame(rows)


def edge_bucket_calibration(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = [-1e-9, 0.01, 0.02, 0.03, 0.04, 0.05, 10.0]
    labels = ["0.00-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05", ">=0.05"]
    for (model, fg), subset in pred.groupby(["model", "feature_group"]):
        for side, cls in [("home", 0), ("draw", 1), ("away", 2)]:
            tmp = subset.copy()
            tmp["edge_bucket"] = pd.cut(tmp[f"{side}_edge"], bins=bins, labels=labels).astype(str)
            for bucket, g in tmp.groupby("edge_bucket"):
                if bucket == "nan" or len(g) < 20:
                    continue
                target = g["target_1x2"].eq(cls).astype(float)
                rows.append(
                    {
                        "model": model,
                        "feature_group": fg,
                        "side": side,
                        "edge_bucket": bucket,
                        "rows": int(len(g)),
                        "average_model_probability": float(g[f"model_{side}_probability"].mean()),
                        "average_market_probability": float(g[f"no_vig_{side}_probability"].mean()),
                        "realised_hit_rate": float(target.mean()),
                        "calibration_error": float(target.mean() - g[f"model_{side}_probability"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def run_negative_controls(data: pd.DataFrame, key: EvalKey) -> pd.DataFrame:
    controls = ["shuffled_train_labels", "random_noise_replacing_market_features", "permuted_market_features_within_league_season", "league_only_without_market_odds", "opposite_label_sanity_check"]
    rows = []
    base_features = feature_groups(data)[key.feature_group]
    for control in controls:
        parts = []
        for year in TEST_YEARS:
            train = data[data["season_end_year"].lt(year - 1)].copy()
            validation = data[data["season_end_year"].eq(year - 1)].copy()
            test = data[data["season_end_year"].eq(year)].copy()
            if len(train) == 0 or len(validation) == 0 or len(test) == 0:
                continue
            original_test = test.copy()
            rng = np.random.default_rng(7000 + year)
            features = base_features.copy()
            if control == "shuffled_train_labels":
                y = train["target_1x2"].to_numpy(copy=True)
                rng.shuffle(y)
                train["target_1x2"] = y
            elif control == "random_noise_replacing_market_features":
                for frame in [train, validation, test]:
                    for column in features:
                        frame[column] = rng.normal(0, 1, len(frame))
            elif control == "permuted_market_features_within_league_season":
                for frame in [train, validation, test]:
                    for _, idx in frame.groupby(["league", "season_end_year"]).groups.items():
                        for column in features:
                            vals = frame.loc[idx, column].to_numpy(copy=True)
                            rng.shuffle(vals)
                            frame.loc[idx, column] = vals
            elif control == "league_only_without_market_odds":
                features = [f"league_code_{league}" for league in LEAGUES]
            elif control == "opposite_label_sanity_check":
                train["target_1x2"] = train["target_1x2"].map({0: 2, 1: 1, 2: 0}).astype(int)
            _, p = fit_model(train, validation, test, key.model, features, 8000 + year)
            parts.append(prediction_frame(original_test, key.model, key.feature_group, year, "test", p))
        if parts:
            row = summarize_predictions(pd.concat(parts, ignore_index=True).assign(control=control), ["control"]).iloc[0].to_dict()
            row["model"] = key.model
            row["feature_group"] = key.feature_group
            rows.append(row)
    return pd.DataFrame(rows)


def run_robustness(data: pd.DataFrame, key: EvalKey, best_season: int, best_league: str) -> pd.DataFrame:
    exclusions = [
        ("exclude_best_performing_season", lambda df: df[df["season_end_year"].ne(best_season)]),
        ("exclude_best_performing_league", lambda df: df[~df["league"].eq(best_league)]),
        ("exclude_pre_2012_data", lambda df: df[df["season_end_year"].ge(2012)]),
        ("exclude_pre_2020_training_data", lambda df: df[df["season_end_year"].ge(2020)]),
        ("exclude_2026", lambda df: df[df["season_end_year"].ne(2026)]),
        ("exclude_SC0", lambda df: df[~df["league"].eq("SC0")]),
        ("exclude_English_lower_leagues", lambda df: df[~df["league"].isin(ENGLISH_LOWER)]),
        ("exclude_Layer_1", lambda df: df[~df["league"].isin(LAYER1)]),
        ("exclude_Layer_2", lambda df: df[~df["league"].isin(LAYER2)]),
    ]
    exclusions.extend((f"exclude_{league}", lambda df, league=league: df[~df["league"].eq(league)]) for league in LEAGUES)
    rows = []
    for name, fn in exclusions:
        frame = fn(data.copy())
        parts = []
        features = feature_groups(frame)[key.feature_group]
        for year in TEST_YEARS:
            train = frame[frame["season_end_year"].lt(year - 1)].copy()
            validation = frame[frame["season_end_year"].eq(year - 1)].copy()
            test = frame[frame["season_end_year"].eq(year)].copy()
            if len(train) == 0 or len(validation) == 0 or len(test) == 0 or train["target_1x2"].nunique() < 3:
                continue
            _, p = fit_model(train, validation, test, key.model, features, 9000 + year)
            parts.append(prediction_frame(test, key.model, key.feature_group, year, "test", p))
        row = {"robustness": name, "rows": 0}
        if parts:
            row = summarize_predictions(pd.concat(parts, ignore_index=True).assign(robustness=name), ["robustness"]).iloc[0].to_dict()
        row["model"] = key.model
        row["feature_group"] = key.feature_group
        rows.append(row)
    return pd.DataFrame(rows)


def select_rule(frame: pd.DataFrame, side: str, edge_threshold: float, min_odds: float) -> pd.DataFrame:
    odds_col = f"avg_{side}_odds"
    edge_col = f"{side}_edge"
    profit_col = f"{side}_profit"
    selected = frame[frame[edge_col].ge(edge_threshold) & frame[odds_col].ge(min_odds)].copy()
    selected["profit"] = selected[profit_col]
    selected["side"] = side
    selected["rule_name"] = f"{side}_edge_{edge_threshold:g}_odds_{min_odds:g}"
    return selected


def bet_summary(selected: pd.DataFrame, label: str, rule_name: str) -> dict[str, object]:
    bets = int(len(selected))
    profit = float(selected["profit"].sum()) if bets else 0.0
    roi = profit / bets if bets else 0.0
    std = float(selected["profit"].std(ddof=1)) if bets > 1 else 0.0
    z = profit / (std * math.sqrt(bets)) if bets > 1 and std > 0 else 0.0
    return {"label": label, "rule_name": rule_name, "bets": bets, "profit": profit, "roi": roi, "z_score": z, "leagues": int(selected["league"].nunique()) if bets else 0, "years": int(selected["test_year"].nunique()) if bets else 0}


def fixed_value_rules(test: pd.DataFrame, model: str, feature_group: str) -> pd.DataFrame:
    rows = []
    for side in ["home", "draw", "away"]:
        for edge, odds in RULE_GRID:
            selected = select_rule(test, side, edge, odds)
            rule_name = selected["rule_name"].iloc[0] if len(selected) else f"{side}_edge_{edge:g}_odds_{odds:g}"
            row = bet_summary(selected, "fixed_rule", rule_name)
            row.update({"model": model, "feature_group": feature_group, "side": side, "edge_threshold": edge, "min_odds": odds})
            rows.append(row)
    return pd.DataFrame(rows)


def validation_rule_passes(selected: pd.DataFrame) -> bool:
    if len(selected) < 150:
        return False
    stats = bet_summary(selected, "validation", selected["rule_name"].iloc[0])
    if stats["roi"] <= 0 or stats["z_score"] <= 0.75 or stats["profit"] <= 0:
        return False
    if selected["league"].nunique() < 4:
        return False
    if selected["league"].value_counts(normalize=True).max() > 0.35:
        return False
    return True


def nested_selection(test: pd.DataFrame, validation: pd.DataFrame, model: str, feature_group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bets = []
    for year in TEST_YEARS:
        prior = validation[validation["test_year"].lt(year)].copy()
        current = test[test["test_year"].eq(year)].copy()
        candidates = []
        if len(prior):
            for side in ["home", "draw", "away"]:
                for edge, odds in RULE_GRID:
                    selected = select_rule(prior, side, edge, odds)
                    if validation_rule_passes(selected):
                        stats = bet_summary(selected, "validation", selected["rule_name"].iloc[0])
                        stats.update({"side": side, "edge_threshold": edge, "min_odds": odds})
                        candidates.append(stats)
        if not candidates:
            rows.append({"model": model, "feature_group": feature_group, "test_year": year, "selected_rule": "", "selection_status": "no_rule_passed", "test_bets": 0, "test_profit": 0.0, "test_roi": 0.0, "test_z": 0.0})
            continue
        chosen = pd.DataFrame(candidates).sort_values(["z_score", "profit", "bets"], ascending=[False, False, False]).iloc[0]
        selected_test = select_rule(current, str(chosen["side"]), float(chosen["edge_threshold"]), float(chosen["min_odds"]))
        stats = bet_summary(selected_test, "test", str(chosen["rule_name"]))
        rows.append({"model": model, "feature_group": feature_group, "test_year": year, "selected_rule": chosen["rule_name"], "selection_status": "selected_prior_only", "test_bets": stats["bets"], "test_profit": stats["profit"], "test_roi": stats["roi"], "test_z": stats["z_score"]})
        bets.append(selected_test)
    return pd.DataFrame(rows), pd.concat(bets, ignore_index=True, sort=False) if bets else pd.DataFrame()


def value_controls(test: pd.DataFrame, selected: pd.DataFrame, model: str, feature_group: str, rule_name: str) -> pd.DataFrame:
    rows = [{"model": model, "feature_group": feature_group, "control": "selected_rule", **bet_summary(selected, "control", rule_name)}]
    if len(selected):
        rng = np.random.default_rng(321)
        sample = test.sample(n=min(len(selected), len(test)), replace=False, random_state=321).copy()
        choices = rng.integers(0, 3, len(sample))
        sample["profit"] = np.select([choices == 0, choices == 1, choices == 2], [sample["home_profit"], sample["draw_profit"], sample["away_profit"]])
        rows.append({"model": model, "feature_group": feature_group, "control": "random_same_bet_count", **bet_summary(sample, "control", "random_same_bet_count")})
    return pd.DataFrame(rows)


def value_robustness(selected: pd.DataFrame, label: str, model: str, feature_group: str) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame([{"model": model, "feature_group": feature_group, "portfolio": label, "robustness": "empty", "bets": 0, "profit": 0.0, "roi": 0.0, "z_score": 0.0}])
    best_season = selected.groupby("test_year")["profit"].sum().sort_values(ascending=False).index[0]
    best_league = selected.groupby("league")["profit"].sum().sort_values(ascending=False).index[0]
    rows = []
    for name, frame in [("all", selected), ("exclude_best_profit_season", selected[selected["test_year"].ne(best_season)]), ("exclude_best_profit_league", selected[~selected["league"].eq(best_league)])]:
        rows.append({"model": model, "feature_group": feature_group, "portfolio": label, "robustness": name, **bet_summary(frame, "robustness", label)})
    return pd.DataFrame(rows)


def classify_value(nested_bets: pd.DataFrame, controls: pd.DataFrame, robustness: pd.DataFrame) -> str:
    if nested_bets.empty:
        return "predictive_only_no_value"
    stats = bet_summary(nested_bets, "nested", "nested")
    season_profit = nested_bets.groupby("test_year")["profit"].sum()
    league_profit = nested_bets.groupby("league")["profit"].sum()
    positive_profit = max(stats["profit"], 0.0)
    no_majority_season = positive_profit > 0 and season_profit.max() <= 0.5 * positive_profit
    no_majority_league = positive_profit > 0 and league_profit.max() <= 0.5 * positive_profit
    robust_row = robustness[robustness["robustness"].eq("exclude_best_profit_season")]
    robust_positive = len(robust_row) > 0 and float(robust_row["profit"].iloc[0]) > 0
    control_best = controls[~controls["control"].eq("selected_rule")]["profit"].max() if len(controls) else 0.0
    controls_fail = pd.isna(control_best) or float(control_best) < stats["profit"]
    if stats["profit"] > 0 and stats["roi"] > 0.02 and stats["z_score"] >= 1.0 and stats["years"] >= 4 and no_majority_season and no_majority_league and robust_positive and controls_fail:
        return "forward_paper_candidate"
    return "research_only" if stats["bets"] > 0 else "predictive_only_no_value"


def advancement_candidates(summary: pd.DataFrame, edge_buckets: pd.DataFrame, robustness: pd.DataFrame, negative: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        if row["model"] == "raw_market_baseline":
            continue
        high = edge_buckets[(edge_buckets["model"].eq(row["model"])) & (edge_buckets["feature_group"].eq(row["feature_group"])) & (edge_buckets["edge_bucket"].eq(">=0.05"))]
        high_sane = len(high) == 0 or abs(float(high["calibration_error"].mean())) <= 0.08
        neg = negative[(negative["model"].eq(row["model"])) & (negative["feature_group"].eq(row["feature_group"]))]
        negative_fail = len(neg) > 0 and not (neg["mean_delta_multiclass_log_loss_vs_raw_market"] < 0).any()
        rob = robustness[(robustness["model"].eq(row["model"])) & (robustness["feature_group"].eq(row["feature_group"]))]
        best_exclusions_positive = True
        for required in ["exclude_best_performing_season", "exclude_best_performing_league"]:
            r = rob[rob["robustness"].eq(required)]
            if len(r) == 0 or float(r["mean_delta_multiclass_log_loss_vs_raw_market"].iloc[0]) >= 0 or float(r["mean_delta_brier_vs_raw_market"].iloc[0]) >= 0:
                best_exclusions_positive = False
        passes = (
            row["mean_delta_multiclass_log_loss_vs_raw_market"] < 0
            and row["mean_delta_brier_vs_raw_market"] < 0
            and row["mean_delta_ece_vs_raw_market"] <= 0.0025
            and int(row["improved_years"]) >= 6
            and negative_fail
            and high_sane
            and best_exclusions_positive
        )
        rows.append({**row.to_dict(), "advancement_gate_passed": bool(passes), "negative_controls_failed": bool(negative_fail), "high_edge_buckets_sane": bool(high_sane), "robustness_best_exclusions_positive": bool(best_exclusions_positive)})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def write_reports(summary: pd.DataFrame, candidates: pd.DataFrame, value_fixed: pd.DataFrame, value_nested: pd.DataFrame, final_classification: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best = summary[summary["summary_scope"].eq("overall")].sort_values("mean_delta_multiclass_log_loss_vs_raw_market").head(30)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# 1X2 Predictive Audit",
                "",
                f"Final classification: `{final_classification}`",
                "",
                "Scope: historical-training modern-test 1X2 audit using local processed football-data only. AvgH/AvgD/AvgA were preferred, with BbAvH/BbAvD/BbAvA used as the legacy average fallback. No live betting, Transfermarkt, player features, lineups, team-name direct model features, closing odds as features/selection inputs, scraping, external APIs, threshold optimization after test results, or confirmed edge claims were used. Simple Elo/rating was skipped because no clean existing implementation was present in the processed match files.",
                "",
                "## Best Predictive Rows",
                "",
                markdown_table(best, ["model", "feature_group", "rows", "test_years", "mean_delta_multiclass_log_loss_vs_raw_market", "mean_delta_brier_vs_raw_market", "mean_delta_ece_vs_raw_market", "improved_years"], 30),
                "",
                "## Advancement Gate",
                "",
                markdown_table(candidates, ["model", "feature_group", "advancement_gate_passed", "mean_delta_multiclass_log_loss_vs_raw_market", "mean_delta_brier_vs_raw_market", "mean_delta_ece_vs_raw_market", "improved_years", "negative_controls_failed", "high_edge_buckets_sane", "robustness_best_exclusions_positive"], 80),
                "",
                "Locked value review is only eligible for rows with `advancement_gate_passed=True`.",
                "",
                "No confirmed edge is claimed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    VALUE_REPORT_PATH.write_text(
        "\n".join(
            [
                "# 1X2 Locked Value Review",
                "",
                f"Final classification: `{final_classification}`",
                "",
                "Value review was run only for models that passed the predictive advancement gate. Fixed candidate rules were predeclared in the audit script.",
                "",
                "## Fixed Rules",
                "",
                markdown_table(value_fixed.sort_values(["profit", "z_score"], ascending=[False, False]) if len(value_fixed) else value_fixed, ["model", "feature_group", "rule_name", "bets", "profit", "roi", "z_score", "leagues", "years"], 60),
                "",
                "## Nested Temporal Selection",
                "",
                markdown_table(value_nested, ["model", "feature_group", "test_year", "selected_rule", "selection_status", "test_bets", "test_profit", "test_roi", "test_z"], 80),
                "",
                "No confirmed edge is claimed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    data = load_data()
    test_pred, val_pred = run_predictions(data)
    overall = summarize_predictions(test_pred, ["model", "feature_group"]).assign(summary_scope="overall")
    summary = pd.concat(
        [
            overall,
            summarize_predictions(test_pred, ["model", "feature_group", "test_year"]).assign(summary_scope="per_year"),
            summarize_predictions(test_pred, ["model", "feature_group", "league"]).assign(summary_scope="per_league"),
            binary_decompositions(test_pred).assign(summary_scope="binary_decomposition"),
        ],
        ignore_index=True,
        sort=False,
    )
    class_cal = class_calibration(test_pred)
    edge_buckets = edge_bucket_calibration(test_pred)
    prelim = overall[
        (~overall["model"].eq("raw_market_baseline"))
        & overall["mean_delta_multiclass_log_loss_vs_raw_market"].lt(0)
        & overall["mean_delta_brier_vs_raw_market"].lt(0)
        & overall["mean_delta_ece_vs_raw_market"].le(0.0025)
        & overall["improved_years"].ge(6)
    ].copy()
    if prelim.empty:
        prelim = overall.sort_values("mean_delta_multiclass_log_loss_vs_raw_market").head(1).copy()
    negative_parts = []
    robustness_parts = []
    for _, candidate in prelim.iterrows():
        key = EvalKey(str(candidate["model"]), str(candidate["feature_group"]))
        candidate_pred = test_pred[test_pred["model"].eq(key.model) & test_pred["feature_group"].eq(key.feature_group)]
        by_season = summarize_predictions(candidate_pred, ["test_year"])
        best_season = int(by_season.sort_values("mean_delta_multiclass_log_loss_vs_raw_market").iloc[0]["test_year"])
        by_league = summarize_predictions(candidate_pred, ["league"])
        best_league = str(by_league.sort_values("mean_delta_multiclass_log_loss_vs_raw_market").iloc[0]["league"])
        negative_parts.append(run_negative_controls(data, key))
        robustness_parts.append(run_robustness(data, key, best_season, best_league))
    negative = pd.concat(negative_parts, ignore_index=True, sort=False) if negative_parts else pd.DataFrame()
    robustness = pd.concat(robustness_parts, ignore_index=True, sort=False) if robustness_parts else pd.DataFrame()
    candidates = advancement_candidates(overall, edge_buckets, robustness, negative)
    passed = candidates[candidates["advancement_gate_passed"]].copy()
    fixed_parts = []
    nested_parts = []
    control_parts = []
    value_robust_parts = []
    value_classes = []
    for _, row in passed.iterrows():
        model = str(row["model"])
        fg = str(row["feature_group"])
        test = test_pred[test_pred["model"].eq(model) & test_pred["feature_group"].eq(fg)].copy()
        val = val_pred[val_pred["model"].eq(model) & val_pred["feature_group"].eq(fg)].copy()
        fixed = fixed_value_rules(test, model, fg)
        nested, nested_bets = nested_selection(test, val, model, fg)
        fixed_parts.append(fixed)
        nested_parts.append(nested)
        if len(fixed):
            best_rule = fixed.sort_values(["profit", "z_score"], ascending=[False, False]).iloc[0]
            best_bets = select_rule(test, str(best_rule["side"]), float(best_rule["edge_threshold"]), float(best_rule["min_odds"]))
            control_parts.append(value_controls(test, best_bets, model, fg, str(best_rule["rule_name"])))
            value_robust_parts.append(value_robustness(best_bets, "best_fixed_rule", model, fg))
        if len(nested_bets):
            control_parts.append(value_controls(test, nested_bets, model, fg, "nested_portfolio"))
            vr = value_robustness(nested_bets, "nested_portfolio", model, fg)
            value_robust_parts.append(vr)
            value_classes.append(classify_value(nested_bets, pd.concat(control_parts, ignore_index=True, sort=False), vr))
    value_fixed = pd.concat(fixed_parts, ignore_index=True, sort=False) if fixed_parts else pd.DataFrame()
    value_nested = pd.concat(nested_parts, ignore_index=True, sort=False) if nested_parts else pd.DataFrame()
    value_controls_df = pd.concat(control_parts, ignore_index=True, sort=False) if control_parts else pd.DataFrame()
    value_robustness_df = pd.concat(value_robust_parts, ignore_index=True, sort=False) if value_robust_parts else pd.DataFrame()
    if "forward_paper_candidate" in value_classes:
        final_classification = "forward_paper_candidate"
    elif len(passed):
        final_classification = "research_only" if "research_only" in value_classes else "predictive_only_no_value"
    else:
        final_classification = "predictive_only_no_value" if (overall["mean_delta_multiclass_log_loss_vs_raw_market"] < 0).any() else "reject"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    class_cal.to_csv(CLASS_CAL_PATH, index=False)
    edge_buckets.to_csv(EDGE_BUCKET_PATH, index=False)
    negative.to_csv(NEGATIVE_PATH, index=False)
    robustness.to_csv(ROBUSTNESS_PATH, index=False)
    value_fixed.to_csv(VALUE_FIXED_PATH, index=False)
    value_nested.to_csv(VALUE_NESTED_PATH, index=False)
    value_controls_df.to_csv(VALUE_CONTROLS_PATH, index=False)
    value_robustness_df.to_csv(VALUE_ROBUSTNESS_PATH, index=False)
    write_reports(summary, candidates, value_fixed, value_nested, final_classification)
    print(
        {
            "data_rows": len(data),
            "test_prediction_rows": len(test_pred),
            "overall_rows": int(len(overall)),
            "gate_passed": int(len(passed)),
            "classification": final_classification,
        }
    )


if __name__ == "__main__":
    main()
