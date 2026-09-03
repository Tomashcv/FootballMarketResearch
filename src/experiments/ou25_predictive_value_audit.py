from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from pandas.errors import PerformanceWarning
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
TARGET = "over_2_5"

REPORT_PATH = Path("outputs/reports/ou25_predictive_audit.md")
SUMMARY_PATH = Path("outputs/reports/ou25_predictive_summary.csv")
BUCKET_PATH = Path("outputs/reports/ou25_edge_bucket_calibration.csv")
NEGATIVE_PATH = Path("outputs/reports/ou25_negative_controls.csv")
ROBUSTNESS_PATH = Path("outputs/reports/ou25_robustness.csv")
VALUE_REPORT_PATH = Path("outputs/reports/ou25_value_review.md")
VALUE_FIXED_PATH = Path("outputs/reports/ou25_value_fixed_rules.csv")
VALUE_NESTED_PATH = Path("outputs/reports/ou25_value_nested_selection.csv")
VALUE_CONTROLS_PATH = Path("outputs/reports/ou25_value_controls.csv")
VALUE_ROBUSTNESS_PATH = Path("outputs/reports/ou25_value_robustness.csv")

MODELS = [
    "raw_market_baseline",
    "market_baseline_calibration_only",
    "logistic_l2",
    "logistic_elasticnet",
    "xgboost_shallow",
    "xgboost_depth3_regularized",
    "xgboost_market_residual",
]

EDGE_THRESHOLDS = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
RULE_GRID = [
    (0.01, 1.80),
    (0.015, 1.80),
    (0.02, 1.80),
    (0.03, 1.80),
    (0.04, 1.80),
    (0.05, 1.80),
    (0.02, 1.85),
    (0.03, 1.85),
    (0.04, 1.85),
    (0.05, 1.85),
    (0.02, 1.90),
    (0.03, 1.90),
    (0.04, 1.90),
    (0.05, 1.90),
]


@dataclass(frozen=True)
class EvalKey:
    model: str
    feature_group: str


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
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["league"] = league
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No processed league match files found.")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    for column in ["season_end_year", "FTHG", "FTAG"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["avg_over_2_5_odds"] = coalesce(data, ["Avg>2.5", "BbAv>2.5"])
    data["avg_under_2_5_odds"] = coalesce(data, ["Avg<2.5", "BbAv<2.5"])
    data["max_over_2_5_odds"] = coalesce(data, ["Max>2.5", "BbMx>2.5"])
    data["max_under_2_5_odds"] = coalesce(data, ["Max<2.5", "BbMx<2.5"])
    raw_over = 1.0 / data["avg_over_2_5_odds"]
    raw_under = 1.0 / data["avg_under_2_5_odds"]
    data["overround"] = raw_over + raw_under
    data["no_vig_over_probability"] = raw_over / data["overround"]
    data["no_vig_under_probability"] = raw_under / data["overround"]
    data["raw_market_over_probability"] = data["no_vig_over_probability"]
    data["raw_market_under_probability"] = data["no_vig_under_probability"]
    data["odds_spread"] = data["avg_over_2_5_odds"] - data["avg_under_2_5_odds"]
    data["total_goals"] = data["FTHG"] + data["FTAG"]
    data[TARGET] = np.where(data["total_goals"].gt(2.5), 1.0, np.where(data["total_goals"].notna(), 0.0, np.nan))
    required = ["Date", "league", "season_end_year", "HomeTeam", "AwayTeam", "FTHG", "FTAG", TARGET, "avg_over_2_5_odds", "avg_under_2_5_odds", "no_vig_over_probability"]
    data = data.dropna(subset=required).copy()
    data = data[data["avg_over_2_5_odds"].gt(1.0) & data["avg_under_2_5_odds"].gt(1.0)].copy()
    data["season_end_year"] = data["season_end_year"].astype(int)
    return data.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def add_structure_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
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
    out["odds_bucket"] = pd.cut(out["avg_over_2_5_odds"], bins=[1.0, 1.6, 1.8, 2.0, 2.25, 2.6, 10.0], labels=False).astype(float)
    out["market_total_balance_bucket"] = pd.cut(out["no_vig_over_probability"], bins=[0.0, 0.42, 0.47, 0.50, 0.53, 0.58, 1.0], labels=False).astype(float)
    for league in LEAGUES:
        out[f"league_code_{league}"] = out["league"].eq(league).astype(float)
    for layer in [1, 2, 3]:
        out[f"league_layer_{layer}"] = out["league_layer"].eq(float(layer)).astype(float)
    for era in [0, 1, 2, 3]:
        out[f"season_era_{era}"] = out["season_era"].eq(float(era)).astype(float)
    for bucket in range(6):
        out[f"odds_bucket_{bucket}"] = out["odds_bucket"].eq(float(bucket)).astype(float)
        out[f"market_balance_bucket_{bucket}"] = out["market_total_balance_bucket"].eq(float(bucket)).astype(float)
    return out


def add_temporal_goal_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).copy()
    out["match_order"] = np.arange(len(out))
    out["goals_per_match"] = out["total_goals"]
    for window in [20, 50, 100]:
        out[f"league_rolling_goals_per_match_{window}"] = (
            out.groupby("league")["goals_per_match"].transform(lambda s: s.shift(1).rolling(window, min_periods=5).mean())
        )
    long_rows = []
    for idx, row in out.iterrows():
        long_rows.append({"row_id": idx, "league": row["league"], "Date": row["Date"], "team": row["HomeTeam"], "gf": row["FTHG"], "ga": row["FTAG"], "is_home_row": True, "is_away_row": False, "over": row[TARGET]})
        long_rows.append({"row_id": idx, "league": row["league"], "Date": row["Date"], "team": row["AwayTeam"], "gf": row["FTAG"], "ga": row["FTHG"], "is_home_row": False, "is_away_row": True, "over": row[TARGET]})
    long = pd.DataFrame(long_rows).sort_values(["league", "team", "Date", "row_id"]).reset_index(drop=True)
    grouped = long.groupby(["league", "team"], sort=False)
    long["previous_match_date"] = grouped["Date"].shift(1)
    long["rest_days"] = (long["Date"] - long["previous_match_date"]).dt.days
    for window in [5, 10, 20]:
        long[f"rolling_gf_{window}"] = grouped["gf"].transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
        long[f"rolling_ga_{window}"] = grouped["ga"].transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
        long[f"rolling_over_rate_{window}"] = grouped["over"].transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
    home = long[long["is_home_row"]].set_index("row_id")
    away = long[long["is_away_row"]].set_index("row_id")
    for window in [5, 10, 20]:
        out[f"home_rolling_gf_{window}"] = home[f"rolling_gf_{window}"]
        out[f"home_rolling_ga_{window}"] = home[f"rolling_ga_{window}"]
        out[f"away_rolling_gf_{window}"] = away[f"rolling_gf_{window}"]
        out[f"away_rolling_ga_{window}"] = away[f"rolling_ga_{window}"]
        out[f"home_rolling_over_2_5_rate_{window}"] = home[f"rolling_over_rate_{window}"]
        out[f"away_rolling_over_2_5_rate_{window}"] = away[f"rolling_over_rate_{window}"]
        out[f"combined_expected_goal_proxy_{window}"] = (
            out[f"home_rolling_gf_{window}"] + out[f"away_rolling_gf_{window}"] + out[f"home_rolling_ga_{window}"] + out[f"away_rolling_ga_{window}"]
        ) / 2.0
    out["home_rest_days"] = home["rest_days"]
    out["away_rest_days"] = away["rest_days"]
    out["rest_days_diff"] = out["home_rest_days"] - out["away_rest_days"]
    out["row_id"] = np.arange(len(out))
    return out.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def load_data() -> pd.DataFrame:
    return add_temporal_goal_features(add_structure_features(load_base_data()))


def feature_groups(data: pd.DataFrame) -> dict[str, list[str]]:
    market = [
        "avg_over_2_5_odds",
        "avg_under_2_5_odds",
        "no_vig_over_probability",
        "no_vig_under_probability",
        "overround",
        "odds_spread",
    ]
    structure = market + [f"league_code_{league}" for league in LEAGUES]
    structure += [f"league_layer_{i}" for i in [1, 2, 3]]
    structure += [f"season_era_{i}" for i in [0, 1, 2, 3]]
    structure += [f"odds_bucket_{i}" for i in range(6)]
    structure += [f"market_balance_bucket_{i}" for i in range(6)]
    temporal = structure.copy()
    for window in [20, 50, 100]:
        temporal.append(f"league_rolling_goals_per_match_{window}")
    for window in [5, 10, 20]:
        temporal += [
            f"home_rolling_gf_{window}",
            f"home_rolling_ga_{window}",
            f"away_rolling_gf_{window}",
            f"away_rolling_ga_{window}",
            f"home_rolling_over_2_5_rate_{window}",
            f"away_rolling_over_2_5_rate_{window}",
            f"combined_expected_goal_proxy_{window}",
        ]
    temporal += ["home_rest_days", "away_rest_days", "rest_days_diff"]
    groups = {
        "ou_market_only": market,
        "ou_market_structure_safe": structure,
        "ou_market_plus_temporal_goals": temporal,
    }
    return {name: [column for column in columns if column in data.columns] for name, columns in groups.items()}


def metric_values(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    y = frame[TARGET].astype(int).to_numpy()
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(expected_calibration_error(y, p)),
    }


def fit_xgb(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str], params: dict, rounds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if xgb is None:
        raise ImportError("xgboost is required for xgboost models.")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_val = imputer.transform(validation[features])
    x_test = imputer.transform(test[features])
    params = {**params, "objective": "binary:logistic", "eval_metric": "logloss", "seed": seed, "verbosity": 0, "tree_method": "hist", "nthread": 4}
    dtrain = xgb.DMatrix(x_train, label=train[TARGET].astype(int).to_numpy(), feature_names=features)
    dval = xgb.DMatrix(x_val, label=validation[TARGET].astype(int).to_numpy(), feature_names=features)
    model = xgb.train(params, dtrain, num_boost_round=rounds, evals=[(dval, "validation")], early_stopping_rounds=12, verbose_eval=False)
    return model.predict(dval), model.predict(xgb.DMatrix(x_test, feature_names=features))


def fit_model(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, model: str, features: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    if model == "raw_market_baseline":
        return validation["no_vig_over_probability"].to_numpy(), test["no_vig_over_probability"].to_numpy()
    if model == "market_baseline_calibration_only":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=seed, C=1.0))])
        pipe.fit(validation[["no_vig_over_probability"]], validation[TARGET].astype(int))
        cls = list(pipe.named_steps["model"].classes_).index(1)
        return pipe.predict_proba(validation[["no_vig_over_probability"]])[:, cls], pipe.predict_proba(test[["no_vig_over_probability"]])[:, cls]
    if model in {"logistic_l2", "logistic_elasticnet"}:
        penalty = "l2" if model == "logistic_l2" else "elasticnet"
        solver = "lbfgs" if model == "logistic_l2" else "saga"
        kwargs = {"l1_ratio": 0.2, "tol": 1e-3, "n_jobs": 4} if model == "logistic_elasticnet" else {"tol": 1e-4}
        max_iter = 200 if model == "logistic_elasticnet" else 1000
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=max_iter, random_state=seed, penalty=penalty, solver=solver, C=0.5, **kwargs))])
        pipe.fit(train[features], train[TARGET].astype(int))
        cls = list(pipe.named_steps["model"].classes_).index(1)
        return pipe.predict_proba(validation[features])[:, cls], pipe.predict_proba(test[features])[:, cls]
    if model == "xgboost_shallow":
        return fit_xgb(train, validation, test, features, {"max_depth": 2, "eta": 0.035, "lambda": 10.0, "alpha": 2.0, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 20.0}, 100, seed)
    if model == "xgboost_depth3_regularized":
        return fit_xgb(train, validation, test, features, {"max_depth": 3, "eta": 0.03, "lambda": 15.0, "alpha": 4.0, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 25.0}, 110, seed)
    if model == "xgboost_market_residual":
        if xgb is None:
            raise ImportError("xgboost is required for residual model.")
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train[features])
        x_val = imputer.transform(validation[features])
        x_test = imputer.transform(test[features])
        train_res = train[TARGET].astype(float).to_numpy() - train["no_vig_over_probability"].to_numpy()
        val_res = validation[TARGET].astype(float).to_numpy() - validation["no_vig_over_probability"].to_numpy()
        params = {"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 2, "eta": 0.025, "lambda": 14.0, "alpha": 4.0, "subsample": 0.9, "colsample_bytree": 0.9, "seed": seed, "verbosity": 0, "tree_method": "hist", "nthread": 4}
        dtrain = xgb.DMatrix(x_train, label=train_res, feature_names=features)
        dval = xgb.DMatrix(x_val, label=val_res, feature_names=features)
        booster = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, "validation")], early_stopping_rounds=12, verbose_eval=False)
        return (
            np.clip(validation["no_vig_over_probability"].to_numpy() + booster.predict(dval), 1e-6, 1 - 1e-6),
            np.clip(test["no_vig_over_probability"].to_numpy() + booster.predict(xgb.DMatrix(x_test, feature_names=features)), 1e-6, 1 - 1e-6),
        )
    if model == "small_mlp_regularized":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(24,), alpha=0.03, learning_rate_init=0.001, max_iter=100, early_stopping=True, validation_fraction=0.15, random_state=seed))])
        pipe.fit(train[features], train[TARGET].astype(int))
        cls = list(pipe.named_steps["model"].classes_).index(1)
        return pipe.predict_proba(validation[features])[:, cls], pipe.predict_proba(test[features])[:, cls]
    raise ValueError(model)


def prediction_frame(frame: pd.DataFrame, model: str, feature_group: str, year: int, role: str, p: np.ndarray) -> pd.DataFrame:
    columns = [
        "row_id",
        "league",
        "season_end_year",
        "Date",
        TARGET,
        "avg_over_2_5_odds",
        "avg_under_2_5_odds",
        "no_vig_over_probability",
        "no_vig_under_probability",
        "raw_market_over_probability",
        "raw_market_under_probability",
    ]
    out = frame[columns].copy()
    out["model"] = model
    out["feature_group"] = feature_group
    out["test_year"] = year
    out["fold_role"] = role
    out["model_over_probability"] = np.clip(p, 1e-6, 1 - 1e-6)
    out["model_under_probability"] = 1.0 - out["model_over_probability"]
    out["over_edge"] = out["model_over_probability"] - out["no_vig_over_probability"]
    out["under_edge"] = out["model_under_probability"] - out["no_vig_under_probability"]
    out["over_profit"] = np.where(out[TARGET].eq(1), out["avg_over_2_5_odds"] - 1.0, -1.0)
    out["under_profit"] = np.where(out[TARGET].eq(0), out["avg_under_2_5_odds"] - 1.0, -1.0)
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
                if len(train) == 0 or len(validation) == 0 or len(test) == 0 or train[TARGET].nunique() < 2:
                    continue
                val_p, test_p = fit_model(train, validation, test, model, features, 1000 + year)
                val_parts.append(prediction_frame(validation, model, fg, year, "validation", val_p))
                test_parts.append(prediction_frame(test, model, fg, year, "test", test_p))
    return pd.concat(test_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)


def summarize_predictions(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pred.groupby(group_cols, dropna=False):
        model_metrics = metric_values(g, g["model_over_probability"].to_numpy())
        market_metrics = metric_values(g, g["no_vig_over_probability"].to_numpy())
        improved = []
        for _, gy in g.groupby("test_year"):
            improved.append(metric_values(gy, gy["model_over_probability"].to_numpy())["log_loss"] < metric_values(gy, gy["no_vig_over_probability"].to_numpy())["log_loss"])
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "rows": int(len(g)),
                "test_years": ";".join(map(str, sorted(g["test_year"].unique()))),
                "mean_delta_log_loss_vs_raw_market": model_metrics["log_loss"] - market_metrics["log_loss"],
                "mean_delta_brier_vs_raw_market": model_metrics["brier"] - market_metrics["brier"],
                "mean_delta_ece_vs_raw_market": model_metrics["ece"] - market_metrics["ece"],
                "improved_years": int(sum(improved)),
                "model_log_loss": model_metrics["log_loss"],
                "market_log_loss": market_metrics["log_loss"],
                "model_brier": model_metrics["brier"],
                "market_brier": market_metrics["brier"],
                "model_ece": model_metrics["ece"],
                "market_ece": market_metrics["ece"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def edge_bucket_calibration(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    edge_bins = [-1e-9, 0.01, 0.02, 0.03, 0.04, 0.05, 10.0]
    edge_labels = ["0.00-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05", ">=0.05"]
    for model, fg in pred[["model", "feature_group"]].drop_duplicates().itertuples(index=False):
        subset = pred[pred["model"].eq(model) & pred["feature_group"].eq(fg)].copy()
        subset["over_edge_bucket"] = pd.cut(subset["over_edge"], bins=edge_bins, labels=edge_labels).astype(str)
        subset["under_edge_bucket"] = pd.cut(subset["under_edge"], bins=edge_bins, labels=edge_labels).astype(str)
        subset["predicted_probability_bucket"] = pd.cut(subset["model_over_probability"], bins=np.linspace(0, 1, 11), include_lowest=True).astype(str)
        for side, bucket_col in [("over", "over_edge_bucket"), ("under", "under_edge_bucket")]:
            for bucket, g in subset.groupby(bucket_col):
                if bucket == "nan" or len(g) < 20:
                    continue
                prob_col = "model_over_probability" if side == "over" else "model_under_probability"
                target_rate = float(g[TARGET].mean()) if side == "over" else float(1.0 - g[TARGET].mean())
                rows.append(
                    {
                        "model": model,
                        "feature_group": fg,
                        "bucket_type": f"{side}_edge",
                        "bucket": bucket,
                        "rows": int(len(g)),
                        "average_model_probability": float(g[prob_col].mean()),
                        "average_market_probability": float((g["no_vig_over_probability"] if side == "over" else g["no_vig_under_probability"]).mean()),
                        "realised_over_rate": float(g[TARGET].mean()),
                        "realised_side_hit_rate": target_rate,
                        "calibration_error": target_rate - float(g[prob_col].mean()),
                    }
                )
        for bucket, g in subset.groupby("predicted_probability_bucket"):
            if bucket == "nan" or len(g) < 20:
                continue
            rows.append(
                {
                    "model": model,
                    "feature_group": fg,
                    "bucket_type": "predicted_probability",
                    "bucket": bucket,
                    "rows": int(len(g)),
                    "average_model_probability": float(g["model_over_probability"].mean()),
                    "average_market_probability": float(g["no_vig_over_probability"].mean()),
                    "realised_over_rate": float(g[TARGET].mean()),
                    "realised_side_hit_rate": float(g[TARGET].mean()),
                    "calibration_error": float(g[TARGET].mean() - g["model_over_probability"].mean()),
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
                y = train[TARGET].to_numpy(copy=True)
                rng.shuffle(y)
                train[TARGET] = y
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
                train[TARGET] = 1 - train[TARGET].astype(int)
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
            if len(train) == 0 or len(validation) == 0 or len(test) == 0 or train[TARGET].nunique() < 2:
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
    if side == "over":
        selected = frame[frame["over_edge"].ge(edge_threshold) & frame["avg_over_2_5_odds"].ge(min_odds)].copy()
        selected["profit"] = selected["over_profit"]
        selected["side"] = "over"
    else:
        selected = frame[frame["under_edge"].ge(edge_threshold) & frame["avg_under_2_5_odds"].ge(min_odds)].copy()
        selected["profit"] = selected["under_profit"]
        selected["side"] = "under"
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
    for side in ["over", "under"]:
        for edge, odds in RULE_GRID:
            selected = select_rule(test, side, edge, odds)
            row = bet_summary(selected, "fixed_rule", selected["rule_name"].iloc[0] if len(selected) else f"{side}_edge_{edge:g}_odds_{odds:g}")
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
            for side in ["over", "under"]:
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
    rows = []
    rows.append({"model": model, "feature_group": feature_group, "control": "selected_rule", **bet_summary(selected, "control", rule_name)})
    if len(selected):
        rng = np.random.default_rng(321)
        sample = test.sample(n=min(len(selected), len(test)), replace=False, random_state=321).copy()
        sample["profit"] = np.where(rng.random(len(sample)) < 0.5, sample["over_profit"], sample["under_profit"])
        rows.append({"model": model, "feature_group": feature_group, "control": "random_same_bet_count", **bet_summary(sample, "control", "random_same_bet_count")})
        inverse = selected.copy()
        inverse["profit"] = np.where(inverse["side"].eq("over"), inverse["under_profit"], inverse["over_profit"])
        rows.append({"model": model, "feature_group": feature_group, "control": "opposite_side_same_matches", **bet_summary(inverse, "control", "opposite_side_same_matches")})
    return pd.DataFrame(rows)


def value_robustness(selected: pd.DataFrame, label: str, model: str, feature_group: str) -> pd.DataFrame:
    rows = []
    if selected.empty:
        return pd.DataFrame([{"model": model, "feature_group": feature_group, "portfolio": label, "robustness": "empty", "bets": 0, "profit": 0.0, "roi": 0.0, "z_score": 0.0}])
    best_season = selected.groupby("test_year")["profit"].sum().sort_values(ascending=False).index[0]
    best_league = selected.groupby("league")["profit"].sum().sort_values(ascending=False).index[0]
    for name, frame in [
        ("all", selected),
        ("exclude_best_profit_season", selected[selected["test_year"].ne(best_season)]),
        ("exclude_best_profit_league", selected[~selected["league"].eq(best_league)]),
    ]:
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


def advancement_candidates(summary: pd.DataFrame, buckets: pd.DataFrame, robustness: pd.DataFrame, negative: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        if row["model"] == "raw_market_baseline":
            continue
        key = (row["model"], row["feature_group"])
        high = buckets[(buckets["model"].eq(key[0])) & (buckets["feature_group"].eq(key[1])) & (buckets["bucket_type"].isin(["over_edge", "under_edge"])) & (buckets["bucket"].eq(">=0.05"))]
        high_sane = len(high) == 0 or abs(float(high["calibration_error"].mean())) <= 0.08
        neg = negative[(negative["model"].eq(key[0])) & (negative["feature_group"].eq(key[1]))]
        negative_fail = len(neg) > 0 and not (neg["mean_delta_log_loss_vs_raw_market"] < 0).any()
        rob = robustness[(robustness["model"].eq(key[0])) & (robustness["feature_group"].eq(key[1]))]
        best_exclusions_positive = True
        for required in ["exclude_best_performing_season", "exclude_best_performing_league"]:
            r = rob[rob["robustness"].eq(required)]
            if len(r) == 0 or float(r["mean_delta_log_loss_vs_raw_market"].iloc[0]) >= 0 or float(r["mean_delta_brier_vs_raw_market"].iloc[0]) >= 0:
                best_exclusions_positive = False
        passes = (
            row["mean_delta_log_loss_vs_raw_market"] < 0
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
    best = summary[summary["summary_scope"].eq("overall")].sort_values("mean_delta_log_loss_vs_raw_market").head(30)
    lines = [
        "# Over/Under 2.5 Predictive Audit",
        "",
        f"Final classification: `{final_classification}`",
        "",
        "Scope: historical-training modern-test O/U 2.5 audit using local processed football-data only. No live betting, Transfermarkt, player features, lineups, team-name model features, closing odds as features/selection inputs, scraping, external APIs, threshold optimization after test results, or confirmed edge claims were used.",
        "",
        "## Best Predictive Rows",
        "",
        markdown_table(best, ["model", "feature_group", "rows", "test_years", "mean_delta_log_loss_vs_raw_market", "mean_delta_brier_vs_raw_market", "mean_delta_ece_vs_raw_market", "improved_years"], 30),
        "",
        "## Advancement Gate",
        "",
        markdown_table(candidates, ["model", "feature_group", "advancement_gate_passed", "mean_delta_log_loss_vs_raw_market", "mean_delta_brier_vs_raw_market", "mean_delta_ece_vs_raw_market", "improved_years", "negative_controls_failed", "high_edge_buckets_sane", "robustness_best_exclusions_positive"], 60),
        "",
        "Locked value review is only eligible for rows with `advancement_gate_passed=True`.",
        "",
        "No confirmed edge is claimed.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    value_lines = [
        "# Over/Under 2.5 Locked Value Review",
        "",
        f"Final classification: `{final_classification}`",
        "",
        "Value review was run only for models that passed the predictive advancement gate. Fixed candidate rules were predeclared in the audit script.",
        "",
        "## Fixed Rules",
        "",
        markdown_table(value_fixed.sort_values(["profit", "z_score"], ascending=[False, False]) if len(value_fixed) else value_fixed, ["model", "feature_group", "rule_name", "bets", "profit", "roi", "z_score", "leagues", "years"], 50),
        "",
        "## Nested Temporal Selection",
        "",
        markdown_table(value_nested, ["model", "feature_group", "test_year", "selected_rule", "selection_status", "test_bets", "test_profit", "test_roi", "test_z"], 80),
        "",
        "No confirmed edge is claimed.",
        "",
    ]
    VALUE_REPORT_PATH.write_text("\n".join(value_lines), encoding="utf-8")


def main() -> None:
    data = load_data()
    test_pred, val_pred = run_predictions(data)
    overall = summarize_predictions(test_pred, ["model", "feature_group"]).assign(summary_scope="overall")
    summary = pd.concat(
        [
            overall,
            summarize_predictions(test_pred, ["model", "feature_group", "test_year"]).assign(summary_scope="per_year"),
            summarize_predictions(test_pred, ["model", "feature_group", "league"]).assign(summary_scope="per_league"),
        ],
        ignore_index=True,
        sort=False,
    )
    buckets = edge_bucket_calibration(test_pred)
    prelim = overall[
        (~overall["model"].eq("raw_market_baseline"))
        & overall["mean_delta_log_loss_vs_raw_market"].lt(0)
        & overall["mean_delta_brier_vs_raw_market"].lt(0)
        & overall["mean_delta_ece_vs_raw_market"].le(0.0025)
        & overall["improved_years"].ge(6)
    ].copy()
    if prelim.empty:
        prelim = overall.sort_values("mean_delta_log_loss_vs_raw_market").head(1).copy()
    negative_parts = []
    robustness_parts = []
    for _, candidate in prelim.iterrows():
        key = EvalKey(str(candidate["model"]), str(candidate["feature_group"]))
        candidate_pred = test_pred[test_pred["model"].eq(key.model) & test_pred["feature_group"].eq(key.feature_group)]
        by_season = summarize_predictions(candidate_pred, ["test_year"])
        best_season = int(by_season.sort_values("mean_delta_log_loss_vs_raw_market").iloc[0]["test_year"])
        by_league = summarize_predictions(candidate_pred, ["league"])
        best_league = str(by_league.sort_values("mean_delta_log_loss_vs_raw_market").iloc[0]["league"])
        negative_parts.append(run_negative_controls(data, key))
        robustness_parts.append(run_robustness(data, key, best_season, best_league))
    negative = pd.concat(negative_parts, ignore_index=True, sort=False) if negative_parts else pd.DataFrame()
    robustness = pd.concat(robustness_parts, ignore_index=True, sort=False) if robustness_parts else pd.DataFrame()
    candidates = advancement_candidates(overall, buckets, robustness, negative)
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
            best_fixed_bets = select_rule(test, str(best_rule["side"]), float(best_rule["edge_threshold"]), float(best_rule["min_odds"]))
            control_parts.append(value_controls(test, best_fixed_bets, model, fg, str(best_rule["rule_name"])))
            value_robust_parts.append(value_robustness(best_fixed_bets, "best_fixed_rule", model, fg))
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
        final_classification = "predictive_only_no_value" if (overall["mean_delta_log_loss_vs_raw_market"] < 0).any() else "reject"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    buckets.to_csv(BUCKET_PATH, index=False)
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
