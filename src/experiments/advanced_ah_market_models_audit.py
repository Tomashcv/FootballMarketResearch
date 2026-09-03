from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import xgboost as xgb

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments.ah_settlement_engine_audit import settle_side
from src.experiments.post_backfill_locked_ah_value_review import Rule
from src.experiments.post_backfill_locked_ah_value_review import classify as classify_value
from src.experiments.post_backfill_locked_ah_value_review import controls as value_controls
from src.experiments.post_backfill_locked_ah_value_review import fill_selected_columns
from src.experiments.post_backfill_locked_ah_value_review import markdown_table
from src.experiments.post_backfill_locked_ah_value_review import nested_selection as value_nested_selection
from src.experiments.post_backfill_locked_ah_value_review import robustness as value_robustness
from src.experiments.post_backfill_locked_ah_value_review import rule_grid
from src.experiments.post_backfill_locked_ah_value_review import select_rule
from src.experiments.post_backfill_locked_ah_value_review import summarize as value_summarize
from src.experiments.transfermarkt_proxy_predictive_audit import ece_binary


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "E1", "E2", "E3", "SC0"]
LAYER1 = {"E0", "D1", "I1", "SP1", "F1", "P1"}
LAYER2 = {"N1", "B1", "T1", "G1", "E1", "E2", "E3"}
ENGLISH_LOWER = {"E1", "E2", "E3"}
TARGET = "target_ah_home_cover"
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
FROZEN_LOGLOSS_DELTA = -0.00445
FROZEN_BRIER_DELTA = -0.00219
FROZEN_ECE_DELTA = 0.00314

REPORT_PATH = Path("outputs/reports/advanced_ah_market_models_predictive_audit.md")
SUMMARY_PATH = Path("outputs/reports/advanced_ah_market_models_predictive_summary.csv")
BUCKET_PATH = Path("outputs/reports/advanced_ah_market_models_edge_bucket_calibration.csv")
NEGATIVE_PATH = Path("outputs/reports/advanced_ah_market_models_negative_controls.csv")
ROBUSTNESS_PATH = Path("outputs/reports/advanced_ah_market_models_robustness.csv")
VALUE_REPORT_PATH = Path("outputs/reports/advanced_ah_market_models_value_review.md")
VALUE_FIXED_PATH = Path("outputs/reports/advanced_ah_market_models_value_fixed_rules.csv")
VALUE_NESTED_PATH = Path("outputs/reports/advanced_ah_market_models_value_nested_selection.csv")
VALUE_CONTROLS_PATH = Path("outputs/reports/advanced_ah_market_models_value_controls.csv")
VALUE_ROBUSTNESS_PATH = Path("outputs/reports/advanced_ah_market_models_value_robustness.csv")


CORE_FEATURES = ["AHh", "AvgAHH", "AvgAHA", "no_vig_ah_home_probability", "no_vig_ah_away_probability"]
MODELS = [
    "raw_market_baseline",
    "market_baseline_calibration_only",
    "logistic_l2",
    "logistic_elasticnet",
    "xgboost_shallow_frozen_control",
    "xgboost_depth3_regularized",
    "xgboost_market_residual",
    "small_mlp_regularized",
    "small_deep_cross_network",
]


@dataclass(frozen=True)
class EvalKey:
    model: str
    feature_group: str


def load_data() -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frame["league"] = league
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    for column in ["season_end_year", "FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA", "AvgCAHH", "AvgCAHA"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    home_raw = 1.0 / data["AvgAHH"]
    away_raw = 1.0 / data["AvgAHA"]
    total = home_raw + away_raw
    data["no_vig_ah_home_probability"] = home_raw / total
    data["no_vig_ah_away_probability"] = away_raw / total
    data["market_home_probability"] = data["no_vig_ah_home_probability"]
    data["market_away_probability"] = data["no_vig_ah_away_probability"]
    data["odds_spread"] = data["AvgAHH"] - data["AvgAHA"]
    data["implied_prob_gap"] = data["no_vig_ah_home_probability"] - data["no_vig_ah_away_probability"]
    data["overround_estimate"] = home_raw + away_raw
    data["abs_AHh"] = data["AHh"].abs()
    data["home_is_favourite_by_line"] = data["AHh"].lt(0).astype(float)
    data["away_is_favourite_by_line"] = data["AHh"].gt(0).astype(float)
    data["line_bucket"] = pd.cut(data["AHh"], bins=[-5, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 5], labels=False).astype("float")
    data["odds_bucket"] = pd.cut(data[["AvgAHH", "AvgAHA"]].min(axis=1), bins=[1.0, 1.8, 1.9, 2.0, 2.2, 10], labels=False).astype("float")
    data["league_layer"] = np.where(data["league"].isin(LAYER1), 1.0, np.where(data["league"].isin(LAYER2), 2.0, 3.0))
    data["season_era"] = np.select(
        [
            data["season_end_year"].lt(2012),
            data["season_end_year"].between(2012, 2016),
            data["season_end_year"].between(2017, 2019),
            data["season_end_year"].ge(2020),
        ],
        [0.0, 1.0, 2.0, 3.0],
        default=np.nan,
    )
    for league in LEAGUES:
        data[f"league_code_{league}"] = data["league"].eq(league).astype(float)
    for layer in [1, 2, 3]:
        data[f"league_layer_{layer}"] = data["league_layer"].eq(float(layer)).astype(float)
    for era in [0, 1, 2, 3]:
        data[f"season_era_{era}"] = data["season_era"].eq(float(era)).astype(float)
    margin = data["FTHG"] - data["FTAG"]
    adjusted = margin + data["AHh"]
    data[TARGET] = np.where(adjusted > 0, 1.0, np.where(adjusted < 0, 0.0, np.nan))
    home_settled = [settle_side(m, line, odds) for m, line, odds in zip(margin, data["AHh"], data["AvgAHH"])]
    away_settled = [settle_side(-m, -line if pd.notna(line) else np.nan, odds) for m, line, odds in zip(margin, data["AHh"], data["AvgAHA"])]
    data["home_profit"] = [item.profit for item in home_settled]
    data["away_profit"] = [item.profit for item in away_settled]
    data["home_label"] = [item.label for item in home_settled]
    data["away_label"] = [item.label for item in away_settled]
    if {"AvgCAHH", "AvgCAHA"}.issubset(data.columns):
        close_home = 1.0 / data["AvgCAHH"]
        close_away = 1.0 / data["AvgCAHA"]
        close_total = close_home + close_away
        data["closing_home_probability"] = close_home / close_total
        data["closing_away_probability"] = close_away / close_total
    else:
        data["closing_home_probability"] = np.nan
        data["closing_away_probability"] = np.nan
    required = ["Date", "league", "season_end_year", TARGET] + CORE_FEATURES
    data = data.dropna(subset=required).copy()
    data = data[data["AvgAHH"].gt(1.0) & data["AvgAHA"].gt(1.0)].copy()
    data["season_end_year"] = data["season_end_year"].astype(int)
    data["row_id"] = np.arange(len(data))
    return data.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def feature_groups(data: pd.DataFrame) -> dict[str, list[str]]:
    league_cols = [f"league_code_{league}" for league in LEAGUES]
    structure = [
        "AHh",
        "abs_AHh",
        "AvgAHH",
        "AvgAHA",
        "no_vig_ah_home_probability",
        "no_vig_ah_away_probability",
        "odds_spread",
        "implied_prob_gap",
        "overround_estimate",
        "home_is_favourite_by_line",
        "away_is_favourite_by_line",
        "line_bucket",
        "odds_bucket",
        "league_layer_1",
        "league_layer_2",
        "league_layer_3",
        "season_era_0",
        "season_era_1",
        "season_era_2",
        "season_era_3",
    ] + league_cols
    return {
        "core_market_5": CORE_FEATURES,
        "market_structure_safe": [column for column in structure if column in data.columns],
    }


def metric_values(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    y = frame[TARGET].astype(int).to_numpy()
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {"log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)), "ece": float(ece_binary(y, p))}


def xgb_fit_predict(train, validation, test, features, params, rounds=250, seed=42):
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_validation = imputer.transform(validation[features])
    x_test = imputer.transform(test[features])
    params = {**params, "objective": "binary:logistic", "eval_metric": "logloss", "seed": seed, "verbosity": 0}
    model = xgb.train(
        params,
        xgb.DMatrix(x_train, label=train[TARGET].astype(int).to_numpy(), feature_names=features),
        num_boost_round=rounds,
        evals=[(xgb.DMatrix(x_validation, label=validation[TARGET].astype(int).to_numpy(), feature_names=features), "validation")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return model.predict(xgb.DMatrix(x_validation, feature_names=features)), model.predict(xgb.DMatrix(x_test, feature_names=features))


def fit_model(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, model: str, features: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    if model == "raw_market_baseline":
        return validation["no_vig_ah_home_probability"].to_numpy(), test["no_vig_ah_home_probability"].to_numpy()
    if model == "market_baseline_calibration_only":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=seed))])
        pipe.fit(validation[["no_vig_ah_home_probability"]], validation[TARGET].astype(int))
        cls = list(pipe.named_steps["model"].classes_).index(1)
        return (
            pipe.predict_proba(validation[["no_vig_ah_home_probability"]])[:, cls],
            pipe.predict_proba(test[["no_vig_ah_home_probability"]])[:, cls],
        )
    if model in {"logistic_l2", "logistic_elasticnet"}:
        penalty = "l2" if model == "logistic_l2" else "elasticnet"
        solver = "lbfgs" if model == "logistic_l2" else "saga"
        kwargs = {"l1_ratio": 0.2} if model == "logistic_elasticnet" else {}
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=seed, penalty=penalty, solver=solver, C=0.5, **kwargs))])
        pipe.fit(train[features], train[TARGET].astype(int))
        cls = list(pipe.named_steps["model"].classes_).index(1)
        return pipe.predict_proba(validation[features])[:, cls], pipe.predict_proba(test[features])[:, cls]
    if model == "xgboost_shallow_frozen_control":
        return xgb_fit_predict(train, validation, test, features, {"max_depth": 2, "eta": 0.03, "lambda": 8.0, "alpha": 2.0}, 250, seed)
    if model == "xgboost_depth3_regularized":
        return xgb_fit_predict(train, validation, test, features, {"max_depth": 3, "eta": 0.025, "lambda": 12.0, "alpha": 4.0, "subsample": 0.85, "colsample_bytree": 0.85}, 250, seed)
    if model == "xgboost_market_residual":
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train[features])
        x_validation = imputer.transform(validation[features])
        x_test = imputer.transform(test[features])
        train_res = train[TARGET].astype(float).to_numpy() - train["no_vig_ah_home_probability"].to_numpy()
        val_res = validation[TARGET].astype(float).to_numpy() - validation["no_vig_ah_home_probability"].to_numpy()
        params = {"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 2, "eta": 0.02, "lambda": 12.0, "alpha": 3.0, "seed": seed, "verbosity": 0}
        booster = xgb.train(params, xgb.DMatrix(x_train, label=train_res, feature_names=features), num_boost_round=200, evals=[(xgb.DMatrix(x_validation, label=val_res, feature_names=features), "validation")], early_stopping_rounds=20, verbose_eval=False)
        return (
            np.clip(validation["no_vig_ah_home_probability"].to_numpy() + booster.predict(xgb.DMatrix(x_validation, feature_names=features)), 1e-6, 1 - 1e-6),
            np.clip(test["no_vig_ah_home_probability"].to_numpy() + booster.predict(xgb.DMatrix(x_test, feature_names=features)), 1e-6, 1 - 1e-6),
        )
    if model == "small_mlp_regularized":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(24,), alpha=0.02, learning_rate_init=0.001, max_iter=120, early_stopping=True, validation_fraction=0.15, random_state=seed))])
        pipe.fit(train[features], train[TARGET].astype(int))
        cls = list(pipe.named_steps["model"].classes_).index(1)
        return pipe.predict_proba(validation[features])[:, cls], pipe.predict_proba(test[features])[:, cls]
    if model == "small_deep_cross_network":
        return fit_dcn(train, validation, test, features, seed)
    raise ValueError(model)


class CrossLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.weight, 0.0, 0.02)

    def forward(self, x0, x):
        return x0 * torch.sum(x * self.weight, dim=1, keepdim=True) + self.bias + x


class DCN(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.cross1 = CrossLayer(width)
        self.cross2 = CrossLayer(width)
        self.deep = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Dropout(0.12), nn.Linear(32, 16), nn.ReLU())
        self.head = nn.Linear(width + 16, 1)

    def forward(self, x):
        crossed = self.cross2(x, self.cross1(x, x))
        return self.head(torch.cat([crossed, self.deep(x)], dim=1)).squeeze(-1)


def fit_dcn(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[features])).astype(np.float32)
    x_val = scaler.transform(imputer.transform(validation[features])).astype(np.float32)
    x_test = scaler.transform(imputer.transform(test[features])).astype(np.float32)
    y_train = train[TARGET].astype(float).to_numpy(np.float32)
    y_val = validation[TARGET].astype(float).to_numpy(np.float32)
    net = DCN(x_train.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    tx = torch.tensor(x_train)
    ty = torch.tensor(y_train)
    vx = torch.tensor(x_val)
    vy = torch.tensor(y_val)
    best_loss = math.inf
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    stale = 0
    gen = torch.Generator().manual_seed(seed)
    for _epoch in range(45):
        net.train()
        order = torch.randperm(len(tx), generator=gen)
        for start in range(0, len(order), 512):
            idx = order[start : start + 512]
            loss = loss_fn(net(tx[idx]), ty[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(vx), vy).item())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 6:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        val_p = torch.sigmoid(net(torch.tensor(x_val))).numpy()
        test_p = torch.sigmoid(net(torch.tensor(x_test))).numpy()
    return np.clip(val_p, 1e-6, 1 - 1e-6), np.clip(test_p, 1e-6, 1 - 1e-6)


def prediction_frame(
    frame: pd.DataFrame,
    model: str,
    feature_group: str,
    year: int,
    role: str,
    p: np.ndarray,
    market_home_probability: np.ndarray | None = None,
) -> pd.DataFrame:
    out = frame[["row_id", "league", "season_end_year", "Date", TARGET, "AHh", "AvgAHH", "AvgAHA", "no_vig_ah_home_probability", "no_vig_ah_away_probability", "market_home_probability", "market_away_probability", "home_profit", "away_profit", "home_label", "away_label", "closing_home_probability", "closing_away_probability"]].copy()
    if market_home_probability is not None:
        out["market_home_probability"] = np.asarray(market_home_probability, dtype=float)
        out["market_away_probability"] = 1.0 - out["market_home_probability"]
    out["model"] = model
    out["feature_group"] = feature_group
    out["test_year"] = year
    out["fold_role"] = role
    out["model_home_probability"] = p
    out["model_away_probability"] = 1.0 - out["model_home_probability"]
    out["home_edge"] = out["model_home_probability"] - out["market_home_probability"]
    out["away_edge"] = out["model_away_probability"] - out["market_away_probability"]
    out["home_clv"] = out["closing_home_probability"] - out["market_home_probability"]
    out["away_clv"] = out["closing_away_probability"] - out["market_away_probability"]
    return out


def run_predictions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = feature_groups(data)
    test_rows = []
    val_rows = []
    for fg, features in groups.items():
        for model in MODELS:
            for year in TEST_YEARS:
                train = data[data["season_end_year"].lt(year - 1)].dropna(subset=[TARGET]).copy()
                validation = data[data["season_end_year"].eq(year - 1)].dropna(subset=[TARGET]).copy()
                test = data[data["season_end_year"].eq(year)].copy()
                if len(train) == 0 or len(validation) == 0 or len(test) == 0 or train[TARGET].nunique() < 2:
                    continue
                val_p, test_p = fit_model(train, validation, test, model, features, seed=1000 + year)
                val_rows.append(prediction_frame(validation, model, fg, year, "validation", val_p))
                test_rows.append(prediction_frame(test, model, fg, year, "test", test_p))
    return pd.concat(test_rows, ignore_index=True), pd.concat(val_rows, ignore_index=True)


def summarize_predictions(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pred.groupby(group_cols):
        m = metric_values(g, g["model_home_probability"].to_numpy())
        b = metric_values(g, g["market_home_probability"].to_numpy())
        per_year = []
        for _year, gy in g.groupby("test_year"):
            my = metric_values(gy, gy["model_home_probability"].to_numpy())
            by = metric_values(gy, gy["market_home_probability"].to_numpy())
            per_year.append(my["log_loss"] < by["log_loss"])
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "rows": len(g),
                "test_years": ";".join(map(str, sorted(g["test_year"].unique()))),
                "mean_delta_log_loss_vs_raw_market": m["log_loss"] - b["log_loss"],
                "mean_delta_brier_vs_raw_market": m["brier"] - b["brier"],
                "mean_delta_ece_vs_raw_market": m["ece"] - b["ece"],
                "improved_years": int(sum(per_year)),
                "model_log_loss": m["log_loss"],
                "market_log_loss": b["log_loss"],
                "model_brier": m["brier"],
                "market_brier": b["brier"],
                "model_ece": m["ece"],
                "market_ece": b["ece"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def edge_buckets(pred: pd.DataFrame) -> pd.DataFrame:
    bins = [-1e-9, 0.01, 0.02, 0.03, 0.04, 0.05, 10]
    labels = ["0.00-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05", ">=0.05"]
    rows = []
    p_bucket = pd.cut(pred["model_home_probability"], bins=np.linspace(0, 1, 11), include_lowest=True)
    edge = pred["model_home_probability"] - pred["market_home_probability"]
    e_bucket = pd.cut(edge, bins=bins, labels=labels)
    for cols, bucket_name, bucket_values in [(["model", "feature_group"], "edge_bucket", e_bucket), (["model", "feature_group"], "probability_bucket", p_bucket)]:
        tmp = pred.copy()
        tmp[bucket_name] = bucket_values.astype(str)
        for key, g in tmp.groupby(cols + [bucket_name]):
            if len(g) < 20:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(cols + [bucket_name], key))
            row.update(
                {
                    "rows": len(g),
                    "average_model_probability": float(g["model_home_probability"].mean()),
                    "average_market_probability": float(g["market_home_probability"].mean()),
                    "realised_cover_rate": float(g[TARGET].mean()),
                    "calibration_error": float(g[TARGET].mean() - g["model_home_probability"].mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_negative_controls(data: pd.DataFrame, key: EvalKey) -> pd.DataFrame:
    features = feature_groups(data)[key.feature_group]
    controls = ["shuffled_train_labels", "random_noise_replacing_market_features", "permuted_market_features_within_league_season", "league_only_without_market_odds", "opposite_label_sanity_check"]
    rows = []
    for control in controls:
        parts = []
        for year in TEST_YEARS:
            train = data[data["season_end_year"].lt(year - 1)].dropna(subset=[TARGET]).copy()
            validation = data[data["season_end_year"].eq(year - 1)].dropna(subset=[TARGET]).copy()
            test = data[data["season_end_year"].eq(year)].copy()
            if len(train) == 0 or len(validation) == 0 or len(test) == 0:
                continue
            original_test_market = test["market_home_probability"].to_numpy(copy=True)
            rng = np.random.default_rng(7000 + year)
            f = features.copy()
            if control == "shuffled_train_labels":
                y = train[TARGET].to_numpy(copy=True)
                rng.shuffle(y)
                train[TARGET] = y
            elif control == "random_noise_replacing_market_features":
                for current in [train, validation, test]:
                    for col in f:
                        current[col] = rng.normal(0, 1, len(current))
            elif control == "permuted_market_features_within_league_season":
                for current in [train, validation, test]:
                    for _, idx in current.groupby(["league", "season_end_year"]).groups.items():
                        for col in f:
                            vals = current.loc[idx, col].to_numpy(copy=True)
                            rng.shuffle(vals)
                            current.loc[idx, col] = vals
            elif control == "league_only_without_market_odds":
                f = [f"league_code_{league}" for league in LEAGUES]
            elif control == "opposite_label_sanity_check":
                train[TARGET] = 1 - train[TARGET].astype(int)
            _, p = fit_model(train, validation, test, key.model, f, 8000 + year)
            parts.append(prediction_frame(test, key.model, key.feature_group, year, "test", p, original_test_market))
        if parts:
            pred = pd.concat(parts, ignore_index=True)
            row = summarize_predictions(pred.assign(control=control), ["control"]).iloc[0].to_dict()
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
            train = frame[frame["season_end_year"].lt(year - 1)].dropna(subset=[TARGET]).copy()
            validation = frame[frame["season_end_year"].eq(year - 1)].dropna(subset=[TARGET]).copy()
            test = frame[frame["season_end_year"].eq(year)].copy()
            if len(train) == 0 or len(validation) == 0 or len(test) == 0 or train[TARGET].nunique() < 2:
                continue
            _, p = fit_model(train, validation, test, key.model, features, 9000 + year)
            parts.append(prediction_frame(test, key.model, key.feature_group, year, "test", p))
        if parts:
            row = summarize_predictions(pd.concat(parts, ignore_index=True).assign(robustness=name), ["robustness"]).iloc[0].to_dict()
        else:
            row = {"robustness": name, "rows": 0}
        row["model"] = key.model
        row["feature_group"] = key.feature_group
        rows.append(row)
    return pd.DataFrame(rows)


def add_value_columns(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["AvgAHH"] = out["AvgAHH"].astype(float)
    out["AvgAHA"] = out["AvgAHA"].astype(float)
    return out


def value_review(test_pred: pd.DataFrame, val_pred: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    fixed_rows = []
    nested_rows = []
    controls_rows = []
    robust_rows = []
    final_classification = "predictive_only_no_value"
    for _, candidate in candidates.iterrows():
        model = candidate["model"]
        fg = candidate["feature_group"]
        test = add_value_columns(test_pred[(test_pred["model"].eq(model)) & (test_pred["feature_group"].eq(fg))].copy())
        val = add_value_columns(val_pred[(val_pred["model"].eq(model)) & (val_pred["feature_group"].eq(fg))].copy())
        selections = {}
        for rule in rule_grid():
            selected = select_rule(test, rule)
            stats = value_summarize(selected, "fixed_rule", rule)
            stats["model"] = model
            stats["feature_group"] = fg
            fixed_rows.append(stats)
            selections[rule.name] = selected
        nested, nested_bets = value_nested_selection(test.rename(columns={"test_year": "fold_test_year"}), val.rename(columns={"test_year": "fold_test_year"}), f"{model}:{fg}")
        nested_rows.append(nested)
        best = pd.DataFrame([r for r in fixed_rows if r.get("model") == model and r.get("feature_group") == fg]).sort_values(["profit", "z_score"], ascending=[False, False]).iloc[0]
        best_rule = next(r for r in rule_grid() if r.name == best["rule_name"])
        best_bets = selections[best_rule.name]
        controls_rows.append(value_controls(test, best_bets, best_rule, f"{model}_{fg}_best_fixed_rule"))
        controls_rows.append(value_controls(test, nested_bets, best_rule if len(nested_bets) else None, f"{model}_{fg}_nested_portfolio"))
        robust_rows.append(value_robustness(best_bets, f"{model}_{fg}_best_fixed_rule"))
        robust_rows.append(value_robustness(nested_bets, f"{model}_{fg}_nested_portfolio"))
        current = classify_value(nested_bets, pd.concat(controls_rows, ignore_index=True), pd.concat(robust_rows, ignore_index=True))
        if current == "forward_paper_candidate":
            final_classification = current
        elif current == "research_only" and final_classification != "forward_paper_candidate":
            final_classification = current
    return (
        pd.DataFrame(fixed_rows),
        pd.concat(nested_rows, ignore_index=True, sort=False) if nested_rows else pd.DataFrame(),
        pd.concat(controls_rows, ignore_index=True, sort=False) if controls_rows else pd.DataFrame(),
        pd.concat(robust_rows, ignore_index=True, sort=False) if robust_rows else pd.DataFrame(),
        final_classification,
    )


def advancement_candidates(summary: pd.DataFrame, buckets: pd.DataFrame, robustness: pd.DataFrame, negative: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        if row["model"] in {"raw_market_baseline", "xgboost_shallow_frozen_control"}:
            continue
        key = (row["model"], row["feature_group"])
        high = buckets[(buckets["model"].eq(key[0])) & (buckets["feature_group"].eq(key[1])) & (buckets.get("edge_bucket", "").eq(">=0.05"))]
        high_sane = len(high) == 0 or abs(float(high["calibration_error"].mean())) <= 0.08
        neg = negative[(negative["model"].eq(key[0])) & (negative["feature_group"].eq(key[1]))]
        controls_fail = len(neg) > 0 and not (neg["mean_delta_log_loss_vs_raw_market"] < 0).any()
        rob = robustness[(robustness["model"].eq(key[0])) & (robustness["feature_group"].eq(key[1]))]
        best_ok = len(rob) > 0
        for required in ["exclude_best_performing_season", "exclude_best_performing_league"]:
            r = rob[rob["robustness"].eq(required)]
            if len(r) == 0 or float(r["mean_delta_log_loss_vs_raw_market"].iloc[0]) >= 0:
                best_ok = False
        passes = (
            row["mean_delta_log_loss_vs_raw_market"] < FROZEN_LOGLOSS_DELTA
            and row["mean_delta_brier_vs_raw_market"] < FROZEN_BRIER_DELTA
            and row["mean_delta_ece_vs_raw_market"] <= FROZEN_ECE_DELTA + 0.002
            and row["improved_years"] >= 6
            and controls_fail
            and high_sane
            and best_ok
        )
        rows.append({**row.to_dict(), "advancement_gate_passed": bool(passes), "high_edge_sane": high_sane, "negative_controls_failed": controls_fail, "robustness_best_exclusions_positive": best_ok})
    return pd.DataFrame(rows)


def write_reports(summary, buckets, negative, robustness, candidates, value_fixed, value_nested, value_controls_df, value_robustness_df, final_classification):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best = summary.sort_values("mean_delta_log_loss_vs_raw_market").head(20)
    lines = [
        "# Advanced AH Market-Only Models Predictive Audit",
        "",
        f"Final classification: `{final_classification}`",
        "",
        "Scope: controlled advanced AH market-only branch. No live betting, Transfermarkt, player features, lineups, team-name features, closing odds as features/selection inputs, scraping, external APIs, or confirmed edge claims were used.",
        "",
        "## Best Predictive Rows",
        "",
        markdown_table(best, ["model", "feature_group", "rows", "mean_delta_log_loss_vs_raw_market", "mean_delta_brier_vs_raw_market", "mean_delta_ece_vs_raw_market", "improved_years"], 30),
        "",
        "## Advancement Gate",
        "",
        markdown_table(candidates, ["model", "feature_group", "advancement_gate_passed", "mean_delta_log_loss_vs_raw_market", "mean_delta_brier_vs_raw_market", "mean_delta_ece_vs_raw_market", "improved_years", "high_edge_sane", "negative_controls_failed", "robustness_best_exclusions_positive"], 40),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    value_lines = [
        "# Advanced AH Market Models Value Review",
        "",
        f"Final classification: `{final_classification}`",
        "",
        "Locked value review was run only for models that passed the advancement gate. No new betting thresholds were searched.",
        "",
        "## Fixed Rules",
        "",
        markdown_table(value_fixed.sort_values(["profit", "z_score"], ascending=[False, False]) if len(value_fixed) else value_fixed, ["model", "feature_group", "rule_name", "bets", "profit", "roi", "z_score"], 40),
        "",
        "## Nested Selection",
        "",
        markdown_table(value_nested, ["regime", "test_year", "selected_rule", "selection_status", "test_bets", "test_profit", "test_roi", "test_z"], 60),
        "",
        "No confirmed edge is claimed.",
        "",
    ]
    VALUE_REPORT_PATH.write_text("\n".join(value_lines), encoding="utf-8")


def main() -> None:
    data = load_data()
    test_pred, val_pred = run_predictions(data)
    summary = summarize_predictions(test_pred, ["model", "feature_group"])
    full_summary = pd.concat(
        [
            summary.assign(summary_scope="overall"),
            summarize_predictions(test_pred, ["model", "feature_group", "test_year"]).assign(summary_scope="per_year"),
            summarize_predictions(test_pred, ["model", "feature_group", "league"]).assign(summary_scope="per_league"),
            summarize_predictions(test_pred, ["model", "feature_group", "league", "test_year"]).assign(summary_scope="per_league_year"),
        ],
        ignore_index=True,
        sort=False,
    )
    buckets = edge_buckets(test_pred)
    gate_rows = summary[
        (~summary["model"].isin(["raw_market_baseline", "xgboost_shallow_frozen_control"]))
        & (summary["mean_delta_log_loss_vs_raw_market"].lt(FROZEN_LOGLOSS_DELTA))
        & (summary["mean_delta_brier_vs_raw_market"].lt(FROZEN_BRIER_DELTA))
        & (summary["mean_delta_ece_vs_raw_market"].le(FROZEN_ECE_DELTA + 0.002))
        & (summary["improved_years"].ge(6))
    ].copy()
    if gate_rows.empty:
        gate_rows = summary.sort_values("mean_delta_log_loss_vs_raw_market").head(1).copy()
    negative_parts = []
    robustness_parts = []
    for _, gate_row in gate_rows.iterrows():
        key = EvalKey(str(gate_row["model"]), str(gate_row["feature_group"]))
        candidate_pred = test_pred[(test_pred["model"].eq(key.model)) & (test_pred["feature_group"].eq(key.feature_group))]
        per_season = summarize_predictions(candidate_pred, ["test_year"])
        best_season = int(per_season.sort_values("mean_delta_log_loss_vs_raw_market").iloc[0]["test_year"])
        per_league = summarize_predictions(candidate_pred, ["league"])
        best_league = str(per_league.sort_values("mean_delta_log_loss_vs_raw_market").iloc[0]["league"])
        negative_parts.append(run_negative_controls(data, key))
        robustness_parts.append(run_robustness(data, key, best_season, best_league))
    negative = pd.concat(negative_parts, ignore_index=True, sort=False) if negative_parts else pd.DataFrame()
    robustness = pd.concat(robustness_parts, ignore_index=True, sort=False) if robustness_parts else pd.DataFrame()
    candidates = advancement_candidates(summary, buckets, robustness, negative)
    passed = candidates[candidates["advancement_gate_passed"]].copy()
    if len(passed):
        value_fixed, value_nested, value_controls_df, value_robustness_df, value_class = value_review(test_pred, val_pred, passed)
    else:
        value_fixed, value_nested, value_controls_df, value_robustness_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        value_class = "predictive_only_no_value" if (summary["mean_delta_log_loss_vs_raw_market"] < 0).any() else "reject"
    final_classification = value_class
    full_summary.to_csv(SUMMARY_PATH, index=False)
    buckets.to_csv(BUCKET_PATH, index=False)
    negative.to_csv(NEGATIVE_PATH, index=False)
    robustness.to_csv(ROBUSTNESS_PATH, index=False)
    value_fixed.to_csv(VALUE_FIXED_PATH, index=False)
    value_nested.to_csv(VALUE_NESTED_PATH, index=False)
    value_controls_df.to_csv(VALUE_CONTROLS_PATH, index=False)
    value_robustness_df.to_csv(VALUE_ROBUSTNESS_PATH, index=False)
    write_reports(summary, buckets, negative, robustness, candidates, value_fixed, value_nested, value_controls_df, value_robustness_df, final_classification)
    print(
        {
            "data_rows": len(data),
            "prediction_rows": len(test_pred),
            "summary_rows": len(summary),
            "advancement_passed": int(len(passed)),
            "value_fixed_rows": len(value_fixed),
            "classification": final_classification,
        }
    )


if __name__ == "__main__":
    main()
