from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments.post_backfill_locked_ah_value_review import classify as classify_value
from src.experiments.post_backfill_locked_ah_value_review import controls as value_controls
from src.experiments.post_backfill_locked_ah_value_review import markdown_table
from src.experiments.post_backfill_locked_ah_value_review import nested_selection as value_nested_selection
from src.experiments.post_backfill_locked_ah_value_review import robustness as value_robustness
from src.experiments.post_backfill_locked_ah_value_review import rule_grid
from src.experiments.post_backfill_locked_ah_value_review import select_rule
from src.experiments.post_backfill_locked_ah_value_review import summarize as value_summarize
from src.experiments.transfermarkt_proxy_predictive_audit import ece_binary


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_PATH = Path("data/processed/features/football_feature_matrix_v1_1.csv")
DICT_PATH = Path("outputs/reports/football_feature_matrix_v1_feature_dictionary.csv")
DELTA_DICT_PATH = Path("outputs/reports/football_feature_matrix_v1_1_feature_dictionary_delta.csv")
TARGET = "target_ah_home_cover"
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "E1", "E2", "E3", "SC0"]
ENGLISH_LOWER = {"E1", "E2", "E3"}
PRIOR_LOGLOSS_DELTA = -0.0050
PRIOR_BRIER_DELTA = -0.0025

REPORT_PATH = Path("outputs/reports/feature_matrix_v1_1_ah_predictive_audit.md")
SUMMARY_PATH = Path("outputs/reports/feature_matrix_v1_1_ah_predictive_summary.csv")
BUCKET_PATH = Path("outputs/reports/feature_matrix_v1_1_ah_edge_bucket_calibration.csv")
NEGATIVE_PATH = Path("outputs/reports/feature_matrix_v1_1_ah_negative_controls.csv")
ROBUSTNESS_PATH = Path("outputs/reports/feature_matrix_v1_1_ah_robustness.csv")
VALUE_REPORT_PATH = Path("outputs/reports/feature_matrix_v1_1_ah_value_review.md")
VALUE_FIXED_PATH = Path("outputs/reports/feature_matrix_v1_1_ah_value_fixed_rules.csv")
VALUE_NESTED_PATH = Path("outputs/reports/feature_matrix_v1_1_ah_value_nested_selection.csv")
VALUE_CONTROLS_PATH = Path("outputs/reports/feature_matrix_v1_1_ah_value_controls.csv")
VALUE_ROBUSTNESS_PATH = Path("outputs/reports/feature_matrix_v1_1_ah_value_robustness.csv")
VALUE_FIXED_COLUMNS = ["scope", "label", "side", "rule_name", "edge_threshold", "min_odds", "bets", "profit", "roi", "z_score", "max_drawdown", "league_concentration_hhi", "top_league_share", "push_rate", "half_win_loss_rate", "average_odds", "average_edge", "average_model_probability", "average_market_probability", "average_clv", "clv_positive_rate", "average_line", "model", "feature_group"]
VALUE_NESTED_COLUMNS = ["regime", "test_year", "selected_rule", "selection_status", "test_bets", "test_profit", "test_roi", "test_z"]
VALUE_CONTROLS_COLUMNS = ["label", "control", "bets", "profit", "roi", "z_score"]
VALUE_ROBUSTNESS_COLUMNS = ["label", "robustness", "bets", "profit", "roi", "z_score"]

MODELS = [
    "raw_market_baseline",
    "logistic_l2",
    "xgboost_shallow",
    "xgboost_depth3_regularized",
    "xgboost_market_residual",
]
CORE_GROUP = "ah_core_market_5"


@dataclass(frozen=True)
class EvalKey:
    model: str
    feature_group: str


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["match_date"] = pd.to_datetime(data["match_date"], errors="coerce").dt.normalize()
    string_columns = {"match_id", "match_date", "league", "home_team", "away_team", "source_processed_file", "feature_matrix_version", "league_era_bucket", "ah_avg_market_source", "ou25_avg_market_source", "x1x2_avg_market_source", "memory_scaling_policy", "target_outcome_1x2"}
    for column in data.columns:
        if column not in string_columns:
            converted = pd.to_numeric(data[column], errors="coerce")
            if converted.notna().sum() or data[column].isna().all():
                data[column] = converted
    for column in ["AHh", "AvgAHH", "AvgAHA", "MaxAHH", "MaxAHA", "no_vig_ah_home_probability", "no_vig_ah_away_probability"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    missing = data["no_vig_ah_home_probability"].isna() | data["no_vig_ah_away_probability"].isna()
    implied_home = 1.0 / data["AvgAHH"]
    implied_away = 1.0 / data["AvgAHA"]
    implied_total = implied_home + implied_away
    data.loc[missing, "no_vig_ah_home_probability"] = implied_home[missing] / implied_total[missing]
    data.loc[missing, "no_vig_ah_away_probability"] = implied_away[missing] / implied_total[missing]
    data["market_home_probability"] = data["no_vig_ah_home_probability"]
    data["market_away_probability"] = data["no_vig_ah_away_probability"]
    data["closing_home_probability"] = np.nan
    data["closing_away_probability"] = np.nan
    target = pd.to_numeric(data[TARGET], errors="coerce")
    home_units = pd.to_numeric(data["ah_settlement_units_home"], errors="coerce")
    away_units = -home_units
    data["home_profit"] = np.where(home_units.gt(0), home_units * (data["AvgAHH"] - 1.0), home_units)
    data["away_profit"] = np.where(away_units.gt(0), away_units * (data["AvgAHA"] - 1.0), away_units)
    label_map = {1.0: "full_win", 0.5: "half_win", 0.0: "push", -0.5: "half_loss", -1.0: "full_loss"}
    data["home_label"] = home_units.round(6).map(label_map).fillna("invalid")
    data["away_label"] = away_units.round(6).map(label_map).fillna("invalid")
    required = ["match_date", "league", "season_end_year", TARGET, "ah_settlement_units_home", "AHh", "AvgAHH", "AvgAHA", "market_home_probability", "market_away_probability"]
    data = data.dropna(subset=required).copy()
    data = data[data["AvgAHH"].gt(1.0) & data["AvgAHA"].gt(1.0)].copy()
    data["season_end_year"] = data["season_end_year"].astype(int)
    data["row_id"] = np.arange(len(data))
    return data.sort_values(["match_date", "league", "match_id"]).reset_index(drop=True)


def dictionary() -> pd.DataFrame:
    frames = [pd.read_csv(DICT_PATH)]
    if DELTA_DICT_PATH.exists():
        frames.append(pd.read_csv(DELTA_DICT_PATH))
    return pd.concat(frames, ignore_index=True, sort=False)


def safe_numeric_columns(data: pd.DataFrame) -> list[str]:
    banned_tokens = [
        "target",
        "settlement",
        "score",
        "result",
        "closing",
        "close",
        "transfermarkt",
        "player",
        "lineup",
        "squad",
        "current_club",
        "source",
        "path",
        "team",
        "club",
    ]
    banned_exact = {
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "league",
        "source_processed_file",
        "feature_matrix_version",
        "target_outcome_1x2",
        "league_era_bucket",
        "ah_avg_market_source",
        "ou25_avg_market_source",
        "x1x2_avg_market_source",
        "memory_scaling_policy",
        "row_id",
        "home_profit",
        "away_profit",
        "home_label",
        "away_label",
        "market_home_probability",
        "market_away_probability",
        "closing_home_probability",
        "closing_away_probability",
    }
    cols = []
    for column in data.columns:
        lower = column.lower()
        if column in banned_exact or column.startswith("C") or any(token in lower for token in banned_tokens):
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            cols.append(column)
    return cols


def feature_groups(data: pd.DataFrame) -> dict[str, list[str]]:
    fd = dictionary()
    by_group = {name: set(fd.loc[fd["feature_group"].eq(name), "column"]) for name in fd["feature_group"].dropna().unique()}
    safe = set(safe_numeric_columns(data))
    core = [
        "AHh",
        "AvgAHH",
        "AvgAHA",
        "no_vig_ah_home_probability",
        "no_vig_ah_away_probability",
    ]
    line_structure = core + [
        "abs_AHh",
        "home_is_ah_favourite",
        "away_is_ah_favourite",
        "ah_line_bucket",
        "ah_price_bucket",
        "AH_overround",
        "ah_market_entropy",
        "ah_odds_spread",
    ]
    bookmaker_specific = [
        "B365AH",
        "B365AHH",
        "B365AHA",
        "BFEAHH",
        "BFEAHA",
        "BbAH",
        "BbAHh",
        "BbMxAHH",
        "BbMxAHA",
        "BbAvAHH",
        "BbAvAHA",
        "GBAH",
        "GBAHH",
        "GBAHA",
        "LBAH",
        "LBAHH",
        "LBAHA",
        "PAHH",
        "PAHA",
        "MaxAHH",
        "MaxAHA",
    ]
    odds_board = sorted(
        (
            (by_group.get("odds_board_features", set()) & safe)
            | {c for c in data.columns if c in bookmaker_specific}
            | {"AH_bookmaker_count", "ah_bookmaker_count", "ah_missing_bookmaker_count", "ah_home_best_vs_avg_price_gap", "ah_away_best_vs_avg_price_gap"}
        )
        & set(data.columns)
        & safe
    )
    form_elo_groups = set().union(
        by_group.get("rolling_team_form", set()),
        by_group.get("rest_and_schedule", set()),
        by_group.get("league_trends", set()),
        by_group.get("internal_past_only_elo", set()),
    )
    form_elo = sorted(form_elo_groups & safe)
    full_safe = sorted(safe | set(core) | {"AH_overround", "AH_bookmaker_count"})
    return {
        "ah_core_market_5": [c for c in core if c in data.columns],
        "ah_market_line_structure": [c for c in line_structure if c in data.columns],
        "ah_market_plus_odds_board": sorted(set(line_structure) | set(odds_board)),
        "ah_market_plus_form_elo": sorted(set(line_structure) | set(form_elo)),
        "ah_feature_matrix_v1_1_full_safe": full_safe,
    }


def metric_values(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    y = frame[TARGET].astype(int).to_numpy()
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {"log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)), "ece": float(ece_binary(y, p))}


def xgb_fit_predict(train, validation, test, features, params, rounds=220, seed=42):
    features = [c for c in features if c in train.columns and train[c].notna().any()]
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_validation = imputer.transform(validation[features])
    x_test = imputer.transform(test[features])
    params = {**params, "objective": "binary:logistic", "eval_metric": "logloss", "seed": seed, "verbosity": 0, "nthread": 4}
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
    features = [c for c in features if c in train.columns and train[c].notna().any()]
    if not features:
        raise ValueError(f"No active features for {model}")
    if model == "raw_market_baseline":
        return validation["market_home_probability"].to_numpy(), test["market_home_probability"].to_numpy()
    if model == "logistic_l2":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=400, tol=1e-3, random_state=seed, penalty="l2", solver="lbfgs", C=0.5))])
        pipe.fit(train[features], train[TARGET].astype(int))
        cls = list(pipe.named_steps["model"].classes_).index(1)
        return pipe.predict_proba(validation[features])[:, cls], pipe.predict_proba(test[features])[:, cls]
    if model == "xgboost_shallow":
        return xgb_fit_predict(train, validation, test, features, {"max_depth": 2, "eta": 0.03, "lambda": 8.0, "alpha": 2.0}, 220, seed)
    if model == "xgboost_depth3_regularized":
        return xgb_fit_predict(train, validation, test, features, {"max_depth": 3, "eta": 0.025, "lambda": 12.0, "alpha": 4.0, "subsample": 0.85, "colsample_bytree": 0.85}, 240, seed)
    if model == "xgboost_market_residual":
        features = [c for c in features if c in train.columns and train[c].notna().any()]
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train[features])
        x_validation = imputer.transform(validation[features])
        x_test = imputer.transform(test[features])
        train_res = train[TARGET].astype(float).to_numpy() - train["market_home_probability"].to_numpy()
        val_res = validation[TARGET].astype(float).to_numpy() - validation["market_home_probability"].to_numpy()
        params = {"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 2, "eta": 0.02, "lambda": 12.0, "alpha": 3.0, "seed": seed, "verbosity": 0, "nthread": 4}
        booster = xgb.train(params, xgb.DMatrix(x_train, label=train_res, feature_names=features), num_boost_round=200, evals=[(xgb.DMatrix(x_validation, label=val_res, feature_names=features), "validation")], early_stopping_rounds=20, verbose_eval=False)
        return (
            np.clip(validation["market_home_probability"].to_numpy() + booster.predict(xgb.DMatrix(x_validation, feature_names=features)), 1e-6, 1 - 1e-6),
            np.clip(test["market_home_probability"].to_numpy() + booster.predict(xgb.DMatrix(x_test, feature_names=features)), 1e-6, 1 - 1e-6),
        )
    raise ValueError(model)


def prediction_frame(frame: pd.DataFrame, model: str, feature_group: str, year: int, role: str, p: np.ndarray, market_home_probability: np.ndarray | None = None) -> pd.DataFrame:
    cols = ["row_id", "match_id", "league", "season_end_year", "match_date", TARGET, "AHh", "AvgAHH", "AvgAHA", "market_home_probability", "market_away_probability", "home_profit", "away_profit", "home_label", "away_label", "closing_home_probability", "closing_away_probability"]
    out = frame[cols].copy()
    if market_home_probability is not None:
        out["market_home_probability"] = np.asarray(market_home_probability, dtype=float)
        out["market_away_probability"] = 1.0 - out["market_home_probability"]
    out["model"] = model
    out["feature_group"] = feature_group
    out["test_year"] = year
    out["fold_test_year"] = year
    out["fold_role"] = role
    out["Date"] = out["match_date"]
    out["model_home_probability"] = p
    out["model_away_probability"] = 1.0 - out["model_home_probability"]
    out["home_edge"] = out["model_home_probability"] - out["market_home_probability"]
    out["away_edge"] = out["model_away_probability"] - out["market_away_probability"]
    out["home_clv"] = np.nan
    out["away_clv"] = np.nan
    return out


def run_predictions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_rows = []
    val_rows = []
    for fg, features in feature_groups(data).items():
        for model in MODELS:
            for year in TEST_YEARS:
                print(f"predictive_fold model={model} feature_group={fg} year={year}", flush=True)
                train = data[data["season_end_year"].lt(year - 1)].copy()
                validation = data[data["season_end_year"].eq(year - 1)].copy()
                test = data[data["season_end_year"].eq(year)].copy()
                if len(train) == 0 or len(validation) == 0 or len(test) == 0 or train[TARGET].nunique() < 2:
                    continue
                val_p, test_p = fit_model(train, validation, test, model, features, seed=1000 + year)
                val_rows.append(prediction_frame(validation, model, fg, year, "validation", val_p))
                test_rows.append(prediction_frame(test, model, fg, year, "test", test_p))
    return pd.concat(test_rows, ignore_index=True), pd.concat(val_rows, ignore_index=True)


def summarize_predictions(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pred.groupby(group_cols, dropna=False):
        m = metric_values(g, g["model_home_probability"].to_numpy())
        b = metric_values(g, g["market_home_probability"].to_numpy())
        per_year = []
        for _, gy in g.groupby("test_year"):
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


def add_core_market_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    metric_cols = {
        "model_log_loss": "delta_log_loss_vs_same_model_core_market_5",
        "model_brier": "delta_brier_vs_same_model_core_market_5",
        "model_ece": "delta_ece_vs_same_model_core_market_5",
    }
    for new_col in metric_cols.values():
        out[new_col] = np.nan
    if "model" not in out.columns or "feature_group" not in out.columns:
        return out
    overall_like = out["feature_group"].notna()
    for model, core in out[overall_like & out["feature_group"].eq(CORE_GROUP)].groupby("model", dropna=False):
        if len(core) == 0:
            continue
        core_row = core.iloc[0]
        mask = out["model"].eq(model)
        for metric, new_col in metric_cols.items():
            out.loc[mask, new_col] = out.loc[mask, metric] - float(core_row[metric])
    return out


def edge_buckets(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = [-1e-9, 0.01, 0.02, 0.03, 0.04, 0.05, 10]
    labels = ["0.00-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05", ">=0.05"]
    tmp = pred.copy()
    tmp["edge_bucket"] = pd.cut(tmp["model_home_probability"] - tmp["market_home_probability"], bins=bins, labels=labels).astype(str)
    tmp["probability_bucket"] = pd.cut(tmp["model_home_probability"], bins=np.linspace(0, 1, 11), include_lowest=True).astype(str)
    for bucket_col in ["edge_bucket", "probability_bucket"]:
        for key, g in tmp.groupby(["model", "feature_group", bucket_col]):
            if len(g) < 20:
                continue
            row = dict(zip(["model", "feature_group", bucket_col], key))
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


def league_only_features(frame: pd.DataFrame) -> list[str]:
    for league in sorted(frame["league"].dropna().unique()):
        frame[f"league_code_{league}"] = frame["league"].eq(league).astype(float)
    return [f"league_code_{league}" for league in sorted(frame["league"].dropna().unique())]


def run_negative_controls(data: pd.DataFrame, key: EvalKey) -> pd.DataFrame:
    controls = ["shuffled_train_labels", "random_noise_replacing_feature_matrix_features", "permuted_features_within_league_season", "league_only_without_market_odds", "opposite_label_sanity_check"]
    base_features = feature_groups(data)[key.feature_group]
    rows = []
    for control in controls:
        parts = []
        for year in TEST_YEARS:
            print(f"negative_control model={key.model} feature_group={key.feature_group} control={control} year={year}", flush=True)
            train = data[data["season_end_year"].lt(year - 1)].copy()
            validation = data[data["season_end_year"].eq(year - 1)].copy()
            test = data[data["season_end_year"].eq(year)].copy()
            if len(train) == 0 or len(validation) == 0 or len(test) == 0 or train[TARGET].nunique() < 2:
                continue
            original_market = test["market_home_probability"].to_numpy(copy=True)
            rng = np.random.default_rng(7000 + year)
            features = base_features.copy()
            protected_market = {"AHh", "AvgAHH", "AvgAHA", "no_vig_ah_home_probability", "no_vig_ah_away_probability"}
            if control == "shuffled_train_labels":
                y = train[TARGET].to_numpy(copy=True)
                rng.shuffle(y)
                train[TARGET] = y
            elif control == "random_noise_replacing_feature_matrix_features":
                replace = [c for c in features if c not in protected_market]
                for current in [train, validation, test]:
                    for col in replace:
                        current[col] = rng.normal(0, 1, len(current))
            elif control == "permuted_features_within_league_season":
                replace = [c for c in features if c not in protected_market]
                for current in [train, validation, test]:
                    for _, idx in current.groupby(["league", "season_end_year"]).groups.items():
                        for col in replace:
                            vals = current.loc[idx, col].to_numpy(copy=True)
                            rng.shuffle(vals)
                            current.loc[idx, col] = vals
            elif control == "league_only_without_market_odds":
                combined = pd.concat([train, validation, test], ignore_index=True, sort=False)
                features = league_only_features(combined)
                train = combined.iloc[: len(train)].copy()
                validation = combined.iloc[len(train) : len(train) + len(validation)].copy()
                test = combined.iloc[len(train) + len(validation) :].copy()
            elif control == "opposite_label_sanity_check":
                train[TARGET] = 1 - train[TARGET].astype(int)
            _, p = fit_model(train, validation, test, key.model, features, 8000 + year)
            parts.append(prediction_frame(test, key.model, key.feature_group, year, "test", p, original_market))
        if parts:
            row = summarize_predictions(pd.concat(parts).assign(control=control), ["control"]).iloc[0].to_dict()
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
    ]
    exclusions.extend((f"exclude_{league}", lambda df, league=league: df[~df["league"].eq(league)]) for league in sorted(data["league"].dropna().unique()))
    rows = []
    for name, fn in exclusions:
        print(f"robustness model={key.model} feature_group={key.feature_group} check={name}", flush=True)
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
        row = summarize_predictions(pd.concat(parts).assign(robustness=name), ["robustness"]).iloc[0].to_dict() if parts else {"robustness": name, "rows": 0}
        row["model"] = key.model
        row["feature_group"] = key.feature_group
        rows.append(row)
    return pd.DataFrame(rows)


def advancement_candidates(summary: pd.DataFrame, buckets: pd.DataFrame, robustness: pd.DataFrame, negative: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        if row["model"] in {"raw_market_baseline", "xgboost_shallow"} or row["feature_group"] == CORE_GROUP:
            continue
        high = buckets[(buckets["model"].eq(row["model"])) & (buckets["feature_group"].eq(row["feature_group"])) & (buckets.get("edge_bucket", pd.Series(dtype=str)).eq(">=0.05"))]
        high_sane = len(high) == 0 or abs(float(high["calibration_error"].mean())) <= 0.08
        neg = negative[(negative["model"].eq(row["model"])) & (negative["feature_group"].eq(row["feature_group"]))]
        controls_fail = len(neg) > 0 and not (neg["mean_delta_log_loss_vs_raw_market"] < 0).any()
        rob = robustness[(robustness["model"].eq(row["model"])) & (robustness["feature_group"].eq(row["feature_group"]))]
        best_ok = True
        for required in ["exclude_best_performing_season", "exclude_best_performing_league"]:
            r = rob[rob["robustness"].eq(required)]
            if len(r) == 0 or float(r["mean_delta_log_loss_vs_raw_market"].iloc[0]) >= 0 or float(r["mean_delta_brier_vs_raw_market"].iloc[0]) >= 0:
                best_ok = False
        beats_core = (
            pd.notna(row.get("delta_log_loss_vs_same_model_core_market_5"))
            and pd.notna(row.get("delta_brier_vs_same_model_core_market_5"))
            and float(row["delta_log_loss_vs_same_model_core_market_5"]) < -0.0002
            and float(row["delta_brier_vs_same_model_core_market_5"]) < -0.0001
        )
        beats_prior = (row["mean_delta_log_loss_vs_raw_market"] < PRIOR_LOGLOSS_DELTA and row["mean_delta_brier_vs_raw_market"] < PRIOR_BRIER_DELTA) or (
            row["mean_delta_log_loss_vs_raw_market"] < 0 and row["mean_delta_brier_vs_raw_market"] < 0 and high_sane and row["mean_delta_ece_vs_raw_market"] <= 0 and beats_core
        )
        passes = (
            row["mean_delta_log_loss_vs_raw_market"] < 0
            and row["mean_delta_brier_vs_raw_market"] < 0
            and beats_core
            and beats_prior
            and row["mean_delta_ece_vs_raw_market"] <= 0.003
            and row["improved_years"] >= 6
            and controls_fail
            and high_sane
            and best_ok
        )
        rows.append({**row.to_dict(), "advancement_gate_passed": bool(passes), "beats_or_justifies_prior_benchmark": bool(beats_prior), "beats_same_model_core_market_5": bool(beats_core), "high_edge_sane": bool(high_sane), "negative_controls_failed": bool(controls_fail), "robustness_best_exclusions_positive": bool(best_ok)})
    return pd.DataFrame(rows)


def value_review(test_pred: pd.DataFrame, val_pred: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    fixed_rows = []
    nested_rows = []
    controls_rows = []
    robust_rows = []
    final_classification = "predictive_only_no_value"
    for _, candidate in candidates.iterrows():
        model = candidate["model"]
        fg = candidate["feature_group"]
        test = test_pred[(test_pred["model"].eq(model)) & (test_pred["feature_group"].eq(fg))].copy()
        val = val_pred[(val_pred["model"].eq(model)) & (val_pred["feature_group"].eq(fg))].copy()
        selections = {}
        for rule in rule_grid():
            selected = select_rule(test, rule)
            stats = value_summarize(selected, "fixed_rule", rule)
            stats["model"] = model
            stats["feature_group"] = fg
            fixed_rows.append(stats)
            selections[rule.name] = selected
        nested, nested_bets = value_nested_selection(test, val, f"{model}:{fg}")
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
    return pd.DataFrame(fixed_rows), pd.concat(nested_rows, ignore_index=True, sort=False) if nested_rows else pd.DataFrame(), pd.concat(controls_rows, ignore_index=True, sort=False) if controls_rows else pd.DataFrame(), pd.concat(robust_rows, ignore_index=True, sort=False) if robust_rows else pd.DataFrame(), final_classification


def write_reports(summary, candidates, value_fixed, value_nested, final_classification, data, groups):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best = summary[summary["summary_scope"].eq("overall")].sort_values("mean_delta_log_loss_vs_raw_market").head(30)
    gate = candidates.sort_values("mean_delta_log_loss_vs_raw_market")
    core = summary[(summary["summary_scope"].eq("overall")) & (summary["feature_group"].eq(CORE_GROUP))].sort_values("mean_delta_log_loss_vs_raw_market")
    lines = [
        "# feature_matrix_v1_1 AH Predictive Audit",
        "",
        f"Final classification: `{final_classification}`",
        "",
        "Scope: historical-training modern-test AH home-cover audit using `data/processed/features/football_feature_matrix_v1_1.csv`. Pushes were excluded via `target_ah_home_cover` missingness. No live betting, confirmed edge claim, Transfermarkt, player features, lineups, team-name direct features, closing-odds features, closing-odds selection, or post-test threshold optimization were used.",
        "",
        f"Rows after AH target/price filtering: `{len(data)}`. Test years: `{';'.join(map(str, TEST_YEARS))}`.",
        "",
        "Feature group widths: " + ", ".join(f"`{k}`={len(v)}" for k, v in groups.items()) + ".",
        "",
        "AH line note: v1.1 restores `AHh` from processed pre-match market rows. The reproduction feature group `ah_core_market_5` uses exactly `AHh`, `AvgAHH`, `AvgAHA`, `no_vig_ah_home_probability`, and `no_vig_ah_away_probability`.",
        "",
        "## Core Market Reproduction",
        "",
        markdown_table(core, ["model", "feature_group", "rows", "mean_delta_log_loss_vs_raw_market", "mean_delta_brier_vs_raw_market", "mean_delta_ece_vs_raw_market", "improved_years"], 20),
        "",
        "## Best Predictive Rows",
        "",
        markdown_table(best, ["model", "feature_group", "rows", "mean_delta_log_loss_vs_raw_market", "mean_delta_brier_vs_raw_market", "mean_delta_ece_vs_raw_market", "delta_log_loss_vs_same_model_core_market_5", "delta_brier_vs_same_model_core_market_5", "improved_years"], 30),
        "",
        "## Advancement Gate",
        "",
        markdown_table(gate, ["model", "feature_group", "advancement_gate_passed", "beats_same_model_core_market_5", "beats_or_justifies_prior_benchmark", "mean_delta_log_loss_vs_raw_market", "mean_delta_brier_vs_raw_market", "mean_delta_ece_vs_raw_market", "delta_log_loss_vs_same_model_core_market_5", "delta_brier_vs_same_model_core_market_5", "improved_years", "high_edge_sane", "negative_controls_failed", "robustness_best_exclusions_positive"], 80),
        "",
        "No confirmed edge is claimed.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    value_lines = [
        "# feature_matrix_v1_1 AH Value Review",
        "",
        f"Final classification: `{final_classification}`",
        "",
        "Locked value review was run only for models that passed the predictive gate. The fixed and nested rules are the predeclared prior AH rules; no live betting or new threshold search was used.",
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
    groups = feature_groups(data)
    test_pred, val_pred = run_predictions(data)
    overall = add_core_market_comparison(summarize_predictions(test_pred, ["model", "feature_group"])).assign(summary_scope="overall")
    full_summary = pd.concat(
        [
            overall,
            summarize_predictions(test_pred, ["model", "feature_group", "test_year"]).assign(summary_scope="per_year"),
            summarize_predictions(test_pred, ["model", "feature_group", "league"]).assign(summary_scope="per_league"),
            summarize_predictions(test_pred, ["model", "feature_group", "league", "test_year"]).assign(summary_scope="per_league_year"),
        ],
        ignore_index=True,
        sort=False,
    )
    buckets = edge_buckets(test_pred)
    prelim = overall[
        (~overall["model"].isin(["raw_market_baseline", "xgboost_shallow"]))
        & (~overall["feature_group"].eq(CORE_GROUP))
        & (overall["mean_delta_log_loss_vs_raw_market"].lt(0))
        & (overall["mean_delta_brier_vs_raw_market"].lt(0))
        & (overall["delta_log_loss_vs_same_model_core_market_5"].lt(0))
        & (overall["delta_brier_vs_same_model_core_market_5"].lt(0))
        & (overall["improved_years"].ge(6))
    ].copy()
    if prelim.empty:
        prelim = overall[(~overall["model"].eq("raw_market_baseline")) & (~overall["feature_group"].eq(CORE_GROUP))].sort_values("mean_delta_log_loss_vs_raw_market").head(2).copy()
    negative_parts = []
    robustness_parts = []
    for _, row in prelim.iterrows():
        key = EvalKey(str(row["model"]), str(row["feature_group"]))
        candidate_pred = test_pred[test_pred["model"].eq(key.model) & test_pred["feature_group"].eq(key.feature_group)]
        per_season = summarize_predictions(candidate_pred, ["test_year"])
        best_season = int(per_season.sort_values("mean_delta_log_loss_vs_raw_market").iloc[0]["test_year"])
        per_league = summarize_predictions(candidate_pred, ["league"])
        best_league = str(per_league.sort_values("mean_delta_log_loss_vs_raw_market").iloc[0]["league"])
        negative_parts.append(run_negative_controls(data, key))
        robustness_parts.append(run_robustness(data, key, best_season, best_league))
    negative = pd.concat(negative_parts, ignore_index=True, sort=False) if negative_parts else pd.DataFrame()
    robustness = pd.concat(robustness_parts, ignore_index=True, sort=False) if robustness_parts else pd.DataFrame()
    candidates = advancement_candidates(overall, buckets, robustness, negative)
    passed = candidates[candidates["advancement_gate_passed"]].copy()
    if len(passed):
        value_fixed, value_nested, value_controls_df, value_robustness_df, value_class = value_review(test_pred, val_pred, passed)
    else:
        value_fixed, value_nested, value_controls_df, value_robustness_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        value_class = "predictive_only_no_value" if (overall["mean_delta_log_loss_vs_raw_market"] < 0).any() else "reject"
    full_summary.to_csv(SUMMARY_PATH, index=False)
    buckets.to_csv(BUCKET_PATH, index=False)
    negative.to_csv(NEGATIVE_PATH, index=False)
    robustness.to_csv(ROBUSTNESS_PATH, index=False)
    (value_fixed if len(value_fixed) else pd.DataFrame(columns=VALUE_FIXED_COLUMNS)).to_csv(VALUE_FIXED_PATH, index=False)
    (value_nested if len(value_nested) else pd.DataFrame(columns=VALUE_NESTED_COLUMNS)).to_csv(VALUE_NESTED_PATH, index=False)
    (value_controls_df if len(value_controls_df) else pd.DataFrame(columns=VALUE_CONTROLS_COLUMNS)).to_csv(VALUE_CONTROLS_PATH, index=False)
    (value_robustness_df if len(value_robustness_df) else pd.DataFrame(columns=VALUE_ROBUSTNESS_COLUMNS)).to_csv(VALUE_ROBUSTNESS_PATH, index=False)
    write_reports(full_summary, candidates, value_fixed, value_nested, value_class, data, groups)
    print(
        {
            "data_rows": len(data),
            "prediction_rows": len(test_pred),
            "summary_rows": len(full_summary),
            "advancement_passed": int(len(passed)),
            "classification": value_class,
        }
    )


if __name__ == "__main__":
    main()
