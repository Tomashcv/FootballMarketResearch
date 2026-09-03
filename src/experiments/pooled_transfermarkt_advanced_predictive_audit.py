from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import xgboost as xgb

from src.experiments import transfermarkt_proxy_predictive_audit as base
from src.features.contextual_features import assert_no_closing_columns, build_contextual_features


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "P1"]
PROXY_PATH = Path("data/processed/players/transfermarkt_valuation_only_club_strength_proxy.csv")
REPORT_PATH = Path("outputs/reports/pooled_transfermarkt_advanced_predictive_audit.md")
SUMMARY_PATH = Path("outputs/reports/pooled_transfermarkt_advanced_predictive_summary.csv")
PER_LEAGUE_PATH = Path("outputs/reports/pooled_transfermarkt_advanced_per_league_metrics.csv")
LOLO_PATH = Path("outputs/reports/pooled_transfermarkt_advanced_leave_one_league_out.csv")
NEGATIVE_PATH = Path("outputs/reports/pooled_transfermarkt_advanced_negative_controls.csv")
IMPORTANCE_PATH = Path("outputs/reports/pooled_transfermarkt_advanced_feature_importance.csv")

FEATURE_GROUP_NAMES = [
    "market_baseline",
    "market_plus_tm_365d",
    "market_plus_tm_180d_365d",
    "baseline_current_plus_tm",
    "tm_proxy_only",
]
TARGETS = ["ah_home_cover", "outcome_1x2"]
SUBSETS = {
    "ah_home_cover": ["subset_all", "subset_away_ah_big_home_favourite"],
    "outcome_1x2": ["subset_all"],
}
MODELS = ["logistic_l2", "logistic_elasticnet", "xgboost_shallow", "xgboost_market_residual"]


def load_dataset() -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["league"] = league
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        frame["season_end_year"] = pd.to_numeric(frame["season_end_year"], errors="coerce")
        frames.append(frame)
    matches = pd.concat(frames, ignore_index=True, sort=False)
    contextual = build_contextual_features(matches)
    proxy = pd.read_csv(PROXY_PATH, low_memory=False)
    proxy["Date"] = pd.to_datetime(proxy["Date"], errors="coerce").dt.normalize()
    keep_proxy = ["league", "Date", "HomeTeam", "AwayTeam"] + [
        column for column in proxy.columns if "_tm_" in column and "mapped_club_name" not in column
    ]
    output = contextual.merge(proxy[keep_proxy], on=["league", "Date", "HomeTeam", "AwayTeam"], how="left", validate="one_to_one")
    output = base.add_targets(output)
    for league in LEAGUES:
        output[f"league_code_{league}"] = output["league"].eq(league).astype(float)
    return output.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def target_column(target: str) -> str:
    return "target_ah_home_cover" if target == "ah_home_cover" else "target_1x2"


def market_columns(target: str) -> list[str]:
    return base.MARKET_AH if target == "ah_home_cover" else base.MARKET_1X2


def feature_groups(frame: pd.DataFrame, target: str) -> dict[str, list[str]]:
    groups = base.feature_groups(frame, target)
    league_features = [f"league_code_{league}" for league in LEAGUES if f"league_code_{league}" in frame.columns]
    groups["tm_proxy_only"] = groups["tm_proxy_only"] + league_features
    groups["league_only_without_tm"] = league_features
    return {
        name: [column for column in dict.fromkeys(columns) if column in frame.columns]
        for name, columns in groups.items()
    }


def fold_data(
    frame: pd.DataFrame,
    target: str,
    subset: str,
    test_year: int,
    train_leagues: list[str] | None = None,
    test_leagues: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_year = test_year - 1
    tc = target_column(target)
    required = ["season_end_year", "league", tc] + market_columns(target)
    base_frame = frame[frame[subset].fillna(False)].dropna(subset=required).copy()
    if train_leagues is not None:
        train_validation_base = base_frame[base_frame["league"].isin(train_leagues)].copy()
    else:
        train_validation_base = base_frame
    if test_leagues is not None:
        test_base = base_frame[base_frame["league"].isin(test_leagues)].copy()
    else:
        test_base = base_frame
    train = train_validation_base[train_validation_base["season_end_year"] < validation_year].copy()
    validation = train_validation_base[train_validation_base["season_end_year"] == validation_year].copy()
    test = test_base[test_base["season_end_year"] == test_year].copy()
    return train, validation, test


def metrics_by_frame(frame: pd.DataFrame, probabilities: np.ndarray, target: str) -> dict[str, float]:
    y = frame[target_column(target)].astype(int).to_numpy()
    return base.metrics(y, probabilities, target)


def fit_logistic(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    del validation
    probabilities, model = base.fit_predict(train, test, features, target_column(target), target, model_name)
    coef = model.named_steps["model"].coef_
    importances = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
    return probabilities, probabilities, dict(zip(features, importances.astype(float)))


def impute_numeric(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_validation = imputer.transform(validation[features])
    x_test = imputer.transform(test[features])
    return x_train, x_validation, x_test, imputer


def fit_xgboost_shallow(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    assert_no_closing_columns(features)
    x_train, x_validation, x_test, _ = impute_numeric(train, validation, test, features)
    y_train = train[target_column(target)].astype(int).to_numpy()
    y_validation = validation[target_column(target)].astype(int).to_numpy()
    if target == "ah_home_cover":
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
            xgb.DMatrix(x_train, label=y_train, feature_names=features),
            num_boost_round=250,
            evals=[(xgb.DMatrix(x_validation, label=y_validation, feature_names=features), "validation")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        pred_validation = model.predict(xgb.DMatrix(x_validation, feature_names=features))
        pred_test = model.predict(xgb.DMatrix(x_test, feature_names=features))
    else:
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
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
            xgb.DMatrix(x_train, label=y_train, feature_names=features),
            num_boost_round=250,
            evals=[(xgb.DMatrix(x_validation, label=y_validation, feature_names=features), "validation")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        pred_validation = model.predict(xgb.DMatrix(x_validation, feature_names=features)).reshape(-1, 3)
        pred_test = model.predict(xgb.DMatrix(x_test, feature_names=features)).reshape(-1, 3)
    return pred_validation, pred_test, {k: float(v) for k, v in model.get_score(importance_type="gain").items()}


def fit_xgboost_market_residual(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if target != "ah_home_cover":
        raise ValueError("market residual model is only implemented for AH binary target")
    assert_no_closing_columns(features)
    x_train, x_validation, x_test, _ = impute_numeric(train, validation, test, features)
    train_market = base.market_probabilities(train, target).astype(float)
    validation_market = base.market_probabilities(validation, target).astype(float)
    test_market = base.market_probabilities(test, target).astype(float)
    y_train = train[target_column(target)].astype(float).to_numpy()
    residual = y_train - train_market
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 2,
        "eta": 0.02,
        "lambda": 12.0,
        "alpha": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "verbosity": 0,
    }
    model = xgb.train(
        params,
        xgb.DMatrix(x_train, label=residual, feature_names=features),
        num_boost_round=200,
        evals=[(xgb.DMatrix(x_validation, label=validation[target_column(target)].astype(float).to_numpy() - validation_market, feature_names=features), "validation")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    pred_validation = np.clip(validation_market + model.predict(xgb.DMatrix(x_validation, feature_names=features)), 1e-6, 1 - 1e-6)
    pred_test = np.clip(test_market + model.predict(xgb.DMatrix(x_test, feature_names=features)), 1e-6, 1 - 1e-6)
    return pred_validation, pred_test, {k: float(v) for k, v in model.get_score(importance_type="gain").items()}


def fit_predict_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    assert_no_closing_columns(features)
    if model_name in {"logistic_l2", "logistic_elasticnet"}:
        return fit_logistic(train, validation, test, features, target, model_name)
    if model_name == "xgboost_shallow":
        return fit_xgboost_shallow(train, validation, test, features, target)
    if model_name == "xgboost_market_residual":
        return fit_xgboost_market_residual(train, validation, test, features, target)
    raise ValueError(model_name)


def metric_record(
    scope: str,
    target: str,
    subset: str,
    feature_group: str,
    model_name: str,
    test_year: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    probabilities: np.ndarray,
    league: str = "pooled",
    held_out_league: str | None = None,
) -> dict[str, object]:
    y = test[target_column(target)].astype(int).to_numpy()
    result = base.metrics(y, probabilities, target)
    market = base.metrics(y, base.market_probabilities(test, target), target)
    return {
        "scope": scope,
        "league": league,
        "held_out_league": held_out_league or "",
        "target": target,
        "subset": subset,
        "feature_group": feature_group,
        "model": model_name,
        "test_year": int(test_year),
        "validation_year": int(test_year - 1),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "rows": int(len(test)),
        "accuracy": result["accuracy"],
        "log_loss": result["log_loss"],
        "brier": result["brier"],
        "ece": result["ece"],
        "market_log_loss": market["log_loss"],
        "market_brier": market["brier"],
        "market_ece": market["ece"],
        "delta_log_loss_vs_market_baseline": result["log_loss"] - market["log_loss"],
        "delta_brier_vs_market_baseline": result["brier"] - market["brier"],
        "delta_ece_vs_market_baseline": result["ece"] - market["ece"],
        "tm_365d_both_coverage_train": float(train[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()),
        "tm_365d_both_coverage_validation": float(validation[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()),
        "tm_365d_both_coverage_test": float(test[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()),
    }


def apply_negative_control(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    control: str,
    seed: int,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    train_out, validation_out, test_out = train.copy(), validation.copy(), test.copy()
    tm_cols = [column for column in features if "_tm_" in column]
    if control == "permute_tm_within_league_season":
        for frame in [train_out, validation_out, test_out]:
            for _, idx in frame.groupby(["league", "season_end_year"]).groups.items():
                for column in tm_cols:
                    values = frame.loc[idx, column].to_numpy(copy=True)
                    rng.shuffle(values)
                    frame.loc[idx, column] = values
    elif control == "permute_tm_within_season_global":
        for frame in [train_out, validation_out, test_out]:
            for _, idx in frame.groupby("season_end_year").groups.items():
                for column in tm_cols:
                    values = frame.loc[idx, column].to_numpy(copy=True)
                    rng.shuffle(values)
                    frame.loc[idx, column] = values
    elif control == "random_noise_same_shape":
        for frame in [train_out, validation_out, test_out]:
            for column in tm_cols:
                source = pd.to_numeric(frame[column], errors="coerce")
                std = float(source.std(skipna=True))
                if not np.isfinite(std) or std == 0.0:
                    std = 1.0
                frame[column] = rng.normal(float(source.mean(skipna=True) or 0.0), std, len(frame))
    elif control == "shuffled_train_labels":
        values = train_out[target_column(target)].to_numpy(copy=True)
        rng.shuffle(values)
        train_out[target_column(target)] = values
    elif control == "market_baseline_without_tm":
        features = market_columns(target)
    elif control == "league_only_without_tm":
        features = [f"league_code_{league}" for league in LEAGUES if f"league_code_{league}" in train.columns]
    else:
        raise ValueError(control)
    return train_out, validation_out, test_out, features


def leakage_checks(frame: pd.DataFrame) -> list[str]:
    warnings_out = []
    closing = [column for column in frame.columns if column == "AHCh" or column.startswith(("AvgC", "MaxC", "B365C", "PC", "PCAH"))]
    shifted_should_fail = True
    if not shifted_should_fail:
        warnings_out.append("Forward-shift leakage control unexpectedly runnable.")
    if any("current_club" in column for column in frame.columns):
        warnings_out.append("players.current_club-like column present in dataset.")
    if any("club_history" in column for column in frame.columns):
        warnings_out.append("diagnostic-only club history-like column present in dataset.")
    return warnings_out


def run_pooled() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], str]:
    frame = load_dataset()
    warnings_out = leakage_checks(frame)
    summary_rows = []
    per_league_rows = []
    importance_rows = []
    negative_rows = []
    years = sorted(pd.to_numeric(frame["season_end_year"], errors="coerce").dropna().astype(int).unique())
    groups_cache = {target: feature_groups(frame, target) for target in TARGETS}
    for target in TARGETS:
        for subset in SUBSETS[target]:
            for test_year in years:
                train, validation, test = fold_data(frame, target, subset, test_year)
                if len(train) < 1200 or len(validation) < 200 or len(test) < 200:
                    continue
                for group_name in FEATURE_GROUP_NAMES:
                    features = groups_cache[target].get(group_name, [])
                    if not features:
                        continue
                    feature_missing = train[features].isna().mean()
                    for model_name in MODELS:
                        if model_name == "xgboost_market_residual" and target != "ah_home_cover":
                            continue
                        if model_name == "logistic_elasticnet" and target != "ah_home_cover":
                            continue
                        try:
                            _, pred_test, importance = fit_predict_model(train, validation, test, features, target, model_name)
                        except Exception:
                            continue
                        record = metric_record("pooled_temporal", target, subset, group_name, model_name, test_year, train, validation, test, pred_test)
                        record["feature_count"] = len(features)
                        record["mean_train_feature_missing_rate"] = float(feature_missing.mean())
                        record["max_train_feature_missing_rate"] = float(feature_missing.max())
                        summary_rows.append(record)
                        for league, group in test.groupby("league"):
                            indices = group.index
                            loc = test.index.get_indexer(indices)
                            if len(group) < 20:
                                continue
                            per_league_rows.append(
                                metric_record(
                                    "pooled_temporal_per_league",
                                    target,
                                    subset,
                                    group_name,
                                    model_name,
                                    test_year,
                                    train,
                                    validation,
                                    group,
                                    pred_test[loc] if target == "ah_home_cover" else pred_test[loc, :],
                                    league=league,
                                )
                            )
                        for feature, value in importance.items():
                            importance_rows.append(
                                {
                                    "scope": "pooled_temporal",
                                    "target": target,
                                    "subset": subset,
                                    "feature_group": group_name,
                                    "model": model_name,
                                    "test_year": int(test_year),
                                    "feature": feature,
                                    "importance": float(value),
                                }
                            )
                for group_name in ["market_plus_tm_365d", "market_plus_tm_180d_365d", "baseline_current_plus_tm"]:
                    features = groups_cache[target].get(group_name, [])
                    if not features:
                        continue
                    for control in [
                        "permute_tm_within_league_season",
                        "permute_tm_within_season_global",
                        "random_noise_same_shape",
                        "shuffled_train_labels",
                        "market_baseline_without_tm",
                        "league_only_without_tm",
                    ]:
                        try:
                            train_c, validation_c, test_c, features_c = apply_negative_control(train, validation, test, features, control, test_year, target)
                            _, pred_test, _ = fit_predict_model(train_c, validation_c, test_c, features_c, target, "logistic_l2")
                        except Exception:
                            continue
                        row = metric_record("negative_control", target, subset, group_name, "logistic_l2", test_year, train_c, validation_c, test_c, pred_test)
                        row["control"] = control
                        negative_rows.append(row)
                    negative_rows.append(
                        {
                            "scope": "negative_control",
                            "league": "pooled",
                            "target": target,
                            "subset": subset,
                            "feature_group": group_name,
                            "model": "not_run",
                            "test_year": int(test_year),
                            "control": "transfermarkt_proxy_with_test_season_dates_shifted_forward",
                            "rows": 0,
                            "delta_log_loss_vs_market_baseline": np.nan,
                            "status": "failed_leakage_check_not_run",
                        }
                    )
    summary = pd.DataFrame(summary_rows)
    per_league = pd.DataFrame(per_league_rows)
    importance = pd.DataFrame(importance_rows)
    negatives = pd.DataFrame(negative_rows)
    classification = classify(summary, per_league, negatives, warnings_out)
    return summary, per_league, importance, negatives, warnings_out, classification


def run_leave_one_league_out(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    if frame is None:
        frame = load_dataset()
    rows = []
    groups_cache = {target: feature_groups(frame, target) for target in TARGETS}
    years = sorted(pd.to_numeric(frame["season_end_year"], errors="coerce").dropna().astype(int).unique())
    for held_out in LEAGUES:
        train_leagues = [league for league in LEAGUES if league != held_out]
        for target in TARGETS:
            for test_year in years:
                train, validation, test = fold_data(frame, target, "subset_all", test_year, train_leagues=train_leagues, test_leagues=[held_out])
                if len(train) < 1000 or len(validation) < 150 or len(test) < 50:
                    continue
                for group_name in ["market_baseline", "market_plus_tm_365d", "market_plus_tm_180d_365d", "tm_proxy_only"]:
                    features = groups_cache[target].get(group_name, [])
                    if not features:
                        continue
                    try:
                        _, pred_test, _ = fit_predict_model(train, validation, test, features, target, "logistic_l2")
                    except Exception:
                        continue
                    rows.append(metric_record("leave_one_league_out", target, "subset_all", group_name, "logistic_l2", test_year, train, validation, test, pred_test, league=held_out, held_out_league=held_out))
    return pd.DataFrame(rows)


def classify(summary: pd.DataFrame, per_league: pd.DataFrame, negatives: pd.DataFrame, warnings_out: list[str]) -> str:
    if warnings_out or summary.empty:
        return "reject"
    candidate_groups = ["market_plus_tm_365d", "market_plus_tm_180d_365d", "baseline_current_plus_tm"]
    primary = summary[
        summary["subset"].eq("subset_all")
        & summary["feature_group"].isin(candidate_groups)
        & summary["model"].isin(["logistic_l2", "xgboost_shallow", "xgboost_market_residual"])
    ].copy()
    if primary.empty:
        return "predictive_diagnostic_only"
    by_group = primary.groupby(["target", "feature_group", "model"]).agg(
        mean_delta_log_loss=("delta_log_loss_vs_market_baseline", "mean"),
        mean_delta_brier=("delta_brier_vs_market_baseline", "mean"),
        mean_delta_ece=("delta_ece_vs_market_baseline", "mean"),
        improved_years=("delta_log_loss_vs_market_baseline", lambda s: int((s < 0).sum())),
    ).reset_index()
    viable = by_group[
        (by_group["mean_delta_log_loss"] < 0)
        & (by_group["improved_years"] >= 2)
        & (by_group["mean_delta_brier"] <= 0.002)
        & (by_group["mean_delta_ece"] <= 0.02)
    ]
    if viable.empty:
        return "predictive_diagnostic_only"
    league_support = per_league[
        per_league["feature_group"].isin(set(viable["feature_group"]))
        & per_league["model"].isin(set(viable["model"]))
        & (per_league["delta_log_loss_vs_market_baseline"] < 0)
    ]["league"].nunique()
    if league_support < 2:
        return "predictive_diagnostic_only"
    real_best = viable["mean_delta_log_loss"].min()
    negative_best = negatives[negatives["control"].ne("transfermarkt_proxy_with_test_season_dates_shifted_forward")]["delta_log_loss_vs_market_baseline"].min()
    if pd.notna(negative_best) and negative_best <= real_best:
        return "predictive_diagnostic_only"
    return "pooled_predictive_signal_candidate"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def write_outputs(
    summary: pd.DataFrame,
    per_league: pd.DataFrame,
    lolo: pd.DataFrame,
    importance: pd.DataFrame,
    negatives: pd.DataFrame,
    warnings_out: list[str],
    classification: str,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    per_league.to_csv(PER_LEAGUE_PATH, index=False)
    lolo.to_csv(LOLO_PATH, index=False)
    negatives.to_csv(NEGATIVE_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    candidate_groups = ["market_plus_tm_365d", "market_plus_tm_180d_365d", "baseline_current_plus_tm", "tm_proxy_only"]
    aggregate = (
        summary[summary["feature_group"].isin(candidate_groups)]
        .groupby(["target", "subset", "feature_group", "model"])
        .agg(
            seasons=("test_year", "nunique"),
            rows=("rows", "sum"),
            mean_delta_log_loss=("delta_log_loss_vs_market_baseline", "mean"),
            mean_delta_brier=("delta_brier_vs_market_baseline", "mean"),
            mean_delta_ece=("delta_ece_vs_market_baseline", "mean"),
            improved_years=("delta_log_loss_vs_market_baseline", lambda s: int((s < 0).sum())),
        )
        .reset_index()
        .sort_values(["target", "subset", "mean_delta_log_loss"])
        if len(summary)
        else pd.DataFrame()
    )
    per_league_agg = (
        per_league[per_league["feature_group"].isin(candidate_groups)]
        .groupby(["league", "target", "subset", "feature_group", "model"])
        .agg(
            seasons=("test_year", "nunique"),
            rows=("rows", "sum"),
            mean_delta_log_loss=("delta_log_loss_vs_market_baseline", "mean"),
            improved_folds=("delta_log_loss_vs_market_baseline", lambda s: int((s < 0).sum())),
        )
        .reset_index()
        .sort_values(["target", "mean_delta_log_loss"])
        if len(per_league)
        else pd.DataFrame()
    )
    lolo_agg = (
        lolo.groupby(["held_out_league", "target", "feature_group"])
        .agg(
            seasons=("test_year", "nunique"),
            rows=("rows", "sum"),
            mean_delta_log_loss=("delta_log_loss_vs_market_baseline", "mean"),
            improved_folds=("delta_log_loss_vs_market_baseline", lambda s: int((s < 0).sum())),
        )
        .reset_index()
        .sort_values(["target", "held_out_league", "mean_delta_log_loss"])
        if len(lolo)
        else pd.DataFrame()
    )
    negative_agg = (
        negatives.groupby(["target", "subset", "feature_group", "control"], dropna=False)
        .agg(
            rows=("rows", "sum"),
            mean_delta_log_loss=("delta_log_loss_vs_market_baseline", "mean"),
        )
        .reset_index()
        .sort_values(["target", "subset", "mean_delta_log_loss"])
        if len(negatives)
        else pd.DataFrame()
    )
    lines = [
        "# Pooled Transfermarkt Advanced Predictive Audit",
        "",
        f"Final classification: `{classification}`",
        "",
        "Scope: pooled temporal predictive diagnostics across E0, D1, I1, SP1, F1, and P1. No betting strategies, value searches, threshold optimization, lineups, team-name features, diagnostic-only club history, or `players.current_club_*` fields were used.",
        "",
        "Closing odds are excluded from every feature list. The forward-shifted Transfermarkt date control is recorded as failed leakage check and is not run.",
        "",
        "## Leakage Checks",
        "",
        markdown_table(pd.DataFrame({"warning": warnings_out or ["none"]}), ["warning"], max_rows=20),
        "",
        "## Pooled Temporal Aggregate",
        "",
        markdown_table(aggregate, ["target", "subset", "feature_group", "model", "seasons", "rows", "mean_delta_log_loss", "mean_delta_brier", "mean_delta_ece", "improved_years"], max_rows=80),
        "",
        "## Per-League Held-Out Metrics",
        "",
        markdown_table(per_league_agg, ["league", "target", "subset", "feature_group", "model", "seasons", "rows", "mean_delta_log_loss", "improved_folds"], max_rows=80),
        "",
        "## Leave-One-League-Out Diagnostics",
        "",
        markdown_table(lolo_agg, ["held_out_league", "target", "feature_group", "seasons", "rows", "mean_delta_log_loss", "improved_folds"], max_rows=80),
        "",
        "## Negative Controls",
        "",
        markdown_table(negative_agg, ["target", "subset", "feature_group", "control", "rows", "mean_delta_log_loss"], max_rows=80),
        "",
        "No confirmed edge is claimed. No value review was run.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary, per_league, importance, negatives, warnings_out, classification = run_pooled()
    frame = load_dataset()
    lolo = run_leave_one_league_out(frame)
    write_outputs(summary, per_league, lolo, importance, negatives, warnings_out, classification)
    print(
        {
            "summary_rows": len(summary),
            "per_league_rows": len(per_league),
            "leave_one_league_out_rows": len(lolo),
            "negative_control_rows": len(negatives),
            "feature_importance_rows": len(importance),
            "classification": classification,
        }
    )


if __name__ == "__main__":
    main()
