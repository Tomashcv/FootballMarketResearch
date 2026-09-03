from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.metrics import expected_calibration_error
from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced
from src.experiments import e0_away_ah_memory_odds_combo_review as memory_review
from src.experiments import market_disagreement_1x2_feature_audit as disagreement
from src.features.contextual_features import assert_no_closing_columns
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS, summarize

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")

REPORT_PATH = Path("outputs/reports/e0_away_ah_related_market_bookmaker_disagreement_review.md")
SUMMARY_PATH = Path("outputs/reports/e0_away_ah_related_market_bookmaker_disagreement_summary.csv")
IMPORTANCE_PATH = Path("outputs/reports/e0_away_ah_related_market_bookmaker_disagreement_feature_importance.csv")
BETS_PATH = Path("outputs/reports/e0_away_ah_related_market_bookmaker_disagreement_bets.csv")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/related_market_bookmaker_disagreement_review")

SEED = 17
MIN_VALIDATION_BETS = 12
SCORE_QUANTILES = [0.60, 0.70, 0.80]
TARGET = advanced.TARGET_COLUMN
MODEL_CONFIGS = [
    ("logistic_l2", "logistic_l2", {"C": 0.5, "penalty": "l2", "solver": "lbfgs"}),
    ("logistic_elastic_net", "logistic_elastic_net", {"C": 0.5, "penalty": "elasticnet", "l1_ratio": 0.25, "solver": "saga"}),
    (
        "xgboost",
        "xgboost",
        {
            "n_estimators": 80,
            "max_depth": 2,
            "learning_rate": 0.04,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 8,
            "reg_lambda": 4.0,
        },
    ),
]
RELATED_MARKET_STRENGTH_COLUMNS = [
    "away_1x2_market_strength_minus_ah_market_strength",
    "home_1x2_market_strength_minus_ah_market_strength",
    "favourite_strength_disagreement",
    "draw_pressure_index",
    "away_ps_prob_minus_avg_prob",
    "max_minus_avg_home",
    "max_minus_avg_draw",
    "max_minus_avg_away",
    "bookmaker_probability_std_home",
    "bookmaker_probability_std_draw",
    "bookmaker_probability_std_away",
    "bookmaker_probability_range_home",
    "bookmaker_probability_range_draw",
    "bookmaker_probability_range_away",
]


def prepare_data() -> pd.DataFrame:
    base = advanced.prepare_e0_data()
    features, _, _ = disagreement.build_features(base)
    data = features.copy()
    data = data[(data["season_end_year"].between(2021, 2025)) & (pd.to_numeric(data["ah_line"], errors="coerce") <= -1.0)].copy()
    return data.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def stable_disagreement_columns(data: pd.DataFrame) -> list[str]:
    excluded_tokens = ("1xbet", "bfe")
    columns = [
        column
        for column in disagreement.feature_columns(data)
        if not any(token in column for token in excluded_tokens)
        and (
            "source_minus_" in column
            or "max_minus_avg" in column
            or "bookmaker_probability_" in column
            or column
            in {
                "home_disagreement_index",
                "draw_disagreement_index",
                "away_disagreement_index",
                "draw_pressure_index",
                "favourite_strength_disagreement",
                "away_1x2_market_strength_minus_ah_market_strength",
                "home_1x2_market_strength_minus_ah_market_strength",
                "away_ps_prob_minus_avg_prob",
                "away_max_prob_minus_avg_prob",
            }
        )
    ]
    return [column for column in columns if data[column].notna().mean() >= 0.70]


def related_market_strength_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in RELATED_MARKET_STRENGTH_COLUMNS if column in data.columns and data[column].notna().mean() >= 0.70]


def sparse_disagreement_columns(data: pd.DataFrame) -> list[str]:
    sparse = [column for column in disagreement.feature_columns(data) if "1xbet" in column or "bfe" in column]
    indicators = []
    for column in sparse:
        indicator = f"{column}_is_missing"
        data[indicator] = data[column].isna().astype(float)
        indicators.append(indicator)
    return sparse + indicators


def feature_groups(data: pd.DataFrame) -> dict[str, tuple[list[str], list[str], str]]:
    current_numeric, current_categorical = advanced.available_feature_columns(data)
    stable = stable_disagreement_columns(data)
    related = related_market_strength_columns(data)
    sparse = sparse_disagreement_columns(data)
    market_numeric = ["away_market_probability", "away_ah_odds", "ah_line"]
    groups = {
        "market_baseline": (market_numeric, [], "baseline"),
        "baseline_current": (current_numeric, current_categorical, "primary"),
        "baseline_plus_stable_1x2_disagreement": (current_numeric + stable, current_categorical, "primary"),
        "baseline_plus_related_market_strength": (current_numeric + related, current_categorical, "primary"),
        "baseline_plus_sparse_bfe_1xbet": (current_numeric + stable + sparse, current_categorical, "diagnostic_sparse"),
        "disagreement_only": (stable, [], "diagnostic"),
    }
    for numeric, categorical, _ in groups.values():
        assert_no_closing_columns(numeric + categorical)
    return groups


def temporal_splits(data: pd.DataFrame) -> list[advanced.TemporalSplit]:
    return advanced.make_temporal_splits(sorted(data["season_end_year"].unique()))


def make_model(model_family: str):
    if model_family == "logistic_l2":
        return LogisticRegression(max_iter=1000, C=0.5, penalty="l2", solver="lbfgs", random_state=SEED)
    if model_family == "logistic_elastic_net":
        return LogisticRegression(max_iter=1500, C=0.5, penalty="elasticnet", l1_ratio=0.25, solver="saga", random_state=SEED)
    if model_family == "xgboost":
        if XGBClassifier is None:
            return None
        return XGBClassifier(
            n_estimators=80,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_lambda=4.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
        )
    raise ValueError(model_family)


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def predict_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    model_family: str,
    *,
    shuffle_labels: bool = False,
    permute_disagreement: bool = False,
    random_noise: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(SEED + int(test["season_end_year"].iloc[0]))
    train_x = train.copy()
    validation_x = validation.copy()
    test_x = test.copy()
    feature_names = numeric + categorical
    if permute_disagreement:
        disagreement_cols = [column for column in numeric if column not in advanced.NUMERIC_FEATURE_COLUMNS]
        for frame in [train_x, validation_x, test_x]:
            for _, idx in frame.groupby("season_end_year").groups.items():
                for column in disagreement_cols:
                    values = frame.loc[idx, column].to_numpy(copy=True)
                    frame.loc[idx, column] = values[rng.permutation(len(values))]
    if random_noise:
        for frame in [train_x, validation_x, test_x]:
            for column in numeric:
                frame[column] = rng.normal(0.0, 1.0, len(frame))

    preprocessor = build_preprocessor(numeric, categorical)
    preprocessor.fit(train_x[feature_names])
    x_train = preprocessor.transform(train_x[feature_names])
    x_validation = preprocessor.transform(validation_x[feature_names])
    x_test = preprocessor.transform(test_x[feature_names])
    y_train = train[TARGET].astype(int).to_numpy()
    if shuffle_labels:
        y_train = rng.permutation(y_train)
    model = make_model(model_family)
    if model is None:
        return np.array([]), np.array([]), {}, pd.DataFrame()
    if len(np.unique(y_train)) < 2:
        val_prob = np.full(len(validation), float(np.mean(y_train)) if len(y_train) else 0.5)
        test_prob = np.full(len(test), float(np.mean(y_train)) if len(y_train) else 0.5)
        return val_prob, test_prob, {}, pd.DataFrame()
    model.fit(x_train, y_train)
    val_prob = model.predict_proba(x_validation)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    importance = feature_importance(model, preprocessor, model_family)
    return val_prob, test_prob, {}, importance


def feature_importance(model, preprocessor: ColumnTransformer, model_family: str) -> pd.DataFrame:
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        names = [f"feature_{idx}" for idx in range(getattr(model, "n_features_in_", 0))]
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return pd.DataFrame()
    return pd.DataFrame({"feature": names[: len(values)], "importance": values, "model_family": model_family})


def metric_row(
    model: str,
    group: str,
    model_family: str,
    split_kind: str,
    year: int,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    train_years: tuple[int, ...] = (),
    validation_year: int | None = None,
    hyperparameters: dict | None = None,
) -> dict:
    y = frame[TARGET].astype(int).to_numpy()
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    market = np.clip(pd.to_numeric(frame["away_market_probability"], errors="coerce").fillna(0.5).to_numpy(), 1e-6, 1 - 1e-6)
    return {
        "strategy": model,
        "feature_group": group,
        "model_family": model_family,
        "kind": split_kind,
        "year": int(year),
        "test_season": int(year) if split_kind in {"test", "negative_control"} else pd.NA,
        "train_seasons": ",".join(str(int(season)) for season in train_years),
        "validation_season": int(validation_year) if validation_year is not None else pd.NA,
        "selected_model": model_family,
        "selected_hyperparameters": repr(hyperparameters or {}),
        "rows": len(frame),
        "accuracy": accuracy_score(y, p >= 0.5),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "market_accuracy": accuracy_score(y, market >= 0.5),
        "market_log_loss": log_loss(y, market, labels=[0, 1]),
        "market_brier": brier_score_loss(y, market),
        "market_ece": expected_calibration_error(y, market),
    }


def candidate_thresholds(scores: pd.Series) -> list[float]:
    clean = pd.to_numeric(scores, errors="coerce").dropna()
    if clean.empty:
        return []
    return sorted(set(float(clean.quantile(q)) for q in SCORE_QUANTILES))


def select_value_candidate(validation: pd.DataFrame, probabilities: np.ndarray) -> dict | None:
    scores = pd.Series(probabilities, index=validation.index) - pd.to_numeric(validation["away_market_probability"], errors="coerce")
    candidates = []
    for ah_threshold in THRESHOLDS:
        for score_threshold in candidate_thresholds(scores):
            selected = validation[(pd.to_numeric(validation["ah_line"], errors="coerce") <= ah_threshold) & (scores >= score_threshold)].copy()
            if len(selected) < MIN_VALIDATION_BETS:
                continue
            result = summarize(selected)
            if result["profit"] <= 0 or result["roi"] <= 0:
                continue
            candidates.append({"selected_threshold": ah_threshold, "selected_score_threshold": score_threshold, **result})
    if not candidates:
        return None
    return pd.DataFrame(candidates).sort_values(["z_score", "roi", "bets"], ascending=[False, False, False]).iloc[0].to_dict()


def selected_test_bets(test: pd.DataFrame, probabilities: np.ndarray, selection: dict, strategy: str) -> pd.DataFrame:
    scores = pd.Series(probabilities, index=test.index) - pd.to_numeric(test["away_market_probability"], errors="coerce")
    bets = test[(pd.to_numeric(test["ah_line"], errors="coerce") <= selection["selected_threshold"]) & (scores >= selection["selected_score_threshold"])].copy()
    bets["strategy"] = strategy
    bets["model_score"] = scores.loc[bets.index].to_numpy()
    bets["model_probability"] = pd.Series(probabilities, index=test.index).loc[bets.index].to_numpy()
    bets["selected_threshold"] = selection["selected_threshold"]
    bets["selected_score_threshold"] = selection["selected_score_threshold"]
    return bets


def run_models(data: pd.DataFrame, groups: dict[str, tuple[list[str], list[str], str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    bet_frames = []
    importance_frames = []
    control_rows = []

    for split in temporal_splits(data):
        train = data[data["season_end_year"].isin(split.train_years)].copy()
        validation = data[data["season_end_year"].eq(split.validation_year)].copy()
        test = data[data["season_end_year"].eq(split.test_year)].copy()
        split_context = {"train_years": split.train_years, "validation_year": split.validation_year}
        market_validation = metric_row(
            "raw_ah_market_probability",
            "market_baseline",
            "market",
            "validation",
            split.validation_year,
            validation,
            validation["away_market_probability"].to_numpy(),
            **split_context,
        )
        metric_rows.append(market_validation)
        market_test = metric_row(
            "raw_ah_market_probability",
            "market_baseline",
            "market",
            "test",
            split.test_year,
            test,
            test["away_market_probability"].to_numpy(),
            **split_context,
        )
        metric_rows.append(market_test)
        fold_baseline_validation_loss: dict[str, float] = {}

        for group_name, (numeric, categorical, group_kind) in groups.items():
            for model_name, model_family, hyperparameters in MODEL_CONFIGS:
                val_prob, test_prob, _, importance = predict_split(train, validation, test, numeric, categorical, model_family)
                if len(val_prob) == 0:
                    continue
                strategy = f"{group_name}_{model_name}"
                val_row = metric_row(
                    strategy,
                    group_name,
                    model_family,
                    "validation",
                    split.validation_year,
                    validation,
                    val_prob,
                    hyperparameters=hyperparameters,
                    **split_context,
                )
                test_row = metric_row(
                    strategy,
                    group_name,
                    model_family,
                    "test",
                    split.test_year,
                    test,
                    test_prob,
                    hyperparameters=hyperparameters,
                    **split_context,
                )
                metric_rows.extend([val_row, test_row])
                if group_name == "baseline_current":
                    fold_baseline_validation_loss[model_family] = val_row["log_loss"]
                baseline_loss = fold_baseline_validation_loss.get(model_family, np.inf)
                if len(importance):
                    importance["strategy"] = strategy
                    importance["feature_group"] = group_name
                    importance["test_year"] = split.test_year
                    importance_frames.append(importance)

                if group_name not in {"market_baseline", "baseline_current"} and val_row["log_loss"] < baseline_loss:
                    selection = select_value_candidate(validation, val_prob)
                    if selection is not None:
                        bets = selected_test_bets(test, test_prob, selection, strategy)
                        if len(bets):
                            bets["nested_test_year"] = split.test_year
                            bets["feature_group"] = group_name
                            bets["model_family"] = model_family
                            bet_frames.append(bets)

                if group_name in {"baseline_plus_stable_1x2_disagreement", "disagreement_only"}:
                    for control_name, kwargs in [
                        ("permute_disagreement_features", {"permute_disagreement": True}),
                        ("random_feature_noise", {"random_noise": True}),
                        ("shuffled_train_labels", {"shuffle_labels": True}),
                    ]:
                        _, control_prob, _, _ = predict_split(train, validation, test, numeric, categorical, model_family, **kwargs)
                        if len(control_prob):
                            control_rows.append(
                                metric_row(
                                    f"{strategy}_{control_name}",
                                    group_name,
                                    model_family,
                                    "negative_control",
                                    split.test_year,
                                    test,
                                    control_prob,
                                    hyperparameters=hyperparameters,
                                    **split_context,
                                )
                            )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame(),
        pd.concat(importance_frames, ignore_index=True, sort=False) if importance_frames else pd.DataFrame(),
        pd.DataFrame(control_rows),
    )


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = metrics[(metrics["kind"].eq("test")) & (metrics["feature_group"].eq("baseline_current"))]
    baseline_by_model = {}
    for family, frame in baseline.groupby("model_family"):
        baseline_by_model[family] = {
            metric: np.average(frame[metric], weights=frame["rows"])
            for metric in ["log_loss", "brier", "ece", "accuracy"]
        }
    for (strategy, group, family, kind), frame in metrics.groupby(["strategy", "feature_group", "model_family", "kind"]):
        row = {
            "strategy": strategy,
            "feature_group": group,
            "model_family": family,
            "kind": kind,
            "rows": int(frame["rows"].sum()),
            "accuracy": np.average(frame["accuracy"], weights=frame["rows"]),
            "log_loss": np.average(frame["log_loss"], weights=frame["rows"]),
            "brier": np.average(frame["brier"], weights=frame["rows"]),
            "ece": np.average(frame["ece"], weights=frame["rows"]),
            "market_log_loss": np.average(frame["market_log_loss"], weights=frame["rows"]),
            "market_brier": np.average(frame["market_brier"], weights=frame["rows"]),
            "market_ece": np.average(frame["market_ece"], weights=frame["rows"]),
        }
        row["delta_log_loss_vs_market"] = row["market_log_loss"] - row["log_loss"]
        row["delta_brier_vs_market"] = row["market_brier"] - row["brier"]
        if family in baseline_by_model:
            row["delta_log_loss_vs_baseline_current"] = float(baseline_by_model[family]["log_loss"] - row["log_loss"])
            row["delta_brier_vs_baseline_current"] = float(baseline_by_model[family]["brier"] - row["brier"])
        else:
            row["delta_log_loss_vs_baseline_current"] = pd.NA
            row["delta_brier_vs_baseline_current"] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def overall_bet_row(strategy: str, bets: pd.DataFrame, kind: str) -> dict:
    row = advanced.overall_row(strategy, bets, "incremental", "market_residual")
    row["kind"] = kind
    return row


def exclusion_rows(strategy: str, bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bets.empty:
        return pd.DataFrame(rows)
    by_season = bets.groupby("season_end_year")["profit"].sum()
    best = int(by_season.sort_values(ascending=False).index[0])
    for season in sorted(bets["season_end_year"].unique()):
        row = overall_bet_row(strategy, bets[bets["season_end_year"].ne(season)].copy(), "season_exclusion")
        row["exclusion_type"] = "exclude_each_season"
        row["excluded"] = int(season)
        rows.append(row)
    row = overall_bet_row(strategy, bets[bets["season_end_year"].ne(best)].copy(), "season_exclusion")
    row["exclusion_type"] = "exclude_best_profit_season"
    row["excluded"] = best
    rows.append(row)
    counts = bets["HomeTeam"].value_counts()
    for count in [1, 2, 3]:
        teams = list(counts.head(count).index)
        row = overall_bet_row(strategy, bets[~bets["HomeTeam"].isin(teams)].copy(), "home_exclusion")
        row["exclusion_type"] = f"exclude_top{count}_home"
        row["excluded"] = ", ".join(teams)
        rows.append(row)
    return pd.DataFrame(rows)


def value_summary(bets: pd.DataFrame, controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    exclusions = []
    if len(bets):
        for strategy, group in bets.groupby("strategy"):
            row = overall_bet_row(strategy, group, "value_review")
            exc = exclusion_rows(strategy, group)
            failures = promotion_failures(row, exc, controls)
            row["classification"] = "research only" if row["profit"] > 0 else "reject"
            row["gate_failures"] = ";".join(failures)
            rows.append(row)
            if len(exc):
                exclusions.append(exc)
    return pd.DataFrame(rows), pd.concat(exclusions, ignore_index=True, sort=False) if exclusions else pd.DataFrame()


def promotion_failures(row: dict, exclusions: pd.DataFrame, controls: pd.DataFrame) -> list[str]:
    no_2025 = exclusions[(exclusions["exclusion_type"].eq("exclude_each_season")) & (exclusions["excluded"].astype(str).eq("2025"))]
    best = exclusions[exclusions["exclusion_type"].eq("exclude_best_profit_season")]
    top3 = exclusions[exclusions["exclusion_type"].eq("exclude_top3_home")]
    control_success = False
    if len(controls):
        control_success = bool((controls["log_loss"] < controls["market_log_loss"]).any())
    gates = {
        "positive_roi": row["roi"] > 0,
        "z_ge_1_5": row["z_score"] >= 1.5,
        "positive_clv": row["avg_clv_pp"] > 0,
        "clv_plus_ge_52": row["clv_positive_rate"] >= 0.52,
        "positive_roi_excluding_2025": bool(len(no_2025) and float(no_2025.iloc[0]["roi"]) > 0),
        "positive_roi_excluding_best_profit_season": bool(len(best) and float(best.iloc[0]["roi"]) > 0),
        "not_destroyed_excluding_top3_home": bool(len(top3) and float(top3.iloc[0]["roi"]) > -0.05),
        "negative_controls_fail": not control_success,
    }
    return [name for name, passed in gates.items() if not passed]


def feature_missingness(data: pd.DataFrame, groups: dict[str, tuple[list[str], list[str], str]]) -> pd.DataFrame:
    rows = []
    for group, (numeric, _, _) in groups.items():
        disagreement_cols = [column for column in numeric if column not in advanced.NUMERIC_FEATURE_COLUMNS]
        for season, frame in data.groupby("season_end_year"):
            for column in disagreement_cols:
                rows.append({"feature_group": group, "season_end_year": int(season), "feature": column, "missing_rate": float(frame[column].isna().mean())})
    return pd.DataFrame(rows)


def feature_stability(data: pd.DataFrame, groups: dict[str, tuple[list[str], list[str], str]]) -> pd.DataFrame:
    rows = []
    selected = sorted(set(column for group, (numeric, _, _) in groups.items() for column in numeric if column not in advanced.NUMERIC_FEATURE_COLUMNS))
    for season, frame in data.groupby("season_end_year"):
        for column in selected:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values):
                rows.append({"season_end_year": int(season), "feature": column, "rows": len(values), "mean": values.mean(), "std": values.std(ddof=0)})
    return pd.DataFrame(rows)


def correlation_diagnostics(data: pd.DataFrame, groups: dict[str, tuple[list[str], list[str], str]]) -> pd.DataFrame:
    rows = []
    selected = sorted(set(column for group, (numeric, _, _) in groups.items() for column in numeric if column not in advanced.NUMERIC_FEATURE_COLUMNS))
    targets = {"line_move_to_away": "closing_movement", TARGET: "ah_cover"}
    for column in selected:
        values = pd.to_numeric(data[column], errors="coerce")
        for target, target_name in targets.items():
            clean = pd.DataFrame({"feature": values, "target": pd.to_numeric(data[target], errors="coerce")}).dropna()
            if len(clean) < 30 or clean["feature"].nunique() < 2 or clean["target"].nunique() < 2:
                continue
            rows.append({"feature": column, "target": target_name, "rows": len(clean), "corr": float(clean["feature"].corr(clean["target"]))})
    return pd.DataFrame(rows)


def rule_benchmark_rows(data: pd.DataFrame) -> pd.DataFrame:
    old = memory_review.prepare_data
    try:
        memory_review.prepare_data = lambda: data.copy()
        rows = []
        for strategy in ["away_odds_ge_1_85", "away_odds_ge_1_85_plus_memory_knn_profit"]:
            _, bets, _ = memory_review.run_nested_strategy(data.copy(), strategy, memory_review.strategy_defs()[strategy])
            if len(bets):
                row = memory_review.overall_row(strategy, bets)
                row["kind"] = "rule_benchmark"
                rows.append(row)
        return pd.DataFrame(rows)
    finally:
        memory_review.prepare_data = old


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str], max_rows: int = 50) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[columns].head(max_rows).to_markdown(index=False, headers=headers, floatfmt=".4f")


def write_outputs(
    metrics: pd.DataFrame,
    bets: pd.DataFrame,
    aggregate: pd.DataFrame,
    value_rows: pd.DataFrame,
    rule_rows: pd.DataFrame,
    importances: pd.DataFrame,
    controls: pd.DataFrame,
    missingness: pd.DataFrame,
    stability: pd.DataFrame,
    correlations: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    fold_summary = metrics.copy()
    fold_summary["row_type"] = "fold_metric"
    aggregate_summary = aggregate.copy()
    aggregate_summary["row_type"] = "aggregate_metric"
    value_summary_rows = value_rows.copy()
    value_summary_rows["row_type"] = "value_review"
    rule_summary_rows = rule_rows.copy()
    rule_summary_rows["row_type"] = "rule_benchmark"
    summary = pd.concat([fold_summary, aggregate_summary, value_summary_rows, rule_summary_rows], ignore_index=True, sort=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    bets.to_csv(BETS_PATH, index=False)
    if len(importances):
        importance_summary = (
            importances.groupby(["strategy", "feature_group", "model_family", "feature"], as_index=False)["importance"].mean()
            .sort_values(["strategy", "importance"], ascending=[True, False])
        )
    else:
        importance_summary = pd.DataFrame(columns=["strategy", "feature_group", "model_family", "feature", "importance"])
    importance_summary.to_csv(IMPORTANCE_PATH, index=False)
    controls.to_csv(DETAIL_DIR / "negative_controls.csv", index=False)
    missingness.to_csv(DETAIL_DIR / "feature_missingness_by_season.csv", index=False)
    stability.to_csv(DETAIL_DIR / "feature_stability_by_season.csv", index=False)
    correlations.to_csv(DETAIL_DIR / "feature_correlation_diagnostics.csv", index=False)
    exclusions.to_csv(DETAIL_DIR / "value_exclusions.csv", index=False)

    test_rows = aggregate[aggregate["kind"].eq("test")].copy()
    fold_test_rows = metrics[metrics["kind"].eq("test")].copy()
    best_predictive = test_rows.sort_values("delta_log_loss_vs_baseline_current", ascending=False).head(15)
    incremental_test = test_rows[
        test_rows["feature_group"].isin(
            [
                "baseline_plus_stable_1x2_disagreement",
                "baseline_plus_related_market_strength",
                "baseline_plus_sparse_bfe_1xbet",
                "disagreement_only",
            ]
        )
    ].copy()
    predictive_improved = bool(
        len(incremental_test)
        and pd.to_numeric(incremental_test["delta_log_loss_vs_baseline_current"], errors="coerce").max() > 0
    )
    any_value_clears_gates = bool(len(value_rows) and value_rows["gate_failures"].fillna("").eq("").any())
    final_classification = (
        "paper challenger candidate pending locked falsification"
        if predictive_improved and any_value_clears_gates
        else ("research only" if predictive_improved and len(value_rows) else "reject")
    )
    lines = [
        "# E0 Away AH Related-Market / Bookmaker-Disagreement Incremental Review",
        "",
        "Scope: E0 Away Asian Handicap big home favourite spots, 2020-2025 availability window. This is a controlled incremental feature review, not a broad strategy search.",
        "",
        "No closing odds were used as bet-time-safe features. Closing odds only appear in CLV diagnostics inherited from the AH framework.",
        "",
        "## Predictive Metrics",
        "",
        "Fold-level rows, including train seasons, validation season, selected model, and selected hyperparameters, are written to the summary CSV.",
        "",
        markdown_table(
            test_rows,
            ["strategy", "feature_group", "model_family", "rows", "accuracy", "log_loss", "brier", "ece", "delta_log_loss_vs_market", "delta_log_loss_vs_baseline_current"],
            ["Strategy", "Group", "Model", "Rows", "Accuracy", "Log loss", "Brier", "ECE", "Delta vs market", "Delta vs current"],
            max_rows=80,
        ),
        "",
        "## Fold-Level Test Metrics",
        "",
        markdown_table(
            fold_test_rows,
            [
                "strategy",
                "feature_group",
                "test_season",
                "train_seasons",
                "validation_season",
                "selected_model",
                "selected_hyperparameters",
                "accuracy",
                "log_loss",
                "brier",
                "ece",
            ],
            ["Strategy", "Group", "Test", "Train", "Validation", "Model", "Hyperparameters", "Accuracy", "Log loss", "Brier", "ECE"],
            max_rows=80,
        ),
        "",
        "## Best Predictive Deltas",
        "",
        markdown_table(
            best_predictive,
            ["strategy", "feature_group", "model_family", "log_loss", "brier", "delta_log_loss_vs_baseline_current", "delta_brier_vs_baseline_current"],
            ["Strategy", "Group", "Model", "Log loss", "Brier", "Delta LL vs current", "Delta Brier vs current"],
            max_rows=20,
        ),
        "",
        "## Value Review",
        "",
        "Value review rows were generated only for fold-level validation cases where the model log loss improved versus the matching `baseline_current` model family. They are not promoted unless aggregate predictive and betting robustness gates also pass.",
        "",
        markdown_table(
            value_rows,
            ["strategy", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "classification", "gate_failures"],
            ["Strategy", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+", "Class", "Gate failures"],
            max_rows=80,
        ),
        "",
        "## Rule Benchmarks",
        "",
        markdown_table(
            rule_rows,
            ["strategy", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate"],
            ["Strategy", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+"],
        ),
        "",
        "## Negative Controls",
        "",
        markdown_table(
            controls,
            ["strategy", "feature_group", "model_family", "year", "rows", "log_loss", "brier", "market_log_loss"],
            ["Control", "Group", "Model", "Year", "Rows", "Log loss", "Brier", "Market LL"],
            max_rows=80,
        ),
        "",
        "## Correlation Diagnostics",
        "",
        markdown_table(
            correlations,
            ["feature", "target", "rows", "corr"],
            ["Feature", "Target", "Rows", "Correlation"],
            max_rows=80,
        ),
        "",
        "## Top Feature Importances",
        "",
        markdown_table(
            importance_summary,
            ["strategy", "feature_group", "model_family", "feature", "importance"],
            ["Strategy", "Group", "Model", "Feature", "Importance"],
            max_rows=80,
        ),
        "",
        "## Leakage Audit",
        "",
        "- Feature groups use only opening/pre-match 1X2/AH/context/memory columns.",
        "- Closing odds are excluded from feature matrices and used only for CLV diagnostics after selection.",
        "- Scalers/encoders are fitted inside train seasons only.",
        "- Validation seasons are used for threshold selection only; held-out test seasons are not optimized.",
        "- Sparse BFE/1XBet group is diagnostic due low coverage.",
        "",
        f"Final classification: **{final_classification}**",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = prepare_data()
    groups = feature_groups(data)
    metrics, bets, importances, controls = run_models(data, groups)
    aggregate = aggregate_metrics(metrics)
    value_rows, exclusions = value_summary(bets, controls)
    rule_rows = rule_benchmark_rows(data)
    missingness = feature_missingness(data, groups)
    stability = feature_stability(data, groups)
    correlations = correlation_diagnostics(data, groups)
    write_outputs(metrics, bets, aggregate, value_rows, rule_rows, importances, controls, missingness, stability, correlations, exclusions)
    print(REPORT_PATH)
    print(SUMMARY_PATH)
    print(IMPORTANCE_PATH)
    print(BETS_PATH)


if __name__ == "__main__":
    main()
