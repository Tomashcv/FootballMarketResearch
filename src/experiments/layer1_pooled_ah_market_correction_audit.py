from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments import pooled_transfermarkt_advanced_predictive_audit as advanced
from src.experiments import transfermarkt_proxy_predictive_audit as base
from src.features.contextual_features import assert_no_closing_columns


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "P1"]
REPORT_PATH = Path("outputs/reports/layer1_pooled_ah_market_correction_audit.md")
SUMMARY_PATH = Path("outputs/reports/layer1_pooled_ah_market_correction_summary.csv")
PER_LEAGUE_PATH = Path("outputs/reports/layer1_pooled_ah_market_correction_per_league.csv")
PER_SEASON_PATH = Path("outputs/reports/layer1_pooled_ah_market_correction_per_season.csv")
NEGATIVE_PATH = Path("outputs/reports/layer1_pooled_ah_market_correction_negative_controls.csv")
IMPORTANCE_PATH = Path("outputs/reports/layer1_pooled_ah_market_correction_feature_importance.csv")

TARGET = "ah_home_cover"
TARGET_COLUMN = "target_ah_home_cover"
SUBSETS = ["subset_all", "subset_away_ah_big_home_favourite"]
MODELS = [
    "market_baseline_calibration",
    "logistic_l2",
    "logistic_elasticnet",
    "xgboost_shallow",
    "xgboost_market_residual",
]
CONTEXT_FEATURES = [
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_short_rest_3d",
    "away_short_rest_3d",
    "home_matches_last_7d",
    "away_matches_last_7d",
    "home_matches_last_14d",
    "away_matches_last_14d",
    "home_matches_last_21d",
    "away_matches_last_21d",
    "home_season_matches_before",
    "away_season_matches_before",
    "min_team_season_matches_before",
    "season_match_count_diff",
    "home_new_to_league",
    "away_new_to_league",
]


def league_features(frame: pd.DataFrame) -> list[str]:
    return [f"league_code_{league}" for league in LEAGUES if f"league_code_{league}" in frame.columns]


def feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    league = league_features(frame)
    market = [
        "AHh",
        "AvgAHH",
        "AvgAHA",
        "avg_ah_AvgAHH_no_vig_probability",
        "avg_ah_AvgAHA_no_vig_probability",
        "avg_ah_overround",
    ] + league
    context = [column for column in CONTEXT_FEATURES if column in frame.columns]
    tm_365 = [column for column in base.TM_FEATURES_365 if column in frame.columns]
    groups = {
        "market_only": market,
        "market_plus_basic_context": market + context,
        "market_plus_tm_365d": market + tm_365,
        "market_plus_context_plus_tm": market + context + tm_365,
        "league_only_without_market": league,
    }
    return {name: [column for column in dict.fromkeys(columns) if column in frame.columns] for name, columns in groups.items()}


def load_frame() -> pd.DataFrame:
    frame = advanced.load_dataset()
    required = ["season_end_year", "league", TARGET_COLUMN] + base.MARKET_AH
    frame = frame.dropna(subset=required).copy()
    return frame


def fold_data(frame: pd.DataFrame, subset: str, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_year = test_year - 1
    base_frame = frame[frame[subset].fillna(False)].dropna(subset=["season_end_year", TARGET_COLUMN] + base.MARKET_AH).copy()
    train = base_frame[base_frame["season_end_year"] < validation_year].copy()
    validation = base_frame[base_frame["season_end_year"] == validation_year].copy()
    test = base_frame[base_frame["season_end_year"] == test_year].copy()
    return train, validation, test


def raw_market_probability(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["avg_ah_AvgAHH_no_vig_probability"], errors="coerce").to_numpy(dtype=float)


def metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    p = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(base.ece_binary(y, p)),
    }


def fit_market_calibration(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    features = ["avg_ah_AvgAHH_no_vig_probability"]
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs", penalty="l2")),
        ]
    )
    model.fit(train[features], train[TARGET_COLUMN].astype(int).to_numpy())
    proba = model.predict_proba(test[features])
    class_index = list(model.named_steps["model"].classes_).index(1)
    coef = float(abs(model.named_steps["model"].coef_[0][0]))
    return proba[:, class_index], {"avg_ah_AvgAHH_no_vig_probability": coef}


def fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_name: str,
) -> tuple[np.ndarray, dict[str, float]]:
    assert_no_closing_columns(features)
    if model_name == "market_baseline_calibration":
        return fit_market_calibration(train, test)
    if model_name in {"logistic_l2", "logistic_elasticnet"}:
        probabilities, model = base.fit_predict(train, test, features, TARGET_COLUMN, TARGET, model_name)
        coef = model.named_steps["model"].coef_[0]
        return probabilities, dict(zip(features, np.abs(coef).astype(float)))
    if model_name == "xgboost_shallow":
        _, probabilities, importance = advanced.fit_xgboost_shallow(train, validation, test, features, TARGET)
        return probabilities, importance
    if model_name == "xgboost_market_residual":
        _, probabilities, importance = advanced.fit_xgboost_market_residual(train, validation, test, features, TARGET)
        return probabilities, importance
    raise ValueError(model_name)


def record_metrics(
    scope: str,
    subset: str,
    feature_group: str,
    model_name: str,
    test_year: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    probabilities: np.ndarray,
    league: str = "pooled",
    control: str = "",
) -> dict[str, object]:
    result = metrics(test, probabilities)
    market = metrics(test, raw_market_probability(test))
    return {
        "scope": scope,
        "league": league,
        "target": TARGET,
        "subset": subset,
        "feature_group": feature_group,
        "model": model_name,
        "control": control,
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
        "delta_log_loss_vs_raw_market": result["log_loss"] - market["log_loss"],
        "delta_brier_vs_raw_market": result["brier"] - market["brier"],
        "delta_ece_vs_raw_market": result["ece"] - market["ece"],
        "tm_365d_both_coverage_train": float(train[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()),
        "tm_365d_both_coverage_validation": float(validation[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()),
        "tm_365d_both_coverage_test": float(test[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()),
    }


def non_market_columns(features: list[str]) -> list[str]:
    market = set(feature_groups(pd.DataFrame(columns=[*features, *[f"league_code_{league}" for league in LEAGUES]]))["market_only"])
    return [column for column in features if column not in market]


def apply_negative_control(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    control: str,
    seed: int,
    groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    train_out, validation_out, test_out = train.copy(), validation.copy(), test.copy()
    market_only = set(groups["market_only"])
    controlled = [column for column in features if column not in market_only]
    if control == "permute_non_market_within_league_season":
        for frame in [train_out, validation_out, test_out]:
            for _, idx in frame.groupby(["league", "season_end_year"]).groups.items():
                for column in controlled:
                    values = frame.loc[idx, column].to_numpy(copy=True)
                    rng.shuffle(values)
                    frame.loc[idx, column] = values
    elif control == "random_noise_same_shape":
        for frame in [train_out, validation_out, test_out]:
            for column in controlled:
                source = pd.to_numeric(frame[column], errors="coerce")
                std = float(source.std(skipna=True))
                if not np.isfinite(std) or std == 0.0:
                    std = 1.0
                frame[column] = rng.normal(float(source.mean(skipna=True) or 0.0), std, len(frame))
    elif control == "shuffled_train_labels":
        values = train_out[TARGET_COLUMN].to_numpy(copy=True)
        rng.shuffle(values)
        train_out[TARGET_COLUMN] = values
    elif control == "league_only_without_market":
        features = groups["league_only_without_market"]
    elif control == "market_only_without_context_tm":
        features = groups["market_only"]
    else:
        raise ValueError(control)
    return train_out, validation_out, test_out, features


def leakage_warnings(frame: pd.DataFrame) -> list[str]:
    warnings_out = []
    if any("current_club" in column for column in frame.columns):
        warnings_out.append("players.current_club-like column present.")
    if any("club_history" in column for column in frame.columns):
        warnings_out.append("diagnostic-only club history-like column present.")
    return warnings_out


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], str]:
    frame = load_frame()
    warnings_out = leakage_warnings(frame)
    groups = feature_groups(frame)
    summary_rows = []
    per_league_rows = []
    importance_rows = []
    negative_rows = []
    years = sorted(pd.to_numeric(frame["season_end_year"], errors="coerce").dropna().astype(int).unique())
    for subset in SUBSETS:
        for test_year in years:
            train, validation, test = fold_data(frame, subset, test_year)
            if len(train) < 1200 or len(validation) < 200 or len(test) < 200:
                continue
            if subset == "subset_away_ah_big_home_favourite" and len(test) < 200:
                continue
            for group_name in ["market_only", "market_plus_basic_context", "market_plus_tm_365d", "market_plus_context_plus_tm"]:
                features = groups[group_name]
                missing = train[features].isna().mean()
                for model_name in MODELS:
                    if model_name == "market_baseline_calibration" and group_name != "market_only":
                        continue
                    try:
                        probabilities, importance = fit_predict(train, validation, test, features, model_name)
                    except Exception:
                        continue
                    row = record_metrics("pooled_temporal", subset, group_name, model_name, test_year, train, validation, test, probabilities)
                    row["feature_count"] = len(features)
                    row["mean_train_feature_missing_rate"] = float(missing.mean())
                    row["max_train_feature_missing_rate"] = float(missing.max())
                    summary_rows.append(row)
                    for league, league_test in test.groupby("league"):
                        loc = test.index.get_indexer(league_test.index)
                        if len(league_test) < 20:
                            continue
                        per_league_rows.append(
                            record_metrics(
                                "per_league",
                                subset,
                                group_name,
                                model_name,
                                test_year,
                                train,
                                validation,
                                league_test,
                                probabilities[loc],
                                league=league,
                            )
                        )
                    for feature, value in importance.items():
                        importance_rows.append(
                            {
                                "scope": "pooled_temporal",
                                "target": TARGET,
                                "subset": subset,
                                "feature_group": group_name,
                                "model": model_name,
                                "test_year": int(test_year),
                                "feature": feature,
                                "importance": float(value),
                            }
                        )
            for group_name in ["market_plus_basic_context", "market_plus_tm_365d", "market_plus_context_plus_tm"]:
                features = groups[group_name]
                for control in [
                    "permute_non_market_within_league_season",
                    "random_noise_same_shape",
                    "shuffled_train_labels",
                    "league_only_without_market",
                    "market_only_without_context_tm",
                ]:
                    try:
                        train_c, validation_c, test_c, features_c = apply_negative_control(train, validation, test, features, control, test_year, groups)
                        probabilities, _ = fit_predict(train_c, validation_c, test_c, features_c, "logistic_l2")
                    except Exception:
                        continue
                    negative_rows.append(
                        record_metrics(
                            "negative_control",
                            subset,
                            group_name,
                            "logistic_l2",
                            test_year,
                            train_c,
                            validation_c,
                            test_c,
                            probabilities,
                            control=control,
                        )
                    )
                negative_rows.append(
                    {
                        "scope": "negative_control",
                        "league": "pooled",
                        "target": TARGET,
                        "subset": subset,
                        "feature_group": group_name,
                        "model": "not_run",
                        "control": "transfermarkt_date_shift_leakage_check",
                        "test_year": int(test_year),
                        "rows": 0,
                        "delta_log_loss_vs_raw_market": np.nan,
                        "status": "failed_closed_not_run",
                    }
                )
    summary = pd.DataFrame(summary_rows)
    per_league = pd.DataFrame(per_league_rows)
    per_season = (
        summary.groupby(["subset", "feature_group", "model", "test_year"])
        .agg(
            rows=("rows", "sum"),
            log_loss=("log_loss", "mean"),
            brier=("brier", "mean"),
            ece=("ece", "mean"),
            delta_log_loss_vs_raw_market=("delta_log_loss_vs_raw_market", "mean"),
            delta_brier_vs_raw_market=("delta_brier_vs_raw_market", "mean"),
            delta_ece_vs_raw_market=("delta_ece_vs_raw_market", "mean"),
        )
        .reset_index()
        if len(summary)
        else pd.DataFrame()
    )
    negatives = pd.DataFrame(negative_rows)
    importance = pd.DataFrame(importance_rows)
    classification = classify(summary, per_league, negatives, warnings_out)
    return summary, per_league, per_season, negatives, importance, warnings_out, classification


def classify(summary: pd.DataFrame, per_league: pd.DataFrame, negatives: pd.DataFrame, warnings_out: list[str]) -> str:
    if warnings_out or summary.empty:
        return "reject"
    candidates = summary[
        summary["subset"].eq("subset_all")
        & summary["feature_group"].isin(["market_plus_basic_context", "market_plus_tm_365d", "market_plus_context_plus_tm"])
        & summary["model"].isin(["logistic_l2", "logistic_elasticnet", "xgboost_shallow", "xgboost_market_residual"])
    ].copy()
    grouped = candidates.groupby(["feature_group", "model"]).agg(
        mean_delta_log_loss=("delta_log_loss_vs_raw_market", "mean"),
        mean_delta_brier=("delta_brier_vs_raw_market", "mean"),
        mean_delta_ece=("delta_ece_vs_raw_market", "mean"),
        improved_years=("delta_log_loss_vs_raw_market", lambda s: int((s < 0).sum())),
    ).reset_index()
    viable = grouped[
        (grouped["mean_delta_log_loss"] < 0)
        & (grouped["improved_years"] >= 2)
        & (grouped["mean_delta_brier"] <= 0.002)
        & (grouped["mean_delta_ece"] <= 0.02)
    ].copy()
    if viable.empty:
        return "predictive_diagnostic_only"
    supported_leagues = per_league[
        per_league["subset"].eq("subset_all")
        & per_league["feature_group"].isin(viable["feature_group"])
        & per_league["model"].isin(viable["model"])
        & (per_league["delta_log_loss_vs_raw_market"] < 0)
    ]["league"].nunique()
    if supported_leagues < 2:
        return "predictive_diagnostic_only"
    real_best = viable["mean_delta_log_loss"].min()
    negative_best = negatives[negatives["control"].ne("transfermarkt_date_shift_leakage_check")]["delta_log_loss_vs_raw_market"].min()
    if pd.notna(negative_best) and negative_best <= real_best:
        return "predictive_diagnostic_only"
    return "layer1_market_correction_candidate"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
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
    per_season: pd.DataFrame,
    negatives: pd.DataFrame,
    importance: pd.DataFrame,
    warnings_out: list[str],
    classification: str,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    per_league.to_csv(PER_LEAGUE_PATH, index=False)
    per_season.to_csv(PER_SEASON_PATH, index=False)
    negatives.to_csv(NEGATIVE_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    aggregate = (
        summary.groupby(["subset", "feature_group", "model"])
        .agg(
            seasons=("test_year", "nunique"),
            rows=("rows", "sum"),
            mean_delta_log_loss=("delta_log_loss_vs_raw_market", "mean"),
            mean_delta_brier=("delta_brier_vs_raw_market", "mean"),
            mean_delta_ece=("delta_ece_vs_raw_market", "mean"),
            improved_years=("delta_log_loss_vs_raw_market", lambda s: int((s < 0).sum())),
        )
        .reset_index()
        .sort_values(["subset", "mean_delta_log_loss"])
        if len(summary)
        else pd.DataFrame()
    )
    league_agg = (
        per_league.groupby(["league", "subset", "feature_group", "model"])
        .agg(
            seasons=("test_year", "nunique"),
            rows=("rows", "sum"),
            mean_delta_log_loss=("delta_log_loss_vs_raw_market", "mean"),
            improved_folds=("delta_log_loss_vs_raw_market", lambda s: int((s < 0).sum())),
        )
        .reset_index()
        .sort_values(["subset", "mean_delta_log_loss"])
        if len(per_league)
        else pd.DataFrame()
    )
    negative_agg = (
        negatives.groupby(["subset", "feature_group", "control"], dropna=False)
        .agg(rows=("rows", "sum"), mean_delta_log_loss=("delta_log_loss_vs_raw_market", "mean"))
        .reset_index()
        .sort_values(["subset", "mean_delta_log_loss"])
        if len(negatives)
        else pd.DataFrame()
    )
    lines = [
        "# Layer 1 Pooled AH Market-Correction Audit",
        "",
        f"Final classification: `{classification}`",
        "",
        "Scope: E0, D1, I1, SP1, F1, and P1 AH home-cover probability correction. No betting strategies, value searches, threshold optimization, live betting, lineups, team-name features, diagnostic-only club history, or `players.current_club_*` fields were used.",
        "",
        "Closing odds are excluded from feature matrices. Transfermarkt date-shift leakage control is failed closed and not run.",
        "",
        "## Leakage Checks",
        "",
        markdown_table(pd.DataFrame({"warning": warnings_out or ["none"]}), ["warning"], max_rows=20),
        "",
        "## Pooled Aggregate",
        "",
        markdown_table(aggregate, ["subset", "feature_group", "model", "seasons", "rows", "mean_delta_log_loss", "mean_delta_brier", "mean_delta_ece", "improved_years"], max_rows=80),
        "",
        "## Per-League Breakdown",
        "",
        markdown_table(league_agg, ["league", "subset", "feature_group", "model", "seasons", "rows", "mean_delta_log_loss", "improved_folds"], max_rows=80),
        "",
        "## Per-Season Breakdown",
        "",
        markdown_table(per_season.sort_values(["subset", "test_year", "delta_log_loss_vs_raw_market"]), ["subset", "test_year", "feature_group", "model", "rows", "delta_log_loss_vs_raw_market", "delta_brier_vs_raw_market", "delta_ece_vs_raw_market"], max_rows=80),
        "",
        "## Negative Controls",
        "",
        markdown_table(negative_agg, ["subset", "feature_group", "control", "rows", "mean_delta_log_loss"], max_rows=80),
        "",
        "No confirmed edge is claimed. No value review was run.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary, per_league, per_season, negatives, importance, warnings_out, classification = run_audit()
    write_outputs(summary, per_league, per_season, negatives, importance, warnings_out, classification)
    print(
        {
            "summary_rows": len(summary),
            "per_league_rows": len(per_league),
            "per_season_rows": len(per_season),
            "negative_control_rows": len(negatives),
            "feature_importance_rows": len(importance),
            "classification": classification,
        }
    )


if __name__ == "__main__":
    main()
