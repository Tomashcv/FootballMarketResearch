from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.features.contextual_features import assert_no_closing_columns
from src.features.contextual_features import build_contextual_features

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


LEAGUES = ["E0", "P1", "I1", "SP1", "D1", "F1"]
PRIMARY_LEAGUES = {"E0", "P1"}
PROXY_PATH = Path("data/processed/players/transfermarkt_valuation_only_club_strength_proxy.csv")
REPORT_PATH = Path("outputs/reports/transfermarkt_proxy_predictive_audit.md")
SUMMARY_PATH = Path("outputs/reports/transfermarkt_proxy_predictive_summary.csv")
IMPORTANCE_PATH = Path("outputs/reports/transfermarkt_proxy_predictive_feature_importance.csv")
NEGATIVE_PATH = Path("outputs/reports/transfermarkt_proxy_predictive_negative_controls.csv")

TM_FEATURES_365 = [
    "home_tm_value_total_365d",
    "away_tm_value_total_365d",
    "home_minus_away_tm_value_total_365d",
    "home_tm_value_top11_365d",
    "away_tm_value_top11_365d",
    "home_minus_away_tm_value_top11_365d",
    "home_tm_value_top5_365d",
    "away_tm_value_top5_365d",
    "home_minus_away_tm_value_top5_365d",
    "home_tm_value_median_365d",
    "away_tm_value_median_365d",
    "home_minus_away_tm_value_median_365d",
    "home_tm_players_count_365d",
    "away_tm_players_count_365d",
    "home_minus_away_tm_players_count_365d",
]
TM_FEATURES_ALL = [column for column in TM_FEATURES_365]
for days in [180]:
    for metric in ["value_total", "value_top11", "value_top5", "value_median", "players_count"]:
        TM_FEATURES_ALL.extend(
            [
                f"home_tm_{metric}_{days}d",
                f"away_tm_{metric}_{days}d",
                f"home_minus_away_tm_{metric}_{days}d",
            ]
        )

MARKET_1X2 = [
    "avg_1x2_AvgH_no_vig_probability",
    "avg_1x2_AvgD_no_vig_probability",
    "avg_1x2_AvgA_no_vig_probability",
    "avg_1x2_overround",
]
MARKET_AH = [
    "AHh",
    "avg_ah_AvgAHH_no_vig_probability",
    "avg_ah_AvgAHA_no_vig_probability",
    "avg_ah_overround",
]
BASELINE_CURRENT_CANDIDATES = [
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
    "home_internal_elo_pre",
    "away_internal_elo_pre",
    "internal_elo_diff_home_minus_away",
    "internal_elo_home_win_prob",
    "internal_elo_away_win_prob",
    "market_home_prob_minus_internal_elo_prob",
    "market_away_prob_minus_internal_elo_prob",
]


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
    keep_proxy = ["league", "Date", "HomeTeam", "AwayTeam"] + [c for c in proxy.columns if "_tm_" in c and "mapped_club_name" not in c]
    output = contextual.merge(proxy[keep_proxy], on=["league", "Date", "HomeTeam", "AwayTeam"], how="left", validate="one_to_one")
    return output.sort_values(["league", "Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["target_1x2"] = output["FTR"].map({"H": 0, "D": 1, "A": 2})
    margin = pd.to_numeric(output["FTHG"], errors="coerce") - pd.to_numeric(output["FTAG"], errors="coerce")
    home_handicap = pd.to_numeric(output["AHh"], errors="coerce")
    adjusted = margin + home_handicap
    output["target_ah_home_cover"] = np.where(adjusted > 0, 1, np.where(adjusted < 0, 0, np.nan))
    output["subset_all"] = True
    output["subset_away_ah_big_home_favourite"] = home_handicap <= -1.0
    return output


def brier_multiclass(y: np.ndarray, probabilities: np.ndarray, labels: list[int]) -> float:
    return float(np.mean([brier_score_loss((y == label).astype(int), probabilities[:, idx]) for idx, label in enumerate(labels)]))


def ece_binary(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (probabilities >= left) & (probabilities <= right if idx == bins - 1 else probabilities < right)
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(probabilities[mask].mean()))
    return total


def ece_multiclass(y: np.ndarray, probabilities: np.ndarray, labels: list[int], bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = np.array(labels)[probabilities.argmax(axis=1)]
    correct = (predicted == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (confidence >= left) & (confidence <= right if idx == bins - 1 else confidence < right)
        if mask.any():
            total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return total


def market_probabilities(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target == "ah_home_cover":
        return pd.to_numeric(frame["avg_ah_AvgAHH_no_vig_probability"], errors="coerce").to_numpy()
    return frame[
        [
            "avg_1x2_AvgH_no_vig_probability",
            "avg_1x2_AvgD_no_vig_probability",
            "avg_1x2_AvgA_no_vig_probability",
        ]
    ].apply(pd.to_numeric, errors="coerce").to_numpy()


def metrics(y: np.ndarray, probabilities: np.ndarray, target: str) -> dict:
    if target == "ah_home_cover":
        p = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
        return {
            "accuracy": float(accuracy_score(y, p >= 0.5)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "ece": float(ece_binary(y, p)),
        }
    labels = [0, 1, 2]
    p = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    p = p / p.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(accuracy_score(y, np.array(labels)[p.argmax(axis=1)])),
        "log_loss": float(log_loss(y, p, labels=labels)),
        "brier": brier_multiclass(y, p, labels),
        "ece": float(ece_multiclass(y, p, labels)),
    }


def feature_groups(frame: pd.DataFrame, target: str) -> dict[str, list[str]]:
    market = MARKET_AH if target == "ah_home_cover" else MARKET_1X2
    baseline_current = [column for column in BASELINE_CURRENT_CANDIDATES if column in frame.columns]
    groups = {
        "market_baseline": market,
        "tm_proxy_only": [column for column in TM_FEATURES_ALL if column in frame.columns],
        "market_plus_tm_365d": market + [column for column in TM_FEATURES_365 if column in frame.columns],
        "market_plus_tm_180d_365d": market + [column for column in TM_FEATURES_ALL if column in frame.columns],
        "baseline_current": market + baseline_current,
        "baseline_current_plus_tm": market + baseline_current + [column for column in TM_FEATURES_ALL if column in frame.columns],
    }
    return {name: list(dict.fromkeys(cols)) for name, cols in groups.items()}


def make_model(target: str, model_name: str) -> Pipeline:
    penalty = "elasticnet" if model_name == "logistic_elasticnet" else "l2"
    solver = "saga" if penalty == "elasticnet" else "lbfgs"
    kwargs = {"max_iter": 1000, "random_state": 42, "solver": solver, "penalty": penalty}
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = 0.2
    model = LogisticRegression(**kwargs)
    return Pipeline(
        [
            ("prep", ColumnTransformer([("num", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), [])])),
            ("model", model),
        ]
    )


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target_column: str, target: str, model_name: str) -> tuple[np.ndarray, Pipeline]:
    assert_no_closing_columns(features)
    model = make_model(target, model_name)
    model.named_steps["prep"].transformers = [("num", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), features)]
    y_train = train[target_column].astype(int).to_numpy()
    model.fit(train[features], y_train)
    proba = model.predict_proba(test[features])
    if target == "ah_home_cover":
        class_index = list(model.named_steps["model"].classes_).index(1)
        return proba[:, class_index], model
    classes = list(model.named_steps["model"].classes_)
    aligned = np.zeros((len(test), 3))
    for idx, label in enumerate([0, 1, 2]):
        if label in classes:
            aligned[:, idx] = proba[:, classes.index(label)]
    aligned = aligned / aligned.sum(axis=1, keepdims=True)
    return aligned, model


def prepare_fold_data(frame: pd.DataFrame, league: str, target: str, subset_column: str, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_column = "target_ah_home_cover" if target == "ah_home_cover" else "target_1x2"
    validation_year = test_year - 1
    base = frame[(frame["league"].eq(league)) & frame[subset_column].fillna(False)].copy()
    required = ["season_end_year", target_column]
    market_cols = MARKET_AH if target == "ah_home_cover" else MARKET_1X2
    base = base.dropna(subset=required + market_cols)
    train = base[base["season_end_year"] < validation_year].copy()
    validation = base[base["season_end_year"] == validation_year].copy()
    test = base[base["season_end_year"] == test_year].copy()
    return train, validation, test


def apply_negative_control(train: pd.DataFrame, test: pd.DataFrame, features: list[str], control: str, seed: int, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_out = train.copy()
    test_out = test.copy()
    tm_cols = [column for column in features if "_tm_" in column]
    if control == "permute_tm_within_season":
        for frame in [train_out, test_out]:
            for _, idx in frame.groupby("season_end_year").groups.items():
                for column in tm_cols:
                    values = frame.loc[idx, column].to_numpy(copy=True)
                    rng.shuffle(values)
                    frame.loc[idx, column] = values
    elif control == "random_noise_same_shape":
        for frame in [train_out, test_out]:
            for column in tm_cols:
                source = pd.to_numeric(frame[column], errors="coerce")
                frame[column] = rng.normal(float(source.mean(skipna=True) or 0.0), float(source.std(skipna=True) or 1.0), len(frame))
    elif control == "shuffled_train_labels":
        values = train_out[target_column].to_numpy(copy=True)
        rng.shuffle(values)
        train_out[target_column] = values
    return train_out, test_out


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    frame = add_targets(load_dataset())
    rows = []
    importance_rows = []
    negative_rows = []
    for league in LEAGUES:
        for target in ["ah_home_cover", "outcome_1x2"]:
            target_column = "target_ah_home_cover" if target == "ah_home_cover" else "target_1x2"
            models = ["logistic_l2", "logistic_elasticnet"] if target == "ah_home_cover" else ["logistic_l2"]
            subsets = ["subset_all"] + (["subset_away_ah_big_home_favourite"] if target == "ah_home_cover" else [])
            years = sorted(pd.to_numeric(frame.loc[frame["league"].eq(league), "season_end_year"], errors="coerce").dropna().astype(int).unique())
            for subset in subsets:
                for test_year in years:
                    train, validation, test = prepare_fold_data(frame, league, target, subset, test_year)
                    if len(train) < 300 or len(validation) < 50 or len(test) < 50:
                        continue
                    groups = feature_groups(frame, target)
                    market_p = market_probabilities(test, target)
                    y_test = test[target_column].astype(int).to_numpy()
                    market_metrics = metrics(y_test, market_p, target)
                    fold_results = {}
                    for group_name, features in groups.items():
                        available = [column for column in features if column in train.columns]
                        if not available:
                            continue
                        fold_missing = train[available].isna().mean()
                        for model_name in models:
                            if model_name == "logistic_elasticnet" and target != "ah_home_cover":
                                continue
                            try:
                                probabilities, model = fit_predict(train, test, available, target_column, target, model_name)
                            except Exception as exc:
                                continue
                            result = metrics(y_test, probabilities, target)
                            record = {
                                "league": league,
                                "primary_league": league in PRIMARY_LEAGUES,
                                "target": target,
                                "subset": subset,
                                "feature_group": group_name,
                                "model": model_name,
                                "test_year": test_year,
                                "validation_year": test_year - 1,
                                "train_rows": len(train),
                                "validation_rows": len(validation),
                                "rows": len(test),
                                "feature_count": len(available),
                                "mean_train_feature_missing_rate": float(fold_missing.mean()),
                                "max_train_feature_missing_rate": float(fold_missing.max()),
                                "tm_365d_both_coverage_train": float(
                                    train[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()
                                ),
                                "tm_365d_both_coverage_validation": float(
                                    validation[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()
                                ),
                                "tm_365d_both_coverage_test": float(
                                    test[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()
                                ),
                                "accuracy": result["accuracy"],
                                "log_loss": result["log_loss"],
                                "brier": result["brier"],
                                "ece": result["ece"],
                                "market_log_loss": market_metrics["log_loss"],
                                "market_brier": market_metrics["brier"],
                                "market_ece": market_metrics["ece"],
                                "delta_log_loss_vs_market_baseline": result["log_loss"] - market_metrics["log_loss"],
                                "delta_brier_vs_market_baseline": result["brier"] - market_metrics["brier"],
                                "delta_ece_vs_market_baseline": result["ece"] - market_metrics["ece"],
                            }
                            fold_results[(group_name, model_name)] = record
                            rows.append(record)
                            coefs = model.named_steps["model"].coef_
                            coef_vector = np.mean(np.abs(coefs), axis=0) if coefs.ndim == 2 else np.abs(coefs)
                            for feature, coefficient in zip(available, coef_vector):
                                importance_rows.append(
                                    {
                                        "league": league,
                                        "target": target,
                                        "subset": subset,
                                        "feature_group": group_name,
                                        "model": model_name,
                                        "test_year": test_year,
                                        "feature": feature,
                                        "importance": float(coefficient),
                                    }
                                )
                    for key, record in list(fold_results.items()):
                        if key[0] == "baseline_current" and key[1] in {k[1] for k in fold_results if k[0] == "baseline_current"}:
                            baseline_current = record
                            for row in rows[-len(fold_results) :]:
                                if row["league"] == league and row["target"] == target and row["subset"] == subset and row["test_year"] == test_year and row["model"] == baseline_current["model"]:
                                    row["delta_log_loss_vs_baseline_current"] = row["log_loss"] - baseline_current["log_loss"]
                                    row["delta_brier_vs_baseline_current"] = row["brier"] - baseline_current["brier"]
                                    row["delta_ece_vs_baseline_current"] = row["ece"] - baseline_current["ece"]
                    for group_name in ["market_plus_tm_365d", "market_plus_tm_180d_365d", "baseline_current_plus_tm"]:
                        features = groups.get(group_name, [])
                        if not any("_tm_" in column for column in features):
                            continue
                        for control in ["permute_tm_within_season", "random_noise_same_shape", "shuffled_train_labels", "market_baseline_without_tm"]:
                            model_name = "logistic_l2"
                            if control == "market_baseline_without_tm":
                                control_features = groups["market_baseline"]
                                train_c, test_c = train, test
                            else:
                                control_features = features
                                train_c, test_c = apply_negative_control(train, test, control_features, control, test_year, target_column)
                            try:
                                probabilities, _ = fit_predict(train_c, test_c, control_features, target_column, target, model_name)
                            except Exception:
                                continue
                            result = metrics(y_test, probabilities, target)
                            negative_rows.append(
                                {
                                    "league": league,
                                    "target": target,
                                    "subset": subset,
                                    "feature_group": group_name,
                                    "control": control,
                                    "model": model_name,
                                    "test_year": test_year,
                                    "rows": len(test),
                                    "log_loss": result["log_loss"],
                                    "brier": result["brier"],
                                    "ece": result["ece"],
                                    "delta_log_loss_vs_market_baseline": result["log_loss"] - market_metrics["log_loss"],
                                }
                            )
    summary = pd.DataFrame(rows)
    if len(summary):
        for metric in ["delta_log_loss_vs_baseline_current", "delta_brier_vs_baseline_current", "delta_ece_vs_baseline_current"]:
            if metric not in summary.columns:
                summary[metric] = np.nan
    return summary, pd.DataFrame(importance_rows), pd.DataFrame(negative_rows), classify(summary, pd.DataFrame(negative_rows))


def classify(summary: pd.DataFrame, negatives: pd.DataFrame) -> str:
    if summary.empty:
        return "reject"
    primary = summary[
        summary["league"].isin(PRIMARY_LEAGUES)
        & summary["target"].eq("ah_home_cover")
        & summary["subset"].eq("subset_all")
        & summary["feature_group"].isin(["market_plus_tm_365d", "market_plus_tm_180d_365d", "baseline_current_plus_tm"])
        & summary["model"].eq("logistic_l2")
    ].copy()
    if primary.empty:
        return "reject"
    stable = primary.groupby(["league", "feature_group"])["delta_log_loss_vs_market_baseline"].agg(["count", lambda x: (x < 0).sum(), "mean"]).reset_index()
    stable.columns = ["league", "feature_group", "seasons", "improved_seasons", "mean_delta_log_loss"]
    candidates = stable[(stable["improved_seasons"] > 1) & (stable["mean_delta_log_loss"] < 0)]
    if candidates.empty:
        return "predictive_diagnostic_only"
    real_best = primary["delta_log_loss_vs_market_baseline"].mean()
    neg_best = negatives[negatives["league"].isin(PRIMARY_LEAGUES)]["delta_log_loss_vs_market_baseline"].min()
    if pd.notna(neg_best) and neg_best <= real_best:
        return "predictive_diagnostic_only"
    brier_bad = primary["delta_brier_vs_market_baseline"].mean() > 0.002
    ece_bad = primary["delta_ece_vs_market_baseline"].mean() > 0.02
    if brier_bad or ece_bad:
        return "predictive_diagnostic_only"
    return "predictive_signal_candidate"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[columns].head(max_rows).fillna("").to_markdown(index=False)


def write_outputs(summary: pd.DataFrame, importance: pd.DataFrame, negatives: pd.DataFrame, classification: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    negatives.to_csv(NEGATIVE_PATH, index=False)

    tm = summary[summary["feature_group"].str.contains("tm", na=False)].copy()
    aggregate = (
        tm.groupby(["league", "target", "subset", "feature_group", "model"])
        .agg(
            seasons=("test_year", "nunique"),
            rows=("rows", "sum"),
            mean_delta_log_loss=("delta_log_loss_vs_market_baseline", "mean"),
            mean_delta_brier=("delta_brier_vs_market_baseline", "mean"),
            mean_delta_ece=("delta_ece_vs_market_baseline", "mean"),
            improved_seasons=("delta_log_loss_vs_market_baseline", lambda s: int((s < 0).sum())),
        )
        .reset_index()
        .sort_values(["league", "target", "mean_delta_log_loss"])
        if len(tm)
        else pd.DataFrame()
    )
    lines = [
        "# Transfermarkt Proxy Predictive Audit",
        "",
        f"Final classification: **{classification}**",
        "",
        "No betting strategies, value searches, threshold optimization, closing-odds features, lineups, diagnostic-only club history, or `players.current_club_*` fields were used.",
        "Models are fixed logistic regressions under temporal train/validation/test season splits. This is a predictive diagnostic, not an edge claim.",
        "",
        "## Aggregate TM Results",
        markdown_table(
            aggregate,
            ["league", "target", "subset", "feature_group", "model", "seasons", "rows", "mean_delta_log_loss", "mean_delta_brier", "mean_delta_ece", "improved_seasons"],
        ),
        "",
        "## Season Diagnostics",
        markdown_table(
            summary.sort_values(["league", "target", "test_year", "feature_group"]),
            ["league", "target", "subset", "feature_group", "model", "test_year", "rows", "log_loss", "brier", "ece", "delta_log_loss_vs_market_baseline"],
            max_rows=60,
        ),
        "",
        "## Negative Controls",
        markdown_table(
            negatives.sort_values(["league", "target", "test_year", "feature_group", "control"]),
            ["league", "target", "subset", "feature_group", "control", "test_year", "rows", "log_loss", "delta_log_loss_vs_market_baseline"],
            max_rows=60,
        ),
        "",
        f"Summary CSV: `{SUMMARY_PATH}`",
        f"Feature importance CSV: `{IMPORTANCE_PATH}`",
        f"Negative controls CSV: `{NEGATIVE_PATH}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary, importance, negatives, classification = run_audit()
    write_outputs(summary, importance, negatives, classification)
    print(f"summary_rows: {len(summary)}")
    print(f"importance_rows: {len(importance)}")
    print(f"negative_control_rows: {len(negatives)}")
    print(f"classification: {classification}")


if __name__ == "__main__":
    main()
