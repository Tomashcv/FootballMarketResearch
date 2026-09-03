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
from src.features.contextual_features import assert_no_closing_columns, build_contextual_features, is_closing_column


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

LAYER1 = ["E0", "D1", "I1", "SP1", "F1", "P1"]
LAYER2 = ["N1", "B1", "T1", "G1", "E1", "E2", "E3"]
LEAGUES = LAYER1 + LAYER2
ENGLISH_LOWER = {"E1", "E2", "E3"}
NON_BIG_FIVE = {"P1", "N1", "B1", "T1", "G1", "E1", "E2", "E3"}

REPORT_PATH = Path("outputs/reports/layer1_layer2_pooled_ah_market_only_correction_audit.md")
SUMMARY_PATH = Path("outputs/reports/layer1_layer2_pooled_ah_market_only_summary.csv")
PER_LEAGUE_PATH = Path("outputs/reports/layer1_layer2_pooled_ah_market_only_per_league.csv")
PER_SEASON_PATH = Path("outputs/reports/layer1_layer2_pooled_ah_market_only_per_season.csv")
PER_LEAGUE_SEASON_PATH = Path("outputs/reports/layer1_layer2_pooled_ah_market_only_per_league_season.csv")
NEGATIVE_PATH = Path("outputs/reports/layer1_layer2_pooled_ah_market_only_negative_controls.csv")
IMPORTANCE_PATH = Path("outputs/reports/layer1_layer2_pooled_ah_market_only_feature_importance.csv")
DIAGNOSTIC_SUBSETS_PATH = Path("outputs/reports/layer1_layer2_pooled_ah_market_only_diagnostic_subsets.csv")

TARGET = "ah_home_cover"
TARGET_COLUMN = "target_ah_home_cover"
SUBSETS = ["subset_all", "subset_away_ah_big_home_favourite"]
FEATURE_GROUPS = [
    "raw_market_baseline",
    "market_calibration_only",
    "market_plus_line_odds",
    "market_plus_league",
    "market_plus_basic_context",
]
MODELS_BY_GROUP = {
    "raw_market_baseline": ["raw_market_baseline"],
    "market_calibration_only": ["market_baseline_calibration"],
    "market_plus_line_odds": ["logistic_l2", "logistic_elasticnet", "xgboost_shallow", "xgboost_market_residual"],
    "market_plus_league": ["logistic_l2", "logistic_elasticnet", "xgboost_shallow", "xgboost_market_residual"],
    "market_plus_basic_context": ["logistic_l2", "logistic_elasticnet", "xgboost_shallow", "xgboost_market_residual"],
}
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


def load_frame() -> pd.DataFrame:
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
    margin = pd.to_numeric(contextual["FTHG"], errors="coerce") - pd.to_numeric(contextual["FTAG"], errors="coerce")
    line = pd.to_numeric(contextual["AHh"], errors="coerce")
    adjusted = margin + line
    contextual[TARGET_COLUMN] = np.where(adjusted > 0, 1, np.where(adjusted < 0, 0, np.nan))
    contextual["subset_all"] = True
    contextual["subset_away_ah_big_home_favourite"] = line <= -1.0
    for league in LEAGUES:
        contextual[f"league_code_{league}"] = contextual["league"].eq(league).astype(float)
    required = ["league", "season_end_year", TARGET_COLUMN, "AHh", "AvgAHH", "AvgAHA", "avg_ah_AvgAHH_no_vig_probability", "avg_ah_AvgAHA_no_vig_probability"]
    contextual = contextual.dropna(subset=required).copy()
    contextual = contextual[contextual["season_end_year"] >= 2020].copy()
    return contextual.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    league = [f"league_code_{league}" for league in LEAGUES if f"league_code_{league}" in frame.columns]
    line_odds = [
        "AHh",
        "AvgAHH",
        "AvgAHA",
        "avg_ah_AvgAHH_no_vig_probability",
        "avg_ah_AvgAHA_no_vig_probability",
    ]
    context = [column for column in CONTEXT_FEATURES if column in frame.columns]
    groups = {
        "raw_market_baseline": ["avg_ah_AvgAHH_no_vig_probability"],
        "market_calibration_only": ["avg_ah_AvgAHH_no_vig_probability"],
        "market_plus_line_odds": line_odds,
        "market_plus_league": line_odds + league,
        "market_plus_basic_context": line_odds + league + context,
        "league_only_without_market": league,
    }
    return {name: [column for column in dict.fromkeys(cols) if column in frame.columns] for name, cols in groups.items()}


def fold_data(frame: pd.DataFrame, subset: str, test_year: int, excluded_league: str | None = None, excluded_season: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_year = test_year - 1
    base_frame = frame[frame[subset].fillna(False)].copy()
    if excluded_league:
        base_frame = base_frame[~base_frame["league"].eq(excluded_league)].copy()
    if excluded_season:
        base_frame = base_frame[~base_frame["season_end_year"].eq(excluded_season)].copy()
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
    return proba[:, class_index], {"avg_ah_AvgAHH_no_vig_probability": float(abs(model.named_steps["model"].coef_[0][0]))}


def fit_predict(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str], model_name: str) -> tuple[np.ndarray, dict[str, float]]:
    assert_no_closing_columns(features)
    if model_name == "raw_market_baseline":
        return raw_market_probability(test), {"avg_ah_AvgAHH_no_vig_probability": 1.0}
    if model_name == "market_baseline_calibration":
        return fit_market_calibration(train, test)
    if model_name in {"logistic_l2", "logistic_elasticnet"}:
        probabilities, model = base.fit_predict(train, test, features, TARGET_COLUMN, TARGET, model_name)
        return probabilities, dict(zip(features, np.abs(model.named_steps["model"].coef_[0]).astype(float)))
    if model_name == "xgboost_shallow":
        _, probabilities, importance = advanced.fit_xgboost_shallow(train, validation, test, features, TARGET)
        return probabilities, importance
    if model_name == "xgboost_market_residual":
        _, probabilities, importance = advanced.fit_xgboost_market_residual(train, validation, test, features, TARGET)
        return probabilities, importance
    raise ValueError(model_name)


def record(scope: str, subset: str, feature_group: str, model: str, test_year: int, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, probabilities: np.ndarray, league: str = "pooled", control: str = "") -> dict[str, object]:
    result = metrics(test, probabilities)
    market = metrics(test, raw_market_probability(test))
    return {
        "scope": scope,
        "league": league,
        "subset": subset,
        "feature_group": feature_group,
        "model": model,
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
    }


def apply_control(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str], control: str, seed: int, groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    train_out, validation_out, test_out = train.copy(), validation.copy(), test.copy()
    market_columns = set(groups["market_plus_line_odds"])
    controlled = [column for column in features if column not in market_columns and not column.startswith("league_code_")]
    if control == "league_only_without_market":
        features = groups["league_only_without_market"]
    elif control == "market_only_without_league_context":
        features = groups["market_plus_line_odds"]
    elif control == "random_noise_same_shape":
        for frame in [train_out, validation_out, test_out]:
            for column in features:
                if column.startswith("league_code_"):
                    continue
                source = pd.to_numeric(frame[column], errors="coerce")
                std = float(source.std(skipna=True))
                if not np.isfinite(std) or std == 0.0:
                    std = 1.0
                frame[column] = rng.normal(float(source.mean(skipna=True) or 0.0), std, len(frame))
    elif control == "shuffled_train_labels":
        values = train_out[TARGET_COLUMN].to_numpy(copy=True)
        rng.shuffle(values)
        train_out[TARGET_COLUMN] = values
    elif control == "permute_non_market_context_within_league_season":
        for frame in [train_out, validation_out, test_out]:
            for _, idx in frame.groupby(["league", "season_end_year"]).groups.items():
                for column in controlled:
                    values = frame.loc[idx, column].to_numpy(copy=True)
                    rng.shuffle(values)
                    frame.loc[idx, column] = values
    else:
        raise ValueError(control)
    return train_out, validation_out, test_out, features


def leakage_warnings(frame: pd.DataFrame, groups: dict[str, list[str]]) -> list[str]:
    warnings_out = []
    all_features = [feature for cols in groups.values() for feature in cols]
    closing = [feature for feature in all_features if is_closing_column(feature)]
    if closing:
        warnings_out.append("closing columns in feature groups: " + ",".join(sorted(set(closing))))
    if any("current_club" in column for column in frame.columns):
        warnings_out.append("players.current_club-like column present")
    if any("club_history" in column for column in frame.columns):
        warnings_out.append("diagnostic-only club history-like column present")
    if any(column in all_features for column in ["HomeTeam", "AwayTeam"]):
        warnings_out.append("team-name feature present")
    return warnings_out


def run_main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], str]:
    frame = load_frame()
    groups = feature_groups(frame)
    warnings_out = leakage_warnings(frame, groups)
    years = sorted(frame["season_end_year"].dropna().astype(int).unique())
    summary_rows: list[dict[str, object]] = []
    per_league_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    for subset in SUBSETS:
        for test_year in years:
            train, validation, test = fold_data(frame, subset, test_year)
            if len(train) < 3000 or len(validation) < 500 or len(test) < 500:
                continue
            if subset == "subset_away_ah_big_home_favourite" and len(test) < 300:
                continue
            for feature_group in FEATURE_GROUPS:
                features = groups[feature_group]
                for model in MODELS_BY_GROUP[feature_group]:
                    try:
                        probabilities, importance = fit_predict(train, validation, test, features, model)
                    except Exception:
                        continue
                    row = record("pooled_temporal", subset, feature_group, model, test_year, train, validation, test, probabilities)
                    row["feature_count"] = len(features)
                    row["mean_train_feature_missing_rate"] = float(train[features].isna().mean().mean()) if features else 0.0
                    summary_rows.append(row)
                    for league, league_test in test.groupby("league"):
                        if len(league_test) < 20:
                            continue
                        loc = test.index.get_indexer(league_test.index)
                        per_league_rows.append(record("per_league", subset, feature_group, model, test_year, train, validation, league_test, probabilities[loc], league=league))
                    for feature, value in importance.items():
                        importance_rows.append({"subset": subset, "feature_group": feature_group, "model": model, "test_year": int(test_year), "feature": feature, "importance": float(value)})
            for feature_group in ["market_plus_league", "market_plus_basic_context"]:
                features = groups[feature_group]
                for control in [
                    "league_only_without_market",
                    "market_only_without_league_context",
                    "random_noise_same_shape",
                    "shuffled_train_labels",
                    "permute_non_market_context_within_league_season",
                ]:
                    try:
                        train_c, validation_c, test_c, features_c = apply_control(train, validation, test, features, control, test_year, groups)
                        probabilities, _ = fit_predict(train_c, validation_c, test_c, features_c, "logistic_l2")
                    except Exception:
                        continue
                    negative_rows.append(record("negative_control", subset, feature_group, "logistic_l2", test_year, train_c, validation_c, test_c, probabilities, control=control))
    summary = pd.DataFrame(summary_rows)
    per_league = pd.DataFrame(per_league_rows)
    negatives = pd.DataFrame(negative_rows)
    importance = pd.DataFrame(importance_rows)
    negatives = pd.concat([negatives, exclusion_controls(frame, summary, groups)], ignore_index=True, sort=False)
    classification = classify(summary, per_league, negatives, warnings_out)
    return summary, per_league, negatives, importance, warnings_out, classification


def exclusion_controls(frame: pd.DataFrame, summary: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    candidates = summary[
        summary["subset"].eq("subset_all")
        & summary["feature_group"].isin(["market_plus_line_odds", "market_plus_league", "market_plus_basic_context"])
        & summary["model"].isin(["logistic_l2", "xgboost_shallow", "xgboost_market_residual"])
    ].copy()
    if candidates.empty:
        return pd.DataFrame()
    agg = candidates.groupby(["feature_group", "model"])["delta_log_loss_vs_raw_market"].mean().sort_values()
    feature_group, model = agg.index[0]
    best_league = (
        summary[summary["subset"].eq("subset_all")]
        .merge(pd.DataFrame({"feature_group": [feature_group], "model": [model]}), on=["feature_group", "model"])
        .pipe(lambda _: pd.DataFrame())
    )
    per_league_proxy = []
    rows = []
    features = groups[feature_group]
    years = sorted(frame["season_end_year"].dropna().astype(int).unique())
    # Identify best league and season from raw per-league/per-season deltas by refitting-free summaries.
    all_train, all_validation, all_test = [], [], []
    for test_year in years:
        train, validation, test = fold_data(frame, "subset_all", test_year)
        if len(train) >= 3000 and len(validation) >= 500 and len(test) >= 500:
            try:
                probabilities, _ = fit_predict(train, validation, test, features, model)
            except Exception:
                continue
            for league, group in test.groupby("league"):
                loc = test.index.get_indexer(group.index)
                per_league_proxy.append(record("best_model_probe", "subset_all", feature_group, model, test_year, train, validation, group, probabilities[loc], league=league))
    probe = pd.DataFrame(per_league_proxy)
    best_league_name = ""
    best_season = None
    if len(probe):
        best_league_name = str(probe.groupby("league")["delta_log_loss_vs_raw_market"].mean().idxmin())
        best_season = int(probe.groupby("test_year")["delta_log_loss_vs_raw_market"].mean().idxmin())
    controls = [
        ("exclude_best_performing_league", best_league_name, None),
        ("exclude_best_performing_season", None, best_season),
        ("exclude_2021", None, 2021),
        ("exclude_2026", None, 2026),
    ]
    rng = np.random.default_rng(20260630)
    for control, excluded_league, excluded_season in controls:
        if (excluded_league is None or excluded_league == "") and excluded_season is None:
            continue
        for test_year in years:
            if excluded_season == test_year:
                continue
            train, validation, test = fold_data(frame, "subset_all", test_year, excluded_league=excluded_league, excluded_season=excluded_season)
            if len(train) < 3000 or len(validation) < 500 or len(test) < 500:
                continue
            try:
                probabilities, _ = fit_predict(train, validation, test, features, model)
            except Exception:
                continue
            rows.append(record("exclusion_control", "subset_all", feature_group, model, test_year, train, validation, test, probabilities, control=control))
    for draw in range(12):
        sampled = sorted(rng.choice(LEAGUES, size=len(LAYER1), replace=False).tolist())
        sub = frame[frame["league"].isin(sampled)].copy()
        for test_year in years:
            train, validation, test = fold_data(sub, "subset_all", test_year)
            if len(train) < 1200 or len(validation) < 250 or len(test) < 250:
                continue
            try:
                probabilities, _ = fit_predict(train, validation, test, features, model)
            except Exception:
                continue
            row = record("random_same_size_league_subset", "subset_all", feature_group, model, test_year, train, validation, test, probabilities, control="random_same_size_league_subsets")
            row["sampled_leagues"] = ",".join(sampled)
            rows.append(row)
    return pd.DataFrame(rows)


def classify(summary: pd.DataFrame, per_league: pd.DataFrame, negatives: pd.DataFrame, warnings_out: list[str]) -> str:
    if warnings_out or summary.empty:
        return "reject"
    candidates = summary[
        summary["subset"].eq("subset_all")
        & ~summary["feature_group"].isin(["raw_market_baseline"])
        & (summary["delta_log_loss_vs_raw_market"] < np.inf)
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
    if supported_leagues < 3:
        return "predictive_diagnostic_only"
    real_best = float(viable["mean_delta_log_loss"].min())
    neg_best = negatives["delta_log_loss_vs_raw_market"].min() if len(negatives) else np.nan
    if pd.notna(neg_best) and neg_best <= real_best:
        return "predictive_diagnostic_only"
    exclusion = negatives[negatives["scope"].eq("exclusion_control")]
    if len(exclusion) and exclusion.groupby("control")["delta_log_loss_vs_raw_market"].mean().min() >= 0:
        return "predictive_diagnostic_only"
    return "pooled_market_only_signal_candidate"


def aggregate(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return (
        summary.groupby(group_cols)
        .agg(
            seasons=("test_year", "nunique"),
            rows=("rows", "sum"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            mean_ece=("ece", "mean"),
            mean_delta_log_loss=("delta_log_loss_vs_raw_market", "mean"),
            mean_delta_brier=("delta_brier_vs_raw_market", "mean"),
            mean_delta_ece=("delta_ece_vs_raw_market", "mean"),
            improved_years=("delta_log_loss_vs_raw_market", lambda s: int((s < 0).sum())),
        )
        .reset_index()
    )


def diagnostic_subsets(summary: pd.DataFrame, per_league: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "layer1_only": set(LAYER1),
        "layer2_only": set(LAYER2),
        "english_lower_only": ENGLISH_LOWER,
        "non_big_five": NON_BIG_FIVE,
    }
    for name, leagues in groups.items():
        subset = per_league[per_league["league"].isin(leagues)].copy()
        if subset.empty:
            continue
        agg = aggregate(subset, ["subset", "feature_group", "model"])
        agg["diagnostic_subset"] = name
        rows.append(agg)
    big = summary[summary["subset"].eq("subset_away_ah_big_home_favourite")].copy()
    if len(big):
        agg = aggregate(big, ["subset", "feature_group", "model"])
        agg["diagnostic_subset"] = "away_ah_big_home_favourite"
        rows.append(agg)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 60) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def write_outputs(summary: pd.DataFrame, per_league: pd.DataFrame, negatives: pd.DataFrame, importance: pd.DataFrame, warnings_out: list[str], classification: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_season = aggregate(summary, ["subset", "feature_group", "model", "test_year"])
    per_league_season = aggregate(per_league, ["league", "subset", "feature_group", "model", "test_year"])
    diag = diagnostic_subsets(summary, per_league)
    summary.to_csv(SUMMARY_PATH, index=False)
    aggregate(per_league, ["league", "subset", "feature_group", "model"]).to_csv(PER_LEAGUE_PATH, index=False)
    per_season.to_csv(PER_SEASON_PATH, index=False)
    per_league_season.to_csv(PER_LEAGUE_SEASON_PATH, index=False)
    negatives.to_csv(NEGATIVE_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    diag.to_csv(DIAGNOSTIC_SUBSETS_PATH, index=False)
    pooled = aggregate(summary, ["subset", "feature_group", "model"]).sort_values(["subset", "mean_delta_log_loss"])
    negative_agg = aggregate(negatives, ["scope", "subset", "feature_group", "model", "control"]).sort_values("mean_delta_log_loss") if len(negatives) else pd.DataFrame()
    lines = [
        "# Layer 1 + Layer 2 Pooled AH Market-Only Correction Audit",
        "",
        f"Final classification: `{classification}`",
        "",
        "Scope: AH home-cover probability correction across Layer 1 and Layer 2 market-only leagues. No Transfermarkt features, player features, lineups, team names, diagnostic-only club history, `players.current_club_*`, betting strategies, value searches, threshold optimization, live betting, or closing-odds features were used.",
        "",
        "## Leakage Checks",
        "",
        markdown_table(pd.DataFrame({"warning": warnings_out or ["none"]}), ["warning"], max_rows=20),
        "",
        "## Pooled Aggregate",
        "",
        markdown_table(pooled, ["subset", "feature_group", "model", "seasons", "rows", "mean_delta_log_loss", "mean_delta_brier", "mean_delta_ece", "improved_years"], max_rows=80),
        "",
        "## Diagnostic Subsets",
        "",
        markdown_table(diag.sort_values(["diagnostic_subset", "mean_delta_log_loss"]) if len(diag) else diag, ["diagnostic_subset", "subset", "feature_group", "model", "seasons", "rows", "mean_delta_log_loss", "improved_years"], max_rows=80),
        "",
        "## Negative Controls And Exclusions",
        "",
        markdown_table(negative_agg, ["scope", "subset", "feature_group", "model", "control", "seasons", "rows", "mean_delta_log_loss", "improved_years"], max_rows=100),
        "",
        "No confirmed edge is claimed. No value review was run.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary, per_league, negatives, importance, warnings_out, classification = run_main()
    write_outputs(summary, per_league, negatives, importance, warnings_out, classification)
    print(
        {
            "summary_rows": len(summary),
            "per_league_rows": len(per_league),
            "negative_control_rows": len(negatives),
            "feature_importance_rows": len(importance),
            "classification": classification,
        }
    )


if __name__ == "__main__":
    main()
