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

import xgboost as xgb

from src.experiments import transfermarkt_proxy_predictive_audit as base
from src.features.contextual_features import assert_no_closing_columns, build_contextual_features, is_closing_column


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

LAYER1 = ["E0", "D1", "I1", "SP1", "F1", "P1"]
LAYER2 = ["N1", "B1", "T1", "G1", "E1", "E2", "E3"]
LEAGUES = LAYER1 + LAYER2
ENGLISH_LOWER = {"E1", "E2", "E3"}
NON_BIG_FIVE = {"P1", "N1", "B1", "T1", "G1", "E1", "E2", "E3"}
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]

TARGET_COLUMN = "target_ah_home_cover"
FEATURES = [
    "AHh",
    "AvgAHH",
    "AvgAHA",
    "avg_ah_AvgAHH_no_vig_probability",
    "avg_ah_AvgAHA_no_vig_probability",
]
RAW_MARKET_FEATURE = "avg_ah_AvgAHH_no_vig_probability"

REPORT_PATH = Path("outputs/reports/layer1_layer2_locked_ah_market_only_replication.md")
SUMMARY_PATH = Path("outputs/reports/layer1_layer2_locked_ah_market_only_replication_summary.csv")
PER_LEAGUE_PATH = Path("outputs/reports/layer1_layer2_locked_ah_market_only_per_league.csv")
PER_SEASON_PATH = Path("outputs/reports/layer1_layer2_locked_ah_market_only_per_season.csv")
PER_LEAGUE_SEASON_PATH = Path("outputs/reports/layer1_layer2_locked_ah_market_only_per_league_season.csv")
NEGATIVE_PATH = Path("outputs/reports/layer1_layer2_locked_ah_market_only_negative_controls.csv")
ROBUSTNESS_PATH = Path("outputs/reports/layer1_layer2_locked_ah_market_only_robustness_exclusions.csv")
SETTLEMENT_PATH = Path("outputs/reports/layer1_layer2_locked_ah_market_only_ah_settlement_sanity.csv")


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
    if not frames:
        raise FileNotFoundError("No processed match files found for locked replication audit")
    matches = pd.concat(frames, ignore_index=True, sort=False)
    contextual = build_contextual_features(matches)
    margin = pd.to_numeric(contextual["FTHG"], errors="coerce") - pd.to_numeric(contextual["FTAG"], errors="coerce")
    line = pd.to_numeric(contextual["AHh"], errors="coerce")
    adjusted = margin + line
    contextual["ah_adjusted_margin"] = adjusted
    contextual[TARGET_COLUMN] = np.where(adjusted > 0, 1, np.where(adjusted < 0, 0, np.nan))
    contextual["ah_push"] = adjusted.eq(0)
    contextual["subset_away_ah_big_home_favourite"] = line <= -1.0
    for league in LEAGUES:
        contextual[f"league_code_{league}"] = contextual["league"].eq(league).astype(float)
    required = ["league", "season_end_year", TARGET_COLUMN] + FEATURES
    contextual = contextual.dropna(subset=required).copy()
    contextual = contextual[contextual["season_end_year"] >= 2020].copy()
    return contextual.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def load_settlement_frame() -> pd.DataFrame:
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
    if not frames:
        raise FileNotFoundError("No processed match files found for AH settlement sanity")
    matches = pd.concat(frames, ignore_index=True, sort=False)
    contextual = build_contextual_features(matches)
    required = ["league", "season_end_year", "FTHG", "FTAG"] + FEATURES
    contextual = contextual.dropna(subset=required).copy()
    contextual = contextual[contextual["season_end_year"] >= 2020].copy()
    margin = pd.to_numeric(contextual["FTHG"], errors="coerce") - pd.to_numeric(contextual["FTAG"], errors="coerce")
    line = pd.to_numeric(contextual["AHh"], errors="coerce")
    adjusted = margin + line
    contextual["ah_adjusted_margin"] = adjusted
    contextual[TARGET_COLUMN] = np.where(adjusted > 0, 1, np.where(adjusted < 0, 0, np.nan))
    contextual["ah_push"] = adjusted.eq(0)
    return contextual.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def fold_data(
    frame: pd.DataFrame,
    test_year: int,
    excluded_leagues: set[str] | None = None,
    excluded_seasons: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    excluded_leagues = excluded_leagues or set()
    excluded_seasons = excluded_seasons or set()
    base_frame = frame[~frame["league"].isin(excluded_leagues)].copy()
    base_frame = base_frame[~base_frame["season_end_year"].isin(excluded_seasons)].copy()
    validation_year = test_year - 1
    train = base_frame[base_frame["season_end_year"] < validation_year].copy()
    validation = base_frame[base_frame["season_end_year"] == validation_year].copy()
    test = base_frame[base_frame["season_end_year"] == test_year].copy()
    return train, validation, test


def raw_market_probability(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame[RAW_MARKET_FEATURE], errors="coerce").to_numpy(dtype=float)


def metric_values(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    p = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(base.ece_binary(y, p)),
    }


def fit_xgboost_shallow(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    assert_no_closing_columns(features)
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_validation = imputer.transform(validation[features])
    x_test = imputer.transform(test[features])
    y_train = train[TARGET_COLUMN].astype(int).to_numpy()
    y_validation = validation[TARGET_COLUMN].astype(int).to_numpy()
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
    probabilities = model.predict(xgb.DMatrix(x_test, feature_names=features))
    return np.clip(probabilities, 1e-6, 1 - 1e-6), {k: float(v) for k, v in model.get_score(importance_type="gain").items()}


def fit_market_calibration(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    features = [RAW_MARKET_FEATURE]
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs", penalty="l2")),
        ]
    )
    model.fit(train[features], train[TARGET_COLUMN].astype(int).to_numpy())
    class_index = list(model.named_steps["model"].classes_).index(1)
    return model.predict_proba(test[features])[:, class_index]


def record(
    scope: str,
    test_year: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    probabilities: np.ndarray,
    league: str = "pooled",
    control: str = "",
) -> dict[str, object]:
    result = metric_values(test, probabilities)
    market = metric_values(test, raw_market_probability(test))
    return {
        "scope": scope,
        "league": league,
        "feature_group": "market_plus_line_odds",
        "model": "xgboost_shallow",
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


def apply_control(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    control: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    train_out, validation_out, test_out = train.copy(), validation.copy(), test.copy()
    features = FEATURES.copy()
    if control == "random_noise_replacing_market_features":
        for frame in [train_out, validation_out, test_out]:
            for column in features:
                source = pd.to_numeric(frame[column], errors="coerce")
                std = float(source.std(skipna=True))
                if not np.isfinite(std) or std == 0.0:
                    std = 1.0
                frame[column] = rng.normal(float(source.mean(skipna=True) or 0.0), std, len(frame))
    elif control == "permute_market_features_within_league_season":
        for frame in [train_out, validation_out, test_out]:
            for _, idx in frame.groupby(["league", "season_end_year"]).groups.items():
                for column in features:
                    values = frame.loc[idx, column].to_numpy(copy=True)
                    rng.shuffle(values)
                    frame.loc[idx, column] = values
    elif control == "shuffled_train_labels":
        values = train_out[TARGET_COLUMN].to_numpy(copy=True)
        rng.shuffle(values)
        train_out[TARGET_COLUMN] = values
    elif control == "league_only_without_market_features":
        features = [f"league_code_{league}" for league in LEAGUES if f"league_code_{league}" in train_out.columns]
    elif control == "market_baseline_calibration_only":
        features = [RAW_MARKET_FEATURE]
    elif control == "opposite_label_sanity_check":
        train_out[TARGET_COLUMN] = 1 - train_out[TARGET_COLUMN].astype(int)
        validation_out[TARGET_COLUMN] = 1 - validation_out[TARGET_COLUMN].astype(int)
    else:
        raise ValueError(control)
    return train_out, validation_out, test_out, features


def leakage_warnings(frame: pd.DataFrame) -> list[str]:
    warnings_out = []
    closing = [feature for feature in FEATURES if is_closing_column(feature)]
    if closing:
        warnings_out.append("closing columns in locked feature group: " + ",".join(closing))
    banned_features = {"HomeTeam", "AwayTeam"}
    if banned_features.intersection(FEATURES):
        warnings_out.append("team-name feature present")
    if any("_tm_" in column for column in FEATURES):
        warnings_out.append("Transfermarkt feature present")
    if any("current_club" in column for column in frame.columns):
        warnings_out.append("players.current_club-like column present in loaded frame")
    if any("club_history" in column for column in frame.columns):
        warnings_out.append("diagnostic-only club history-like column present in loaded frame")
    return warnings_out


def aggregate(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(group_cols)
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


def run_locked_replication(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    per_league_rows = []
    per_league_season_rows = []
    importance_rows = []
    for test_year in TEST_YEARS:
        train, validation, test = fold_data(frame, test_year)
        if len(train) < 3000 or len(validation) < 500 or len(test) < 500:
            continue
        probabilities, importance = fit_xgboost_shallow(train, validation, test, FEATURES)
        summary_rows.append(record("locked_replication", test_year, train, validation, test, probabilities))
        for league, league_test in test.groupby("league"):
            if len(league_test) < 20:
                continue
            loc = test.index.get_indexer(league_test.index)
            row = record("per_league_season", test_year, train, validation, league_test, probabilities[loc], league=league)
            per_league_season_rows.append(row)
            per_league_rows.append(row)
        for feature, value in importance.items():
            importance_rows.append({"test_year": int(test_year), "feature": feature, "importance": float(value)})
    return pd.DataFrame(summary_rows), pd.DataFrame(per_league_rows), pd.DataFrame(per_league_season_rows), pd.DataFrame(importance_rows)


def run_negative_controls(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    controls = [
        "shuffled_train_labels",
        "random_noise_replacing_market_features",
        "permute_market_features_within_league_season",
        "league_only_without_market_features",
        "market_baseline_calibration_only",
        "opposite_label_sanity_check",
    ]
    for test_year in TEST_YEARS:
        train, validation, test = fold_data(frame, test_year)
        if len(train) < 3000 or len(validation) < 500 or len(test) < 500:
            continue
        for control in controls:
            try:
                train_c, validation_c, test_c, features = apply_control(train, validation, test, control, 20260630 + test_year, )
                if control == "market_baseline_calibration_only":
                    probabilities = fit_market_calibration(train_c, test_c)
                else:
                    probabilities, _ = fit_xgboost_shallow(train_c, validation_c, test_c, features)
                rows.append(record("negative_control", test_year, train_c, validation_c, test_c, probabilities, control=control))
            except Exception as exc:
                rows.append(
                    {
                        "scope": "negative_control",
                        "league": "pooled",
                        "feature_group": "market_plus_line_odds",
                        "model": "xgboost_shallow",
                        "control": control,
                        "test_year": int(test_year),
                        "error": str(exc),
                    }
                )
    rng = np.random.default_rng(20260630)
    for draw in range(12):
        sampled = sorted(rng.choice(LEAGUES, size=len(LAYER1), replace=False).tolist())
        subset = frame[frame["league"].isin(sampled)].copy()
        for test_year in TEST_YEARS:
            train, validation, test = fold_data(subset, test_year)
            if len(train) < 1200 or len(validation) < 250 or len(test) < 250:
                continue
            probabilities, _ = fit_xgboost_shallow(train, validation, test, FEATURES)
            row = record("random_same_size_league_subset", test_year, train, validation, test, probabilities, control="random_same_size_league_subsets")
            row["sampled_leagues"] = ",".join(sampled)
            rows.append(row)
    return pd.DataFrame(rows)


def run_robustness_exclusions(frame: pd.DataFrame, per_league_season: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    best_league = ""
    best_season = None
    if len(per_league_season):
        best_league = str(per_league_season.groupby("league")["delta_log_loss_vs_raw_market"].mean().idxmin())
    if len(summary):
        best_season = int(summary.groupby("test_year")["delta_log_loss_vs_raw_market"].mean().idxmin())
    controls: list[tuple[str, set[str], set[int]]] = [
        ("exclude_2021_from_history", set(), {2021}),
        ("exclude_2026_from_testing", set(), set()),
        ("exclude_best_performing_league", {best_league} if best_league else set(), set()),
        ("exclude_best_performing_season", set(), {best_season} if best_season else set()),
        ("exclude_english_lower_leagues", ENGLISH_LOWER, set()),
        ("exclude_layer2", set(LAYER2), set()),
        ("exclude_layer1", set(LAYER1), set()),
    ]
    controls.extend((f"exclude_league_{league}", {league}, set()) for league in LEAGUES)
    for control, excluded_leagues, excluded_seasons in controls:
        for test_year in TEST_YEARS:
            if control == "exclude_2026_from_testing" and test_year == 2026:
                continue
            if test_year in excluded_seasons:
                continue
            train, validation, test = fold_data(frame, test_year, excluded_leagues=excluded_leagues, excluded_seasons=excluded_seasons)
            if len(train) < 1200 or len(validation) < 250 or len(test) < 250:
                continue
            try:
                probabilities, _ = fit_xgboost_shallow(train, validation, test, FEATURES)
                rows.append(record("robustness_exclusion", test_year, train, validation, test, probabilities, control=control))
            except Exception as exc:
                rows.append({"scope": "robustness_exclusion", "control": control, "test_year": int(test_year), "error": str(exc)})
    return pd.DataFrame(rows)


def settlement_sanity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source = frame.copy()
    raw_line = pd.to_numeric(source["AHh"], errors="coerce")
    adjusted = pd.to_numeric(source["ah_adjusted_margin"], errors="coerce")
    source["settlement_bucket"] = np.select(
        [
            adjusted >= 1.0,
            (adjusted > 0.0) & (adjusted < 1.0),
            adjusted == 0.0,
            (adjusted < 0.0) & (adjusted > -1.0),
            adjusted <= -1.0,
        ],
        ["full_home_cover", "partial_home_cover", "push", "partial_home_no_cover", "full_home_no_cover"],
        default="unknown",
    )
    source["half_line"] = np.isclose(np.mod(np.abs(raw_line) * 2.0, 2.0), 1.0)
    source["quarter_line"] = np.isclose(np.mod(np.abs(raw_line) * 4.0, 4.0), 1.0) | np.isclose(np.mod(np.abs(raw_line) * 4.0, 4.0), 3.0)
    for keys, group in source.groupby(["league", "season_end_year"], dropna=False):
        league, season = keys
        bucket_counts = group["settlement_bucket"].value_counts(normalize=True)
        rows.append(
            {
                "league": league,
                "season_end_year": int(season) if pd.notna(season) else np.nan,
                "rows": int(len(group)),
                "push_rate": float(group["ah_push"].mean()),
                "half_line_rate": float(group["half_line"].mean()),
                "quarter_line_rate": float(group["quarter_line"].mean()),
                "full_home_cover_rate": float(bucket_counts.get("full_home_cover", 0.0)),
                "partial_home_cover_rate": float(bucket_counts.get("partial_home_cover", 0.0)),
                "full_home_no_cover_rate": float(bucket_counts.get("full_home_no_cover", 0.0)),
                "partial_home_no_cover_rate": float(bucket_counts.get("partial_home_no_cover", 0.0)),
                "target_excludes_pushes": True,
                "target_collapses_half_outcomes": True,
                "payout_compatibility": "binary_predictive_target_only; separate AH payout settlement still required before value review",
            }
        )
    return pd.DataFrame(rows)


def diagnostic_subsets(per_league: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "layer1_only": set(LAYER1),
        "layer2_only": set(LAYER2),
        "english_lower_only": ENGLISH_LOWER,
        "non_big_five": NON_BIG_FIVE,
    }
    for name, leagues in groups.items():
        subset = per_league[per_league["league"].isin(leagues)].copy()
        agg = aggregate(subset, ["feature_group", "model"]) if len(subset) else pd.DataFrame()
        if len(agg):
            agg["diagnostic_subset"] = name
            rows.append(agg)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def classify(
    summary: pd.DataFrame,
    per_league: pd.DataFrame,
    negatives: pd.DataFrame,
    robustness: pd.DataFrame,
    settlement: pd.DataFrame,
    warnings_out: list[str],
) -> str:
    if warnings_out or summary.empty:
        return "reject"
    aggregate_summary = aggregate(summary, ["feature_group", "model"])
    if aggregate_summary.empty:
        return "predictive_diagnostic_only"
    main = aggregate_summary.iloc[0]
    improved_years = int(main["improved_years"])
    improved_leagues = per_league.groupby("league")["delta_log_loss_vs_raw_market"].mean().lt(0).sum() if len(per_league) else 0
    brier_ok = float(main["mean_delta_brier"]) < 0
    ece_ok = float(main["mean_delta_ece"]) <= 0.002
    negative_agg = aggregate(negatives[negatives["scope"].ne("random_same_size_league_subset")], ["scope", "control"])
    neg_best = float(negative_agg["mean_delta_log_loss"].min()) if len(negative_agg) else np.nan
    real_delta = float(main["mean_delta_log_loss"])
    controls_ok = pd.isna(neg_best) or neg_best > real_delta
    robust_agg = aggregate(robustness, ["control"])
    must_survive = {
        "exclude_best_performing_league",
        "exclude_best_performing_season",
        "exclude_2021_from_history",
        "exclude_2026_from_testing",
    }
    survived = set(robust_agg.loc[robust_agg["mean_delta_log_loss"] < 0, "control"]) if len(robust_agg) else set()
    robustness_ok = must_survive.issubset(survived)
    settlement_ok = False
    if len(settlement):
        settlement_ok = not bool(settlement["target_collapses_half_outcomes"].any())
    if (
        real_delta < 0
        and improved_years >= 4
        and int(improved_leagues) >= 5
        and brier_ok
        and ece_ok
        and controls_ok
        and robustness_ok
        and settlement_ok
    ):
        return "ready_for_locked_value_review"
    if real_delta < 0 and improved_years >= 4 and int(improved_leagues) >= 5 and brier_ok and controls_ok and robustness_ok:
        return "locked_predictive_signal_candidate"
    return "predictive_diagnostic_only"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 60) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def write_outputs(
    summary: pd.DataFrame,
    per_league_rows: pd.DataFrame,
    per_league_season_rows: pd.DataFrame,
    negatives: pd.DataFrame,
    robustness: pd.DataFrame,
    settlement: pd.DataFrame,
    warnings_out: list[str],
    classification: str,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_league = aggregate(per_league_rows, ["league", "feature_group", "model"])
    per_season = aggregate(summary, ["feature_group", "model", "test_year"])
    per_league_season = aggregate(per_league_season_rows, ["league", "feature_group", "model", "test_year"])
    summary.to_csv(SUMMARY_PATH, index=False)
    per_league.to_csv(PER_LEAGUE_PATH, index=False)
    per_season.to_csv(PER_SEASON_PATH, index=False)
    per_league_season.to_csv(PER_LEAGUE_SEASON_PATH, index=False)
    negatives.to_csv(NEGATIVE_PATH, index=False)
    robustness.to_csv(ROBUSTNESS_PATH, index=False)
    settlement.to_csv(SETTLEMENT_PATH, index=False)
    diag = diagnostic_subsets(per_league_rows)
    negative_agg = aggregate(negatives, ["scope", "control"]).sort_values("mean_delta_log_loss") if len(negatives) else pd.DataFrame()
    robust_agg = aggregate(robustness, ["control"]).sort_values("mean_delta_log_loss") if len(robustness) else pd.DataFrame()
    lines = [
        "# Layer 1 + Layer 2 Locked AH Market-Only Replication",
        "",
        f"Final classification: `{classification}`",
        "",
        "Frozen setup: AH home cover target, `market_plus_line_odds` only, and the prior shallow XGBoost parameters (`max_depth=2`, `eta=0.03`, `lambda=8`, `alpha=2`, 250 rounds, early stopping on validation). No Transfermarkt features, player features, lineups, team names, closing-odds features, value search, threshold optimization, betting strategy, or live betting were used.",
        "",
        "## Leakage Checks",
        "",
        markdown_table(pd.DataFrame({"warning": warnings_out or ["none"]}), ["warning"]),
        "",
        "## Locked Pooled Result",
        "",
        markdown_table(aggregate(summary, ["feature_group", "model"]), ["feature_group", "model", "seasons", "rows", "mean_delta_log_loss", "mean_delta_brier", "mean_delta_ece", "improved_years"]),
        "",
        "## Per Season",
        "",
        markdown_table(per_season, ["test_year", "rows", "mean_delta_log_loss", "mean_delta_brier", "mean_delta_ece"]),
        "",
        "## Per League",
        "",
        markdown_table(per_league.sort_values("mean_delta_log_loss"), ["league", "seasons", "rows", "mean_delta_log_loss", "mean_delta_brier", "mean_delta_ece", "improved_years"], max_rows=30),
        "",
        "## Diagnostic Subsets",
        "",
        markdown_table(diag.sort_values("mean_delta_log_loss") if len(diag) else diag, ["diagnostic_subset", "seasons", "rows", "mean_delta_log_loss", "mean_delta_brier", "mean_delta_ece", "improved_years"], max_rows=20),
        "",
        "## Negative Controls",
        "",
        markdown_table(negative_agg, ["scope", "control", "seasons", "rows", "mean_delta_log_loss", "mean_delta_brier", "mean_delta_ece", "improved_years"], max_rows=50),
        "",
        "## Robustness Exclusions",
        "",
        markdown_table(robust_agg, ["control", "seasons", "rows", "mean_delta_log_loss", "mean_delta_brier", "mean_delta_ece", "improved_years"], max_rows=80),
        "",
        "## AH Settlement Sanity",
        "",
        "The predictive target excludes exact pushes and collapses partial AH outcomes into a binary cover/no-cover label. That is compatible with probability diagnostics, but separate AH payout settlement must be verified before any value review.",
        "",
        markdown_table(settlement.groupby("league").agg(rows=("rows", "sum"), push_rate=("push_rate", "mean"), half_line_rate=("half_line_rate", "mean"), quarter_line_rate=("quarter_line_rate", "mean"), partial_home_cover_rate=("partial_home_cover_rate", "mean"), partial_home_no_cover_rate=("partial_home_no_cover_rate", "mean")).reset_index(), ["league", "rows", "push_rate", "half_line_rate", "quarter_line_rate", "partial_home_cover_rate", "partial_home_no_cover_rate"], max_rows=30),
        "",
        "No confirmed edge is claimed. No value review was run.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame = load_frame()
    settlement_frame = load_settlement_frame()
    warnings_out = leakage_warnings(frame)
    summary, per_league_rows, per_league_season_rows, importance = run_locked_replication(frame)
    del importance
    negatives = run_negative_controls(frame)
    robustness = run_robustness_exclusions(frame, per_league_season_rows, summary)
    settlement = settlement_sanity(settlement_frame)
    classification = classify(summary, per_league_rows, negatives, robustness, settlement, warnings_out)
    write_outputs(summary, per_league_rows, per_league_season_rows, negatives, robustness, settlement, warnings_out, classification)
    print(
        {
            "summary_rows": len(summary),
            "per_league_rows": len(per_league_rows),
            "negative_rows": len(negatives),
            "robustness_rows": len(robustness),
            "settlement_rows": len(settlement),
            "classification": classification,
        }
    )


if __name__ == "__main__":
    main()
