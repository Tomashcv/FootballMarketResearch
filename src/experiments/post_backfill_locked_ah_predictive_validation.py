from __future__ import annotations

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

from src.experiments.transfermarkt_proxy_predictive_audit import ece_binary


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "E1", "E2", "E3", "SC0"]
LAYER1 = {"E0", "D1", "I1", "SP1", "F1", "P1"}
LAYER2 = {"N1", "B1", "T1", "G1", "E1", "E2", "E3"}
ENGLISH_LOWER = {"E1", "E2", "E3"}
NON_BIG_FIVE = {"P1", "N1", "B1", "T1", "G1", "E1", "E2", "E3", "SC0"}

TARGET = "target_ah_home_cover"
FEATURES = ["AHh", "AvgAHH", "AvgAHA", "avg_ah_home_no_vig_probability", "avg_ah_away_no_vig_probability"]
RAW_MARKET = "avg_ah_home_no_vig_probability"

REPORT_PATH = Path("outputs/reports/post_backfill_locked_ah_predictive_validation.md")
SUMMARY_PATH = Path("outputs/reports/post_backfill_locked_ah_predictive_summary.csv")
PER_SEASON_PATH = Path("outputs/reports/post_backfill_locked_ah_predictive_per_season.csv")
PER_LEAGUE_PATH = Path("outputs/reports/post_backfill_locked_ah_predictive_per_league.csv")
PER_LEAGUE_SEASON_PATH = Path("outputs/reports/post_backfill_locked_ah_predictive_per_league_season.csv")
NEGATIVE_PATH = Path("outputs/reports/post_backfill_locked_ah_predictive_negative_controls.csv")
ROBUSTNESS_PATH = Path("outputs/reports/post_backfill_locked_ah_predictive_robustness.csv")


def load_data() -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["league"] = league
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No processed match files found")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    for column in ["season_end_year", "FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    home_raw = 1.0 / data["AvgAHH"]
    away_raw = 1.0 / data["AvgAHA"]
    total = home_raw + away_raw
    data["avg_ah_home_no_vig_probability"] = home_raw / total
    data["avg_ah_away_no_vig_probability"] = away_raw / total
    adjusted = data["FTHG"] - data["FTAG"] + data["AHh"]
    data["ah_adjusted_margin"] = adjusted
    data[TARGET] = np.where(adjusted > 0, 1.0, np.where(adjusted < 0, 0.0, np.nan))
    data["is_push"] = adjusted.eq(0)
    required = ["league", "Date", "season_end_year", TARGET] + FEATURES
    data = data.dropna(subset=required).copy()
    data = data[data["AvgAHH"].gt(1.0) & data["AvgAHA"].gt(1.0)].copy()
    data["season_end_year"] = data["season_end_year"].astype(int)
    return data.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def xgb_params(seed: int = 42) -> dict[str, object]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 2,
        "eta": 0.03,
        "lambda": 8.0,
        "alpha": 2.0,
        "seed": seed,
        "verbosity": 0,
    }


def fit_xgboost(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int = 42) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_validation = imputer.transform(validation[features])
    x_test = imputer.transform(test[features])
    y_train = train[TARGET].astype(int).to_numpy()
    y_validation = validation[TARGET].astype(int).to_numpy()
    model = xgb.train(
        xgb_params(seed),
        xgb.DMatrix(x_train, label=y_train, feature_names=features),
        num_boost_round=250,
        evals=[(xgb.DMatrix(x_validation, label=y_validation, feature_names=features), "validation")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return np.clip(model.predict(xgb.DMatrix(x_test, feature_names=features)), 1e-6, 1 - 1e-6)


def fit_market_calibration(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs", penalty="l2")),
        ]
    )
    model.fit(train[[RAW_MARKET]], train[TARGET].astype(int))
    class_index = list(model.named_steps["model"].classes_).index(1)
    return np.clip(model.predict_proba(test[[RAW_MARKET]])[:, class_index], 1e-6, 1 - 1e-6)


def metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    y = frame[TARGET].astype(int).to_numpy()
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece_binary(y, p)),
    }


def prediction_record(
    regime: str,
    test_year: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    pred: np.ndarray,
    market_probability: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    out = test[["league", "season_end_year", TARGET, RAW_MARKET, "AHh", "AvgAHH", "AvgAHA"]].copy()
    out["regime"] = regime
    out["test_year"] = test_year
    out["train_rows"] = len(train)
    out["validation_rows"] = len(validation)
    out["model_probability"] = pred
    if market_probability is None:
        out["market_probability"] = out[RAW_MARKET].astype(float).to_numpy()
    else:
        out["market_probability"] = np.asarray(market_probability, dtype=float)
    return out


def run_regime(data: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == "A_recent_only_reproduction":
        frame = data[data["season_end_year"].between(2020, 2026)].copy()
        test_years = [2022, 2023, 2024, 2025, 2026]
    elif regime == "B_historical_training_modern_test":
        frame = data.copy()
        test_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    elif regime == "C_full_historical_temporal_diagnostic":
        frame = data.copy()
        years = sorted(frame["season_end_year"].unique())
        test_years = [year for year in years if len(frame[frame["season_end_year"].lt(year - 1)]) > 0 and len(frame[frame["season_end_year"].eq(year - 1)]) > 0]
    else:
        raise ValueError(regime)
    rows = []
    for test_year in test_years:
        validation_year = test_year - 1
        train = frame[frame["season_end_year"].lt(validation_year)].copy()
        validation = frame[frame["season_end_year"].eq(validation_year)].copy()
        test = frame[frame["season_end_year"].eq(test_year)].copy()
        if len(train) == 0 or len(validation) == 0 or len(test) == 0 or train[TARGET].nunique() < 2:
            continue
        pred = fit_xgboost(train, validation, test, FEATURES, seed=42 + int(test_year))
        rows.append(prediction_record(regime, test_year, train, validation, test, pred))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def subset_mask(frame: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "pooled":
        return pd.Series(True, index=frame.index)
    if subset == "layer1_only":
        return frame["league"].isin(LAYER1)
    if subset == "layer2_only":
        return frame["league"].isin(LAYER2)
    if subset == "sc0_only":
        return frame["league"].eq("SC0")
    if subset == "english_lower_only":
        return frame["league"].isin(ENGLISH_LOWER)
    if subset == "non_big_five_only":
        return frame["league"].isin(NON_BIG_FIVE)
    raise ValueError(subset)


def summarize_predictions(predictions: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    rows = []
    for key, group in predictions.groupby(group_cols, dropna=False):
        if len(group) == 0 or group[TARGET].nunique() < 2:
            continue
        model = metrics(group, group["model_probability"].to_numpy())
        market = metrics(group, group["market_probability"].to_numpy())
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "scope": scope,
                "rows": int(len(group)),
                "seasons": int(group["season_end_year"].nunique()),
                "leagues_included": ";".join(sorted(group["league"].unique())),
                "log_loss": model["log_loss"],
                "brier": model["brier"],
                "ece": model["ece"],
                "market_log_loss": market["log_loss"],
                "market_brier": market["brier"],
                "market_ece": market["ece"],
                "delta_log_loss_vs_raw_market": model["log_loss"] - market["log_loss"],
                "delta_brier_vs_raw_market": model["brier"] - market["brier"],
                "delta_ece_vs_raw_market": model["ece"] - market["ece"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def regime_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets = ["pooled", "layer1_only", "layer2_only", "sc0_only", "english_lower_only", "non_big_five_only"]
    for regime in sorted(predictions["regime"].unique()):
        reg = predictions[predictions["regime"].eq(regime)].copy()
        per_year = summarize_predictions(reg, ["test_year"], "year")
        improved_years = int((per_year["delta_log_loss_vs_raw_market"] < 0).sum()) if len(per_year) else 0
        for subset in subsets:
            sub = reg[subset_mask(reg, subset)].copy()
            if len(sub) == 0 or sub[TARGET].nunique() < 2:
                continue
            model = metrics(sub, sub["model_probability"].to_numpy())
            market = metrics(sub, sub["market_probability"].to_numpy())
            rows.append(
                {
                    "regime": regime,
                    "subset": subset,
                    "rows": int(len(sub)),
                    "seasons": int(sub["season_end_year"].nunique()),
                    "leagues_included": ";".join(sorted(sub["league"].unique())),
                    "mean_delta_log_loss_vs_raw_market": model["log_loss"] - market["log_loss"],
                    "mean_delta_brier_vs_raw_market": model["brier"] - market["brier"],
                    "mean_delta_ece_vs_raw_market": model["ece"] - market["ece"],
                    "improved_years": improved_years if subset == "pooled" else np.nan,
                    "model_log_loss": model["log_loss"],
                    "market_log_loss": market["log_loss"],
                    "model_brier": model["brier"],
                    "market_brier": market["brier"],
                    "model_ece": model["ece"],
                    "market_ece": market["ece"],
                }
            )
    return pd.DataFrame(rows)


def run_negative_controls(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = data.copy()
    test_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    controls = [
        "shuffled_train_labels",
        "random_noise_replacing_market_features",
        "permute_features_within_league_season",
        "market_baseline_calibration_only",
        "league_only_without_market_features",
        "opposite_label_sanity_check",
    ]
    for control in controls:
        pred_rows = []
        for test_year in test_years:
            validation_year = test_year - 1
            train = frame[frame["season_end_year"].lt(validation_year)].copy()
            validation = frame[frame["season_end_year"].eq(validation_year)].copy()
            test = frame[frame["season_end_year"].eq(test_year)].copy()
            if len(train) == 0 or len(validation) == 0 or len(test) == 0 or train[TARGET].nunique() < 2:
                continue
            original_test_market = test[RAW_MARKET].astype(float).to_numpy(copy=True)
            features = FEATURES.copy()
            rng = np.random.default_rng(1000 + test_year)
            if control == "shuffled_train_labels":
                values = train[TARGET].to_numpy(copy=True)
                rng.shuffle(values)
                train[TARGET] = values
            elif control == "random_noise_replacing_market_features":
                for current in [train, validation, test]:
                    for column in features:
                        current[column] = rng.normal(0.0, 1.0, len(current))
            elif control == "permute_features_within_league_season":
                for current in [train, validation, test]:
                    for _, idx in current.groupby(["league", "season_end_year"]).groups.items():
                        for column in features:
                            values = current.loc[idx, column].to_numpy(copy=True)
                            rng.shuffle(values)
                            current.loc[idx, column] = values
            elif control == "market_baseline_calibration_only":
                pred = fit_market_calibration(train, test)
                pred_rows.append(prediction_record("negative_control", test_year, train, validation, test, pred, original_test_market))
                continue
            elif control == "league_only_without_market_features":
                for league in LEAGUES:
                    for current in [train, validation, test]:
                        current[f"league_{league}"] = current["league"].eq(league).astype(float)
                features = [f"league_{league}" for league in LEAGUES]
            elif control == "opposite_label_sanity_check":
                train[TARGET] = 1 - train[TARGET].astype(int)
            pred = fit_xgboost(train, validation, test, features, seed=5000 + test_year)
            pred_rows.append(prediction_record("negative_control", test_year, train, validation, test, pred, original_test_market))
        if pred_rows:
            predictions = pd.concat(pred_rows, ignore_index=True)
            model = metrics(predictions, predictions["model_probability"].to_numpy())
            market = metrics(predictions, predictions["market_probability"].to_numpy())
            rows.append(
                {
                    "control": control,
                    "rows": int(len(predictions)),
                    "test_years": ";".join(map(str, sorted(predictions["test_year"].unique()))),
                    "delta_log_loss_vs_raw_market": model["log_loss"] - market["log_loss"],
                    "delta_brier_vs_raw_market": model["brier"] - market["brier"],
                    "delta_ece_vs_raw_market": model["ece"] - market["ece"],
                    "model_log_loss": model["log_loss"],
                    "market_log_loss": market["log_loss"],
                    "status": "computed",
                }
            )
    return pd.DataFrame(rows)


def run_robustness(data: pd.DataFrame, base_predictions: pd.DataFrame) -> pd.DataFrame:
    b = base_predictions[base_predictions["regime"].eq("B_historical_training_modern_test")].copy()
    per_season = summarize_predictions(b, ["test_year"], "season")
    per_league = summarize_predictions(b, ["league"], "league")
    best_season = int(per_season.sort_values("delta_log_loss_vs_raw_market").iloc[0]["test_year"]) if len(per_season) else None
    best_league = str(per_league.sort_values("delta_log_loss_vs_raw_market").iloc[0]["league"]) if len(per_league) else None
    exclusions: list[tuple[str, callable]] = [
        ("exclude_best_performing_season", lambda df: df[df["season_end_year"].ne(best_season)] if best_season is not None else df),
        ("exclude_best_performing_league", lambda df: df[~df["league"].eq(best_league)] if best_league else df),
        ("exclude_oldest_pre_2012_data", lambda df: df[df["season_end_year"].ge(2012)]),
        ("exclude_pre_2020_data", lambda df: df[df["season_end_year"].ge(2020)]),
        ("exclude_2026", lambda df: df[df["season_end_year"].ne(2026)]),
        ("exclude_SC0", lambda df: df[~df["league"].eq("SC0")]),
        ("exclude_English_lower_leagues", lambda df: df[~df["league"].isin(ENGLISH_LOWER)]),
        ("exclude_Layer_1", lambda df: df[~df["league"].isin(LAYER1)]),
        ("exclude_Layer_2", lambda df: df[~df["league"].isin(LAYER2)]),
    ]
    exclusions.extend((f"exclude_{league}", lambda df, league=league: df[~df["league"].eq(league)]) for league in LEAGUES if league != "SC0")
    rows = []
    for name, fn in exclusions:
        frame = fn(data.copy())
        test_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
        pred_rows = []
        for test_year in test_years:
            validation_year = test_year - 1
            train = frame[frame["season_end_year"].lt(validation_year)].copy()
            validation = frame[frame["season_end_year"].eq(validation_year)].copy()
            test = frame[frame["season_end_year"].eq(test_year)].copy()
            if len(train) == 0 or len(validation) == 0 or len(test) == 0 or train[TARGET].nunique() < 2:
                continue
            pred = fit_xgboost(train, validation, test, FEATURES, seed=9000 + test_year)
            pred_rows.append(prediction_record(name, test_year, train, validation, test, pred))
        if not pred_rows:
            rows.append({"robustness_check": name, "rows": 0, "status": "not_computable"})
            continue
        predictions = pd.concat(pred_rows, ignore_index=True)
        model = metrics(predictions, predictions["model_probability"].to_numpy())
        market = metrics(predictions, predictions["market_probability"].to_numpy())
        rows.append(
            {
                "robustness_check": name,
                "rows": int(len(predictions)),
                "test_years": ";".join(map(str, sorted(predictions["test_year"].unique()))),
                "leagues_included": ";".join(sorted(predictions["league"].unique())),
                "delta_log_loss_vs_raw_market": model["log_loss"] - market["log_loss"],
                "delta_brier_vs_raw_market": model["brier"] - market["brier"],
                "delta_ece_vs_raw_market": model["ece"] - market["ece"],
                "model_log_loss": model["log_loss"],
                "market_log_loss": market["log_loss"],
                "status": "computed",
            }
        )
    return pd.DataFrame(rows)


def classify(summary: pd.DataFrame, negative: pd.DataFrame, robustness: pd.DataFrame) -> str:
    pooled = summary[summary["subset"].eq("pooled")].set_index("regime")
    if "A_recent_only_reproduction" not in pooled.index or "B_historical_training_modern_test" not in pooled.index:
        return "reject"
    a = float(pooled.loc["A_recent_only_reproduction", "mean_delta_log_loss_vs_raw_market"])
    b = float(pooled.loc["B_historical_training_modern_test", "mean_delta_log_loss_vs_raw_market"])
    b_brier = float(pooled.loc["B_historical_training_modern_test", "mean_delta_brier_vs_raw_market"])
    controls_good = True
    harmful_controls = {"shuffled_train_labels", "random_noise_replacing_market_features", "opposite_label_sanity_check"}
    if len(negative):
        bad = negative[negative["control"].isin(harmful_controls) & negative["delta_log_loss_vs_raw_market"].lt(0)]
        controls_good = len(bad) == 0
    robust_good = len(robustness[robustness["status"].eq("computed") & robustness["delta_log_loss_vs_raw_market"].lt(0)]) >= max(3, int(0.4 * len(robustness[robustness["status"].eq("computed")])))
    if b < 0 and b < a and b_brier < 0 and controls_good and robust_good:
        return "locked_predictive_signal_candidate"
    if b < 0 and b < a:
        return "predictive_diagnostic_only"
    return "reject"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.5f}")
    return view.to_markdown(index=False)


def write_report(summary: pd.DataFrame, per_season: pd.DataFrame, negative: pd.DataFrame, robustness: pd.DataFrame, classification: str) -> None:
    pooled = summary[summary["subset"].eq("pooled")].copy()
    lines = [
        "# Post-Backfill Locked AH Predictive Validation",
        "",
        f"Final classification: `{classification}`",
        "",
        "Scope: frozen market-only AH home-cover prediction using `AHh`, `AvgAHH`, `AvgAHA`, and no-vig AH home/away probabilities. Pushes are excluded. No betting strategies, value searches, threshold optimization, Transfermarkt features, player features, lineups, team-name features, closing odds as features, live betting, or confirmed edge claims were used.",
        "",
        "Model: `xgboost_shallow`, `max_depth=2`, `eta=0.03`, `lambda=8`, `alpha=2`, 250 rounds, early stopping on validation only. No hyperparameter search.",
        "",
        "## Regime Summary",
        "",
        markdown_table(pooled, ["regime", "rows", "seasons", "mean_delta_log_loss_vs_raw_market", "mean_delta_brier_vs_raw_market", "mean_delta_ece_vs_raw_market", "improved_years", "model_log_loss", "market_log_loss"], max_rows=20),
        "",
        "Key comparison: Regime B is the relevant post-backfill test. It must improve against Regime A and the raw market baseline before any later value-review rerun is justified.",
        "",
        "## Diagnostic Subsets",
        "",
        markdown_table(summary, ["regime", "subset", "rows", "mean_delta_log_loss_vs_raw_market", "mean_delta_brier_vs_raw_market", "mean_delta_ece_vs_raw_market"], max_rows=120),
        "",
        "## Per-Season Results",
        "",
        markdown_table(per_season, ["regime", "test_year", "rows", "delta_log_loss_vs_raw_market", "delta_brier_vs_raw_market", "delta_ece_vs_raw_market"], max_rows=120),
        "",
        "## Negative Controls",
        "",
        markdown_table(negative, ["control", "rows", "delta_log_loss_vs_raw_market", "delta_brier_vs_raw_market", "delta_ece_vs_raw_market", "status"], max_rows=40),
        "",
        "## Robustness",
        "",
        markdown_table(robustness, ["robustness_check", "rows", "delta_log_loss_vs_raw_market", "delta_brier_vs_raw_market", "delta_ece_vs_raw_market", "status"], max_rows=80),
        "",
        "No value review was run. No confirmed edge is claimed.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = load_data()
    predictions = pd.concat(
        [
            run_regime(data, "A_recent_only_reproduction"),
            run_regime(data, "B_historical_training_modern_test"),
            run_regime(data, "C_full_historical_temporal_diagnostic"),
        ],
        ignore_index=True,
    )
    summary = regime_summary(predictions)
    per_season = summarize_predictions(predictions, ["regime", "test_year"], "per_season")
    per_league = summarize_predictions(predictions, ["regime", "league"], "per_league")
    per_league_season = summarize_predictions(predictions, ["regime", "league", "test_year"], "per_league_season")
    negative = run_negative_controls(data)
    robustness = run_robustness(data, predictions)
    classification = classify(summary, negative, robustness)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    per_season.to_csv(PER_SEASON_PATH, index=False)
    per_league.to_csv(PER_LEAGUE_PATH, index=False)
    per_league_season.to_csv(PER_LEAGUE_SEASON_PATH, index=False)
    negative.to_csv(NEGATIVE_PATH, index=False)
    robustness.to_csv(ROBUSTNESS_PATH, index=False)
    write_report(summary, per_season, negative, robustness, classification)
    print(
        {
            "usable_rows": int(len(data)),
            "prediction_rows": int(len(predictions)),
            "summary_rows": int(len(summary)),
            "per_season_rows": int(len(per_season)),
            "per_league_rows": int(len(per_league)),
            "per_league_season_rows": int(len(per_league_season)),
            "negative_controls": int(len(negative)),
            "robustness_rows": int(len(robustness)),
            "classification": classification,
        }
    )


if __name__ == "__main__":
    main()
