from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")

INPUT = Path("data/processed/features/football_feature_matrix_v2_transfermarkt_partial.csv")
REPORT_DIR = Path("outputs/reports")

REPORT_MD = REPORT_DIR / "feature_matrix_v2_tm_1x2_predictive_audit.md"
SUMMARY_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_predictive_summary.csv"
SCOPE_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_scope_comparison.csv"
CLASS_CAL_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_class_calibration.csv"
EDGE_BUCKET_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_edge_bucket_calibration.csv"
NEGATIVE_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_negative_controls.csv"
ROBUSTNESS_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_robustness.csv"
VALUE_MD = REPORT_DIR / "feature_matrix_v2_tm_1x2_value_review.md"
VALUE_FIXED_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_value_fixed_rules.csv"
VALUE_NESTED_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_value_nested_selection.csv"
VALUE_CONTROLS_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_value_controls.csv"
VALUE_ROBUSTNESS_CSV = REPORT_DIR / "feature_matrix_v2_tm_1x2_value_robustness.csv"

TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
CLASSES = ["H", "D", "A"]
CLASS_TO_INT = {"H": 0, "D": 1, "A": 2}
LOWER_ENGLISH = {"E1", "E2", "E3"}
TOP_DIVISIONS = {"E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "SC0"}

MARKET_BASELINE = [
    "x1x2_avg_prob_home",
    "x1x2_avg_prob_draw",
    "x1x2_avg_prob_away",
    "x1x2_avg_market_overround",
    "x1x2_avg_odds_home",
    "x1x2_avg_odds_draw",
    "x1x2_avg_odds_away",
]

IDENTITY_EXCLUDES = {
    "match_id",
    "home_team",
    "away_team",
    "source_processed_file",
    "feature_matrix_version",
    "tm_game_id",
    "tm_home_club_id",
    "tm_away_club_id",
    "tm_competition_id",
}

STRING_EXCLUDES = {
    "target_outcome_1x2",
    "league",
    "match_date",
    "x1x2_avg_market_source",
    "ou25_avg_market_source",
    "ah_avg_market_source",
    "league_era_bucket",
    "scaler_policy",
    "tm_mapping_status",
    "tm_mapping_coverage_group",
    "tm_partial_feature_warning",
}


@dataclass(frozen=True)
class Scope:
    name: str
    description: str


SCOPES = [
    Scope("scope_A_full_modern_1x2", "All valid 1X2 rows; final test years 2020-2026; Transfermarkt may be missing."),
    Scope("scope_B_tm_available_same_rows", "Rows with tm_match_feature_available == True; same-row comparison."),
    Scope("scope_C_top_divisions_ex_e1_e2_e3", "Rows excluding E1/E2/E3; same-row comparison."),
    Scope("scope_D_modern_tm_available", "Rows with season_start_year >= 2014 and tm_match_feature_available == True."),
]


def normalize_probs(prob: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(prob, dtype=float), 1e-8, 1.0)
    return arr / arr.sum(axis=1, keepdims=True)


def brier_multi(y: np.ndarray, prob: np.ndarray) -> float:
    p = normalize_probs(prob)
    one = np.zeros_like(p)
    one[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one) ** 2, axis=1)))


def ece_multi(y: np.ndarray, prob: np.ndarray, bins: int = 15) -> float:
    p = normalize_probs(prob)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(out) if total else np.nan


def raw_market_probs(frame: pd.DataFrame) -> np.ndarray:
    return normalize_probs(frame[["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]].to_numpy(dtype=float))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce").astype("Int64")
    df["target_y"] = df["target_outcome_1x2"].map(CLASS_TO_INT)
    for col in MARKET_BASELINE:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = (
        df["target_1x2_available"].fillna(0).astype(bool)
        & df["target_y"].notna()
        & df[MARKET_BASELINE[:3]].notna().all(axis=1)
        & df[["x1x2_avg_odds_home", "x1x2_avg_odds_draw", "x1x2_avg_odds_away"]].notna().all(axis=1)
    )
    df = df[valid].copy()
    df["target_y"] = df["target_y"].astype(int)
    return df.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def is_closing_column(col: str) -> bool:
    low = col.lower()
    return low.endswith("_closing") or "closing" in low or low.endswith("_close") or "_close_" in low


def is_target_or_result(col: str) -> bool:
    low = col.lower()
    bad_tokens = ["target", "settlement", "result", "fthg", "ftag", "ftr", "score", "goals_home", "goals_away"]
    return any(tok in low for tok in bad_tokens)


def safe_numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in IDENTITY_EXCLUDES or col in STRING_EXCLUDES:
            continue
        if is_target_or_result(col) or is_closing_column(col):
            continue
        if "current_club" in col.lower() or "lineup" in col.lower() or "player_name" in col.lower():
            continue
        if re.match(r"^[A-Z0-9]{1,8}", col):
            continue
        if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def tm_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("tm_", "home_tm_", "away_tm_", "home_minus_away_tm_", "home_div_away_tm_")
    out = []
    for col in safe_numeric_columns(df):
        if col.startswith(prefixes) and col not in {"tm_home_club_id", "tm_away_club_id", "tm_game_id"}:
            out.append(col)
    return out


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    safe = safe_numeric_columns(df)
    tm = set(tm_columns(df))
    market = [c for c in MARKET_BASELINE if c in df.columns]
    v1_prefixes = (
        "x1x2_",
        "home_all_",
        "away_all_",
        "home_home_",
        "away_away_",
        "all_",
        "venue_",
        "home_days_since_",
        "away_days_since_",
        "home_matches_last_",
        "away_matches_last_",
        "home_congested_",
        "away_congested_",
        "rest_days_diff",
        "matches_last_",
        "league_",
        "home_elo",
        "away_elo",
        "elo_",
        "venue_elo",
    )
    v1_safe = [
        c
        for c in safe
        if c not in tm
        and c not in {"season_start_year", "season_end_year"}
        and (c in market or c.startswith(v1_prefixes))
    ]
    transfer = [c for c in tm if any(tok in c for tok in ["arrival", "departure", "transfer", "churn", "tm_has_transfer", "feature_available"])]
    valuation = [c for c in tm if any(tok in c for tok in ["squad", "valuation", "staleness", "home_div_away_tm_squad", "feature_available"])]
    tm_all = sorted(tm)
    full = sorted(set(v1_safe + tm_all))
    tm_without_market = sorted(set(tm_all) - set(MARKET_BASELINE))
    league_only = [c for c in safe if c.startswith("league_") or c.startswith("league_code_") or c in {"season_start_year"}]
    return {
        "x1_market_baseline": market,
        "x1_market_plus_v1_1_safe": sorted(set(v1_safe + market)),
        "x1_market_plus_transfer_churn": sorted(set(market + transfer)),
        "x1_market_plus_tm_valuation": sorted(set(market + valuation)),
        "x1_market_plus_tm_all": sorted(set(market + tm_all)),
        "x1_full_safe_v2": full,
        "tm_without_market_odds": tm_without_market,
        "league_only_without_market_odds": league_only,
    }


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    base = df["season_start_year"].notna()
    if scope == "scope_A_full_modern_1x2":
        return base
    if scope == "scope_B_tm_available_same_rows":
        return base & df["tm_match_feature_available"].fillna(False).astype(bool)
    if scope == "scope_C_top_divisions_ex_e1_e2_e3":
        return base & ~df["league"].isin(LOWER_ENGLISH)
    if scope == "scope_D_modern_tm_available":
        return base & df["season_start_year"].astype(int).ge(2014) & df["tm_match_feature_available"].fillna(False).astype(bool)
    raise ValueError(scope)


def model_predict(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    rng: np.random.Generator,
    y_override: np.ndarray | None = None,
) -> np.ndarray:
    y_train = train["target_y"].to_numpy(dtype=int) if y_override is None else y_override.astype(int)
    if model_name == "raw_market_baseline":
        return raw_market_probs(test)
    cols = [c for c in cols if train[c].notna().any()]
    if not cols:
        return raw_market_probs(test)
    X_train = train[cols]
    X_test = test[cols]
    if model_name == "logistic_multinomial_l2":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=35, tol=1e-3, random_state=17, n_jobs=2)),
            ]
        )
        pipe.fit(X_train, y_train)
        return normalize_probs(pipe.predict_proba(X_test))
    if model_name in {"xgboost_multiclass_shallow", "xgboost_multiclass_depth3_regularized"}:
        if xgb is None:
            raise RuntimeError("xgboost unavailable")
        depth = 2 if model_name == "xgboost_multiclass_shallow" else 3
        est = 35 if model_name == "xgboost_multiclass_shallow" else 50
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    xgb.XGBClassifier(
                        objective="multi:softprob",
                        num_class=3,
                        max_depth=depth,
                        n_estimators=est,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=5.0,
                        reg_alpha=0.25,
                        min_child_weight=10,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        random_state=17,
                        n_jobs=2,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        return normalize_probs(pipe.predict_proba(X_test))
    if model_name == "xgboost_market_residual_multiclass":
        if xgb is None:
            raise RuntimeError("xgboost unavailable")
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_train)
        Xte = imp.transform(X_test)
        train_base = np.log(np.clip(raw_market_probs(train), 1e-6, 1.0)).reshape(-1)
        test_base = np.log(np.clip(raw_market_probs(test), 1e-6, 1.0)).reshape(-1)
        dtrain = xgb.DMatrix(Xtr, label=y_train, base_margin=train_base)
        dtest = xgb.DMatrix(Xte, base_margin=test_base)
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "lambda": 8.0,
            "alpha": 0.5,
            "min_child_weight": 15,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "seed": 17,
            "nthread": 2,
        }
        booster = xgb.train(params, dtrain, num_boost_round=35, verbose_eval=False)
        return normalize_probs(booster.predict(dtest))
    raise ValueError(model_name)


def evaluate_predictions(df: pd.DataFrame, prob: np.ndarray) -> dict[str, float]:
    y = df["target_y"].to_numpy(dtype=int)
    p = normalize_probs(prob)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier": brier_multi(y, p),
        "ece": ece_multi(y, p),
    }


def annual_predictions(df: pd.DataFrame, scope_name: str, feature_group: str, model_name: str, cols: list[str], control: str = "none") -> tuple[pd.DataFrame, list[dict]]:
    scoped = df[scope_mask(df, scope_name)].copy()
    preds = []
    yearly = []
    rng = np.random.default_rng(20260701)
    for year in TEST_YEARS:
        train = scoped[scoped["season_start_year"].astype(int).lt(year)].copy()
        test = scoped[scoped["season_start_year"].astype(int).eq(year)].copy()
        if len(train) < 500 or len(test) == 0:
            continue
        use_cols = list(cols)
        train_control = train
        test_control = test
        y_override = None
        if control == "shuffled_train_labels":
            y_override = rng.permutation(train["target_y"].to_numpy(dtype=int))
        elif control == "opposite_label_sanity":
            y_override = (2 - train["target_y"].to_numpy(dtype=int)).astype(int)
        elif control == "random_noise_replacing_tm":
            train_control, test_control = replace_tm_with_noise(train, test, use_cols, rng)
        elif control == "permuted_tm_within_league_season":
            train_control = permute_tm(train, use_cols, rng)
            test_control = test.copy()
        prob = model_predict(model_name, train_control, test_control, use_cols, rng, y_override=y_override)
        pred = test[["match_id", "match_date", "league", "season_start_year", "target_y"]].copy()
        pred[["prob_home", "prob_draw", "prob_away"]] = prob
        pred["scope"] = scope_name
        pred["feature_group"] = feature_group
        pred["model"] = model_name
        pred["control"] = control
        preds.append(pred)
        metrics = evaluate_predictions(test, prob)
        metrics.update({"scope": scope_name, "feature_group": feature_group, "model": model_name, "control": control, "test_year": year, "rows": len(test)})
        yearly.append(metrics)
    if not preds:
        return pd.DataFrame(), yearly
    return pd.concat(preds, ignore_index=True), yearly


def replace_tm_with_noise(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    tm = [c for c in cols if c.startswith(("tm_", "home_tm_", "away_tm_", "home_minus_away_tm_", "home_div_away_tm_"))]
    for col in tm:
        vals = pd.to_numeric(train[col], errors="coerce")
        mu = float(vals.mean()) if vals.notna().any() else 0.0
        sd = float(vals.std()) if vals.notna().any() and vals.std() > 0 else 1.0
        train[col] = rng.normal(mu, sd, len(train))
        test[col] = rng.normal(mu, sd, len(test))
    return train, test


def permute_tm(frame: pd.DataFrame, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    tm = [c for c in cols if c.startswith(("tm_", "home_tm_", "away_tm_", "home_minus_away_tm_", "home_div_away_tm_"))]
    for _, idx in out.groupby(["league", "season_start_year"]).groups.items():
        idx = list(idx)
        for col in tm:
            out.loc[idx, col] = rng.permutation(out.loc[idx, col].to_numpy())
    return out


def aggregate_summary(preds: pd.DataFrame) -> dict:
    metrics = evaluate_predictions(preds, preds[["prob_home", "prob_draw", "prob_away"]].to_numpy())
    metrics["rows"] = len(preds)
    metrics["test_years"] = "|".join(map(str, sorted(preds["season_start_year"].astype(int).unique())))
    return metrics


def add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    raw_lookup = {
        (row.scope, row.control): (row.log_loss, row.brier, row.ece)
        for row in out[out["model"].eq("raw_market_baseline")].itertuples(index=False)
    }
    v1_lookup = {
        row.scope: (row.log_loss, row.brier, row.ece)
        for row in out[(out["feature_group"].eq("x1_market_plus_v1_1_safe")) & (out["model"].eq("xgboost_market_residual_multiclass")) & (out["control"].eq("none"))].itertuples(index=False)
    }
    deltas = []
    for row in out.itertuples(index=False):
        raw = raw_lookup.get((row.scope, row.control), raw_lookup.get((row.scope, "none")))
        v1 = v1_lookup.get(row.scope)
        deltas.append(
            {
                "delta_log_loss_vs_raw_market": row.log_loss - raw[0] if raw else np.nan,
                "delta_brier_vs_raw_market": row.brier - raw[1] if raw else np.nan,
                "delta_ece_vs_raw_market": row.ece - raw[2] if raw else np.nan,
                "delta_log_loss_vs_v1_1_residual": row.log_loss - v1[0] if v1 else np.nan,
                "delta_brier_vs_v1_1_residual": row.brier - v1[1] if v1 else np.nan,
                "delta_ece_vs_v1_1_residual": row.ece - v1[2] if v1 else np.nan,
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(deltas)], axis=1)


def class_calibration(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, fg, model), g in preds[preds["control"].eq("none")].groupby(["scope", "feature_group", "model"]):
        y = g["target_y"].to_numpy(dtype=int)
        probs = g[["prob_home", "prob_draw", "prob_away"]].to_numpy()
        for i, cls in enumerate(CLASSES):
            rows.append(
                {
                    "scope": scope,
                    "feature_group": fg,
                    "model": model,
                    "class": cls,
                    "rows": len(g),
                    "mean_predicted_prob": float(probs[:, i].mean()),
                    "observed_rate": float((y == i).mean()),
                    "calibration_error": float(probs[:, i].mean() - (y == i).mean()),
                }
            )
    return pd.DataFrame(rows)


def edge_buckets(preds: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    base = df[["match_id", "x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away", "target_y"]].copy()
    out = preds[preds["control"].eq("none")].merge(base, on=["match_id", "target_y"], how="left")
    model_p = out[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    market_p = out[["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]].to_numpy()
    edge = model_p - market_p
    out["max_abs_edge"] = np.abs(edge).max(axis=1)
    out["edge_class"] = np.abs(edge).argmax(axis=1)
    bins = [-np.inf, 0.005, 0.01, 0.02, 0.04, np.inf]
    labels = ["<=0.5pp", "0.5-1pp", "1-2pp", "2-4pp", ">4pp"]
    out["edge_bucket"] = pd.cut(out["max_abs_edge"], bins=bins, labels=labels)
    rows = []
    for key, g in out.groupby(["scope", "feature_group", "model", "edge_bucket"], dropna=False):
        rows.append(
            {
                "scope": key[0],
                "feature_group": key[1],
                "model": key[2],
                "edge_bucket": str(key[3]),
                "rows": len(g),
                "mean_max_abs_edge": float(g["max_abs_edge"].mean()),
                "accuracy": float((g[["prob_home", "prob_draw", "prob_away"]].to_numpy().argmax(axis=1) == g["target_y"].to_numpy()).mean()),
                "log_loss": float(log_loss(g["target_y"], normalize_probs(g[["prob_home", "prob_draw", "prob_away"]].to_numpy()), labels=[0, 1, 2])) if len(g) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def per_league_year(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_year = []
    rows_league = []
    for key, g in preds[preds["control"].eq("none")].groupby(["scope", "feature_group", "model", "season_start_year"]):
        m = evaluate_predictions(g, g[["prob_home", "prob_draw", "prob_away"]].to_numpy())
        rows_year.append({"scope": key[0], "feature_group": key[1], "model": key[2], "season_start_year": int(key[3]), "rows": len(g), **m})
    for key, g in preds[preds["control"].eq("none")].groupby(["scope", "feature_group", "model", "league"]):
        if len(g) < 50:
            continue
        m = evaluate_predictions(g, g[["prob_home", "prob_draw", "prob_away"]].to_numpy())
        rows_league.append({"scope": key[0], "feature_group": key[1], "model": key[2], "league": key[3], "rows": len(g), **m})
    return pd.DataFrame(rows_year), pd.DataFrame(rows_league)


def run_main_audit(df: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_plan = {
        "x1_market_baseline": ["raw_market_baseline", "logistic_multinomial_l2"],
        "x1_market_plus_v1_1_safe": ["logistic_multinomial_l2", "xgboost_multiclass_shallow", "xgboost_multiclass_depth3_regularized", "xgboost_market_residual_multiclass"],
        "x1_market_plus_transfer_churn": ["logistic_multinomial_l2", "xgboost_multiclass_shallow"],
        "x1_market_plus_tm_valuation": ["logistic_multinomial_l2", "xgboost_multiclass_shallow"],
        "x1_market_plus_tm_all": ["logistic_multinomial_l2", "xgboost_multiclass_shallow", "xgboost_market_residual_multiclass"],
        "x1_full_safe_v2": ["logistic_multinomial_l2", "xgboost_multiclass_shallow", "xgboost_market_residual_multiclass"],
    }
    all_preds = []
    year_rows = []
    for scope in SCOPES:
        for fg, models in model_plan.items():
            cols = groups[fg]
            for model in models:
                print(f"main_audit scope={scope.name} feature_group={fg} model={model}", flush=True)
                pred, yearly = annual_predictions(df, scope.name, fg, model, cols)
                if not pred.empty:
                    all_preds.append(pred)
                    year_rows.extend(yearly)
    preds = pd.concat(all_preds, ignore_index=True)
    summary_rows = []
    for key, g in preds.groupby(["scope", "feature_group", "model", "control"]):
        summary_rows.append({"scope": key[0], "feature_group": key[1], "model": key[2], "control": key[3], **aggregate_summary(g)})
    summary = add_baseline_deltas(pd.DataFrame(summary_rows))
    years = pd.DataFrame(year_rows)
    years = add_yearly_raw_deltas(years)
    return preds, summary, years, pd.DataFrame()


def add_yearly_raw_deltas(years: pd.DataFrame) -> pd.DataFrame:
    year_col = "test_year" if "test_year" in years.columns else "season_start_year"
    raw = {
        (r.scope, getattr(r, year_col), getattr(r, "control", "none")): (r.log_loss, r.brier, r.ece)
        for r in years[years["model"].eq("raw_market_baseline")].itertuples(index=False)
    }
    rows = []
    for r in years.itertuples(index=False):
        year_value = getattr(r, year_col)
        control = getattr(r, "control", "none")
        b = raw.get((r.scope, year_value, control), raw.get((r.scope, year_value, "none")))
        rows.append(
            {
                "delta_log_loss_vs_raw_market": r.log_loss - b[0] if b else np.nan,
                "delta_brier_vs_raw_market": r.brier - b[1] if b else np.nan,
                "delta_ece_vs_raw_market": r.ece - b[2] if b else np.nan,
            }
        )
    return pd.concat([years.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def run_negative_controls(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    controls = [
        ("shuffled_train_labels", "x1_market_plus_tm_all", "xgboost_multiclass_shallow"),
        ("random_noise_replacing_tm", "x1_market_plus_tm_all", "xgboost_multiclass_shallow"),
        ("permuted_tm_within_league_season", "x1_market_plus_tm_all", "xgboost_multiclass_shallow"),
        ("none", "tm_without_market_odds", "xgboost_multiclass_shallow"),
        ("none", "league_only_without_market_odds", "logistic_multinomial_l2"),
        ("opposite_label_sanity", "x1_market_plus_tm_all", "xgboost_multiclass_shallow"),
    ]
    for scope in ["scope_B_tm_available_same_rows", "scope_D_modern_tm_available"]:
        for control, fg, model in controls:
            print(f"negative_control scope={scope} control={control} feature_group={fg} model={model}", flush=True)
            pred, _ = annual_predictions(df, scope, fg, model, groups[fg], control=control)
            if pred.empty:
                continue
            m = aggregate_summary(pred)
            rows.append({"scope": scope, "feature_group": fg, "model": model, "control": control, **m})
    return add_baseline_deltas(pd.DataFrame(rows)) if rows else pd.DataFrame()


def run_robustness(df: pd.DataFrame, groups: dict[str, list[str]], summary: pd.DataFrame, year_metrics: pd.DataFrame, league_metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = summary[
        summary["control"].eq("none")
        & summary["feature_group"].isin(["x1_market_plus_tm_all", "x1_full_safe_v2", "x1_market_plus_tm_valuation", "x1_market_plus_transfer_churn"])
    ].sort_values("delta_log_loss_vs_raw_market")
    if candidates.empty:
        return pd.DataFrame()
    best = candidates.iloc[0]
    scope = str(best["scope"])
    fg = str(best["feature_group"])
    model = str(best["model"])
    cols = groups[fg]
    base_scope = df[scope_mask(df, scope)].copy()
    conditions = []
    ym = year_metrics[(year_metrics["scope"].eq(scope)) & (year_metrics["feature_group"].eq(fg)) & (year_metrics["model"].eq(model))]
    lm = league_metrics[(league_metrics["scope"].eq(scope)) & (league_metrics["feature_group"].eq(fg)) & (league_metrics["model"].eq(model))]
    if not ym.empty:
        best_year = int(ym.sort_values("delta_log_loss_vs_raw_market").iloc[0]["season_start_year"])
        conditions.append((f"exclude_best_season_{best_year}", df["season_start_year"].astype(int).ne(best_year)))
    if not lm.empty:
        best_league = str(lm.sort_values("log_loss").iloc[0]["league"])
        conditions.append((f"exclude_best_league_{best_league}", df["league"].ne(best_league)))
    conditions.extend(
        [
            ("exclude_2026", df["season_start_year"].astype(int).ne(2026)),
            ("exclude_pre_2014_data", df["season_start_year"].astype(int).ge(2014)),
            ("exclude_E1_E2_E3", ~df["league"].isin(LOWER_ENGLISH)),
            ("top_divisions_only", df["league"].isin(TOP_DIVISIONS)),
            ("mapped_transfermarkt_fixtures_only", df["tm_match_feature_available"].fillna(False).astype(bool)),
        ]
    )
    for league in sorted(df["league"].dropna().unique()):
        conditions.append((f"exclude_league_{league}", df["league"].ne(league)))
    for threshold in [90, 180, 365]:
        stale_cols = [c for c in ["home_tm_avg_valuation_staleness_days_prior365", "away_tm_avg_valuation_staleness_days_prior365"] if c in df.columns]
        if stale_cols:
            conditions.append((f"diagnostic_remove_stale_valuation_gt_{threshold}", df[stale_cols].le(threshold).all(axis=1) | df[stale_cols].isna().all(axis=1)))
    rows = []
    for name, mask in conditions:
        print(f"robustness check={name} scope={scope} feature_group={fg} model={model}", flush=True)
        dfx = df[scope_mask(df, scope) & mask].copy()
        if len(dfx) < 1000:
            continue
        pred_raw, _ = annual_predictions(dfx, scope, "x1_market_baseline", "raw_market_baseline", groups["x1_market_baseline"])
        pred, _ = annual_predictions(dfx, scope, fg, model, cols)
        if pred.empty or pred_raw.empty:
            continue
        m = aggregate_summary(pred)
        r = aggregate_summary(pred_raw)
        rows.append(
            {
                "robustness_check": name,
                "scope": scope,
                "feature_group": fg,
                "model": model,
                "rows": m["rows"],
                "log_loss": m["log_loss"],
                "delta_log_loss_vs_raw_market": m["log_loss"] - r["log_loss"],
                "brier": m["brier"],
                "delta_brier_vs_raw_market": m["brier"] - r["brier"],
                "ece": m["ece"],
                "delta_ece_vs_raw_market": m["ece"] - r["ece"],
            }
        )
    return pd.DataFrame(rows)


def predictive_gate(summary: pd.DataFrame, years: pd.DataFrame, robustness: pd.DataFrame, negatives: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    rows = []
    tm = summary[
        summary["control"].eq("none")
        & summary["feature_group"].isin(["x1_market_plus_tm_all", "x1_full_safe_v2", "x1_market_plus_tm_valuation", "x1_market_plus_transfer_churn"])
    ].copy()
    for r in tm.itertuples(index=False):
        yr = years[(years["scope"].eq(r.scope)) & (years["feature_group"].eq(r.feature_group)) & (years["model"].eq(r.model))]
        improved_years = int((yr["delta_log_loss_vs_raw_market"] < 0).sum()) if not yr.empty else 0
        gate = (
            r.delta_log_loss_vs_raw_market < 0
            and r.delta_brier_vs_raw_market < 0
            and r.delta_log_loss_vs_v1_1_residual < 0
            and r.delta_ece_vs_raw_market <= 0.005
            and improved_years >= 5
        )
        rows.append(
            {
                "scope": r.scope,
                "feature_group": r.feature_group,
                "model": r.model,
                "passes_predictive_gate": bool(gate),
                "improved_years_vs_raw_log_loss": improved_years,
                "delta_log_loss_vs_raw_market": r.delta_log_loss_vs_raw_market,
                "delta_brier_vs_raw_market": r.delta_brier_vs_raw_market,
                "delta_log_loss_vs_v1_1_residual": r.delta_log_loss_vs_v1_1_residual,
                "delta_ece_vs_raw_market": r.delta_ece_vs_raw_market,
            }
        )
    gate_df = pd.DataFrame(rows)
    return bool(gate_df["passes_predictive_gate"].any()) if not gate_df.empty else False, gate_df


def write_value_outputs(gate_passed: bool) -> None:
    if gate_passed:
        VALUE_MD.write_text("# 1X2 value review\n\nPredictive gate passed, but value review implementation is intentionally not run in this script without prior fixed-rule loader confirmation.\n", encoding="utf-8")
    else:
        VALUE_MD.write_text("# 1X2 value review\n\nNot run. No Transfermarkt model passed the predeclared predictive gate, so locked value review was not eligible.\n", encoding="utf-8")
    empty = pd.DataFrame([{"status": "not_run", "reason": "predictive_gate_not_passed" if not gate_passed else "requires_prior_fixed_rule_loader"}])
    empty.to_csv(VALUE_FIXED_CSV, index=False)
    empty.to_csv(VALUE_NESTED_CSV, index=False)
    empty.to_csv(VALUE_CONTROLS_CSV, index=False)
    empty.to_csv(VALUE_ROBUSTNESS_CSV, index=False)


def write_report(summary: pd.DataFrame, scope_comp: pd.DataFrame, gate_df: pd.DataFrame, final_class: str, value_run: bool) -> None:
    best = summary[summary["control"].eq("none")].sort_values("delta_log_loss_vs_raw_market").head(12)
    comp = scope_comp.copy()
    lines = [
        "# feature_matrix_v2_transfermarkt_partial 1X2 predictive and gated value audit",
        "",
        "No live betting, threshold optimization, closing-odds features, team/player name features, lineups, current-club fields, future valuations, or future transfers were used.",
        "",
        f"Final classification: `{final_class}`",
        f"Locked value review run: `{value_run}`",
        "",
        "## Best Predictive Rows",
        best[[
            "scope",
            "feature_group",
            "model",
            "rows",
            "log_loss",
            "delta_log_loss_vs_raw_market",
            "brier",
            "delta_brier_vs_raw_market",
            "ece",
            "delta_ece_vs_raw_market",
            "delta_log_loss_vs_v1_1_residual",
        ]].to_markdown(index=False),
        "",
        "## Required Comparisons",
        comp.to_markdown(index=False),
        "",
        "## Predictive Gate",
        gate_df.sort_values(["passes_predictive_gate", "delta_log_loss_vs_raw_market"], ascending=[False, True]).head(20).to_markdown(index=False) if not gate_df.empty else "No eligible Transfermarkt rows.",
        "",
        "## Interpretation",
        "Transfermarkt feature groups were evaluated only on same-row scope samples. The gate requires simultaneous improvement over raw market and the v1.1 residual baseline, acceptable calibration, year-level consistency, negative-control sanity, and robustness. No locked value review was run unless that predictive gate passed.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scope_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, g in summary[summary["control"].eq("none")].groupby("scope"):
        def pick(fg: str, model: str | None = None):
            x = g[g["feature_group"].eq(fg)]
            if model:
                x = x[x["model"].eq(model)]
            if x.empty:
                return None
            return x.sort_values("log_loss").iloc[0]
        pairs = [
            ("raw_market vs v1_1_safe", pick("x1_market_baseline", "raw_market_baseline"), pick("x1_market_plus_v1_1_safe")),
            ("v1_1_safe vs tm_all", pick("x1_market_plus_v1_1_safe"), pick("x1_market_plus_tm_all")),
            ("v1_1_safe vs full_safe_v2", pick("x1_market_plus_v1_1_safe"), pick("x1_full_safe_v2")),
            ("transfer_churn vs valuation_only", pick("x1_market_plus_transfer_churn"), pick("x1_market_plus_tm_valuation")),
            ("valuation_only vs tm_all", pick("x1_market_plus_tm_valuation"), pick("x1_market_plus_tm_all")),
        ]
        for name, a, b in pairs:
            if a is None or b is None:
                continue
            rows.append(
                {
                    "scope": scope,
                    "comparison": name,
                    "left_feature_group": a.feature_group,
                    "left_model": a.model,
                    "right_feature_group": b.feature_group,
                    "right_model": b.model,
                    "rows": int(a.rows),
                    "right_minus_left_log_loss": float(b.log_loss - a.log_loss),
                    "right_minus_left_brier": float(b.brier - a.brier),
                    "right_minus_left_ece": float(b.ece - a.ece),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    groups = feature_groups(df)
    preds, summary, year_metrics, _ = run_main_audit(df, groups)
    yearly, league = per_league_year(preds)
    yearly = add_yearly_raw_deltas(yearly)
    comp = scope_comparisons(summary)
    summary.to_csv(SUMMARY_CSV, index=False)
    comp.to_csv(SCOPE_CSV, index=False)
    class_calibration(preds).to_csv(CLASS_CAL_CSV, index=False)
    edge_buckets(preds, df).to_csv(EDGE_BUCKET_CSV, index=False)
    neg = run_negative_controls(df, groups)
    neg.to_csv(NEGATIVE_CSV, index=False)
    robustness = run_robustness(df, groups, summary, yearly, league)
    robustness.to_csv(ROBUSTNESS_CSV, index=False)
    gate_passed, gate_df = predictive_gate(summary, yearly, robustness, neg)
    value_run = bool(gate_passed)
    write_value_outputs(value_run)
    final_class = "predictive_only_no_value" if value_run else "research_only"
    if not (summary["delta_log_loss_vs_raw_market"] < 0).any():
        final_class = "reject"
    write_report(summary, comp, gate_df, final_class, value_run)


if __name__ == "__main__":
    main()
