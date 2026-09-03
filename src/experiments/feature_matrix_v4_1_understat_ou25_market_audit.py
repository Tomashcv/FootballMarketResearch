from __future__ import annotations

from pathlib import Path
import math
import re
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

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments.feature_matrix_v2_tm_1x2_predictive_audit import is_closing_column, is_target_or_result


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

V3_MATRIX = Path("data/processed/features/football_feature_matrix_v3_clubelo_partial.csv")
V4_MATRIX = Path("data/processed/features/football_feature_matrix_v4_1_understat_partial_v2.csv")
V4_LEAKAGE = Path("outputs/reports/feature_matrix_v4_1_understat_v2_leakage_checks.csv")
REPORT_DIR = Path("outputs/reports")

REPORT_MD = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_predictive_audit.md"
SUMMARY_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_predictive_summary.csv"
CALIBRATION_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_calibration.csv"
EDGE_BUCKET_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_edge_bucket_calibration.csv"
NEGATIVE_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_negative_controls.csv"
STALENESS_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_staleness_diagnostics.csv"
VALUE_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_value_diagnostics.csv"
VALUE_NESTED_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_value_nested_selection.csv"
VALUE_YEAR_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_value_year_breakdown.csv"
VALUE_LEAGUE_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_value_league_breakdown.csv"
LEAKAGE_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_leakage_checks.csv"
DECISION_MD = REPORT_DIR / "feature_matrix_v4_1_understat_ou25_decision.md"

TARGET = "target_over_2_5"
TARGET_AVAILABLE = "target_ou25_available"
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
TOP5 = {"E0", "D1", "SP1", "I1", "F1"}
SOURCE_END = pd.Timestamp("2024-09-29")

MARKET_COLS = [
    "ou25_avg_prob_over",
    "ou25_avg_prob_under",
    "ou25_avg_odds_over",
    "ou25_avg_odds_under",
    "ou25_avg_market_overround",
]
V3_BASELINE = "ou_v3_safe"
CORE = "ou_v4_1_understat_core"
FULL = "ou_v4_1_understat_full"
BASELINE = "ou_market_baseline"
MODELS = ["raw_market_baseline", "logistic_l2", "xgboost_shallow", "xgboost_depth3_regularized", "xgboost_market_residual_binary"]
VALUE_RULES = [
    ("over", 0.01, 1.5),
    ("over", 0.015, 1.5),
    ("over", 0.02, 1.5),
    ("under", 0.01, 1.5),
    ("under", 0.015, 1.5),
    ("under", 0.02, 1.5),
]


def ece_binary(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(out)


def z_score(profit: pd.Series) -> float:
    n = int(len(profit))
    if n <= 1:
        return 0.0
    sd = float(profit.std(ddof=1))
    return float(profit.sum() / (sd * math.sqrt(n))) if sd > 0 else 0.0


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return view.to_markdown(index=False)


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce").astype("Int64")
    for col in MARKET_COLS + [TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = (
        df[TARGET_AVAILABLE].fillna(False).astype(bool)
        & df[TARGET].notna()
        & df[MARKET_COLS].notna().all(axis=1)
        & df["ou25_avg_odds_over"].gt(1.0)
        & df["ou25_avg_odds_under"].gt(1.0)
        & df["season_start_year"].notna()
    )
    out = df[valid].copy()
    out[TARGET] = out[TARGET].astype(int)
    return out.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "full_available_ou_scope":
        return pd.Series(True, index=df.index)
    if scope == "top5_only":
        return df["league"].isin(TOP5)
    if scope == "top5_understat_available":
        return df["league"].isin(TOP5) & df["understat_both_available_flag"].fillna(False).astype(bool)
    if scope == "modern_top5_2020_2025":
        return df["league"].isin(TOP5) & df["season_start_year"].between(2020, 2025)
    raise ValueError(scope)


def safe_numeric_columns(df: pd.DataFrame) -> list[str]:
    excludes = {
        "match_id",
        "home_team",
        "away_team",
        "match_date",
        "league",
        "target_outcome_1x2",
        "ou25_avg_market_source",
        "x1x2_avg_market_source",
        "ah_avg_market_source",
        "source_processed_file",
        "feature_matrix_version",
    }
    cols = []
    for col in df.columns:
        low = col.lower()
        if col in excludes or col.startswith("home_understat_") or col.startswith("away_understat_") or "understat" in col:
            continue
        if is_target_or_result(col) or is_closing_column(col):
            continue
        if any(tok in low for tok in ["score", "fthg", "ftag", "result", "settlement", "current_club", "lineup", "player_name"]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)
    return cols


def understat_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    under = [c for c in df.columns if "understat" in c and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    flags = [
        "understat_home_history_count",
        "understat_away_history_count",
        "understat_home_latest_days_ago",
        "understat_away_latest_days_ago",
        "understat_home_available_flag",
        "understat_away_available_flag",
        "understat_both_available_flag",
    ]
    core_tokens = ["xg_for", "xg_against", "npxg_for", "npxg_against", "xg_diff", "npxg_diff"]
    core = [c for c in under if any(tok in c for tok in core_tokens) or c in flags]
    return sorted(set(core)), sorted(set(under))


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    safe = safe_numeric_columns(df)
    v3 = sorted(set(safe + MARKET_COLS))
    core, full = understat_cols(df)
    return {
        BASELINE: MARKET_COLS,
        V3_BASELINE: v3,
        CORE: sorted(set(v3 + core)),
        FULL: sorted(set(v3 + full)),
    }


def market_prob(frame: pd.DataFrame) -> np.ndarray:
    p = pd.to_numeric(frame["ou25_avg_prob_over"], errors="coerce").to_numpy(dtype=float)
    return np.clip(p, 1e-6, 1 - 1e-6)


def predict_model(model: str, train: pd.DataFrame, test: pd.DataFrame, cols: list[str], seed: int, y_override: np.ndarray | None = None) -> np.ndarray:
    if model == "raw_market_baseline":
        return market_prob(test)
    cols = [c for c in cols if c in train.columns and train[c].notna().any()]
    y = train[TARGET].to_numpy(dtype=int) if y_override is None else y_override.astype(int)
    if not cols:
        return market_prob(test)
    if model == "logistic_l2":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=250, random_state=seed)),
            ]
        )
        pipe.fit(train[cols], y)
        return np.clip(pipe.predict_proba(test[cols])[:, 1], 1e-6, 1 - 1e-6)
    if xgb is None:
        raise RuntimeError("xgboost unavailable")
    imp = SimpleImputer(strategy="median")
    xtr = imp.fit_transform(train[cols])
    xte = imp.transform(test[cols])
    if model == "xgboost_market_residual_binary":
        base_tr = np.log(market_prob(train) / (1 - market_prob(train)))
        base_te = np.log(market_prob(test) / (1 - market_prob(test)))
        dtrain = xgb.DMatrix(xtr, label=y, base_margin=base_tr)
        dtest = xgb.DMatrix(xte, base_margin=base_te)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "lambda": 8.0,
            "alpha": 0.5,
            "min_child_weight": 15,
            "tree_method": "hist",
            "seed": seed,
            "nthread": 2,
        }
        booster = xgb.train(params, dtrain, num_boost_round=35, verbose_eval=False)
        return np.clip(booster.predict(dtest), 1e-6, 1 - 1e-6)
    depth = 2 if model == "xgboost_shallow" else 3
    rounds = 35 if model == "xgboost_shallow" else 50
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=depth,
        n_estimators=rounds,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        reg_alpha=0.25,
        min_child_weight=10,
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
    )
    clf.fit(xtr, y)
    return np.clip(clf.predict_proba(xte)[:, 1], 1e-6, 1 - 1e-6)


def metrics(frame: pd.DataFrame, p: np.ndarray) -> dict[str, float]:
    y = frame[TARGET].to_numpy(dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece_binary(y, p),
    }


def run_predictions(df: pd.DataFrame, scope: str, group: str, model: str, cols: list[str]) -> pd.DataFrame:
    scoped = df[scope_mask(df, scope)].copy()
    rows = []
    for year in TEST_YEARS:
        train = scoped[scoped["season_start_year"].astype(int).lt(year)].copy()
        test = scoped[scoped["season_start_year"].astype(int).eq(year)].copy()
        if len(train) < 500 or len(test) == 0:
            continue
        print(f"ou25 scope={scope} group={group} model={model} test_year={year} rows={len(test)}", flush=True)
        p = predict_model(model, train, test, cols, 20260702)
        out = test[["match_id", "match_date", "league", "season_start_year", TARGET, "ou25_avg_prob_over", "ou25_avg_prob_under", "ou25_avg_odds_over", "ou25_avg_odds_under", "understat_both_available_flag", "understat_home_latest_days_ago", "understat_away_latest_days_ago"]].copy()
        out["scope"] = scope
        out["feature_group"] = group
        out["model"] = model
        out["model_prob_over"] = p
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_predictions(pred: pd.DataFrame) -> dict[str, object]:
    market = metrics(pred, pred["ou25_avg_prob_over"].to_numpy(dtype=float))
    model = metrics(pred, pred["model_prob_over"].to_numpy(dtype=float))
    return {
        "scope": pred["scope"].iloc[0],
        "feature_group": pred["feature_group"].iloc[0],
        "model": pred["model"].iloc[0],
        "rows": int(len(pred)),
        "market_accuracy": market["accuracy"],
        "model_accuracy": model["accuracy"],
        "market_log_loss": market["log_loss"],
        "model_log_loss": model["log_loss"],
        "delta_log_loss": model["log_loss"] - market["log_loss"],
        "market_brier": market["brier"],
        "model_brier": model["brier"],
        "delta_brier": model["brier"] - market["brier"],
        "market_ece": market["ece"],
        "model_ece": model["ece"],
        "delta_ece": model["ece"] - market["ece"],
        "seasons_improved_log_loss": int(pred.groupby("season_start_year").apply(lambda g: metrics(g, g["model_prob_over"])["log_loss"] < metrics(g, g["ou25_avg_prob_over"])["log_loss"]).sum()),
        "test_seasons": int(pred["season_start_year"].nunique()),
    }


def calibration(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = pred.copy()
    frame["prob_bucket"] = pd.cut(frame["model_prob_over"], bins=np.linspace(0, 1, 11), include_lowest=True)
    for bucket, g in frame.groupby("prob_bucket", dropna=False):
        rows.append(
            {
                "scope": frame["scope"].iloc[0],
                "feature_group": frame["feature_group"].iloc[0],
                "model": frame["model"].iloc[0],
                "bucket": str(bucket),
                "rows": int(len(g)),
                "mean_model_prob_over": float(g["model_prob_over"].mean()) if len(g) else np.nan,
                "observed_over_rate": float(g[TARGET].mean()) if len(g) else np.nan,
                "observed_under_rate": float(1.0 - g[TARGET].mean()) if len(g) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def edge_buckets(pred: pd.DataFrame) -> pd.DataFrame:
    frame = pred.copy()
    frame["over_edge"] = frame["model_prob_over"] - frame["ou25_avg_prob_over"]
    frame["under_edge"] = (1.0 - frame["model_prob_over"]) - frame["ou25_avg_prob_under"]
    frame["max_abs_edge"] = np.maximum(frame["over_edge"].abs(), frame["under_edge"].abs())
    frame["edge_bucket"] = pd.cut(frame["max_abs_edge"], bins=[-np.inf, 0.005, 0.01, 0.02, 0.04, np.inf], labels=["<=0.5pp", "0.5-1pp", "1-2pp", "2-4pp", ">4pp"])
    rows = []
    for bucket, g in frame.groupby("edge_bucket", dropna=False):
        rows.append(
            {
                "scope": frame["scope"].iloc[0],
                "feature_group": frame["feature_group"].iloc[0],
                "model": frame["model"].iloc[0],
                "edge_bucket": str(bucket),
                "rows": int(len(g)),
                "mean_max_abs_edge": float(g["max_abs_edge"].mean()) if len(g) else np.nan,
                "mean_over_edge": float(g["over_edge"].mean()) if len(g) else np.nan,
                "observed_over_rate": float(g[TARGET].mean()) if len(g) else np.nan,
                "log_loss": metrics(g, g["model_prob_over"])["log_loss"] if len(g) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def latest_days(pred: pd.DataFrame) -> pd.Series:
    return pd.concat(
        [
            pd.to_numeric(pred["understat_home_latest_days_ago"], errors="coerce"),
            pd.to_numeric(pred["understat_away_latest_days_ago"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)


def staleness_diag(pred: pd.DataFrame) -> pd.DataFrame:
    latest = latest_days(pred)
    both = pred["understat_both_available_flag"].fillna(False).astype(bool)
    after_source = pd.to_datetime(pred["match_date"]).gt(SOURCE_END)
    masks = {
        "all_test_seasons": pd.Series(True, index=pred.index),
        "exclude_2025": pred["season_start_year"].astype(int).ne(2025),
        "exclude_after_2024_09_29": ~after_source,
        "both_understat_available": both,
        "latest_days_le_180": latest.le(180),
        "latest_days_le_365": latest.le(365),
        "latest_days_le_730": latest.le(730),
    }
    rows = []
    for segment, mask in masks.items():
        g = pred[mask]
        if len(g):
            m = summarize_predictions(g.assign(scope=pred["scope"].iloc[0], feature_group=pred["feature_group"].iloc[0], model=pred["model"].iloc[0]))
            m["segment"] = segment
            rows.append(m)
        else:
            rows.append({"scope": pred["scope"].iloc[0], "feature_group": pred["feature_group"].iloc[0], "model": pred["model"].iloc[0], "segment": segment, "rows": 0})
    return pd.DataFrame(rows)


def negative_controls(df: pd.DataFrame, candidates: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260702)
    for cand in candidates.head(2).itertuples(index=False):
        scoped = df[scope_mask(df, cand.scope)].copy()
        group = str(cand.feature_group)
        model = str(cand.model)
        cols = groups[group]
        if model == "raw_market_baseline":
            continue
        for control in ["shuffled_target", "shuffled_understat_features"]:
            parts = []
            for year in TEST_YEARS:
                train = scoped[scoped["season_start_year"].astype(int).lt(year)].copy()
                test = scoped[scoped["season_start_year"].astype(int).eq(year)].copy()
                if len(train) < 500 or len(test) == 0:
                    continue
                y_override = None
                train_use = train.copy()
                if control == "shuffled_target":
                    y_override = rng.permutation(train[TARGET].to_numpy(dtype=int))
                elif control == "shuffled_understat_features":
                    under = [c for c in cols if "understat" in c]
                    for col in under:
                        train_use[col] = rng.permutation(train_use[col].to_numpy())
                elif control == "season_permuted_understat":
                    under = [c for c in cols if "understat" in c]
                    shuffled = train_use[under].sample(frac=1.0, random_state=20260702 + year).reset_index(drop=True)
                    train_use.loc[:, under] = shuffled.to_numpy()
                p = predict_model(model, train_use, test, cols, 20260703, y_override=y_override)
                part = test[["match_id", "match_date", "league", "season_start_year", TARGET, "ou25_avg_prob_over", "ou25_avg_prob_under", "ou25_avg_odds_over", "ou25_avg_odds_under", "understat_both_available_flag", "understat_home_latest_days_ago", "understat_away_latest_days_ago"]].copy()
                part["scope"] = cand.scope
                part["feature_group"] = group
                part["model"] = model
                part["model_prob_over"] = p
                parts.append(part)
            if parts:
                pred = pd.concat(parts, ignore_index=True)
                m = summarize_predictions(pred)
                m["control"] = control
                rows.append(m)
        rows.append(
            {
                "scope": cand.scope,
                "feature_group": group,
                "model": model,
                "control": "season_permuted_understat",
                "rows": 0,
                "delta_log_loss": np.nan,
                "delta_brier": np.nan,
                "delta_ece": np.nan,
                "control_status": "skipped_not_cheap_for_full_matrix_run",
            }
        )
    return pd.DataFrame(rows)


def predictive_gate(summary: pd.DataFrame, negatives: pd.DataFrame, checks_ok: bool) -> pd.DataFrame:
    if not checks_ok:
        return pd.DataFrame()
    cand = summary[
        summary["model"].ne("raw_market_baseline")
        & summary["delta_log_loss"].lt(0)
        & summary["delta_brier"].le(0.0005)
        & summary["delta_ece"].le(0.005)
        & (
            (summary["seasons_improved_log_loss"].ge(4))
            | (summary["delta_log_loss"].lt(-0.0005))
        )
    ].copy()
    if cand.empty or negatives.empty:
        return pd.DataFrame()
    neg_key = negatives.groupby(["scope", "feature_group", "model"])["delta_log_loss"].min().reset_index(name="best_control_delta_log_loss")
    cand = cand.merge(neg_key, on=["scope", "feature_group", "model"], how="left")
    cand = cand[cand["best_control_delta_log_loss"].fillna(0.0).ge(cand["delta_log_loss"] * 0.25)]
    return cand.sort_values(["delta_log_loss", "delta_brier"])


def add_value_cols(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["over_edge"] = out["model_prob_over"] - out["ou25_avg_prob_over"]
    out["under_edge"] = (1.0 - out["model_prob_over"]) - out["ou25_avg_prob_under"]
    out["over_profit"] = np.where(out[TARGET].eq(1), out["ou25_avg_odds_over"] - 1.0, -1.0)
    out["under_profit"] = np.where(out[TARGET].eq(0), out["ou25_avg_odds_under"] - 1.0, -1.0)
    return out


def select_rule(frame: pd.DataFrame, side: str, edge: float, odds: float) -> pd.DataFrame:
    side_odds = "ou25_avg_odds_over" if side == "over" else "ou25_avg_odds_under"
    selected = frame[frame[f"{side}_edge"].ge(edge) & frame[side_odds].ge(odds)].copy()
    selected["side"] = side
    selected["selected_rule"] = f"{side}_edge_{edge:g}_odds_{odds:g}"
    selected["profit"] = selected[f"{side}_profit"]
    return selected


def bet_summary(selected: pd.DataFrame) -> dict[str, object]:
    bets = int(len(selected))
    profit = float(selected["profit"].sum()) if bets else 0.0
    best_year = selected.groupby("season_start_year")["profit"].sum().idxmax() if bets else ""
    best_league = selected.groupby("league")["profit"].sum().idxmax() if bets else ""
    best_year_profit = float(selected.groupby("season_start_year")["profit"].sum().max()) if bets else 0.0
    best_league_profit = float(selected.groupby("league")["profit"].sum().max()) if bets else 0.0
    cum = selected["profit"].cumsum() if bets else pd.Series(dtype=float)
    drawdown = float((cum.cummax() - cum).max()) if bets else 0.0
    return {
        "bets": bets,
        "profit": profit,
        "roi": profit / bets if bets else 0.0,
        "z": z_score(selected["profit"]) if bets else 0.0,
        "max_drawdown": drawdown,
        "best_year": best_year,
        "best_league": best_league,
        "best_year_share": best_year_profit / profit if profit > 0 else np.nan,
        "best_league_share": best_league_profit / profit if profit > 0 else np.nan,
        "profit_ex_best_year": float(selected[selected["season_start_year"].ne(best_year)]["profit"].sum()) if bets else 0.0,
        "profit_ex_best_league": float(selected[~selected["league"].eq(best_league)]["profit"].sum()) if bets else 0.0,
    }


def nested_value(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_value_cols(pred)
    rows, bets = [], []
    for year in TEST_YEARS:
        prior = data[data["season_start_year"].astype(int).lt(year)]
        current = data[data["season_start_year"].astype(int).eq(year)]
        candidates = []
        for side, edge, odds in VALUE_RULES:
            selected = select_rule(prior, side, edge, odds)
            if len(selected) >= 100:
                stats = bet_summary(selected)
                if stats["profit"] > 0 and stats["z"] > 0.5 and selected["league"].nunique() >= 3:
                    candidates.append({"side": side, "edge": edge, "odds": odds, "rule": f"{side}_edge_{edge:g}_odds_{odds:g}", **stats})
        if not candidates:
            rows.append({"test_year": year, "selected_rule": "", "selection_status": "no_prior_rule_passed", "test_bets": 0, "test_profit": 0.0, "test_roi": 0.0, "test_z": 0.0})
            continue
        chosen = pd.DataFrame(candidates).sort_values(["z", "profit", "bets"], ascending=[False, False, False]).iloc[0]
        selected_test = select_rule(current, str(chosen["side"]), float(chosen["edge"]), float(chosen["odds"]))
        stats = bet_summary(selected_test)
        rows.append({"test_year": year, "selected_rule": chosen["rule"], "selection_status": "selected_prior_out_of_sample_only", "test_bets": stats["bets"], "test_profit": stats["profit"], "test_roi": stats["roi"], "test_z": stats["z"]})
        bets.append(selected_test.assign(selected_rule=str(chosen["rule"])))
    return pd.DataFrame(rows), pd.concat(bets, ignore_index=True, sort=False) if bets else pd.DataFrame()


def leakage_checks(v3_raw: pd.DataFrame, v4_raw: pd.DataFrame, groups: dict[str, list[str]], selected: pd.DataFrame | None) -> pd.DataFrame:
    build = pd.read_csv(V4_LEAKAGE)
    all_cols = sorted(set(sum(groups.values(), [])))
    bad_current = [c for c in all_cols if re.search(r"current|scored|missed|result|score|fthg|ftag", c, re.I)]
    bad_target = [c for c in all_cols if is_target_or_result(c)]
    closing = [c for c in all_cols if is_closing_column(c)]
    bad_under_direct = [c for c in all_cols if re.fullmatch(r"(xg|npxg|xga|goals|pts|scored|missed)", c.lower())]
    unchanged_bad = []
    common = [c for c in v3_raw.columns if c in v4_raw.columns]
    for col in common:
        left, right = v3_raw[col], v4_raw[col]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            ok = np.allclose(pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce"), equal_nan=True, rtol=1e-12, atol=1e-12)
        else:
            ok = left.astype("string").fillna("<NA>").equals(right.astype("string").fillna("<NA>"))
        if not ok:
            unchanged_bad.append(col)
    dup = int(selected.duplicated(["match_id", "side"]).sum()) if selected is not None and len(selected) else 0
    profit_bad = 0
    if selected is not None and len(selected):
        expected = np.where(
            selected["side"].eq("over"),
            np.where(selected[TARGET].eq(1), selected["ou25_avg_odds_over"] - 1.0, -1.0),
            np.where(selected[TARGET].eq(0), selected["ou25_avg_odds_under"] - 1.0, -1.0),
        )
        profit_bad = int((np.round(selected["profit"].to_numpy(dtype=float), 12) != np.round(expected, 12)).sum())
    rows = [
        ("v4_1_row_count_equals_v3", len(v3_raw) == len(v4_raw), len(v4_raw), f"v3={len(v3_raw)} v4={len(v4_raw)}"),
        ("v3_columns_unchanged_in_v4", len(unchanged_bad) == 0, len(unchanged_bad), "|".join(unchanged_bad[:20])),
        ("no_current_match_understat_xg_result_scored_missed_direct_features", len(bad_current) + len(bad_under_direct) == 0, len(bad_current) + len(bad_under_direct), "|".join((bad_current + bad_under_direct)[:20])),
        ("no_target_result_score_columns_used_as_features", len(bad_target) == 0, len(bad_target), "|".join(bad_target[:20])),
        ("no_closing_odds_used_for_selection", len(closing) == 0, len(closing), "|".join(closing)),
        ("no_duplicate_selected_matches", dup == 0, dup, ""),
        ("profit_formula_correct_for_over_under", profit_bad == 0, profit_bad, ""),
    ]
    for name in ["all_contributing_understat_rows_strictly_before_match_date", "no_same_day_understat_joins", "no_future_understat_joins"]:
        row = build[build["check"].eq(name)]
        rows.append((name, bool(len(row) and row["status"].iloc[0] == "pass"), int(row["count"].iloc[0]) if len(row) else -1, "from v4.1 build audit"))
    return pd.DataFrame([{"check": n, "status": "pass" if ok else "fail", "count": int(count), "detail": detail} for n, ok, count, detail in rows])


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    v3_raw = pd.read_csv(V3_MATRIX, low_memory=False)
    v4_raw = pd.read_csv(V4_MATRIX, low_memory=False)
    df = load_matrix(V4_MATRIX)
    groups = feature_groups(df)
    preds = {}
    summaries, cals, edges = [], [], []
    for scope in ["full_available_ou_scope", "top5_only", "top5_understat_available", "modern_top5_2020_2025"]:
        for group, cols in groups.items():
            models = ["raw_market_baseline"] if group == BASELINE else MODELS[1:]
            for model in models:
                pred = run_predictions(df, scope, group, model, cols)
                if pred.empty:
                    continue
                preds[(scope, group, model)] = pred
                summaries.append(summarize_predictions(pred))
                cals.append(calibration(pred))
                edges.append(edge_buckets(pred))
    summary = pd.DataFrame(summaries).sort_values(["scope", "delta_log_loss", "delta_brier"])
    pd.concat(cals, ignore_index=True).to_csv(CALIBRATION_CSV, index=False)
    pd.concat(edges, ignore_index=True).to_csv(EDGE_BUCKET_CSV, index=False)

    prelim_checks = leakage_checks(v3_raw, v4_raw, groups, None)
    prelim_ok = prelim_checks["status"].eq("pass").all()
    provisional = summary[summary["delta_log_loss"].lt(0)].sort_values(["delta_log_loss", "delta_brier"])
    negatives = negative_controls(df, provisional, groups) if len(provisional) else pd.DataFrame()
    negatives.to_csv(NEGATIVE_CSV, index=False)
    gate = predictive_gate(summary, negatives, prelim_ok)
    summary["predictive_gate_pass"] = summary.set_index(["scope", "feature_group", "model"]).index.isin(gate.set_index(["scope", "feature_group", "model"]).index) if len(gate) else False
    summary.to_csv(SUMMARY_CSV, index=False)

    stale_parts = []
    for key, pred in preds.items():
        if key[1] in {CORE, FULL}:
            stale_parts.append(staleness_diag(pred))
    stale = pd.concat(stale_parts, ignore_index=True) if stale_parts else pd.DataFrame()
    stale.to_csv(STALENESS_CSV, index=False)

    value_rows, nested_rows, year_rows, league_rows = [], [], [], []
    selected_all = []
    if len(gate):
        for cand in gate.itertuples(index=False):
            key = (cand.scope, cand.feature_group, cand.model)
            nested, selected = nested_value(preds[key])
            nested = nested.assign(scope=cand.scope, feature_group=cand.feature_group, model=cand.model)
            nested_rows.append(nested)
            if len(selected):
                selected = selected.assign(scope=cand.scope, feature_group=cand.feature_group, model=cand.model)
                selected_all.append(selected)
                value_rows.append({"scope": cand.scope, "feature_group": cand.feature_group, "model": cand.model, **bet_summary(selected)})
                for year, g in selected.groupby("season_start_year"):
                    year_rows.append({"scope": cand.scope, "feature_group": cand.feature_group, "model": cand.model, "season_start_year": year, **bet_summary(g)})
                for league, g in selected.groupby("league"):
                    league_rows.append({"scope": cand.scope, "feature_group": cand.feature_group, "model": cand.model, "league": league, **bet_summary(g)})
            else:
                value_rows.append({"scope": cand.scope, "feature_group": cand.feature_group, "model": cand.model, **bet_summary(pd.DataFrame(columns=["profit"]))})
    pd.DataFrame(value_rows).to_csv(VALUE_CSV, index=False)
    (pd.concat(nested_rows, ignore_index=True) if nested_rows else pd.DataFrame()).to_csv(VALUE_NESTED_CSV, index=False)
    pd.DataFrame(year_rows).to_csv(VALUE_YEAR_CSV, index=False)
    pd.DataFrame(league_rows).to_csv(VALUE_LEAGUE_CSV, index=False)
    selected_combined = pd.concat(selected_all, ignore_index=True) if selected_all else pd.DataFrame()
    checks = leakage_checks(v3_raw, v4_raw, groups, selected_combined if len(selected_combined) else None)
    checks.to_csv(LEAKAGE_CSV, index=False)

    if checks["status"].ne("pass").any():
        decision = "ou_v4_1_rejected_bug_or_leakage"
    elif gate.empty:
        decision = "ou_v4_1_rejected_no_predictive_gain"
    elif not value_rows or pd.DataFrame(value_rows)["profit"].max() <= 0:
        decision = "ou_v4_1_predictive_only"
    else:
        value_df = pd.DataFrame(value_rows)
        best = value_df.sort_values("profit", ascending=False).iloc[0]
        if best["profit"] > 0 and best["best_year_share"] < 0.6 and best["best_league_share"] < 0.6:
            decision = "ou_v4_1_forward_paper_candidate"
        else:
            decision = "ou_v4_1_value_research_candidate"

    REPORT_MD.write_text(
        "\n".join(
            [
                "# V4.1 Understat O/U 2.5 Predictive Audit",
                "",
                f"Decision: `{decision}`",
                "",
                "No broad model search, threshold optimization, FBref data, locked v3 1X2 candidate change, closing-odds selection, or current-match Understat feature use was run. No confirmed edge is claimed.",
                "",
                "## Predictive Summary",
                md_table(summary[["scope", "feature_group", "model", "rows", "delta_log_loss", "delta_brier", "delta_ece", "seasons_improved_log_loss", "predictive_gate_pass"]], 40),
                "",
                "## Negative Controls",
                md_table(negatives[["scope", "feature_group", "model", "control", "rows", "delta_log_loss", "delta_brier", "delta_ece"]] if len(negatives) else negatives, 40),
                "",
                "## Value Diagnostics",
                md_table(pd.DataFrame(value_rows), 40),
                "",
                "## Leakage Checks",
                md_table(checks, 40),
                "",
            ]
        ),
        encoding="utf-8",
    )
    DECISION_MD.write_text(
        "\n".join(["# V4.1 Understat O/U 2.5 Decision", "", f"Decision: `{decision}`", "", "No confirmed edge is claimed.", ""]),
        encoding="utf-8",
    )
    print(
        {
            "decision": decision,
            "rows": int(len(df)),
            "gate_passes": int(len(gate)),
            "failed_checks": int(checks["status"].ne("pass").sum()),
            "best_delta_log_loss": round(float(summary["delta_log_loss"].min()), 8) if len(summary) else None,
            "best_value_profit": round(float(pd.DataFrame(value_rows)["profit"].max()), 2) if value_rows else 0.0,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
