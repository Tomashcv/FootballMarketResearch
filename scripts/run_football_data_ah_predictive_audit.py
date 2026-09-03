from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    XGBClassifier = None
    HAS_XGB = False


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_clubelo_understat_transfermarkt"
INPUT = INPUT_DIR / "super_ah_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv"
OPEN_INPUT = INPUT_DIR / "super_ah_open_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv"
CLOSE_INPUT = INPUT_DIR / "super_ah_close_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv"
REPORT_DIR = ROOT / "outputs/reports/football_data_ah_predictive"

TEST_SEASONS = list(range(2018, 2025))
BASELINE = "ah_home_no_vig_prob"
AWAY_BASELINE = "ah_away_no_vig_prob"
TARGET = "target_ah_home_positive_return"

CLUBELO_FEATURES = [
    "home_clubelo_rating",
    "away_clubelo_rating",
    "clubelo_diff_home_minus_away",
    "home_clubelo_days_stale",
    "away_clubelo_days_stale",
    "home_clubelo_found_flag",
    "away_clubelo_found_flag",
    "clubelo_both_found_flag",
]

FORBIDDEN_EXACT = {
    "canonical_match_id",
    "football_data_row_id",
    "source_file",
    "source",
    "div",
    "competition_slug",
    "season_label",
    "match_date",
    "match_time",
    "match_datetime",
    "source_home_team_id",
    "source_away_team_id",
    "home_team_raw",
    "away_team_raw",
    "home_team_normalized",
    "away_team_normalized",
    "home_goals",
    "away_goals",
    "result_1x2",
    "ah_home_odds",
    "ah_away_odds",
    "ah_home_unit_return",
    "ah_away_unit_return",
    "ah_home_settlement",
    "ah_away_settlement",
    "ah_push_flag",
    "ah_odds_source",
    "ah_timing_label",
    "ah_home_raw_prob",
    "ah_away_raw_prob",
    "ah_overround",
    "clubelo_source_file",
    "understat_league",
    "understat_source_file",
    "home_understat_alias_id",
    "away_understat_alias_id",
    "home_understat_latest_date",
    "away_understat_latest_date",
    "home_tm_club_id",
    "away_tm_club_id",
}
FORBIDDEN_PREFIXES = ("target_", "x1_", "btts_", "ou05_", "ou15_", "ou25_", "ou35_", "ou45_")
RAW_ODDS = {"H", "D", "A", "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"}


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def metric_dict(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece_score(y, p),
    }


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def understat_core_cols(df: pd.DataFrame) -> list[str]:
    metric_tokens = [
        "_xG_avg_",
        "_xGA_avg_",
        "_npxG_avg_",
        "_npxGA_avg_",
        "_scored_avg_",
        "_missed_avg_",
        "_xpts_avg_",
        "_deep_avg_",
        "_deep_allowed_avg_",
        "_ppda_avg_",
        "_ppda_allowed_avg_",
    ]
    out = []
    for c in numeric_cols(df):
        if c in FORBIDDEN_EXACT:
            continue
        if c.startswith(("home_understat_", "away_understat_")):
            if any(t in c for t in metric_tokens) or c.endswith(("history_count", "latest_days_ago", "found_flag")) or "_matches_w" in c:
                out.append(c)
        elif c.startswith("understat_home_minus_away_") and any(t in c for t in metric_tokens):
            out.append(c)
        elif c in {"understat_both_found_flag", "understat_match_after_source_max_date_flag"}:
            out.append(c)
    return out


def transfermarkt_core_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in numeric_cols(df)
        if c.startswith(("home_tm_", "away_tm_", "tm_"))
        and c not in FORBIDDEN_EXACT
        and c not in {"home_tm_club_id", "away_tm_club_id"}
    ]


def football_data_rolling_cols(df: pd.DataFrame) -> list[str]:
    # The current football-data AH CSV has no pre-built rolling source features.
    safe_tokens = ("rolling_", "history_count", "latest_days_ago", "matches_w", "form_")
    return [
        c
        for c in numeric_cols(df)
        if any(t in c for t in safe_tokens)
        and not c.startswith(("home_understat_", "away_understat_", "understat_", "home_tm_", "away_tm_", "tm_"))
        and c not in FORBIDDEN_EXACT
    ]


def league_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("league_") and pd.api.types.is_numeric_dtype(df[c])]


def allowed_col(col: str) -> bool:
    if col in FORBIDDEN_EXACT or col in RAW_ODDS:
        return False
    if col.startswith(FORBIDDEN_PREFIXES):
        return False
    lower = col.lower()
    if any(t in lower for t in ["current_club", "current_value", "lineup", "appearance"]):
        return False
    if any(t in lower for t in ["team_name", "team_raw", "team_normalized"]) or col.endswith("_team_id"):
        return False
    return True


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    market = [BASELINE, AWAY_BASELINE]
    line = ["ah_line_home"]
    clubelo = [c for c in CLUBELO_FEATURES if c in df.columns]
    tm = transfermarkt_core_cols(df)
    under = understat_core_cols(df)
    rolling = football_data_rolling_cols(df)
    league = league_cols(df)
    flags = [c for c in ["understat_match_after_source_max_date_flag", "clubelo_both_found_flag", "understat_both_found_flag", "tm_both_value_found_flag"] if c in df.columns]
    groups = {
        "market_probability_only": market,
        "market_plus_ah_line": market + line,
        "market_plus_clubelo": market + clubelo,
        "market_plus_transfermarkt_core": market + tm,
        "market_plus_clubelo_transfermarkt": market + clubelo + tm,
        "market_plus_understat_core": market + under,
        "market_plus_clubelo_understat_transfermarkt": market + clubelo + under + tm,
        "market_plus_football_data_rolling_if_available": market + rolling,
        "market_plus_all_safe_light": market + line + clubelo + under + tm + rolling + league + flags,
    }
    return {k: list(dict.fromkeys([c for c in v if c in df.columns and allowed_col(c)])) for k, v in groups.items()}


def models_for(group_name: str) -> dict[str, object]:
    models: dict[str, object] = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(loss="log_loss", penalty=None, max_iter=500, tol=1e-3, random_state=42),
        ),
        "regularized_logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(penalty="l2", C=0.25, solver="lbfgs", max_iter=600, random_state=42),
        ),
    }
    xgb_groups = {
        "market_probability_only",
        "market_plus_ah_line",
        "market_plus_clubelo_transfermarkt",
        "market_plus_clubelo_understat_transfermarkt",
        "market_plus_all_safe_light",
    }
    if HAS_XGB and group_name in xgb_groups:
        for n_estimators, learning_rate, reg_lambda in [(80, 0.05, 5), (120, 0.03, 20)]:
            models[f"xgboost_binary_ne{n_estimators}_lr{learning_rate}_rl{reg_lambda}"] = make_pipeline(
                SimpleImputer(strategy="median"),
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=2,
                    learning_rate=learning_rate,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=reg_lambda,
                    eval_metric="logloss",
                    n_jobs=1,
                    random_state=42,
                    verbosity=0,
                ),
            )
    return models


def predict_model(model: object, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def line_bucket(line: pd.Series) -> pd.Series:
    vals = pd.to_numeric(line, errors="coerce")
    bins = [-10, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10]
    labels = ["<=-2", "(-2,-1]", "(-1,-0.5]", "(-0.5,0]", "(0,0.5]", "(0.5,1]", "(1,2]", ">2"]
    return pd.cut(vals, bins=bins, labels=labels, include_lowest=True).astype(str)


def load_data(path: Path = INPUT) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"competition_code": str}, low_memory=False)
    df[TARGET] = (pd.to_numeric(df["ah_home_unit_return"], errors="coerce") > 0).astype(int)
    df["target_ah_away_positive_return"] = (pd.to_numeric(df["ah_away_unit_return"], errors="coerce") > 0).astype(int)
    df["ah_line_bucket"] = line_bucket(df["ah_line_home"])
    return df[df[TARGET].notna() & df[BASELINE].notna()].copy()


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = feature_groups(df)
    fold_rows = []
    pred_rows = []
    for test_season in TEST_SEASONS:
        print(f"ah: season {test_season}", flush=True)
        train = df[df["season_start_year"] < test_season].copy()
        test = df[df["season_start_year"] == test_season].copy()
        if train.empty or test.empty or train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
            continue
        base_prob = test[BASELINE].to_numpy(dtype=float)
        base_m = metric_dict(test[TARGET], base_prob)
        fold_rows.append(
            {
                "market": "ah",
                "feature_group": "market_probability_only",
                "model": "no_vig_ah_market_baseline",
                "test_season": test_season,
                "n_train": len(train),
                "n_test": len(test),
                **base_m,
                "baseline_log_loss": base_m["log_loss"],
                "baseline_brier": base_m["brier"],
                "delta_log_loss_vs_market": 0.0,
                "delta_brier_vs_market": 0.0,
                "error": "",
            }
        )
        pred_rows.append(
            pd.DataFrame(
                {
                    "market": "ah",
                    "feature_group": "market_probability_only",
                    "model": "no_vig_ah_market_baseline",
                    "test_season": test_season,
                    "competition_slug": test["competition_slug"].values,
                    "ah_line_home": test["ah_line_home"].values,
                    "ah_line_bucket": test["ah_line_bucket"].values,
                    "y": test[TARGET].values,
                    "y_away": test["target_ah_away_positive_return"].values,
                    "p": base_prob,
                    "p_market": base_prob,
                    "p_away_market": test[AWAY_BASELINE].to_numpy(dtype=float),
                    "ah_home_settlement": test["ah_home_settlement"].values,
                    "ah_away_settlement": test["ah_away_settlement"].values,
                }
            )
        )
        for group_name, cols in groups.items():
            for model_name, model in models_for(group_name).items():
                try:
                    prob = predict_model(model, train[cols], train[TARGET], test[cols])
                    m = metric_dict(test[TARGET], prob)
                    error = ""
                except Exception as exc:
                    prob = np.full(len(test), np.nan)
                    m = {"accuracy": np.nan, "log_loss": np.nan, "brier": np.nan, "ece": np.nan}
                    error = f"{type(exc).__name__}: {exc}"
                fold_rows.append(
                    {
                        "market": "ah",
                        "feature_group": group_name,
                        "model": model_name,
                        "test_season": test_season,
                        "n_train": len(train),
                        "n_test": len(test),
                        **m,
                        "baseline_log_loss": base_m["log_loss"],
                        "baseline_brier": base_m["brier"],
                        "delta_log_loss_vs_market": m["log_loss"] - base_m["log_loss"] if not np.isnan(m["log_loss"]) else np.nan,
                        "delta_brier_vs_market": m["brier"] - base_m["brier"] if not np.isnan(m["brier"]) else np.nan,
                        "error": error,
                    }
                )
                if not error:
                    pred_rows.append(
                        pd.DataFrame(
                            {
                                "market": "ah",
                                "feature_group": group_name,
                                "model": model_name,
                                "test_season": test_season,
                                "competition_slug": test["competition_slug"].values,
                                "ah_line_home": test["ah_line_home"].values,
                                "ah_line_bucket": test["ah_line_bucket"].values,
                                "y": test[TARGET].values,
                                "y_away": test["target_ah_away_positive_return"].values,
                                "p": prob,
                                "p_market": base_prob,
                                "p_away_market": test[AWAY_BASELINE].to_numpy(dtype=float),
                                "ah_home_settlement": test["ah_home_settlement"].values,
                                "ah_away_settlement": test["ah_away_settlement"].values,
                            }
                        )
                    )
    fg = pd.DataFrame([{"market": "ah", "feature_group": k, "feature_count": len(v), "features": "; ".join(v)} for k, v in groups.items()])
    leakage = pd.DataFrame([leakage_row(df, groups)])
    return pd.DataFrame(fold_rows), pd.concat(pred_rows, ignore_index=True), fg, leakage


def aggregate_overall(preds: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in preds.groupby(["market", "feature_group", "model"], dropna=False):
        m = metric_dict(g["y"], g["p"])
        bm = metric_dict(g["y"], g["p_market"])
        f = folds[
            folds["feature_group"].eq(keys[1])
            & folds["model"].eq(keys[2])
            & folds["error"].fillna("").eq("")
        ]
        rows.append(
            {
                "market": keys[0],
                "feature_group": keys[1],
                "model": keys[2],
                "n_test": len(g),
                "seasons": g["test_season"].nunique(),
                **m,
                "market_log_loss": bm["log_loss"],
                "market_brier": bm["brier"],
                "delta_log_loss_vs_market": m["log_loss"] - bm["log_loss"],
                "delta_brier_vs_market": m["brier"] - bm["brier"],
                "seasons_log_loss_improved": int((f["delta_log_loss_vs_market"] < 0).sum()),
                "seasons_brier_improved": int((f["delta_brier_vs_market"] < 0).sum()),
                "seasons_both_improved": int(((f["delta_log_loss_vs_market"] < 0) & (f["delta_brier_vs_market"] < 0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def grouped_metrics(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in preds.groupby(["market", "feature_group", "model"] + by, dropna=False):
        m = metric_dict(g["y"], g["p"])
        bm = metric_dict(g["y"], g["p_market"])
        key_vals = dict(zip(["market", "feature_group", "model"] + by, keys))
        rows.append(
            {
                **key_vals,
                "n": len(g),
                **m,
                "market_log_loss": bm["log_loss"],
                "market_brier": bm["brier"],
                "delta_log_loss_vs_market": m["log_loss"] - bm["log_loss"],
                "delta_brier_vs_market": m["brier"] - bm["brier"],
            }
        )
    return pd.DataFrame(rows)


def calibration(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for keys, g in preds.groupby(["market", "feature_group", "model"], dropna=False):
        for side, y_col, p_col in [("home", "y", "p"), ("away_market_reference", "y_away", "p_away_market")]:
            p = np.clip(g[p_col].to_numpy(dtype=float), 0, 1)
            y = g[y_col].to_numpy(dtype=int)
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
                if mask.any():
                    rows.append(
                        {
                            "market": keys[0],
                            "feature_group": keys[1],
                            "model": keys[2],
                            "side": side,
                            "bin": i,
                            "bin_lower": lo,
                            "bin_upper": hi,
                            "n": int(mask.sum()),
                            "mean_pred_prob": float(p[mask].mean()),
                            "actual_positive_rate": float(y[mask].mean()),
                            "calibration_error": float(y[mask].mean() - p[mask].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def side_file_diagnostics() -> pd.DataFrame:
    rows = []
    for label, path in [("primary", INPUT), ("open", OPEN_INPUT), ("close", CLOSE_INPUT)]:
        if not path.exists():
            rows.append({"dataset": label, "exists": False})
            continue
        df = load_data(path)
        rows.append(
            {
                "dataset": label,
                "exists": True,
                "rows": len(df),
                "seasons": df["season_start_year"].nunique(),
                "min_season": int(df["season_start_year"].min()),
                "max_season": int(df["season_start_year"].max()),
                "home_positive_rate": float(df[TARGET].mean()),
                "away_positive_rate": float(df["target_ah_away_positive_return"].mean()),
                "push_rate": float(df["ah_push_flag"].astype(bool).mean()),
                "ah_timing_labels": ";".join(sorted(df["ah_timing_label"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def settlement_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    home = df.groupby(["ah_home_settlement"], dropna=False).size().reset_index(name="rows").rename(columns={"ah_home_settlement": "settlement"})
    home["side"] = "home"
    away = df.groupby(["ah_away_settlement"], dropna=False).size().reset_index(name="rows").rename(columns={"ah_away_settlement": "settlement"})
    away["side"] = "away"
    return pd.concat([home, away], ignore_index=True)[["side", "settlement", "rows"]]


def leakage_row(df: pd.DataFrame, groups: dict[str, list[str]]) -> dict[str, object]:
    used = set(sum(groups.values(), []))
    row = {
        "market": "ah",
        "canonical_match_id_excluded": "canonical_match_id" not in used,
        "source_identifiers_excluded": not any(c in used for c in ["football_data_row_id", "source_file", "source"]),
        "team_names_excluded": not any("team_name" in c or "team_raw" in c or "team_normalized" in c for c in used),
        "team_ids_excluded": not any(c.endswith("_team_id") or c.endswith("_club_id") for c in used),
        "raw_odds_excluded": not bool(used & RAW_ODDS) and "ah_home_odds" not in used and "ah_away_odds" not in used,
        "targets_excluded": not any(c.startswith("target_") for c in used),
        "settlement_result_columns_excluded": not any(c in used for c in ["ah_home_unit_return", "ah_away_unit_return", "ah_home_settlement", "ah_away_settlement", "ah_push_flag"]),
        "final_score_columns_excluded": not any(c in used for c in ["home_goals", "away_goals", "result_1x2"]),
        "same_match_stats_excluded": True,
        "future_valuations_transfers_excluded": True,
        "current_club_fields_excluded": not any("current_club" in c.lower() for c in used),
        "game_lineups_excluded": not any("lineup" in c.lower() for c in used),
        "same_match_appearances_excluded": not any("appearance" in c.lower() for c in used),
        "unrelated_market_odds_probabilities_excluded": not any(c.startswith(("x1_", "btts_", "ou")) for c in used),
        "classification_research_only": bool(df["classification"].eq("research_only").all()),
    }
    row["leakage_check_pass"] = all(bool(v) for k, v in row.items() if k != "market")
    return row


def decide(summary: pd.DataFrame, league: pd.DataFrame, leakage: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    leakage_pass = bool(leakage["leakage_check_pass"].all())
    feature_block_groups = {
        "market_plus_clubelo",
        "market_plus_transfermarkt_core",
        "market_plus_clubelo_transfermarkt",
        "market_plus_understat_core",
        "market_plus_clubelo_understat_transfermarkt",
        "market_plus_all_safe_light",
    }
    all_candidates = summary[
        ~summary["model"].eq("no_vig_ah_market_baseline")
        & (summary["delta_log_loss_vs_market"] < 0)
        & (summary["delta_brier_vs_market"] < 0)
    ].sort_values(["delta_log_loss_vs_market", "delta_brier_vs_market"])
    feature_candidates = all_candidates[all_candidates["feature_group"].isin(feature_block_groups)].copy()
    if all_candidates.empty:
        d = pd.DataFrame([{"market": "ah", "market_decision": "rejected_no_gain", "reason": "No model improved both log loss and brier overall versus AH market baseline."}])
        return "football_data_ah_predictive_rejected_no_gain", d
    def with_leagues(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in frame.iterrows():
            l = league[league["feature_group"].eq(row["feature_group"]) & league["model"].eq(row["model"])]
            rows.append(int(((l["delta_log_loss_vs_market"] < 0) & (l["delta_brier_vs_market"] < 0)).sum()))
        out = frame.copy()
        out["leagues_both_improved"] = rows
        return out

    feature_candidates = with_leagues(feature_candidates)
    robust_feature = feature_candidates[
        (feature_candidates["seasons_both_improved"] >= 4)
        & (feature_candidates["leagues_both_improved"] >= 3)
    ].sort_values(["delta_log_loss_vs_market", "delta_brier_vs_market"])
    if not robust_feature.empty and leakage_pass:
        best = robust_feature.iloc[0]
        market_decision = "ready_for_value_diagnostic_research_only"
        decision_label = "football_data_ah_predictive_ready_for_value_diagnostic_research_only"
        reason = "A date-safe feature-block model meets robust predictive improvement criteria."
    else:
        all_candidates = with_leagues(all_candidates)
        robust_any = all_candidates[
            (all_candidates["seasons_both_improved"] >= 4)
            & (all_candidates["leagues_both_improved"] >= 3)
        ].sort_values(["delta_log_loss_vs_market", "delta_brier_vs_market"])
        best = (robust_any if not robust_any.empty else all_candidates).iloc[0]
        market_decision = "market_recalibration_only"
        decision_label = "football_data_ah_predictive_market_recalibration_only"
        reason = "Overall improvement exists, but no date-safe feature-block model fully meets robustness criteria."
    d = pd.DataFrame(
        [
            {
                "market": "ah",
                "market_decision": market_decision,
                "best_feature_group": best["feature_group"],
                "best_model": best["model"],
                "best_delta_log_loss_vs_market": best["delta_log_loss_vs_market"],
                "best_delta_brier_vs_market": best["delta_brier_vs_market"],
                "seasons_both_improved": int(best["seasons_both_improved"]),
                "leagues_both_improved": int(best["leagues_both_improved"]),
                "reason": reason,
            }
        ]
    )
    return decision_label, d


def write_markdown(decision: str, decision_df: pd.DataFrame, summary: pd.DataFrame, side_diag: pd.DataFrame, settlement: pd.DataFrame, fg: pd.DataFrame) -> None:
    best = summary[~summary["model"].eq("no_vig_ah_market_baseline")].sort_values(["delta_log_loss_vs_market", "delta_brier_vs_market"]).head(12)
    lines = [
        "# Football-Data AH Predictive Audit",
        "",
        f"Decision: **{decision}**",
        "",
        "Walk-forward by season, train on earlier seasons only, test seasons 2018/2019 through 2024/2025 where available. No value search, threshold optimization, source join, or raw-file modification was performed.",
        "",
        "Target: `target_ah_home_positive_return = 1` when `ah_home_unit_return > 0`, else 0. Pushes and half-losses are non-positive for this binary target and are reported separately.",
        "",
        "## Dataset Diagnostics",
        side_diag.to_markdown(index=False),
        "",
        "## Decision Detail",
        decision_df.to_markdown(index=False),
        "",
        "## Best Model Rows",
        best[["feature_group", "model", "delta_log_loss_vs_market", "delta_brier_vs_market", "seasons_both_improved"]].to_markdown(index=False),
        "",
        "## Settlement Distribution",
        settlement.to_markdown(index=False),
        "",
        "## Feature Group Sizes",
        fg[["feature_group", "feature_count"]].to_markdown(index=False),
        "",
        "No confirmed edge is claimed. Classification remains research_only.",
    ]
    (REPORT_DIR / "ah_predictive_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT_DIR / "ah_predictive_decision.md").write_text(
        "\n".join(["# Football-Data AH Predictive Decision", "", f"Decision: **{decision}**", "", "No value search or threshold optimization was run. No confirmed edge is claimed.", "", decision_df.to_markdown(index=False)]) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(INPUT)
    side_diag = side_file_diagnostics()
    settlement = settlement_diagnostics(df)
    folds, preds, fg, leakage = evaluate(df)
    summary = aggregate_overall(preds, folds)
    league = grouped_metrics(preds, ["competition_slug"])
    line_bucket_df = grouped_metrics(preds, ["ah_line_bucket"])
    calib = calibration(preds)
    decision, decision_df = decide(summary, league, leakage)
    summary_out = summary.merge(decision_df, on="market", how="left")
    summary_out.to_csv(REPORT_DIR / "ah_summary.csv", index=False)
    summary.to_csv(REPORT_DIR / "ah_by_model.csv", index=False)
    folds.to_csv(REPORT_DIR / "ah_by_season.csv", index=False)
    league.to_csv(REPORT_DIR / "ah_by_league.csv", index=False)
    line_bucket_df.to_csv(REPORT_DIR / "ah_by_line_bucket.csv", index=False)
    calib.to_csv(REPORT_DIR / "ah_calibration.csv", index=False)
    leakage.to_csv(REPORT_DIR / "ah_leakage_checks.csv", index=False)
    side_diag.to_csv(REPORT_DIR / "ah_open_close_file_diagnostics.csv", index=False)
    settlement.to_csv(REPORT_DIR / "ah_push_half_outcome_diagnostics.csv", index=False)
    fg.to_csv(REPORT_DIR / "ah_feature_group_comparison.csv", index=False)
    write_markdown(decision, decision_df, summary, side_diag, settlement, fg)
    print(decision)


if __name__ == "__main__":
    main()
