from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    XGBClassifier = None
    HAS_XGB = False


RUN_XGB = False

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/clubelo_understat"
REPORT_DIR = ROOT / "outputs/reports/understat_predictive"

MARKETS = {
    "btts": {
        "file": "super_btts_footiqo_top5_clubelo_understat_research_v1.csv",
        "target": "target_btts_yes",
        "baseline": "btts_yes_no_vig_prob",
    },
    "ou25": {
        "file": "super_ou25_footiqo_top5_clubelo_understat_research_v1.csv",
        "target": "target_over_2_5",
        "baseline": "ou25_over_no_vig_prob",
    },
    "ou35": {
        "file": "super_ou35_footiqo_top5_clubelo_understat_research_v1.csv",
        "target": "target_over_3_5",
        "baseline": "ou35_over_no_vig_prob",
    },
}

TEST_SEASONS = list(range(2018, 2025))

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
    "source",
    "source_match_id",
    "source_league_slug",
    "primary_source",
    "match_datetime",
    "season_label",
    "competition_slug",
    "league_name",
    "home_team_normalized",
    "away_team_normalized",
    "understat_league",
    "understat_source_file",
    "home_understat_alias_id",
    "away_understat_alias_id",
    "home_understat_latest_date",
    "away_understat_latest_date",
    "clubelo_source_file",
}
RAW_ODDS = {"BTTSY", "BTTSN", "O15", "U15", "O25", "U25", "O35", "U35", "O45", "U45"}
UNRELATED_PROB_TOKENS = [
    "btts_",
    "ou15_",
    "ou25_",
    "ou35_",
    "ou45_",
]


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


def footiqo_goals_form_cols(df: pd.DataFrame) -> list[str]:
    tokens = [
        "history_count",
        "latest_days_ago",
        "matches_w",
        "goals_for",
        "goals_against",
        "total_goals_in_matches",
        "btts_rate",
        "over_1_5_rate",
        "over_2_5_rate",
        "over_3_5_rate",
        "over_4_5_rate",
        "points_per_match",
        "win_rate",
        "draw_rate",
        "loss_rate",
    ]
    return [
        c
        for c in numeric_cols(df)
        if c.startswith(("home_", "away_", "home_minus_away_"))
        and not c.startswith(("home_understat_", "away_understat_", "understat_"))
        and not c.startswith(("home_clubelo_", "away_clubelo_", "clubelo_"))
        and any(t in c for t in tokens)
    ]


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


def league_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("league_") and pd.api.types.is_numeric_dtype(df[c])]


def allowed_base_col(col: str, target: str, baseline: str) -> bool:
    if col in FORBIDDEN_EXACT or col in RAW_ODDS or col == target:
        return False
    if col.startswith("target_"):
        return False
    if col == baseline:
        return False
    if "raw_prob" in col or "overround" in col:
        return False
    for token in UNRELATED_PROB_TOKENS:
        if col.startswith(token) and col != baseline:
            return False
    return True


def feature_groups(df: pd.DataFrame, target: str, baseline: str) -> dict[str, list[str]]:
    clubelo = [c for c in CLUBELO_FEATURES if c in df.columns]
    under = understat_core_cols(df)
    goals = footiqo_goals_form_cols(df)
    league = league_cols(df)
    light_flags = [
        c
        for c in [
            "rolling_features_date_safe_flag",
            "external_sources_joined_flag",
            "both_history_available_flag",
            "match_week_index",
        ]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    groups = {
        "market_probability_only": [baseline],
        "market_plus_clubelo": [baseline] + clubelo,
        "market_plus_understat_core": [baseline] + under,
        "market_plus_understat_core_plus_clubelo": [baseline] + under + clubelo,
        "market_plus_footiqo_rolling_goals_form": [baseline] + goals,
        "market_plus_footiqo_rolling_goals_form_plus_understat_core": [baseline] + goals + under,
        "market_plus_footiqo_rolling_goals_form_plus_understat_core_plus_clubelo": [baseline] + goals + under + clubelo,
        "market_plus_all_safe_light": [baseline] + goals + under + clubelo + league + light_flags,
    }
    return {k: list(dict.fromkeys([c for c in v if c in df.columns and allowed_base_col(c, target, baseline) or c == baseline])) for k, v in groups.items()}


def models_for(group_name: str) -> dict[str, object]:
    models: dict[str, object] = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(loss="log_loss", penalty=None, max_iter=300, tol=1e-3, random_state=42),
        ),
        "regularized_logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=300, tol=1e-3, random_state=42),
        ),
    }
    xgb_groups = {
        "market_plus_understat_core",
        "market_plus_understat_core_plus_clubelo",
        "market_plus_footiqo_rolling_goals_form_plus_understat_core",
    }
    if HAS_XGB and RUN_XGB and group_name in xgb_groups:
        models["xgboost_binary"] = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=80,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=5,
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


def evaluate_market(market: str, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(INPUT_DIR / spec["file"], dtype={"competition_code": str})
    target = spec["target"]
    baseline = spec["baseline"]
    df = df[df[target].notna() & df[baseline].notna()].copy()
    df[target] = df[target].astype(int)
    groups = feature_groups(df, target, baseline)
    fold_rows = []
    pred_rows = []
    for test_season in TEST_SEASONS:
        print(f"{market}: season {test_season}", flush=True)
        train = df[df["season_start_year"] < test_season].copy()
        test = df[df["season_start_year"] == test_season].copy()
        if train.empty or test.empty or train[target].nunique() < 2 or test[target].nunique() < 2:
            continue
        base_prob = test[baseline].to_numpy()
        base_m = metric_dict(test[target], base_prob)
        fold_rows.append(
            {
                "market": market,
                "feature_group": "market_probability_only",
                "model": "no_vig_market_baseline",
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
                    "market": market,
                    "feature_group": "market_probability_only",
                    "model": "no_vig_market_baseline",
                    "test_season": test_season,
                    "competition_slug": test["competition_slug"].values,
                    "y": test[target].values,
                    "p": base_prob,
                    "p_market": base_prob,
                    "understat_stale": test["understat_match_after_source_max_date_flag"].astype(bool).values,
                }
            )
        )
        for group_name, cols in groups.items():
            for model_name, model in models_for(group_name).items():
                try:
                    prob = predict_model(model, train[cols], train[target], test[cols])
                    m = metric_dict(test[target], prob)
                    error = ""
                except Exception as exc:
                    m = {"accuracy": np.nan, "log_loss": np.nan, "brier": np.nan, "ece": np.nan}
                    prob = np.full(len(test), np.nan)
                    error = f"{type(exc).__name__}: {exc}"
                fold_rows.append(
                    {
                        "market": market,
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
                                "market": market,
                                "feature_group": group_name,
                                "model": model_name,
                                "test_season": test_season,
                                "competition_slug": test["competition_slug"].values,
                                "y": test[target].values,
                                "p": prob,
                                "p_market": base_prob,
                                "understat_stale": test["understat_match_after_source_max_date_flag"].astype(bool).values,
                            }
                        )
                    )
    fg_rows = [
        {"market": market, "feature_group": name, "feature_count": len(cols), "features": "; ".join(cols)}
        for name, cols in groups.items()
    ]
    leakage = leakage_row(market, df, groups)
    return pd.DataFrame(fold_rows), pd.concat(pred_rows, ignore_index=True), pd.DataFrame(fg_rows), pd.DataFrame([leakage])


def leakage_row(market: str, df: pd.DataFrame, groups: dict[str, list[str]]) -> dict[str, object]:
    used = set(sum(groups.values(), []))
    forbidden_exact_excluded = not bool(used & FORBIDDEN_EXACT)
    raw_odds_excluded = not bool(used & RAW_ODDS)
    targets_excluded = not any(c.startswith("target_") for c in used)
    source_ids_excluded = not any("source_match_id" in c or c.endswith("_alias_id") for c in used)
    team_names_excluded = not any("team_normalized" in c or "team_raw" in c or "team_name" in c for c in used)
    current_fixture_xg_excluded = not any(c in {"xG", "xGA", "npxG", "npxGA"} for c in used)
    rejected_alias_not_used = not (
        df.get("home_understat_alias_id", pd.Series(dtype=float)).eq(384).any()
        or df.get("away_understat_alias_id", pd.Series(dtype=float)).eq(384).any()
    )
    latest = df[["match_datetime", "home_understat_latest_date", "away_understat_latest_date"]].copy()
    latest["match_date"] = pd.to_datetime(latest["match_datetime"], errors="coerce").dt.floor("D")
    latest["home_understat_latest_date"] = pd.to_datetime(latest["home_understat_latest_date"], errors="coerce")
    latest["away_understat_latest_date"] = pd.to_datetime(latest["away_understat_latest_date"], errors="coerce")
    no_future = not (
        (latest["home_understat_latest_date"].notna() & (latest["home_understat_latest_date"] >= latest["match_date"])).any()
        or (latest["away_understat_latest_date"].notna() & (latest["away_understat_latest_date"] >= latest["match_date"])).any()
    )
    row = {
        "market": market,
        "forbidden_id_features_excluded": "canonical_match_id" not in used,
        "source_identifiers_excluded": source_ids_excluded,
        "team_names_excluded": team_names_excluded,
        "raw_odds_excluded": raw_odds_excluded,
        "targets_excluded": targets_excluded,
        "forbidden_exact_columns_excluded": forbidden_exact_excluded,
        "current_fixture_xg_excluded": current_fixture_xg_excluded,
        "rejected_manual_aliases_excluded": rejected_alias_not_used,
        "no_future_understat_rows_used": no_future,
        "odds_timing_unknown": bool(df["odds_timing_flag"].eq("unknown").all()),
        "understat_staleness_documented": "understat_match_after_source_max_date_flag" in df.columns,
        "classification": "research_only",
    }
    row["leakage_check_pass"] = all(bool(v) for k, v in row.items() if k not in {"market", "classification"})
    return row


def aggregate_overall(preds: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in preds.groupby(["market", "feature_group", "model"], dropna=False):
        market, fg, model = keys
        m = metric_dict(g["y"], g["p"])
        bm = metric_dict(g["y"], g["p_market"])
        f = folds[
            (folds["market"].eq(market))
            & (folds["feature_group"].eq(fg))
            & (folds["model"].eq(model))
            & folds["error"].fillna("").eq("")
        ]
        rows.append(
            {
                "market": market,
                "feature_group": fg,
                "model": model,
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


def by_league(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in preds.groupby(["market", "feature_group", "model", "competition_slug"], dropna=False):
        m = metric_dict(g["y"], g["p"])
        bm = metric_dict(g["y"], g["p_market"])
        rows.append(
            {
                "market": keys[0],
                "feature_group": keys[1],
                "model": keys[2],
                "competition_slug": keys[3],
                "n": len(g),
                **m,
                "market_log_loss": bm["log_loss"],
                "market_brier": bm["brier"],
                "delta_log_loss_vs_market": m["log_loss"] - bm["log_loss"],
                "delta_brier_vs_market": m["brier"] - bm["brier"],
            }
        )
    return pd.DataFrame(rows)


def staleness_diagnostics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in preds.groupby(["market", "feature_group", "model", "understat_stale"], dropna=False):
        m = metric_dict(g["y"], g["p"])
        bm = metric_dict(g["y"], g["p_market"])
        rows.append(
            {
                "market": keys[0],
                "feature_group": keys[1],
                "model": keys[2],
                "understat_stale": bool(keys[3]),
                "n": len(g),
                **m,
                "market_log_loss": bm["log_loss"],
                "market_brier": bm["brier"],
                "delta_log_loss_vs_market": m["log_loss"] - bm["log_loss"],
                "delta_brier_vs_market": m["brier"] - bm["brier"],
            }
        )
    return pd.DataFrame(rows)


def leave_one_league_out(league: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in league.groupby(["market", "feature_group", "model"], dropna=False):
        for league_name in sorted(g["competition_slug"].dropna().unique()):
            rest = g[g["competition_slug"].ne(league_name)]
            if rest.empty:
                continue
            rows.append(
                {
                    "market": keys[0],
                    "feature_group": keys[1],
                    "model": keys[2],
                    "left_out_league": league_name,
                    "remaining_leagues": rest["competition_slug"].nunique(),
                    "mean_delta_log_loss_vs_market": rest["delta_log_loss_vs_market"].mean(),
                    "mean_delta_brier_vs_market": rest["delta_brier_vs_market"].mean(),
                    "all_remaining_leagues_improve_both": bool(
                        ((rest["delta_log_loss_vs_market"] < 0) & (rest["delta_brier_vs_market"] < 0)).all()
                    ),
                }
            )
    return pd.DataFrame(rows)


def previous_reports() -> pd.DataFrame:
    candidates = [
        ("clubelo_only", ROOT / "outputs/reports/clubelo_predictive/clubelo_by_market_model.csv"),
        ("clubelo_only", ROOT / "outputs/reports/clubelo_predictive/clubelo_multimarket_summary.csv"),
        ("footiqo_only_btts", ROOT / "outputs/reports/footiqo_top5_btts_predictive_summary.csv"),
        ("footiqo_only_ou25", ROOT / "outputs/reports/footiqo_top5_ou25_predictive_summary.csv"),
        ("footiqo_only_ou35", ROOT / "outputs/reports/footiqo_top5_ou35_predictive_summary.csv"),
    ]
    rows = []
    for kind, p in candidates:
        if not p.exists():
            rows.append({"comparison_type": kind, "report": str(p.relative_to(ROOT)), "found": False})
            continue
        df = pd.read_csv(p)
        best = {}
        if {"delta_log_loss_vs_market", "delta_brier_vs_market"}.issubset(df.columns):
            tmp = df.dropna(subset=["delta_log_loss_vs_market", "delta_brier_vs_market"]).copy()
            if not tmp.empty:
                b = tmp.sort_values(["delta_log_loss_vs_market", "delta_brier_vs_market"]).iloc[0]
                best = {
                    "best_delta_log_loss_vs_market": b.get("delta_log_loss_vs_market"),
                    "best_delta_brier_vs_market": b.get("delta_brier_vs_market"),
                    "best_model": b.get("model", ""),
                    "best_feature_group": b.get("feature_group", ""),
                }
        rows.append(
            {
                "comparison_type": kind,
                "report": str(p.relative_to(ROOT)),
                "found": True,
                "rows": len(df),
                "columns": "; ".join(df.columns[:12]),
                **best,
            }
        )
    return pd.DataFrame(rows)


def decide(summary: pd.DataFrame, league: pd.DataFrame, leakage: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    decisions = []
    leakage_pass = bool(leakage["leakage_check_pass"].all())
    for market in MARKETS:
        c = summary[
            (summary["market"].eq(market))
            & ~summary["model"].eq("no_vig_market_baseline")
            & (summary["delta_log_loss_vs_market"] < 0)
            & (summary["delta_brier_vs_market"] < 0)
        ].sort_values(["delta_log_loss_vs_market", "delta_brier_vs_market"])
        if c.empty:
            decisions.append({"market": market, "market_decision": "rejected_no_gain", "reason": "No model improved both log loss and brier overall."})
            continue
        best = c.iloc[0]
        l = league[
            league["market"].eq(market)
            & league["feature_group"].eq(best["feature_group"])
            & league["model"].eq(best["model"])
        ]
        leagues_both = int(((l["delta_log_loss_vs_market"] < 0) & (l["delta_brier_vs_market"] < 0)).sum())
        robust = int(best["seasons_both_improved"]) >= 4 and leagues_both >= 3 and leakage_pass
        decisions.append(
            {
                "market": market,
                "market_decision": "ready_for_value_diagnostic_research_only" if robust else "market_recalibration_only",
                "best_feature_group": best["feature_group"],
                "best_model": best["model"],
                "best_delta_log_loss_vs_market": best["delta_log_loss_vs_market"],
                "best_delta_brier_vs_market": best["delta_brier_vs_market"],
                "seasons_both_improved": int(best["seasons_both_improved"]),
                "leagues_both_improved": leagues_both,
                "reason": "Meets robust predictive improvement criteria." if robust else "Overall improvement exists but robustness criteria are not fully met.",
            }
        )
    ddf = pd.DataFrame(decisions)
    if (ddf["market_decision"] == "ready_for_value_diagnostic_research_only").any() and leakage_pass:
        decision = "understat_predictive_ready_for_value_diagnostic_research_only"
    elif (ddf["market_decision"] == "market_recalibration_only").any():
        decision = "understat_predictive_market_recalibration_only"
    else:
        decision = "understat_predictive_rejected_no_gain"
    return decision, ddf


def write_markdown(decision: str, decision_df: pd.DataFrame, summary: pd.DataFrame, prev: pd.DataFrame) -> None:
    best = (
        summary[~summary["model"].eq("no_vig_market_baseline")]
        .sort_values(["market", "delta_log_loss_vs_market", "delta_brier_vs_market"])
        .groupby("market")
        .head(1)
    )
    lines = [
        "# Understat Controlled Predictive Audit",
        "",
        f"Decision: **{decision}**",
        "",
        "Walk-forward by season using only earlier seasons for training. Test seasons: 2018/2019 through 2024/2025. No value search, threshold optimization, or extra source join was run.",
        "",
        "XGBoost was available only as an optional bounded runtime path and was skipped in this run; logistic and regularized logistic models were run for all requested feature groups.",
        "",
        "## Market Decisions",
        decision_df.to_markdown(index=False),
        "",
        "## Best Model Rows",
        best[["market", "feature_group", "model", "delta_log_loss_vs_market", "delta_brier_vs_market", "seasons_both_improved"]].to_markdown(index=False),
        "",
        "## Previous Reports Checked",
        prev.to_markdown(index=False),
        "",
        "## Conservative Notes",
        "- Odds timing remains unknown and classification remains research_only.",
        "- Understat source staleness is documented separately; 2024/2025 rows can be stale after the archive max date.",
        "- Predictive improvement, if any, is not a betting edge.",
        "- No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "understat_controlled_predictive_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision_lines = [
        "# Understat Predictive Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "This is a research-only predictive audit. No value search, threshold optimization, or edge claim was made.",
        "",
        decision_df.to_markdown(index=False),
    ]
    (REPORT_DIR / "understat_predictive_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_folds, all_preds, all_fg, all_leakage = [], [], [], []
    for market, spec in MARKETS.items():
        folds, preds, fg, leakage = evaluate_market(market, spec)
        all_folds.append(folds)
        all_preds.append(preds)
        all_fg.append(fg)
        all_leakage.append(leakage)
    folds = pd.concat(all_folds, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    fg = pd.concat(all_fg, ignore_index=True)
    leakage = pd.concat(all_leakage, ignore_index=True)
    summary = aggregate_overall(preds, folds)
    league = by_league(preds)
    lolo = leave_one_league_out(league)
    stale = staleness_diagnostics(preds)
    prev = previous_reports()
    decision, decision_df = decide(summary, league, leakage)
    summary_out = summary.merge(decision_df, on="market", how="left")

    summary_out.to_csv(REPORT_DIR / "understat_controlled_summary.csv", index=False)
    summary.to_csv(REPORT_DIR / "understat_by_market_model.csv", index=False)
    folds.to_csv(REPORT_DIR / "understat_by_market_season.csv", index=False)
    league.to_csv(REPORT_DIR / "understat_by_market_league.csv", index=False)
    lolo.to_csv(REPORT_DIR / "understat_leave_one_league_out.csv", index=False)
    fg.to_csv(REPORT_DIR / "understat_feature_group_comparison.csv", index=False)
    stale.to_csv(REPORT_DIR / "understat_staleness_diagnostics.csv", index=False)
    leakage.to_csv(REPORT_DIR / "understat_leakage_checks.csv", index=False)
    prev.to_csv(REPORT_DIR / "understat_previous_report_comparison.csv", index=False)
    write_markdown(decision, decision_df, summary, prev)
    print(decision)


if __name__ == "__main__":
    main()
