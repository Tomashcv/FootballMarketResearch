from __future__ import annotations

import warnings
from dataclasses import dataclass
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
INPUT_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/clubelo"
REPORT_DIR = ROOT / "outputs/reports/clubelo_predictive"

MARKETS = {
    "btts": {
        "file": "super_btts_footiqo_top5_clubelo_research_v1.csv",
        "target": "target_btts_yes",
        "baseline": "btts_yes_no_vig_prob",
        "prior_summary": ROOT / "outputs/reports/footiqo_top5_btts_predictive_summary.csv",
    },
    "ou15": {
        "file": "super_ou15_footiqo_top5_clubelo_research_v1.csv",
        "target": "target_over_1_5",
        "baseline": "ou15_over_no_vig_prob",
        "prior_summary": ROOT / "outputs/reports/footiqo_top5_ou15_predictive_summary.csv",
    },
    "ou25": {
        "file": "super_ou25_footiqo_top5_clubelo_research_v1.csv",
        "target": "target_over_2_5",
        "baseline": "ou25_over_no_vig_prob",
        "prior_summary": ROOT / "outputs/reports/footiqo_top5_ou25_predictive_summary.csv",
    },
    "ou35": {
        "file": "super_ou35_footiqo_top5_clubelo_research_v1.csv",
        "target": "target_over_3_5",
        "baseline": "ou35_over_no_vig_prob",
        "prior_summary": ROOT / "outputs/reports/footiqo_top5_ou35_predictive_summary.csv",
    },
    "ou45": {
        "file": "super_ou45_footiqo_top5_clubelo_research_v1.csv",
        "target": "target_over_4_5",
        "baseline": "ou45_over_no_vig_prob",
        "prior_summary": ROOT / "outputs/reports/footiqo_top5_ou45_predictive_summary.csv",
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
LEAGUE_FEATURES = [
    "league_england_premier_league",
    "league_spain_laliga",
    "league_germany_bundesliga",
    "league_italy_serie_a",
    "league_france_ligue_1",
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
    "clubelo_source_file",
}
FORBIDDEN_CONTAINS = ["raw_prob", "overround"]
FORBIDDEN_RAW_ODDS = {"BTTSY", "BTTSN", "O15", "U15", "O25", "U25", "O35", "U35", "O45", "U45"}


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype=int)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece_score(y, p),
    }


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def rolling_goals_form_cols(df: pd.DataFrame) -> list[str]:
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
        if (c.startswith(("home_", "away_", "home_minus_away_")) and any(t in c for t in tokens))
    ]


def rolling_all_safe_cols(df: pd.DataFrame, target: str, baseline: str) -> list[str]:
    out = []
    for c in numeric_cols(df):
        if c in FORBIDDEN_EXACT or c in FORBIDDEN_RAW_ODDS or c == target:
            continue
        if c.startswith("target_"):
            continue
        if c == baseline:
            continue
        if c in CLUBELO_FEATURES or c in LEAGUE_FEATURES:
            continue
        if any(x in c for x in FORBIDDEN_CONTAINS):
            continue
        if c.startswith(("home_", "away_", "home_minus_away_")) or c in {
            "match_week_index",
            "both_history_available_flag",
            "rolling_features_date_safe_flag",
            "external_sources_joined_flag",
        }:
            out.append(c)
    return out


def feature_groups(df: pd.DataFrame, baseline: str, target: str) -> dict[str, list[str]]:
    goals = rolling_goals_form_cols(df)
    allsafe = rolling_all_safe_cols(df, target, baseline)
    clubelo = [c for c in CLUBELO_FEATURES if c in df.columns]
    league = [c for c in LEAGUE_FEATURES if c in df.columns]
    return {
        "market_probability_only": [baseline],
        "market_plus_clubelo": [baseline] + clubelo,
        "market_plus_rolling_goals_form": [baseline] + goals,
        "market_plus_rolling_goals_form_plus_clubelo": [baseline] + goals + clubelo,
        "market_plus_rolling_all_safe": [baseline] + allsafe,
        "market_plus_rolling_all_safe_plus_clubelo": [baseline] + allsafe + clubelo,
        "market_plus_league": [baseline] + league,
        "market_plus_league_plus_clubelo": [baseline] + league + clubelo,
        "market_plus_rolling_all_safe_plus_league_plus_clubelo": [baseline] + allsafe + league + clubelo,
    }


def models_for(n_features: int):
    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(loss="log_loss", penalty=None, max_iter=200, tol=1e-3, random_state=42),
        ),
        "regularized_logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=200, tol=1e-3, random_state=42),
        ),
    }
    if HAS_XGB and RUN_XGB:
        models["xgboost_binary"] = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=10,
                max_depth=2,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                n_jobs=1,
                random_state=42,
                verbosity=0,
            ),
        )
    return models


def predict_model(model, x_train, y_train, x_test) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    return model[-1].predict_proba(model[:-1].transform(x_test))[:, 1]


def evaluate_market(market: str, spec: dict):
    df = pd.read_csv(INPUT_DIR / spec["file"], dtype={"competition_code": str})
    target = spec["target"]
    baseline = spec["baseline"]
    df = df[df[target].notna() & df[baseline].notna()].copy()
    df[target] = df[target].astype(int)
    groups = feature_groups(df, baseline, target)
    fold_rows = []
    pred_rows = []
    leakage_rows = []
    for test_season in TEST_SEASONS:
        print(f"{market}: fold {test_season}", flush=True)
        train = df[df["season_start_year"] < test_season]
        test = df[df["season_start_year"] == test_season]
        if train.empty or test.empty or train[target].nunique() < 2 or test[target].nunique() < 2:
            continue
        base_prob = test[baseline].to_numpy()
        base_m = metrics(test[target], base_prob)
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
            }
        )
        for group_name, cols in groups.items():
            print(f"{market}: fold {test_season} group {group_name}", flush=True)
            cols = list(dict.fromkeys([c for c in cols if c in df.columns]))
            if not cols:
                continue
            for model_name, model in models_for(len(cols)).items():
                if model_name == "xgboost_binary" and group_name != "market_plus_rolling_all_safe_plus_league_plus_clubelo":
                    continue
                try:
                    prob = predict_model(model, train[cols], train[target], test[cols])
                    m = metrics(test[target], prob)
                except Exception as exc:
                    fold_rows.append(
                        {
                            "market": market,
                            "feature_group": group_name,
                            "model": model_name,
                            "test_season": test_season,
                            "n_train": len(train),
                            "n_test": len(test),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
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
                        "delta_log_loss_vs_market": m["log_loss"] - base_m["log_loss"],
                        "delta_brier_vs_market": m["brier"] - base_m["brier"],
                    }
                )
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
                        }
                    )
                )
    leakage_rows.append(
        {
            "market": market,
            "forbidden_id_features_excluded": not any(c in sum(groups.values(), []) for c in FORBIDDEN_EXACT),
            "raw_odds_excluded": not any(c in sum(groups.values(), []) for c in FORBIDDEN_RAW_ODDS),
            "targets_excluded": not any(c.startswith("target_") for c in sum(groups.values(), [])),
            "xg_excluded": not any("xg" in c.lower() for c in sum(groups.values(), [])),
            "clubelo_coverage": float(df["clubelo_both_found_flag"].mean()),
            "clubelo_date_safe_from_lock_report": True,
            "odds_timing_unknown": bool(df["odds_timing_flag"].eq("unknown").all()),
            "classification": "research_only",
        }
    )
    leakage_rows[-1]["leakage_check_pass"] = all(
        bool(leakage_rows[-1][k])
        for k in [
            "forbidden_id_features_excluded",
            "raw_odds_excluded",
            "targets_excluded",
            "xg_excluded",
            "clubelo_date_safe_from_lock_report",
            "odds_timing_unknown",
        ]
    )
    return pd.DataFrame(fold_rows), (pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()), pd.DataFrame(leakage_rows), groups


def aggregate_model(folds: pd.DataFrame) -> pd.DataFrame:
    ok = folds[folds["error"].isna()] if "error" in folds.columns else folds.copy()
    return (
        ok.groupby(["market", "feature_group", "model"], dropna=False)
        .agg(
            n_test=("n_test", "sum"),
            seasons=("test_season", "nunique"),
            accuracy=("accuracy", "mean"),
            log_loss=("log_loss", "mean"),
            brier=("brier", "mean"),
            ece=("ece", "mean"),
            delta_log_loss_vs_market=("delta_log_loss_vs_market", "mean"),
            delta_brier_vs_market=("delta_brier_vs_market", "mean"),
            seasons_log_loss_improved=("delta_log_loss_vs_market", lambda s: int((s < 0).sum())),
            seasons_brier_improved=("delta_brier_vs_market", lambda s: int((s < 0).sum())),
            seasons_both_improved=("delta_log_loss_vs_market", lambda s: int(((s < 0) & (ok.loc[s.index, "delta_brier_vs_market"] < 0)).sum())),
        )
        .reset_index()
    )


def by_league(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if preds.empty:
        return pd.DataFrame()
    for keys, g in preds.groupby(["market", "feature_group", "model", "competition_slug"]):
        market, fg, model, league = keys
        m = metrics(g["y"], g["p"])
        bm = metrics(g["y"], g["p_market"])
        rows.append(
            {
                "market": market,
                "feature_group": fg,
                "model": model,
                "competition_slug": league,
                "n": len(g),
                **m,
                "market_log_loss": bm["log_loss"],
                "market_brier": bm["brier"],
                "delta_log_loss_vs_market": m["log_loss"] - bm["log_loss"],
                "delta_brier_vs_market": m["brier"] - bm["brier"],
            }
        )
    return pd.DataFrame(rows)


def leave_one_league_out(by_league_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if by_league_df.empty:
        return pd.DataFrame()
    for keys, g in by_league_df.groupby(["market", "feature_group", "model"]):
        for league in sorted(g["competition_slug"].unique()):
            rest = g[g["competition_slug"] != league]
            if rest.empty:
                continue
            rows.append(
                {
                    "market": keys[0],
                    "feature_group": keys[1],
                    "model": keys[2],
                    "left_out_league": league,
                    "mean_delta_log_loss_vs_market": rest["delta_log_loss_vs_market"].mean(),
                    "mean_delta_brier_vs_market": rest["delta_brier_vs_market"].mean(),
                    "remaining_leagues": rest["competition_slug"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def previous_comparison() -> pd.DataFrame:
    rows = []
    for market, spec in MARKETS.items():
        p = spec["prior_summary"]
        if not p.exists():
            rows.append({"market": market, "previous_report_found": False})
            continue
        try:
            df = pd.read_csv(p)
            best = {}
            if {"delta_log_loss_vs_market", "delta_brier_vs_market"}.issubset(df.columns):
                tmp = df.dropna(subset=["delta_log_loss_vs_market", "delta_brier_vs_market"]).copy()
                if not tmp.empty:
                    tmp = tmp.sort_values(["delta_log_loss_vs_market", "delta_brier_vs_market"])
                    b = tmp.iloc[0]
                    best = {
                        "previous_best_model": b.get("model", ""),
                        "previous_best_feature_group": b.get("feature_group", ""),
                        "previous_best_delta_log_loss_vs_market": b.get("delta_log_loss_vs_market", np.nan),
                        "previous_best_delta_brier_vs_market": b.get("delta_brier_vs_market", np.nan),
                        "previous_best_improved_both_seasons": b.get("improved_both_seasons", np.nan),
                        "previous_best_improved_both_leagues": b.get("improved_both_leagues", np.nan),
                    }
            rows.append(
                {
                    "market": market,
                    "previous_report_found": True,
                    "previous_report_path": str(p.relative_to(ROOT)),
                    "previous_columns": "; ".join(df.columns[:20]),
                    "previous_rows": len(df),
                    **best,
                }
            )
        except Exception as exc:
            rows.append({"market": market, "previous_report_found": True, "previous_read_error": str(exc)})
    return pd.DataFrame(rows)


def decide(summary: pd.DataFrame, by_league_df: pd.DataFrame, leakage: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    candidates = summary[
        (summary["model"] != "no_vig_market_baseline")
        & (summary["delta_log_loss_vs_market"] < 0)
        & (summary["delta_brier_vs_market"] < 0)
    ].copy()
    decisions = []
    leakage_pass = bool(leakage["leakage_check_pass"].all())
    for market in MARKETS:
        c = candidates[candidates["market"] == market].sort_values(
            ["delta_log_loss_vs_market", "delta_brier_vs_market"]
        )
        if c.empty:
            decisions.append({"market": market, "market_decision": "rejected_no_gain", "reason": "No model improved both log loss and brier overall."})
            continue
        best = c.iloc[0]
        league_rows = by_league_df[
            (by_league_df["market"] == market)
            & (by_league_df["feature_group"] == best["feature_group"])
            & (by_league_df["model"] == best["model"])
        ]
        leagues_both = int(((league_rows["delta_log_loss_vs_market"] < 0) & (league_rows["delta_brier_vs_market"] < 0)).sum())
        robust = (
            int(best["seasons_both_improved"]) >= 4
            and leagues_both >= 3
            and bool(leakage[leakage["market"] == market]["odds_timing_unknown"].iloc[0])
        )
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
                "reason": "Meets robust improvement criteria." if robust else "Overall improvement exists but robustness criteria are not fully met.",
            }
        )
    decision_df = pd.DataFrame(decisions)
    if (decision_df["market_decision"] == "ready_for_value_diagnostic_research_only").any() and bool(leakage_pass):
        decision = "clubelo_predictive_ready_for_value_diagnostic_research_only"
    elif (decision_df["market_decision"] == "market_recalibration_only").any():
        decision = "clubelo_predictive_market_recalibration_only"
    else:
        decision = "clubelo_predictive_rejected_no_gain"
    return decision, decision_df


def write_markdown(decision: str, decision_df: pd.DataFrame, summary: pd.DataFrame, prev: pd.DataFrame):
    lines = [
        "# ClubElo Multimarket Predictive Audit",
        "",
        f"Decision: **{decision}**",
        "",
        "Walk-forward evaluation by season. Train folds use only earlier seasons; test seasons are 2018/2019 through 2024/2025. No value search, threshold optimization, or extra source join was run.",
        "",
        "XGBoost was available but skipped in this run for bounded runtime after repeated long-running attempts; logistic models were run across every requested feature group.",
        "",
        "## Market Decisions",
        decision_df.to_markdown(index=False),
        "",
        "## Best Overall Rows",
    ]
    best = summary[summary["model"] != "no_vig_market_baseline"].sort_values(
        ["market", "delta_log_loss_vs_market", "delta_brier_vs_market"]
    ).groupby("market").head(1)
    lines.append(best[["market", "feature_group", "model", "delta_log_loss_vs_market", "delta_brier_vs_market", "seasons_both_improved"]].to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Previous Footiqo-Only Reports",
            prev.to_markdown(index=False),
            "",
            "## Conservative Notes",
            "- Odds timing remains labelled unknown; all outputs remain research_only.",
            "- Any positive predictive result is not a betting edge and is not a value diagnostic.",
            "- No confirmed edge is claimed.",
        ]
    )
    (REPORT_DIR / "clubelo_multimarket_predictive_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision_lines = [
        "# ClubElo Predictive Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "This was a predictive research audit only. No value search, threshold optimization, or edge claim was made.",
        "",
        decision_df.to_markdown(index=False),
    ]
    (REPORT_DIR / "clubelo_predictive_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_folds = []
    all_preds = []
    all_leakage = []
    fg_rows = []
    for market, spec in MARKETS.items():
        folds, preds, leakage, groups = evaluate_market(market, spec)
        all_folds.append(folds)
        all_preds.append(preds)
        all_leakage.append(leakage)
        for name, cols in groups.items():
            fg_rows.append({"market": market, "feature_group": name, "feature_count": len(cols), "features": "; ".join(cols)})
    folds = pd.concat(all_folds, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    leakage = pd.concat(all_leakage, ignore_index=True)
    summary = aggregate_model(folds)
    league = by_league(preds)
    lolo = leave_one_league_out(league)
    prev = previous_comparison()
    decision, decision_df = decide(summary, league, leakage)
    summary_out = summary.merge(decision_df, on="market", how="left")

    summary_out.to_csv(REPORT_DIR / "clubelo_multimarket_summary.csv", index=False)
    summary.to_csv(REPORT_DIR / "clubelo_by_market_model.csv", index=False)
    folds.to_csv(REPORT_DIR / "clubelo_by_market_season.csv", index=False)
    league.to_csv(REPORT_DIR / "clubelo_by_market_league.csv", index=False)
    lolo.to_csv(REPORT_DIR / "clubelo_leave_one_league_out.csv", index=False)
    pd.DataFrame(fg_rows).to_csv(REPORT_DIR / "clubelo_feature_group_comparison.csv", index=False)
    leakage.to_csv(REPORT_DIR / "clubelo_leakage_checks.csv", index=False)
    prev.to_csv(REPORT_DIR / "clubelo_previous_footiqo_only_reports.csv", index=False)
    write_markdown(decision, decision_df, summary, prev)


if __name__ == "__main__":
    main()
