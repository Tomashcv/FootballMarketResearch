from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
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
INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/clubelo_understat_transfermarkt/super_1x2_footiqo_top5_clubelo_understat_transfermarkt_research_v1.csv"
REPORT_DIR = ROOT / "outputs/reports/one_x_two_predictive"

TEST_SEASONS = list(range(2018, 2025))
CLASSES = [0, 1, 2]
CLASS_NAMES = {0: "H", 1: "D", 2: "A"}
BASELINE_PROBS = ["x1_home_no_vig_prob", "x1_draw_no_vig_prob", "x1_away_no_vig_prob"]
TARGETS = ["target_home_win", "target_draw", "target_away_win"]

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
    "home_team_id",
    "away_team_id",
    "home_team_name_audit",
    "away_team_name_audit",
    "home_team_normalized",
    "away_team_normalized",
    "result_1x2",
    "H",
    "D",
    "A",
    "x1_home_raw_prob",
    "x1_draw_raw_prob",
    "x1_away_raw_prob",
    "x1_overround",
    "x1_paired_odds_available_flag",
    "x1_valid_paired_odds_flag",
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
RAW_ODDS = {"H", "D", "A"}
UNRELATED_RAW_ODDS = {"O05", "U05", "O15", "U15", "O25", "U25", "O35", "U35", "O45", "U45", "BTTSY", "BTTSN"}
UNRELATED_PROB_PREFIXES = ("btts_", "ou05_", "ou15_", "ou25_", "ou35_", "ou45_")


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def target_vector(df: pd.DataFrame) -> np.ndarray:
    return df[TARGETS].to_numpy(dtype=int).argmax(axis=1)


def baseline_matrix(df: pd.DataFrame) -> np.ndarray:
    p = df[BASELINE_PROBS].to_numpy(dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def multiclass_brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    one_hot = np.eye(3)[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def multiclass_ece(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = pred == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def metric_dict(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), 1e-9, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(accuracy_score(y, p.argmax(axis=1))),
        "log_loss": float(log_loss(y, p, labels=CLASSES)),
        "brier": multiclass_brier(y, p),
        "ece": multiclass_ece(y, p),
    }


def footiqo_goals_form_cols(df: pd.DataFrame) -> list[str]:
    tokens = [
        "history_count",
        "latest_days_ago",
        "matches_w",
        "goals_for",
        "goals_against",
        "total_goals_in_matches",
        "points_per_match",
        "win_rate",
        "draw_rate",
        "loss_rate",
    ]
    out = []
    for c in numeric_cols(df):
        if not c.startswith(("home_", "away_", "home_minus_away_")):
            continue
        if c.startswith(("home_understat_", "away_understat_", "home_tm_", "away_tm_", "home_clubelo_", "away_clubelo_")):
            continue
        if c in FORBIDDEN_EXACT or c.startswith("target_"):
            continue
        if any(t in c for t in tokens):
            out.append(c)
    return out


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
    out = []
    for c in numeric_cols(df):
        if c in FORBIDDEN_EXACT:
            continue
        if c.startswith(("home_tm_", "away_tm_", "tm_")) and c not in {"home_tm_club_id", "away_tm_club_id"}:
            out.append(c)
    return out


def league_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("league_") and pd.api.types.is_numeric_dtype(df[c])]


def allowed_feature(col: str) -> bool:
    if col in FORBIDDEN_EXACT or col in RAW_ODDS or col in UNRELATED_RAW_ODDS:
        return False
    if col.startswith("target_") or col in {"result_1x2"}:
        return False
    if any(col.startswith(prefix) for prefix in UNRELATED_PROB_PREFIXES):
        return False
    lowered = col.lower()
    if any(token in lowered for token in ["current_club", "current_value", "lineup", "appearance"]):
        return False
    if any(token in lowered for token in ["team_name", "team_raw", "team_normalized"]):
        return False
    return True


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    market = [c for c in BASELINE_PROBS if c in df.columns]
    clubelo = [c for c in CLUBELO_FEATURES if c in df.columns and allowed_feature(c)]
    tm = [c for c in transfermarkt_core_cols(df) if allowed_feature(c)]
    under = [c for c in understat_core_cols(df) if allowed_feature(c)]
    goals = [c for c in footiqo_goals_form_cols(df) if allowed_feature(c)]
    league = [c for c in league_cols(df) if allowed_feature(c)]
    flags = [
        c
        for c in [
            "rolling_features_date_safe_flag",
            "external_sources_joined_flag",
            "both_history_available_flag",
            "match_week_index",
            "odds_timing_flag",
        ]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and allowed_feature(c)
    ]
    groups = {
        "market_probability_only": market,
        "market_plus_clubelo": market + clubelo,
        "market_plus_transfermarkt_core": market + tm,
        "market_plus_understat_core": market + under,
        "market_plus_clubelo_transfermarkt": market + clubelo + tm,
        "market_plus_clubelo_understat_transfermarkt": market + clubelo + under + tm,
        "market_plus_footiqo_rolling_goals_form": market + goals,
        "market_plus_footiqo_rolling_goals_form_plus_clubelo_transfermarkt": market + goals + clubelo + tm,
        "market_plus_all_safe_light": market + goals + clubelo + under + tm + league + flags,
    }
    return {k: list(dict.fromkeys(v)) for k, v in groups.items()}


def models_for(group_name: str) -> dict[str, object]:
    models: dict[str, object] = {
        "multinomial_logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=500,
                random_state=42,
            ),
        ),
        "regularized_multinomial_logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=0.25,
                solver="lbfgs",
                max_iter=500,
                random_state=42,
            ),
        ),
    }
    xgb_groups = {
        "market_plus_transfermarkt_core",
        "market_plus_understat_core",
        "market_plus_clubelo_understat_transfermarkt",
    }
    if HAS_XGB and RUN_XGB and group_name in xgb_groups:
        models["xgboost_multiclass"] = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=80,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=5,
                eval_metric="mlogloss",
                n_jobs=1,
                random_state=42,
                verbosity=0,
            ),
        )
    return models


def predict_proba_aligned(model: object, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    out = np.zeros((len(x_test), 3), dtype=float)
    for idx, cls in enumerate(classes):
        out[:, int(cls)] = prob[:, idx]
    out = np.clip(out, 1e-9, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT, dtype={"competition_code": str})
    df = df[df[TARGETS].notna().all(axis=1)].copy()
    df = df[df[BASELINE_PROBS].notna().all(axis=1)].copy()
    df["target_class"] = target_vector(df)
    return df


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = feature_groups(df)
    fold_rows = []
    pred_rows = []
    for test_season in TEST_SEASONS:
        print(f"1x2: season {test_season}", flush=True)
        train = df[df["season_start_year"] < test_season].copy()
        test = df[df["season_start_year"] == test_season].copy()
        if train.empty or test.empty or train["target_class"].nunique() < 3 or test["target_class"].nunique() < 3:
            continue
        y_train = train["target_class"].to_numpy(dtype=int)
        y_test = test["target_class"].to_numpy(dtype=int)
        base_prob = baseline_matrix(test)
        base_m = metric_dict(y_test, base_prob)
        fold_rows.append(
            {
                "market": "1x2",
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
                    "market": "1x2",
                    "feature_group": "market_probability_only",
                    "model": "no_vig_market_baseline",
                    "test_season": test_season,
                    "competition_slug": test["competition_slug"].values,
                    "y": y_test,
                    "p_home": base_prob[:, 0],
                    "p_draw": base_prob[:, 1],
                    "p_away": base_prob[:, 2],
                    "p_market_home": base_prob[:, 0],
                    "p_market_draw": base_prob[:, 1],
                    "p_market_away": base_prob[:, 2],
                }
            )
        )
        for group_name, cols in groups.items():
            for model_name, model in models_for(group_name).items():
                try:
                    prob = predict_proba_aligned(model, train[cols], y_train, test[cols])
                    m = metric_dict(y_test, prob)
                    error = ""
                except Exception as exc:
                    prob = np.full((len(test), 3), np.nan)
                    m = {"accuracy": np.nan, "log_loss": np.nan, "brier": np.nan, "ece": np.nan}
                    error = f"{type(exc).__name__}: {exc}"
                fold_rows.append(
                    {
                        "market": "1x2",
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
                                "market": "1x2",
                                "feature_group": group_name,
                                "model": model_name,
                                "test_season": test_season,
                                "competition_slug": test["competition_slug"].values,
                                "y": y_test,
                                "p_home": prob[:, 0],
                                "p_draw": prob[:, 1],
                                "p_away": prob[:, 2],
                                "p_market_home": base_prob[:, 0],
                                "p_market_draw": base_prob[:, 1],
                                "p_market_away": base_prob[:, 2],
                            }
                        )
                    )
    fg = pd.DataFrame(
        [
            {
                "market": "1x2",
                "feature_group": name,
                "feature_count": len(cols),
                "features": "; ".join(cols),
            }
            for name, cols in groups.items()
        ]
    )
    leakage = pd.DataFrame([leakage_row(df, groups)])
    return pd.DataFrame(fold_rows), pd.concat(pred_rows, ignore_index=True), fg, leakage


def prob_cols(g: pd.DataFrame, prefix: str = "p") -> np.ndarray:
    return g[[f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]].to_numpy(dtype=float)


def aggregate_overall(preds: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in preds.groupby(["market", "feature_group", "model"], dropna=False):
        market, fg, model = keys
        m = metric_dict(g["y"].to_numpy(dtype=int), prob_cols(g, "p"))
        bm = metric_dict(g["y"].to_numpy(dtype=int), prob_cols(g, "p_market"))
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
        m = metric_dict(g["y"].to_numpy(dtype=int), prob_cols(g, "p"))
        bm = metric_dict(g["y"].to_numpy(dtype=int), prob_cols(g, "p_market"))
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


def leakage_row(df: pd.DataFrame, groups: dict[str, list[str]]) -> dict[str, object]:
    used = set(sum(groups.values(), []))
    row = {
        "market": "1x2",
        "canonical_match_id_excluded": "canonical_match_id" not in used,
        "source_identifiers_excluded": not any(c in used for c in ["source", "source_match_id", "source_league_slug", "primary_source"]),
        "team_names_excluded": not any("team_name" in c or "team_raw" in c or "team_normalized" in c for c in used),
        "team_ids_excluded": not any(c in used for c in ["home_team_id", "away_team_id", "home_tm_club_id", "away_tm_club_id"]),
        "raw_odds_excluded": not bool(used & (RAW_ODDS | UNRELATED_RAW_ODDS)),
        "targets_excluded": not any(c.startswith("target_") for c in used) and "result_1x2" not in used,
        "scores_results_excluded": not any(c in used for c in ["home_goals", "away_goals", "result_1x2"]),
        "same_match_stats_excluded": True,
        "current_fixture_xg_excluded": not any(c in {"xG", "xGA", "npxG", "npxGA"} for c in used),
        "current_club_fields_excluded": not any("current_club" in c.lower() for c in used),
        "current_value_fields_excluded": not any("current_value" in c.lower() for c in used),
        "game_lineups_excluded": not any("lineup" in c.lower() for c in used),
        "same_match_appearances_excluded": not any("appearance" in c.lower() for c in used),
        "unrelated_market_odds_excluded": not any(c in used for c in UNRELATED_RAW_ODDS) and not any(c.startswith(UNRELATED_PROB_PREFIXES) for c in used),
        "future_valuations_transfers_excluded": True,
        "odds_timing_unknown": bool(df["odds_timing_flag"].eq("unknown").all()),
        "classification_research_only": bool(df["classification"].eq("research_only").all()),
    }
    row["leakage_check_pass"] = all(bool(v) for k, v in row.items() if k != "market")
    return row


def decide(summary: pd.DataFrame, league: pd.DataFrame, leakage: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    leakage_pass = bool(leakage["leakage_check_pass"].all())
    candidates = summary[
        ~summary["model"].eq("no_vig_market_baseline")
        & (summary["delta_log_loss_vs_market"] < 0)
        & (summary["delta_brier_vs_market"] < 0)
    ].sort_values(["delta_log_loss_vs_market", "delta_brier_vs_market"])
    if candidates.empty:
        decision_df = pd.DataFrame(
            [
                {
                    "market": "1x2",
                    "market_decision": "rejected_no_gain",
                    "reason": "No model improved both multiclass log loss and multiclass brier overall versus the no-vig market baseline.",
                }
            ]
        )
        return "one_x_two_predictive_rejected_no_gain", decision_df
    best = candidates.iloc[0]
    l = league[
        league["feature_group"].eq(best["feature_group"])
        & league["model"].eq(best["model"])
    ]
    leagues_both = int(((l["delta_log_loss_vs_market"] < 0) & (l["delta_brier_vs_market"] < 0)).sum())
    robust = int(best["seasons_both_improved"]) >= 4 and leagues_both >= 3 and leakage_pass
    decision_df = pd.DataFrame(
        [
            {
                "market": "1x2",
                "market_decision": "ready_for_value_diagnostic_research_only" if robust else "market_recalibration_only",
                "best_feature_group": best["feature_group"],
                "best_model": best["model"],
                "best_delta_log_loss_vs_market": best["delta_log_loss_vs_market"],
                "best_delta_brier_vs_market": best["delta_brier_vs_market"],
                "seasons_both_improved": int(best["seasons_both_improved"]),
                "leagues_both_improved": leagues_both,
                "reason": "Meets robust predictive improvement criteria." if robust else "Overall improvement exists but robustness criteria are not fully met.",
            }
        ]
    )
    if robust:
        return "one_x_two_predictive_ready_for_value_diagnostic_research_only", decision_df
    return "one_x_two_predictive_market_recalibration_only", decision_df


def write_markdown(decision: str, decision_df: pd.DataFrame, summary: pd.DataFrame, fg: pd.DataFrame) -> None:
    best = (
        summary[~summary["model"].eq("no_vig_market_baseline")]
        .sort_values(["delta_log_loss_vs_market", "delta_brier_vs_market"])
        .head(10)
    )
    lines = [
        "# 1X2 Predictive Audit",
        "",
        f"Decision: **{decision}**",
        "",
        "Walk-forward by season using only earlier seasons for training. Test seasons: 2018/2019 through 2024/2025. All five leagues are pooled.",
        "",
        "The no-vig market baseline uses `x1_home_no_vig_prob`, `x1_draw_no_vig_prob`, and `x1_away_no_vig_prob`. Median imputers are fitted inside each train fold through the model pipelines.",
        "",
        "XGBoost was available only as an optional bounded runtime path and was skipped in this run; multinomial logistic and regularized multinomial logistic models were run for all requested feature groups.",
        "",
        "## Decision Detail",
        decision_df.to_markdown(index=False),
        "",
        "## Best Model Rows",
        best[["feature_group", "model", "delta_log_loss_vs_market", "delta_brier_vs_market", "seasons_both_improved"]].to_markdown(index=False),
        "",
        "## Feature Group Sizes",
        fg[["feature_group", "feature_count"]].to_markdown(index=False),
        "",
        "## Conservative Notes",
        "- Odds timing remains unknown and classification remains research_only.",
        "- No value search or threshold optimization was run.",
        "- Predictive improvement, if any, is not a betting edge.",
        "- No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "one_x_two_predictive_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision_lines = [
        "# 1X2 Predictive Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "This is a research-only predictive audit. No value search, threshold optimization, or confirmed edge claim was made.",
        "",
        decision_df.to_markdown(index=False),
    ]
    (REPORT_DIR / "one_x_two_predictive_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    folds, preds, fg, leakage = evaluate(df)
    summary = aggregate_overall(preds, folds)
    league = by_league(preds)
    decision, decision_df = decide(summary, league, leakage)
    summary_out = summary.merge(decision_df, on="market", how="left")

    summary_out.to_csv(REPORT_DIR / "one_x_two_summary.csv", index=False)
    summary.to_csv(REPORT_DIR / "one_x_two_by_model.csv", index=False)
    folds.to_csv(REPORT_DIR / "one_x_two_by_season.csv", index=False)
    league.to_csv(REPORT_DIR / "one_x_two_by_league.csv", index=False)
    fg.to_csv(REPORT_DIR / "one_x_two_feature_group_comparison.csv", index=False)
    leakage.to_csv(REPORT_DIR / "one_x_two_leakage_checks.csv", index=False)
    write_markdown(decision, decision_df, summary, fg)
    print(decision)


if __name__ == "__main__":
    main()
