from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    XGBClassifier = None
    HAS_XGB = False


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_extended/super_1x2_football_data_top5_extended_full_features_research_v1_deduped_review_fixed.csv"
REPORT_DIR = ROOT / "outputs/reports/football_data_extended_1x2_predictive"

TEST_SEASONS = list(range(2018, 2025))
CLASSES = [0, 1, 2]
CLASS_LABELS = {0: "home", 1: "draw", 2: "away"}
TARGETS = ["target_home_win", "target_draw", "target_away_win"]
BASELINE = ["x1_home_no_vig_prob", "x1_draw_no_vig_prob", "x1_away_no_vig_prob"]

XGB_GROUPS = {
    "market_probability_only",
    "market_plus_clubelo",
    "market_plus_football_data_rolling",
    "market_plus_clubelo_plus_football_data_rolling",
    "market_plus_all_safe_light",
}
UNREGULARIZED_LOGISTIC_GROUPS = {
    "market_probability_only",
    "market_plus_clubelo",
    "market_plus_football_data_rolling",
    "market_plus_clubelo_plus_football_data_rolling",
    "market_plus_external_availability_flags",
}
XGB_SPECS = [
    ("xgboost_multiclass_ne80_d2_lr0.05_rl5", 80, 2, 0.05, 5),
    ("xgboost_multiclass_ne120_d2_lr0.03_rl20", 120, 2, 0.03, 20),
]


def target_vector(df: pd.DataFrame) -> np.ndarray:
    return df[TARGETS].to_numpy(dtype=int).argmax(axis=1)


def baseline_matrix(df: pd.DataFrame) -> np.ndarray:
    p = df[BASELINE].to_numpy(dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    return float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))


def ece(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = pred == y
    total = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(prob, dtype=float), 1e-9, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(accuracy_score(y_true, p.argmax(axis=1))),
        "log_loss": float(log_loss(y_true, p, labels=CLASSES)),
        "brier": brier(y_true, p),
        "ece": ece(y_true, p),
    }


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def allowed_feature(col: str) -> bool:
    exact_forbidden = {
        "canonical_match_id",
        "extended_canonical_match_id",
        "existing_locked_canonical_match_id",
        "football_data_row_id",
        "home_team_id",
        "away_team_id",
        "competition_type",
        "competition_code",
        "season_start_year",
        "home_goals",
        "away_goals",
        "target_home_win",
        "target_draw",
        "target_away_win",
        "x1_home_odds",
        "x1_draw_odds",
        "x1_away_odds",
        "x1_home_raw_prob",
        "x1_draw_raw_prob",
        "x1_away_raw_prob",
        "x1_overround",
        "home_tm_club_id",
        "away_tm_club_id",
    }
    if col in exact_forbidden or col == "result_1x2":
        return False
    low = col.lower()
    forbidden_tokens = [
        "source",
        "team_raw",
        "team_name",
        "team_normalized",
        "logical_match_key",
        "match_date",
        "match_time",
        "match_datetime",
        "season_label",
        "current_club",
        "current_value",
        "game_lineups",
        "lineup",
        "appearance",
        "quarantine",
        "settlement",
    ]
    if any(t in low for t in forbidden_tokens):
        return False
    unrelated = ("ah_", "btts_", "ou05_", "ou15_", "ou25_", "ou35_", "ou45_")
    if low.startswith(unrelated):
        return False
    return True


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    market = [c for c in BASELINE if c in df.columns]
    nums = [c for c in numeric_cols(df) if allowed_feature(c)]
    clubelo = [
        c
        for c in nums
        if c
        in {
            "home_clubelo_rating",
            "away_clubelo_rating",
            "clubelo_diff_home_minus_away",
            "home_clubelo_days_stale",
            "away_clubelo_days_stale",
            "home_clubelo_found_flag",
            "away_clubelo_found_flag",
            "clubelo_both_found_flag",
        }
    ]
    fd = [c for c in nums if c.startswith(("home_fd_", "away_fd_", "fd_"))]
    tm = [c for c in nums if c.startswith(("home_tm_", "away_tm_", "tm_", "transfermarkt_"))]
    under = [c for c in nums if c.startswith(("home_understat_", "away_understat_", "understat_"))]
    flags = [
        c
        for c in nums
        if c
        in {
            "fd_rolling_features_available",
            "clubelo_available",
            "understat_available",
            "understat_missing_due_to_pre_source_era",
            "understat_after_source_max_date",
            "transfermarkt_available",
            "transfermarkt_value_both_found",
            "clubelo_both_found_flag",
            "understat_both_found_flag",
            "tm_both_value_found_flag",
        }
    ]
    groups = {
        "market_probability_only": market,
        "market_plus_clubelo": market + clubelo,
        "market_plus_football_data_rolling": market + fd,
        "market_plus_clubelo_plus_football_data_rolling": market + clubelo + fd,
        "market_plus_transfermarkt_core": market + tm,
        "market_plus_understat_core": market + under,
        "market_plus_clubelo_transfermarkt": market + clubelo + tm,
        "market_plus_clubelo_understat_transfermarkt": market + clubelo + under + tm,
        "market_plus_external_availability_flags": market + flags,
        "market_plus_all_safe_light": market + fd + clubelo + tm + under + flags,
    }
    return {k: list(dict.fromkeys(v)) for k, v in groups.items()}


def train_mask_for_policy(df: pd.DataFrame, test_season: int, policy: str) -> pd.Series:
    season = pd.to_numeric(df["season_start_year"], errors="coerce")
    if policy == "expanding_2004_plus":
        return season < test_season
    if policy == "expanding_2015_plus":
        return (season >= 2015) & (season < test_season)
    if policy == "rolling_8_seasons":
        return (season >= test_season - 8) & (season < test_season)
    if policy == "rolling_10_seasons":
        return (season >= test_season - 10) & (season < test_season)
    raise ValueError(policy)


def logistic_model(regularized: bool):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            penalty="l2" if regularized else None,
            C=0.25 if regularized else 1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=42,
        ),
    )


def xgb_model(n_estimators: int, max_depth: int, learning_rate: float, reg_lambda: float):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=reg_lambda,
            eval_metric="mlogloss",
            n_jobs=1,
            random_state=42,
            verbosity=0,
        ),
    )


def predict_fold(model, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    p = model.predict_proba(x_test)
    out = np.zeros((len(x_test), 3), dtype=float)
    classes = getattr(model[-1], "classes_", CLASSES) if hasattr(model, "__getitem__") else CLASSES
    for j, cls in enumerate(classes):
        out[:, int(cls)] = p[:, j]
    out = np.clip(out, 1e-9, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def run_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = feature_groups(df)
    policies = ["expanding_2004_plus", "expanding_2015_plus", "rolling_8_seasons", "rolling_10_seasons"]
    pred_frames = []
    by_feature_rows = []
    y_all = target_vector(df)
    for policy in policies:
        for group_name, cols in groups.items():
            if len(cols) != len(set(cols)):
                cols = list(dict.fromkeys(cols))
            model_specs: list[tuple[str, object]] = []
            if group_name in UNREGULARIZED_LOGISTIC_GROUPS:
                model_specs.append(("multinomial_logistic_regression", logistic_model(False)))
            model_specs.append(("regularized_multinomial_logistic_regression", logistic_model(True)))
            if HAS_XGB and group_name in XGB_GROUPS:
                for name, ne, depth, lr, rl in XGB_SPECS:
                    model_specs.append((name, xgb_model(ne, depth, lr, rl)))
            by_feature_rows.append(
                {
                    "training_policy": policy,
                    "feature_group": group_name,
                    "feature_count": len(cols),
                    "features": ";".join(cols),
                    "xgboost_run": HAS_XGB and group_name in XGB_GROUPS,
                }
            )
            for model_name, model in model_specs:
                for test_season in TEST_SEASONS:
                    train_mask = train_mask_for_policy(df, test_season, policy)
                    test_mask = df["season_start_year"].eq(test_season)
                    if not train_mask.any() or not test_mask.any():
                        continue
                    train = df.loc[train_mask].copy()
                    test = df.loc[test_mask].copy()
                    y_train = target_vector(train)
                    y_test = target_vector(test)
                    if len(np.unique(y_train)) < 3:
                        continue
                    print(f"{policy} | {group_name} | {model_name} | season {test_season}", flush=True)
                    prob = predict_fold(model, train[cols], y_train, test[cols])
                    base = baseline_matrix(test)
                    frame = test[
                        [
                            "canonical_match_id",
                            "season_start_year",
                            "competition_slug",
                            "result_1x2",
                            "target_home_win",
                            "target_draw",
                            "target_away_win",
                        ]
                    ].copy()
                    frame.insert(0, "training_policy", policy)
                    frame.insert(1, "feature_group", group_name)
                    frame.insert(2, "model", model_name)
                    frame["y_true"] = y_test
                    for i, label in CLASS_LABELS.items():
                        frame[f"pred_{label}"] = prob[:, i]
                        frame[f"market_{label}"] = base[:, i]
                    pred_frames.append(frame)
    return pd.concat(pred_frames, ignore_index=True), pd.DataFrame(by_feature_rows)


def summarize(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prob_cols = ["pred_home", "pred_draw", "pred_away"]
    market_cols = ["market_home", "market_draw", "market_away"]
    rows, season_rows, league_rows, class_rows, cm_rows = [], [], [], [], []
    for keys, g in preds.groupby(["training_policy", "feature_group", "model"], dropna=False):
        policy, group, model = keys
        y = g["y_true"].to_numpy(dtype=int)
        p = g[prob_cols].to_numpy(dtype=float)
        b = g[market_cols].to_numpy(dtype=float)
        m = metrics(y, p)
        bm = metrics(y, b)
        season_good = 0
        league_good = 0
        worst_season_delta = np.nan
        worst_league_delta = np.nan
        for season, sg in g.groupby("season_start_year"):
            sm = metrics(sg["y_true"].to_numpy(dtype=int), sg[prob_cols].to_numpy(dtype=float))
            sb = metrics(sg["y_true"].to_numpy(dtype=int), sg[market_cols].to_numpy(dtype=float))
            dll = sm["log_loss"] - sb["log_loss"]
            db = sm["brier"] - sb["brier"]
            season_good += int(dll < 0 and db < 0)
            worst_season_delta = max(worst_season_delta, dll) if pd.notna(worst_season_delta) else dll
            season_rows.append(
                {
                    "training_policy": policy,
                    "feature_group": group,
                    "model": model,
                    "season_start_year": season,
                    "n": len(sg),
                    **sm,
                    "market_log_loss": sb["log_loss"],
                    "market_brier": sb["brier"],
                    "delta_log_loss_vs_market": dll,
                    "delta_brier_vs_market": db,
                    "both_improved": dll < 0 and db < 0,
                }
            )
        for league, lg in g.groupby("competition_slug"):
            lm = metrics(lg["y_true"].to_numpy(dtype=int), lg[prob_cols].to_numpy(dtype=float))
            lb = metrics(lg["y_true"].to_numpy(dtype=int), lg[market_cols].to_numpy(dtype=float))
            dll = lm["log_loss"] - lb["log_loss"]
            db = lm["brier"] - lb["brier"]
            league_good += int(dll < 0 and db < 0)
            worst_league_delta = max(worst_league_delta, dll) if pd.notna(worst_league_delta) else dll
            league_rows.append(
                {
                    "training_policy": policy,
                    "feature_group": group,
                    "model": model,
                    "competition_slug": league,
                    "n": len(lg),
                    **lm,
                    "market_log_loss": lb["log_loss"],
                    "market_brier": lb["brier"],
                    "delta_log_loss_vs_market": dll,
                    "delta_brier_vs_market": db,
                    "both_improved": dll < 0 and db < 0,
                }
            )
        pred = p.argmax(axis=1)
        cm = confusion_matrix(y, pred, labels=CLASSES)
        for i, true_label in CLASS_LABELS.items():
            class_mask = y == i
            class_rows.append(
                {
                    "training_policy": policy,
                    "feature_group": group,
                    "model": model,
                    "class": true_label,
                    "support": int(class_mask.sum()),
                    "mean_predicted_probability": float(p[class_mask, i].mean()) if class_mask.any() else np.nan,
                    "one_vs_rest_brier_component": float(np.mean((p[:, i] - class_mask.astype(int)) ** 2)),
                }
            )
            for j, pred_label in CLASS_LABELS.items():
                cm_rows.append(
                    {
                        "training_policy": policy,
                        "feature_group": group,
                        "model": model,
                        "true_class": true_label,
                        "predicted_class": pred_label,
                        "count": int(cm[i, j]),
                    }
                )
        rows.append(
            {
                "training_policy": policy,
                "feature_group": group,
                "model": model,
                "n": len(g),
                **m,
                "market_log_loss": bm["log_loss"],
                "market_brier": bm["brier"],
                "delta_log_loss_vs_market": m["log_loss"] - bm["log_loss"],
                "delta_brier_vs_market": m["brier"] - bm["brier"],
                "seasons_both_improved": season_good,
                "leagues_both_improved": league_good,
                "worst_season_delta_log_loss": worst_season_delta,
                "worst_league_delta_log_loss": worst_league_delta,
                "not_driven_by_one_season": season_good >= 4,
                "not_driven_by_one_league": league_good >= 3,
                "ready_for_value_diagnostic_research_only": (m["log_loss"] < bm["log_loss"])
                and (m["brier"] < bm["brier"])
                and season_good >= 4
                and league_good >= 3,
            }
        )
    summary = pd.DataFrame(rows)
    by_policy = (
        summary.groupby("training_policy")
        .agg(
            models=("model", "count"),
            best_delta_log_loss_vs_market=("delta_log_loss_vs_market", "min"),
            best_delta_brier_vs_market=("delta_brier_vs_market", "min"),
            ready_candidates=("ready_for_value_diagnostic_research_only", "sum"),
        )
        .reset_index()
    )
    return summary, by_policy, pd.DataFrame(season_rows), pd.DataFrame(league_rows), pd.DataFrame(class_rows), pd.DataFrame(cm_rows)


def leakage_checks(df: pd.DataFrame, feature_info: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    feature_text = ";".join(feature_info["features"].fillna("").astype(str))
    forbidden_hits = [
        token
        for token in [
            "canonical_match_id",
            "football_data_row_id",
            "source_file",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "result_1x2",
            "target_",
            "current_club",
            "current_value",
            "game_lineups",
            "appearance",
            "quarantine",
        ]
        if token in feature_text
    ]
    rows = [
        {
            "check_name": "input_research_only",
            "status": "pass" if df["classification"].eq("research_only").all() else "fail",
            "details": df["classification"].value_counts().to_dict(),
        },
        {
            "check_name": "closed_test_seasons_only",
            "status": "pass" if set(TEST_SEASONS).issubset(set(df["season_start_year"].unique())) and df["season_start_year"].max() <= 2024 else "fail",
            "details": f"season_range={df['season_start_year'].min()}-{df['season_start_year'].max()}",
        },
        {
            "check_name": "forbidden_features_excluded",
            "status": "pass" if not forbidden_hits else "fail",
            "details": ";".join(forbidden_hits),
        },
        {
            "check_name": "baseline_probabilities_valid",
            "status": "pass" if df[BASELINE].notna().all().all() and np.allclose(df[BASELINE].sum(axis=1), 1.0, atol=1e-6) else "fail",
            "details": "1X2 no-vig probabilities present",
        },
        {
            "check_name": "models_compared_to_market_only",
            "status": "pass" if len(summary) > 0 else "fail",
            "details": f"model_rows={len(summary)}",
        },
        {
            "check_name": "no_value_search_or_threshold_optimization",
            "status": "pass",
            "details": "predictive audit only",
        },
        {
            "check_name": "no_extra_sources_joined",
            "status": "pass",
            "details": "used supplied full-feature CSV only",
        },
    ]
    return pd.DataFrame(rows)


def write_reports(df: pd.DataFrame, preds: pd.DataFrame, feature_info: pd.DataFrame) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary, by_policy, by_season, by_league, by_class, cm = summarize(preds)
    leakage = leakage_checks(df, feature_info, summary)
    summary.to_csv(REPORT_DIR / "extended_1x2_summary.csv", index=False)
    summary.to_csv(REPORT_DIR / "extended_1x2_by_model.csv", index=False)
    by_policy.to_csv(REPORT_DIR / "extended_1x2_by_training_policy.csv", index=False)
    by_season.to_csv(REPORT_DIR / "extended_1x2_by_season.csv", index=False)
    by_league.to_csv(REPORT_DIR / "extended_1x2_by_league.csv", index=False)
    by_class.to_csv(REPORT_DIR / "extended_1x2_by_class.csv", index=False)
    feature_info.to_csv(REPORT_DIR / "extended_1x2_feature_group_comparison.csv", index=False)
    cm.to_csv(REPORT_DIR / "extended_1x2_confusion_matrix.csv", index=False)
    leakage.to_csv(REPORT_DIR / "extended_1x2_leakage_checks.csv", index=False)
    ready = summary[summary["ready_for_value_diagnostic_research_only"].astype(bool)].copy()
    if leakage["status"].ne("pass").any() or summary.empty:
        decision = "football_data_extended_1x2_predictive_rejected_no_gain"
    elif ready.empty:
        decision = "football_data_extended_1x2_predictive_rejected_no_gain"
    else:
        best_ready = ready.sort_values("delta_log_loss_vs_market").iloc[0]
        if best_ready["feature_group"] == "market_probability_only":
            decision = "football_data_extended_1x2_predictive_market_recalibration_only"
        else:
            decision = "football_data_extended_1x2_predictive_ready_for_value_diagnostic_research_only"
    best = summary.sort_values("delta_log_loss_vs_market").head(10)
    policy_pivot = summary.pivot_table(
        index=["feature_group", "model"],
        columns="training_policy",
        values="delta_log_loss_vs_market",
        aggfunc="min",
    ).reset_index()
    report = [
        "# Extended Football-Data 1X2 Predictive Audit",
        "",
        f"Decision: **{decision}**",
        "",
        f"Rows: {len(df)}",
        f"Test seasons: {TEST_SEASONS[0]}-{TEST_SEASONS[-1]}",
        f"Prediction rows: {len(preds)}",
        "",
        "No value search, threshold optimization, extra source joins, raw-file modification, or confirmed-edge claim was performed.",
        "",
        "## Best Overall Models",
        "",
        best[
            [
                "training_policy",
                "feature_group",
                "model",
                "log_loss",
                "brier",
                "delta_log_loss_vs_market",
                "delta_brier_vs_market",
                "seasons_both_improved",
                "leagues_both_improved",
                "ready_for_value_diagnostic_research_only",
            ]
        ].to_markdown(index=False),
        "",
        "## Training History Comparison",
        "",
        policy_pivot.head(20).to_markdown(index=False),
        "",
        "Classification remains research_only. Any value diagnostic would require a separate settlement/price protocol and no confirmed edge is claimed here.",
    ]
    (REPORT_DIR / "extended_1x2_predictive_audit.md").write_text("\n".join(report) + "\n")
    (REPORT_DIR / "extended_1x2_predictive_decision.md").write_text(
        f"# Extended 1X2 Predictive Decision\n\nDecision: **{decision}**\n\nNo confirmed edge is claimed. Classification remains research_only.\n"
    )
    return decision


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    df = df[df["season_start_year"].le(2024)].copy()
    df = df[df[TARGETS].sum(axis=1).eq(1)].copy()
    df = df[df[BASELINE].notna().all(axis=1)].copy()
    preds, feature_info = run_audit(df)
    decision = write_reports(df, preds, feature_info)
    print(decision)
    print(f"rows={len(df)} predictions={len(preds)}")


if __name__ == "__main__":
    main()
