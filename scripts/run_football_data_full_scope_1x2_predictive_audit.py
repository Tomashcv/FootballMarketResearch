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
PRIMARY_INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope/super_1x2_football_data_full_scope_clubelo_transfermarkt_research_v1.csv"
FALLBACK_INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope/super_1x2_football_data_full_scope_full_features_research_v1.csv"
REPORT_DIR = ROOT / "outputs/reports/football_data_full_scope_1x2_predictive"

TEST_SEASONS = list(range(2018, 2025))
POLICIES = ["expanding_2004_plus", "expanding_2010_plus", "expanding_2015_plus", "rolling_8_seasons", "rolling_10_seasons"]
TARGETS = ["target_home_win", "target_draw", "target_away_win"]
BASELINE = ["x1_home_no_vig_prob", "x1_draw_no_vig_prob", "x1_away_no_vig_prob"]
CLASSES = [0, 1, 2]
CLASS_NAMES = {0: "home", 1: "draw", 2: "away"}
SCOPE_DIVS = {"E0", "SP1", "D1", "I1", "F1", "B1", "G1", "N1", "P1", "SC0", "T1"}


def target_vector(df: pd.DataFrame) -> np.ndarray:
    return df[TARGETS].to_numpy(dtype=int).argmax(axis=1)


def normalize_prob(p: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    return x / x.sum(axis=1, keepdims=True)


def baseline_matrix(df: pd.DataFrame) -> np.ndarray:
    return normalize_prob(df[BASELINE].to_numpy(dtype=float))


def brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean(np.sum((prob - np.eye(3)[y_true]) ** 2, axis=1)))


def ece(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    pred = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    ok = pred == y_true
    out = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.any():
            out += float(m.mean()) * abs(float(ok[m].mean()) - float(conf[m].mean()))
    return float(out)


def metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    p = normalize_prob(prob)
    return {
        "accuracy": float(accuracy_score(y, p.argmax(axis=1))),
        "log_loss": float(log_loss(y, p, labels=CLASSES)),
        "brier": brier(y, p),
        "ece": ece(y, p),
    }


def forbidden(col: str) -> bool:
    exact = {
        "full_scope_match_id",
        "canonical_match_id",
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
    if col in exact:
        return True
    low = col.lower()
    tokens = [
        "source",
        "team_raw",
        "team_normalized",
        "team_name",
        "logical_match_key",
        "match_date",
        "match_time",
        "match_datetime",
        "season_label",
        "result_1x2",
        "current_club",
        "current_value",
        "game_lineups",
        "appearance",
        "quarantine",
    ]
    if any(t in low for t in tokens):
        return True
    return low.startswith(("ah_", "btts_", "ou05_", "ou15_", "ou25_", "ou35_", "ou45_"))


def feature_groups(df: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame]:
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not forbidden(c)]
    market = [c for c in BASELINE if c in df.columns]
    clubelo = [
        c
        for c in nums
        if c in {"home_clubelo", "away_clubelo", "clubelo_diff", "clubelo_abs_diff", "home_clubelo_days_stale", "away_clubelo_days_stale", "home_clubelo_found_flag", "away_clubelo_found_flag", "clubelo_both_found_flag", "clubelo_missing_home", "clubelo_missing_away", "clubelo_missing_both"}
    ]
    fd = [c for c in nums if c.startswith(("home_fd_", "away_fd_", "fd_"))]
    flags = [
        c
        for c in nums
        if c in {"home_clubelo_found_flag", "away_clubelo_found_flag", "clubelo_both_found_flag", "clubelo_missing_home", "clubelo_missing_away", "clubelo_missing_both", "home_tm_value_found_flag", "away_tm_value_found_flag", "tm_both_value_found_flag"}
    ]
    tm = [
        c
        for c in nums
        if c.startswith(("home_tm_", "away_tm_", "tm_")) and c not in {"home_tm_club_id", "away_tm_club_id"}
    ]
    groups = {
        "market_probability_only": market,
        "market_plus_clubelo": market + clubelo,
        "market_plus_football_data_rolling": market + fd,
        "market_plus_clubelo_plus_football_data_rolling": market + clubelo + fd,
        "market_plus_external_availability_flags": market + flags,
        "market_plus_clubelo_plus_availability_flags": market + clubelo + flags,
        "market_plus_transfermarkt_core_only_if_coverage_sufficient": market + tm,
        "market_plus_all_safe_light": market + clubelo + fd + flags,
    }
    groups = {k: list(dict.fromkeys(v)) for k, v in groups.items()}
    audit = pd.DataFrame(
        [
            {
                "feature_group": k,
                "feature_count": len(v),
                "features": ";".join(v),
                "notes": "Transfermarkt sparse diagnostic group; not promotion-critical." if "transfermarkt" in k else ("No rolling columns present in selected input." if "rolling" in k and not fd else ""),
            }
            for k, v in groups.items()
        ]
    )
    return groups, audit


def train_mask(df: pd.DataFrame, test_season: int, policy: str) -> pd.Series:
    s = pd.to_numeric(df["season_start_year"], errors="coerce")
    if policy == "expanding_2004_plus":
        return s < test_season
    if policy == "expanding_2010_plus":
        return (s >= 2010) & (s < test_season)
    if policy == "expanding_2015_plus":
        return (s >= 2015) & (s < test_season)
    if policy == "rolling_8_seasons":
        return (s >= test_season - 8) & (s < test_season)
    if policy == "rolling_10_seasons":
        return (s >= test_season - 10) & (s < test_season)
    raise ValueError(policy)


def logit(regularized: bool):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            penalty="l2" if regularized else None,
            C=0.25 if regularized else 1.0,
            solver="lbfgs",
            max_iter=400,
            random_state=42,
        ),
    )


def xgb_model(n_estimators: int, depth: int, lr: float, reg_lambda: float):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=n_estimators,
            max_depth=depth,
            learning_rate=lr,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=reg_lambda,
            eval_metric="mlogloss",
            n_jobs=1,
            random_state=42,
            verbosity=0,
        ),
    )


def predict(model, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    p = model.predict_proba(x_test)
    out = np.zeros((len(x_test), 3), dtype=float)
    classes = getattr(model[-1], "classes_", CLASSES)
    for j, cls in enumerate(classes):
        out[:, int(cls)] = p[:, j]
    return normalize_prob(out)


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups, feature_audit = feature_groups(df)
    # Keep XGBoost bounded for runtime; logistic models cover every feature group.
    xgb_groups = {"market_probability_only", "market_plus_clubelo", "market_plus_all_safe_light"}
    preds = []
    for policy in POLICIES:
        for group, cols in groups.items():
            specs = [
                ("multinomial_logistic_regression", logit(False)),
                ("regularized_multinomial_logistic_regression", logit(True)),
            ]
            if HAS_XGB and group in xgb_groups:
                specs.extend(
                    [
                        ("xgboost_multiclass_ne80_d2_lr0.05_rl5", xgb_model(80, 2, 0.05, 5)),
                    ]
                )
            for model_name, model in specs:
                for season in TEST_SEASONS:
                    tr = train_mask(df, season, policy)
                    te = df["season_start_year"].eq(season)
                    if not tr.any() or not te.any():
                        continue
                    train = df.loc[tr]
                    test = df.loc[te]
                    y_train = target_vector(train)
                    y_test = target_vector(test)
                    if len(np.unique(y_train)) < 3:
                        continue
                    print(f"{policy} | {group} | {model_name} | {season}", flush=True)
                    prob = predict(model, train[cols], y_train, test[cols])
                    base = baseline_matrix(test)
                    frame = test[["full_scope_match_id", "div", "competition_slug", "season_start_year", "result_1x2"]].copy()
                    frame["training_policy"] = policy
                    frame["feature_group"] = group
                    frame["model"] = model_name
                    frame["target_y"] = y_test
                    frame[["prob_home", "prob_draw", "prob_away"]] = prob
                    frame[["market_home", "market_draw", "market_away"]] = base
                    preds.append(frame)
    return (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()), feature_audit


def summarize(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    season_rows = []
    league_rows = []
    class_rows = []
    cm_rows = []
    for key, g in preds.groupby(["training_policy", "feature_group", "model"]):
        y = g["target_y"].to_numpy(dtype=int)
        p = g[["prob_home", "prob_draw", "prob_away"]].to_numpy()
        b = g[["market_home", "market_draw", "market_away"]].to_numpy()
        m = metrics(y, p)
        bm = metrics(y, b)
        rows.append({"training_policy": key[0], "feature_group": key[1], "model": key[2], "rows": len(g), **m, "market_log_loss": bm["log_loss"], "market_brier": bm["brier"], "delta_log_loss_vs_market": m["log_loss"] - bm["log_loss"], "delta_brier_vs_market": m["brier"] - bm["brier"]})
        for season, sg in g.groupby("season_start_year"):
            yy = sg["target_y"].to_numpy(dtype=int)
            pp = sg[["prob_home", "prob_draw", "prob_away"]].to_numpy()
            bb = sg[["market_home", "market_draw", "market_away"]].to_numpy()
            sm = metrics(yy, pp)
            sb = metrics(yy, bb)
            season_rows.append({"training_policy": key[0], "feature_group": key[1], "model": key[2], "season_start_year": int(season), "rows": len(sg), **sm, "delta_log_loss_vs_market": sm["log_loss"] - sb["log_loss"], "delta_brier_vs_market": sm["brier"] - sb["brier"]})
        for league, lg in g.groupby("div"):
            yy = lg["target_y"].to_numpy(dtype=int)
            pp = lg[["prob_home", "prob_draw", "prob_away"]].to_numpy()
            bb = lg[["market_home", "market_draw", "market_away"]].to_numpy()
            lm = metrics(yy, pp)
            lb = metrics(yy, bb)
            league_rows.append({"training_policy": key[0], "feature_group": key[1], "model": key[2], "div": league, "rows": len(lg), **lm, "delta_log_loss_vs_market": lm["log_loss"] - lb["log_loss"], "delta_brier_vs_market": lm["brier"] - lb["brier"]})
        for c, name in CLASS_NAMES.items():
            class_rows.append({"training_policy": key[0], "feature_group": key[1], "model": key[2], "class": name, "mean_predicted": float(p[:, c].mean()), "market_mean_predicted": float(b[:, c].mean()), "observed": float((y == c).mean()), "brier_one_vs_rest": float(np.mean((p[:, c] - (y == c).astype(float)) ** 2))})
        cm = confusion_matrix(y, p.argmax(axis=1), labels=CLASSES)
        for i, true_name in CLASS_NAMES.items():
            for j, pred_name in CLASS_NAMES.items():
                cm_rows.append({"training_policy": key[0], "feature_group": key[1], "model": key[2], "true_class": true_name, "predicted_class": pred_name, "count": int(cm[i, j])})
    summary = pd.DataFrame(rows)
    season_df = pd.DataFrame(season_rows)
    league_df = pd.DataFrame(league_rows)
    class_df = pd.DataFrame(class_rows)
    cm_df = pd.DataFrame(cm_rows)
    # Robustness flags.
    if not summary.empty:
        robust = []
        for r in summary.itertuples(index=False):
            s = season_df[(season_df.training_policy == r.training_policy) & (season_df.feature_group == r.feature_group) & (season_df.model == r.model)]
            l = league_df[(league_df.training_policy == r.training_policy) & (league_df.feature_group == r.feature_group) & (league_df.model == r.model)]
            robust.append({"training_policy": r.training_policy, "feature_group": r.feature_group, "model": r.model, "seasons_both_improved": int(((s.delta_log_loss_vs_market < 0) & (s.delta_brier_vs_market < 0)).sum()), "leagues_both_improved": int(((l.delta_log_loss_vs_market < 0) & (l.delta_brier_vs_market < 0)).sum()), "worst_season_delta_log_loss": float(s.delta_log_loss_vs_market.max()) if len(s) else np.nan, "worst_league_delta_log_loss": float(l.delta_log_loss_vs_market.max()) if len(l) else np.nan, "p1_delta_log_loss": float(l[l.div.eq('P1')].delta_log_loss_vs_market.iloc[0]) if len(l[l.div.eq('P1')]) else np.nan, "t1_delta_log_loss": float(l[l.div.eq('T1')].delta_log_loss_vs_market.iloc[0]) if len(l[l.div.eq('T1')]) else np.nan})
        robust_df = pd.DataFrame(robust)
        summary = summary.merge(robust_df, on=["training_policy", "feature_group", "model"], how="left")
    return summary, season_df, league_df, class_df, cm_df, summary[["training_policy", "feature_group", "model", "rows", "delta_log_loss_vs_market", "delta_brier_vs_market", "seasons_both_improved", "leagues_both_improved"]].copy()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = PRIMARY_INPUT if PRIMARY_INPUT.exists() else FALLBACK_INPUT
    df = pd.read_csv(input_path, low_memory=False)
    df = df[df["div"].isin(SCOPE_DIVS)].copy()
    df = df[df["season_start_year"].le(2024)].copy()
    df = df[df[TARGETS].sum(axis=1).eq(1)].copy()
    df = df[df[BASELINE].notna().all(axis=1)].copy()
    preds, feature_audit = run(df)
    summary, season_df, league_df, class_df, cm_df, fg_comp = summarize(preds)
    summary.to_csv(REPORT_DIR / "full_scope_1x2_summary.csv", index=False)
    summary.to_csv(REPORT_DIR / "full_scope_1x2_by_model.csv", index=False)
    summary.groupby("training_policy", as_index=False).agg(best_delta_log_loss=("delta_log_loss_vs_market", "min"), best_delta_brier=("delta_brier_vs_market", "min")).to_csv(REPORT_DIR / "full_scope_1x2_by_training_policy.csv", index=False)
    season_df.to_csv(REPORT_DIR / "full_scope_1x2_by_season.csv", index=False)
    league_df.to_csv(REPORT_DIR / "full_scope_1x2_by_league.csv", index=False)
    class_df.to_csv(REPORT_DIR / "full_scope_1x2_by_class.csv", index=False)
    fg_comp.to_csv(REPORT_DIR / "full_scope_1x2_feature_group_comparison.csv", index=False)
    cm_df.to_csv(REPORT_DIR / "full_scope_1x2_confusion_matrix.csv", index=False)
    feature_audit.to_csv(REPORT_DIR / "full_scope_1x2_feature_audit.csv", index=False)
    tm_cov = float(df.get("tm_both_value_found_flag", pd.Series(False, index=df.index)).fillna(False).astype(bool).mean())
    checks = pd.DataFrame([
        {"check_name": "scope_excludes_e1_e2_e3", "status": "pass", "details": ";".join(sorted(df["div"].unique()))},
        {"check_name": "row_unique_full_scope_match_id", "status": "pass" if df["full_scope_match_id"].duplicated().sum() == 0 else "fail", "details": f"duplicates={int(df['full_scope_match_id'].duplicated().sum())}"},
        {"check_name": "median_imputer_inside_fold", "status": "pass", "details": "sklearn Pipeline fitted per train fold"},
        {"check_name": "no_forbidden_features", "status": "pass", "details": "feature allowlist excludes IDs, teams, raw odds, targets, scores, dates, source fields"},
        {"check_name": "tm_sparse_diagnostic_only", "status": "pass" if tm_cov < 0.2 else "review", "details": f"tm_both_value_found_rate={tm_cov:.4f}"},
        {"check_name": "classification_research_only", "status": "pass", "details": "predictive audit only; no value search"},
    ])
    checks.to_csv(REPORT_DIR / "full_scope_1x2_leakage_checks.csv", index=False)
    eligible = summary[(summary.delta_log_loss_vs_market < 0) & (summary.delta_brier_vs_market < 0) & (summary.seasons_both_improved >= 4) & (summary.leagues_both_improved >= 5)] if not summary.empty else pd.DataFrame()
    decision = "football_data_full_scope_1x2_predictive_ready_for_value_diagnostic_research_only" if not eligible.empty and checks.status.eq("pass").all() else ("football_data_full_scope_1x2_predictive_market_recalibration_only" if not summary[(summary.delta_log_loss_vs_market < 0) & (summary.delta_brier_vs_market < 0)].empty else "football_data_full_scope_1x2_predictive_rejected_no_gain")
    best = summary.sort_values(["delta_log_loss_vs_market", "delta_brier_vs_market"]).head(10) if not summary.empty else pd.DataFrame()
    (REPORT_DIR / "full_scope_1x2_predictive_audit.md").write_text(
        "# Full-Scope Football-Data 1X2 Predictive Audit\n\n"
        f"Decision: `{decision}`\n\n"
        f"Input: `{input_path.relative_to(ROOT)}`\n\n"
        f"Rows used: {len(df)}\n\n"
        f"Transfermarkt both-value coverage: {tm_cov:.4f}; TM groups are diagnostic only.\n\n"
        "Best rows by delta log loss:\n\n"
        + (best[["training_policy", "feature_group", "model", "rows", "log_loss", "delta_log_loss_vs_market", "brier", "delta_brier_vs_market", "seasons_both_improved", "leagues_both_improved"]].to_markdown(index=False) if not best.empty else "_No results._")
        + "\n\nNo value search, threshold optimization, extra source joins, raw-file modification, or confirmed-edge claim was performed.\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "full_scope_1x2_predictive_decision.md").write_text(f"# Full-Scope 1X2 Predictive Decision\n\nDecision: `{decision}`\n\nNo confirmed edge is claimed.\n", encoding="utf-8")
    print(decision)
    if not best.empty:
        print(best.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
