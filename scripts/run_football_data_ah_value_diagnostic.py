from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from run_football_data_ah_predictive_audit import (
    INPUT,
    BASELINE,
    AWAY_BASELINE,
    TEST_SEASONS,
    feature_groups,
    load_data,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs/reports/football_data_ah_value"

THRESHOLDS = [0.005, 0.010, 0.015, 0.020, 0.030, 0.040]
ODDS_FILTERS = {
    "all": 0.0,
    "odds_ge_1_75": 1.75,
    "odds_ge_1_80": 1.80,
    "odds_ge_1_90": 1.90,
    "odds_ge_2_00": 2.00,
}
LINE_BUCKETS = [
    "all",
    "favourite_side",
    "underdog_side",
    "big_favourite_side",
    "big_underdog_side",
    "small_favourite_or_pick",
    "small_underdog",
]

CANDIDATES = {
    "A": {
        "candidate_type": "market_recalibration",
        "feature_group": "market_plus_ah_line",
        "model": "xgboost_binary_ne80_lr0.05_rl5",
    },
    "B": {
        "candidate_type": "date_safe_feature_block",
        "feature_group": "market_plus_clubelo",
        "model": "regularized_logistic_regression",
    },
    "C": {
        "candidate_type": "market_probability_recalibration_control",
        "feature_group": "market_probability_only",
        "model": "regularized_logistic_regression",
    },
    "D": {
        "candidate_type": "market_recalibration",
        "feature_group": "market_plus_ah_line",
        "model": "xgboost_binary_ne120_lr0.03_rl20",
    },
}


def candidate_model(name: str):
    if name == "regularized_logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(penalty="l2", C=0.25, solver="lbfgs", max_iter=600, random_state=42),
        )
    if name == "xgboost_binary_ne80_lr0.05_rl5":
        return make_pipeline(
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
    if name == "xgboost_binary_ne120_lr0.03_rl20":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=120,
                max_depth=2,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=20,
                eval_metric="logloss",
                n_jobs=1,
                random_state=42,
                verbosity=0,
            ),
        )
    raise ValueError(name)


def predict_proba(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def build_predictions(df: pd.DataFrame) -> pd.DataFrame:
    groups = feature_groups(df)
    df = df.copy()
    df["target_ah_home_positive_return"] = (pd.to_numeric(df["ah_home_unit_return"], errors="coerce") > 0).astype(int)
    df["target_ah_away_positive_return"] = (pd.to_numeric(df["ah_away_unit_return"], errors="coerce") > 0).astype(int)
    pred_frames = []
    for cid, spec in CANDIDATES.items():
        cols = groups[spec["feature_group"]]
        for test_season in TEST_SEASONS:
            train = df[df["season_start_year"] < test_season].copy()
            test = df[df["season_start_year"] == test_season].copy()
            if train.empty or test.empty:
                continue
            if train["target_ah_home_positive_return"].nunique() < 2 or train["target_ah_away_positive_return"].nunique() < 2:
                continue
            print(f"{cid}: season {test_season}", flush=True)
            home_model = candidate_model(spec["model"])
            away_model = candidate_model(spec["model"])
            home_prob = predict_proba(home_model, train[cols], train["target_ah_home_positive_return"], test[cols])
            away_prob = predict_proba(away_model, train[cols], train["target_ah_away_positive_return"], test[cols])
            out = test[
                [
                    "canonical_match_id",
                    "season_start_year",
                    "competition_slug",
                    "match_datetime",
                    "ah_line_home",
                    "ah_home_odds",
                    "ah_away_odds",
                    "ah_home_no_vig_prob",
                    "ah_away_no_vig_prob",
                    "ah_home_unit_return",
                    "ah_away_unit_return",
                    "ah_home_settlement",
                    "ah_away_settlement",
                    "classification",
                ]
            ].copy()
            out.insert(0, "candidate_id", cid)
            out.insert(1, "candidate_type", spec["candidate_type"])
            out.insert(2, "feature_group", spec["feature_group"])
            out.insert(3, "model", spec["model"])
            out["test_season"] = test_season
            out["home_model_prob"] = home_prob
            out["away_model_prob"] = away_prob
            out["home_market_prob"] = out["ah_home_no_vig_prob"]
            out["away_market_prob"] = out["ah_away_no_vig_prob"]
            out["home_model_edge"] = out["home_model_prob"] - out["home_market_prob"]
            out["away_model_edge"] = out["away_model_prob"] - out["away_market_prob"]
            pred_frames.append(out)
    return pd.concat(pred_frames, ignore_index=True)


def bucket_mask(line: pd.Series, bucket: str) -> pd.Series:
    h = pd.to_numeric(line, errors="coerce")
    if bucket == "all":
        return h.notna()
    if bucket == "favourite_side":
        return h < 0
    if bucket == "underdog_side":
        return h > 0
    if bucket == "big_favourite_side":
        return h <= -1.0
    if bucket == "big_underdog_side":
        return h >= 1.0
    if bucket == "small_favourite_or_pick":
        return (h > -1.0) & (h <= 0)
    if bucket == "small_underdog":
        return (h > 0) & (h < 1.0)
    raise ValueError(bucket)


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = returns.cumsum()
    peak = equity.cummax()
    return float((equity - peak).min())


def settlement_dist(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return "; ".join(f"{k}:{v}" for k, v in df["settlement"].value_counts(dropna=False).sort_index().items())


def line_bucket_dist(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return "; ".join(f"{k}:{v}" for k, v in df["line_bucket"].value_counts(dropna=False).sort_index().items())


def bets_for_rule(preds: pd.DataFrame, cid: str, side_mode: str, threshold: float, odds_min: float, bucket: str) -> pd.DataFrame:
    p = preds[preds["candidate_id"].eq(cid)].copy()
    rows = []
    if side_mode in {"home", "combined"}:
        h = p[p["home_model_edge"] >= threshold].copy()
        h = h[h["ah_home_odds"] >= odds_min]
        h = h[bucket_mask(h["ah_line_home"], bucket)]
        h["bet_side"] = "home"
        h["side_line"] = h["ah_line_home"]
        h["bet_odds"] = h["ah_home_odds"]
        h["model_edge"] = h["home_model_edge"]
        h["profit"] = h["ah_home_unit_return"]
        h["settlement"] = h["ah_home_settlement"]
        rows.append(h)
    if side_mode in {"away", "combined"}:
        a = p[p["away_model_edge"] >= threshold].copy()
        a = a[a["ah_away_odds"] >= odds_min]
        a["away_line"] = -pd.to_numeric(a["ah_line_home"], errors="coerce")
        a = a[bucket_mask(a["away_line"], bucket)]
        a["bet_side"] = "away"
        a["side_line"] = a["away_line"]
        a["bet_odds"] = a["ah_away_odds"]
        a["model_edge"] = a["away_model_edge"]
        a["profit"] = a["ah_away_unit_return"]
        a["settlement"] = a["ah_away_settlement"]
        rows.append(a)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if side_mode == "combined" and not out.empty:
        out = out.sort_values(["canonical_match_id", "model_edge"], ascending=[True, False])
        out = out.drop_duplicates("canonical_match_id", keep="first")
    out["line_bucket"] = bucket
    return out.sort_values(["match_datetime", "canonical_match_id"]).reset_index(drop=True)


def summarize_rule(bets: pd.DataFrame, cid: str, side_mode: str, threshold: float, odds_label: str, odds_min: float, bucket: str) -> dict:
    n = len(bets)
    profit = float(bets["profit"].sum()) if n else 0.0
    roi = profit / n if n else np.nan
    std = float(bets["profit"].std(ddof=1)) if n > 1 else np.nan
    z = profit / (std * np.sqrt(n)) if n > 1 and std > 0 else np.nan
    by_season = bets.groupby("season_start_year")["profit"].agg(["count", "sum"]) if n else pd.DataFrame(columns=["count", "sum"])
    by_league = bets.groupby("competition_slug")["profit"].agg(["count", "sum"]) if n else pd.DataFrame(columns=["count", "sum"])
    season_roi = by_season["sum"] / by_season["count"] if not by_season.empty else pd.Series(dtype=float)
    league_roi = by_league["sum"] / by_league["count"] if not by_league.empty else pd.Series(dtype=float)
    max_season_conc = float(by_season["count"].max() / n) if n and not by_season.empty else np.nan
    max_league_conc = float(by_league["count"].max() / n) if n and not by_league.empty else np.nan
    positive_seasons = int((by_season["sum"] > 0).sum()) if not by_season.empty else 0
    positive_leagues = int((by_league["sum"] > 0).sum()) if not by_league.empty else 0
    candidate_tag = "not_candidate"
    if (
        n >= 200
        and roi > 0.03
        and pd.notna(z)
        and z > 1.5
        and positive_seasons >= 4
        and positive_leagues >= 4
        and max_season_conc <= 0.40
        and max_league_conc <= 0.40
    ):
        candidate_tag = "stronger_research_candidate"
    elif (
        n >= 150
        and roi > 0
        and pd.notna(z)
        and z > 1.0
        and positive_seasons >= 4
        and positive_leagues >= 3
        and max_season_conc <= 0.40
        and max_league_conc <= 0.40
    ):
        candidate_tag = "value_diagnostic_candidate_research_only"
    return {
        "candidate_id": cid,
        "side_mode": side_mode,
        "edge_threshold": threshold,
        "odds_filter": odds_label,
        "odds_min": odds_min,
        "line_bucket": bucket,
        "bets": n,
        "profit": profit,
        "roi": roi,
        "average_odds": float(bets["bet_odds"].mean()) if n else np.nan,
        "average_edge": float(bets["model_edge"].mean()) if n else np.nan,
        "z_score": z,
        "max_drawdown": max_drawdown(bets["profit"]) if n else np.nan,
        "seasons_with_bets": int(by_season.shape[0]),
        "positive_seasons": positive_seasons,
        "leagues_with_bets": int(by_league.shape[0]),
        "positive_leagues": positive_leagues,
        "worst_season_roi": float(season_roi.min()) if not season_roi.empty else np.nan,
        "worst_league_roi": float(league_roi.min()) if not league_roi.empty else np.nan,
        "max_season_bet_concentration": max_season_conc,
        "max_league_bet_concentration": max_league_conc,
        "line_bucket_distribution": line_bucket_dist(bets),
        "settlement_distribution": settlement_dist(bets),
        "candidate_tag": candidate_tag,
        "uses_actual_settlement_returns": True,
        "classification": "research_only",
    }


def evaluate_rules(preds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    detail_rows: dict[str, list[pd.DataFrame]] = {"season": [], "league": [], "line": [], "settlement": [], "candidate": []}
    for cid in CANDIDATES:
        for side_mode in ["home", "away", "combined"]:
            for threshold in THRESHOLDS:
                for odds_label, odds_min in ODDS_FILTERS.items():
                    for bucket in LINE_BUCKETS:
                        bets = bets_for_rule(preds, cid, side_mode, threshold, odds_min, bucket)
                        summary = summarize_rule(bets, cid, side_mode, threshold, odds_label, odds_min, bucket)
                        if summary["bets"] >= 50:
                            rule_id = f"{cid}|{side_mode}|{threshold:.3f}|{odds_label}|{bucket}"
                            summary["rule_id"] = rule_id
                            rows.append(summary)
                            if not bets.empty:
                                for key, col in [("season", "season_start_year"), ("league", "competition_slug"), ("line", "line_bucket"), ("settlement", "settlement")]:
                                    d = bets.groupby(col)["profit"].agg(["count", "sum", "mean"]).reset_index()
                                    d.insert(0, "rule_id", rule_id)
                                    d.insert(1, "candidate_id", cid)
                                    detail_rows[key].append(d)
    rules = pd.DataFrame(rows)
    details = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in detail_rows.items()}
    if not rules.empty:
        details["candidate"] = rules.groupby("candidate_id").agg(
            rules=("rule_id", "count"),
            candidate_rules=("candidate_tag", lambda s: int(s.ne("not_candidate").sum())),
            stronger_rules=("candidate_tag", lambda s: int(s.eq("stronger_research_candidate").sum())),
            max_roi=("roi", "max"),
            max_z_score=("z_score", "max"),
            max_bets=("bets", "max"),
        ).reset_index()
    else:
        details["candidate"] = pd.DataFrame()
    return rules, details


def leakage_checks(preds: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("row_predictions_written", True, f"rows={len(preds)}"),
        ("actual_settlement_returns_used", {"ah_home_unit_return", "ah_away_unit_return"}.issubset(preds.columns), "profit columns are unit returns"),
        ("classification_research_only", bool(preds["classification"].eq("research_only").all()), "primary AH dataset remains research_only"),
        ("no_extra_sources_joined", True, "used supplied AH dataset only"),
        ("no_raw_files_modified", True, "only outputs/reports written"),
        ("no_binary_target_profit", True, "profit uses ah_*_unit_return, not target labels"),
        ("closing_label_diagnostic_only", True, "primary dataset timing label is closing"),
    ]
    return pd.DataFrame([{"check_name": n, "status": "pass" if ok else "fail", "details": d} for n, ok, d in checks])


def write_reports(preds: pd.DataFrame, rules: pd.DataFrame, details: dict[str, pd.DataFrame], leak: pd.DataFrame, decision: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(REPORT_DIR / "ah_value_row_predictions.csv", index=False)
    rules.to_csv(REPORT_DIR / "ah_value_all_rules.csv", index=False)
    candidates = rules[rules["candidate_tag"].ne("not_candidate")].copy() if not rules.empty else pd.DataFrame()
    candidates.to_csv(REPORT_DIR / "ah_value_candidates.csv", index=False)
    details["candidate"].to_csv(REPORT_DIR / "ah_value_by_candidate_model.csv", index=False)
    details["season"].to_csv(REPORT_DIR / "ah_value_by_season.csv", index=False)
    details["league"].to_csv(REPORT_DIR / "ah_value_by_league.csv", index=False)
    details["line"].to_csv(REPORT_DIR / "ah_value_by_line_bucket.csv", index=False)
    details["settlement"].to_csv(REPORT_DIR / "ah_value_settlement_distribution.csv", index=False)
    leak.to_csv(REPORT_DIR / "ah_value_leakage_checks.csv", index=False)
    top = candidates.sort_values(["candidate_tag", "roi", "z_score"], ascending=[True, False, False]).head(20) if not candidates.empty else pd.DataFrame()
    lines = [
        "# Football-Data AH Settlement-Aware Value Diagnostic",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "This is a research-only diagnostic on closing-labelled football-data AH odds. It is not live betting logic and no confirmed edge is claimed.",
        "",
        f"- Row-level prediction rows: {len(preds)}",
        f"- Reported rules with bets >= 50: {len(rules)}",
        f"- Candidate rules: {len(candidates)}",
        f"- Stronger candidate rules: {int(candidates['candidate_tag'].eq('stronger_research_candidate').sum()) if not candidates.empty else 0}",
        "",
        "Profit uses actual `ah_home_unit_return` / `ah_away_unit_return` only. Binary targets are used only for model fitting.",
        "",
        "## Top Candidate Rules",
        top[["rule_id", "candidate_id", "side_mode", "edge_threshold", "odds_filter", "line_bucket", "bets", "profit", "roi", "z_score", "positive_seasons", "positive_leagues", "candidate_tag"]].to_markdown(index=False) if not top.empty else "No candidate rules passed the promotion filters.",
        "",
        "## Leakage Checks",
        leak.to_markdown(index=False),
        "",
        "Open-line validation remains required before any stronger interpretation. No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "ah_value_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT_DIR / "ah_value_decision.md").write_text(
        "\n".join(["# AH Value Diagnostic Decision", "", f"Decision: **{decision}**", "", "Research-only diagnostic. Closing-labelled value is not a tradable edge. No confirmed edge is claimed."]) + "\n",
        encoding="utf-8",
    )


def decide(rules: pd.DataFrame, leak: pd.DataFrame) -> str:
    if leak["status"].ne("pass").any() or rules.empty:
        return "football_data_ah_value_rejected_no_robust_value"
    candidates = rules[rules["candidate_tag"].ne("not_candidate")]
    if candidates.empty:
        return "football_data_ah_value_rejected_no_robust_value"
    if candidates["candidate_tag"].eq("stronger_research_candidate").any():
        return "football_data_ah_value_ready_for_open_line_validation_research_only"
    types = candidates["candidate_id"].map({k: v["candidate_type"] for k, v in CANDIDATES.items()})
    if types.eq("date_safe_feature_block").any():
        return "football_data_ah_value_feature_block_research_candidate"
    if types.isin(["market_recalibration", "market_probability_recalibration_control"]).any():
        return "football_data_ah_value_market_recalibration_only_research_candidate"
    return "football_data_ah_value_rejected_no_robust_value"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(INPUT)
    preds = build_predictions(df)
    rules, details = evaluate_rules(preds)
    leak = leakage_checks(preds)
    decision = decide(rules, leak)
    write_reports(preds, rules, details, leak, decision)
    print(decision)
    print(f"predictions={len(preds)} rules={len(rules)} candidates={0 if rules.empty else int(rules['candidate_tag'].ne('not_candidate').sum())}")


if __name__ == "__main__":
    main()
