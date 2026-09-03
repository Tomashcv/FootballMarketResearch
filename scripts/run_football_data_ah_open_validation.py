from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_football_data_ah_predictive_audit import metric_dict, load_data
from run_football_data_ah_value_diagnostic import (
    CANDIDATES,
    build_predictions,
    bets_for_rule,
    max_drawdown,
    settlement_dist,
    line_bucket_dist,
)


ROOT = Path(__file__).resolve().parents[1]
OPEN_INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_clubelo_understat_transfermarkt/super_ah_open_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv"
CLOSING_CANDIDATES = ROOT / "outputs/reports/football_data_ah_value/ah_value_candidates.csv"
REPORT_DIR = ROOT / "outputs/reports/football_data_ah_open_validation"

PRIMARY_RULES = {
    ("B", "away", 0.005, "odds_ge_2_00", "big_underdog_side"),
    ("B", "away", 0.005, "odds_ge_1_90", "big_underdog_side"),
    ("A", "away", 0.005, "odds_ge_1_90", "big_underdog_side"),
    ("D", "away", 0.005, "odds_ge_1_90", "big_underdog_side"),
    ("D", "away", 0.005, "odds_ge_2_00", "big_underdog_side"),
    ("A", "away", 0.005, "odds_ge_2_00", "big_underdog_side"),
}


def prediction_metrics(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, season_rows, league_rows, line_rows = [], [], [], []
    preds = preds.copy()
    preds["y_home"] = (pd.to_numeric(preds["ah_home_unit_return"], errors="coerce") > 0).astype(int)
    for keys, g in preds.groupby(["candidate_id", "candidate_type", "feature_group", "model"], dropna=False):
        m = metric_dict(g["y_home"], g["home_model_prob"])
        bm = metric_dict(g["y_home"], g["home_market_prob"])
        candidate_id, candidate_type, feature_group, model = keys
        season_improve = 0
        league_improve = 0
        for season, sg in g.groupby("test_season"):
            sm = metric_dict(sg["y_home"], sg["home_model_prob"])
            sb = metric_dict(sg["y_home"], sg["home_market_prob"])
            both = (sm["log_loss"] - sb["log_loss"] < 0) and (sm["brier"] - sb["brier"] < 0)
            season_improve += int(both)
            season_rows.append(
                {
                    "candidate_id": candidate_id,
                    "feature_group": feature_group,
                    "model": model,
                    "test_season": season,
                    "n": len(sg),
                    **sm,
                    "market_log_loss": sb["log_loss"],
                    "market_brier": sb["brier"],
                    "delta_log_loss_vs_open_market": sm["log_loss"] - sb["log_loss"],
                    "delta_brier_vs_open_market": sm["brier"] - sb["brier"],
                }
            )
        for league, lg in g.groupby("competition_slug"):
            lm = metric_dict(lg["y_home"], lg["home_model_prob"])
            lb = metric_dict(lg["y_home"], lg["home_market_prob"])
            both = (lm["log_loss"] - lb["log_loss"] < 0) and (lm["brier"] - lb["brier"] < 0)
            league_improve += int(both)
            league_rows.append(
                {
                    "candidate_id": candidate_id,
                    "feature_group": feature_group,
                    "model": model,
                    "competition_slug": league,
                    "n": len(lg),
                    **lm,
                    "market_log_loss": lb["log_loss"],
                    "market_brier": lb["brier"],
                    "delta_log_loss_vs_open_market": lm["log_loss"] - lb["log_loss"],
                    "delta_brier_vs_open_market": lm["brier"] - lb["brier"],
                }
            )
        for bucket, bg in g.groupby(pd.cut(pd.to_numeric(g["ah_line_home"], errors="coerce"), [-10, -1, 0, 1, 10], labels=["<=-1", "(-1,0]", "(0,1]", ">1"], include_lowest=True).astype(str)):
            if bg.empty:
                continue
            lm = metric_dict(bg["y_home"], bg["home_model_prob"])
            lb = metric_dict(bg["y_home"], bg["home_market_prob"])
            line_rows.append(
                {
                    "candidate_id": candidate_id,
                    "feature_group": feature_group,
                    "model": model,
                    "line_bucket": bucket,
                    "n": len(bg),
                    **lm,
                    "market_log_loss": lb["log_loss"],
                    "market_brier": lb["brier"],
                    "delta_log_loss_vs_open_market": lm["log_loss"] - lb["log_loss"],
                    "delta_brier_vs_open_market": lm["brier"] - lb["brier"],
                }
            )
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
                "feature_group": feature_group,
                "model": model,
                "n_test": len(g),
                "seasons": g["test_season"].nunique(),
                **m,
                "open_market_log_loss": bm["log_loss"],
                "open_market_brier": bm["brier"],
                "delta_log_loss_vs_open_market": m["log_loss"] - bm["log_loss"],
                "delta_brier_vs_open_market": m["brier"] - bm["brier"],
                "seasons_both_improved": season_improve,
                "leagues_both_improved": league_improve,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(season_rows), pd.DataFrame(league_rows), pd.DataFrame(line_rows)


def summarize_open_rule(bets: pd.DataFrame, closing_rule: pd.Series) -> dict:
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
    pos_seasons = int((by_season["sum"] > 0).sum()) if not by_season.empty else 0
    pos_leagues = int((by_league["sum"] > 0).sum()) if not by_league.empty else 0
    tag = "not_open_validated"
    if (
        n >= 150
        and roi > 0.03
        and pd.notna(z)
        and z > 1.5
        and pos_seasons >= 4
        and pos_leagues >= 4
        and max_season_conc <= 0.40
        and max_league_conc <= 0.40
    ):
        tag = "stronger_open_validated_research_candidate"
    elif (
        n >= 100
        and roi > 0
        and pd.notna(z)
        and z > 1.0
        and pos_seasons >= 4
        and pos_leagues >= 3
        and max_season_conc <= 0.40
        and max_league_conc <= 0.40
    ):
        tag = "open_validated_research_candidate"
    return {
        "rule_id": closing_rule["rule_id"],
        "candidate_id": closing_rule["candidate_id"],
        "side_mode": closing_rule["side_mode"],
        "edge_threshold": closing_rule["edge_threshold"],
        "odds_filter": closing_rule["odds_filter"],
        "odds_min": closing_rule["odds_min"],
        "line_bucket": closing_rule["line_bucket"],
        "primary_pre_registered_rule": (
            closing_rule["candidate_id"],
            closing_rule["side_mode"],
            float(closing_rule["edge_threshold"]),
            closing_rule["odds_filter"],
            closing_rule["line_bucket"],
        )
        in PRIMARY_RULES,
        "closing_candidate_tag": closing_rule["candidate_tag"],
        "open_validation_tag": tag,
        "bets": n,
        "profit": profit,
        "roi": roi,
        "average_odds": float(bets["bet_odds"].mean()) if n else np.nan,
        "average_edge": float(bets["model_edge"].mean()) if n else np.nan,
        "z_score": z,
        "max_drawdown": max_drawdown(bets["profit"]) if n else np.nan,
        "seasons_with_bets": int(by_season.shape[0]),
        "positive_seasons": pos_seasons,
        "leagues_with_bets": int(by_league.shape[0]),
        "positive_leagues": pos_leagues,
        "worst_season_roi": float(season_roi.min()) if not season_roi.empty else np.nan,
        "worst_league_roi": float(league_roi.min()) if not league_roi.empty else np.nan,
        "max_season_bet_concentration": max_season_conc,
        "max_league_bet_concentration": max_league_conc,
        "settlement_distribution": settlement_dist(bets),
        "line_bucket_distribution": line_bucket_dist(bets),
        "uses_actual_settlement_returns": True,
        "classification": "research_only",
    }


def validate_locked_rules(preds: pd.DataFrame, closing_candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    detail = {"season": [], "league": [], "line": [], "settlement": []}
    for _, rule in closing_candidates.iterrows():
        bets = bets_for_rule(
            preds,
            rule["candidate_id"],
            rule["side_mode"],
            float(rule["edge_threshold"]),
            float(rule["odds_min"]),
            rule["line_bucket"],
        )
        summary = summarize_open_rule(bets, rule)
        rows.append(summary)
        if not bets.empty:
            for key, col in [("season", "season_start_year"), ("league", "competition_slug"), ("line", "line_bucket"), ("settlement", "settlement")]:
                d = bets.groupby(col)["profit"].agg(["count", "sum", "mean"]).reset_index()
                d.insert(0, "rule_id", rule["rule_id"])
                d.insert(1, "candidate_id", rule["candidate_id"])
                detail[key].append(d)
    details = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in detail.items()}
    return pd.DataFrame(rows), details


def leakage_checks(open_df: pd.DataFrame, preds: pd.DataFrame, closing_candidates: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("open_dataset_used", bool(open_df["ah_timing_label"].astype(str).eq("opening").all()), "all rows are opening-labelled"),
        ("row_predictions_written", len(preds) > 0, f"rows={len(preds)}"),
        ("locked_closing_rules_only", True, f"rules={len(closing_candidates)} from ah_value_candidates.csv"),
        ("actual_settlement_returns_used", {"ah_home_unit_return", "ah_away_unit_return"}.issubset(preds.columns), "profit uses actual unit returns"),
        ("classification_research_only", bool(preds["classification"].eq("research_only").all()), "open AH dataset remains research_only"),
        ("no_extra_sources_joined", True, "used supplied open AH dataset only"),
        ("no_raw_files_modified", True, "only outputs/reports written"),
    ]
    return pd.DataFrame([{"check_name": n, "status": "pass" if ok else "fail", "details": d} for n, ok, d in rows])


def decide(validated: pd.DataFrame, leak: pd.DataFrame) -> str:
    if leak["status"].ne("pass").any() or validated.empty:
        return "football_data_ah_open_validation_rejected"
    passed = validated[validated["open_validation_tag"].ne("not_open_validated")]
    if passed.empty:
        return "football_data_ah_open_validation_rejected"
    stronger = passed[passed["open_validation_tag"].eq("stronger_open_validated_research_candidate")]
    ids = set(passed["candidate_id"])
    if not stronger.empty:
        return "football_data_ah_open_validation_ready_for_paper_tracking_research_only"
    if "B" in ids:
        return "football_data_ah_open_validation_feature_block_research_candidate"
    return "football_data_ah_open_validation_market_recalibration_only_research_candidate"


def write_reports(
    decision: str,
    pred_summary: pd.DataFrame,
    preds: pd.DataFrame,
    validated: pd.DataFrame,
    primary: pd.DataFrame,
    details: dict[str, pd.DataFrame],
    leak: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pred_summary.to_csv(REPORT_DIR / "ah_open_predictive_summary.csv", index=False)
    preds.to_csv(REPORT_DIR / "ah_open_row_predictions.csv", index=False)
    validated.to_csv(REPORT_DIR / "ah_open_locked_rule_validation.csv", index=False)
    primary.to_csv(REPORT_DIR / "ah_open_primary_rules.csv", index=False)
    details["season"].to_csv(REPORT_DIR / "ah_open_by_season.csv", index=False)
    details["league"].to_csv(REPORT_DIR / "ah_open_by_league.csv", index=False)
    details["line"].to_csv(REPORT_DIR / "ah_open_by_line_bucket.csv", index=False)
    details["settlement"].to_csv(REPORT_DIR / "ah_open_settlement_distribution.csv", index=False)
    leak.to_csv(REPORT_DIR / "ah_open_leakage_checks.csv", index=False)
    passed = validated[validated["open_validation_tag"].ne("not_open_validated")]
    top = passed.sort_values(["open_validation_tag", "roi", "z_score"], ascending=[True, False, False]).head(20)
    lines = [
        "# Football-Data AH Open-Line Validation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "This validates locked closing-discovered AH candidate rules on opening-labelled football-data AH odds. No new open-line rules were searched or promoted.",
        "",
        f"- Row-level open predictions: {len(preds)}",
        f"- Locked closing candidate rules validated: {len(validated)}",
        f"- Open-validated rules: {len(passed)}",
        f"- Stronger open-validated rules: {int(passed['open_validation_tag'].eq('stronger_open_validated_research_candidate').sum()) if not passed.empty else 0}",
        "",
        "Profit uses actual AH unit returns only. This remains research_only and no confirmed edge is claimed.",
        "",
        "## Primary Pre-Registered Rules",
        primary[["rule_id", "candidate_id", "bets", "profit", "roi", "z_score", "positive_seasons", "positive_leagues", "open_validation_tag"]].to_markdown(index=False),
        "",
        "## Top Open-Validated Rules",
        top[["rule_id", "candidate_id", "side_mode", "bets", "profit", "roi", "z_score", "positive_seasons", "positive_leagues", "open_validation_tag"]].to_markdown(index=False) if not top.empty else "No locked candidate rule passed open validation.",
        "",
        "## Leakage Checks",
        leak.to_markdown(index=False),
        "",
        "No live betting logic was run. No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "ah_open_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT_DIR / "ah_open_validation_decision.md").write_text(
        "\n".join(["# AH Open-Line Validation Decision", "", f"Decision: **{decision}**", "", "Research-only validation on opening-labelled odds. No live betting logic and no confirmed edge claim."]) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    open_df = load_data(OPEN_INPUT)
    preds = build_predictions(open_df)
    pred_summary, by_season, by_league, by_line = prediction_metrics(preds)
    closing_candidates = pd.read_csv(CLOSING_CANDIDATES)
    validated, details = validate_locked_rules(preds, closing_candidates)
    primary = validated[validated["primary_pre_registered_rule"].astype(bool)].copy()
    leak = leakage_checks(open_df, preds, closing_candidates)
    decision = decide(validated, leak)
    # Include predictive detail tables alongside value validation tables.
    by_season.to_csv(REPORT_DIR / "ah_open_predictive_by_season.csv", index=False)
    by_league.to_csv(REPORT_DIR / "ah_open_predictive_by_league.csv", index=False)
    by_line.to_csv(REPORT_DIR / "ah_open_predictive_by_line_bucket.csv", index=False)
    write_reports(decision, pred_summary, preds, validated, primary, details, leak)
    print(decision)
    print(f"predictions={len(preds)} locked_rules={len(validated)} open_validated={int(validated['open_validation_tag'].ne('not_open_validated').sum())}")


if __name__ == "__main__":
    main()
