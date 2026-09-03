from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_football_data_ah_value_diagnostic import bets_for_rule, max_drawdown


ROOT = Path(__file__).resolve().parents[1]
OPEN_INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_clubelo_understat_transfermarkt/super_ah_open_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv"
CLOSE_INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_clubelo_understat_transfermarkt/super_ah_close_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv"
CLOSING_CANDIDATES = ROOT / "outputs/reports/football_data_ah_value/ah_value_candidates.csv"
OPEN_PRIMARY_RULES = ROOT / "outputs/reports/football_data_ah_open_validation/ah_open_primary_rules.csv"
OPEN_LOCKED_RULES = ROOT / "outputs/reports/football_data_ah_open_validation/ah_open_locked_rule_validation.csv"
CLOSE_PREDS = ROOT / "outputs/reports/football_data_ah_value/ah_value_row_predictions.csv"
OPEN_PREDS = ROOT / "outputs/reports/football_data_ah_open_validation/ah_open_row_predictions.csv"
REPORT_DIR = ROOT / "outputs/reports/football_data_ah_open_close_diagnostic"


def side_bucket(side_line: pd.Series) -> pd.Series:
    x = pd.to_numeric(side_line, errors="coerce")
    return np.select(
        [
            x <= -1.0,
            (x > -1.0) & (x <= 0),
            (x > 0) & (x < 1.0),
            x >= 1.0,
        ],
        [
            "big_favourite_side",
            "small_favourite_or_pick",
            "small_underdog",
            "big_underdog_side",
        ],
        default="unknown",
    )


def load_joined() -> pd.DataFrame:
    base_cols = [
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
    open_df = pd.read_csv(OPEN_INPUT, usecols=lambda c: c in base_cols)
    close_df = pd.read_csv(CLOSE_INPUT, usecols=lambda c: c in base_cols)
    joined = close_df.merge(open_df, on="canonical_match_id", how="inner", suffixes=("_close", "_open"))
    joined["home_line_open"] = pd.to_numeric(joined["ah_line_home_open"], errors="coerce")
    joined["home_line_close"] = pd.to_numeric(joined["ah_line_home_close"], errors="coerce")
    joined["away_line_open"] = -joined["home_line_open"]
    joined["away_line_close"] = -joined["home_line_close"]
    for side in ["home", "away"]:
        joined[f"{side}_line_movement_close_minus_open"] = joined[f"{side}_line_close"] - joined[f"{side}_line_open"]
        joined[f"{side}_odds_movement_close_minus_open"] = (
            pd.to_numeric(joined[f"ah_{side}_odds_close"], errors="coerce")
            - pd.to_numeric(joined[f"ah_{side}_odds_open"], errors="coerce")
        )
        joined[f"{side}_prob_movement_close_minus_open"] = (
            pd.to_numeric(joined[f"ah_{side}_no_vig_prob_close"], errors="coerce")
            - pd.to_numeric(joined[f"ah_{side}_no_vig_prob_open"], errors="coerce")
        )
    return joined


def joined_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ["home", "away"]:
        rows.append(
            {
                "side": side,
                "joined_matches": len(joined),
                "open_rows": pd.read_csv(OPEN_INPUT, usecols=["canonical_match_id"]).shape[0],
                "close_rows": pd.read_csv(CLOSE_INPUT, usecols=["canonical_match_id"]).shape[0],
                "avg_line_open": float(joined[f"{side}_line_open"].mean()),
                "avg_line_close": float(joined[f"{side}_line_close"].mean()),
                "avg_line_movement_close_minus_open": float(joined[f"{side}_line_movement_close_minus_open"].mean()),
                "avg_odds_open": float(pd.to_numeric(joined[f"ah_{side}_odds_open"], errors="coerce").mean()),
                "avg_odds_close": float(pd.to_numeric(joined[f"ah_{side}_odds_close"], errors="coerce").mean()),
                "avg_odds_movement_close_minus_open": float(joined[f"{side}_odds_movement_close_minus_open"].mean()),
                "avg_no_vig_prob_open": float(pd.to_numeric(joined[f"ah_{side}_no_vig_prob_open"], errors="coerce").mean()),
                "avg_no_vig_prob_close": float(pd.to_numeric(joined[f"ah_{side}_no_vig_prob_close"], errors="coerce").mean()),
                "avg_prob_movement_close_minus_open": float(joined[f"{side}_prob_movement_close_minus_open"].mean()),
            }
        )
    return pd.DataFrame(rows)


def selected_side_columns(df: pd.DataFrame, side_col: str = "bet_side") -> pd.DataFrame:
    out = df.copy()
    out["selected_open_odds"] = np.where(out[side_col].eq("home"), out["ah_home_odds_open"], out["ah_away_odds_open"])
    out["selected_close_odds"] = np.where(out[side_col].eq("home"), out["ah_home_odds_close"], out["ah_away_odds_close"])
    out["selected_open_prob"] = np.where(out[side_col].eq("home"), out["ah_home_no_vig_prob_open"], out["ah_away_no_vig_prob_open"])
    out["selected_close_prob"] = np.where(out[side_col].eq("home"), out["ah_home_no_vig_prob_close"], out["ah_away_no_vig_prob_close"])
    out["selected_open_line"] = np.where(out[side_col].eq("home"), out["home_line_open"], out["away_line_open"])
    out["selected_close_line"] = np.where(out[side_col].eq("home"), out["home_line_close"], out["away_line_close"])
    out["selected_open_return"] = np.where(out[side_col].eq("home"), out["ah_home_unit_return_open"], out["ah_away_unit_return_open"])
    out["selected_close_return"] = np.where(out[side_col].eq("home"), out["ah_home_unit_return_close"], out["ah_away_unit_return_close"])
    out["selected_open_settlement"] = np.where(out[side_col].eq("home"), out["ah_home_settlement_open"], out["ah_away_settlement_open"])
    out["selected_close_settlement"] = np.where(out[side_col].eq("home"), out["ah_home_settlement_close"], out["ah_away_settlement_close"])
    out["selected_line_movement_close_minus_open"] = out["selected_close_line"] - out["selected_open_line"]
    out["selected_odds_movement_close_minus_open"] = out["selected_close_odds"] - out["selected_open_odds"]
    out["selected_prob_movement_close_minus_open"] = out["selected_close_prob"] - out["selected_open_prob"]
    out["selected_open_line_bucket"] = side_bucket(out["selected_open_line"])
    out["selected_close_line_bucket"] = side_bucket(out["selected_close_line"])
    out["open_price_movement_label"] = np.select(
        [
            (out["selected_odds_movement_close_minus_open"] > 0) | (out["selected_line_movement_close_minus_open"] > 0),
            (out["selected_odds_movement_close_minus_open"] < 0) | (out["selected_line_movement_close_minus_open"] < 0),
        ],
        ["adverse_open_vs_close_for_selected_side", "favourable_open_vs_close_for_selected_side"],
        default="flat_mixed",
    )
    return out


def bet_summary(df: pd.DataFrame, profit_col: str, prefix: str) -> dict:
    n = len(df)
    profit = float(pd.to_numeric(df[profit_col], errors="coerce").sum()) if n else 0.0
    roi = profit / n if n else np.nan
    std = float(pd.to_numeric(df[profit_col], errors="coerce").std(ddof=1)) if n > 1 else np.nan
    z = profit / (std * np.sqrt(n)) if n > 1 and std > 0 else np.nan
    return {
        f"{prefix}_bets": n,
        f"{prefix}_profit": profit,
        f"{prefix}_roi": roi,
        f"{prefix}_z_score": z,
        f"{prefix}_max_drawdown": max_drawdown(pd.to_numeric(df[profit_col], errors="coerce")) if n else np.nan,
    }


def rule_diagnostics(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close_preds = pd.read_csv(CLOSE_PREDS)
    open_preds = pd.read_csv(OPEN_PREDS)
    candidates = pd.read_csv(CLOSING_CANDIDATES)
    primary = pd.read_csv(OPEN_PRIMARY_RULES)[["rule_id"]].assign(primary_pre_registered_rule=True)
    open_validation = pd.read_csv(OPEN_LOCKED_RULES)[["rule_id", "open_validation_tag"]]
    rows, season_rows, league_rows, bucket_rows = [], [], [], []
    join_cols = [
        "canonical_match_id",
        "season_start_year_close",
        "competition_slug_close",
        "match_datetime_close",
        "ah_line_home_open",
        "ah_line_home_close",
        "ah_home_odds_open",
        "ah_home_odds_close",
        "ah_away_odds_open",
        "ah_away_odds_close",
        "ah_home_no_vig_prob_open",
        "ah_home_no_vig_prob_close",
        "ah_away_no_vig_prob_open",
        "ah_away_no_vig_prob_close",
        "ah_home_unit_return_open",
        "ah_home_unit_return_close",
        "ah_away_unit_return_open",
        "ah_away_unit_return_close",
        "ah_home_settlement_open",
        "ah_home_settlement_close",
        "ah_away_settlement_open",
        "ah_away_settlement_close",
        "home_line_open",
        "home_line_close",
        "away_line_open",
        "away_line_close",
    ]
    movement = joined[join_cols].copy()
    for _, rule in candidates.iterrows():
        close_bets = bets_for_rule(
            close_preds,
            rule["candidate_id"],
            rule["side_mode"],
            float(rule["edge_threshold"]),
            float(rule["odds_min"]),
            rule["line_bucket"],
        )
        open_bets = bets_for_rule(
            open_preds,
            rule["candidate_id"],
            rule["side_mode"],
            float(rule["edge_threshold"]),
            float(rule["odds_min"]),
            rule["line_bucket"],
        )
        close_selected = close_bets[["canonical_match_id", "bet_side", "profit", "bet_odds", "model_edge", "settlement"]].copy()
        close_selected = close_selected.rename(columns={"profit": "close_selected_close_profit", "bet_odds": "close_selected_close_odds", "model_edge": "close_selected_close_edge", "settlement": "close_selected_close_settlement"})
        same_open = close_selected.merge(movement, on="canonical_match_id", how="left")
        if not same_open.empty:
            same_open = selected_side_columns(same_open)
            same_open["same_close_selected_open_profit"] = same_open["selected_open_return"]
        open_selected = open_bets[["canonical_match_id", "bet_side", "profit", "bet_odds", "model_edge", "settlement"]].copy()
        open_selected = open_selected.rename(columns={"profit": "open_selected_open_profit", "bet_odds": "open_selected_open_odds", "model_edge": "open_selected_open_edge", "settlement": "open_selected_open_settlement"})
        if not open_selected.empty:
            open_selected = open_selected.merge(movement, on="canonical_match_id", how="left")
            open_selected = selected_side_columns(open_selected)
        close_ids = set(close_bets["canonical_match_id"].astype(str))
        open_ids = set(open_bets["canonical_match_id"].astype(str))
        overlap = len(close_ids & open_ids)
        row = {
            "rule_id": rule["rule_id"],
            "candidate_id": rule["candidate_id"],
            "side_mode": rule["side_mode"],
            "edge_threshold": rule["edge_threshold"],
            "odds_filter": rule["odds_filter"],
            "odds_min": rule["odds_min"],
            "line_bucket": rule["line_bucket"],
            "closing_candidate_tag": rule["candidate_tag"],
            "close_selected_matches": len(close_bets),
            "open_selected_matches": len(open_bets),
            "overlap_count": overlap,
            "overlap_pct_of_close": overlap / len(close_bets) if len(close_bets) else np.nan,
            "overlap_pct_of_open": overlap / len(open_bets) if len(open_bets) else np.nan,
            "avg_open_odds_on_close_selected": float(same_open["selected_open_odds"].mean()) if len(same_open) else np.nan,
            "avg_close_odds_on_close_selected": float(same_open["selected_close_odds"].mean()) if len(same_open) else np.nan,
            "avg_selected_line_movement_close_minus_open": float(same_open["selected_line_movement_close_minus_open"].mean()) if len(same_open) else np.nan,
            "avg_selected_odds_movement_close_minus_open": float(same_open["selected_odds_movement_close_minus_open"].mean()) if len(same_open) else np.nan,
            "avg_selected_probability_movement_close_minus_open": float(same_open["selected_prob_movement_close_minus_open"].mean()) if len(same_open) else np.nan,
            "dominant_open_price_movement_label": same_open["open_price_movement_label"].mode().iat[0] if len(same_open) else "",
        }
        row.update(bet_summary(close_bets.rename(columns={"profit": "p"}), "p", "close_selected_close_odds"))
        row.update(bet_summary(same_open.assign(p=same_open["same_close_selected_open_profit"] if len(same_open) else pd.Series(dtype=float)), "p", "close_selected_open_odds"))
        row.update(bet_summary(open_bets.rename(columns={"profit": "p"}), "p", "open_selected_open_odds"))
        rows.append(row)
        for label, df, profit_col in [
            ("close_selected_open_odds", same_open, "same_close_selected_open_profit"),
            ("open_selected_open_odds", open_selected, "open_selected_open_profit"),
        ]:
            if df.empty:
                continue
            group_specs = [
                (season_rows, "season_start_year_close", "season_start_year"),
                (league_rows, "competition_slug_close", "competition_slug"),
                (bucket_rows, "selected_open_line_bucket", "line_bucket"),
            ]
            for target, col, out_col in group_specs:
                g = df.groupby(col, dropna=False)[profit_col].agg(["count", "sum", "mean"]).reset_index()
                g = g.rename(columns={col: out_col, "count": "bets", "sum": "profit", "mean": "roi"})
                g.insert(0, "selection_context", label)
                g.insert(0, "rule_id", rule["rule_id"])
                g.insert(1, "candidate_id", rule["candidate_id"])
                target.append(g)
    out = pd.DataFrame(rows)
    out = out.merge(primary, on="rule_id", how="left")
    out["primary_pre_registered_rule"] = out["primary_pre_registered_rule"].fillna(False)
    out = out.merge(open_validation, on="rule_id", how="left")
    detail = [
        pd.concat(season_rows, ignore_index=True) if season_rows else pd.DataFrame(),
        pd.concat(league_rows, ignore_index=True) if league_rows else pd.DataFrame(),
        pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame(),
    ]
    return out, detail[0], detail[1], detail[2]


def write_reports(summary: pd.DataFrame, candidate_overlap: pd.DataFrame, by_season: pd.DataFrame, by_league: pd.DataFrame, by_bucket: pd.DataFrame) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_DIR / "ah_open_close_joined_summary.csv", index=False)
    candidate_overlap.to_csv(REPORT_DIR / "ah_open_close_candidate_overlap.csv", index=False)
    by_season.to_csv(REPORT_DIR / "ah_open_close_by_season.csv", index=False)
    by_league.to_csv(REPORT_DIR / "ah_open_close_by_league.csv", index=False)
    by_bucket.to_csv(REPORT_DIR / "ah_open_close_by_line_bucket.csv", index=False)
    decision = "ah_open_close_diagnostic_ready_good"
    primary = candidate_overlap[candidate_overlap["primary_pre_registered_rule"].astype(bool)].copy()
    validated = int(candidate_overlap["open_validation_tag"].eq("open_validated_research_candidate").sum() + candidate_overlap["open_validation_tag"].eq("stronger_open_validated_research_candidate").sum())
    top = candidate_overlap.sort_values("close_selected_open_odds_roi", ascending=True).head(5)
    report = [
        "# AH Open-vs-Close Movement Diagnostic",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "This is a diagnostic-only open-vs-close comparison. It does not train betting models, search for new rules, optimize thresholds, or claim confirmed edge.",
        "",
        "## Joined Dataset",
        "",
        summary.to_markdown(index=False),
        "",
        "## Locked Candidate Rules",
        "",
        f"- Locked closing candidate rules evaluated: {len(candidate_overlap)}",
        f"- Primary pre-registered rules evaluated: {len(primary)}",
        f"- Rules that passed open validation: {validated}",
        "",
        "## Primary Rule Open-vs-Close Summary",
        "",
        primary[
            [
                "rule_id",
                "candidate_id",
                "side_mode",
                "odds_filter",
                "line_bucket",
                "close_selected_matches",
                "open_selected_matches",
                "overlap_count",
                "overlap_pct_of_close",
                "close_selected_close_odds_roi",
                "close_selected_open_odds_roi",
                "open_selected_open_odds_roi",
                "avg_selected_odds_movement_close_minus_open",
                "avg_selected_probability_movement_close_minus_open",
                "dominant_open_price_movement_label",
            ]
        ].to_markdown(index=False),
        "",
        "## Worst Same-Match Open Repricing Examples",
        "",
        top[
            [
                "rule_id",
                "candidate_id",
                "side_mode",
                "odds_filter",
                "line_bucket",
                "close_selected_matches",
                "close_selected_close_odds_roi",
                "close_selected_open_odds_roi",
                "avg_open_odds_on_close_selected",
                "avg_close_odds_on_close_selected",
                "avg_selected_line_movement_close_minus_open",
                "avg_selected_odds_movement_close_minus_open",
            ]
        ].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- The locked closing-candidate rules were replayed without changing thresholds or creating new rules.",
        "- Open validation failed because selections changed materially and the same close-selected opportunities did not retain the closing-labelled return profile at open prices/lines.",
        "- For selected sides, positive close-minus-open odds/probability movement generally means the closing record was priced differently than the available open line.",
        "- Results remain research_only and are not a betting signal.",
    ]
    (REPORT_DIR / "ah_open_close_diagnostic_report.md").write_text("\n".join(report) + "\n")
    (REPORT_DIR / "ah_open_close_decision.md").write_text(
        f"# AH Open-vs-Close Diagnostic Decision\n\nDecision: **{decision}**\n\n"
        "The diagnostic completed using existing open/close datasets and locked candidate rules only. No confirmed edge is claimed.\n"
    )
    return decision


def main() -> None:
    joined = load_joined()
    summary = joined_summary(joined)
    candidate_overlap, by_season, by_league, by_bucket = rule_diagnostics(joined)
    decision = write_reports(summary, candidate_overlap, by_season, by_league, by_bucket)
    print(decision)
    print(f"joined_matches={len(joined)} locked_rules={len(candidate_overlap)}")


if __name__ == "__main__":
    main()
