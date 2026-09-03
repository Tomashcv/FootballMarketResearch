from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data/processed/super_csvs/research_ready/football_data"
OUT_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_clubelo_understat_transfermarkt"
REPORT_DIR = ROOT / "outputs/reports/football_data_feature_merge"

CLUBELO = ROOT / "data/processed/feature_blocks/clubelo/clubelo_features_footiqo_top5_v1_locked.csv"
UNDERSTAT = ROOT / "data/processed/feature_blocks/understat/understat_features_footiqo_top5_v1_locked.csv"
TRANSFERMARKT = ROOT / "data/processed/feature_blocks/transfermarkt/transfermarkt_features_footiqo_top5_v1_locked.csv"

MARKETS = {
    "1x2": {
        "input": BASE_DIR / "super_1x2_football_data_top5_research_v1_locked.csv",
        "output": OUT_DIR / "super_1x2_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv",
        "required": True,
        "market_no_vig": ["x1_home_no_vig_prob", "x1_draw_no_vig_prob", "x1_away_no_vig_prob"],
    },
    "ah": {
        "input": BASE_DIR / "super_ah_football_data_top5_research_v1_locked.csv",
        "output": OUT_DIR / "super_ah_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv",
        "required": True,
        "market_no_vig": ["ah_home_no_vig_prob", "ah_away_no_vig_prob"],
    },
    "ah_open": {
        "input": BASE_DIR / "super_ah_open_football_data_top5_research_v1_locked.csv",
        "output": OUT_DIR / "super_ah_open_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv",
        "required": False,
        "market_no_vig": ["ah_home_no_vig_prob", "ah_away_no_vig_prob"],
    },
    "ah_close": {
        "input": BASE_DIR / "super_ah_close_football_data_top5_research_v1_locked.csv",
        "output": OUT_DIR / "super_ah_close_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv",
        "required": False,
        "market_no_vig": ["ah_home_no_vig_prob", "ah_away_no_vig_prob"],
    },
}

FEATURE_BLOCKS = {
    "clubelo": CLUBELO,
    "understat": UNDERSTAT,
    "transfermarkt": TRANSFERMARKT,
}

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
    "target_home_win",
    "target_draw",
    "target_away_win",
    "x1_home_odds",
    "x1_draw_odds",
    "x1_away_odds",
    "x1_odds_source",
    "x1_odds_timing_label",
    "x1_home_raw_prob",
    "x1_draw_raw_prob",
    "x1_away_raw_prob",
    "x1_overround",
    "ah_line_home",
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

FORBIDDEN_SUBSTRINGS = [
    "current_club",
    "current_value",
    "lineup",
    "appearance",
]


def load_feature_blocks() -> dict[str, pd.DataFrame]:
    blocks = {}
    for name, path in FEATURE_BLOCKS.items():
        df = pd.read_csv(path)
        df["canonical_match_id"] = df["canonical_match_id"].astype("int64")
        if df["canonical_match_id"].duplicated().any():
            raise ValueError(f"{name} feature block has duplicate canonical_match_id")
        blocks[name] = df
    return blocks


def merge_market(market: str, spec: dict, blocks: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict, dict, dict]:
    base = pd.read_csv(spec["input"], dtype={"competition_code": str})
    base["canonical_match_id"] = base["canonical_match_id"].astype("int64")
    before_rows = len(base)
    before_cols = len(base.columns)
    merged = base.copy()
    added_cols = []
    for block_name, block in blocks.items():
        conflicts = sorted((set(merged.columns) & set(block.columns)) - {"canonical_match_id"})
        if conflicts:
            raise ValueError(f"{market}: {block_name} column conflicts would overwrite existing columns: {conflicts[:20]}")
        merged = merged.merge(block, on="canonical_match_id", how="left", validate="one_to_one")
        added_cols.extend([c for c in block.columns if c != "canonical_match_id"])
    spec["output"].parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(spec["output"], index=False)

    row_count = {
        "market": market,
        "input_file": str(spec["input"].relative_to(ROOT)),
        "output_file": str(spec["output"].relative_to(ROOT)),
        "row_count_before": before_rows,
        "row_count_after": len(merged),
        "column_count_before": before_cols,
        "column_count_after": len(merged.columns),
        "row_count_preserved": before_rows == len(merged),
        "canonical_match_id_unique": not merged["canonical_match_id"].duplicated().any(),
        "duplicate_canonical_match_id_count": int(merged["canonical_match_id"].duplicated().sum()),
        "row_multiplication_detected": before_rows != len(merged),
    }

    understat_stale = (
        int(merged["understat_match_after_source_max_date_flag"].fillna(False).astype(bool).sum())
        if "understat_match_after_source_max_date_flag" in merged.columns
        else 0
    )
    suspicious_tm = int(
        (
            (pd.to_numeric(merged.get("home_tm_total_market_value"), errors="coerce") > 2_000_000_000)
            | (pd.to_numeric(merged.get("away_tm_total_market_value"), errors="coerce") > 2_000_000_000)
        ).sum()
    )
    coverage = {
        "market": market,
        "row_count": len(merged),
        "clubelo_both_found_rate": float(merged["clubelo_both_found_flag"].mean()) if "clubelo_both_found_flag" in merged.columns else 0.0,
        "clubelo_missing_feature_rows": int(merged["clubelo_both_found_flag"].isna().sum()) if "clubelo_both_found_flag" in merged.columns else len(merged),
        "understat_both_found_rate": float(merged["understat_both_found_flag"].mean()) if "understat_both_found_flag" in merged.columns else 0.0,
        "understat_missing_feature_rows": int(merged["understat_both_found_flag"].isna().sum()) if "understat_both_found_flag" in merged.columns else len(merged),
        "understat_after_source_max_date_rows": understat_stale,
        "transfermarkt_both_value_found_rate": float(merged["tm_both_value_found_flag"].mean()) if "tm_both_value_found_flag" in merged.columns else 0.0,
        "transfermarkt_missing_feature_rows": int(merged["tm_both_value_found_flag"].isna().sum()) if "tm_both_value_found_flag" in merged.columns else len(merged),
        "transfermarkt_suspicious_value_outlier_rows_gt_2b": suspicious_tm,
    }

    ah_settlement_ok = True
    if market.startswith("ah"):
        ah_settlement_ok = {"ah_home_unit_return", "ah_away_unit_return", "ah_home_settlement", "ah_away_settlement"}.issubset(merged.columns)
    timing_cols = [c for c in ["x1_odds_timing_label", "ah_timing_label"] if c in merged.columns]
    leakage = {
        "market": market,
        "row_count_preserved": before_rows == len(merged),
        "canonical_match_id_unique": not merged["canonical_match_id"].duplicated().any(),
        "no_row_multiplication": before_rows == len(merged),
        "no_current_club_columns_present": not any("current_club" in c.lower() for c in merged.columns),
        "no_current_value_columns_present": not any("current_value" in c.lower() for c in merged.columns),
        "no_game_lineups_columns_present": not any("lineup" in c.lower() for c in merged.columns),
        "no_same_match_appearance_features_present": not any("appearance" in c.lower() for c in merged.columns),
        "no_current_fixture_xg_stats_features_present": not any(c in {"xG", "xGA", "npxG", "npxGA"} for c in added_cols),
        "odds_timing_or_timing_labels_retained": bool(timing_cols),
        "ah_settlement_columns_preserved": ah_settlement_ok,
        "classification_research_only": bool(merged["classification"].eq("research_only").all()) if "classification" in merged.columns else False,
        "no_suspicious_transfermarkt_outliers": suspicious_tm == 0,
    }
    leakage["leakage_check_pass"] = all(bool(v) for k, v in leakage.items() if k != "market")
    return merged, row_count, coverage, leakage


def is_allowed_feature(col: str, market: str, market_no_vig: list[str]) -> tuple[bool, str]:
    if col in market_no_vig:
        return True, "Market-specific no-vig probability."
    if col in FORBIDDEN_EXACT or col.startswith("target_"):
        return False, "Identifier, target, raw odds, settlement/audit, team/source, or outcome column."
    lower = col.lower()
    if any(token in lower for token in FORBIDDEN_SUBSTRINGS):
        return False, "Forbidden leakage-prone source field."
    if "team_name" in lower or "team_raw" in lower or "team_normalized" in lower or col.endswith("_team_id"):
        return False, "Team identifier/name forbidden as model feature."
    if col.startswith(("home_clubelo_", "away_clubelo_", "clubelo_")) and col != "clubelo_source_file":
        return True, "Locked ClubElo strict-before-match feature."
    if col.startswith(("home_understat_", "away_understat_", "understat_")) and col not in FORBIDDEN_EXACT:
        return True, "Locked Understat lagged feature or staleness/found flag."
    if col.startswith(("home_tm_", "away_tm_", "tm_")) and col not in {"home_tm_club_id", "away_tm_club_id"}:
        return True, "Locked Transfermarkt point-in-time feature or staleness/found flag."
    return False, "Not part of approved football-data market/external feature blocks."


def allowlist_and_forbidden(outputs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    allow_rows = []
    forbidden_rows = []
    for market, df in outputs.items():
        market_no_vig = MARKETS[market]["market_no_vig"]
        for col in df.columns:
            allowed, reason = is_allowed_feature(col, market, market_no_vig)
            if allowed:
                allow_rows.append(
                    {
                        "market": market,
                        "column": col,
                        "feature_block": "football_data_clubelo_understat_transfermarkt",
                        "allowlist_status": "frozen_allowed",
                        "review_note": reason,
                    }
                )
            elif reason != "Not part of approved football-data market/external feature blocks." or col in df.columns:
                if col in FORBIDDEN_EXACT or col.startswith("target_") or any(token in col.lower() for token in FORBIDDEN_SUBSTRINGS) or col in {"classification"}:
                    forbidden_rows.append(
                        {
                            "market": market,
                            "column": col,
                            "role": "forbidden_or_audit",
                            "feature_block": "football_data_clubelo_understat_transfermarkt",
                            "forbidden_status": "forbidden_as_model_feature",
                            "leakage_note": reason,
                        }
                    )
    return pd.DataFrame(allow_rows), pd.DataFrame(forbidden_rows)


def write_reports(
    row_counts: pd.DataFrame,
    coverage: pd.DataFrame,
    leakage: pd.DataFrame,
    allow: pd.DataFrame,
    forbidden: pd.DataFrame,
    decision: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    row_counts.to_csv(REPORT_DIR / "football_data_feature_merge_row_counts.csv", index=False)
    coverage.to_csv(REPORT_DIR / "football_data_feature_merge_coverage.csv", index=False)
    allow.to_csv(REPORT_DIR / "football_data_feature_merge_feature_allowlist.csv", index=False)
    forbidden.to_csv(REPORT_DIR / "football_data_feature_merge_forbidden_columns.csv", index=False)
    leakage.to_csv(REPORT_DIR / "football_data_feature_merge_leakage_checks.csv", index=False)

    report = [
        "# Football-Data Feature Merge Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Merged locked ClubElo, Understat, and Transfermarkt feature blocks into locked football-data 1X2/AH research CSVs.",
        "",
        "Merge policy:",
        "- Left join from each football-data market CSV.",
        "- Join key: `canonical_match_id` only.",
        "- No existing columns overwritten.",
        "- Classification remains `research_only`.",
        "",
        "Outputs:",
    ]
    for _, row in row_counts.iterrows():
        report.append(f"- `{row['output_file']}`")
    report.extend(
        [
            "",
            "Validation summary:",
            f"- Row counts preserved: {bool(row_counts['row_count_preserved'].all())}",
            f"- Duplicate canonical IDs present: {bool((row_counts['duplicate_canonical_match_id_count'] > 0).any())}",
            f"- Leakage checks passing: {int(leakage['leakage_check_pass'].sum())}/{len(leakage)}",
            f"- Minimum ClubElo both-found coverage: {coverage['clubelo_both_found_rate'].min():.4f}",
            f"- Minimum Understat both-found coverage: {coverage['understat_both_found_rate'].min():.4f}",
            f"- Minimum Transfermarkt both-value coverage: {coverage['transfermarkt_both_value_found_rate'].min():.4f}",
            "",
            "No modeling, value search, threshold optimization, or confirmed-edge claim was performed.",
        ]
    )
    (REPORT_DIR / "football_data_feature_merge_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision_md = [
        "# Football-Data Feature Merge Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "These outputs are research-only datasets. No modeling or value search was performed, and no confirmed edge is claimed.",
    ]
    (REPORT_DIR / "football_data_feature_merge_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = load_feature_blocks()
    outputs: dict[str, pd.DataFrame] = {}
    row_counts = []
    coverage = []
    leakage = []
    missing_required = []
    for market, spec in MARKETS.items():
        if not spec["input"].exists():
            if spec["required"]:
                missing_required.append(str(spec["input"].relative_to(ROOT)))
            continue
        merged, rc, cov, leak = merge_market(market, spec, blocks)
        outputs[market] = merged
        row_counts.append(rc)
        coverage.append(cov)
        leakage.append(leak)
    row_counts_df = pd.DataFrame(row_counts)
    coverage_df = pd.DataFrame(coverage)
    leakage_df = pd.DataFrame(leakage)
    allow, forbidden = allowlist_and_forbidden(outputs)
    if (
        missing_required
        or row_counts_df.empty
        or not row_counts_df["row_count_preserved"].all()
        or (row_counts_df["duplicate_canonical_match_id_count"] > 0).any()
        or row_counts_df["row_multiplication_detected"].any()
        or not leakage_df["leakage_check_pass"].all()
    ):
        decision = "football_data_feature_merge_failed"
    elif coverage_df[["clubelo_both_found_rate", "understat_both_found_rate", "transfermarkt_both_value_found_rate"]].min().min() < 0.90:
        decision = "football_data_feature_merge_ready_needs_review"
    else:
        decision = "football_data_feature_merge_ready_good"
    write_reports(row_counts_df, coverage_df, leakage_df, allow, forbidden, decision)
    print(decision)
    print(f"outputs={len(outputs)} rows_preserved={bool(row_counts_df['row_count_preserved'].all()) if not row_counts_df.empty else False}")


if __name__ == "__main__":
    main()
