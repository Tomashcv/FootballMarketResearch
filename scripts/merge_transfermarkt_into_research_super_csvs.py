from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/clubelo_understat"
OUT_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/clubelo_understat_transfermarkt"
REPORT_DIR = ROOT / "outputs/reports/super_csvs/transfermarkt_merge"
TM_PATH = ROOT / "data/processed/feature_blocks/transfermarkt/transfermarkt_features_footiqo_top5_v1_locked.csv"
TM_LOCKED_CHECKS = ROOT / "outputs/reports/feature_blocks/transfermarkt/transfermarkt_locked_leakage_checks.csv"
BASE_ALLOWLIST = ROOT / "outputs/reports/super_csvs/understat_merge/understat_merge_feature_allowlist.csv"
BASE_FORBIDDEN = ROOT / "outputs/reports/super_csvs/understat_merge/understat_merge_forbidden_columns.csv"

MARKETS = {
    "btts": {
        "input": "super_btts_footiqo_top5_clubelo_understat_research_v1.csv",
        "output": "super_btts_footiqo_top5_clubelo_understat_transfermarkt_research_v1.csv",
    },
    "ou15": {
        "input": "super_ou15_footiqo_top5_clubelo_understat_research_v1.csv",
        "output": "super_ou15_footiqo_top5_clubelo_understat_transfermarkt_research_v1.csv",
    },
    "ou25": {
        "input": "super_ou25_footiqo_top5_clubelo_understat_research_v1.csv",
        "output": "super_ou25_footiqo_top5_clubelo_understat_transfermarkt_research_v1.csv",
    },
    "ou35": {
        "input": "super_ou35_footiqo_top5_clubelo_understat_research_v1.csv",
        "output": "super_ou35_footiqo_top5_clubelo_understat_transfermarkt_research_v1.csv",
    },
    "ou45": {
        "input": "super_ou45_footiqo_top5_clubelo_understat_research_v1.csv",
        "output": "super_ou45_footiqo_top5_clubelo_understat_transfermarkt_research_v1.csv",
    },
}

FORBIDDEN_TM_COLUMNS = {
    "home_tm_club_id",
    "away_tm_club_id",
}


def load_transfermarkt() -> pd.DataFrame:
    df = pd.read_csv(TM_PATH)
    df["canonical_match_id"] = df["canonical_match_id"].astype("int64")
    if df["canonical_match_id"].duplicated().any():
        raise ValueError("Transfermarkt feature block has duplicate canonical_match_id")
    return df


def transfermarkt_model_features(df: pd.DataFrame) -> list[str]:
    allowed = []
    for c in df.columns:
        if c == "canonical_match_id" or c in FORBIDDEN_TM_COLUMNS:
            continue
        if c.startswith(("home_tm_", "away_tm_", "tm_")):
            allowed.append(c)
    return allowed


def merge_market(market: str, spec: dict, tm_df: pd.DataFrame, tm_allowed: list[str]) -> tuple[dict, dict, dict]:
    base = pd.read_csv(BASE_DIR / spec["input"], dtype={"competition_code": str})
    base["canonical_match_id"] = base["canonical_match_id"].astype("int64")
    before_rows = len(base)
    conflicts = sorted((set(base.columns) & set(tm_df.columns)) - {"canonical_match_id"})
    if conflicts:
        raise ValueError(f"{market}: Transfermarkt column conflicts would overwrite existing columns: {conflicts[:20]}")
    merged = base.merge(tm_df, on="canonical_match_id", how="left", validate="one_to_one")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DIR / spec["output"], index=False)

    suspicious = int(
        (
            (merged["home_tm_total_market_value"] > 2_000_000_000)
            | (merged["away_tm_total_market_value"] > 2_000_000_000)
        ).sum()
    )
    row_count = {
        "market": market,
        "input_file": spec["input"],
        "output_file": spec["output"],
        "row_count_before": before_rows,
        "row_count_after": len(merged),
        "column_count_before": len(base.columns),
        "column_count_after": len(merged.columns),
        "row_count_preserved": before_rows == len(merged),
        "canonical_match_id_unique": not merged["canonical_match_id"].duplicated().any(),
        "duplicate_canonical_match_id_count": int(merged["canonical_match_id"].duplicated().sum()),
        "row_multiplication_detected": before_rows != len(merged),
    }
    coverage = {
        "market": market,
        "row_count": len(merged),
        "missing_transfermarkt_feature_rows": int(merged["tm_both_value_found_flag"].isna().sum()),
        "tm_both_value_found_rate": float(merged["tm_both_value_found_flag"].mean()),
        "tm_home_value_found_rate": float(merged["home_tm_value_found_flag"].mean()),
        "tm_away_value_found_rate": float(merged["away_tm_value_found_flag"].mean()),
        "tm_both_missing_rows": int((~merged["tm_both_value_found_flag"].astype(bool)).sum()),
        "home_tm_players_coverage_median": float(merged["home_tm_players_coverage_count"].median()),
        "away_tm_players_coverage_median": float(merged["away_tm_players_coverage_count"].median()),
    }
    leakage = {
        "market": market,
        "no_target_leakage_columns_added": not any(c.startswith("target_") for c in tm_allowed),
        "no_current_club_columns_present": not any("current_club" in c for c in merged.columns),
        "no_current_value_columns_present": not any("current_value" in c for c in merged.columns),
        "no_game_lineups_columns_present": not any("lineup" in c.lower() for c in merged.columns),
        "no_same_match_appearance_features_present": not any("appearance" in c.lower() for c in merged.columns),
        "no_team_names_added_as_model_features": not any("team_name" in c or "team_normalized" in c for c in tm_allowed),
        "no_source_identifiers_as_model_features": not bool(set(tm_allowed) & FORBIDDEN_TM_COLUMNS),
        "no_canonical_match_id_as_model_feature": "canonical_match_id" not in tm_allowed,
        "suspicious_valuation_outliers_absent": suspicious == 0,
        "odds_timing_remains_unknown": bool(merged["odds_timing_flag"].eq("unknown").all()) if "odds_timing_flag" in merged.columns else False,
        "classification": "research_only",
    }
    leakage["leakage_check_pass"] = all(bool(v) for k, v in leakage.items() if k not in {"market", "classification"})
    return row_count, coverage, leakage


def build_allowlist_and_forbidden(tm_allowed: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if BASE_ALLOWLIST.exists():
        base_allow = pd.read_csv(BASE_ALLOWLIST)
    else:
        base_allow = pd.DataFrame(columns=["market", "column", "feature_block", "allowlist_status", "review_note"])
    if BASE_FORBIDDEN.exists():
        base_forbidden = pd.read_csv(BASE_FORBIDDEN)
    else:
        base_forbidden = pd.DataFrame(columns=["market", "column", "role", "feature_block", "forbidden_status", "leakage_note"])
    allow_rows = []
    forbidden_rows = []
    for market in MARKETS:
        for col in tm_allowed:
            allow_rows.append(
                {
                    "market": market,
                    "column": col,
                    "feature_block": "transfermarkt_point_in_time",
                    "allowlist_status": "frozen_allowed",
                    "review_note": "Transfermarkt point-in-time feature built from dated valuations/transfers/appearances strictly before match_date.",
                }
            )
        for col in sorted(FORBIDDEN_TM_COLUMNS):
            forbidden_rows.append(
                {
                    "market": market,
                    "column": col,
                    "role": "source_identifier",
                    "feature_block": "transfermarkt_point_in_time",
                    "forbidden_status": "forbidden_as_model_feature",
                    "leakage_note": "Transfermarkt club ID retained for audit only; not a model feature.",
                }
            )
    return (
        pd.concat([base_allow, pd.DataFrame(allow_rows)], ignore_index=True),
        pd.concat([base_forbidden, pd.DataFrame(forbidden_rows)], ignore_index=True),
    )


def write_reports(row_counts: pd.DataFrame, coverage: pd.DataFrame, leakage: pd.DataFrame, decision: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    row_counts.to_csv(REPORT_DIR / "transfermarkt_merge_row_counts.csv", index=False)
    coverage.to_csv(REPORT_DIR / "transfermarkt_merge_coverage.csv", index=False)
    leakage.to_csv(REPORT_DIR / "transfermarkt_merge_leakage_checks.csv", index=False)

    report = [
        "# Transfermarkt Super CSV Merge Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Scope: merge locked point-in-time Transfermarkt feature block into Footiqo top-5 ClubElo + Understat research CSVs. No modeling, value search, threshold optimization, or extra source join was run.",
        "",
        "## Merge Policy",
        "- Left join from each market CSV to Transfermarkt features.",
        "- Join key: `canonical_match_id` only.",
        "- Existing columns are not overwritten.",
        "- Transfermarkt club IDs are retained only as audit identifiers and are forbidden as model features.",
        "- Classification remains `research_only` because odds timing remains unknown.",
        "",
        "## Outputs",
    ]
    for _, r in row_counts.iterrows():
        report.append(f"- data/processed/super_csvs/research_ready_plus/clubelo_understat_transfermarkt/{r['output_file']}")
    report.extend(
        [
            "",
            "## Summary",
            f"- All row counts preserved: {bool(row_counts['row_count_preserved'].all())}",
            f"- Any duplicate canonical IDs: {bool((row_counts['duplicate_canonical_match_id_count'] > 0).any())}",
            f"- Minimum Transfermarkt both-team value coverage: {coverage['tm_both_value_found_rate'].min():.4f}",
            f"- Leakage checks passing: {int(leakage['leakage_check_pass'].sum())}/{len(leakage)}",
            "",
            "No confirmed edge is claimed.",
        ]
    )
    (REPORT_DIR / "transfermarkt_merge_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    decision_md = [
        "# Transfermarkt Merge Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "The Transfermarkt-enhanced market CSVs are research-only datasets. They preserve row counts, canonical ID uniqueness, and existing market availability filters.",
        "",
        "No modeling was performed and no confirmed edge is claimed.",
    ]
    (REPORT_DIR / "transfermarkt_merge_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tm_df = load_transfermarkt()
    tm_allowed = transfermarkt_model_features(tm_df)
    row_counts, coverage, leakage = [], [], []
    for market, spec in MARKETS.items():
        rc, cov, leak = merge_market(market, spec, tm_df, tm_allowed)
        row_counts.append(rc)
        coverage.append(cov)
        leakage.append(leak)
    row_counts_df = pd.DataFrame(row_counts)
    coverage_df = pd.DataFrame(coverage)
    leakage_df = pd.DataFrame(leakage)
    allow, forbidden = build_allowlist_and_forbidden(tm_allowed)
    allow.to_csv(REPORT_DIR / "transfermarkt_merge_feature_allowlist.csv", index=False)
    forbidden.to_csv(REPORT_DIR / "transfermarkt_merge_forbidden_columns.csv", index=False)

    upstream_checks_pass = True
    if TM_LOCKED_CHECKS.exists():
        upstream = pd.read_csv(TM_LOCKED_CHECKS)
        upstream_checks_pass = bool(upstream["status"].eq("pass").all())

    if (
        not upstream_checks_pass
        or not row_counts_df["row_count_preserved"].all()
        or (row_counts_df["duplicate_canonical_match_id_count"] > 0).any()
        or row_counts_df["row_multiplication_detected"].any()
        or not leakage_df["leakage_check_pass"].all()
    ):
        decision = "transfermarkt_super_csv_merge_failed"
    elif coverage_df["tm_both_value_found_rate"].min() < 0.90:
        decision = "transfermarkt_super_csv_merge_ready_needs_review"
    else:
        decision = "transfermarkt_super_csv_merge_ready_good"
    write_reports(row_counts_df, coverage_df, leakage_df, decision)
    print(decision)


if __name__ == "__main__":
    main()
