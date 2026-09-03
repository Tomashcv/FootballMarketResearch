from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
READY_DIR = ROOT / "data/processed/super_csvs/research_ready"
OUT_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/clubelo"
REPORT_DIR = ROOT / "outputs/reports/super_csvs/clubelo_merge"
CLUBELO_PATH = ROOT / "data/processed/feature_blocks/clubelo/clubelo_features_footiqo_top5_v1_locked.csv"
BASE_ALLOWLIST_PATH = ROOT / "outputs/reports/super_csvs/review/super_csv_model_feature_allowlist.csv"
BASE_FORBIDDEN_PATH = ROOT / "outputs/reports/super_csvs/review/super_csv_forbidden_columns.csv"

MARKETS = {
    "btts": {
        "input": "super_btts_footiqo_top5_research_v1.csv",
        "output": "super_btts_footiqo_top5_clubelo_research_v1.csv",
    },
    "ou15": {
        "input": "super_ou15_footiqo_top5_research_v1.csv",
        "output": "super_ou15_footiqo_top5_clubelo_research_v1.csv",
    },
    "ou25": {
        "input": "super_ou25_footiqo_top5_research_v1.csv",
        "output": "super_ou25_footiqo_top5_clubelo_research_v1.csv",
    },
    "ou35": {
        "input": "super_ou35_footiqo_top5_research_v1.csv",
        "output": "super_ou35_footiqo_top5_clubelo_research_v1.csv",
    },
    "ou45": {
        "input": "super_ou45_footiqo_top5_research_v1.csv",
        "output": "super_ou45_footiqo_top5_clubelo_research_v1.csv",
    },
}

CLUBELO_FEATURE_COLUMNS = [
    "home_clubelo_rating",
    "away_clubelo_rating",
    "clubelo_diff_home_minus_away",
    "home_clubelo_days_stale",
    "away_clubelo_days_stale",
    "home_clubelo_found_flag",
    "away_clubelo_found_flag",
    "clubelo_both_found_flag",
    "clubelo_source_file",
]

SAME_MATCH_FORBIDDEN = {
    "home_goals",
    "away_goals",
    "result_1x2",
    "FTHG",
    "FTAG",
    "FTR",
    "total_goals",
    "total_corners",
    "total_yellow_cards",
    "total_shots",
    "total_shots_on_target",
}
TEAM_NAME_COLUMNS = {"home_team_raw", "away_team_raw", "homeTeam", "awayTeam"}


def load_clubelo() -> pd.DataFrame:
    clubelo = pd.read_csv(CLUBELO_PATH)
    clubelo["canonical_match_id"] = clubelo["canonical_match_id"].astype("int64")
    if clubelo["canonical_match_id"].duplicated().any():
        raise ValueError("ClubElo feature block has duplicate canonical_match_id")
    return clubelo[["canonical_match_id"] + CLUBELO_FEATURE_COLUMNS].copy()


def merge_market(market: str, spec: dict, clubelo: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, object]]:
    base = pd.read_csv(READY_DIR / spec["input"], dtype={"competition_code": str})
    base["canonical_match_id"] = base["canonical_match_id"].astype("int64")
    before_rows = len(base)
    before_cols = len(base.columns)
    conflicts = sorted(set(base.columns).intersection(CLUBELO_FEATURE_COLUMNS))
    if conflicts:
        raise ValueError(f"{market}: ClubElo column conflicts would overwrite existing columns: {conflicts}")
    merged = base.merge(clubelo, on="canonical_match_id", how="left", validate="one_to_one")
    out_path = OUT_DIR / spec["output"]
    merged.to_csv(out_path, index=False)
    after_rows = len(merged)
    row_count = {
        "market": market,
        "input_file": spec["input"],
        "output_file": spec["output"],
        "row_count_before": before_rows,
        "row_count_after": after_rows,
        "column_count_before": before_cols,
        "column_count_after": len(merged.columns),
        "row_count_preserved": before_rows == after_rows,
        "canonical_match_id_unique": not merged["canonical_match_id"].duplicated().any(),
        "duplicate_canonical_match_id_count": int(merged["canonical_match_id"].duplicated().sum()),
        "row_multiplication_detected": after_rows != before_rows,
        "column_conflicts": "; ".join(conflicts),
    }
    clubelo_missing_any = merged[CLUBELO_FEATURE_COLUMNS].isna().any(axis=1)
    coverage = {
        "market": market,
        "row_count": after_rows,
        "missing_any_clubelo_feature_rows": int(clubelo_missing_any.sum()),
        "home_clubelo_found_rate": float(merged["home_clubelo_found_flag"].mean()),
        "away_clubelo_found_rate": float(merged["away_clubelo_found_flag"].mean()),
        "clubelo_both_found_rate": float(merged["clubelo_both_found_flag"].mean()),
        "clubelo_both_missing_rows": int((~merged["clubelo_both_found_flag"].astype(bool)).sum()),
        "clubelo_source_files": "; ".join(sorted(merged["clubelo_source_file"].dropna().astype(str).unique())),
    }
    added = set(CLUBELO_FEATURE_COLUMNS)
    leakage = {
        "market": market,
        "future_rating_check_applicable": False,
        "future_rating_check_note": "Locked ClubElo feature block does not expose rating date columns; strict-before-date validation is documented in the ClubElo lock reports.",
        "no_target_leakage_columns_added": not any(c.startswith("target_") for c in added),
        "no_same_match_stats_added": not bool(added.intersection(SAME_MATCH_FORBIDDEN)),
        "no_team_names_added_as_model_features": not bool(added.intersection(TEAM_NAME_COLUMNS)),
        "xg_not_added": not any("xg" in c.lower() for c in added),
        "odds_timing_remains_unknown": bool(merged["odds_timing_flag"].eq("unknown").all()),
        "classification": "research_only",
        "leakage_check_pass": True,
    }
    leakage["leakage_check_pass"] = all(
        bool(leakage[k])
        for k in [
            "no_target_leakage_columns_added",
            "no_same_match_stats_added",
            "no_team_names_added_as_model_features",
            "xg_not_added",
            "odds_timing_remains_unknown",
        ]
    )
    return merged, row_count, coverage, leakage


def build_allowlist_and_forbidden() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_allow = pd.read_csv(BASE_ALLOWLIST_PATH)
    base_forbidden = pd.read_csv(BASE_FORBIDDEN_PATH)
    rows = []
    for market in MARKETS:
        for col in [
            "home_clubelo_rating",
            "away_clubelo_rating",
            "clubelo_diff_home_minus_away",
            "home_clubelo_days_stale",
            "away_clubelo_days_stale",
            "home_clubelo_found_flag",
            "away_clubelo_found_flag",
            "clubelo_both_found_flag",
        ]:
            rows.append(
                {
                    "market": market,
                    "column": col,
                    "feature_block": "clubelo_locked",
                    "allowlist_status": "frozen_allowed",
                    "review_note": "Locked ClubElo feature joined strictly by canonical_match_id; source ratings were date-safe before match date.",
                }
            )
    allow = pd.concat([base_allow, pd.DataFrame(rows)], ignore_index=True)
    forbidden_rows = []
    for market in MARKETS:
        forbidden_rows.append(
            {
                "market": market,
                "column": "clubelo_source_file",
                "role": "source_provenance",
                "feature_block": "clubelo_locked",
                "forbidden_status": "forbidden_as_model_feature",
                "leakage_note": "Source file provenance only; not a model feature.",
            }
        )
    forbidden = pd.concat([base_forbidden, pd.DataFrame(forbidden_rows)], ignore_index=True)
    return allow, forbidden


def write_reports(row_counts: pd.DataFrame, coverage: pd.DataFrame, leakage: pd.DataFrame, decision: str) -> None:
    report = [
        "# ClubElo Super CSV Merge Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Scope: merge locked ClubElo feature block into Footiqo top-5 research-ready market CSVs. No other external source was joined. No modeling, value search, or threshold optimization was run.",
        "",
        "## Merge Policy",
        "- Left join from each market CSV to ClubElo.",
        "- Join key: `canonical_match_id` only.",
        "- Existing columns are not overwritten.",
        "- Classification remains `research_only` because odds timing remains unknown.",
        "",
        "## Outputs",
    ]
    for _, r in row_counts.iterrows():
        report.append(f"- data/processed/super_csvs/research_ready_plus/clubelo/{r['output_file']}")
    report.extend(
        [
            "",
            "## Summary",
            f"- All row counts preserved: {bool(row_counts['row_count_preserved'].all())}",
            f"- Any duplicate canonical IDs: {bool((row_counts['duplicate_canonical_match_id_count'] > 0).any())}",
            f"- Minimum ClubElo both-found coverage: {coverage['clubelo_both_found_rate'].min():.4f}",
            f"- Leakage checks passing: {int(leakage['leakage_check_pass'].sum())}/{len(leakage)}",
            "",
            "No confirmed edge is claimed.",
        ]
    )
    (REPORT_DIR / "clubelo_merge_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision_md = [
        "# ClubElo Merge Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "The ClubElo-enhanced market CSVs are research-only datasets. They preserve row counts, canonical ID uniqueness, and the existing market availability filters.",
        "",
        "No modeling was performed and no confirmed edge is claimed.",
    ]
    (REPORT_DIR / "clubelo_merge_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    clubelo = load_clubelo()
    row_counts = []
    coverages = []
    leakages = []
    all_written = True
    for market, spec in MARKETS.items():
        _, rc, cov, leak = merge_market(market, spec, clubelo)
        row_counts.append(rc)
        coverages.append(cov)
        leakages.append(leak)
        all_written = all_written and (OUT_DIR / spec["output"]).exists()
    row_counts_df = pd.DataFrame(row_counts)
    coverage_df = pd.DataFrame(coverages)
    leakage_df = pd.DataFrame(leakages)
    allow, forbidden = build_allowlist_and_forbidden()

    ready_good = (
        all_written
        and row_counts_df["row_count_preserved"].all()
        and row_counts_df["canonical_match_id_unique"].all()
        and row_counts_df["duplicate_canonical_match_id_count"].eq(0).all()
        and ~row_counts_df["row_multiplication_detected"].any()
        and coverage_df["clubelo_both_found_rate"].ge(1.0).all()
        and leakage_df["leakage_check_pass"].all()
        and not allow.empty
        and leakage_df["classification"].eq("research_only").all()
    )
    if ready_good:
        decision = "clubelo_super_csv_merge_ready_good"
    elif all_written:
        decision = "clubelo_super_csv_merge_ready_needs_review"
    else:
        decision = "clubelo_super_csv_merge_failed"

    row_counts_df.to_csv(REPORT_DIR / "clubelo_merge_row_counts.csv", index=False)
    coverage_df.to_csv(REPORT_DIR / "clubelo_merge_coverage.csv", index=False)
    allow.to_csv(REPORT_DIR / "clubelo_merge_feature_allowlist.csv", index=False)
    forbidden.to_csv(REPORT_DIR / "clubelo_merge_forbidden_columns.csv", index=False)
    leakage_df.to_csv(REPORT_DIR / "clubelo_merge_leakage_checks.csv", index=False)
    write_reports(row_counts_df, coverage_df, leakage_df, decision)


if __name__ == "__main__":
    main()
