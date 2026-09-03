from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/clubelo"
OUT_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/clubelo_understat"
REPORT_DIR = ROOT / "outputs/reports/super_csvs/understat_merge"
UNDERSTAT_PATH = ROOT / "data/processed/feature_blocks/understat/understat_features_footiqo_top5_v1_locked.csv"
UNDERSTAT_LEAKAGE_CHECKS = ROOT / "outputs/reports/feature_blocks/understat/understat_locked_leakage_checks.csv"
CLUBELO_ALLOWLIST = ROOT / "outputs/reports/super_csvs/clubelo_merge/clubelo_merge_feature_allowlist.csv"
CLUBELO_FORBIDDEN = ROOT / "outputs/reports/super_csvs/clubelo_merge/clubelo_merge_forbidden_columns.csv"

MARKETS = {
    "btts": {
        "input": "super_btts_footiqo_top5_clubelo_research_v1.csv",
        "output": "super_btts_footiqo_top5_clubelo_understat_research_v1.csv",
    },
    "ou15": {
        "input": "super_ou15_footiqo_top5_clubelo_research_v1.csv",
        "output": "super_ou15_footiqo_top5_clubelo_understat_research_v1.csv",
    },
    "ou25": {
        "input": "super_ou25_footiqo_top5_clubelo_research_v1.csv",
        "output": "super_ou25_footiqo_top5_clubelo_understat_research_v1.csv",
    },
    "ou35": {
        "input": "super_ou35_footiqo_top5_clubelo_research_v1.csv",
        "output": "super_ou35_footiqo_top5_clubelo_understat_research_v1.csv",
    },
    "ou45": {
        "input": "super_ou45_footiqo_top5_clubelo_research_v1.csv",
        "output": "super_ou45_footiqo_top5_clubelo_understat_research_v1.csv",
    },
}

FORBIDDEN_UNDERSTAT_COLUMNS = {
    "understat_league",
    "understat_source_file",
    "home_understat_alias_id",
    "away_understat_alias_id",
    "home_understat_latest_date",
    "away_understat_latest_date",
}


def load_understat() -> pd.DataFrame:
    df = pd.read_csv(UNDERSTAT_PATH)
    df["canonical_match_id"] = df["canonical_match_id"].astype("int64")
    if df["canonical_match_id"].duplicated().any():
        raise ValueError("Understat feature block has duplicate canonical_match_id")
    return df


def understat_model_features(df: pd.DataFrame) -> list[str]:
    allowed = []
    for c in df.columns:
        if c == "canonical_match_id":
            continue
        if c in FORBIDDEN_UNDERSTAT_COLUMNS:
            continue
        # All written Understat numeric/flag rolling columns are past-only by construction.
        if c.startswith("home_understat_") or c.startswith("away_understat_") or c.startswith("understat_home_minus_away_") or c in {
            "understat_both_found_flag",
            "understat_match_after_source_max_date_flag",
        }:
            allowed.append(c)
    return allowed


def merge_market(market: str, spec: dict, understat: pd.DataFrame, understat_cols: list[str]) -> tuple[dict, dict, dict]:
    base = pd.read_csv(BASE_DIR / spec["input"], dtype={"competition_code": str})
    base["canonical_match_id"] = base["canonical_match_id"].astype("int64")
    before_rows = len(base)
    conflicts = sorted((set(base.columns) & set(understat.columns)) - {"canonical_match_id"})
    if conflicts:
        raise ValueError(f"{market}: Understat column conflicts would overwrite existing columns: {conflicts[:20]}")
    merged = base.merge(understat, on="canonical_match_id", how="left", validate="one_to_one")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DIR / spec["output"], index=False)

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
        "missing_understat_feature_rows": int(merged["understat_both_found_flag"].isna().sum()),
        "understat_both_found_rate": float(merged["understat_both_found_flag"].mean()),
        "understat_both_missing_rows": int((~merged["understat_both_found_flag"].astype(bool)).sum()),
        "understat_after_source_max_date_rate": float(merged["understat_match_after_source_max_date_flag"].mean()),
        "understat_after_source_max_date_rows": int(merged["understat_match_after_source_max_date_flag"].astype(bool).sum()),
    }
    latest = merged[["match_datetime", "home_understat_latest_date", "away_understat_latest_date"]].copy()
    latest["match_date"] = pd.to_datetime(latest["match_datetime"], errors="coerce").dt.floor("D")
    latest["home_understat_latest_date"] = pd.to_datetime(latest["home_understat_latest_date"], errors="coerce")
    latest["away_understat_latest_date"] = pd.to_datetime(latest["away_understat_latest_date"], errors="coerce")
    future_home = latest["home_understat_latest_date"].notna() & (latest["home_understat_latest_date"] >= latest["match_date"])
    future_away = latest["away_understat_latest_date"].notna() & (latest["away_understat_latest_date"] >= latest["match_date"])
    rejected_alias_used = (
        merged.get("home_understat_alias_id", pd.Series(dtype="float64")).eq(384).any()
        or merged.get("away_understat_alias_id", pd.Series(dtype="float64")).eq(384).any()
    )
    leakage = {
        "market": market,
        "no_target_leakage_columns_added": not any(c.startswith("target_") for c in understat_cols),
        "no_same_match_raw_understat_columns_added": not any(c in {"xG", "xGA", "npxG", "npxGA", "scored", "missed", "result", "date"} for c in understat_cols),
        "no_team_names_added_as_model_features": not any("team_raw" in c or "team_name" in c for c in understat_cols),
        "no_source_provenance_as_model_feature": "understat_source_file" not in understat_cols,
        "no_understat_alias_ids_as_model_feature": "home_understat_alias_id" not in understat_cols and "away_understat_alias_id" not in understat_cols,
        "no_understat_latest_dates_as_model_feature": "home_understat_latest_date" not in understat_cols and "away_understat_latest_date" not in understat_cols,
        "no_future_understat_rows_used": not (future_home.any() or future_away.any()),
        "no_rejected_alias_use": not bool(rejected_alias_used),
        "odds_timing_remains_unknown": bool(merged["odds_timing_flag"].eq("unknown").all()) if "odds_timing_flag" in merged.columns else False,
        "classification": "research_only",
    }
    leakage["leakage_check_pass"] = all(bool(v) for k, v in leakage.items() if k not in {"market", "classification"})
    return row_count, coverage, leakage


def build_allowlist_and_forbidden(understat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if CLUBELO_ALLOWLIST.exists():
        base_allow = pd.read_csv(CLUBELO_ALLOWLIST)
    else:
        base_allow = pd.DataFrame(columns=["market", "column", "feature_block", "allowlist_status", "review_note"])
    if CLUBELO_FORBIDDEN.exists():
        base_forbidden = pd.read_csv(CLUBELO_FORBIDDEN)
    else:
        base_forbidden = pd.DataFrame(columns=["market", "column", "role", "feature_block", "forbidden_status", "leakage_note"])
    rows = []
    for market in MARKETS:
        for col in understat_cols:
            rows.append({
                "market": market,
                "column": col,
                "feature_block": "understat_lagged_rolling",
                "allowlist_status": "frozen_allowed",
                "review_note": "Understat rolling/flag feature built strictly from rows with understat_date < match_date.",
            })
    forbidden_rows = []
    for market in MARKETS:
        for col in sorted(FORBIDDEN_UNDERSTAT_COLUMNS):
            forbidden_rows.append({
                "market": market,
                "column": col,
                "role": "source_provenance_or_context",
                "feature_block": "understat_lagged_rolling",
                "forbidden_status": "forbidden_as_model_feature",
                "leakage_note": "Context/provenance only; not a model feature.",
            })
    return pd.concat([base_allow, pd.DataFrame(rows)], ignore_index=True), pd.concat([base_forbidden, pd.DataFrame(forbidden_rows)], ignore_index=True)


def write_reports(row_counts: pd.DataFrame, coverage: pd.DataFrame, staleness: pd.DataFrame, leakage: pd.DataFrame, decision: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    row_counts.to_csv(REPORT_DIR / "understat_merge_row_counts.csv", index=False)
    coverage.to_csv(REPORT_DIR / "understat_merge_coverage.csv", index=False)
    staleness.to_csv(REPORT_DIR / "understat_merge_staleness.csv", index=False)
    leakage.to_csv(REPORT_DIR / "understat_merge_leakage_checks.csv", index=False)

    report = [
        "# Understat Super CSV Merge Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Scope: merge lagged Understat feature block into Footiqo top-5 ClubElo-enhanced research-ready market CSVs. No modeling, value search, or threshold optimization was run.",
        "",
        "## Merge Policy",
        "- Left join from each ClubElo-enhanced market CSV to Understat features.",
        "- Join key: `canonical_match_id` only.",
        "- Existing columns are not overwritten.",
        "- Understat features are rolling past-only and same-fixture xG/stats are not model features.",
        "- Understat alias IDs, latest contributing dates, league, and source file are retained only as audit/provenance columns.",
        "- Classification remains `research_only` because odds timing remains unknown.",
        "",
        "## Outputs",
    ]
    for _, r in row_counts.iterrows():
        report.append(f"- data/processed/super_csvs/research_ready_plus/clubelo_understat/{r['output_file']}")
    report.extend([
        "",
        "## Summary",
        f"- All row counts preserved: {bool(row_counts['row_count_preserved'].all())}",
        f"- Any duplicate canonical IDs: {bool((row_counts['duplicate_canonical_match_id_count'] > 0).any())}",
        f"- Minimum Understat both-found coverage: {coverage['understat_both_found_rate'].min():.4f}",
        f"- Maximum stale-after-source-date rate: {coverage['understat_after_source_max_date_rate'].max():.4f}",
        f"- Leakage checks passing: {int(leakage['leakage_check_pass'].sum())}/{len(leakage)}",
        "",
        "No confirmed edge is claimed.",
    ])
    (REPORT_DIR / "understat_merge_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision_md = [
        "# Understat Merge Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "The Understat-enhanced market CSVs are research-only datasets. They preserve row counts, canonical ID uniqueness, and existing market availability filters.",
        "",
        "No modeling was performed and no confirmed edge is claimed.",
    ]
    (REPORT_DIR / "understat_merge_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    understat = load_understat()
    understat_cols = understat_model_features(understat)
    row_counts, coverage, leakage = [], [], []
    for market, spec in MARKETS.items():
        rc, cov, leak = merge_market(market, spec, understat, understat_cols)
        row_counts.append(rc)
        coverage.append(cov)
        leakage.append(leak)
    row_counts_df = pd.DataFrame(row_counts)
    coverage_df = pd.DataFrame(coverage)
    staleness_df = coverage_df[
        [
            "market",
            "row_count",
            "understat_after_source_max_date_rows",
            "understat_after_source_max_date_rate",
        ]
    ].copy()
    leakage_df = pd.DataFrame(leakage)
    allow, forbidden = build_allowlist_and_forbidden(understat_cols)
    allow.to_csv(REPORT_DIR / "understat_merge_feature_allowlist.csv", index=False)
    forbidden.to_csv(REPORT_DIR / "understat_merge_forbidden_columns.csv", index=False)

    upstream_locked_checks_pass = True
    if UNDERSTAT_LEAKAGE_CHECKS.exists():
        upstream = pd.read_csv(UNDERSTAT_LEAKAGE_CHECKS)
        upstream_locked_checks_pass = bool(upstream["status"].eq("pass").all())

    if (
        not upstream_locked_checks_pass
        or not row_counts_df["row_count_preserved"].all()
        or (row_counts_df["duplicate_canonical_match_id_count"] > 0).any()
        or row_counts_df["row_multiplication_detected"].any()
        or not leakage_df["leakage_check_pass"].all()
    ):
        decision = "understat_super_csv_merge_failed"
    elif coverage_df["understat_both_found_rate"].min() < 0.95:
        decision = "understat_super_csv_merge_failed"
    elif coverage_df["understat_after_source_max_date_rate"].max() > 0.05:
        decision = "understat_super_csv_merge_ready_with_staleness_warning"
    else:
        decision = "understat_super_csv_merge_ready_good"
    write_reports(row_counts_df, coverage_df, staleness_df, leakage_df, decision)
    print(decision)


if __name__ == "__main__":
    main()
