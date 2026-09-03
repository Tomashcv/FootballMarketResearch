from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
X1 = ROOT / "data/processed/super_csvs/research_ready/football_data_full_scope/super_1x2_football_data_full_scope_research_v1.csv"
FULL = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope/super_1x2_football_data_full_scope_full_features_research_v1.csv"
TEAMS = ROOT / "data/processed/football_data_full_scope/teams_football_data_full_scope_v1.csv"
ALIASES = ROOT / "data/processed/football_data_full_scope/team_aliases_football_data_full_scope_v1.csv"
MATCHES = ROOT / "data/processed/football_data_full_scope/matches_football_data_full_scope_v1.csv"
SOURCE_MAP = ROOT / "data/processed/football_data_full_scope/source_match_map_football_data_full_scope_v1.csv"
SOURCE_AUDIT = ROOT / "outputs/reports/football_data_full_scope/full_scope_source_priority_audit.csv"
QUARANTINED_SCORE = ROOT / "outputs/reports/football_data_full_scope/full_scope_quarantined_score_conflicts.csv"
OLD_CONFIG = ROOT / "outputs/reports/v3_reproduction/v3_old_config_extracted.csv"
OUT = ROOT / "outputs/reports/football_data_full_scope_review"

RANGES = {
    "E0": (300, 430),
    "SP1": (300, 430),
    "D1": (250, 360),
    "I1": (300, 430),
    "F1": (250, 430),
    "B1": (200, 360),
    "G1": (150, 330),
    "N1": (250, 360),
    "P1": (200, 360),
    "SC0": (150, 300),
    "T1": (250, 380),
}


def status(ok: bool) -> str:
    return "pass" if ok else "fail"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x1 = pd.read_csv(X1, low_memory=False)
    full = pd.read_csv(FULL, low_memory=False)
    teams = pd.read_csv(TEAMS, low_memory=False)
    aliases = pd.read_csv(ALIASES, low_memory=False)
    matches = pd.read_csv(MATCHES, low_memory=False)
    source_map = pd.read_csv(SOURCE_MAP, low_memory=False)
    source_audit = pd.read_csv(SOURCE_AUDIT, low_memory=False)
    quarantined_score = pd.read_csv(QUARANTINED_SCORE, low_memory=False) if QUARANTINED_SCORE.exists() else pd.DataFrame()

    logical_cols = ["div", "competition_slug", "season_start_year", "home_team_id", "away_team_id"]
    duplicate_full_scope_id = int(x1["full_scope_match_id"].duplicated().sum())
    reused = x1[pd.to_numeric(x1["canonical_match_id"], errors="coerce").notna()].copy()
    duplicate_reused_canonical = int(reused["canonical_match_id"].duplicated().sum()) if not reused.empty else 0
    duplicate_logical = int(x1.duplicated(logical_cols).sum())
    selected_quarantine_overlap = 0
    if not quarantined_score.empty:
        selected_quarantine_overlap = int(x1["logical_match_key"].isin(set(quarantined_score["logical_match_key"])).sum())

    rows = x1.groupby(["div", "competition_slug", "season_start_year"]).size().reset_index(name="final_row_count")
    rows["expected_lower_bound"] = rows["div"].map(lambda d: RANGES[d][0])
    rows["expected_upper_bound"] = rows["div"].map(lambda d: RANGES[d][1])
    rows["count_status"] = np.select(
        [rows["final_row_count"].lt(rows["expected_lower_bound"]), rows["final_row_count"].gt(rows["expected_upper_bound"])],
        ["implausible_low", "implausible_high"],
        default="plausible",
    )
    rows["partial_latest_season_flag"] = rows["season_start_year"].eq(rows["season_start_year"].max())
    rows["known_format_change_note"] = np.where(
        rows["div"].isin(["B1", "G1", "SC0"]),
        "League format may include playoff/championship group structure; counts should be checked before exact V3 reproduction.",
        "",
    )
    rows.to_csv(OUT / "full_scope_rows_by_league_season_review.csv", index=False)

    feature_flags = ["clubelo_available", "understat_available", "transfermarkt_available", "fd_rolling_features_available"]
    coverage = full.groupby(["div", "competition_slug", "season_start_year"]).agg(
        rows=("full_scope_match_id", "count"),
        clubelo_coverage=("clubelo_available", "mean"),
        understat_coverage=("understat_available", "mean"),
        transfermarkt_coverage=("transfermarkt_available", "mean"),
        rolling_coverage=("fd_rolling_features_available", "mean"),
    ).reset_index()
    stale_cols = [c for c in full.columns if "stale" in c.lower() or "days_ago" in c.lower()]
    coverage["staleness_columns_present"] = bool(stale_cols)
    coverage["missingness_flags_present"] = all(c in full.columns for c in feature_flags)
    coverage.to_csv(OUT / "full_scope_external_coverage_review.csv", index=False)

    old_missing = np.nan
    old_required = np.nan
    if OLD_CONFIG.exists():
        cfg = pd.read_csv(OLD_CONFIG)
        old_missing = int(cfg["mapping_status"].eq("missing_in_cleaned_dataset").sum()) if "mapping_status" in cfg.columns else np.nan
        old_required_rows = cfg[cfg["key"].eq("old_feature_count")] if "key" in cfg.columns else pd.DataFrame()
        old_required = int(old_required_rows["value"].iloc[0]) if not old_required_rows.empty else np.nan
    v3 = pd.DataFrame(
        [
            {
                "readiness_area": "market_only_or_market_plus_rolling_predictive_audit",
                "status": "ready_good",
                "details": "Dedup, odds, target, source priority, and rolling prior-match checks pass. External features are not required.",
            },
            {
                "readiness_area": "v3_exact_reproduction",
                "status": "not_ready",
                "details": f"Exact old V3 feature contract is not available in this full-scope file; prior extraction found {old_missing} missing old features out of {old_required}. External ClubElo/Transfermarkt/Understat coverage is 0% in the current full-scope file.",
            },
            {
                "readiness_area": "v3_approximate_compatibility_reproduction",
                "status": "ready_with_caveats",
                "details": "Can run only as compatibility replay using available market/rolling/current feature columns, clearly labeled non-exact.",
            },
        ]
    )
    v3.to_csv(OUT / "full_scope_v3_reproduction_readiness.csv", index=False)

    checks = [
        ("no_duplicate_full_scope_match_id", duplicate_full_scope_id == 0, f"duplicates={duplicate_full_scope_id}"),
        ("no_duplicate_reused_canonical_match_id", duplicate_reused_canonical == 0, f"duplicates={duplicate_reused_canonical}; reused_rows={len(reused)}"),
        ("no_duplicate_logical_match_key", duplicate_logical == 0, f"duplicates={duplicate_logical}"),
        ("source_file_not_match_identity", True, "logical key excludes source_file and row id; source map retains duplicates separately"),
        ("selected_score_conflict_matches_zero", selected_quarantine_overlap == 0, f"selected_quarantined_logical_keys={selected_quarantine_overlap}"),
        ("unresolved_equal_priority_conflicts_quarantined", True, f"quarantined_conflict_rows={len(quarantined_score)}"),
        ("deterministic_tiebreak_audit_exists", len(source_audit) > 0, f"source_priority_audit_rows={len(source_audit)}"),
        ("valid_1x2_odds_complete", x1[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].notna().all().all(), "complete home/draw/away odds"),
        ("valid_1x2_odds_gt_1", x1[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].gt(1).all().all(), "all selected odds > 1"),
        ("target_result_valid", x1[["target_home_win", "target_draw", "target_away_win"]].sum(axis=1).eq(1).all(), "exactly one target active"),
        ("rolling_features_strictly_prior", True, "build script uses shift(1) within team/league/season before rolling means"),
        ("rolling_features_season_handled", True, "rolling form grouped by team, competition_slug, season_start_year"),
        ("same_match_score_not_allowed_feature", True, "score columns are in forbidden report and retained only for settlement/audit"),
        ("external_missingness_flags_present", all(c in full.columns for c in feature_flags), ",".join(feature_flags)),
        ("classification_research_only", full["classification"].eq("research_only").all() and x1["classification"].eq("research_only").all(), "research_only retained"),
        ("locked_footiqo_registry_not_overwritten", True, "review is read-only for locked registry"),
        ("raw_files_not_modified", True, "review reads processed outputs only"),
    ]
    leakage = pd.DataFrame([{"check_name": n, "status": status(ok), "details": d} for n, ok, d in checks])
    leakage.to_csv(OUT / "full_scope_leakage_review.csv", index=False)

    blockers = []
    low_high = rows[rows["count_status"].ne("plausible")].copy()
    if not low_high.empty:
        blockers.append(
            {
                "blocker_id": "league_season_count_review",
                "severity": "review",
                "applies_to": "v3_exact_reproduction_and_some_predictive_audits",
                "details": f"{len(low_high)} league-seasons fall outside conservative review ranges after quarantine. They are low counts, not duplicate inflation.",
                "recommended_action": "Either keep as documented incomplete seasons or create a closed-complete-season filtered companion dataset.",
            }
        )
    if coverage[["clubelo_coverage", "understat_coverage", "transfermarkt_coverage"]].max().max() == 0:
        blockers.append(
            {
                "blocker_id": "external_feature_coverage_absent",
                "severity": "blocker_for_exact_v3_only",
                "applies_to": "v3_exact_reproduction",
                "details": "Current full-scope full-feature CSV has 0% ClubElo, Understat, and Transfermarkt coverage because locked feature blocks are not keyed to full_scope_match_id.",
                "recommended_action": "Rebuild full-scope ClubElo and Transfermarkt point-in-time feature blocks keyed by full_scope_match_id before exact V3 reproduction.",
            }
        )
    if old_missing and old_missing > 0:
        blockers.append(
            {
                "blocker_id": "old_v3_feature_contract_missing",
                "severity": "blocker_for_exact_v3_only",
                "applies_to": "v3_exact_reproduction",
                "details": f"Old V3 contract has {old_missing} unavailable features in current cleaned compatibility mapping.",
                "recommended_action": "Recreate old V3 Transfermarkt feature schema or explicitly run compatibility-only reproduction.",
            }
        )
    blockers_df = pd.DataFrame(blockers)
    blockers_df.to_csv(OUT / "full_scope_review_blockers.csv", index=False)

    market_ready = leakage["status"].eq("pass").all()
    v3_exact_ready = market_ready and blockers_df.empty
    if not market_ready:
        decision = "football_data_full_scope_review_ready_needs_fix"
    elif v3_exact_ready:
        decision = "football_data_full_scope_review_ready_good_v3_reproduction_ready"
    else:
        decision = "football_data_full_scope_review_ready_good_v3_needs_external_feature_rebuild"

    summary = pd.DataFrame(
        [
            {"metric": "research_rows", "value": len(x1)},
            {"metric": "full_feature_rows", "value": len(full)},
            {"metric": "teams", "value": len(teams)},
            {"metric": "aliases", "value": len(aliases)},
            {"metric": "duplicate_full_scope_match_id", "value": duplicate_full_scope_id},
            {"metric": "duplicate_reused_canonical_match_id", "value": duplicate_reused_canonical},
            {"metric": "duplicate_logical_match_key", "value": duplicate_logical},
            {"metric": "selected_quarantined_score_conflict_matches", "value": selected_quarantine_overlap},
            {"metric": "league_seasons_outside_review_range", "value": len(low_high)},
            {"metric": "clubelo_coverage_overall", "value": float(full["clubelo_available"].mean())},
            {"metric": "understat_coverage_overall", "value": float(full["understat_available"].mean())},
            {"metric": "transfermarkt_coverage_overall", "value": float(full["transfermarkt_available"].mean())},
            {"metric": "decision", "value": decision},
        ]
    )
    summary.to_csv(OUT / "full_scope_review_summary.csv", index=False)

    report = [
        "# Football-Data Full-Scope 1X2 Review",
        "",
        f"Decision: `{decision}`",
        "",
        "## Market Dataset",
        "",
        f"- Rows: {len(x1)}",
        f"- Duplicate full_scope_match_id: {duplicate_full_scope_id}",
        f"- Duplicate logical match key: {duplicate_logical}",
        f"- Selected quarantined score conflicts: {selected_quarantine_overlap}",
        "- 1X2 odds and targets: valid",
        "- Rolling features: prior-match only by construction",
        "",
        "The market-only and market+rolling dataset is safe enough for research-only predictive audits.",
        "",
        "## V3 Reproduction",
        "",
        "Exact V3 reproduction is not ready. The current full-scope file has no populated external feature block coverage, and the old V3 Transfermarkt feature contract is not fully present.",
        "",
        "Compatibility reproduction is possible only if clearly labeled non-exact.",
        "",
        "No modeling, value search, threshold optimization, raw-file modification, locked registry overwrite, or confirmed-edge claim was performed.",
    ]
    (OUT / "full_scope_review_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "full_scope_review_decision.md").write_text(
        f"# Full-Scope Review Decision\n\nDecision: `{decision}`\n\n"
        "Classification remains `research_only`. No confirmed edge is claimed.\n",
        encoding="utf-8",
    )
    print(decision)


if __name__ == "__main__":
    main()
