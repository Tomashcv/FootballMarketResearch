from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data/processed/match_registry/canonical_match_registry_v1_prototype.csv"
ENTITY_MATCHES = ROOT / "data/processed/entity_registry/matches_v1_locked.csv"
FOOTIQO = ROOT / "data/processed/footiqo/footiqo_top5_rolling_features_v1.csv"
CLUBELO = ROOT / "data/processed/feature_blocks/clubelo/clubelo_features_footiqo_top5_v1_locked.csv"
UNDERSTAT = ROOT / "data/processed/feature_blocks/understat/understat_features_footiqo_top5_v1_locked.csv"
TRANSFERMARKT = ROOT / "data/processed/feature_blocks/transfermarkt/transfermarkt_features_footiqo_top5_v1_locked.csv"
OUT = ROOT / "data/processed/super_csvs/research_ready_plus/clubelo_understat_transfermarkt/super_1x2_footiqo_top5_clubelo_understat_transfermarkt_research_v1.csv"
REPORT_DIR = ROOT / "outputs/reports/super_csvs/one_x_two"


RAW_ODDS = {"H", "D", "A"}
UNRELATED_MARKET_PREFIXES = ("btts_", "ou05_", "ou15_", "ou25_", "ou35_", "ou45_")
UNRELATED_RAW_ODDS = {"O05", "U05", "O15", "U15", "O25", "U25", "O35", "U35", "O45", "U45", "BTTSY", "BTTSN"}
FORBIDDEN_EXACT = {
    "canonical_match_id",
    "source",
    "source_match_id",
    "source_league_slug",
    "primary_source",
    "match_datetime",
    "season_label",
    "competition_slug",
    "league_name",
    "home_team_normalized",
    "away_team_normalized",
    "home_team_raw",
    "away_team_raw",
    "home_team_id",
    "away_team_id",
    "home_team_name_audit",
    "away_team_name_audit",
    "homeTeam",
    "awayTeam",
    "home_goals",
    "away_goals",
    "result_1x2",
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
TARGETS = {"target_home_win", "target_draw", "target_away_win"}


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def detect_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    candidates = []
    expected = {"home": "H", "draw": "D", "away": "A"}
    for role, col in expected.items():
        exists = col in df.columns
        valid_count = int((pd.to_numeric(df[col], errors="coerce") > 1).sum()) if exists else 0
        candidates.append(
            {
                "market": "1x2",
                "role": role,
                "detected_column": col if exists else "",
                "exists": exists,
                "valid_odds_gt_1_count": valid_count,
                "decision": "selected" if exists else "missing",
                "notes": "Detected canonical Footiqo 1X2 odds column." if exists else "Required column not found.",
            }
        )
    return pd.DataFrame(candidates)


def base_universe() -> pd.DataFrame:
    canonical = pd.read_csv(CANONICAL, dtype={"competition_code": str})
    entity = pd.read_csv(ENTITY_MATCHES, dtype={"competition_code": str})[
        [
            "canonical_match_id",
            "home_team_id",
            "away_team_id",
            "home_team_name_audit",
            "away_team_name_audit",
        ]
    ]
    foot = pd.read_csv(FOOTIQO)
    foot["match_datetime"] = pd.to_datetime(foot["match_datetime"], errors="coerce")
    foot["home_team_normalized"] = foot["homeTeam"].map(normalize_name)
    foot["away_team_normalized"] = foot["awayTeam"].map(normalize_name)
    canonical["match_datetime"] = pd.to_datetime(canonical["match_datetime"], errors="coerce")
    join_keys = ["match_datetime", "season_start_year", "home_team_normalized", "away_team_normalized"]
    merged = canonical.merge(foot, on=join_keys, how="left", suffixes=("", "_footiqo"), validate="one_to_one")
    merged = merged.merge(entity, on="canonical_match_id", how="left", validate="one_to_one")
    merged["source"] = "footiqo"
    merged["source_match_id"] = merged["id"]
    merged["source_league_slug"] = merged["source_league_slug"]
    return merged


def compute_market(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_home_win"] = out["target_home_win"].fillna(out["result_1x2"].eq("H").astype(int))
    out["target_draw"] = out["target_draw"].fillna(out["result_1x2"].eq("D").astype(int))
    out["target_away_win"] = out["target_away_win"].fillna(out["result_1x2"].eq("A").astype(int))
    out["x1_home_raw_prob"] = 1.0 / pd.to_numeric(out["H"], errors="coerce")
    out["x1_draw_raw_prob"] = 1.0 / pd.to_numeric(out["D"], errors="coerce")
    out["x1_away_raw_prob"] = 1.0 / pd.to_numeric(out["A"], errors="coerce")
    out["x1_overround"] = out[["x1_home_raw_prob", "x1_draw_raw_prob", "x1_away_raw_prob"]].sum(axis=1)
    out["x1_home_no_vig_prob"] = out["x1_home_raw_prob"] / out["x1_overround"]
    out["x1_draw_no_vig_prob"] = out["x1_draw_raw_prob"] / out["x1_overround"]
    out["x1_away_no_vig_prob"] = out["x1_away_raw_prob"] / out["x1_overround"]
    out["odds_timing_flag"] = "unknown"
    out["classification"] = "research_only"
    out["x1_paired_odds_available_flag"] = out[["H", "D", "A"]].notna().all(axis=1)
    out["x1_valid_paired_odds_flag"] = out[["H", "D", "A"]].apply(lambda s: pd.to_numeric(s, errors="coerce")).gt(1).all(axis=1)
    return out


def merge_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    feature_paths = [
        ("clubelo", CLUBELO),
        ("understat", UNDERSTAT),
        ("transfermarkt", TRANSFERMARKT),
    ]
    out = df.copy()
    stats = []
    before = len(out)
    for name, path in feature_paths:
        feat = pd.read_csv(path)
        feat["canonical_match_id"] = feat["canonical_match_id"].astype("int64")
        if feat["canonical_match_id"].duplicated().any():
            raise ValueError(f"{name} feature block has duplicate canonical_match_id")
        conflicts = sorted((set(out.columns) & set(feat.columns)) - {"canonical_match_id"})
        if conflicts:
            raise ValueError(f"{name} feature block column conflicts: {conflicts[:20]}")
        out = out.merge(feat, on="canonical_match_id", how="left", validate="one_to_one")
        stats.append(
            {
                "feature_block": name,
                "row_count_before": before,
                "row_count_after": len(out),
                "row_multiplication_detected": len(out) != before,
            }
        )
    return out, stats


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "canonical_match_id",
        "source",
        "source_match_id",
        "source_league_slug",
        "primary_source",
        "match_datetime",
        "season_start_year",
        "season_label",
        "competition_type",
        "competition_code",
        "competition_slug",
        "league_name",
        "home_team_id",
        "away_team_id",
        "home_team_name_audit",
        "away_team_name_audit",
        "home_team_normalized",
        "away_team_normalized",
    ]
    market = [
        "result_1x2",
        "target_home_win",
        "target_draw",
        "target_away_win",
        "H",
        "D",
        "A",
        "x1_home_raw_prob",
        "x1_draw_raw_prob",
        "x1_away_raw_prob",
        "x1_overround",
        "x1_home_no_vig_prob",
        "x1_draw_no_vig_prob",
        "x1_away_no_vig_prob",
        "x1_paired_odds_available_flag",
        "x1_valid_paired_odds_flag",
        "odds_timing_flag",
        "classification",
    ]
    rolling = [
        c
        for c in df.columns
        if (
            c.startswith(("home_", "away_", "home_minus_away_"))
            or c.startswith("league_")
            or c in {"match_week_index", "both_history_available_flag", "rolling_features_date_safe_flag", "external_sources_joined_flag"}
        )
        and c not in set(id_cols)
        and c not in {"home_goals", "away_goals"}
        and c not in FORBIDDEN_EXACT
        and not c.startswith(("home_understat_", "away_understat_", "home_tm_", "away_tm_"))
    ]
    external = [
        c
        for c in df.columns
        if c.startswith(("home_clubelo_", "away_clubelo_", "clubelo_", "home_understat_", "away_understat_", "understat_", "home_tm_", "away_tm_", "tm_"))
    ]
    ordered = list(dict.fromkeys(id_cols + market + rolling + external))
    return df[[c for c in ordered if c in df.columns]].copy()


def feature_allowlist_and_forbidden(cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed = []
    forbidden = []
    allow_exact = {"x1_home_no_vig_prob", "x1_draw_no_vig_prob", "x1_away_no_vig_prob"}
    for col in cols:
        reason = ""
        if col in allow_exact:
            reason = "1X2 target-specific no-vig market probability."
        elif col.startswith(("home_clubelo_", "away_clubelo_", "clubelo_")) and col != "clubelo_source_file":
            reason = "Locked date-safe ClubElo feature."
        elif col.startswith(("home_understat_", "away_understat_", "understat_")) and col not in FORBIDDEN_EXACT:
            reason = "Locked lagged Understat feature."
        elif col.startswith(("home_tm_", "away_tm_", "tm_")) and col not in FORBIDDEN_EXACT:
            reason = "Locked point-in-time Transfermarkt feature."
        elif (
            col.startswith(("home_", "away_", "home_minus_away_", "league_"))
            or col in {"match_week_index", "both_history_available_flag", "rolling_features_date_safe_flag", "external_sources_joined_flag"}
        ) and col not in FORBIDDEN_EXACT and col not in TARGETS:
            reason = "Date-safe Footiqo rolling/control feature."
        if reason and col not in RAW_ODDS and not col.startswith("target_") and not any(col.startswith(p) for p in UNRELATED_MARKET_PREFIXES):
            allowed.append({"market": "1x2", "column": col, "feature_block": "one_x_two_super_csv", "allowlist_status": "frozen_allowed", "review_note": reason})
        else:
            forbidden_reason = None
            if col in FORBIDDEN_EXACT or col in RAW_ODDS or col in TARGETS or col.startswith("target_"):
                forbidden_reason = "Identifier, raw odds, target, team/source audit, or outcome column."
            elif any(col.startswith(p) for p in UNRELATED_MARKET_PREFIXES) or col in UNRELATED_RAW_ODDS:
                forbidden_reason = "Unrelated market odds/probability column."
            elif "current_club" in col or "current_value" in col or "lineup" in col.lower() or "appearance" in col.lower():
                forbidden_reason = "Forbidden leakage-prone source field."
            if forbidden_reason:
                forbidden.append({"market": "1x2", "column": col, "role": "forbidden_or_audit", "feature_block": "one_x_two_super_csv", "forbidden_status": "forbidden_as_model_feature", "leakage_note": forbidden_reason})
    return pd.DataFrame(allowed), pd.DataFrame(forbidden)


def write_reports(full: pd.DataFrame, filtered: pd.DataFrame, detection: pd.DataFrame, merge_stats: list[dict], allow: pd.DataFrame, forbidden: pd.DataFrame, decision: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detection.to_csv(REPORT_DIR / "one_x_two_odds_column_detection.csv", index=False)
    coverage = pd.DataFrame(
        [
            {
                "rows_before_filter": len(full),
                "rows_after_valid_1x2_filter": len(filtered),
                "missing_any_1x2_odds_rows": int((~full["x1_paired_odds_available_flag"]).sum()),
                "invalid_odds_le_1_rows": int((full["x1_paired_odds_available_flag"] & ~full["x1_valid_paired_odds_flag"]).sum()),
                "missing_target_rows": int(full[list(TARGETS)].isna().any(axis=1).sum()),
                "clubelo_both_found_rate": float(filtered["clubelo_both_found_flag"].mean()),
                "understat_both_found_rate": float(filtered["understat_both_found_flag"].mean()),
                "understat_after_source_max_date_rows": int(filtered["understat_match_after_source_max_date_flag"].astype(bool).sum()),
                "transfermarkt_both_value_found_rate": float(filtered["tm_both_value_found_flag"].mean()),
            }
        ]
    )
    coverage.to_csv(REPORT_DIR / "one_x_two_market_coverage.csv", index=False)
    allow.to_csv(REPORT_DIR / "one_x_two_feature_allowlist.csv", index=False)
    forbidden.to_csv(REPORT_DIR / "one_x_two_forbidden_columns.csv", index=False)
    leakage = pd.DataFrame(
        [
            {"check_name": "valid_1x2_odds_columns_detected", "status": "pass" if detection["exists"].all() else "fail", "details": "Selected H/D/A."},
            {"check_name": "research_csv_written", "status": "pass" if OUT.exists() else "fail", "details": str(OUT.relative_to(ROOT))},
            {"check_name": "target_columns_valid", "status": "pass" if filtered[list(TARGETS)].notna().all().all() and filtered[list(TARGETS)].sum(axis=1).eq(1).all() else "fail", "details": "One of target_home_win/target_draw/target_away_win is 1 per row."},
            {"check_name": "canonical_match_id_unique", "status": "pass" if not filtered["canonical_match_id"].duplicated().any() else "fail", "details": f"duplicates={int(filtered['canonical_match_id'].duplicated().sum())}"},
            {"check_name": "paired_odds_complete", "status": "pass" if filtered[["H", "D", "A"]].notna().all().all() else "fail", "details": "H/D/A complete after filter."},
            {"check_name": "odds_gt_1", "status": "pass" if filtered[["H", "D", "A"]].apply(lambda s: pd.to_numeric(s, errors="coerce")).gt(1).all().all() else "fail", "details": "H/D/A > 1 after filter."},
            {"check_name": "no_row_multiplication_after_feature_merges", "status": "pass" if not any(s["row_multiplication_detected"] for s in merge_stats) else "fail", "details": str(merge_stats)},
            {"check_name": "no_current_club_or_current_value_columns", "status": "pass" if not any("current_club" in c or "current_value" in c for c in filtered.columns) else "fail", "details": "No current_* columns present."},
            {"check_name": "no_lineup_or_appearance_columns", "status": "pass" if not any("lineup" in c.lower() or "appearance" in c.lower() for c in filtered.columns) else "fail", "details": "No lineup/appearance columns present."},
            {"check_name": "odds_timing_unknown", "status": "pass" if filtered["odds_timing_flag"].eq("unknown").all() else "fail", "details": "Footiqo odds timing remains unknown."},
            {"check_name": "classification_research_only", "status": "pass" if filtered["classification"].eq("research_only").all() else "fail", "details": "No production/value classification."},
        ]
    )
    leakage.to_csv(REPORT_DIR / "one_x_two_leakage_checks.csv", index=False)
    report = [
        "# 1X2 Super CSV Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Detected 1X2 odds columns: `H`, `D`, `A`. Built a research-only 1X2 CSV with locked ClubElo, Understat, and Transfermarkt features joined by `canonical_match_id` only.",
        "",
        f"- Rows before filter: {len(full)}",
        f"- Rows after valid 1X2 odds filter: {len(filtered)}",
        f"- Output: `{OUT.relative_to(ROOT)}`",
        "",
        "No modeling, value search, threshold optimization, raw-file modification, or confirmed edge claim was performed.",
    ]
    (REPORT_DIR / "one_x_two_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (REPORT_DIR / "one_x_two_decision.md").write_text(
        "\n".join(["# 1X2 Super CSV Decision", "", f"Decision: **{decision}**", "", "The 1X2 CSV is research-only. Odds timing remains unknown.", "", "No confirmed edge is claimed."]) + "\n",
        encoding="utf-8",
    )
    return leakage


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    foot = pd.read_csv(FOOTIQO)
    detection = detect_odds_columns(foot)
    if not detection["exists"].all():
        decision = "one_x_two_super_csv_failed"
        write_reports(pd.DataFrame(), pd.DataFrame(), detection, [], pd.DataFrame(), pd.DataFrame(), decision)
        print(decision)
        return
    base = compute_market(base_universe())
    merged, merge_stats = merge_features(base)
    selected = select_columns(merged)
    full = selected.copy()
    valid = (
        full[list(TARGETS)].notna().all(axis=1)
        & full[list(TARGETS)].sum(axis=1).eq(1)
        & full["x1_paired_odds_available_flag"].astype(bool)
        & full["x1_valid_paired_odds_flag"].astype(bool)
    )
    filtered = full[valid].copy()
    filtered["canonical_match_id"] = filtered["canonical_match_id"].astype("int64")
    filtered.to_csv(OUT, index=False)
    allow, forbidden = feature_allowlist_and_forbidden(filtered.columns.tolist())
    leakage = write_reports(full, filtered, detection, merge_stats, allow, forbidden, "one_x_two_super_csv_ready_good")
    decision = "one_x_two_super_csv_ready_good"
    if leakage["status"].eq("fail").any() or filtered.empty:
        decision = "one_x_two_super_csv_failed"
        write_reports(full, filtered, detection, merge_stats, allow, forbidden, decision)
    print(decision)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
