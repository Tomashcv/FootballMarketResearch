"""Leakage-safe V4 feature matrix assembly from locked blocks."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.v4.data.market_panel import PANEL_PATH, PROCESSED_DIR, ROOT, normalize_team
from src.v4.data.phase1b_audit import OUT_DIR
from src.v4.models.dynamic_scoreline import FEATURE_PATH


MATRIX_PATH = PROCESSED_DIR / "v4_model_matrix_v1.csv"
GROUPS_PATH = PROCESSED_DIR / "v4_model_feature_groups_v1.json"


def _rename_features(frame: pd.DataFrame, prefix: str, id_col: str) -> pd.DataFrame:
    return frame.rename(columns={c: prefix + c for c in frame.columns if c != id_col})


def build_matrix() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    score = pd.read_csv(FEATURE_PATH, low_memory=False)
    matrix = panel.merge(score, on="id__canonical_match_id", how="left", validate="one_to_one")

    elo_path = ROOT / "data/processed/feature_blocks/internal_elo_full_scope/internal_elo_features_football_data_full_scope_v1.csv"
    elo_cols = ["full_scope_match_id","home_internal_elo","away_internal_elo","internal_elo_diff","internal_elo_home_advantage"]
    elo = pd.read_csv(elo_path, usecols=elo_cols, low_memory=False).drop_duplicates("full_scope_match_id")
    elo = _rename_features(elo, "feature_external__", "full_scope_match_id")
    matrix = matrix.merge(elo, left_on="id__full_scope_match_id", right_on="full_scope_match_id", how="left", validate="many_to_one").drop(columns="full_scope_match_id")

    club_path = ROOT / "data/processed/feature_blocks/clubelo_full_scope/clubelo_features_football_data_full_scope_v1.csv"
    club_cols = ["full_scope_match_id","home_clubelo","away_clubelo","clubelo_diff","home_clubelo_days_stale","away_clubelo_days_stale","clubelo_both_found_flag","home_clubelo_latest_date","away_clubelo_latest_date"]
    club = pd.read_csv(club_path, usecols=club_cols, low_memory=False).drop_duplicates("full_scope_match_id")
    club = _rename_features(club, "feature_external__", "full_scope_match_id")
    matrix = matrix.merge(club, left_on="id__full_scope_match_id", right_on="full_scope_match_id", how="left", validate="many_to_one").drop(columns="full_scope_match_id")

    tm_path = ROOT / "data/processed/feature_blocks/transfermarkt_full_scope/transfermarkt_features_football_data_full_scope_v1.csv"
    tm_cols = [
        "full_scope_match_id","tm_fixture_mapped","tm_match_feature_available","home_tm_total_value","away_tm_total_value","tm_total_value_diff","tm_total_value_ratio",
        "home_tm_top11_value","away_tm_top11_value","tm_top11_value_diff","home_tm_player_count","away_tm_player_count","tm_player_count_diff",
        "home_tm_value_days_stale","away_tm_value_days_stale","home_tm_latest_valuation_date","away_tm_latest_valuation_date",
        "home_tm_arrivals_count_90d","away_tm_arrivals_count_90d","home_tm_departures_count_90d","away_tm_departures_count_90d",
    ]
    tm_header = pd.read_csv(tm_path, nrows=0).columns
    tm_cols = [c for c in tm_cols if c in tm_header]
    tm = pd.read_csv(tm_path, usecols=tm_cols, low_memory=False).drop_duplicates("full_scope_match_id")
    tm = _rename_features(tm, "feature_external__", "full_scope_match_id")
    matrix = matrix.merge(tm, left_on="id__full_scope_match_id", right_on="full_scope_match_id", how="left", validate="many_to_one").drop(columns="full_scope_match_id")

    # Reuse the locked/lagged Understat columns from the existing matrix, joined
    # only by deterministic fixture identity. No same-match Understat fields.
    under_path = ROOT / "data/processed/features/football_feature_matrix_v4_1_understat_partial_v2.csv"
    under_candidates = [
        "match_date","league","home_team","away_team",
        "home_understat_xg_for_roll5","away_understat_xg_for_roll5","home_minus_away_understat_xg_for_roll5",
        "home_understat_xg_against_roll5","away_understat_xg_against_roll5","home_minus_away_understat_xg_against_roll5",
        "home_understat_xpts_roll5","away_understat_xpts_roll5","home_minus_away_understat_xpts_roll5",
        "understat_home_history_count","understat_away_history_count","understat_home_latest_days_ago","understat_away_latest_days_ago","understat_both_available_flag",
    ]
    header = pd.read_csv(under_path, nrows=0).columns
    use = [c for c in under_candidates if c in header]
    under = pd.read_csv(under_path, usecols=use, low_memory=False)
    under["_key"] = under["league"].astype(str)+"|"+under["match_date"].astype(str)+"|"+under["home_team"].map(normalize_team)+"|"+under["away_team"].map(normalize_team)
    under = under.drop_duplicates("_key")
    keep_under = [c for c in use if "understat" in c]
    under = under[["_key"]+keep_under].rename(columns={c:"feature_external__"+c for c in keep_under})
    matrix["_key"] = matrix["id__league"].astype(str)+"|"+matrix["id__match_date"].astype(str)+"|"+matrix["id__home_team"].map(normalize_team)+"|"+matrix["id__away_team"].map(normalize_team)
    matrix = matrix.merge(under, on="_key", how="left", validate="many_to_one").drop(columns="_key")

    league_codes = {league:i for i,league in enumerate(sorted(matrix["id__league"].unique()))}
    matrix["feature_external__league_code"] = matrix["id__league"].map(league_codes)
    matrix["feature_external__season_year"] = matrix["id__season_start_year"]
    matrix["feature_external__internal_elo_available"] = matrix["feature_external__internal_elo_diff"].notna()
    matrix["feature_external__clubelo_available"] = matrix["feature_external__clubelo_diff"].notna()
    matrix["feature_external__transfermarkt_available"] = matrix.get("feature_external__tm_match_feature_available", pd.Series(False,index=matrix.index)).fillna(False)
    matrix["feature_external__understat_available"] = matrix.get("feature_external__understat_both_available_flag", pd.Series(False,index=matrix.index)).fillna(False)

    snapshot_core = [
        "feature_snapshot__consensus_prob_home","feature_snapshot__consensus_prob_draw","feature_snapshot__consensus_prob_away",
        "feature_snapshot__bookmaker_count","feature_snapshot__overround_mean","feature_snapshot__overround_min","feature_snapshot__overround_max",
        "feature_snapshot__prob_dispersion_home","feature_snapshot__prob_dispersion_draw","feature_snapshot__prob_dispersion_away",
        "feature_snapshot__best_to_consensus_home","feature_snapshot__best_to_consensus_draw","feature_snapshot__best_to_consensus_away",
    ]
    score_cols=[c for c in matrix if c.startswith("feature_history__")]
    elo_cols2=[c for c in matrix if c.startswith("feature_external__") and ("elo" in c)]
    tm_cols2=[c for c in matrix if c.startswith("feature_external__") and ("tm_" in c or "transfermarkt" in c)]
    under_cols=[c for c in matrix if c.startswith("feature_external__") and "understat" in c]
    context=["feature_external__league_code","feature_external__season_year"]
    cross=[c for c in matrix if c.startswith("feature_snapshot__ah_") or c.startswith("feature_snapshot__ou25_")]
    disagreement=[c for c in snapshot_core if "dispersion" in c or "best_to_consensus" in c or "bookmaker_count" in c or "overround_" in c]
    external=list(dict.fromkeys(elo_cols2+tm_cols2+under_cols+context))
    full=list(dict.fromkeys(snapshot_core+score_cols+external+cross))
    groups={
        "snapshot_market_only":snapshot_core,
        "snapshot_plus_scoreline":list(dict.fromkeys(snapshot_core+score_cols)),
        "snapshot_plus_elo":list(dict.fromkeys(snapshot_core+elo_cols2)),
        "snapshot_plus_external":list(dict.fromkeys(snapshot_core+external)),
        "snapshot_plus_cross_market":list(dict.fromkeys(snapshot_core+cross)),
        "snapshot_plus_bookmaker_disagreement":list(dict.fromkeys(snapshot_core+disagreement)),
        "full_v4":full,
        "full_v4_without_transfermarkt":[c for c in full if c not in tm_cols2],
        "full_v4_without_understat":[c for c in full if c not in under_cols],
        "full_v4_without_clubelo":[c for c in full if "clubelo" not in c],
        "full_v4_without_scoreline":[c for c in full if c not in score_cols],
        "full_v4_without_cross_market":[c for c in full if c not in cross],
    }
    groups={k:[c for c in v if c in matrix.columns] for k,v in groups.items()}

    match_date=pd.to_datetime(matrix["id__match_date"])
    checks=[]
    def add(name, ok, details=""): checks.append({"check":name,"status":"pass" if ok else "fail","details":details})
    add("unique_fixture_rows",not matrix["id__canonical_match_id"].duplicated().any())
    add("no_closing_in_feature_groups",not any(c.startswith("label_close__") for g in groups.values() for c in g))
    add("no_results_in_feature_groups",not any(c.startswith("result__") for g in groups.values() for c in g))
    add("feature_namespaces_only",all(c.startswith(("feature_snapshot__","feature_history__","feature_external__")) for g in groups.values() for c in g))
    for side in ("home","away"):
        col=f"feature_external__{side}_clubelo_latest_date"
        if col in matrix:
            dates=pd.to_datetime(matrix[col],errors="coerce"); add(f"{side}_clubelo_strictly_before_match",bool((dates.dropna()<match_date.loc[dates.notna()]).all()))
        col=f"feature_external__{side}_tm_latest_valuation_date"
        if col in matrix:
            dates=pd.to_datetime(matrix[col],errors="coerce"); add(f"{side}_tm_not_after_match",bool((dates.dropna()<=match_date.loc[dates.notna()]).all()))
    add("exact_join_row_accounting",len(matrix)==len(panel),f"panel={len(panel)} matrix={len(matrix)}")
    add("missing_source_indicators_present",all(c in matrix for c in ["feature_external__internal_elo_available","feature_external__clubelo_available","feature_external__transfermarkt_available","feature_external__understat_available"]))
    return matrix,groups,pd.DataFrame(checks)


def run_phase4() -> dict[str,object]:
    matrix,groups,checks=build_matrix()
    matrix.to_csv(MATRIX_PATH,index=False);GROUPS_PATH.write_text(json.dumps(groups,indent=2,sort_keys=True),encoding="utf-8")
    checks.to_csv(OUT_DIR/"v4_phase4_leakage_checks.csv",index=False)
    coverage=pd.DataFrame([{"source":s,"coverage":float(matrix[c].astype(bool).mean())} for s,c in {
        "internal_elo":"feature_external__internal_elo_available","clubelo":"feature_external__clubelo_available","transfermarkt":"feature_external__transfermarkt_available","understat":"feature_external__understat_available"}.items()])
    coverage.to_csv(OUT_DIR/"v4_phase4_source_coverage.csv",index=False)
    decision="v4_phase4_matrix_ready_research_only" if checks.status.eq("pass").all() else "v4_phase4_matrix_blocked"
    (OUT_DIR/"v4_phase4_feature_matrix_report.md").write_text(f"# V4 Phase 4 Feature Matrix\n\nDecision: **{decision}**\n\nRows={len(matrix)}, columns={len(matrix.columns)}, groups={len(groups)}, checks={int(checks.status.eq('pass').sum())}/{len(checks)}. Locked prior-only blocks were reused; no closing/result field appears in a feature group.\n",encoding="utf-8")
    (OUT_DIR/"v4_phase4_decision.md").write_text(f"# V4 Phase 4 Decision\n\n**{decision}**\n",encoding="utf-8")
    return {"decision":decision,"rows":len(matrix),"columns":len(matrix.columns),"groups":len(groups),"checks_passed":int(checks.status.eq('pass').sum()),"checks":len(checks)}
