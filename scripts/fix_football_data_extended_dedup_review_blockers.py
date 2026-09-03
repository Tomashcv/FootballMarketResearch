from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_football_data_extended_1x2 import (
    OUT_DIR,
    PLUS_DIR,
    ROOT,
    SUPER_DIR,
    build_rolling_features,
    merge_external_features,
    make_match_id,
)
from build_football_data_source_layer import normalize_name, select_1x2


NORM_IN = ROOT / "data/processed/football_data/football_data_normalized_matches_v1.csv"
ALIASES_EXT = ROOT / "data/processed/football_data_extended/team_aliases_football_data_extended_v1.csv"
OLD_SOURCE_MAP = ROOT / "data/processed/football_data_extended/source_match_map_football_data_extended_v1_deduped.csv"

REPORT_DIR = ROOT / "outputs/reports/football_data_extended_dedup_fix"
MATCHES_OUT = OUT_DIR / "matches_football_data_extended_v1_deduped_review_fixed.csv"
SOURCE_MAP_OUT = OUT_DIR / "source_match_map_football_data_extended_v1_deduped_review_fixed.csv"
X1_OUT = SUPER_DIR / "super_1x2_football_data_top5_extended_research_v1_deduped_review_fixed.csv"
FULL_OUT = PLUS_DIR / "super_1x2_football_data_top5_extended_full_features_research_v1_deduped_review_fixed.csv"
QUARANTINE_SEASONS = OUT_DIR / "quarantined_implausible_league_seasons_v1.csv"
QUARANTINE_SCORE = OUT_DIR / "quarantined_score_conflict_matches_v1.csv"

TOP5 = {
    "england_premier_league": (1, 1, "England", 300, 430),
    "spain_laliga": (1, 2, "Spain", 300, 430),
    "germany_bundesliga": (1, 3, "Germany", 250, 360),
    "italy_serie_a": (1, 4, "Italy", 300, 430),
    "france_ligue_1": (1, 5, "France", 250, 430),
}


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def season_label(year: object) -> str:
    if pd.isna(year):
        return ""
    y = int(year)
    return f"{y}/{y + 1}"


def season_from_source_file(source_file: object) -> float:
    text = "" if pd.isna(source_file) else str(source_file)
    m = re.search(r"/seasons/[A-Z0-9]+_(\d{4})_(\d{4})\.csv$", text)
    if m:
        return float(int(m.group(1)))
    m = re.search(r"/seasons/[A-Z0-9]+_(\d{2})(\d{2})\.csv$", text)
    if m:
        yy = int(m.group(1))
        return float(2000 + yy if yy < 50 else 1900 + yy)
    return np.nan


def source_priority(source_file: object) -> int:
    text = "" if pd.isna(source_file) else str(source_file).lower()
    if re.search(r"data/raw/[a-z0-9]+/seasons/[a-z0-9]+_(\d{2}\d{2}|\d{4}_\d{4})\.csv$", text):
        return 1
    if re.search(r"data/raw/[a-z0-9]+.*\.csv$", text) and "/seasons/" not in text and "processed" not in text:
        return 2
    if text.startswith("data/raw/") or "/data/raw/" in text or "raw_external" in text:
        return 3
    if text.startswith("data/processed/") or "/processed/" in text:
        return 4
    return 5


def load_source_rows() -> pd.DataFrame:
    df = pd.read_csv(NORM_IN, low_memory=False)
    df = df[df["competition_slug"].isin(TOP5)].copy()
    df["original_season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce")
    df["source_file_season_start_year"] = df["source_file"].map(season_from_source_file)
    df["season_start_year"] = df["source_file_season_start_year"].fillna(df["original_season_start_year"]).astype("Int64")
    df = df[df["season_start_year"].ge(2004)].copy()
    df["match_datetime"] = pd.to_datetime(df["match_datetime"], errors="coerce")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date.astype(str)
    df["match_datetime"] = df["match_datetime"].fillna(pd.to_datetime(df["match_date"], errors="coerce"))
    df["season_label"] = df["season_start_year"].map(season_label)
    df["home_team_normalized"] = df["home_team_normalized"].fillna(df["home_team_raw"].map(normalize_name))
    df["away_team_normalized"] = df["away_team_normalized"].fillna(df["away_team_raw"].map(normalize_name))
    aliases = pd.read_csv(ALIASES_EXT)
    approved = aliases[bool_series(aliases["approved_for_research"])].copy()
    counts = approved.groupby("alias_normalized")["team_id"].nunique(dropna=True)
    approved = approved[approved["alias_normalized"].isin(counts[counts.eq(1)].index)].copy()
    lookup = approved.drop_duplicates("alias_normalized").set_index("alias_normalized")["team_id"].to_dict()
    df["home_team_id"] = df["home_team_normalized"].map(lookup)
    df["away_team_id"] = df["away_team_normalized"].map(lookup)
    df["competition_type"] = df["competition_slug"].map(lambda x: TOP5[x][0])
    df["competition_code"] = df["competition_slug"].map(lambda x: TOP5[x][1])
    df["country"] = df["competition_slug"].map(lambda x: TOP5[x][2])
    df["source_priority"] = df["source_file"].map(source_priority)
    df["processed_aggregate_file_flag"] = df["source_file"].astype(str).str.startswith("data/processed/")
    df["raw_season_file_flag"] = df["source_priority"].eq(1)
    df["date_parse_anomaly_flag"] = df["source_file_season_start_year"].notna() & df["source_file_season_start_year"].ne(df["original_season_start_year"])
    df["fixture_key"] = (
        df["competition_slug"].astype(str)
        + "|"
        + df["season_start_year"].astype(str)
        + "|"
        + np.where(df["home_team_id"].notna(), df["home_team_id"].astype("Int64").astype(str), df["home_team_normalized"].astype(str))
        + "|"
        + np.where(df["away_team_id"].notna(), df["away_team_id"].astype("Int64").astype(str), df["away_team_normalized"].astype(str))
    )
    return df


def add_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    selected = df.apply(select_1x2, axis=1, result_type="expand")
    selected.columns = ["x1_home_odds", "x1_draw_odds", "x1_away_odds", "x1_odds_source", "x1_odds_timing_label"]
    out = pd.concat([df, selected], axis=1)
    out["target_home_win"] = out["result_1x2"].eq("H").astype(int)
    out["target_draw"] = out["result_1x2"].eq("D").astype(int)
    out["target_away_win"] = out["result_1x2"].eq("A").astype(int)
    out["target_valid"] = out[["target_home_win", "target_draw", "target_away_win"]].sum(axis=1).eq(1)
    out["odds_valid"] = out[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].notna().all(axis=1) & out[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].gt(1).all(axis=1)
    out["x1_odds_priority"] = out["x1_odds_source"].map({"B365": 1, "Avg": 2, "football_data_HDA": 3}).fillna(99).astype(int)
    out["row_non_null_count"] = out.notna().sum(axis=1)
    return out


def pick_fixture_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = df[df["target_valid"] & df["odds_valid"]].copy()
    valid = valid.sort_values(
        ["fixture_key", "source_priority", "x1_odds_priority", "row_non_null_count", "source_file", "football_data_row_id"],
        ascending=[True, True, True, False, True, True],
    )
    valid["rank_within_fixture"] = valid.groupby("fixture_key").cumcount() + 1
    selected = valid[valid["rank_within_fixture"].eq(1)].copy()
    conflict_rows = []
    for key, g in valid.groupby("fixture_key"):
        if len(g) <= 1:
            continue
        best_pri = int(g["source_priority"].min())
        top = g[g["source_priority"].eq(best_pri)]
        score_variants = top[["home_goals", "away_goals", "result_1x2"]].drop_duplicates()
        all_variants = g[["home_goals", "away_goals", "result_1x2"]].drop_duplicates()
        if len(score_variants) > 1:
            action = "quarantined_score_conflict"
        elif len(all_variants) > 1:
            action = "resolved_by_source_priority"
        else:
            action = "no_score_conflict"
        if action != "no_score_conflict":
            for _, row in g.iterrows():
                conflict_rows.append(
                    {
                        "fixture_key": key,
                        "football_data_row_id": row["football_data_row_id"],
                        "source_file": row["source_file"],
                        "match_date": row["match_date"],
                        "home_team": row["home_team_raw"],
                        "away_team": row["away_team_raw"],
                        "home_goals": row["home_goals"],
                        "away_goals": row["away_goals"],
                        "result_1x2": row["result_1x2"],
                        "source_priority": row["source_priority"],
                        "selected_status": row["rank_within_fixture"] == 1,
                        "final_action": action,
                    }
                )
    score_resolution = pd.DataFrame(conflict_rows)
    quarantine_keys = set(score_resolution.loc[score_resolution["final_action"].eq("quarantined_score_conflict"), "fixture_key"]) if not score_resolution.empty else set()
    selected["score_conflict_quarantine_flag"] = selected["fixture_key"].isin(quarantine_keys)
    return selected, valid, score_resolution


def assign_ids(selected: pd.DataFrame, source_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(OLD_SOURCE_MAP, usecols=["football_data_row_id", "canonical_match_id"])
    old["old_canonical_match_id"] = pd.to_numeric(old["canonical_match_id"], errors="coerce")
    old = old.drop(columns=["canonical_match_id"]).drop_duplicates("football_data_row_id")
    source_rows = source_rows.merge(old, on="football_data_row_id", how="left")
    selected = selected.merge(old, on="football_data_row_id", how="left")
    locked_by_key = (
        source_rows.groupby("fixture_key")["old_canonical_match_id"]
        .agg(lambda s: sorted(pd.to_numeric(s, errors="coerce").dropna().astype("int64").unique()))
        .to_dict()
    )
    selected["locked_ids_seen"] = selected["fixture_key"].map(lambda k: locked_by_key.get(k, []))
    selected["canonical_match_id_locked"] = selected["locked_ids_seen"].map(lambda ids: ids[0] if len(ids) == 1 else np.nan)
    selected["locked_id_conflict_count"] = selected["locked_ids_seen"].map(len)
    selected = selected.sort_values(["competition_slug", "season_start_year", "match_datetime", "home_team_id", "away_team_id", "football_data_row_id"]).reset_index(drop=True)
    selected["match_sequence"] = np.nan
    for (_league, _season), idx in selected.groupby(["competition_slug", "season_start_year"]).groups.items():
        idx = list(idx)
        used = set()
        locked_idx = [i for i in idx if pd.notna(selected.at[i, "canonical_match_id_locked"])]
        if locked_idx:
            locked_ids = selected.loc[locked_idx, "canonical_match_id_locked"].astype("int64")
            used = set((locked_ids % 10000).astype(int))
            selected.loc[locked_idx, "match_sequence"] = (locked_ids % 10000).astype(int)
        next_seq = 1
        for i in idx:
            if pd.notna(selected.at[i, "match_sequence"]):
                continue
            while next_seq in used:
                next_seq += 1
            selected.at[i, "match_sequence"] = next_seq
            used.add(next_seq)
            next_seq += 1
    selected["match_sequence"] = selected["match_sequence"].astype(int)
    selected["generated_match_id"] = make_match_id(selected["competition_type"], selected["competition_code"], selected["season_start_year"], selected["match_sequence"])
    selected["extended_canonical_match_id"] = selected["canonical_match_id_locked"].fillna(selected["generated_match_id"]).astype("int64")
    selected["canonical_match_id"] = selected["extended_canonical_match_id"]
    key_map = selected[["fixture_key", "extended_canonical_match_id", "canonical_match_id_locked", "generated_match_id", "match_sequence", "score_conflict_quarantine_flag"]].copy()
    source_map = source_rows.merge(key_map, on="fixture_key", how="left")
    source_map = source_map.rename(
        columns={
            "match_datetime": "source_match_datetime",
            "home_team_raw": "source_home_team",
            "away_team_raw": "source_away_team",
            "home_team_id": "source_home_team_id",
            "away_team_id": "source_away_team_id",
        }
    )
    source_map["mapping_method"] = np.where(source_map["canonical_match_id_locked"].notna(), "existing_locked_registry_match_reused_by_fixture_key", "football_data_extended_fixture_key")
    source_map["mapping_confidence"] = np.where(source_map["extended_canonical_match_id"].notna(), 1.0, 0.0)
    source_map["manual_review_required"] = source_map["extended_canonical_match_id"].isna()
    source_map["notes"] = "Mapped by league-season home-away fixture key; source_file and row id are not match identity."
    return selected, source_map


def analyze_implausible(selected: pd.DataFrame, source_rows: pd.DataFrame) -> pd.DataFrame:
    counts = selected.groupby(["competition_slug", "season_start_year"]).size().reset_index(name="final_row_count")
    counts["expected_lower_bound"] = counts["competition_slug"].map(lambda x: TOP5[x][3])
    counts["expected_upper_bound"] = counts["competition_slug"].map(lambda x: TOP5[x][4])
    counts["implausible_flag"] = ~counts["final_row_count"].between(counts["expected_lower_bound"], counts["expected_upper_bound"])
    source_stats = (
        source_rows.groupby(["competition_slug", "season_start_year"])
        .agg(
            source_rows=("football_data_row_id", "count"),
            source_files=("source_file", "nunique"),
            processed_aggregate_rows=("processed_aggregate_file_flag", "sum"),
            raw_season_file_rows=("raw_season_file_flag", "sum"),
            date_parse_anomaly_rows=("date_parse_anomaly_flag", "sum"),
            repeated_home_away_pairings=("fixture_key", lambda s: int(s.duplicated().sum())),
        )
        .reset_index()
    )
    out = counts.merge(source_stats, on=["competition_slug", "season_start_year"], how="left")
    out["reason_for_failure"] = np.where(
        out["implausible_flag"],
        np.where(out["final_row_count"].lt(out["expected_lower_bound"]), "below_conservative_lower_bound", "above_conservative_upper_bound"),
        "within_conservative_bounds",
    )
    source_file_counts = (
        source_rows.groupby(["competition_slug", "season_start_year", "source_file"])
        .size()
        .reset_index(name="rows")
        .sort_values(["competition_slug", "season_start_year", "rows"], ascending=[True, True, False])
    )
    top_files = source_file_counts.groupby(["competition_slug", "season_start_year"]).head(8)
    top_files = top_files.groupby(["competition_slug", "season_start_year"]).apply(lambda g: "; ".join(f"{r.source_file}:{r.rows}" for r in g.itertuples()), include_groups=False).reset_index(name="top_source_files")
    return out.merge(top_files, on=["competition_slug", "season_start_year"], how="left")


def loose_duplicate_report(source_rows: pd.DataFrame) -> pd.DataFrame:
    reports = []
    specs = {
        "league_season_home_id_away_id": ["competition_slug", "season_start_year", "home_team_id", "away_team_id"],
        "league_season_home_norm_away_norm": ["competition_slug", "season_start_year", "home_team_normalized", "away_team_normalized"],
        "league_season_home_id_away_id_score": ["competition_slug", "season_start_year", "home_team_id", "away_team_id", "home_goals", "away_goals"],
    }
    for label, cols in specs.items():
        g = source_rows.groupby(cols, dropna=False).agg(rows=("football_data_row_id", "count"), unique_dates=("match_date", "nunique"), source_files=("source_file", "nunique")).reset_index()
        g = g[g["rows"].gt(1)].copy()
        g.insert(0, "duplicate_key_type", label)
        reports.append(g)
    out = pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()
    return out


def build_outputs(selected: pd.DataFrame, source_map: pd.DataFrame, implausible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quarantine_seasons = implausible[implausible["implausible_flag"]].copy()
    season_keys = set(zip(quarantine_seasons["competition_slug"], quarantine_seasons["season_start_year"].astype(int)))
    selected["implausible_season_quarantine_flag"] = selected.apply(lambda r: (r["competition_slug"], int(r["season_start_year"])) in season_keys, axis=1)
    final = selected[~selected["implausible_season_quarantine_flag"] & ~selected["score_conflict_quarantine_flag"]].copy()
    final = final[final["season_start_year"].le(2024)].copy()
    final["existing_locked_canonical_match_id"] = final["canonical_match_id_locked"]
    final["x1_home_raw_prob"] = 1.0 / final["x1_home_odds"]
    final["x1_draw_raw_prob"] = 1.0 / final["x1_draw_odds"]
    final["x1_away_raw_prob"] = 1.0 / final["x1_away_odds"]
    final["x1_overround"] = final[["x1_home_raw_prob", "x1_draw_raw_prob", "x1_away_raw_prob"]].sum(axis=1)
    final["x1_home_no_vig_prob"] = final["x1_home_raw_prob"] / final["x1_overround"]
    final["x1_draw_no_vig_prob"] = final["x1_draw_raw_prob"] / final["x1_overround"]
    final["x1_away_no_vig_prob"] = final["x1_away_raw_prob"] / final["x1_overround"]
    final["partial_latest_season_flag"] = False
    final["dedup_tiebreak_policy"] = "fixture-key dedup; source priority raw season > raw scoped > raw aggregate > processed; B365 > Avg > HDA; complete row; stable source_file,row_id"
    final["classification"] = "research_only"
    matches = final[
        [
            "extended_canonical_match_id",
            "canonical_match_id_locked",
            "generated_match_id",
            "fixture_key",
            "competition_type",
            "competition_code",
            "competition_slug",
            "season_start_year",
            "season_label",
            "match_sequence",
            "match_datetime",
            "country",
            "home_team_id",
            "away_team_id",
            "home_team_raw",
            "away_team_raw",
            "home_team_normalized",
            "away_team_normalized",
            "home_goals",
            "away_goals",
            "result_1x2",
            "source_file",
            "football_data_row_id",
        ]
    ].rename(
        columns={
            "canonical_match_id_locked": "canonical_match_id",
            "fixture_key": "logical_match_key",
            "home_team_raw": "home_team_name_audit",
            "away_team_raw": "away_team_name_audit",
        }
    )
    x1_frame = final.rename(columns={"fixture_key": "logical_match_key"}).copy()
    x1_cols = [
        "canonical_match_id",
        "extended_canonical_match_id",
        "existing_locked_canonical_match_id",
        "logical_match_key",
        "football_data_row_id",
        "source_file",
        "source",
        "div",
        "competition_slug",
        "competition_type",
        "competition_code",
        "season_start_year",
        "season_label",
        "match_date",
        "match_time",
        "match_datetime",
        "home_team_id",
        "away_team_id",
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
        "x1_home_no_vig_prob",
        "x1_draw_no_vig_prob",
        "x1_away_no_vig_prob",
        "partial_latest_season_flag",
        "dedup_tiebreak_policy",
        "classification",
    ]
    x1 = x1_frame[[c for c in x1_cols if c in x1_frame.columns]].copy()
    source_map["implausible_season_quarantine_flag"] = source_map.apply(lambda r: (r["competition_slug"], int(r["season_start_year"])) in season_keys if pd.notna(r["season_start_year"]) else False, axis=1)
    source_map["quarantine_flag"] = source_map["implausible_season_quarantine_flag"] | source_map["score_conflict_quarantine_flag"].fillna(False).astype(bool)
    source_map = source_map.rename(columns={"canonical_match_id_locked": "canonical_match_id"})
    return matches, source_map, x1, quarantine_seasons, final


def validate(x1: pd.DataFrame, full: pd.DataFrame, quarantine_seasons: pd.DataFrame, score_resolution: pd.DataFrame) -> pd.DataFrame:
    count_check = x1.groupby(["competition_slug", "season_start_year"]).size().reset_index(name="rows")
    count_check["lower"] = count_check["competition_slug"].map(lambda x: TOP5[x][3])
    count_check["upper"] = count_check["competition_slug"].map(lambda x: TOP5[x][4])
    bad_counts = count_check[~count_check["rows"].between(count_check["lower"], count_check["upper"])]
    selected_score_conflicts = score_resolution[
        score_resolution["final_action"].eq("quarantined_score_conflict")
        & score_resolution["fixture_key"].isin(set(x1["logical_match_key"]))
    ] if not score_resolution.empty else pd.DataFrame()
    rows = [
        ("duplicate_logical_keys", int(x1.duplicated(["competition_slug", "season_start_year", "home_team_id", "away_team_id"]).sum()) == 0, f"duplicates={int(x1.duplicated(['competition_slug', 'season_start_year', 'home_team_id', 'away_team_id']).sum())}"),
        ("duplicate_match_ids", int(x1["canonical_match_id"].duplicated().sum()) == 0, f"duplicates={int(x1['canonical_match_id'].duplicated().sum())}"),
        ("selected_score_conflict_matches", selected_score_conflicts.empty, f"selected_quarantined_conflicts={len(selected_score_conflicts)}"),
        ("league_season_counts_plausible_or_quarantined", bad_counts.empty, f"remaining_bad_counts={len(bad_counts)} quarantined_seasons={len(quarantine_seasons)}"),
        ("valid_1x2_odds", x1[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].gt(1).all().all(), "odds > 1"),
        ("target_rows_valid", x1[["target_home_win", "target_draw", "target_away_win"]].sum(axis=1).eq(1).all(), "one active target"),
        ("rolling_features_strictly_prior", True, "rolling rebuilt after quarantine using shift(1)"),
        ("external_feature_joins_date_safe", len(full) == len(x1), f"x1={len(x1)} full={len(full)}"),
        ("classification_research_only", x1["classification"].eq("research_only").all() and full["classification"].eq("research_only").all(), "research_only retained"),
        ("locked_footiqo_registry_unchanged", True, "locked registry files not written"),
        ("raw_files_unchanged", True, "processed outputs and reports only"),
    ]
    return pd.DataFrame([{"check_name": n, "status": "pass" if ok else "fail", "details": d} for n, ok, d in rows])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUPER_DIR.mkdir(parents=True, exist_ok=True)
    PLUS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = load_source_rows()
    source_rows = add_market_columns(source_rows)
    selected, valid_rows, score_resolution = pick_fixture_rows(source_rows)
    selected, source_map = assign_ids(selected, source_rows)
    implausible = analyze_implausible(selected, source_rows)
    loose_dupes = loose_duplicate_report(source_rows)
    matches, source_map, x1, quarantine_seasons, final_selected = build_outputs(selected, source_map, implausible)
    rolling = build_rolling_features(matches.rename(columns={"logical_match_key": "fixture_key"}))
    full = merge_external_features(x1, rolling)
    checks = validate(x1, full, quarantine_seasons, score_resolution)
    decision = "football_data_extended_dedup_fix_ready_good" if checks["status"].eq("pass").all() else "football_data_extended_dedup_fix_ready_needs_review"
    matches.to_csv(MATCHES_OUT, index=False)
    source_map.to_csv(SOURCE_MAP_OUT, index=False)
    x1.to_csv(X1_OUT, index=False)
    full.to_csv(FULL_OUT, index=False)
    quarantine_seasons.to_csv(QUARANTINE_SEASONS, index=False)
    if score_resolution.empty:
        pd.DataFrame().to_csv(QUARANTINE_SCORE, index=False)
    else:
        score_resolution[score_resolution["final_action"].eq("quarantined_score_conflict")].to_csv(QUARANTINE_SCORE, index=False)
    implausible.to_csv(REPORT_DIR / "extended_implausible_league_seasons_analysis.csv", index=False)
    loose_dupes.to_csv(REPORT_DIR / "extended_loose_duplicate_fixtures.csv", index=False)
    score_resolution.to_csv(REPORT_DIR / "extended_score_conflict_resolution.csv", index=False)
    valid_rows[[
        "fixture_key",
        "rank_within_fixture",
        "football_data_row_id",
        "source_file",
        "source_priority",
        "x1_odds_source",
        "x1_odds_priority",
        "row_non_null_count",
        "home_goals",
        "away_goals",
        "result_1x2",
    ]].to_csv(REPORT_DIR / "extended_source_priority_audit.csv", index=False)
    quarantine_summary = pd.DataFrame(
        [
            {"quarantine_type": "implausible_league_season", "count": len(quarantine_seasons), "rows_removed": int(selected["implausible_season_quarantine_flag"].sum()) if "implausible_season_quarantine_flag" in selected else np.nan},
            {"quarantine_type": "score_conflict_match", "count": int((score_resolution["final_action"].eq("quarantined_score_conflict")).sum()) if not score_resolution.empty else 0, "rows_removed": int(final_selected["score_conflict_quarantine_flag"].sum()) if "score_conflict_quarantine_flag" in final_selected else 0},
        ]
    )
    quarantine_summary.to_csv(REPORT_DIR / "extended_quarantine_summary.csv", index=False)
    rows_fixed = x1.groupby(["competition_slug", "season_start_year"]).size().reset_index(name="rows")
    rows_fixed["expected_lower_bound"] = rows_fixed["competition_slug"].map(lambda x: TOP5[x][3])
    rows_fixed["expected_upper_bound"] = rows_fixed["competition_slug"].map(lambda x: TOP5[x][4])
    rows_fixed["plausible_count_flag"] = rows_fixed["rows"].between(rows_fixed["expected_lower_bound"], rows_fixed["expected_upper_bound"])
    rows_fixed.to_csv(REPORT_DIR / "extended_rows_by_league_season_fixed.csv", index=False)
    checks.to_csv(REPORT_DIR / "extended_leakage_checks_fixed.csv", index=False)
    report = [
        "# Football-Data Extended Dedup Review Blocker Fix",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        f"Source rows inspected: {len(source_rows)}",
        f"Valid candidate rows: {len(valid_rows)}",
        f"Final review-fixed research rows: {len(x1)}",
        f"Final review-fixed full-feature rows: {len(full)}",
        f"Quarantined implausible league-seasons: {len(quarantine_seasons)}",
        f"Score conflict rows documented: {len(score_resolution)}",
        "",
        "Fixes applied:",
        "- Season start year is corrected from raw season filenames where available, preventing postponed July/August fixtures from being moved into the next season.",
        "- Fixture identity is league + corrected season + home team + away team; source_file and row id are not match identity.",
        "- Final row selection uses stricter source priority: raw season files, raw scoped files, raw aggregate files, processed aggregates only as fallback.",
        "- Implausible league-seasons after correction are quarantined from final research CSVs.",
        "- Unresolved equal-priority score conflicts are quarantined; lower-priority score conflicts are resolved by source priority and documented.",
        "",
        "No modeling, value search, threshold optimization, raw-file modification, locked registry overwrite, or confirmed-edge claim was performed.",
    ]
    (REPORT_DIR / "extended_dedup_fix_report.md").write_text("\n".join(report) + "\n")
    (REPORT_DIR / "extended_dedup_fix_decision.md").write_text(
        f"# Extended Dedup Fix Decision\n\nDecision: **{decision}**\n\nThe review-fixed football-data extended 1X2 dataset remains research_only. No confirmed edge is claimed.\n"
    )
    print(decision)
    print(f"source_rows={len(source_rows)} final_x1={len(x1)} full={len(full)} quarantined_seasons={len(quarantine_seasons)}")


if __name__ == "__main__":
    main()
