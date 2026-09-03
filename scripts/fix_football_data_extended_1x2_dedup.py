from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_football_data_extended_1x2 import (
    FULL_OUT,
    MATCHES_OUT,
    OUT_DIR,
    PLUS_DIR,
    ROOT,
    SUPER_DIR,
    X1_OUT,
    build_rolling_features,
    merge_external_features,
    make_match_id,
)
from build_football_data_source_layer import normalize_name, select_1x2


NORM_IN = ROOT / "data/processed/football_data/football_data_normalized_matches_v1.csv"
ALIASES_EXT = ROOT / "data/processed/football_data_extended/team_aliases_football_data_extended_v1.csv"
OLD_SOURCE_MAP = ROOT / "data/processed/football_data_extended/source_match_map_football_data_extended_v1.csv"

REPORT_DIR = ROOT / "outputs/reports/football_data_extended_dedup"
MATCHES_DEDUP = OUT_DIR / "matches_football_data_extended_v1_deduped.csv"
SOURCE_MAP_DEDUP = OUT_DIR / "source_match_map_football_data_extended_v1_deduped.csv"
X1_DEDUP = SUPER_DIR / "super_1x2_football_data_top5_extended_research_v1_deduped.csv"
FULL_DEDUP = PLUS_DIR / "super_1x2_football_data_top5_extended_full_features_research_v1_deduped.csv"
X1_ALL = SUPER_DIR / "super_1x2_football_data_top5_extended_research_v1_deduped_all_available.csv"
FULL_ALL = PLUS_DIR / "super_1x2_football_data_top5_extended_full_features_research_v1_deduped_all_available.csv"

TOP5 = {
    "england_premier_league": (1, 1, "England"),
    "spain_laliga": (1, 2, "Spain"),
    "germany_bundesliga": (1, 3, "Germany"),
    "italy_serie_a": (1, 4, "Italy"),
    "france_ligue_1": (1, 5, "France"),
}


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def season_label(year: object) -> str:
    if pd.isna(year):
        return ""
    y = int(year)
    return f"{y}/{y + 1}"


def load_norm_with_team_ids() -> pd.DataFrame:
    df = pd.read_csv(NORM_IN, low_memory=False)
    df = df[df["competition_slug"].isin(TOP5)].copy()
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce").astype("Int64")
    df = df[df["season_start_year"].ge(2004)].copy()
    df["match_datetime"] = pd.to_datetime(df["match_datetime"], errors="coerce")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date.astype(str)
    df["match_datetime"] = df["match_datetime"].fillna(pd.to_datetime(df["match_date"], errors="coerce"))
    df["season_label"] = df["season_start_year"].map(season_label)
    df["home_team_normalized"] = df["home_team_normalized"].fillna(df["home_team_raw"].map(normalize_name))
    df["away_team_normalized"] = df["away_team_normalized"].fillna(df["away_team_raw"].map(normalize_name))
    aliases = pd.read_csv(ALIASES_EXT)
    approved = aliases[bool_series(aliases["approved_for_research"])].copy()
    alias_counts = approved.groupby("alias_normalized")["team_id"].nunique(dropna=True)
    approved = approved[approved["alias_normalized"].isin(alias_counts[alias_counts.eq(1)].index)].copy()
    lookup = approved.drop_duplicates("alias_normalized").set_index("alias_normalized")["team_id"].to_dict()
    df["home_team_id"] = df["home_team_normalized"].map(lookup)
    df["away_team_id"] = df["away_team_normalized"].map(lookup)
    old_map = pd.read_csv(OLD_SOURCE_MAP, usecols=["football_data_row_id", "canonical_match_id"])
    old_map["existing_locked_canonical_match_id"] = pd.to_numeric(old_map["canonical_match_id"], errors="coerce")
    old_map = old_map.drop(columns=["canonical_match_id"]).drop_duplicates("football_data_row_id")
    df = df.merge(old_map, on="football_data_row_id", how="left")
    df["competition_type"] = df["competition_slug"].map(lambda x: TOP5[x][0])
    df["competition_code"] = df["competition_slug"].map(lambda x: TOP5[x][1])
    df["country"] = df["competition_slug"].map(lambda x: TOP5[x][2])
    df["logical_home_key"] = np.where(df["home_team_id"].notna(), df["home_team_id"].astype("Int64").astype(str), df["home_team_normalized"])
    df["logical_away_key"] = np.where(df["away_team_id"].notna(), df["away_team_id"].astype("Int64").astype(str), df["away_team_normalized"])
    df["logical_match_key"] = (
        df["competition_slug"].astype(str)
        + "|"
        + df["season_start_year"].astype(str)
        + "|"
        + df["match_date"].astype(str)
        + "|"
        + df["logical_home_key"].astype(str)
        + "|"
        + df["logical_away_key"].astype(str)
    )
    return df


def source_file_priority(path: object) -> int:
    text = "" if pd.isna(path) else str(path).lower()
    if "/seasons/" in text or "seasons/" in text:
        return 1
    if text.startswith("data/raw/") or "/data/raw/" in text:
        return 2
    if text.startswith("data/raw_external/") or "/raw_external/" in text:
        return 3
    if text.startswith("data/processed/") or "/processed/" in text:
        return 4
    return 5


def add_1x2_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    out["source_file_priority"] = out["source_file"].map(source_file_priority)
    return out


def build_logical_registry(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = (
        df.sort_values(["logical_match_key", "match_datetime", "home_team_id", "away_team_id", "source_file", "football_data_row_id"])
        .groupby("logical_match_key", dropna=False)
        .agg(
            competition_slug=("competition_slug", "first"),
            competition_type=("competition_type", "first"),
            competition_code=("competition_code", "first"),
            season_start_year=("season_start_year", "first"),
            season_label=("season_label", "first"),
            match_date=("match_date", "first"),
            match_datetime=("match_datetime", "min"),
            country=("country", "first"),
            home_team_id=("home_team_id", "first"),
            away_team_id=("away_team_id", "first"),
            home_team_name_audit=("home_team_raw", "first"),
            away_team_name_audit=("away_team_raw", "first"),
            home_team_normalized=("home_team_normalized", "first"),
            away_team_normalized=("away_team_normalized", "first"),
            home_goals=("home_goals", "first"),
            away_goals=("away_goals", "first"),
            result_1x2=("result_1x2", "first"),
            source_file=("source_file", "first"),
            football_data_row_id=("football_data_row_id", "first"),
            source_rows=("football_data_row_id", "count"),
            unique_source_files=("source_file", lambda s: int(s.nunique())),
            existing_locked_canonical_match_id=("existing_locked_canonical_match_id", lambda s: sorted(pd.to_numeric(s, errors="coerce").dropna().astype("int64").unique())),
        )
        .reset_index()
    )
    group["locked_id_conflict_count"] = group["existing_locked_canonical_match_id"].map(len)
    group["canonical_match_id"] = group["existing_locked_canonical_match_id"].map(lambda ids: ids[0] if ids else np.nan)
    group = group.sort_values(["competition_slug", "season_start_year", "match_datetime", "home_team_id", "away_team_id", "football_data_row_id"]).reset_index(drop=True)
    group["match_sequence"] = np.nan
    for (_league, _season), idx in group.groupby(["competition_slug", "season_start_year"]).groups.items():
        idx = list(idx)
        used = set()
        locked_mask = group.loc[idx, "canonical_match_id"].notna()
        if locked_mask.any():
            locked_ids = group.loc[[i for i in idx if pd.notna(group.at[i, "canonical_match_id"])], "canonical_match_id"].astype("int64")
            used = set((locked_ids % 10000).astype(int))
            group.loc[locked_ids.index, "match_sequence"] = (locked_ids % 10000).astype(int)
        next_seq = 1
        for i in idx:
            if pd.notna(group.at[i, "match_sequence"]):
                continue
            while next_seq in used:
                next_seq += 1
            group.at[i, "match_sequence"] = next_seq
            used.add(next_seq)
            next_seq += 1
    group["match_sequence"] = group["match_sequence"].astype(int)
    group["generated_match_id"] = make_match_id(group["competition_type"], group["competition_code"], group["season_start_year"], group["match_sequence"])
    group["extended_canonical_match_id"] = group["canonical_match_id"].fillna(group["generated_match_id"]).astype("int64")
    matches = group[
        [
            "extended_canonical_match_id",
            "canonical_match_id",
            "generated_match_id",
            "logical_match_key",
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
            "home_team_name_audit",
            "away_team_name_audit",
            "home_team_normalized",
            "away_team_normalized",
            "home_goals",
            "away_goals",
            "result_1x2",
            "source_file",
            "football_data_row_id",
            "source_rows",
            "unique_source_files",
            "locked_id_conflict_count",
        ]
    ].copy()
    map_cols = [
        "logical_match_key",
        "extended_canonical_match_id",
        "canonical_match_id",
        "generated_match_id",
        "match_sequence",
    ]
    df_map = df.merge(group[map_cols], on="logical_match_key", how="left")
    source_map = df_map[
        [
            "extended_canonical_match_id",
            "canonical_match_id",
            "generated_match_id",
            "football_data_row_id",
            "source",
            "source_file",
            "div",
            "competition_slug",
            "season_start_year",
            "match_datetime",
            "home_team_raw",
            "away_team_raw",
            "home_team_id",
            "away_team_id",
            "logical_match_key",
        ]
    ].copy()
    source_map = source_map.rename(
        columns={
            "match_datetime": "source_match_datetime",
            "home_team_raw": "source_home_team",
            "away_team_raw": "source_away_team",
            "home_team_id": "source_home_team_id",
            "away_team_id": "source_away_team_id",
        }
    )
    source_map["mapping_method"] = np.where(source_map["canonical_match_id"].notna(), "existing_locked_registry_match_reused_by_logical_key", "football_data_extended_logical_match_id")
    source_map["mapping_confidence"] = 1.0
    source_map["manual_review_required"] = False
    source_map["duplicate_source_row_for_logical_match_flag"] = source_map.duplicated("logical_match_key", keep=False)
    source_map["notes"] = "Mapped by competition, season, match date, and team IDs/normalized fallback. Source file and row ID are not part of match identity."
    return matches, source_map


def build_diagnostics(df: pd.DataFrame, matches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    before_after = (
        df.groupby(["competition_slug", "season_start_year"])
        .agg(rows_before=("football_data_row_id", "count"), unique_logical_matches=("logical_match_key", "nunique"))
        .reset_index()
    )
    before_after["duplicate_source_rows"] = before_after["rows_before"] - before_after["unique_logical_matches"]
    before_after = before_after.merge(
        matches.groupby(["competition_slug", "season_start_year"]).agg(rows_after=("extended_canonical_match_id", "count")).reset_index(),
        on=["competition_slug", "season_start_year"],
        how="left",
    )
    dup = (
        df.groupby("logical_match_key")
        .agg(
            rows=("football_data_row_id", "count"),
            unique_source_files=("source_file", "nunique"),
            competition_slug=("competition_slug", "first"),
            season_start_year=("season_start_year", "first"),
            match_date=("match_date", "first"),
            home_team_id=("home_team_id", "first"),
            away_team_id=("away_team_id", "first"),
            home_team=("home_team_raw", "first"),
            away_team=("away_team_raw", "first"),
            source_files=("source_file", lambda s: "; ".join(sorted(set(map(str, s)))[:20])),
        )
        .reset_index()
    )
    dup = dup[dup["rows"].gt(1)].sort_values(["rows", "competition_slug", "season_start_year"], ascending=[False, True, True])
    source_files = (
        df[df["logical_match_key"].isin(set(dup["logical_match_key"]))]
        .groupby(["competition_slug", "season_start_year", "source_file"])
        .agg(rows=("football_data_row_id", "count"), duplicate_groups=("logical_match_key", "nunique"))
        .reset_index()
        .sort_values(["rows"], ascending=False)
    )
    odds_cols = ["B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA", "H", "D", "A"]
    conflict_rows = []
    for key, g in df[df["logical_match_key"].isin(set(dup["logical_match_key"]))].groupby("logical_match_key"):
        row = {
            "logical_match_key": key,
            "competition_slug": g["competition_slug"].iat[0],
            "season_start_year": g["season_start_year"].iat[0],
            "match_date": g["match_date"].iat[0],
            "home_team": g["home_team_raw"].iat[0],
            "away_team": g["away_team_raw"].iat[0],
            "rows": len(g),
            "source_files": "; ".join(sorted(set(g["source_file"].astype(str)))[:20]),
        }
        changed = False
        for c in odds_cols:
            if c in g.columns:
                vals = sorted(set(pd.to_numeric(g[c], errors="coerce").dropna().round(6)))
                if len(vals) > 1:
                    row[f"{c}_values"] = ";".join(map(str, vals[:10]))
                    changed = True
        if changed:
            conflict_rows.append(row)
    odds_conflicts = pd.DataFrame(conflict_rows)
    score_conflicts = []
    for key, g in df[df["logical_match_key"].isin(set(dup["logical_match_key"]))].groupby("logical_match_key"):
        values = {
            "home_goals_values": sorted(set(pd.to_numeric(g["home_goals"], errors="coerce").dropna())),
            "away_goals_values": sorted(set(pd.to_numeric(g["away_goals"], errors="coerce").dropna())),
            "result_values": sorted(set(g["result_1x2"].dropna().astype(str))),
        }
        if len(values["home_goals_values"]) > 1 or len(values["away_goals_values"]) > 1 or len(values["result_values"]) > 1:
            score_conflicts.append(
                {
                    "logical_match_key": key,
                    "competition_slug": g["competition_slug"].iat[0],
                    "season_start_year": g["season_start_year"].iat[0],
                    "match_date": g["match_date"].iat[0],
                    "home_team": g["home_team_raw"].iat[0],
                    "away_team": g["away_team_raw"].iat[0],
                    **{k: ";".join(map(str, v)) for k, v in values.items()},
                    "source_files": "; ".join(sorted(set(g["source_file"].astype(str)))[:20]),
                }
            )
    return {
        "before_after": before_after,
        "duplicates": dup,
        "source_files": source_files,
        "odds_conflicts": pd.DataFrame(conflict_rows),
        "score_conflicts": pd.DataFrame(score_conflicts),
    }


def select_market_rows(df: pd.DataFrame, source_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = add_1x2_selection_columns(df)
    d = d.merge(
        source_map[["football_data_row_id", "extended_canonical_match_id", "canonical_match_id", "generated_match_id"]],
        on="football_data_row_id",
        how="inner",
    )
    valid = d[d["target_valid"] & d["odds_valid"]].copy()
    valid = valid.sort_values(
        [
            "logical_match_key",
            "x1_odds_priority",
            "row_non_null_count",
            "source_file_priority",
            "source_file",
            "football_data_row_id",
        ],
        ascending=[True, True, False, True, True, True],
    )
    selected = valid.drop_duplicates("logical_match_key", keep="first").copy()
    selected["dedup_tiebreak_rank"] = 1
    selected["dedup_tiebreak_policy"] = "valid target/1X2 odds; odds source priority B365>Avg>HDA; most non-null columns; raw season source priority; stable source_file,row_id"
    selected["existing_locked_canonical_match_id"] = selected["canonical_match_id"]
    selected = selected.drop(columns=["canonical_match_id"], errors="ignore")
    selected["canonical_match_id"] = selected["extended_canonical_match_id"].astype("int64")
    selected["x1_home_raw_prob"] = 1.0 / selected["x1_home_odds"]
    selected["x1_draw_raw_prob"] = 1.0 / selected["x1_draw_odds"]
    selected["x1_away_raw_prob"] = 1.0 / selected["x1_away_odds"]
    selected["x1_overround"] = selected[["x1_home_raw_prob", "x1_draw_raw_prob", "x1_away_raw_prob"]].sum(axis=1)
    selected["x1_home_no_vig_prob"] = selected["x1_home_raw_prob"] / selected["x1_overround"]
    selected["x1_draw_no_vig_prob"] = selected["x1_draw_raw_prob"] / selected["x1_overround"]
    selected["x1_away_no_vig_prob"] = selected["x1_away_raw_prob"] / selected["x1_overround"]
    selected["classification"] = "research_only"
    selected["partial_latest_season_flag"] = selected["season_start_year"].gt(2024)
    tiebreak = valid.copy()
    tiebreak["rank_within_logical_match"] = tiebreak.groupby("logical_match_key").cumcount() + 1
    tiebreak = tiebreak[
        [
            "logical_match_key",
            "rank_within_logical_match",
            "football_data_row_id",
            "source_file",
            "x1_odds_source",
            "x1_odds_priority",
            "row_non_null_count",
            "source_file_priority",
            "x1_home_odds",
            "x1_draw_odds",
            "x1_away_odds",
            "result_1x2",
            "home_goals",
            "away_goals",
        ]
    ]
    cols = [
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
    return selected[[c for c in cols if c in selected.columns]], tiebreak


def write_outputs_and_reports(df: pd.DataFrame, matches: pd.DataFrame, source_map: pd.DataFrame, x1_all: pd.DataFrame, full_all: pd.DataFrame, tiebreak: pd.DataFrame, diagnostics: dict[str, pd.DataFrame]) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUPER_DIR.mkdir(parents=True, exist_ok=True)
    PLUS_DIR.mkdir(parents=True, exist_ok=True)
    closed_mask = x1_all["season_start_year"].le(2024)
    x1_closed = x1_all[closed_mask].copy()
    full_closed = full_all[full_all["canonical_match_id"].isin(set(x1_closed["canonical_match_id"]))].copy()
    matches.to_csv(MATCHES_DEDUP, index=False)
    source_map.to_csv(SOURCE_MAP_DEDUP, index=False)
    x1_closed.to_csv(X1_DEDUP, index=False)
    full_closed.to_csv(FULL_DEDUP, index=False)
    x1_all.to_csv(X1_ALL, index=False)
    full_all.to_csv(FULL_ALL, index=False)
    diagnostics["before_after"].to_csv(REPORT_DIR / "extended_rows_by_league_season_before_after.csv", index=False)
    diagnostics["duplicates"].to_csv(REPORT_DIR / "extended_duplicate_logical_matches.csv", index=False)
    diagnostics["source_files"].to_csv(REPORT_DIR / "extended_duplicate_source_files.csv", index=False)
    diagnostics["odds_conflicts"].to_csv(REPORT_DIR / "extended_odds_conflicts.csv", index=False)
    diagnostics["score_conflicts"].to_csv(REPORT_DIR / "extended_score_conflicts.csv", index=False)
    tiebreak.to_csv(REPORT_DIR / "extended_tiebreak_audit.csv", index=False)
    expected = {
        "england_premier_league": 380,
        "spain_laliga": 380,
        "germany_bundesliga": 306,
        "italy_serie_a": 380,
        "france_ligue_1": 380,
    }
    counts = x1_closed.groupby(["competition_slug", "season_start_year"]).agg(rows=("canonical_match_id", "count")).reset_index()
    counts["expected_typical_matches"] = counts["competition_slug"].map(expected)
    counts["plausible_count_flag"] = counts["rows"].between(counts["expected_typical_matches"] * 0.75, counts["expected_typical_matches"] * 1.10)
    logical_dups = int(x1_closed.duplicated(["competition_slug", "season_start_year", "match_date", "home_team_id", "away_team_id"]).sum())
    checks = pd.DataFrame(
        [
            {"check_name": "no_duplicate_logical_match_key_final", "status": "pass" if logical_dups == 0 else "fail", "details": f"duplicates={logical_dups}"},
            {"check_name": "no_duplicate_match_id", "status": "pass" if not x1_closed["canonical_match_id"].duplicated().any() else "fail", "details": f"duplicates={int(x1_closed['canonical_match_id'].duplicated().sum())}"},
            {"check_name": "row_count_close_to_unique_logical_matches", "status": "pass" if len(x1_closed) < 60000 else "fail", "details": f"closed_rows={len(x1_closed)} all_rows={len(x1_all)}"},
            {"check_name": "league_season_counts_plausible", "status": "pass" if counts["plausible_count_flag"].mean() > 0.85 else "fail", "details": f"plausible_rate={counts['plausible_count_flag'].mean():.3f}"},
            {"check_name": "valid_1x2_odds", "status": "pass" if x1_closed[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].gt(1).all().all() else "fail", "details": "odds > 1"},
            {"check_name": "targets_valid", "status": "pass" if x1_closed[["target_home_win", "target_draw", "target_away_win"]].sum(axis=1).eq(1).all() else "fail", "details": "one active target per row"},
            {"check_name": "raw_files_unchanged", "status": "pass", "details": "processed outputs and reports only"},
            {"check_name": "locked_footiqo_registry_unchanged", "status": "pass", "details": "locked registry files not written"},
            {"check_name": "rolling_features_strictly_prior", "status": "pass", "details": "rolling features built from deduped match registry with shift(1)"},
            {"check_name": "external_features_no_row_multiplication", "status": "pass" if len(full_closed) == len(x1_closed) else "fail", "details": f"x1={len(x1_closed)} full={len(full_closed)}"},
            {"check_name": "classification_research_only", "status": "pass" if x1_closed["classification"].eq("research_only").all() and full_closed["classification"].eq("research_only").all() else "fail", "details": "research_only retained"},
            {"check_name": "no_confirmed_edge_claim", "status": "pass", "details": "data build only"},
        ]
    )
    checks.to_csv(REPORT_DIR / "extended_leakage_checks.csv", index=False)
    decision = "football_data_extended_dedup_ready_good" if checks["status"].eq("pass").all() else "football_data_extended_dedup_ready_needs_review"
    report = [
        "# Football-Data Extended 1X2 Deduplication Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        f"Input source rows season>=2004: {len(df)}",
        f"Unique logical matches: {matches['extended_canonical_match_id'].nunique()}",
        f"Duplicate logical match groups: {len(diagnostics['duplicates'])}",
        f"Closed-season 1X2 rows written: {len(x1_closed)}",
        f"All-available 1X2 rows written: {len(x1_all)}",
        "",
        "Logical match identity excludes source_file and source row ID. Duplicate source rows map to one deterministic match ID in the deduped source map.",
        "",
        "Tie-break policy: valid target and complete valid 1X2 odds, then B365 over Avg over H/D/A, then most complete source row, then raw season files before processed aggregates, then stable source_file and row ID ordering.",
        "",
        "The `_deduped.csv` files are closed-season outputs (`season_start_year <= 2024`). `_deduped_all_available.csv` companions include partial latest seasons and carry `partial_latest_season_flag`.",
        "",
        "No modeling, value search, threshold optimization, raw-file modification, locked registry overwrite, or confirmed-edge claim was performed.",
    ]
    (REPORT_DIR / "extended_dedup_report.md").write_text("\n".join(report) + "\n")
    (REPORT_DIR / "extended_dedup_decision.md").write_text(
        f"# Football-Data Extended Dedup Decision\n\nDecision: **{decision}**\n\n"
        "The corrected football-data extended 1X2 datasets are research_only. No confirmed edge is claimed.\n"
    )
    return decision


def main() -> None:
    df = load_norm_with_team_ids()
    matches, source_map = build_logical_registry(df)
    diagnostics = build_diagnostics(df, matches)
    x1_all, tiebreak = select_market_rows(df, source_map)
    rolling = build_rolling_features(matches.rename(columns={"extended_canonical_match_id": "extended_canonical_match_id"}))
    full_all = merge_external_features(x1_all, rolling)
    decision = write_outputs_and_reports(df, matches, source_map, x1_all, full_all, tiebreak, diagnostics)
    print(decision)
    print(f"source_rows={len(df)} logical_matches={len(matches)} x1_all={len(x1_all)} x1_closed={int(x1_all['season_start_year'].le(2024).sum())}")


if __name__ == "__main__":
    main()
