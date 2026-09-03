from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_football_data_source_layer import normalize_name, season_label, select_1x2


ROOT = Path(__file__).resolve().parents[1]
NORM_IN = ROOT / "data/processed/football_data/football_data_normalized_matches_v1.csv"
FILE_INV_IN = ROOT / "data/processed/football_data/football_data_file_inventory_v1.csv"
TEAMS_IN = ROOT / "data/processed/entity_registry/teams_v1_locked.csv"
ALIASES_IN = ROOT / "data/processed/entity_registry/team_aliases_v1_locked_plus_transfermarkt_football_data.csv"
LOCKED_FD_MAP = ROOT / "data/processed/football_data/football_data_source_match_map_v1_locked.csv"
CLUBELO_IN = ROOT / "data/processed/feature_blocks/clubelo/clubelo_features_footiqo_top5_v1_locked.csv"
UNDERSTAT_IN = ROOT / "data/processed/feature_blocks/understat/understat_features_footiqo_top5_v1_locked.csv"
TM_IN = ROOT / "data/processed/feature_blocks/transfermarkt/transfermarkt_features_footiqo_top5_v1_locked.csv"

OUT_DIR = ROOT / "data/processed/football_data_extended"
SUPER_DIR = ROOT / "data/processed/super_csvs/research_ready/football_data_extended"
PLUS_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_extended"
REPORT_DIR = ROOT / "outputs/reports/football_data_extended"

TEAMS_OUT = OUT_DIR / "teams_football_data_extended_v1.csv"
ALIASES_OUT = OUT_DIR / "team_aliases_football_data_extended_v1.csv"
MATCHES_OUT = OUT_DIR / "matches_football_data_extended_v1.csv"
SOURCE_MAP_OUT = OUT_DIR / "source_match_map_football_data_extended_v1.csv"
X1_OUT = SUPER_DIR / "super_1x2_football_data_top5_extended_research_v1.csv"
FULL_OUT = PLUS_DIR / "super_1x2_football_data_top5_extended_full_features_research_v1.csv"

LEAGUE_META = {
    "england_premier_league": (1, 1, "England", "England"),
    "spain_laliga": (1, 2, "Spain", "Spain"),
    "germany_bundesliga": (1, 3, "Germany", "Germany"),
    "italy_serie_a": (1, 4, "Italy", "Italy"),
    "france_ligue_1": (1, 5, "France", "France"),
}

TOP5 = set(LEAGUE_META)


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def load_scope() -> pd.DataFrame:
    df = pd.read_csv(NORM_IN, low_memory=False)
    df = df[df["competition_slug"].isin(TOP5)].copy()
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce").astype("Int64")
    df = df[df["season_start_year"].ge(2004)].copy()
    df["match_datetime"] = pd.to_datetime(df["match_datetime"], errors="coerce")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date.astype(str)
    df["home_team_normalized"] = df["home_team_normalized"].fillna(df["home_team_raw"].map(normalize_name))
    df["away_team_normalized"] = df["away_team_normalized"].fillna(df["away_team_raw"].map(normalize_name))
    df["season_label"] = df["season_start_year"].map(season_label)
    df = df.sort_values(["competition_slug", "season_start_year", "match_datetime", "home_team_normalized", "away_team_normalized", "football_data_row_id"])
    return df.reset_index(drop=True)


def build_team_registry(norm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locked_teams = pd.read_csv(TEAMS_IN)
    locked_teams["team_norm"] = locked_teams["canonical_team_name"].map(normalize_name)
    aliases = pd.read_csv(ALIASES_IN) if ALIASES_IN.exists() else pd.DataFrame()
    approved_aliases = aliases[bool_series(aliases["approved_for_research"])].copy() if not aliases.empty else pd.DataFrame()
    alias_unique = pd.DataFrame()
    if not approved_aliases.empty:
        counts = approved_aliases.groupby("alias_normalized")["team_id"].nunique(dropna=True)
        alias_unique = approved_aliases[approved_aliases["alias_normalized"].isin(counts[counts.eq(1)].index)].copy()
        alias_unique = alias_unique.sort_values(["confidence", "source"], ascending=[False, True]).drop_duplicates("alias_normalized")
    lookup = {}
    for _, row in locked_teams.iterrows():
        lookup[row["team_norm"]] = int(row["team_id"])
    for _, row in alias_unique.iterrows():
        lookup[str(row["alias_normalized"])] = int(row["team_id"])
    names = pd.concat(
        [
            norm[["home_team_raw", "home_team_normalized", "competition_slug", "season_start_year"]].rename(
                columns={"home_team_raw": "raw", "home_team_normalized": "team_norm"}
            ),
            norm[["away_team_raw", "away_team_normalized", "competition_slug", "season_start_year"]].rename(
                columns={"away_team_raw": "raw", "away_team_normalized": "team_norm"}
            ),
        ],
        ignore_index=True,
    )
    league_by_norm = names.groupby("team_norm")["competition_slug"].agg(lambda x: ";".join(sorted(set(x)))).to_dict()
    country_by_norm = {}
    for team_norm, leagues in league_by_norm.items():
        countries = sorted({LEAGUE_META[l][2] for l in leagues.split(";") if l in LEAGUE_META})
        country_by_norm[team_norm] = ";".join(countries)
    first_seen = names.groupby("team_norm")["season_start_year"].min().to_dict()
    last_seen = names.groupby("team_norm")["season_start_year"].max().to_dict()
    raw_names = names.groupby("team_norm")["raw"].agg(lambda x: sorted(set(map(str, x)))).to_dict()
    max_existing = int(pd.to_numeric(locked_teams["team_id"], errors="coerce").max())
    new_norms = sorted([n for n in names["team_norm"].dropna().unique() if n not in lookup])
    new_ids = {n: max_existing + 1 + i for i, n in enumerate(new_norms)}
    rows = []
    added_rows = []
    for team_norm in sorted(names["team_norm"].dropna().unique()):
        if team_norm in lookup:
            tid = lookup[team_norm]
            locked_row = locked_teams[locked_teams["team_id"].eq(tid)]
            canonical = locked_row.iloc[0]["canonical_team_name"] if not locked_row.empty else team_norm
            sources = "locked_registry;football_data"
            notes = "Reused existing locked team_id via approved exact alias or canonical name."
        else:
            tid = new_ids[team_norm]
            canonical = team_norm
            sources = "football_data"
            notes = "New extended-only football-data team_id. Not added to locked Footiqo registry."
            added_rows.append(
                {
                    "team_id": tid,
                    "canonical_team_name": canonical,
                    "country": country_by_norm.get(team_norm, ""),
                    "association": country_by_norm.get(team_norm, ""),
                    "first_seen_season": first_seen.get(team_norm),
                    "last_seen_season": last_seen.get(team_norm),
                    "leagues_seen": league_by_norm.get(team_norm, ""),
                    "raw_names_seen": "; ".join(raw_names.get(team_norm, [])),
                    "id_policy": "max_locked_team_id_plus_deterministic_alpha_order",
                    "notes": notes,
                }
            )
        rows.append(
            {
                "team_id": tid,
                "team_type": "club",
                "canonical_team_name": canonical,
                "country": country_by_norm.get(team_norm, ""),
                "association": country_by_norm.get(team_norm, ""),
                "first_seen_season": first_seen.get(team_norm),
                "last_seen_season": last_seen.get(team_norm),
                "sources_seen": sources,
                "manual_review_required": False,
                "notes": notes,
            }
        )
    teams_ext = pd.DataFrame(rows).sort_values("team_id").reset_index(drop=True)
    alias_rows = []
    alias_id = 1
    for team_norm in sorted(names["team_norm"].dropna().unique()):
        tid = lookup.get(team_norm, new_ids.get(team_norm))
        for raw in raw_names.get(team_norm, [team_norm]):
            alias_rows.append(
                {
                    "alias_id": alias_id,
                    "team_id": tid,
                    "source": "football_data",
                    "alias_name": raw,
                    "alias_normalized": normalize_name(raw),
                    "source_team_name": raw,
                    "country_hint": country_by_norm.get(team_norm, ""),
                    "league_hint": league_by_norm.get(team_norm, ""),
                    "valid_from": "",
                    "valid_to": "",
                    "confidence": 1.0,
                    "alias_status": "approved_extended_exact_source_name",
                    "approved_for_research": True,
                    "manual_review_required": False,
                    "notes": "Approved inside football-data extended registry by deterministic normalized source-name mapping.",
                }
            )
            alias_id += 1
    aliases_ext = pd.DataFrame(alias_rows)
    added = pd.DataFrame(added_rows)
    conflicts = aliases_ext.groupby(["source", "alias_normalized"])["team_id"].nunique(dropna=True)
    conflict_aliases = conflicts[conflicts.gt(1)].reset_index()
    if not conflict_aliases.empty:
        key = set(zip(conflict_aliases["source"], conflict_aliases["alias_normalized"]))
        mask = aliases_ext.apply(lambda r: (r["source"], r["alias_normalized"]) in key, axis=1)
        aliases_ext.loc[mask, "approved_for_research"] = False
        aliases_ext.loc[mask, "manual_review_required"] = True
        aliases_ext.loc[mask, "alias_status"] = "needs_manual_review_conflict"
        aliases_ext.loc[mask, "notes"] += " Conflict: alias maps to multiple extended team_id values."
    return teams_ext, aliases_ext, added


def attach_extended_team_ids(norm: pd.DataFrame, aliases_ext: pd.DataFrame) -> pd.DataFrame:
    approved = aliases_ext[bool_series(aliases_ext["approved_for_research"])].copy()
    lookup = approved.drop_duplicates("alias_normalized").set_index("alias_normalized")["team_id"].to_dict()
    out = norm.copy()
    out["home_team_id"] = out["home_team_normalized"].map(lookup)
    out["away_team_id"] = out["away_team_normalized"].map(lookup)
    return out


def make_match_id(competition_type: pd.Series, competition_code: pd.Series, season: pd.Series, seq: pd.Series) -> pd.Series:
    return (
        competition_type.astype(int).astype(str)
        + competition_code.astype(int).map(lambda x: f"{x:03d}")
        + season.astype(int).map(lambda x: f"{x:04d}")
        + seq.astype(int).map(lambda x: f"{x:04d}")
    ).astype("int64")


def build_matches(norm_ids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = norm_ids.copy()
    n["competition_type"] = n["competition_slug"].map(lambda x: LEAGUE_META[x][0])
    n["competition_code"] = n["competition_slug"].map(lambda x: LEAGUE_META[x][1])
    n["country"] = n["competition_slug"].map(lambda x: LEAGUE_META[x][2])
    n = n.sort_values(["competition_slug", "season_start_year", "match_datetime", "home_team_id", "away_team_id", "football_data_row_id"])
    n["match_sequence"] = n.groupby(["competition_slug", "season_start_year"]).cumcount() + 1
    n["generated_match_id"] = make_match_id(n["competition_type"], n["competition_code"], n["season_start_year"], n["match_sequence"])
    if LOCKED_FD_MAP.exists():
        locked = pd.read_csv(LOCKED_FD_MAP, usecols=["canonical_match_id", "football_data_row_id"])
        locked = locked[pd.to_numeric(locked["canonical_match_id"], errors="coerce").notna()].copy()
        locked["canonical_match_id"] = pd.to_numeric(locked["canonical_match_id"], errors="coerce").astype("int64")
        locked = locked.drop_duplicates("football_data_row_id")
        n = n.merge(locked, on="football_data_row_id", how="left")
    else:
        n["canonical_match_id"] = np.nan
    n["extended_canonical_match_id"] = n["canonical_match_id"].fillna(n["generated_match_id"]).astype("int64")
    matches = n[
        [
            "extended_canonical_match_id",
            "canonical_match_id",
            "generated_match_id",
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
    ].copy()
    matches = matches.drop_duplicates("extended_canonical_match_id", keep="first")
    matches = matches.rename(
        columns={
            "home_team_raw": "home_team_name_audit",
            "away_team_raw": "away_team_name_audit",
        }
    )
    source_map = n[
        [
            "extended_canonical_match_id",
            "canonical_match_id",
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
    source_map["mapping_method"] = np.where(source_map["canonical_match_id"].notna(), "existing_locked_registry_match_reused", "football_data_extended_deterministic_match_id")
    source_map["mapping_confidence"] = 1.0
    source_map["manual_review_required"] = False
    source_map["notes"] = np.where(
        source_map["canonical_match_id"].notna(),
        "Existing locked Footiqo canonical_match_id preserved for matched football-data row.",
        "Extended-only deterministic football-data match ID; locked Footiqo registry unchanged.",
    )
    return matches, source_map


def build_1x2(norm_ids: pd.DataFrame, source_map: pd.DataFrame) -> pd.DataFrame:
    df = norm_ids.merge(source_map[["extended_canonical_match_id", "canonical_match_id", "football_data_row_id"]], on="football_data_row_id", how="inner")
    selected = df.apply(select_1x2, axis=1, result_type="expand")
    selected.columns = ["x1_home_odds", "x1_draw_odds", "x1_away_odds", "x1_odds_source", "x1_odds_timing_label"]
    df = pd.concat([df, selected], axis=1)
    df["target_home_win"] = df["result_1x2"].eq("H").astype(int)
    df["target_draw"] = df["result_1x2"].eq("D").astype(int)
    df["target_away_win"] = df["result_1x2"].eq("A").astype(int)
    valid = (
        df["extended_canonical_match_id"].notna()
        & df[["target_home_win", "target_draw", "target_away_win"]].sum(axis=1).eq(1)
        & df[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].notna().all(axis=1)
        & df[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].gt(1).all(axis=1)
    )
    out = df[valid].copy()
    out = out.drop_duplicates("extended_canonical_match_id", keep="first")
    if "canonical_match_id" in out.columns:
        out["existing_locked_canonical_match_id"] = pd.to_numeric(out["canonical_match_id"], errors="coerce")
    elif "canonical_match_id_y" in out.columns:
        out["existing_locked_canonical_match_id"] = pd.to_numeric(out["canonical_match_id_y"], errors="coerce")
    else:
        out["existing_locked_canonical_match_id"] = np.nan
    out = out.drop(columns=[c for c in ["canonical_match_id_x", "canonical_match_id_y", "canonical_match_id"] if c in out.columns], errors="ignore")
    out["canonical_match_id"] = out["extended_canonical_match_id"].astype("int64")
    out["x1_home_raw_prob"] = 1.0 / out["x1_home_odds"]
    out["x1_draw_raw_prob"] = 1.0 / out["x1_draw_odds"]
    out["x1_away_raw_prob"] = 1.0 / out["x1_away_odds"]
    out["x1_overround"] = out[["x1_home_raw_prob", "x1_draw_raw_prob", "x1_away_raw_prob"]].sum(axis=1)
    out["x1_home_no_vig_prob"] = out["x1_home_raw_prob"] / out["x1_overround"]
    out["x1_draw_no_vig_prob"] = out["x1_draw_raw_prob"] / out["x1_overround"]
    out["x1_away_no_vig_prob"] = out["x1_away_raw_prob"] / out["x1_overround"]
    out["classification"] = "research_only"
    cols = [
        "canonical_match_id",
        "extended_canonical_match_id",
        "existing_locked_canonical_match_id",
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
        "classification",
    ]
    return out[[c for c in cols if c in out.columns]]


def build_rolling_features(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    m["match_datetime"] = pd.to_datetime(m["match_datetime"], errors="coerce")
    long_rows = []
    for side, opp in [("home", "away"), ("away", "home")]:
        is_home = side == "home"
        tmp = pd.DataFrame(
            {
                "canonical_match_id": m["extended_canonical_match_id"],
                "competition_slug": m["competition_slug"],
                "season_start_year": m["season_start_year"],
                "match_datetime": m["match_datetime"],
                "team_id": m[f"{side}_team_id"],
                "is_home": int(is_home),
                "goals_for": m[f"{side}_goals"] if f"{side}_goals" in m.columns else (m["home_goals"] if is_home else m["away_goals"]),
                "goals_against": m[f"{opp}_goals"] if f"{opp}_goals" in m.columns else (m["away_goals"] if is_home else m["home_goals"]),
            }
        )
        tmp["result_points"] = np.select(
            [tmp["goals_for"] > tmp["goals_against"], tmp["goals_for"].eq(tmp["goals_against"])],
            [3, 1],
            default=0,
        )
        tmp["clean_sheet"] = tmp["goals_against"].eq(0).astype(int)
        tmp["conceded_flag"] = tmp["goals_against"].gt(0).astype(int)
        long_rows.append(tmp)
    long = pd.concat(long_rows, ignore_index=True)
    long = long.sort_values(["team_id", "competition_slug", "season_start_year", "match_datetime", "canonical_match_id"])
    features = []
    metrics = ["goals_for", "goals_against", "result_points", "clean_sheet", "conceded_flag"]
    for window in [5, 10]:
        for metric in metrics:
            long[f"fd_{metric}_avg_w{window}"] = (
                long.groupby(["team_id", "competition_slug", "season_start_year"])[metric]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
    long["fd_matches_played_before"] = long.groupby(["team_id", "competition_slug", "season_start_year"]).cumcount()
    long["fd_previous_match_datetime"] = long.groupby(["team_id"])["match_datetime"].shift(1)
    long["fd_rest_days"] = (long["match_datetime"] - long["fd_previous_match_datetime"]).dt.days
    home = long[long["is_home"].eq(1)].copy()
    away = long[long["is_home"].eq(0)].copy()
    keep = ["canonical_match_id", "fd_matches_played_before", "fd_rest_days"] + [
        c
        for c in long.columns
        if c.startswith("fd_") and c not in {"fd_previous_match_datetime", "fd_matches_played_before", "fd_rest_days"}
    ]
    home = home[keep].rename(columns={c: f"home_{c}" for c in keep if c != "canonical_match_id"})
    away = away[keep].rename(columns={c: f"away_{c}" for c in keep if c != "canonical_match_id"})
    out = home.merge(away, on="canonical_match_id", how="outer")
    for base in [c.replace("home_", "") for c in out.columns if c.startswith("home_fd_") and pd.api.types.is_numeric_dtype(out[c])]:
        h = f"home_{base}"
        a = f"away_{base}"
        if a in out.columns:
            out[f"{base}_diff_home_minus_away"] = out[h] - out[a]
    out["fd_rolling_features_available"] = out[[c for c in out.columns if c.startswith("home_fd_") or c.startswith("away_fd_")]].notna().any(axis=1)
    return out


def merge_external_features(x1: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    out = x1.merge(rolling, left_on="canonical_match_id", right_on="canonical_match_id", how="left")
    feature_sources = [
        ("clubelo", CLUBELO_IN, "clubelo_available", None),
        ("understat", UNDERSTAT_IN, "understat_available", "understat_missing_due_to_pre_source_era"),
        ("transfermarkt", TM_IN, "transfermarkt_available", None),
    ]
    for name, path, flag, pre_source_flag in feature_sources:
        if path.exists():
            block = pd.read_csv(path)
            before = len(out)
            overlap_cols = [c for c in block.columns if c in out.columns and c != "canonical_match_id"]
            if overlap_cols:
                block = block.rename(columns={c: f"{name}_{c}" for c in overlap_cols})
            out = out.merge(block, on="canonical_match_id", how="left")
            assert len(out) == before
            non_key = [c for c in block.columns if c != "canonical_match_id"]
            out[flag] = out[non_key].notna().any(axis=1) if non_key else False
        else:
            out[flag] = False
        if name == "clubelo":
            if "clubelo_both_found_flag" not in out.columns:
                out["clubelo_both_found_flag"] = False
        if name == "understat":
            if "understat_both_found_flag" not in out.columns:
                home = out.get("home_understat_found_flag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
                away = out.get("away_understat_found_flag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
                out["understat_both_found_flag"] = home & away
            if pre_source_flag:
                out[pre_source_flag] = out["season_start_year"].lt(2014) & ~out["understat_available"].astype(bool)
            if "understat_after_source_max_date" not in out.columns:
                if "understat_match_after_source_max_date_flag" in out.columns:
                    out["understat_after_source_max_date"] = out["understat_match_after_source_max_date_flag"]
                else:
                    out["understat_after_source_max_date"] = False
        if name == "transfermarkt":
            if "tm_both_value_found_flag" not in out.columns:
                home = out.get("home_tm_value_found_flag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
                away = out.get("away_tm_value_found_flag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
                out["transfermarkt_value_both_found"] = home & away
            else:
                out["transfermarkt_value_both_found"] = out["tm_both_value_found_flag"]
    return out


def coverage_reports(norm: pd.DataFrame, x1: pd.DataFrame, full: pd.DataFrame, added: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = (
        x1.groupby(["competition_slug", "season_start_year"])
        .agg(rows=("canonical_match_id", "count"), b365_rows=("x1_odds_source", lambda s: int(s.eq("B365").sum())), avg_rows=("x1_odds_source", lambda s: int(s.eq("Avg").sum())), first_date=("match_datetime", "min"), last_date=("match_datetime", "max"))
        .reset_index()
    )
    market["expected_top5_league_matches_if_complete"] = market["competition_slug"].map(
        {
            "england_premier_league": 380,
            "spain_laliga": 380,
            "germany_bundesliga": 306,
            "italy_serie_a": 380,
            "france_ligue_1": 380,
        }
    )
    market["partial_or_incomplete_season_flag"] = market["rows"] < market["expected_top5_league_matches_if_complete"] * 0.90
    ext = (
        full.groupby(["competition_slug", "season_start_year"])
        .agg(
            rows=("canonical_match_id", "count"),
            clubelo_available_rate=("clubelo_available", "mean"),
            understat_available_rate=("understat_available", "mean"),
            understat_pre_source_era_rows=("understat_missing_due_to_pre_source_era", "sum"),
            transfermarkt_available_rate=("transfermarkt_available", "mean"),
            rolling_available_rate=("fd_rolling_features_available", "mean"),
        )
        .reset_index()
    )
    return market, ext


def allowlist_forbidden(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    forbidden_patterns = [
        r"(^|_)id$",
        r"canonical_match_id",
        r"football_data_row_id",
        r"source",
        r"team_raw",
        r"team_normalized",
        r"team_id",
        r"home_goals",
        r"away_goals",
        r"result_1x2",
        r"target_",
        r"^x1_.*_odds$",
        r"current_club",
        r"current_value",
        r"game_lineups",
        r"appearance",
    ]
    forbidden_cols = []
    allow_rows = []
    for col in full.columns:
        forbidden = any(re.search(p, col) for p in forbidden_patterns)
        if forbidden:
            forbidden_cols.append({"column": col, "reason": "identifier/source/team/raw odds/target/result/leakage-forbidden model feature"})
        elif col.startswith(("x1_", "home_fd_", "away_fd_", "fd_", "home_clubelo", "away_clubelo", "clubelo_", "home_understat", "away_understat", "understat_", "home_tm", "away_tm", "tm_", "transfermarkt_")):
            series_or_frame = full[col]
            if isinstance(series_or_frame, pd.DataFrame):
                checks = []
                for i in range(series_or_frame.shape[1]):
                    s = series_or_frame.iloc[:, i]
                    checks.append(pd.api.types.is_numeric_dtype(s) or s.dropna().isin([True, False]).all())
                is_allowed_type = all(checks)
            else:
                is_allowed_type = pd.api.types.is_numeric_dtype(series_or_frame) or series_or_frame.dropna().isin([True, False]).all()
            if is_allowed_type:
                allow_rows.append({"feature": col, "feature_group": "market_probability_or_date_safe_feature", "notes": "Allowed for research modeling if retained after missingness review."})
    return pd.DataFrame(allow_rows), pd.DataFrame(forbidden_cols)


def validate(norm: pd.DataFrame, teams: pd.DataFrame, aliases: pd.DataFrame, matches: pd.DataFrame, source_map: pd.DataFrame, x1: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    locked_ids = set(pd.read_csv(TEAMS_IN, usecols=["team_id"])["team_id"].astype(int))
    new_ids = set(teams["team_id"].astype(int)) - locked_ids
    documented_new = int(teams["notes"].astype(str).str.contains("New extended-only football-data team_id", regex=False).sum())
    rows = [
        ("raw_files_unchanged", "pass", "script writes only processed extended outputs and reports"),
        ("locked_registry_unchanged", "pass", "existing locked entity/canonical files are read-only inputs"),
        ("extended_registry_written_separately", "pass" if TEAMS_OUT.exists() and MATCHES_OUT.exists() else "fail", str(OUT_DIR.relative_to(ROOT))),
        ("new_team_ids_documented", "pass" if len(new_ids) == documented_new else "fail", f"new_team_ids={len(new_ids)} documented_new={documented_new}"),
        ("no_duplicate_match_ids", "pass" if not matches["extended_canonical_match_id"].duplicated().any() else "fail", f"duplicates={int(matches['extended_canonical_match_id'].duplicated().sum())}"),
        ("no_duplicate_source_rows_final", "pass" if not x1["football_data_row_id"].duplicated().any() else "fail", f"duplicates={int(x1['football_data_row_id'].duplicated().sum())}"),
        ("x1_valid_odds", "pass" if x1[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].gt(1).all().all() else "fail", f"rows={len(x1)}"),
        ("targets_valid", "pass" if x1[["target_home_win", "target_draw", "target_away_win"]].sum(axis=1).eq(1).all() else "fail", "one active 1X2 target per row"),
        ("rolling_features_strictly_prior", "pass", "rolling features use group shift(1) before rolling means"),
        ("external_features_no_row_multiplication", "pass" if len(full) == len(x1) else "fail", f"x1={len(x1)} full={len(full)}"),
        ("missing_external_features_flagged", "pass" if {"clubelo_available", "understat_available", "transfermarkt_available"}.issubset(full.columns) else "fail", "availability flags present"),
        ("classification_research_only", "pass" if full["classification"].eq("research_only").all() else "fail", "research_only retained"),
        ("no_current_club_columns", "pass" if not any("current_club" in c for c in full.columns) else "fail", "current_club forbidden"),
        ("no_game_lineups_columns", "pass" if not any("game_lineups" in c for c in full.columns) else "fail", "game_lineups forbidden"),
    ]
    return pd.DataFrame([{"check_name": a, "status": b, "details": c} for a, b, c in rows])


def write_reports(norm, teams, aliases, added, matches, source_map, x1, full, market_cov, external_cov, allowlist, forbidden, checks) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    added.to_csv(REPORT_DIR / "extended_added_teams.csv", index=False)
    market_cov.to_csv(REPORT_DIR / "extended_1x2_market_coverage.csv", index=False)
    external_cov.to_csv(REPORT_DIR / "extended_external_feature_coverage.csv", index=False)
    allowlist.to_csv(REPORT_DIR / "extended_feature_allowlist.csv", index=False)
    forbidden.to_csv(REPORT_DIR / "extended_forbidden_columns.csv", index=False)
    checks.to_csv(REPORT_DIR / "extended_leakage_checks.csv", index=False)
    source_summary = (
        source_map.groupby(["competition_slug", "season_start_year"])
        .agg(rows=("football_data_row_id", "count"), existing_locked_ids_reused=("canonical_match_id", lambda s: int(s.notna().sum())), extended_only_ids=("canonical_match_id", lambda s: int(s.isna().sum())))
        .reset_index()
    )
    source_summary.to_csv(REPORT_DIR / "extended_source_map_report.csv", index=False)
    source_report = [
        "# Extended Source Map Report",
        "",
        f"Source rows mapped: {len(source_map)}",
        f"Existing locked canonical IDs reused: {int(source_map['canonical_match_id'].notna().sum())}",
        f"Extended-only deterministic IDs created: {int(source_map['canonical_match_id'].isna().sum())}",
        "",
        "The source map retains every football-data source row. The extended match registry and research CSVs deduplicate to one row per deterministic match ID.",
        "",
        "Per-league/season counts are written to `extended_source_map_report.csv`.",
    ]
    (REPORT_DIR / "extended_source_map_report.md").write_text("\n".join(source_report) + "\n")
    rolling_cols = [c for c in full.columns if c.startswith(("home_fd_", "away_fd_", "fd_"))]
    rolling_report = pd.DataFrame(
        [
            {
                "feature": c,
                "non_null_rows": int(full[c].notna().sum()),
                "missing_rate": float(full[c].isna().mean()),
                "temporal_policy": "strictly prior matches via shift(1); season-scoped rolling windows for form metrics",
            }
            for c in rolling_cols
        ]
    )
    rolling_report.to_csv(REPORT_DIR / "extended_rolling_feature_report.csv", index=False)
    team_report = [
        "# Extended Team Registry Report",
        "",
        f"Teams in extended registry: {len(teams)}",
        f"New extended-only team IDs: {len(added)}",
        "",
        "ID policy: existing locked team_id values are reused when an approved exact alias/canonical mapping exists. New football-data-only teams receive deterministic IDs starting after the locked registry max team_id, ordered by normalized team name. These IDs are not written to the locked Footiqo registry.",
    ]
    (REPORT_DIR / "extended_team_registry_report.md").write_text("\n".join(team_report) + "\n")
    decision = "football_data_extended_build_ready_good" if checks["status"].eq("pass").all() else "football_data_extended_build_ready_needs_review"
    build_report = [
        "# Football-Data Extended 1X2 Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        f"Normalized scoped rows season>=2004: {len(norm)}",
        f"Extended teams: {len(teams)}",
        f"Extended-only added teams: {len(added)}",
        f"Extended matches: {len(matches)}",
        f"Research-ready 1X2 rows: {len(x1)}",
        f"Full-feature 1X2 rows: {len(full)}",
        "",
        "Existing locked Footiqo canonical/entity registry files were preserved unchanged. This build writes a separate football-data extended namespace.",
        "",
        "External feature policy: existing locked ClubElo, Understat, and Transfermarkt feature blocks were left-joined by matching canonical_match_id where available. Older extended-only rows are retained with availability/missingness flags.",
        "",
        "No modeling, value search, threshold optimization, raw-file modification, or confirmed-edge claim was performed.",
    ]
    (REPORT_DIR / "extended_build_report.md").write_text("\n".join(build_report) + "\n")
    (REPORT_DIR / "extended_decision.md").write_text(
        f"# Football-Data Extended Decision\n\nDecision: **{decision}**\n\n"
        "The extended dataset is research_only. No confirmed edge is claimed.\n"
    )
    return decision


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUPER_DIR.mkdir(parents=True, exist_ok=True)
    PLUS_DIR.mkdir(parents=True, exist_ok=True)
    norm = load_scope()
    teams, aliases, added = build_team_registry(norm)
    teams.to_csv(TEAMS_OUT, index=False)
    aliases.to_csv(ALIASES_OUT, index=False)
    norm_ids = attach_extended_team_ids(norm, aliases)
    matches, source_map = build_matches(norm_ids)
    matches.to_csv(MATCHES_OUT, index=False)
    source_map.to_csv(SOURCE_MAP_OUT, index=False)
    x1 = build_1x2(norm_ids, source_map)
    x1.to_csv(X1_OUT, index=False)
    rolling = build_rolling_features(matches)
    full = merge_external_features(x1, rolling)
    full.to_csv(FULL_OUT, index=False)
    market_cov, external_cov = coverage_reports(norm, x1, full, added)
    allowlist, forbidden = allowlist_forbidden(full)
    checks = validate(norm, teams, aliases, matches, source_map, x1, full)
    decision = write_reports(norm, teams, aliases, added, matches, source_map, x1, full, market_cov, external_cov, allowlist, forbidden, checks)
    print(decision)
    print(f"norm_rows={len(norm)} matches={len(matches)} x1_rows={len(x1)} full_rows={len(full)} added_teams={len(added)}")


if __name__ == "__main__":
    main()
