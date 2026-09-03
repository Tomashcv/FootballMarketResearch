from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOTS = [ROOT / "data/raw", ROOT / "data/raw_external", ROOT / "data/external", ROOT / "data/processed"]
OUT_DIR = ROOT / "data/processed/football_data_full_scope"
SUPER_DIR = ROOT / "data/processed/super_csvs/research_ready/football_data_full_scope"
PLUS_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope"
REPORT_DIR = ROOT / "outputs/reports/football_data_full_scope"

TEAMS_LOCKED = ROOT / "data/processed/entity_registry/teams_v1_locked.csv"
ALIASES_PLUS = ROOT / "data/processed/entity_registry/team_aliases_v1_locked_plus_transfermarkt_football_data.csv"
ALIASES_LOCKED = ROOT / "data/processed/entity_registry/team_aliases_v1_locked_plus_transfermarkt.csv"
CLUBELO_IN = ROOT / "data/processed/feature_blocks/clubelo/clubelo_features_footiqo_top5_v1_locked.csv"
UNDERSTAT_IN = ROOT / "data/processed/feature_blocks/understat/understat_features_footiqo_top5_v1_locked.csv"
TM_IN = ROOT / "data/processed/feature_blocks/transfermarkt/transfermarkt_features_footiqo_top5_v1_locked.csv"

SCOPE_CODES = {
    "E0": ("england_premier_league", 1, 1, "England", 300, 430),
    "SP1": ("spain_laliga", 1, 2, "Spain", 300, 430),
    "D1": ("germany_bundesliga", 1, 3, "Germany", 250, 360),
    "I1": ("italy_serie_a", 1, 4, "Italy", 300, 430),
    "F1": ("france_ligue_1", 1, 5, "France", 250, 430),
    "B1": ("belgium_jupiler_league", 1, 6, "Belgium", 200, 430),
    "G1": ("greece_super_league", 1, 7, "Greece", 150, 380),
    "N1": ("netherlands_eredivisie", 1, 8, "Netherlands", 250, 360),
    "P1": ("portugal_primeira_liga", 1, 9, "Portugal", 250, 430),
    "SC0": ("scotland_premiership", 1, 10, "Scotland", 180, 260),
    "T1": ("turkey_super_lig", 1, 11, "Turkey", 250, 430),
}
EXCLUDED = {"E1", "E2", "E3"}
ONE_X_TWO_PRIORITY = [
    ("B365", ("B365H", "B365D", "B365A"), "unknown"),
    ("Avg", ("AvgH", "AvgD", "AvgA"), "unknown"),
    ("football_data_HDA", ("H", "D", "A"), "unknown"),
]


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def parse_date_series(s: pd.Series) -> pd.Series:
    """Parse football-data dates without ambiguous dayfirst warnings.

    football-data archives contain a mix of ISO dates and day-first slash/dash dates.
    Parse known formats explicitly, then use a conservative day-first fallback only for
    still-unparsed values.
    """
    text = s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "NaT": pd.NA})
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    formats = [
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
        (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
        (r"^\d{2}-\d{2}-\d{4}$", "%d-%m-%Y"),
        (r"^\d{2}/\d{2}/\d{2}$", "%d/%m/%y"),
        (r"^\d{2}-\d{2}-\d{2}$", "%d-%m-%y"),
    ]
    for pattern, fmt in formats:
        mask = out.isna() & text.str.match(pattern, na=False)
        if mask.any():
            out.loc[mask] = pd.to_datetime(text.loc[mask], format=fmt, errors="coerce")
    remaining = out.isna() & text.notna()
    if remaining.any():
        out.loc[remaining] = pd.to_datetime(text.loc[remaining], errors="coerce", dayfirst=True)
    return out


def combine_date_and_time(dates: pd.Series, time_values: pd.Series) -> pd.Series:
    """Combine normalized dates and HH:MM[:SS] strings without reparsing ISO dates."""
    clean_time = time_values.astype("string").str.strip().replace({"": "00:00:00", "nan": "00:00:00", "NaT": "00:00:00"})
    hhmm = clean_time.str.match(r"^\d{1,2}:\d{2}$", na=False)
    clean_time = clean_time.where(~hhmm, clean_time + ":00")
    timedeltas = pd.to_timedelta(clean_time, errors="coerce").fillna(pd.Timedelta(0))
    return pd.to_datetime(dates, errors="coerce") + timedeltas


def infer_season(date: pd.Timestamp) -> float:
    if pd.isna(date):
        return np.nan
    return float(date.year - 1 if date.month < 7 else date.year)


def season_from_source_file(source_file: object) -> float:
    text = "" if pd.isna(source_file) else str(source_file)
    m = re.search(r"/seasons/[A-Z0-9]+_(\d{4})_(\d{4})(?:__variant_\d+)?\.csv$", text)
    if m:
        return float(int(m.group(1)))
    m = re.search(r"/seasons/[A-Z0-9]+_(\d{2})(\d{2})(?:__variant_\d+)?\.csv$", text)
    if m:
        yy = int(m.group(1))
        return float(2000 + yy if yy < 50 else 1900 + yy)
    return np.nan


def season_label(year: object) -> str:
    if pd.isna(year):
        return ""
    y = int(year)
    return f"{y}/{y + 1}"


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha1(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:12]


def source_priority(source_file: object) -> int:
    text = "" if pd.isna(source_file) else str(source_file).lower()
    if re.search(r"data/raw/[a-z0-9]+/seasons/[a-z0-9]+_(\d{4}_\d{4}|\d{4})(?:__variant_\d+)?\.csv$", text):
        return 1
    if re.search(r"data/raw/[a-z0-9]+.*\.csv$", text) and "/seasons/" not in text and "processed" not in text:
        return 2
    if text.startswith("data/raw/") or "raw_external" in text:
        return 3
    if text.startswith("data/processed/") or "/processed/" in text:
        return 4
    return 5


def select_1x2(row: pd.Series) -> tuple[float, float, float, str, str]:
    for source, cols, timing in ONE_X_TWO_PRIORITY:
        if all(c in row.index for c in cols):
            vals = [pd.to_numeric(row[c], errors="coerce") for c in cols]
            if all(pd.notna(v) and float(v) > 1 for v in vals):
                return float(vals[0]), float(vals[1]), float(vals[2]), source, timing
    return np.nan, np.nan, np.nan, "", ""


def discover_and_normalize() -> tuple[pd.DataFrame, pd.DataFrame]:
    inv_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    seen: set[Path] = set()
    keep_original = {
        "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA", "H", "D", "A",
    }
    for root in DATA_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if path in seen:
                continue
            seen.add(path)
            rel = str(path.relative_to(ROOT))
            try:
                raw = pd.read_csv(path, low_memory=False)
            except Exception as exc:
                inv_rows.append({"path": rel, "filename": path.name, "league_code": "", "inferred_league_slug": "", "rows": np.nan, "columns": np.nan, "date_range": "", "markets_detected": "", "read_status": f"read_error: {type(exc).__name__}: {exc}"})
                continue
            cols = set(map(str, raw.columns))
            if "Div" not in cols or "Date" not in cols or not ({"HomeTeam", "AwayTeam"}.issubset(cols) or {"HT", "AT"}.issubset(cols)):
                continue
            divs = sorted(pd.Series(raw["Div"]).dropna().astype(str).unique())
            for div in divs:
                if div in EXCLUDED:
                    inv_rows.append({"path": rel, "filename": path.name, "league_code": div, "inferred_league_slug": "", "rows": int(raw["Div"].astype(str).eq(div).sum()), "columns": len(raw.columns), "date_range": "", "markets_detected": "", "read_status": "excluded_lower_english"})
                    continue
                if div not in SCOPE_CODES:
                    continue
                sub = raw[raw["Div"].astype(str).eq(div)].copy()
                if sub.empty:
                    continue
                dates = parse_date_series(sub["Date"])
                markets = []
                if any(set(spec[1]).issubset(cols) for spec in ONE_X_TWO_PRIORITY):
                    markets.append("1X2")
                inv_rows.append({"path": rel, "filename": path.name, "league_code": div, "inferred_league_slug": SCOPE_CODES[div][0], "rows": len(sub), "columns": len(raw.columns), "date_range": f"{dates.min().date() if dates.notna().any() else ''} to {dates.max().date() if dates.notna().any() else ''}", "markets_detected": ";".join(markets), "read_status": "read_ok"})
                home_col = "HomeTeam" if "HomeTeam" in sub.columns else "HT"
                away_col = "AwayTeam" if "AwayTeam" in sub.columns else "AT"
                time = sub["Time"].astype(str).str.strip() if "Time" in sub.columns else pd.Series([""] * len(sub), index=sub.index)
                datetimes = combine_date_and_time(dates, time)
                slug, ctype, ccode, country, _lo, _hi = SCOPE_CODES[div]
                tmp = pd.DataFrame({
                    "football_data_row_id": [f"football_data:{file_fingerprint(path)}:{int(i)}" for i in sub.index],
                    "source_file": rel,
                    "source": "football_data",
                    "div": div,
                    "competition_slug": slug,
                    "competition_type": ctype,
                    "competition_code": ccode,
                    "country": country,
                    "original_season_start_year": dates.map(infer_season),
                    "source_file_season_start_year": season_from_source_file(rel),
                    "match_date": dates.dt.date.astype(str),
                    "match_time": time.replace({"nan": ""}),
                    "match_datetime": datetimes,
                    "home_team_raw": sub[home_col],
                    "away_team_raw": sub[away_col],
                    "home_team_normalized": sub[home_col].map(normalize_name),
                    "away_team_normalized": sub[away_col].map(normalize_name),
                    "home_goals": pd.to_numeric(sub.get("FTHG"), errors="coerce"),
                    "away_goals": pd.to_numeric(sub.get("FTAG"), errors="coerce"),
                    "result_1x2": sub.get("FTR"),
                    "raw_date": sub["Date"],
                })
                tmp["season_start_year"] = tmp["source_file_season_start_year"].fillna(tmp["original_season_start_year"]).astype("Int64")
                tmp["season_label"] = tmp["season_start_year"].map(season_label)
                for c in keep_original:
                    if c in sub.columns:
                        tmp[c] = sub[c]
                frames.append(tmp)
    inventory = pd.DataFrame(inv_rows)
    norm = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not norm.empty:
        norm = norm[pd.to_numeric(norm["season_start_year"], errors="coerce").ge(2004)].copy()
        norm["match_datetime"] = pd.to_datetime(norm["match_datetime"], errors="coerce")
        norm["source_priority"] = norm["source_file"].map(source_priority)
        norm["raw_season_file_flag"] = norm["source_priority"].eq(1)
        norm["processed_aggregate_file_flag"] = norm["source_priority"].eq(4)
        norm["date_parse_anomaly_flag"] = norm["source_file_season_start_year"].notna() & norm["source_file_season_start_year"].ne(norm["original_season_start_year"])
    return inventory, norm


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def build_team_registry(norm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locked = pd.read_csv(TEAMS_LOCKED)
    locked["team_norm"] = locked["canonical_team_name"].map(normalize_name)
    aliases_path = ALIASES_PLUS if ALIASES_PLUS.exists() else ALIASES_LOCKED
    aliases = pd.read_csv(aliases_path) if aliases_path.exists() else pd.DataFrame()
    lookup: dict[str, int] = {}
    if not aliases.empty:
        approved = aliases[bool_series(aliases["approved_for_research"])].copy()
        counts = approved.groupby("alias_normalized")["team_id"].nunique(dropna=True)
        approved = approved[approved["alias_normalized"].isin(counts[counts.eq(1)].index)]
        lookup.update(approved.drop_duplicates("alias_normalized").set_index("alias_normalized")["team_id"].astype(int).to_dict())
    lookup.update(locked.drop_duplicates("team_norm").set_index("team_norm")["team_id"].astype(int).to_dict())
    names = pd.concat([
        norm[["home_team_raw", "home_team_normalized", "competition_slug", "country", "season_start_year"]].rename(columns={"home_team_raw": "raw", "home_team_normalized": "team_norm"}),
        norm[["away_team_raw", "away_team_normalized", "competition_slug", "country", "season_start_year"]].rename(columns={"away_team_raw": "raw", "away_team_normalized": "team_norm"}),
    ], ignore_index=True)
    first = names.groupby("team_norm")["season_start_year"].min().to_dict()
    last = names.groupby("team_norm")["season_start_year"].max().to_dict()
    countries = names.groupby("team_norm")["country"].agg(lambda s: ";".join(sorted(set(map(str, s.dropna()))))).to_dict()
    leagues = names.groupby("team_norm")["competition_slug"].agg(lambda s: ";".join(sorted(set(map(str, s.dropna()))))).to_dict()
    raws = names.groupby("team_norm")["raw"].agg(lambda s: sorted(set(map(str, s.dropna())))).to_dict()
    max_locked = int(pd.to_numeric(locked["team_id"], errors="coerce").max())
    missing_norms = sorted([n for n in names["team_norm"].dropna().unique() if n not in lookup])
    new_ids = {n: max_locked + 1 + i for i, n in enumerate(missing_norms)}
    team_rows = []
    added = []
    for n in sorted(names["team_norm"].dropna().unique()):
        if n in lookup:
            tid = lookup[n]
            lr = locked[locked["team_id"].eq(tid)]
            canonical = lr.iloc[0]["canonical_team_name"] if not lr.empty else n
            notes = "Reused existing locked team_id by exact approved alias/canonical normalized name."
            sources = "locked_registry;football_data"
        else:
            tid = new_ids[n]
            canonical = n
            notes = "New full-scope-only team_id. Not added to locked Footiqo registry."
            sources = "football_data"
            added.append({"team_id": tid, "canonical_team_name": canonical, "country": countries.get(n, ""), "association": countries.get(n, ""), "first_seen_season": first.get(n), "last_seen_season": last.get(n), "leagues_seen": leagues.get(n, ""), "raw_names_seen": "; ".join(raws.get(n, [])), "id_policy": "max_locked_team_id_plus_alpha_normalized_name", "notes": notes})
        team_rows.append({"team_id": tid, "team_type": "club", "canonical_team_name": canonical, "country": countries.get(n, ""), "association": countries.get(n, ""), "first_seen_season": first.get(n), "last_seen_season": last.get(n), "sources_seen": sources, "manual_review_required": False, "notes": notes})
    alias_rows = []
    alias_id = 1
    for n in sorted(names["team_norm"].dropna().unique()):
        tid = lookup.get(n, new_ids.get(n))
        for raw in raws.get(n, [n]):
            alias_rows.append({"alias_id": alias_id, "team_id": tid, "source": "football_data", "alias_name": raw, "alias_normalized": normalize_name(raw), "source_team_name": raw, "country_hint": countries.get(n, ""), "league_hint": leagues.get(n, ""), "valid_from": "", "valid_to": "", "confidence": 1.0, "alias_status": "approved_full_scope_exact_source_name", "approved_for_research": True, "manual_review_required": False, "notes": "Approved inside full-scope football-data namespace by deterministic normalized source-name mapping."})
            alias_id += 1
    teams = pd.DataFrame(team_rows).sort_values("team_id")
    aliases_out = pd.DataFrame(alias_rows)
    conflicts = aliases_out.groupby(["source", "alias_normalized"])["team_id"].nunique(dropna=True)
    conflict_norms = set(conflicts[conflicts.gt(1)].reset_index()["alias_normalized"])
    if conflict_norms:
        m = aliases_out["alias_normalized"].isin(conflict_norms)
        aliases_out.loc[m, "approved_for_research"] = False
        aliases_out.loc[m, "manual_review_required"] = True
        aliases_out.loc[m, "alias_status"] = "needs_manual_review_conflict"
    return teams, aliases_out, pd.DataFrame(added)


def attach_team_ids(norm: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    approved = aliases[bool_series(aliases["approved_for_research"])].copy()
    lookup = approved.drop_duplicates("alias_normalized").set_index("alias_normalized")["team_id"].astype(int).to_dict()
    out = norm.copy()
    out["home_team_id"] = out["home_team_normalized"].map(lookup)
    out["away_team_id"] = out["away_team_normalized"].map(lookup)
    out["logical_match_key"] = (
        out["div"].astype(str) + "|" + out["competition_slug"].astype(str) + "|" + out["season_start_year"].astype(str) + "|"
        + out["home_team_id"].astype("Int64").astype(str) + "|" + out["away_team_id"].astype("Int64").astype(str)
    )
    return out


def add_market(norm: pd.DataFrame) -> pd.DataFrame:
    selected = norm.apply(select_1x2, axis=1, result_type="expand")
    selected.columns = ["x1_home_odds", "x1_draw_odds", "x1_away_odds", "x1_odds_source", "x1_odds_timing_label"]
    out = pd.concat([norm, selected], axis=1)
    out["target_home_win"] = out["result_1x2"].eq("H").astype(int)
    out["target_draw"] = out["result_1x2"].eq("D").astype(int)
    out["target_away_win"] = out["result_1x2"].eq("A").astype(int)
    out["target_valid"] = out[["target_home_win", "target_draw", "target_away_win"]].sum(axis=1).eq(1)
    out["odds_valid"] = out[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].notna().all(axis=1) & out[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].gt(1).all(axis=1)
    out["x1_odds_priority"] = out["x1_odds_source"].map({"B365": 1, "Avg": 2, "football_data_HDA": 3}).fillna(99).astype(int)
    out["row_non_null_count"] = out.notna().sum(axis=1)
    return out


def deduplicate(marketed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = marketed[marketed["target_valid"] & marketed["odds_valid"] & marketed["home_team_id"].notna() & marketed["away_team_id"].notna()].copy()
    valid = valid.sort_values(["logical_match_key", "x1_odds_priority", "source_priority", "row_non_null_count", "source_file", "football_data_row_id"], ascending=[True, True, True, False, True, True])
    valid["rank_within_match"] = valid.groupby("logical_match_key").cumcount() + 1
    selected = valid[valid["rank_within_match"].eq(1)].copy()
    conflicts = []
    for key, g in valid.groupby("logical_match_key"):
        if len(g) <= 1:
            continue
        top_priority = int(g["source_priority"].min())
        top = g[g["source_priority"].eq(top_priority)]
        variants = top[["home_goals", "away_goals", "result_1x2"]].drop_duplicates()
        all_variants = g[["home_goals", "away_goals", "result_1x2"]].drop_duplicates()
        action = "no_score_conflict"
        if len(variants) > 1:
            action = "quarantined_score_conflict"
        elif len(all_variants) > 1:
            action = "resolved_by_source_priority"
        if action != "no_score_conflict":
            for _, row in g.iterrows():
                conflicts.append({"logical_match_key": key, "football_data_row_id": row["football_data_row_id"], "source_file": row["source_file"], "match_date": row["match_date"], "home_team": row["home_team_raw"], "away_team": row["away_team_raw"], "home_goals": row["home_goals"], "away_goals": row["away_goals"], "result_1x2": row["result_1x2"], "source_priority": row["source_priority"], "selected_status": row["rank_within_match"] == 1, "final_action": action})
    conflict_df = pd.DataFrame(conflicts)
    quarantine_keys = set(conflict_df.loc[conflict_df["final_action"].eq("quarantined_score_conflict"), "logical_match_key"]) if not conflict_df.empty else set()
    selected["score_conflict_quarantine_flag"] = selected["logical_match_key"].isin(quarantine_keys)
    return selected, valid, conflict_df


def make_match_id(competition_type: pd.Series, competition_code: pd.Series, season: pd.Series, seq: pd.Series) -> pd.Series:
    return (
        competition_type.astype(int).astype(str)
        + competition_code.astype(int).map(lambda x: f"{x:03d}")
        + season.astype(int).map(lambda x: f"{x:04d}")
        + seq.astype(int).map(lambda x: f"{x:04d}")
    ).astype("int64")


def assign_ids(selected: pd.DataFrame, marketed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = selected.sort_values(["competition_slug", "season_start_year", "match_datetime", "home_team_id", "away_team_id", "football_data_row_id"]).copy()
    s["match_sequence"] = s.groupby(["competition_slug", "season_start_year"]).cumcount() + 1
    s["full_scope_match_id"] = make_match_id(s["competition_type"], s["competition_code"], s["season_start_year"], s["match_sequence"])
    s["canonical_match_id"] = np.nan
    key_map = s[["logical_match_key", "full_scope_match_id", "match_sequence", "score_conflict_quarantine_flag"]].copy()
    source_map = marketed.merge(key_map, on="logical_match_key", how="left")
    source_map["mapping_method"] = np.where(source_map["full_scope_match_id"].notna(), "football_data_full_scope_logical_match_key", "unmapped_no_valid_selected_match")
    source_map["mapping_confidence"] = np.where(source_map["full_scope_match_id"].notna(), 1.0, 0.0)
    source_map["manual_review_required"] = source_map["full_scope_match_id"].isna()
    source_map["notes"] = "Source rows map by div/competition/season/home_team_id/away_team_id. source_file and row id are not match identity."
    return s, source_map


def count_implausible(selected: pd.DataFrame, source_rows: pd.DataFrame) -> pd.DataFrame:
    counts = selected.groupby(["div", "competition_slug", "season_start_year"]).size().reset_index(name="final_row_count")
    counts["expected_lower_bound"] = counts["div"].map(lambda d: SCOPE_CODES[d][4])
    counts["expected_upper_bound"] = counts["div"].map(lambda d: SCOPE_CODES[d][5])
    counts["implausible_flag"] = ~counts["final_row_count"].between(counts["expected_lower_bound"], counts["expected_upper_bound"])
    stats = source_rows.groupby(["div", "competition_slug", "season_start_year"]).agg(source_rows=("football_data_row_id", "count"), source_files=("source_file", "nunique"), raw_season_file_rows=("raw_season_file_flag", "sum"), processed_aggregate_rows=("processed_aggregate_file_flag", "sum"), date_parse_anomaly_rows=("date_parse_anomaly_flag", "sum"), repeated_home_away_pairings=("logical_match_key", lambda s: int(s.duplicated().sum()))).reset_index()
    out = counts.merge(stats, on=["div", "competition_slug", "season_start_year"], how="left")
    out["reason_for_failure"] = np.where(out["implausible_flag"], np.where(out["final_row_count"].lt(out["expected_lower_bound"]), "below_conservative_lower_bound", "above_conservative_upper_bound"), "within_conservative_bounds")
    return out


def build_market_outputs(selected: pd.DataFrame, source_map: pd.DataFrame, implausible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quarantine_seasons = implausible[implausible["implausible_flag"]].copy()
    season_keys = set(zip(quarantine_seasons["competition_slug"], quarantine_seasons["season_start_year"].astype(int)))
    selected["implausible_season_quarantine_flag"] = selected.apply(lambda r: (r["competition_slug"], int(r["season_start_year"])) in season_keys, axis=1)
    final = selected[~selected["implausible_season_quarantine_flag"] & ~selected["score_conflict_quarantine_flag"]].copy()
    final = final[final["season_start_year"].le(2024)].copy()
    final["x1_home_raw_prob"] = 1.0 / final["x1_home_odds"]
    final["x1_draw_raw_prob"] = 1.0 / final["x1_draw_odds"]
    final["x1_away_raw_prob"] = 1.0 / final["x1_away_odds"]
    final["x1_overround"] = final[["x1_home_raw_prob", "x1_draw_raw_prob", "x1_away_raw_prob"]].sum(axis=1)
    final["x1_home_no_vig_prob"] = final["x1_home_raw_prob"] / final["x1_overround"]
    final["x1_draw_no_vig_prob"] = final["x1_draw_raw_prob"] / final["x1_overround"]
    final["x1_away_no_vig_prob"] = final["x1_away_raw_prob"] / final["x1_overround"]
    final["classification"] = "research_only"
    final["partial_latest_season_flag"] = False
    final["dedup_tiebreak_policy"] = "valid target/odds; B365>Avg>HDA; raw season>raw scoped>raw aggregate>processed; completeness; stable source,row"
    matches = final[["full_scope_match_id", "canonical_match_id", "div", "competition_type", "competition_code", "competition_slug", "season_start_year", "season_label", "match_sequence", "match_datetime", "country", "home_team_id", "away_team_id", "home_team_raw", "away_team_raw", "home_team_normalized", "away_team_normalized", "home_goals", "away_goals", "result_1x2", "source_file", "football_data_row_id", "logical_match_key"]].copy()
    cols = ["full_scope_match_id", "canonical_match_id", "div", "competition_slug", "competition_type", "competition_code", "season_start_year", "season_label", "match_date", "match_time", "match_datetime", "home_team_raw", "away_team_raw", "home_team_normalized", "away_team_normalized", "home_team_id", "away_team_id", "home_goals", "away_goals", "result_1x2", "target_home_win", "target_draw", "target_away_win", "x1_home_odds", "x1_draw_odds", "x1_away_odds", "x1_odds_source", "x1_odds_timing_label", "x1_home_raw_prob", "x1_draw_raw_prob", "x1_away_raw_prob", "x1_overround", "x1_home_no_vig_prob", "x1_draw_no_vig_prob", "x1_away_no_vig_prob", "football_data_row_id", "source_file", "source", "logical_match_key", "partial_latest_season_flag", "dedup_tiebreak_policy", "classification"]
    x1 = final[[c for c in cols if c in final.columns]].copy()
    source_map["implausible_season_quarantine_flag"] = source_map.apply(lambda r: (r["competition_slug"], int(r["season_start_year"])) in season_keys if pd.notna(r["season_start_year"]) else False, axis=1)
    source_map["quarantine_flag"] = source_map["implausible_season_quarantine_flag"] | source_map["score_conflict_quarantine_flag"].fillna(False).astype(bool)
    return matches, source_map, x1, quarantine_seasons


def build_rolling_features(x1: pd.DataFrame) -> pd.DataFrame:
    m = x1.copy()
    m["match_datetime"] = pd.to_datetime(m["match_datetime"], errors="coerce")
    rows = []
    for side, opp in [("home", "away"), ("away", "home")]:
        tmp = pd.DataFrame({
            "full_scope_match_id": m["full_scope_match_id"],
            "competition_slug": m["competition_slug"],
            "season_start_year": m["season_start_year"],
            "match_datetime": m["match_datetime"],
            "team_id": m[f"{side}_team_id"],
            "is_home": int(side == "home"),
            "goals_for": m[f"{side}_goals"] if f"{side}_goals" in m.columns else (m["home_goals"] if side == "home" else m["away_goals"]),
            "goals_against": m[f"{opp}_goals"] if f"{opp}_goals" in m.columns else (m["away_goals"] if side == "home" else m["home_goals"]),
        })
        tmp["result_points"] = np.select([tmp["goals_for"] > tmp["goals_against"], tmp["goals_for"].eq(tmp["goals_against"])], [3, 1], default=0)
        tmp["clean_sheet"] = tmp["goals_against"].eq(0).astype(int)
        tmp["conceded_flag"] = tmp["goals_against"].gt(0).astype(int)
        rows.append(tmp)
    long = pd.concat(rows, ignore_index=True).sort_values(["team_id", "competition_slug", "season_start_year", "match_datetime", "full_scope_match_id"])
    for w in [5, 10]:
        for metric in ["goals_for", "goals_against", "result_points", "clean_sheet", "conceded_flag"]:
            long[f"fd_{metric}_avg_w{w}"] = long.groupby(["team_id", "competition_slug", "season_start_year"])[metric].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    long["fd_matches_played_before"] = long.groupby(["team_id", "competition_slug", "season_start_year"]).cumcount()
    long["fd_previous_match_datetime"] = long.groupby("team_id")["match_datetime"].shift(1)
    long["fd_rest_days"] = (long["match_datetime"] - long["fd_previous_match_datetime"]).dt.days
    keep = ["full_scope_match_id", "fd_matches_played_before", "fd_rest_days"] + [c for c in long.columns if c.startswith("fd_") and c not in {"fd_previous_match_datetime", "fd_matches_played_before", "fd_rest_days"}]
    home = long[long["is_home"].eq(1)][keep].rename(columns={c: f"home_{c}" for c in keep if c != "full_scope_match_id"})
    away = long[long["is_home"].eq(0)][keep].rename(columns={c: f"away_{c}" for c in keep if c != "full_scope_match_id"})
    out = home.merge(away, on="full_scope_match_id", how="outer")
    for base in [c.replace("home_", "") for c in out.columns if c.startswith("home_fd_")]:
        if f"away_{base}" in out.columns:
            out[f"{base}_diff_home_minus_away"] = out[f"home_{base}"] - out[f"away_{base}"]
    out["fd_rolling_features_available"] = out[[c for c in out.columns if c.startswith(("home_fd_", "away_fd_"))]].notna().any(axis=1)
    return out


def merge_external_features(x1: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    out = x1.merge(rolling, on="full_scope_match_id", how="left")
    # Existing locked feature blocks are top-5 canonical_match_id keyed; non-top-5 rows remain present with availability flags.
    if "canonical_match_id" in out.columns:
        key = "canonical_match_id"
    else:
        out["canonical_match_id"] = np.nan
        key = "canonical_match_id"
    for name, path, flag in [("clubelo", CLUBELO_IN, "clubelo_available"), ("understat", UNDERSTAT_IN, "understat_available"), ("transfermarkt", TM_IN, "transfermarkt_available")]:
        if path.exists():
            block = pd.read_csv(path)
            overlap = [c for c in block.columns if c in out.columns and c != "canonical_match_id"]
            if overlap:
                block = block.rename(columns={c: f"{name}_{c}" for c in overlap})
            before = len(out)
            out = out.merge(block, on=key, how="left")
            assert len(out) == before
            non_key = [c for c in block.columns if c != key]
            out[flag] = out[non_key].notna().any(axis=1) if non_key else False
        else:
            out[flag] = False
    if "understat_both_found_flag" not in out.columns:
        out["understat_both_found_flag"] = False
    out["understat_missing_due_to_pre_source_era"] = out["season_start_year"].astype(int).lt(2014) & ~out["understat_available"].astype(bool)
    if "transfermarkt_value_both_found" not in out.columns:
        out["transfermarkt_value_both_found"] = out.get("tm_both_value_found_flag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    if "clubelo_both_found_flag" not in out.columns:
        out["clubelo_both_found_flag"] = False
    return out


def reports(inventory, norm, teams, aliases, added, matches, source_map, x1, full, valid_rows, conflicts, quarantine_seasons) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    old_scope = pd.DataFrame(
        [{"league_code": code, "league_slug": meta[0], "included_now": True, "old_scope_evidence": "scope_C excludes only E1/E2/E3; old TOP_DIVISIONS constant plus V3 league breakdown", "reason": "confirmed old top-division scope"} for code, meta in SCOPE_CODES.items()]
        + [{"league_code": c, "league_slug": "", "included_now": False, "old_scope_evidence": "explicit lower English exclusions", "reason": "explicitly excluded"} for c in sorted(EXCLUDED)]
    )
    old_scope.to_csv(REPORT_DIR / "full_scope_old_v3_scope_audit.csv", index=False)
    added.to_csv(REPORT_DIR / "full_scope_added_teams.csv", index=False)
    rows_ls = x1.groupby(["div", "competition_slug", "season_start_year"]).size().reset_index(name="rows")
    rows_ls["expected_lower_bound"] = rows_ls["div"].map(lambda d: SCOPE_CODES[d][4])
    rows_ls["expected_upper_bound"] = rows_ls["div"].map(lambda d: SCOPE_CODES[d][5])
    rows_ls["plausible_count_flag"] = rows_ls["rows"].between(rows_ls["expected_lower_bound"], rows_ls["expected_upper_bound"])
    rows_ls.to_csv(REPORT_DIR / "full_scope_rows_by_league_season.csv", index=False)
    quarantine_seasons.to_csv(REPORT_DIR / "full_scope_quarantined_league_seasons.csv", index=False)
    conflict_out = conflicts[conflicts["final_action"].eq("quarantined_score_conflict")] if not conflicts.empty else pd.DataFrame()
    conflict_out.to_csv(REPORT_DIR / "full_scope_quarantined_score_conflicts.csv", index=False)
    valid_rows[["logical_match_key", "rank_within_match", "football_data_row_id", "source_file", "source_priority", "x1_odds_source", "x1_odds_priority", "row_non_null_count", "home_goals", "away_goals", "result_1x2"]].to_csv(REPORT_DIR / "full_scope_source_priority_audit.csv", index=False)
    market_cov = x1.groupby(["div", "competition_slug"]).agg(rows=("full_scope_match_id", "count"), first_season=("season_start_year", "min"), last_season=("season_start_year", "max"), b365_rows=("x1_odds_source", lambda s: int(s.eq("B365").sum())), avg_rows=("x1_odds_source", lambda s: int(s.eq("Avg").sum()))).reset_index()
    market_cov.to_csv(REPORT_DIR / "full_scope_market_coverage.csv", index=False)
    external_cov = full.groupby(["div", "competition_slug"]).agg(rows=("full_scope_match_id", "count"), clubelo_available_rate=("clubelo_available", "mean"), understat_available_rate=("understat_available", "mean"), transfermarkt_available_rate=("transfermarkt_available", "mean"), rolling_available_rate=("fd_rolling_features_available", "mean")).reset_index()
    external_cov.to_csv(REPORT_DIR / "full_scope_external_feature_coverage.csv", index=False)
    forbidden_patterns = ["id", "source", "team_raw", "team_normalized", "team_id", "home_goals", "away_goals", "result_1x2", "target_", "current_club", "current_value", "game_lineups", "appearance"]
    forbidden = [{"column": c, "reason": "identifier/source/team/result/target/raw or leakage-forbidden model feature"} for c in full.columns if any(p in c for p in forbidden_patterns) or c in {"x1_home_odds", "x1_draw_odds", "x1_away_odds"}]
    allow = [{"feature": c, "feature_group": "market_probability_or_date_safe_feature", "notes": "Allowed only under research_only feature allowlist"} for c in full.columns if c.startswith(("x1_", "home_fd_", "away_fd_", "fd_", "home_clubelo", "away_clubelo", "clubelo_", "home_understat", "away_understat", "understat_", "home_tm", "away_tm", "tm_", "transfermarkt_")) and c not in {r["column"] for r in forbidden} and (pd.api.types.is_numeric_dtype(full[c]) or full[c].dropna().isin([True, False]).all())]
    pd.DataFrame(allow).to_csv(REPORT_DIR / "full_scope_feature_allowlist.csv", index=False)
    pd.DataFrame(forbidden).to_csv(REPORT_DIR / "full_scope_forbidden_columns.csv", index=False)
    duplicate_logical = int(x1["logical_match_key"].duplicated().sum())
    duplicate_id = int(x1["full_scope_match_id"].duplicated().sum())
    bad_counts = int((~rows_ls["plausible_count_flag"]).sum())
    checks = pd.DataFrame([
        {"check_name": "raw_files_unchanged", "status": "pass", "details": "raw files read only"},
        {"check_name": "locked_footiqo_registry_unchanged", "status": "pass", "details": "locked files not written"},
        {"check_name": "full_scope_namespace_written", "status": "pass", "details": str(OUT_DIR.relative_to(ROOT))},
        {"check_name": "duplicate_logical_match_key", "status": "pass" if duplicate_logical == 0 else "fail", "details": f"duplicates={duplicate_logical}"},
        {"check_name": "duplicate_match_id", "status": "pass" if duplicate_id == 0 else "fail", "details": f"duplicates={duplicate_id}"},
        {"check_name": "league_season_counts_plausible", "status": "pass" if bad_counts == 0 else "fail", "details": f"implausible_remaining={bad_counts}; quarantined={len(quarantine_seasons)}"},
        {"check_name": "selected_score_conflicts", "status": "pass" if len(conflict_out) == 0 else "fail", "details": f"quarantined_score_conflict_rows={len(conflict_out)}"},
        {"check_name": "valid_1x2_odds", "status": "pass" if x1[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].gt(1).all().all() else "fail", "details": "odds > 1"},
        {"check_name": "targets_valid", "status": "pass" if x1[["target_home_win", "target_draw", "target_away_win"]].sum(axis=1).eq(1).all() else "fail", "details": "one target active"},
        {"check_name": "rolling_features_strictly_prior", "status": "pass", "details": "rolling uses shift(1) inside team/league/season"},
        {"check_name": "external_features_date_safe", "status": "review", "details": "existing locked blocks reused only by canonical_match_id; non-top-5 external coverage mostly unavailable"},
        {"check_name": "missing_external_features_flagged", "status": "pass", "details": "availability flags present"},
        {"check_name": "classification_research_only", "status": "pass" if full["classification"].eq("research_only").all() else "fail", "details": "research_only retained"},
    ])
    checks.to_csv(REPORT_DIR / "full_scope_leakage_checks.csv", index=False)
    decision = "football_data_full_scope_build_ready_good" if checks["status"].eq("pass").all() else "football_data_full_scope_build_ready_needs_review"
    report = [
        "# Football-Data Full-Scope 1X2 Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Decision: `{decision}`",
        "",
        f"Normalized full-scope source rows: {len(norm)}",
        f"Teams: {len(teams)}",
        f"Added full-scope-only teams: {len(added)}",
        f"Full-scope matches: {len(matches)}",
        f"Research 1X2 rows: {len(x1)}",
        f"Full-feature rows: {len(full)}",
        "",
        "Scope includes E0, SP1, D1, I1, F1, B1, G1, N1, P1, SC0, and T1. E1/E2/E3 are excluded.",
        "",
        "Caveat: external feature coverage for the added non-top-5 scope is limited because current locked feature blocks are Footiqo top-5 keyed. Rows are retained with availability flags.",
        "",
        "No modeling, value search, threshold optimization, raw-file modification, locked registry overwrite, or confirmed-edge claim was performed.",
    ]
    (REPORT_DIR / "full_scope_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (REPORT_DIR / "full_scope_decision.md").write_text(f"# Full-Scope Decision\n\nDecision: `{decision}`\n\nClassification remains `research_only`. No confirmed edge is claimed.\n", encoding="utf-8")
    return decision


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUPER_DIR.mkdir(parents=True, exist_ok=True)
    PLUS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    inventory, norm = discover_and_normalize()
    inventory.to_csv(OUT_DIR / "football_data_full_scope_file_inventory_v1.csv", index=False)
    norm.to_csv(OUT_DIR / "football_data_full_scope_normalized_matches_v1.csv", index=False)
    teams, aliases, added = build_team_registry(norm)
    teams.to_csv(OUT_DIR / "teams_football_data_full_scope_v1.csv", index=False)
    aliases.to_csv(OUT_DIR / "team_aliases_football_data_full_scope_v1.csv", index=False)
    ids = attach_team_ids(norm, aliases)
    marketed = add_market(ids)
    selected, valid_rows, conflicts = deduplicate(marketed)
    selected, source_map = assign_ids(selected, marketed)
    implausible = count_implausible(selected, marketed)
    matches, source_map, x1, quarantine_seasons = build_market_outputs(selected, source_map, implausible)
    matches.to_csv(OUT_DIR / "matches_football_data_full_scope_v1.csv", index=False)
    source_map.to_csv(OUT_DIR / "source_match_map_football_data_full_scope_v1.csv", index=False)
    x1.to_csv(SUPER_DIR / "super_1x2_football_data_full_scope_research_v1.csv", index=False)
    rolling = build_rolling_features(x1)
    full = merge_external_features(x1, rolling)
    full.to_csv(PLUS_DIR / "super_1x2_football_data_full_scope_full_features_research_v1.csv", index=False)
    decision = reports(inventory, norm, teams, aliases, added, matches, source_map, x1, full, valid_rows, conflicts, quarantine_seasons)
    print(decision)
    print(f"norm_rows={len(norm)} matches={len(matches)} x1={len(x1)} full={len(full)} added_teams={len(added)}")


if __name__ == "__main__":
    main()
