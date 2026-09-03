from __future__ import annotations

import csv
import gzip
import json
import math
import os
import re
import sqlite3
import warnings
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [
    ROOT / "data",
    ROOT / "data/raw",
    ROOT / "data/raw_external",
    ROOT / "data/external",
    ROOT / "data/interim",
    ROOT / "data/processed",
]
REPORT_DIR = ROOT / "outputs/reports/data_inventory"
REGISTRY_DIR = ROOT / "data/processed/match_registry"
SUPPORTED_EXTS = {
    ".csv",
    ".csv.gz",
    ".xlsx",
    ".xls",
    ".parquet",
    ".json",
    ".json.gz",
    ".sqlite",
    ".db",
    ".zip",
    ".txt",
    ".md",
}
GENERATED_OUTPUTS = {
    ROOT / "data/processed/match_registry/competition_registry_draft.csv",
}

LEAGUE_CODE_MAP = {
    "E0": ("england_premier_league", "England", "Premier League"),
    "SP1": ("spain_laliga", "Spain", "La Liga"),
    "D1": ("germany_bundesliga", "Germany", "Bundesliga"),
    "I1": ("italy_serie_a", "Italy", "Serie A"),
    "F1": ("france_ligue_1", "France", "Ligue 1"),
    "E1": ("england_championship", "England", "Championship"),
    "E2": ("england_league_one", "England", "League One"),
    "E3": ("england_league_two", "England", "League Two"),
    "B1": ("belgium_first_division_a", "Belgium", "First Division A"),
    "G1": ("greece_super_league", "Greece", "Super League"),
    "N1": ("netherlands_eredivisie", "Netherlands", "Eredivisie"),
    "P1": ("portugal_primeira_liga", "Portugal", "Primeira Liga"),
    "SC0": ("scotland_premiership", "Scotland", "Premiership"),
    "T1": ("turkey_super_lig", "Turkey", "Super Lig"),
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def extension(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv.gz"):
        return ".csv.gz"
    if name.endswith(".json.gz"):
        return ".json.gz"
    return path.suffix.lower()


def stage(path: Path) -> str:
    parts = path.relative_to(ROOT).parts
    if "raw_external" in parts:
        return "raw"
    if "raw" in parts:
        return "raw"
    if "interim" in parts:
        return "interim"
    if "processed" in parts:
        return "processed"
    if "outputs" in parts or "output" in parts:
        return "output"
    if "external" in parts:
        return "external"
    return "other"


def likely_source(path: Path) -> str:
    p = rel(path).lower()
    if "football_data" in p or re.search(r"data/raw/([a-z]{1,3}\d|sc0)", p, re.I):
        return "football-data.co.uk"
    if "footiqo" in p:
        return "footiqo"
    if "clubelo" in p or "elo" in path.name.lower():
        return "clubelo"
    if "understat" in p:
        return "understat"
    if "fbref" in p:
        return "fbref"
    if "transfermarkt" in p or "player_scores" in p:
        return "transfermarkt"
    if "beat_the_bookie" in p:
        return "beat_the_bookie"
    if "weather" in p or "open_meteo" in p:
        return "open_meteo_weather"
    if "stadium" in p:
        return "stadiums"
    if "manual" in p:
        return "manual"
    return "unknown"


def discover_files() -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.resolve() in {p.resolve() for p in GENERATED_OUTPUTS}:
                continue
            if path.is_file() and extension(path) in SUPPORTED_EXTS:
                real = path.resolve()
                if real not in seen:
                    seen.add(real)
                    files.append(path)
    return sorted(files, key=lambda p: rel(p).lower())


def candidate_cols(columns: list[str], patterns: list[str]) -> list[str]:
    out = []
    for col in columns:
        c = col.lower()
        if any(re.search(pat, c) for pat in patterns):
            out.append(col)
    return out


def infer_column_groups(columns: list[str]) -> dict[str, list[str]]:
    return {
        "candidate_date_columns": candidate_cols(columns, [r"(^|_)date($|_)", r"time", r"datetime", r"kick", r"matchday"]),
        "candidate_league_columns": candidate_cols(columns, [r"league", r"div", r"competition", r"comp", r"tournament"]),
        "candidate_season_columns": candidate_cols(columns, [r"season", r"year", r"campaign"]),
        "candidate_home_team_columns": candidate_cols(columns, [r"home.*team", r"hometeam", r"^home$", r"home_club", r"home_name"]),
        "candidate_away_team_columns": candidate_cols(columns, [r"away.*team", r"awayteam", r"^away$", r"away_club", r"away_name"]),
        "candidate_score_result_columns": candidate_cols(columns, [r"fthg", r"ftag", r"ftr", r"hthg", r"htag", r"htr", r"score", r"goal", r"result", r"winner"]),
        "candidate_odds_columns": candidate_cols(columns, [r"odds", r"odd", r"book", r"b365", r"ps[had]?", r"avg[had]", r"max[had]", r"over", r"under", r"ah", r"handicap", r"btts", r"draw no bet", r"dnb", r"double chance"]),
        "candidate_stats_columns": candidate_cols(columns, [r"shot", r"poss", r"corner", r"card", r"foul", r"xg", r"stat", r"touch", r"pass", r"deep", r"ppda"]),
        "candidate_player_team_valuation_columns": candidate_cols(columns, [r"player", r"squad", r"value", r"valuation", r"market_value", r"transfer", r"club"]),
        "candidate_referee_columns": candidate_cols(columns, [r"referee", r"ref_"]),
    }


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def parse_dates(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce", dayfirst=True, utc=False)


def summarize_missing(sample: pd.DataFrame) -> dict[str, float]:
    if sample.empty:
        return {}
    ratios = sample.isna().mean().sort_values(ascending=False)
    top = ratios[ratios > 0].head(20)
    return {str(k): round(float(v), 4) for k, v in top.items()}


def duplicate_id_summary(sample: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for col in sample.columns:
        c = col.lower()
        if c == "id" or c.endswith("_id") or "match_id" in c or "game_id" in c:
            s = sample[col].dropna()
            if len(s) > 0:
                out[col] = {
                    "sample_non_null": int(len(s)),
                    "sample_duplicate_values": int(s.duplicated().sum()),
                }
    return out


def date_bounds_from_sample(sample: pd.DataFrame, date_cols: list[str]) -> tuple[str, str]:
    mins = []
    maxs = []
    if not date_cols:
        return "", ""
    for col in date_cols[:5]:
        if col in sample.columns:
            parsed = parse_dates(sample[col])
            if parsed.notna().any():
                mins.append(parsed.min())
                maxs.append(parsed.max())
    if not mins:
        return "", ""
    return str(min(mins).date()), str(max(maxs).date())


def inspect_csv(path: Path, gz: bool = False) -> dict[str, Any]:
    kwargs = {"compression": "gzip"} if gz else {}
    sample = pd.read_csv(path, nrows=5000, low_memory=False, **kwargs)
    columns = list(sample.columns)
    groups = infer_column_groups(columns)
    row_count = 0
    try:
        for chunk in pd.read_csv(path, chunksize=100000, usecols=[columns[0]], low_memory=False, **kwargs):
            row_count += len(chunk)
    except Exception:
        row_count = len(sample)
    first_date, last_date = date_bounds_from_sample(sample, groups["candidate_date_columns"])
    return {
        "read_status": "read",
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "first_date": first_date,
        "last_date": last_date,
        "missingness_summary": summarize_missing(sample),
        "duplicate_candidate_ids": duplicate_id_summary(sample),
        **groups,
    }


def inspect_json(path: Path, gz: bool = False) -> dict[str, Any]:
    opener = gzip.open if gz else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    rows = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        sample = pd.DataFrame(rows[:5000])
        columns = list(sample.columns)
        groups = infer_column_groups(columns)
        return {
            "read_status": "read",
            "row_count": len(rows),
            "column_count": len(columns),
            "columns": columns,
            "first_date": "",
            "last_date": "",
            "missingness_summary": summarize_missing(sample),
            "duplicate_candidate_ids": duplicate_id_summary(sample),
            **groups,
        }
    keys = list(data.keys()) if isinstance(data, dict) else []
    return {
        "read_status": "read_metadata_only",
        "row_count": 1 if isinstance(data, dict) else 0,
        "column_count": len(keys),
        "columns": keys,
        "first_date": "",
        "last_date": "",
        "missingness_summary": {},
        "duplicate_candidate_ids": {},
        **infer_column_groups(keys),
    }


def inspect_sqlite(path: Path) -> dict[str, Any]:
    con = sqlite3.connect(path)
    try:
        tables = pd.read_sql_query("select name from sqlite_master where type='table'", con)["name"].tolist()
        columns = []
        row_counts = {}
        for table in tables:
            cnt = pd.read_sql_query(f'select count(*) as n from "{table}"', con)["n"].iloc[0]
            row_counts[table] = int(cnt)
            info = pd.read_sql_query(f'pragma table_info("{table}")', con)
            columns.extend([f"{table}.{c}" for c in info["name"].tolist()])
        groups = infer_column_groups(columns)
        return {
            "read_status": "read_metadata_only",
            "row_count": sum(row_counts.values()),
            "column_count": len(columns),
            "columns": columns,
            "first_date": "",
            "last_date": "",
            "missingness_summary": {},
            "duplicate_candidate_ids": {},
            "sqlite_tables": row_counts,
            **groups,
        }
    finally:
        con.close()


def inspect_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    return {
        "read_status": "archive_listed",
        "row_count": "",
        "column_count": len(names),
        "columns": names[:200],
        "first_date": "",
        "last_date": "",
        "missingness_summary": {},
        "duplicate_candidate_ids": {},
        **infer_column_groups(names),
    }


def inspect_file(path: Path) -> dict[str, Any]:
    ext = extension(path)
    try:
        if ext == ".csv":
            return inspect_csv(path)
        if ext == ".csv.gz":
            return inspect_csv(path, gz=True)
        if ext == ".json":
            return inspect_json(path)
        if ext == ".json.gz":
            return inspect_json(path, gz=True)
        if ext in {".sqlite", ".db"}:
            return inspect_sqlite(path)
        if ext == ".zip":
            return inspect_zip(path)
        if ext in {".xlsx", ".xls", ".parquet"}:
            return {
                "read_status": "unsupported_optional_dependency",
                "row_count": "",
                "column_count": "",
                "columns": [],
                "first_date": "",
                "last_date": "",
                "missingness_summary": {},
                "duplicate_candidate_ids": {},
                **infer_column_groups([]),
            }
        return {
            "read_status": "metadata_only",
            "row_count": "",
            "column_count": "",
            "columns": [],
            "first_date": "",
            "last_date": "",
            "missingness_summary": {},
            "duplicate_candidate_ids": {},
            **infer_column_groups([]),
        }
    except Exception as exc:
        return {
            "read_status": f"read_error: {type(exc).__name__}: {exc}",
            "row_count": "",
            "column_count": "",
            "columns": [],
            "first_date": "",
            "last_date": "",
            "missingness_summary": {},
            "duplicate_candidate_ids": {},
            **infer_column_groups([]),
        }


def classify_source(path: Path, cols: list[str]) -> list[str]:
    p = rel(path).lower()
    c = " ".join(x.lower() for x in cols)
    labels = set()
    if any(x in c for x in ["hometeam", "awayteam", "home_team", "away_team", "fthg", "ftag"]) or "matches" in p or "/seasons/" in p:
        labels.add("match_results")
    if any(x in c for x in ["b365h", "b365d", "b365a", "psh", "psd", "psa", "maxh", "maxd", "maxa"]):
        labels.add("market_odds_1x2")
    if any(x in c for x in ["over", "under", "o2.5", "u2.5", ">2.5", "<2.5", "avg>2.5", "b365>2.5"]):
        labels.add("market_odds_ou")
    if "btts" in c:
        labels.add("market_odds_btts")
    if any(x in c for x in ["ahh", "aha", "handicap", "asian"]):
        labels.add("market_odds_ah")
    if any(x in c for x in ["double chance", "dnb", "draw no bet"]):
        labels.add("market_odds_dc_dnb")
    if any(x in c for x in ["open", "close", "closing", "movement"]):
        labels.add("odds_movement")
    if any(x in c for x in ["shot", "poss", "xg", "corner", "card", "foul"]):
        labels.add("postmatch_team_stats")
    if any(x in c for x in ["corner", "card"]):
        labels.add("corners_cards_stats")
    if any(x in c for x in ["shot", "poss"]):
        labels.add("shots_possession_stats")
    if "xg" in c or "understat" in p:
        labels.add("xg_stats")
    if any(x in p for x in ["players", "player_scores", "appearances", "lineups"]) or "player" in c:
        labels.add("player_data")
    if "transfermarkt" in p or "market_value" in c or "valuation" in c:
        labels.add("transfermarkt_values")
    if "clubelo" in p or "elo" in c:
        labels.add("clubelo_ratings")
    if "fbref" in p:
        labels.add("fbref_prior_season")
    if "understat" in p:
        labels.add("understat_match_stats")
    if "referee" in p or "referee" in c:
        labels.add("referee_context")
    if not labels:
        labels.add("unknown_needs_review")
    return sorted(labels)


def classify_leakage(path: Path, cols: list[str], source_labels: list[str]) -> list[str]:
    p = rel(path).lower()
    c = " ".join(x.lower() for x in cols)
    risks = set()
    if "outputs/" in p or "selected_bets" in p or "candidate" in p or "prediction" in p or "leakage_audit" in p:
        risks.add("diagnostic_only")
    if any(label in source_labels for label in ["match_results", "postmatch_team_stats", "xg_stats", "understat_match_stats", "corners_cards_stats", "shots_possession_stats"]):
        risks.add("post_match_only")
    if "footiqo" in p or any(label.startswith("market_odds") for label in source_labels) or "odds_movement" in source_labels:
        risks.add("odds_timing_unknown")
    if "transfermarkt" in p and any(x in c for x in ["current_club", "club_name", "current"]):
        risks.add("future_leakage_risk")
    if "clubelo" in p:
        risks.add("pre_match_safe_if_lagged")
    if "rolling" in p or "feature" in p or "elo" in c or "weather" in p or "stadium" in p:
        risks.add("pre_match_safe_if_lagged")
    if "referee" in p and any(x in c for x in ["season", "aggregate", "avg", "mean", "total"]):
        risks.add("all_seasons_aggregate_context_only")
    if not risks:
        risks.add("raw_safe")
    return sorted(risks)


def league_values(sample_cols: list[str], path: Path) -> str:
    p = rel(path)
    found = []
    for code, (_, _, name) in LEAGUE_CODE_MAP.items():
        if re.search(rf"(^|[/_ ]){re.escape(code)}($|[/_ .(])", p):
            found.append(name)
    return "; ".join(sorted(set(found)))


def write_competition_registry(schema_rows: list[dict[str, Any]]) -> None:
    base = [
        (1, 1, "england_premier_league", "Premier League", "England", "domestic_club_league"),
        (1, 2, "spain_laliga", "La Liga", "Spain", "domestic_club_league"),
        (1, 3, "germany_bundesliga", "Bundesliga", "Germany", "domestic_club_league"),
        (1, 4, "italy_serie_a", "Serie A", "Italy", "domestic_club_league"),
        (1, 5, "france_ligue_1", "Ligue 1", "France", "domestic_club_league"),
        (2, 1, "uefa_champions_league", "UEFA Champions League", "", "international_club_competition"),
        (2, 2, "uefa_europa_league", "UEFA Europa League", "", "international_club_competition"),
        (2, 3, "uefa_conference_league", "UEFA Conference League", "", "international_club_competition"),
        (2, 4, "fifa_club_world_cup", "FIFA Club World Cup", "", "international_club_competition"),
        (2, 5, "copa_libertadores", "Copa Libertadores", "", "international_club_competition"),
        (2, 6, "copa_sudamericana", "Copa Sudamericana", "", "international_club_competition"),
        (3, 1, "england_fa_cup", "FA Cup", "England", "domestic_club_cup"),
        (3, 2, "england_efl_cup", "EFL Cup", "England", "domestic_club_cup"),
        (3, 3, "spain_copa_del_rey", "Copa del Rey", "Spain", "domestic_club_cup"),
        (3, 4, "germany_dfb_pokal", "DFB-Pokal", "Germany", "domestic_club_cup"),
        (3, 5, "italy_coppa_italia", "Coppa Italia", "Italy", "domestic_club_cup"),
        (3, 6, "france_coupe_de_france", "Coupe de France", "France", "domestic_club_cup"),
        (4, 1, "england_community_shield", "Community Shield", "England", "club_supercup"),
        (4, 2, "uefa_super_cup", "UEFA Super Cup", "", "club_supercup"),
        (4, 3, "italy_supercoppa", "Supercoppa Italiana", "Italy", "club_supercup"),
        (4, 4, "spain_supercopa", "Supercopa de Espana", "Spain", "club_supercup"),
        (4, 5, "germany_supercup", "DFL-Supercup", "Germany", "club_supercup"),
        (4, 6, "france_trophee_champions", "Trophee des Champions", "France", "club_supercup"),
        (5, 1, "world_cup", "World Cup", "", "national_team_competition"),
        (5, 2, "euro", "European Championship", "", "national_team_competition"),
        (5, 3, "nations_league", "Nations League", "", "national_team_competition"),
        (5, 4, "copa_america", "Copa America", "", "national_team_competition"),
        (5, 5, "afcon", "Africa Cup of Nations", "", "national_team_competition"),
    ]
    seen_sources = defaultdict(set)
    for row in schema_rows:
        p = row["path"]
        src = row["likely_source"]
        for code, (slug, _, _) in LEAGUE_CODE_MAP.items():
            if re.search(rf"(^|[/_ ]){re.escape(code)}($|[/_ .(])", p):
                seen_sources[slug].add(src)
        low = p.lower()
        for slug in [x[2] for x in base]:
            if slug in low or slug.replace("_", "-") in low:
                seen_sources[slug].add(src)
    rows = []
    for typ, code, slug, name, country, scope in base:
        rows.append(
            {
                "competition_type": typ,
                "competition_code": f"{code:03d}",
                "competition_slug": slug,
                "competition_name": name,
                "country": country,
                "scope": scope,
                "source_names_seen": "; ".join(sorted(seen_sources.get(slug, []))),
                "notes": "Seed draft code; requires audit before canonical use.",
            }
        )
    with (REGISTRY_DIR / "competition_registry_draft.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def market_matrix(schema_rows: list[dict[str, Any]], class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_path = {r["path"]: r for r in schema_rows}
    classes = {r["path"]: r["source_classes"].split("; ") for r in class_rows}
    markets = {
        "1X2": {"support": ["market_odds_1x2", "match_results"], "real_odds": ["market_odds_1x2"]},
        "Asian Handicap": {"support": ["market_odds_ah", "match_results"], "real_odds": ["market_odds_ah"]},
        "O/U 0.5": {"support": ["market_odds_ou", "match_results"], "real_odds": ["market_odds_ou"]},
        "O/U 1.5": {"support": ["market_odds_ou", "match_results"], "real_odds": ["market_odds_ou"]},
        "O/U 2.5": {"support": ["market_odds_ou", "match_results"], "real_odds": ["market_odds_ou"]},
        "O/U 3.5": {"support": ["market_odds_ou", "match_results"], "real_odds": ["market_odds_ou"]},
        "O/U 4.5": {"support": ["market_odds_ou", "match_results"], "real_odds": ["market_odds_ou"]},
        "BTTS": {"support": ["market_odds_btts", "match_results"], "real_odds": ["market_odds_btts"]},
        "corners": {"support": ["corners_cards_stats"], "real_odds": []},
        "cards": {"support": ["corners_cards_stats"], "real_odds": []},
        "team totals": {"support": ["match_results", "market_odds_ou"], "real_odds": []},
        "double chance": {"support": ["market_odds_dc_dnb", "market_odds_1x2", "match_results"], "real_odds": ["market_odds_dc_dnb"]},
        "draw no bet": {"support": ["market_odds_dc_dnb", "market_odds_1x2", "match_results"], "real_odds": ["market_odds_dc_dnb"]},
    }
    rows = []
    for market, spec in markets.items():
        wanted = spec["support"]
        real_odds_labels = spec["real_odds"]
        paths = []
        leagues = set()
        first_dates = []
        last_dates = []
        has_real_odds = False
        has_inferable_1x2 = False
        for path, labels in classes.items():
            if any(w in labels for w in wanted):
                paths.append(path)
                s = by_path[path]
                if s.get("leagues"):
                    leagues.update(x.strip() for x in s["leagues"].split(";") if x.strip())
                if s.get("first_date"):
                    first_dates.append(s["first_date"])
                if s.get("last_date"):
                    last_dates.append(s["last_date"])
                if any(x in labels for x in real_odds_labels):
                    has_real_odds = True
                if "market_odds_1x2" in labels:
                    has_inferable_1x2 = True
        has_target = any("match_results" in classes.get(p, []) for p in paths)
        if market in {"corners", "cards"}:
            has_target = bool(paths)
            paired = "corner/card stat targets found; real corner/card odds not confirmed"
        elif market == "team totals":
            paired = "team-total targets may be derivable from scores; real team-total odds not confirmed"
        elif market in {"double chance", "draw no bet"}:
            paired = "synthetic probabilities inferable from paired 1X2 odds; direct real DC/DNB odds not confirmed unless market_odds_dc_dnb rows are audited" if has_inferable_1x2 else "needs direct odds audit"
        elif "O/U" in market:
            paired = "available_if_exact_line_over_under_pairs_present"
        else:
            paired = "needs_manual_audit"
        recommendation = "manual review before registry build; timing and pairing must be documented" if has_real_odds else "target may be derivable, but direct real odds are limited, absent, or unconfirmed"
        rows.append(
            {
                "market": market,
                "has_target": bool(has_target or market in {"corners", "cards"}),
                "has_real_odds": bool(has_real_odds),
                "paired_odds_availability": paired,
                "source_files": "; ".join(paths[:80]),
                "date_range": f"{min(first_dates) if first_dates else ''} to {max(last_dates) if last_dates else ''}",
                "leagues": "; ".join(sorted(leagues)),
                "recommendation": recommendation,
            }
        )
    return rows


def markdown_reports(file_rows, schema_rows, class_rows, leakage_rows, market_rows) -> None:
    status_counts = Counter(r["read_status"].split(":")[0] for r in schema_rows)
    class_counts = Counter()
    risk_counts = Counter()
    for r in class_rows:
        class_counts.update(r["source_classes"].split("; "))
    for r in leakage_rows:
        risk_counts.update(r["leakage_risks"].split("; "))
    decision = "data_inventory_ready_needs_manual_review"
    summary = [
        "# Football Data Inventory Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "This inventory is read-only with respect to source data. No modeling, value search, source joins, final super CSVs, or canonical match registry build were performed.",
        "",
        "## Scope",
        f"- Files discovered: {len(file_rows)}",
        f"- Files inspected/read or metadata-listed: {len(schema_rows)}",
        f"- Read status counts: {dict(status_counts)}",
        f"- Source class counts: {dict(class_counts)}",
        f"- Leakage risk counts: {dict(risk_counts)}",
        "",
        "## Conservative Interpretation",
        "- Football-data style CSVs appear to provide broad match results and several odds markets, but bookmaker timing must be treated as unknown unless documented per source.",
        "- Understat/FBref/team stat files are post-match for same-fixture prediction and only candidates for lagged feature blocks after strict temporal construction.",
        "- ClubElo can be pre-match safe only when ratings are joined strictly before kickoff.",
        "- Transfermarkt/player data requires date-safe point-in-time handling; current club/value columns are treated as future leakage risk until audited.",
        "- Referee season aggregates are context-only for historical backtests unless rebuilt as lagged pre-match aggregates.",
        "",
        "## Output Files",
        "- full_file_inventory.csv",
        "- source_schema_inventory.csv",
        "- source_classification.csv",
        "- leakage_risk_inventory.csv",
        "- market_availability_matrix.csv",
        "- canonical_registry_plan.md",
        "- super_csv_plan.md",
        "- recommended_next_steps.md",
        "- data/processed/match_registry/competition_registry_draft.csv",
    ]
    (REPORT_DIR / "data_inventory_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    canonical = """# Canonical Registry Plan

Decision: **data_inventory_ready_needs_manual_review**

Do not build the full registry until source timing, league aliases, and team aliases are audited.

## Canonical ID Policy

Use int64 `canonical_match_id` with format `TLLLYYYYGGGG`.

- `competition_type`: 1 domestic club leagues, 2 international club competitions, 3 domestic club cups, 4 club supercups/single-match club trophies, 5 national team competitions.
- `competition_code`: three-digit code within type; type 0 is forbidden.
- `season_start_year`: domestic season start year or tournament year.
- `match_sequence`: deterministic sequence within competition and season/year.

Match sequence ordering:

1. `match_datetime`
2. `home_team_normalized`
3. `away_team_normalized`
4. `source_match_id` if available

## Proposed Fields

`canonical_match_id`, `competition_type`, `competition_code`, `season_start_year`, `match_sequence`, `source`, `source_match_id`, `match_datetime`, `season_label`, `country`, `league`, `home_team_raw`, `away_team_raw`, `home_team_normalized`, `away_team_normalized`, `home_goals`, `away_goals`, `result_1x2`.

## Source-to-Canonical Mapping Strategy

- Use exact source IDs where available and stable within source.
- Otherwise build a source-local match key from league/season/date/home/away.
- Cross-source joins should use normalized league/date/home/away only after league and team alias tables are reviewed.
- Fuzzy matching is allowed only after audit and should emit confidence/review flags.
- Maintain manual alias tables for teams and leagues.
- Create a required `source_match_map` table with `canonical_match_id`, `source`, `source_match_id`, source-local key fields, match status, confidence, and audit notes.

## Required Pre-Build Audits

- Confirm each source's kickoff date/time timezone handling.
- Confirm source odds timing and closing/opening semantics.
- Resolve duplicate raw season files and variant files.
- Review quarantined/mislabeled archives before any inclusion.
- Lock competition code assignments only after all observed competitions are mapped.
"""
    (REPORT_DIR / "canonical_registry_plan.md").write_text(canonical, encoding="utf-8")

    super_plan = """# Super CSV Plan

No super CSVs were built.

All super CSVs must be generated from the canonical registry plus audited source feature blocks. Every row should retain `canonical_match_id`, source provenance, temporal flags, and market-specific availability flags.

## Proposed Files

### super_1x2_v1.csv
- Target columns: `home_win`, `draw`, `away_win`, `result_1x2`.
- Required real odds: paired home/draw/away odds from the same bookmaker/timestamp where possible.
- No-vig probabilities: `p_home_novig`, `p_draw_novig`, `p_away_novig`.
- Allowed feature blocks: lagged team form, lagged Elo, lagged squad/value snapshots, lagged weather/travel/stadium context, lagged referee context.
- Forbidden columns: same-match score, full-time result, same-match stats/xG/cards/corners, future transfers/current club snapshots, post-result aggregates.

### super_ah_v1.csv
- Target columns: handicap line, home/away AH settlement, push flag.
- Required real odds: paired AH home/away odds on the same line and timestamp.
- No-vig probabilities: `p_ah_home_novig`, `p_ah_away_novig`.
- Allowed feature blocks: same as 1X2 plus audited market context.
- Forbidden columns: same-match outcomes except target settlement, post-match stats, future leakage fields.

### super_ou15_v1.csv / super_ou25_v1.csv / super_ou35_v1.csv / super_ou45_v1.csv
- Target columns: total goals, line-specific over/under target, push flag if applicable.
- Required real odds: paired over/under odds for the exact line and timestamp.
- No-vig probabilities: `p_over_novig`, `p_under_novig`.
- Allowed feature blocks: lagged scoring/conceding form, lagged xG, lagged Elo, lagged squad/value, schedule/weather/referee context.
- Forbidden columns: same-match goals except target, same-match shots/xG/corners/cards/possession, future team/player state.

### super_btts_v1.csv
- Target columns: `btts_yes`, home goals, away goals for target audit.
- Required real odds: paired BTTS yes/no odds from same timestamp.
- No-vig probabilities: `p_btts_yes_novig`, `p_btts_no_novig`.
- Allowed feature blocks: lagged attack/defense, lagged xG, lagged squad/value, schedule/weather/referee context.
- Forbidden columns: same-match score/stats and future player/club fields.

## Flags Needed

`source_presence_*`, `odds_timestamp_known`, `odds_pair_complete`, `feature_lag_verified`, `team_alias_reviewed`, `league_alias_reviewed`, `missing_feature_block_*`, `stale_feature_block_*`, `postponed_or_neutral_flag`, `manual_review_required`.

## Leakage Risks

All post-match statistics and same-fixture xG are forbidden as predictors. Odds timing is unknown for several sources. Transfermarkt current-club/current-value fields need point-in-time reconstruction before use. Referee and season aggregates must be rebuilt as lagged features or marked context-only.
"""
    (REPORT_DIR / "super_csv_plan.md").write_text(super_plan, encoding="utf-8")

    next_steps = """# Recommended Next Steps

Decision: **data_inventory_ready_needs_manual_review**

1. Review `source_schema_inventory.csv` rows with `read_error`, unsupported dependencies, or unknown classification.
2. Audit duplicate/variant football-data season files before choosing canonical source copies.
3. Build and review team and league alias tables before cross-source joins.
4. Document odds timing for football-data, Footiqo, Beat The Bookie, and any processed odds exports.
5. Decide competition codes beyond the seeded registry, then lock `competition_registry.csv`.
6. Build a small canonical registry prototype for one league/season only, including `source_match_map`.
7. Validate deterministic match sequencing and int64 ID formatting.
8. Only after manual review, build market-specific super CSVs from canonical IDs and lag-verified feature blocks.

No confirmed edge should be claimed from this inventory.
"""
    (REPORT_DIR / "recommended_next_steps.md").write_text(next_steps, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    files = discover_files()

    file_rows = []
    schema_rows = []
    class_rows = []
    leakage_rows = []

    for path in files:
        stat = path.stat()
        ext = extension(path)
        frow = {
            "path": rel(path),
            "file_name": path.name,
            "extension": ext,
            "size_mb": round(stat.st_size / (1024 * 1024), 6),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "data_stage": stage(path),
            "likely_source": likely_source(path),
        }
        file_rows.append(frow)

        inspected = inspect_file(path)
        cols = inspected.pop("columns", [])
        srow = {
            **frow,
            "read_status": inspected.get("read_status", ""),
            "row_count": inspected.get("row_count", ""),
            "column_count": inspected.get("column_count", ""),
            "first_date": inspected.get("first_date", ""),
            "last_date": inspected.get("last_date", ""),
            "columns_sample": safe_json(cols[:200]),
            "leagues": league_values(cols, path),
            "missingness_summary": safe_json(inspected.get("missingness_summary", {})),
            "duplicate_candidate_ids": safe_json(inspected.get("duplicate_candidate_ids", {})),
        }
        for key in [
            "candidate_date_columns",
            "candidate_league_columns",
            "candidate_season_columns",
            "candidate_home_team_columns",
            "candidate_away_team_columns",
            "candidate_score_result_columns",
            "candidate_odds_columns",
            "candidate_stats_columns",
            "candidate_player_team_valuation_columns",
            "candidate_referee_columns",
        ]:
            srow[key] = safe_json(inspected.get(key, []))
        if "sqlite_tables" in inspected:
            srow["sqlite_tables"] = safe_json(inspected["sqlite_tables"])
        else:
            srow["sqlite_tables"] = ""
        schema_rows.append(srow)

        labels = classify_source(path, cols)
        risks = classify_leakage(path, cols, labels)
        class_rows.append(
            {
                "path": rel(path),
                "likely_source": frow["likely_source"],
                "source_classes": "; ".join(labels),
                "classification_notes": "Heuristic inventory classification; requires manual review before canonical use.",
            }
        )
        leakage_rows.append(
            {
                "path": rel(path),
                "likely_source": frow["likely_source"],
                "leakage_risks": "; ".join(risks),
                "leakage_notes": "Conservative heuristic. Treat odds timing and point-in-time availability as unconfirmed until documented.",
            }
        )

    market_rows = market_matrix(schema_rows, class_rows)

    pd.DataFrame(file_rows).to_csv(REPORT_DIR / "full_file_inventory.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(REPORT_DIR / "source_schema_inventory.csv", index=False)
    pd.DataFrame(class_rows).to_csv(REPORT_DIR / "source_classification.csv", index=False)
    pd.DataFrame(leakage_rows).to_csv(REPORT_DIR / "leakage_risk_inventory.csv", index=False)
    pd.DataFrame(market_rows).to_csv(REPORT_DIR / "market_availability_matrix.csv", index=False)
    write_competition_registry(schema_rows)
    markdown_reports(file_rows, schema_rows, class_rows, leakage_rows, market_rows)


if __name__ == "__main__":
    main()
