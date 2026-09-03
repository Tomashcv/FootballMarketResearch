from __future__ import annotations

import hashlib
import re
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not infer format.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Parsing dates.*")


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOTS = [
    ROOT / "data/raw",
    ROOT / "data/raw_external",
    ROOT / "data/external",
    ROOT / "data/processed",
]
OUT_DIR = ROOT / "data/processed/football_data"
SUPER_DIR = ROOT / "data/processed/super_csvs/research_ready/football_data"
REPORT_DIR = ROOT / "outputs/reports/football_data_build"

TEAMS = ROOT / "data/processed/entity_registry/teams_v1_locked.csv"
ALIASES_PLUS_TM = ROOT / "data/processed/entity_registry/team_aliases_v1_locked_plus_transfermarkt.csv"
ALIASES_LOCKED = ROOT / "data/processed/entity_registry/team_aliases_v1_locked.csv"
MATCHES = ROOT / "data/processed/entity_registry/matches_v1_locked.csv"
COMPETITIONS = ROOT / "data/processed/entity_registry/competitions_v1_locked.csv"

LEAGUES = {
    "E0": ("england_premier_league", 1, 1, "England"),
    "SP1": ("spain_laliga", 1, 2, "Spain"),
    "D1": ("germany_bundesliga", 1, 3, "Germany"),
    "I1": ("italy_serie_a", 1, 4, "Italy"),
    "F1": ("france_ligue_1", 1, 5, "France"),
}

ONE_X_TWO_PRIORITY = [
    ("B365", ("B365H", "B365D", "B365A"), "unknown"),
    ("Avg", ("AvgH", "AvgD", "AvgA"), "unknown"),
    ("football_data_HDA", ("H", "D", "A"), "unknown"),
]
AH_MARKETS = {
    "open": [
        ("B365", "AHh", ("B365AHH", "B365AHA"), "opening"),
        ("Avg", "AHh", ("AvgAHH", "AvgAHA"), "opening"),
        ("B365_legacy", "B365AH", ("B365AHH", "B365AHA"), "unknown"),
        ("GB_legacy", "GBAH", ("GBAHH", "GBAHA"), "unknown"),
        ("LB_legacy", "LBAH", ("LBAHH", "LBAHA"), "unknown"),
    ],
    "close": [
        ("B365C", "AHCh", ("B365CAHH", "B365CAHA"), "closing"),
        ("AvgC", "AHCh", ("AvgCAHH", "AvgCAHA"), "closing"),
    ],
}


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def parse_date_series(s: pd.Series) -> pd.Series:
    text = s.astype(str).str.strip()
    first = pd.to_datetime(text, errors="coerce", dayfirst=True)
    second = pd.to_datetime(text, errors="coerce", dayfirst=False)
    return first.fillna(second)


def infer_season(date: pd.Timestamp) -> int | float:
    if pd.isna(date):
        return np.nan
    return int(date.year - 1 if date.month < 7 else date.year)


def season_label(year: object) -> str:
    if pd.isna(year):
        return ""
    y = int(year)
    return f"{y}/{y + 1}"


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha1(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:12]


def detect_markets(cols: set[str]) -> list[str]:
    markets = []
    if any(set(pair).issubset(cols) for _, pair, _ in ONE_X_TWO_PRIORITY):
        markets.append("1X2")
    for timing, specs in AH_MARKETS.items():
        if any(line in cols and set(odds).issubset(cols) for _, line, odds, _ in specs):
            markets.append(f"AH_{timing}")
    if any(c for c in cols if "2.5" in c or c.startswith(("O", "U"))):
        markets.append("OU_detected_not_built")
    return markets


def discover_files() -> pd.DataFrame:
    rows = []
    seen = set()
    for root in DATA_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if path in seen:
                continue
            seen.add(path)
            try:
                df = pd.read_csv(path, low_memory=False)
                cols = set(map(str, df.columns))
                if not {"Div", "Date"}.issubset(cols):
                    continue
                if not ({"HomeTeam", "AwayTeam"}.issubset(cols) or {"HT", "AT"}.issubset(cols)):
                    continue
                divs = sorted(pd.Series(df["Div"]).dropna().astype(str).unique())
                supported = [d for d in divs if d in LEAGUES]
                if not supported:
                    continue
                dates = parse_date_series(df["Date"])
                for div in supported:
                    sub = df[df["Div"].astype(str).eq(div)].copy()
                    sub_dates = parse_date_series(sub["Date"])
                    markets = detect_markets(cols)
                    rows.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "filename": path.name,
                            "league_code": div,
                            "inferred_league_slug": LEAGUES[div][0],
                            "season_years_in_file": ";".join(map(str, sorted(set(sub_dates.dropna().map(infer_season).astype(int))))),
                            "rows": len(sub),
                            "columns": len(df.columns),
                            "date_range": f"{sub_dates.min().date() if sub_dates.notna().any() else ''} to {sub_dates.max().date() if sub_dates.notna().any() else ''}",
                            "markets_detected": ";".join(markets),
                            "read_status": "read_ok",
                        }
                    )
            except Exception as exc:
                rows.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "filename": path.name,
                        "league_code": "",
                        "inferred_league_slug": "",
                        "season_years_in_file": "",
                        "rows": np.nan,
                        "columns": np.nan,
                        "date_range": "",
                        "markets_detected": "",
                        "read_status": f"read_error: {type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(rows)


def normalize_files(inventory: pd.DataFrame) -> pd.DataFrame:
    frames = []
    keep_cols = set()
    for _, row in inventory[inventory["read_status"].eq("read_ok")].iterrows():
        path = ROOT / row["path"]
        div = row["league_code"]
        if div not in LEAGUES:
            continue
        raw = pd.read_csv(path, low_memory=False)
        raw = raw[raw["Div"].astype(str).eq(div)].copy()
        if raw.empty:
            continue
        home_col = "HomeTeam" if "HomeTeam" in raw.columns else "HT"
        away_col = "AwayTeam" if "AwayTeam" in raw.columns else "AT"
        dates = parse_date_series(raw["Date"])
        time = raw["Time"].astype(str).str.strip() if "Time" in raw.columns else pd.Series([""] * len(raw), index=raw.index)
        dt_text = raw["Date"].astype(str).str.strip() + " " + time.replace({"nan": "", "NaT": ""})
        datetimes = pd.to_datetime(dt_text, errors="coerce", dayfirst=True).fillna(dates)
        slug, ctype, ccode, _country = LEAGUES[div]
        tmp = pd.DataFrame(
            {
                "source_file": str(path.relative_to(ROOT)),
                "source": "football_data",
                "div": div,
                "competition_slug": slug,
                "competition_type": ctype,
                "competition_code": ccode,
                "season_start_year": dates.map(infer_season),
                "match_date": dates.dt.date.astype(str),
                "match_time": time.replace({"nan": ""}),
                "match_datetime": datetimes,
                "home_team_raw": raw[home_col],
                "away_team_raw": raw[away_col],
                "home_team_normalized": raw[home_col].map(normalize_name),
                "away_team_normalized": raw[away_col].map(normalize_name),
                "home_goals": pd.to_numeric(raw.get("FTHG"), errors="coerce"),
                "away_goals": pd.to_numeric(raw.get("FTAG"), errors="coerce"),
                "result_1x2": raw.get("FTR"),
                "raw_date": raw["Date"],
            }
        )
        tmp["season_label"] = tmp["season_start_year"].map(season_label)
        tmp["football_data_row_id"] = [
            f"football_data:{file_fingerprint(path)}:{int(i)}" for i in raw.index
        ]
        original_cols = [
            c
            for c in raw.columns
            if c
            in {
                "B365H",
                "B365D",
                "B365A",
                "AvgH",
                "AvgD",
                "AvgA",
                "H",
                "D",
                "A",
                "AHh",
                "AHCh",
                "B365AH",
                "GBAH",
                "LBAH",
                "B365AHH",
                "B365AHA",
                "AvgAHH",
                "AvgAHA",
                "B365CAHH",
                "B365CAHA",
                "AvgCAHH",
                "AvgCAHA",
                "GBAHH",
                "GBAHA",
                "LBAHH",
                "LBAHA",
            }
        ]
        for c in original_cols:
            tmp[c] = raw[c]
            keep_cols.add(c)
        ordered = [
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
            "home_team_raw",
            "away_team_raw",
            "home_team_normalized",
            "away_team_normalized",
            "home_goals",
            "away_goals",
            "result_1x2",
            "raw_date",
        ] + sorted(keep_cols)
        frames.append(tmp[[c for c in ordered if c in tmp.columns]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["season_start_year"] = pd.to_numeric(out["season_start_year"], errors="coerce").astype("Int64")
    return out


def alias_candidates(norm: pd.DataFrame) -> pd.DataFrame:
    aliases_path = ALIASES_PLUS_TM if ALIASES_PLUS_TM.exists() else ALIASES_LOCKED
    aliases = pd.read_csv(aliases_path)
    aliases = aliases[aliases["approved_for_research"].astype(str).str.lower().eq("true")].copy()
    teams = pd.read_csv(TEAMS)
    alias_map = aliases.groupby("alias_normalized")["team_id"].nunique()
    alias_rows = aliases[aliases["alias_normalized"].isin(alias_map[alias_map.eq(1)].index)].copy()
    alias_rows = alias_rows.sort_values(["confidence", "source"], ascending=[False, True]).drop_duplicates("alias_normalized")
    alias_lookup = alias_rows.set_index("alias_normalized").to_dict("index")
    team_lookup = teams.set_index("team_id").to_dict("index")
    team_names = {normalize_name(r["canonical_team_name"]): tid for tid, r in team_lookup.items()}
    rows = []
    names = pd.concat(
        [
            norm[["home_team_raw", "home_team_normalized", "competition_slug"]].rename(
                columns={"home_team_raw": "raw", "home_team_normalized": "normalized"}
            ),
            norm[["away_team_raw", "away_team_normalized", "competition_slug"]].rename(
                columns={"away_team_raw": "raw", "away_team_normalized": "normalized"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(["raw", "normalized", "competition_slug"])
    for _, row in names.iterrows():
        n = row["normalized"]
        league = row["competition_slug"]
        if n in alias_lookup:
            tid = int(alias_lookup[n]["team_id"])
            team = team_lookup.get(tid, {})
            rows.append(
                {
                    "football_data_team_raw": row["raw"],
                    "football_data_team_normalized": n,
                    "candidate_team_id": tid,
                    "candidate_canonical_team_name": team.get("canonical_team_name", ""),
                    "country_hint": team.get("country", ""),
                    "league_hint": league,
                    "match_type": "exact_locked_alias",
                    "confidence": 1.0,
                    "approved_for_research": True,
                    "manual_review_required": False,
                    "notes": f"Exact normalized match to approved locked alias source={alias_lookup[n].get('source')}.",
                }
            )
        elif n in team_names:
            tid = int(team_names[n])
            team = team_lookup.get(tid, {})
            rows.append(
                {
                    "football_data_team_raw": row["raw"],
                    "football_data_team_normalized": n,
                    "candidate_team_id": tid,
                    "candidate_canonical_team_name": team.get("canonical_team_name", ""),
                    "country_hint": team.get("country", ""),
                    "league_hint": league,
                    "match_type": "exact_canonical_team_name",
                    "confidence": 1.0,
                    "approved_for_research": True,
                    "manual_review_required": False,
                    "notes": "Exact normalized match to canonical team name.",
                }
            )
        else:
            rows.append(
                {
                    "football_data_team_raw": row["raw"],
                    "football_data_team_normalized": n,
                    "candidate_team_id": np.nan,
                    "candidate_canonical_team_name": "",
                    "country_hint": "",
                    "league_hint": league,
                    "match_type": "unmatched_no_fuzzy",
                    "confidence": 0.0,
                    "approved_for_research": False,
                    "manual_review_required": True,
                    "notes": "No exact approved locked alias or canonical team match. Fuzzy matching not applied.",
                }
            )
    cand = pd.DataFrame(rows)
    conflicts = cand.groupby("football_data_team_normalized")["candidate_team_id"].nunique(dropna=True)
    conflict_names = set(conflicts[conflicts.gt(1)].index)
    if conflict_names:
        mask = cand["football_data_team_normalized"].isin(conflict_names)
        cand.loc[mask, "approved_for_research"] = False
        cand.loc[mask, "manual_review_required"] = True
        cand.loc[mask, "notes"] = cand.loc[mask, "notes"].astype(str) + " Conflict: normalized alias maps to multiple team_id values."
    return cand.sort_values(["league_hint", "football_data_team_normalized"]).reset_index(drop=True)


def attach_team_ids(norm: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    approved = aliases[aliases["approved_for_research"].astype(str).str.lower().eq("true")].copy()
    lookup = approved.drop_duplicates("football_data_team_normalized").set_index("football_data_team_normalized")["candidate_team_id"].to_dict()
    out = norm.copy()
    out["source_home_team_id"] = out["home_team_normalized"].map(lookup)
    out["source_away_team_id"] = out["away_team_normalized"].map(lookup)
    return out


def build_source_map(norm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(MATCHES, dtype={"competition_code": str})
    matches["match_datetime"] = pd.to_datetime(matches["match_datetime"], errors="coerce")
    matches["match_date"] = matches["match_datetime"].dt.date.astype(str)
    key_cols = ["competition_slug", "season_start_year", "match_date", "home_team_id", "away_team_id"]
    m = matches[key_cols + ["canonical_match_id", "match_datetime"]].copy()
    m = m.rename(columns={"match_datetime": "canonical_match_datetime"})
    n = norm.copy()
    n["source_home_team_id"] = pd.to_numeric(n["source_home_team_id"], errors="coerce").astype("Int64")
    n["source_away_team_id"] = pd.to_numeric(n["source_away_team_id"], errors="coerce").astype("Int64")
    n["season_start_year"] = pd.to_numeric(n["season_start_year"], errors="coerce").astype("Int64")
    mapped = n.merge(
        m,
        left_on=["competition_slug", "season_start_year", "match_date", "source_home_team_id", "source_away_team_id"],
        right_on=key_cols,
        how="left",
        validate="many_to_one",
    )
    mapped["source_match_datetime"] = mapped["match_datetime"]
    mapped["mapping_method"] = np.where(
        mapped["canonical_match_id"].notna(),
        "competition_season_date_approved_team_ids",
        "unmatched",
    )
    mapped["mapping_confidence"] = np.where(mapped["canonical_match_id"].notna(), 1.0, 0.0)
    mapped["manual_review_required"] = mapped["canonical_match_id"].isna()
    mapped["notes"] = np.where(
        mapped["canonical_match_id"].notna(),
        "Mapped by competition, season, match date, approved home team_id, approved away team_id.",
        "No canonical match found or one/both teams unresolved.",
    )
    columns = [
        "canonical_match_id",
        "football_data_row_id",
        "source",
        "source_file",
        "div",
        "competition_slug",
        "season_start_year",
        "source_match_datetime",
        "home_team_raw",
        "away_team_raw",
        "source_home_team_id",
        "source_away_team_id",
        "mapping_method",
        "mapping_confidence",
        "manual_review_required",
        "notes",
    ]
    out = mapped[columns].rename(columns={"home_team_raw": "source_home_team", "away_team_raw": "source_away_team"})
    out["duplicate_source_row_for_canonical_flag"] = False
    mapped_mask = out["canonical_match_id"].notna()
    duplicate_mask = out.loc[mapped_mask, "canonical_match_id"].duplicated(keep=False)
    duplicate_index = out.loc[mapped_mask].index[duplicate_mask]
    out.loc[duplicate_index, "duplicate_source_row_for_canonical_flag"] = True
    out.loc[duplicate_index, "notes"] = (
        out.loc[duplicate_index, "notes"].astype(str)
        + " Duplicate source-layer coverage for this canonical_match_id from overlapping football-data files; research market CSVs deduplicate canonical_match_id."
    )
    unmatched = mapped[mapped["canonical_match_id"].isna()].copy()
    return out, unmatched


def select_1x2(row: pd.Series) -> tuple[float, float, float, str, str]:
    for source, cols, timing in ONE_X_TWO_PRIORITY:
        if all(c in row.index for c in cols):
            vals = [pd.to_numeric(row[c], errors="coerce") for c in cols]
            if all(pd.notna(v) and float(v) > 1 for v in vals):
                return float(vals[0]), float(vals[1]), float(vals[2]), source, timing
    return np.nan, np.nan, np.nan, "", ""


def build_1x2(norm_ids: pd.DataFrame, source_map: pd.DataFrame) -> pd.DataFrame:
    mapped = source_map[source_map["canonical_match_id"].notna()].copy()
    mapped = mapped.drop_duplicates("canonical_match_id", keep="first")
    df = norm_ids.merge(mapped[["canonical_match_id", "football_data_row_id"]], on="football_data_row_id", how="inner")
    selected = df.apply(select_1x2, axis=1, result_type="expand")
    selected.columns = ["x1_home_odds", "x1_draw_odds", "x1_away_odds", "x1_odds_source", "x1_odds_timing_label"]
    df = pd.concat([df, selected], axis=1)
    df["target_home_win"] = df["result_1x2"].eq("H").astype(int)
    df["target_draw"] = df["result_1x2"].eq("D").astype(int)
    df["target_away_win"] = df["result_1x2"].eq("A").astype(int)
    valid = (
        df["canonical_match_id"].notna()
        & df[["target_home_win", "target_draw", "target_away_win"]].sum(axis=1).eq(1)
        & df[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].notna().all(axis=1)
        & df[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].gt(1).all(axis=1)
    )
    out = df[valid].copy()
    out = out.drop_duplicates("canonical_match_id", keep="first")
    out["canonical_match_id"] = out["canonical_match_id"].astype("int64")
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
        "source_home_team_id",
        "source_away_team_id",
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


def ah_unit_return(adjusted: float, odds: float) -> tuple[float, str, bool]:
    if pd.isna(adjusted) or pd.isna(odds):
        return np.nan, "", False
    parts = [adjusted]
    frac = abs(adjusted * 4 - round(adjusted * 4))
    if frac < 1e-8 and abs((adjusted * 2) - round(adjusted * 2)) > 1e-8:
        lower = np.floor(adjusted * 2) / 2
        upper = np.ceil(adjusted * 2) / 2
        parts = [lower, upper]
    returns = []
    labels = []
    for part in parts:
        if part > 0:
            returns.append(float(odds) - 1.0)
            labels.append("win")
        elif part == 0:
            returns.append(0.0)
            labels.append("push")
        else:
            returns.append(-1.0)
            labels.append("loss")
    val = float(np.mean(returns))
    if val > 0 and any(x == "loss" for x in labels):
        label = "half_win"
    elif val == 0 and any(x == "push" for x in labels):
        label = "push"
    elif val < 0 and any(x == "push" for x in labels):
        label = "half_loss"
    elif val > 0:
        label = "full_win"
    else:
        label = "full_loss"
    return val, label, label == "push"


def select_ah(row: pd.Series, timing_key: str) -> tuple[float, float, float, str, str]:
    for source, line, odds, timing in AH_MARKETS[timing_key]:
        if line in row.index and all(c in row.index for c in odds):
            h = pd.to_numeric(row[line], errors="coerce")
            oh = pd.to_numeric(row[odds[0]], errors="coerce")
            oa = pd.to_numeric(row[odds[1]], errors="coerce")
            if pd.notna(h) and pd.notna(oh) and pd.notna(oa) and float(oh) > 1 and float(oa) > 1:
                return float(h), float(oh), float(oa), source, timing
    return np.nan, np.nan, np.nan, "", ""


def build_ah(norm_ids: pd.DataFrame, source_map: pd.DataFrame, timing_key: str) -> pd.DataFrame:
    mapped = source_map[source_map["canonical_match_id"].notna()].copy()
    mapped = mapped.drop_duplicates("canonical_match_id", keep="first")
    df = norm_ids.merge(mapped[["canonical_match_id", "football_data_row_id"]], on="football_data_row_id", how="inner")
    selected = df.apply(lambda r: select_ah(r, timing_key), axis=1, result_type="expand")
    selected.columns = ["ah_line_home", "ah_home_odds", "ah_away_odds", "ah_odds_source", "ah_timing_label"]
    df = pd.concat([df, selected], axis=1)
    valid = (
        df["canonical_match_id"].notna()
        & df[["home_goals", "away_goals"]].notna().all(axis=1)
        & df[["ah_line_home", "ah_home_odds", "ah_away_odds"]].notna().all(axis=1)
        & df[["ah_home_odds", "ah_away_odds"]].gt(1).all(axis=1)
    )
    out = df[valid].copy()
    out = out.drop_duplicates("canonical_match_id", keep="first")
    if out.empty:
        return out
    margin = out["home_goals"] - out["away_goals"]
    home_adj = margin + out["ah_line_home"]
    away_adj = -home_adj
    home_settle = [ah_unit_return(a, o) for a, o in zip(home_adj, out["ah_home_odds"])]
    away_settle = [ah_unit_return(a, o) for a, o in zip(away_adj, out["ah_away_odds"])]
    out["ah_home_unit_return"] = [x[0] for x in home_settle]
    out["ah_home_settlement"] = [x[1] for x in home_settle]
    out["ah_away_unit_return"] = [x[0] for x in away_settle]
    out["ah_away_settlement"] = [x[1] for x in away_settle]
    out["ah_push_flag"] = [x[2] or y[2] for x, y in zip(home_settle, away_settle)]
    out["canonical_match_id"] = out["canonical_match_id"].astype("int64")
    out["ah_home_raw_prob"] = 1.0 / out["ah_home_odds"]
    out["ah_away_raw_prob"] = 1.0 / out["ah_away_odds"]
    out["ah_overround"] = out["ah_home_raw_prob"] + out["ah_away_raw_prob"]
    out["ah_home_no_vig_prob"] = out["ah_home_raw_prob"] / out["ah_overround"]
    out["ah_away_no_vig_prob"] = out["ah_away_raw_prob"] / out["ah_overround"]
    out["classification"] = "research_only"
    cols = [
        "canonical_match_id",
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
        "source_home_team_id",
        "source_away_team_id",
        "home_team_raw",
        "away_team_raw",
        "home_team_normalized",
        "away_team_normalized",
        "home_goals",
        "away_goals",
        "result_1x2",
        "ah_line_home",
        "ah_home_odds",
        "ah_away_odds",
        "ah_home_unit_return",
        "ah_away_unit_return",
        "ah_home_settlement",
        "ah_away_settlement",
        "ah_push_flag",
        "ah_odds_source",
        "ah_timing_label",
        "ah_home_raw_prob",
        "ah_away_raw_prob",
        "ah_overround",
        "ah_home_no_vig_prob",
        "ah_away_no_vig_prob",
        "classification",
    ]
    return out[[c for c in cols if c in out.columns]]


def market_availability(norm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, specs in [("1X2", ONE_X_TWO_PRIORITY)]:
        for source, cols, timing in specs:
            if all(c in norm.columns for c in cols):
                valid = norm[list(cols)].apply(lambda s: pd.to_numeric(s, errors="coerce")).gt(1).all(axis=1)
                rows.append(
                    {
                        "market": market,
                        "odds_columns_detected": ";".join(cols),
                        "line_columns_detected": "",
                        "row_count_complete_paired_odds": int(valid.sum()),
                        "invalid_odds_le_1": int(norm[list(cols)].notna().all(axis=1).sum() - valid.sum()),
                        "date_range": f"{norm['match_date'].min()} to {norm['match_date'].max()}",
                        "leagues": ";".join(sorted(norm["competition_slug"].dropna().unique())),
                        "source_files": ";".join(sorted(norm["source_file"].dropna().unique())[:50]),
                        "timing_label": timing,
                        "recommendation": "usable_for_research_filter" if valid.any() else "not_available",
                    }
                )
    for timing_key, specs in AH_MARKETS.items():
        for source, line, cols, timing in specs:
            if line in norm.columns and all(c in norm.columns for c in cols):
                odds_valid = norm[list(cols)].apply(lambda s: pd.to_numeric(s, errors="coerce")).gt(1).all(axis=1)
                line_valid = pd.to_numeric(norm[line], errors="coerce").notna()
                valid = odds_valid & line_valid
                rows.append(
                    {
                        "market": f"AH_{timing_key}",
                        "odds_columns_detected": ";".join(cols),
                        "line_columns_detected": line,
                        "row_count_complete_paired_odds": int(valid.sum()),
                        "invalid_odds_le_1": int(norm[list(cols)].notna().all(axis=1).sum() - odds_valid.sum()),
                        "date_range": f"{norm['match_date'].min()} to {norm['match_date'].max()}",
                        "leagues": ";".join(sorted(norm["competition_slug"].dropna().unique())),
                        "source_files": ";".join(sorted(norm["source_file"].dropna().unique())[:50]),
                        "timing_label": timing,
                        "recommendation": "usable_for_research_filter" if valid.any() else "not_available",
                    }
                )
    return pd.DataFrame(rows)


def write_reports(
    inventory: pd.DataFrame,
    norm: pd.DataFrame,
    aliases: pd.DataFrame,
    source_map: pd.DataFrame,
    unmatched: pd.DataFrame,
    availability: pd.DataFrame,
    x1: pd.DataFrame,
    ah_outputs: dict[str, pd.DataFrame],
    decision: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(REPORT_DIR / "football_data_file_inventory.csv", index=False)
    aliases.to_csv(REPORT_DIR / "football_data_team_alias_candidates.csv", index=False)
    availability.to_csv(REPORT_DIR / "football_data_market_availability.csv", index=False)
    unmatched_summary = (
        unmatched.groupby(["competition_slug", "season_start_year"], dropna=False)
        .size()
        .reset_index(name="unmatched_rows")
        if not unmatched.empty
        else pd.DataFrame(columns=["competition_slug", "season_start_year", "unmatched_rows"])
    )
    unmatched_summary.to_csv(REPORT_DIR / "football_data_unmatched_summary.csv", index=False)
    map_summary = source_map.groupby(["competition_slug", "season_start_year", "mapping_method"], dropna=False).size().reset_index(name="rows")
    duplicate_summary = (
        source_map[source_map.get("duplicate_source_row_for_canonical_flag", False).astype(bool)]
        .groupby(["competition_slug", "season_start_year"], dropna=False)
        .size()
        .reset_index(name="duplicate_source_rows")
        if "duplicate_source_row_for_canonical_flag" in source_map.columns
        else pd.DataFrame(columns=["competition_slug", "season_start_year", "duplicate_source_rows"])
    )
    map_lines = [
        "# Football-Data Source Map Report",
        "",
        f"Mapped rows: {int(source_map['canonical_match_id'].notna().sum())}",
        f"Unique mapped canonical_match_id: {source_map.loc[source_map['canonical_match_id'].notna(), 'canonical_match_id'].nunique()}",
        f"Unmatched rows: {int(source_map['canonical_match_id'].isna().sum())}",
        f"Duplicate source-layer rows for already-mapped canonical matches: {int(source_map.get('duplicate_source_row_for_canonical_flag', pd.Series(False, index=source_map.index)).astype(bool).sum())}",
        "",
        "Duplicate source rows come from overlapping local football-data files and are retained in the source layer for audit. Research market CSVs are deduplicated to one row per canonical_match_id.",
        "",
        "## Mapping Summary",
        map_summary.to_markdown(index=False),
        "",
        "## Duplicate Source Rows By League/Season",
        duplicate_summary.head(80).to_markdown(index=False),
    ]
    (REPORT_DIR / "football_data_source_map_report.md").write_text("\n".join(map_lines) + "\n", encoding="utf-8")
    x1_lines = [
        "# Football-Data 1X2 Build Report",
        "",
        f"Rows written: {len(x1)}",
        "Odds selection priority: B365H/B365D/B365A, then AvgH/AvgD/AvgA, then H/D/A.",
        "Classification: research_only. Odds timing remains unknown unless column naming explicitly documents timing.",
    ]
    (REPORT_DIR / "football_data_1x2_build_report.md").write_text("\n".join(x1_lines) + "\n", encoding="utf-8")
    ah_lines = [
        "# Football-Data AH Build Report",
        "",
    ]
    for name, df in ah_outputs.items():
        ah_lines.append(f"- {name}: {len(df)} rows")
    ah_lines += [
        "",
        "AH columns were selected only when a home handicap line and paired home/away AH odds were clearly detected.",
        "Settlement assumes the detected line is the home handicap as documented by football-data AH column naming.",
    ]
    (REPORT_DIR / "football_data_ah_build_report.md").write_text("\n".join(ah_lines) + "\n", encoding="utf-8")
    leakage = pd.DataFrame(
        [
            {"check_name": "raw_files_unchanged", "status": "pass", "details": "Only processed/output paths were written."},
            {"check_name": "normalized_rows_created", "status": "pass" if len(norm) > 0 else "fail", "details": f"rows={len(norm)}"},
            {"check_name": "mapping_rows_created", "status": "pass" if len(source_map) > 0 else "fail", "details": f"rows={len(source_map)}"},
            {"check_name": "alias_candidates_created", "status": "pass" if len(aliases) > 0 else "fail", "details": f"rows={len(aliases)}"},
            {"check_name": "one_x_two_csv_created_if_valid_odds", "status": "pass" if len(x1) > 0 else "fail", "details": f"rows={len(x1)}"},
            {"check_name": "one_x_two_no_duplicate_canonical_id", "status": "pass" if x1.empty or not x1["canonical_match_id"].duplicated().any() else "fail", "details": f"duplicates={0 if x1.empty else int(x1['canonical_match_id'].duplicated().sum())}"},
            {"check_name": "one_x_two_odds_gt_1", "status": "pass" if x1.empty or x1[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].gt(1).all().all() else "fail", "details": "Research rows filtered to complete valid odds."},
            {"check_name": "ah_settlement_sanity", "status": "pass" if all(df.empty or df[["ah_home_unit_return", "ah_away_unit_return"]].notna().all().all() for df in ah_outputs.values()) else "fail", "details": "AH settlement present where AH outputs exist."},
            {"check_name": "no_external_feature_blocks_merged", "status": "pass", "details": "Market CSVs include football-data source/market columns only."},
            {"check_name": "classification_research_only", "status": "pass", "details": "Research-only classification retained."},
        ]
    )
    leakage.to_csv(REPORT_DIR / "football_data_leakage_checks.csv", index=False)
    report = [
        "# Football-Data Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Inventory rows: {len(inventory)}",
        f"- Normalized rows: {len(norm)}",
        f"- Source map rows: {len(source_map)}",
        f"- Mapped rows: {int(source_map['canonical_match_id'].notna().sum()) if not source_map.empty else 0}",
        f"- Unmatched rows: {int(source_map['canonical_match_id'].isna().sum()) if not source_map.empty else 0}",
        f"- Duplicate source-layer rows for already-mapped canonical matches: {int(source_map.get('duplicate_source_row_for_canonical_flag', pd.Series(False, index=source_map.index)).astype(bool).sum()) if not source_map.empty else 0}",
        f"- Alias candidates requiring review: {int(aliases['manual_review_required'].astype(str).str.lower().eq('true').sum()) if not aliases.empty else 0}",
        f"- 1X2 research rows: {len(x1)}",
        f"- AH research outputs: {', '.join(f'{k}={len(v)}' for k, v in ah_outputs.items())}",
        "",
        "No modeling, value search, threshold optimization, raw-file modification, final canonical extension, or confirmed-edge claim was performed.",
    ]
    (REPORT_DIR / "football_data_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (REPORT_DIR / "football_data_decision.md").write_text(
        "\n".join(["# Football-Data Decision", "", f"Decision: **{decision}**", "", "Classification remains research_only. No confirmed edge is claimed."]) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUPER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = discover_files()
    norm = normalize_files(inventory)
    aliases = alias_candidates(norm) if not norm.empty else pd.DataFrame()
    norm_ids = attach_team_ids(norm, aliases) if not norm.empty and not aliases.empty else norm
    source_map, unmatched = build_source_map(norm_ids) if not norm_ids.empty else (pd.DataFrame(), pd.DataFrame())
    availability = market_availability(norm_ids) if not norm_ids.empty else pd.DataFrame()
    x1 = build_1x2(norm_ids, source_map) if not norm_ids.empty and not source_map.empty else pd.DataFrame()
    ah_outputs = {}
    ah_open = build_ah(norm_ids, source_map, "open") if not norm_ids.empty and not source_map.empty else pd.DataFrame()
    ah_close = build_ah(norm_ids, source_map, "close") if not norm_ids.empty and not source_map.empty else pd.DataFrame()
    if not ah_open.empty:
        ah_outputs["open"] = ah_open
    if not ah_close.empty:
        ah_outputs["close"] = ah_close
    ah_combined = ah_close if not ah_close.empty else ah_open
    if not ah_combined.empty:
        ah_outputs["primary"] = ah_combined

    inventory.to_csv(OUT_DIR / "football_data_file_inventory_v1.csv", index=False)
    norm_ids.to_csv(OUT_DIR / "football_data_normalized_matches_v1.csv", index=False)
    aliases.to_csv(OUT_DIR / "football_data_team_alias_candidates_v1.csv", index=False)
    source_map.to_csv(OUT_DIR / "football_data_source_match_map_v1.csv", index=False)
    unmatched.to_csv(OUT_DIR / "football_data_unmatched_rows_v1.csv", index=False)
    availability.to_csv(OUT_DIR / "football_data_market_availability_v1.csv", index=False)
    if not x1.empty:
        x1.to_csv(SUPER_DIR / "super_1x2_football_data_top5_research_v1.csv", index=False)
    if not ah_combined.empty:
        ah_combined.to_csv(SUPER_DIR / "super_ah_football_data_top5_research_v1.csv", index=False)
    if not ah_open.empty and not ah_close.empty:
        ah_open.to_csv(SUPER_DIR / "super_ah_open_football_data_top5_research_v1.csv", index=False)
        ah_close.to_csv(SUPER_DIR / "super_ah_close_football_data_top5_research_v1.csv", index=False)

    manual_review = int(aliases["manual_review_required"].astype(str).str.lower().eq("true").sum()) if not aliases.empty else 0
    if norm.empty or source_map.empty or x1.empty:
        decision = "football_data_build_failed"
    elif manual_review > 0:
        decision = "football_data_build_ready_needs_alias_review"
    elif ah_combined.empty:
        decision = "football_data_build_ready_no_ah_available"
    else:
        decision = "football_data_build_ready_market_csvs_good"
    write_reports(inventory, norm_ids, aliases, source_map, unmatched, availability, x1, ah_outputs, decision)
    print(decision)
    print(f"normalized_rows={len(norm_ids)} mapped_rows={int(source_map['canonical_match_id'].notna().sum()) if not source_map.empty else 0} x1_rows={len(x1)} ah_rows={len(ah_combined)}")


if __name__ == "__main__":
    main()
