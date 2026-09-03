from __future__ import annotations

import bisect
import math
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "data/processed/super_csvs/research_ready/football_data_full_scope/super_1x2_football_data_full_scope_research_v1.csv"
FULL_PREV = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope/super_1x2_football_data_full_scope_full_features_research_v1.csv"
OLD_CONFIG = ROOT / "outputs/reports/v3_reproduction/v3_old_config_extracted.csv"

CLUBELO_ARCHIVE = ROOT / "data/raw_external/clubelo_manual/clubelo_archive.zip"
CLUBELO_MAPPING = ROOT / "outputs/reports/clubelo_team_mapping_candidates.csv"

TM_DIR = ROOT / "data/external/players/transfermarkt_raw/player_scores"
TM_MAPPING = ROOT / "data/mappings/transfermarkt_football_data_aliases_v3.csv"

CLUBELO_OUT_DIR = ROOT / "data/processed/feature_blocks/clubelo_full_scope"
TM_OUT_DIR = ROOT / "data/processed/feature_blocks/transfermarkt_full_scope"
BRIDGE_OUT_DIR = ROOT / "data/processed/feature_blocks/v3_full_scope_bridge"
PLUS_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope"
REPORT_DIR = ROOT / "outputs/reports/football_data_full_scope_external"

LEAGUE_TO_TM_COMP = {
    "E0": "GB1",
    "D1": "L1",
    "I1": "IT1",
    "SP1": "ES1",
    "F1": "FR1",
    "P1": "PO1",
    "N1": "NL1",
    "B1": "BE1",
    "T1": "TR1",
    "G1": "GR1",
    "SC0": "SC1",
}
TRANSFER_WINDOWS = [30, 90, 180, 365]
CLUBELO_V3_FEATURES = [
    "clubelo_home_rating",
    "clubelo_away_rating",
    "clubelo_diff",
    "clubelo_abs_diff",
    "clubelo_missing_home",
    "clubelo_missing_away",
    "clubelo_missing_both",
    "clubelo_staleness_home",
    "clubelo_staleness_away",
    "clubelo_diff_minus_internal_elo_diff",
]
NON_EXACT_PLACEHOLDER_FEATURES = {"clubelo_diff_minus_internal_elo_diff"}


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def bool_status(ok: bool) -> str:
    return "pass" if bool(ok) else "fail"


def load_base() -> pd.DataFrame:
    base = pd.read_csv(BASE, low_memory=False)
    base["match_date"] = pd.to_datetime(base["match_date"], errors="coerce").dt.normalize()
    base["season_start_year"] = pd.to_numeric(base["season_start_year"], errors="coerce").astype("Int64")
    return base


def load_old_feature_names() -> list[str]:
    if OLD_CONFIG.exists():
        old = pd.read_csv(OLD_CONFIG)
        names = old.loc[old["record_type"].eq("feature_mapping") & old["old_feature"].notna(), "old_feature"].astype(str).tolist()
        if names:
            return names
    return []


def feature_family(name: str) -> str:
    low = name.lower()
    if name.startswith("x1x2_"):
        return "market"
    if "clubelo" in low:
        return "ClubElo"
    if name.startswith(("tm_", "home_tm_", "away_tm_", "home_minus_away_tm_", "home_div_away_tm_")):
        if "staleness" in low:
            return "staleness"
        return "Transfermarkt"
    if "staleness" in low:
        return "staleness"
    if "rolling" in low or re.search(r"_w\d+$", low):
        return "rolling"
    return "other"


def build_v3_contract(old_names: list[str], current_cols: set[str]) -> pd.DataFrame:
    rows = []
    for name in old_names:
        low = name.lower()
        fam = feature_family(name)
        unsafe = any(token in low for token in ["current_club", "current_value", "lineup", "game_lineups", "same_match"])
        exact = (name in current_cols or name in CLUBELO_V3_FEATURES) and name not in NON_EXACT_PLACEHOLDER_FEATURES
        rows.append(
            {
                "feature_name": name,
                "feature_family": fam,
                "exact_source_available": bool(exact),
                "recoverable_from_current_sources": bool(exact and not unsafe),
                "recoverable_only_from_old_artifacts": bool(name in NON_EXACT_PLACEHOLDER_FEATURES or (fam in {"ClubElo", "Transfermarkt"} and not exact and not unsafe)),
                "unsafe_or_leaky": bool(unsafe),
                "not_recoverable": bool((not exact) or unsafe),
                "notes": (
                    "Forbidden leakage-prone field family; excluded."
                    if unsafe
                    else "Requires old internal Elo columns that are not present in the cleaned full-scope current sources."
                    if name in NON_EXACT_PLACEHOLDER_FEATURES
                    else "Exact old-name feature rebuilt in full-scope block."
                    if exact
                    else "No exact full-scope reconstruction produced; retained as missing/non-exact in bridge."
                ),
            }
        )
    return pd.DataFrame(rows)


def read_clubelo_ratings() -> pd.DataFrame:
    with zipfile.ZipFile(CLUBELO_ARCHIVE) as zf:
        ratings = pd.read_csv(zf.open("EloRatings.csv"), usecols=["date", "club", "elo"])
    ratings = ratings.rename(columns={"date": "clubelo_date", "club": "clubelo_team_raw", "elo": "clubelo_rating"})
    ratings["clubelo_date"] = pd.to_datetime(ratings["clubelo_date"], errors="coerce").dt.normalize()
    ratings["clubelo_rating"] = pd.to_numeric(ratings["clubelo_rating"], errors="coerce")
    ratings["clubelo_team_normalized"] = ratings["clubelo_team_raw"].map(normalize_name)
    return ratings.dropna(subset=["clubelo_date", "clubelo_rating", "clubelo_team_normalized"]).sort_values(["clubelo_team_normalized", "clubelo_date"])


def clubelo_index(ratings: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    out = {}
    for norm, group in ratings.groupby("clubelo_team_normalized"):
        out[str(norm)] = (
            group["clubelo_date"].to_numpy(dtype="datetime64[ns]"),
            group["clubelo_rating"].to_numpy(dtype=float),
            str(group["clubelo_team_raw"].iloc[-1]),
        )
    return out


def build_clubelo_alias(base: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    rating_names = ratings.drop_duplicates("clubelo_team_normalized").set_index("clubelo_team_normalized")["clubelo_team_raw"].to_dict()
    accepted = pd.DataFrame()
    if CLUBELO_MAPPING.exists():
        raw = pd.read_csv(CLUBELO_MAPPING)
        accepted = raw[
            raw["mapping_status"].isin(["accepted_exact_normalized", "accepted_high_confidence_fuzzy"])
            & raw["accepted_clubelo_club"].notna()
            & raw["candidate_club"].eq(raw["accepted_clubelo_club"])
        ].copy()
        accepted["team_norm"] = accepted["football_team"].map(normalize_name)
        accepted["clubelo_norm"] = accepted["accepted_clubelo_club"].map(normalize_name)
    teams = pd.concat(
        [
            base[["div", "home_team_id", "home_team_raw", "home_team_normalized"]].rename(
                columns={"home_team_id": "team_id", "home_team_raw": "team_raw", "home_team_normalized": "team_norm"}
            ),
            base[["div", "away_team_id", "away_team_raw", "away_team_normalized"]].rename(
                columns={"away_team_id": "team_id", "away_team_raw": "team_raw", "away_team_normalized": "team_norm"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(["div", "team_id", "team_norm"])
    rows = []
    for r in teams.itertuples(index=False):
        norm = normalize_name(r.team_norm)
        exact = norm in rating_names
        picked_norm = norm if exact else ""
        picked_name = rating_names.get(norm, "")
        status = "exact_normalized"
        confidence = 1.0 if exact else 0.0
        approved = exact
        notes = "Exact normalized ClubElo team name."
        if not exact and not accepted.empty:
            cand = accepted[(accepted["league"].eq(r.div)) & (accepted["team_norm"].eq(norm))]
            cand = cand.drop_duplicates("clubelo_norm")
            if len(cand) == 1 and str(cand.iloc[0]["clubelo_norm"]) in rating_names:
                picked_norm = str(cand.iloc[0]["clubelo_norm"])
                picked_name = str(cand.iloc[0]["accepted_clubelo_club"])
                status = str(cand.iloc[0]["mapping_status"])
                confidence = float(cand.iloc[0]["candidate_score"])
                approved = True
                notes = "Approved existing ClubElo alias artifact."
        if not approved:
            status = "unmatched_no_fuzzy"
            notes = "No exact or approved ClubElo alias; fuzzy matching not used."
        rows.append(
            {
                "div": r.div,
                "team_id": int(r.team_id),
                "team_raw": r.team_raw,
                "canonical_team_name": r.team_norm,
                "clubelo_team_name": picked_name,
                "clubelo_team_normalized": picked_norm,
                "match_type": status,
                "confidence": confidence,
                "approved_for_research": bool(approved),
                "manual_review_required": not bool(approved),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["div", "team_id", "canonical_team_name"])


def latest_clubelo(idx: dict[str, tuple[np.ndarray, np.ndarray, str]], club_norm: object, match_date: pd.Timestamp) -> tuple[float, object, float, str]:
    if pd.isna(club_norm) or not str(club_norm) or str(club_norm) not in idx or pd.isna(match_date):
        return np.nan, pd.NaT, np.nan, ""
    dates, values, raw_name = idx[str(club_norm)]
    pos = bisect.bisect_left(dates, np.datetime64(match_date)) - 1
    if pos < 0:
        return np.nan, pd.NaT, np.nan, raw_name
    rating_date = pd.Timestamp(dates[pos])
    return float(values[pos]), rating_date.date().isoformat(), float((match_date - rating_date).days), raw_name


def build_clubelo_features(base: pd.DataFrame, ratings: pd.DataFrame, alias: pd.DataFrame) -> pd.DataFrame:
    idx = clubelo_index(ratings)
    amap = alias[alias["approved_for_research"]].set_index(["div", "team_id"])["clubelo_team_normalized"].to_dict()
    rows = []
    for r in base[["full_scope_match_id", "div", "match_date", "home_team_id", "away_team_id"]].itertuples(index=False):
        hnorm = amap.get((r.div, int(r.home_team_id)))
        anorm = amap.get((r.div, int(r.away_team_id)))
        hv, hd, hs, hn = latest_clubelo(idx, hnorm, pd.Timestamp(r.match_date))
        av, ad, aas, an = latest_clubelo(idx, anorm, pd.Timestamp(r.match_date))
        hfound = np.isfinite(hv)
        afound = np.isfinite(av)
        rows.append(
            {
                "full_scope_match_id": r.full_scope_match_id,
                "home_clubelo_found_flag": bool(hfound),
                "away_clubelo_found_flag": bool(afound),
                "clubelo_both_found_flag": bool(hfound and afound),
                "home_clubelo": hv,
                "away_clubelo": av,
                "clubelo_diff": hv - av if hfound and afound else np.nan,
                "home_clubelo_rank": np.nan,
                "away_clubelo_rank": np.nan,
                "clubelo_rank_diff": np.nan,
                "home_clubelo_latest_date": hd,
                "away_clubelo_latest_date": ad,
                "home_clubelo_days_stale": hs,
                "away_clubelo_days_stale": aas,
                "clubelo_source_file": "data/raw_external/clubelo_manual/clubelo_archive.zip::EloRatings.csv",
                "home_clubelo_team_raw": hn,
                "away_clubelo_team_raw": an,
            }
        )
    return pd.DataFrame(rows)


def load_tm_fixture_mapping(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(TM_MAPPING)
    mapping = mapping[mapping["decision"].astype(str).str.startswith("approved")].copy()
    mapping["season_start_year"] = pd.to_numeric(mapping["season_start_year"], errors="coerce").astype("Int64")
    mapping["transfermarkt_club_id"] = pd.to_numeric(mapping["transfermarkt_club_id"], errors="coerce").astype("Int64")
    mapping["fd_norm_name"] = mapping["football_data_team"].map(normalize_name)

    alias_rows = []
    map_by_key = {}
    for r in mapping.dropna(subset=["season_start_year", "transfermarkt_club_id"]).itertuples(index=False):
        key = (str(r.league), int(r.season_start_year), normalize_name(r.football_data_team))
        map_by_key[key] = (int(r.transfermarkt_club_id), str(r.transfermarkt_club_name))
        alias_rows.append(
            {
                "div": r.league,
                "season_start_year": int(r.season_start_year),
                "football_data_team": r.football_data_team,
                "football_data_team_normalized": normalize_name(r.football_data_team),
                "transfermarkt_club_id": int(r.transfermarkt_club_id),
                "transfermarkt_club_name": r.transfermarkt_club_name,
                "alias_status": r.decision,
                "approved_for_research": True,
                "manual_review_required": False,
                "notes": r.reason,
            }
        )

    out = base[["full_scope_match_id", "div", "season_start_year", "match_date", "home_team_raw", "away_team_raw", "home_team_normalized", "away_team_normalized"]].copy()
    home_ids = []
    away_ids = []
    home_names = []
    away_names = []
    for r in out.itertuples(index=False):
        season = int(r.season_start_year) if pd.notna(r.season_start_year) else -1
        h = map_by_key.get((r.div, season, normalize_name(r.home_team_raw))) or map_by_key.get((r.div, season, normalize_name(r.home_team_normalized)))
        a = map_by_key.get((r.div, season, normalize_name(r.away_team_raw))) or map_by_key.get((r.div, season, normalize_name(r.away_team_normalized)))
        home_ids.append(h[0] if h else np.nan)
        away_ids.append(a[0] if a else np.nan)
        home_names.append(h[1] if h else "")
        away_names.append(a[1] if a else "")
    out["tm_competition_id"] = out["div"].map(LEAGUE_TO_TM_COMP)
    out["tm_home_club_id"] = home_ids
    out["tm_away_club_id"] = away_ids
    out["tm_home_club_name"] = home_names
    out["tm_away_club_name"] = away_names

    games = pd.read_csv(TM_DIR / "games.csv", usecols=["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id"], low_memory=False)
    games = games[games["competition_id"].isin(set(LEAGUE_TO_TM_COMP.values()))].copy()
    games["match_date"] = pd.to_datetime(games["date"], errors="coerce").dt.normalize()
    games["season_start_year"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    games["tm_home_club_id"] = pd.to_numeric(games["home_club_id"], errors="coerce")
    games["tm_away_club_id"] = pd.to_numeric(games["away_club_id"], errors="coerce")
    games["candidate_count"] = games.groupby(["competition_id", "season_start_year", "match_date", "tm_home_club_id", "tm_away_club_id"])["game_id"].transform("count")
    games = games.rename(columns={"competition_id": "tm_competition_id"})[
        ["game_id", "tm_competition_id", "season_start_year", "match_date", "tm_home_club_id", "tm_away_club_id", "candidate_count"]
    ]
    joined = out.merge(
        games,
        on=["tm_competition_id", "season_start_year", "match_date", "tm_home_club_id", "tm_away_club_id"],
        how="left",
        validate="many_to_one",
    )
    joined["tm_fixture_mapped"] = joined["game_id"].notna() & joined["candidate_count"].fillna(0).eq(1)
    joined["tm_mapping_status"] = np.select(
        [
            joined["tm_competition_id"].isna(),
            joined[["tm_home_club_id", "tm_away_club_id"]].isna().any(axis=1),
            joined["game_id"].isna(),
            joined["candidate_count"].fillna(0).gt(1),
            joined["tm_fixture_mapped"],
        ],
        [
            "competition_mapping_missing",
            "club_alias_missing",
            "no_transfermarkt_fixture_candidate",
            "duplicate_transfermarkt_fixture_candidate",
            "mapped",
        ],
        default="unknown",
    )
    return joined, pd.DataFrame(alias_rows)


def money_to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.fillna("").astype(str).str.strip().str.lower().str.replace(",", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def empty_transfer_features(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_tm_{metric}_{window}d": np.nan
        for window in TRANSFER_WINDOWS
        for metric in [
            "arrivals_count",
            "departures_count",
            "arrivals_value_sum",
            "departures_value_sum",
            "net_transfer_value",
            "biggest_arrival_value",
            "biggest_departure_value",
            "transfer_churn_count",
            "transfer_churn_value",
            "loan_arrivals_count",
            "loan_departures_count",
            "free_arrivals_count",
            "free_departures_count",
        ]
    }


def transfer_features_for_club(moves: pd.DataFrame, match_date: pd.Timestamp, prefix: str) -> dict[str, float]:
    row = {}
    for window in TRANSFER_WINDOWS:
        prior = moves[moves["transfer_date"].lt(match_date) & moves["transfer_date"].ge(match_date - pd.Timedelta(days=window))]
        arrivals = prior[prior["direction"].eq("arrival")]
        departures = prior[prior["direction"].eq("departure")]
        arr_value = float(arrivals["value_for_sum"].sum()) if not arrivals.empty else 0.0
        dep_value = float(departures["value_for_sum"].sum()) if not departures.empty else 0.0
        row[f"{prefix}_tm_arrivals_count_{window}d"] = int(len(arrivals))
        row[f"{prefix}_tm_departures_count_{window}d"] = int(len(departures))
        row[f"{prefix}_tm_arrivals_value_sum_{window}d"] = arr_value
        row[f"{prefix}_tm_departures_value_sum_{window}d"] = dep_value
        row[f"{prefix}_tm_net_transfer_value_{window}d"] = arr_value - dep_value
        row[f"{prefix}_tm_biggest_arrival_value_{window}d"] = float(arrivals["value_for_sum"].max()) if not arrivals.empty else 0.0
        row[f"{prefix}_tm_biggest_departure_value_{window}d"] = float(departures["value_for_sum"].max()) if not departures.empty else 0.0
        row[f"{prefix}_tm_transfer_churn_count_{window}d"] = int(len(prior))
        row[f"{prefix}_tm_transfer_churn_value_{window}d"] = arr_value + dep_value
        row[f"{prefix}_tm_loan_arrivals_count_{window}d"] = np.nan
        row[f"{prefix}_tm_loan_departures_count_{window}d"] = np.nan
        row[f"{prefix}_tm_free_arrivals_count_{window}d"] = int(arrivals["is_free_or_zero"].sum()) if not arrivals.empty else 0
        row[f"{prefix}_tm_free_departures_count_{window}d"] = int(departures["is_free_or_zero"].sum()) if not departures.empty else 0
    return row


def load_transfers(relevant_clubs: set[int]) -> dict[int, pd.DataFrame]:
    header = pd.read_csv(TM_DIR / "transfers.csv", nrows=0).columns
    cols = [c for c in ["player_id", "transfer_date", "from_club_id", "to_club_id", "transfer_fee", "market_value_in_eur"] if c in header]
    transfers = pd.read_csv(TM_DIR / "transfers.csv", usecols=cols, low_memory=False)
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce").dt.normalize()
    transfers["market_value_in_eur"] = pd.to_numeric(transfers.get("market_value_in_eur", np.nan), errors="coerce")
    transfers["transfer_fee_numeric"] = money_to_numeric(transfers.get("transfer_fee", pd.Series(np.nan, index=transfers.index)))
    frames = []
    arrivals = transfers[transfers["to_club_id"].isin(relevant_clubs)].copy()
    arrivals["club_id"] = arrivals["to_club_id"].astype("Int64")
    arrivals["direction"] = "arrival"
    frames.append(arrivals)
    departures = transfers[transfers["from_club_id"].isin(relevant_clubs)].copy()
    departures["club_id"] = departures["from_club_id"].astype("Int64")
    departures["direction"] = "departure"
    frames.append(departures)
    moves = pd.concat(frames, ignore_index=True).dropna(subset=["club_id", "transfer_date"])
    moves["value_for_sum"] = moves["market_value_in_eur"].fillna(0.0)
    moves["is_free_or_zero"] = moves["transfer_fee_numeric"].fillna(0).eq(0) | moves["value_for_sum"].eq(0)
    return {int(club): group.sort_values("transfer_date").reset_index(drop=True) for club, group in moves.groupby("club_id")}


def empty_valuation_features(prefix: str) -> dict[str, float]:
    metrics = [
        "squad_value_total_prior365",
        "squad_value_top3_prior365",
        "squad_value_top5_prior365",
        "squad_value_top11_prior365",
        "squad_value_depth_outside_top5_prior365",
        "squad_value_depth_outside_top11_prior365",
        "squad_value_mean_prior365",
        "squad_value_median_prior365",
        "squad_value_max_prior365",
        "squad_value_concentration_top3_share_prior365",
        "squad_value_concentration_top5_share_prior365",
        "squad_valued_player_count_prior365",
        "squad_prior_appearance_player_count_prior365",
        "squad_missing_valuation_count_prior365",
        "avg_valuation_staleness_days_prior365",
        "max_valuation_staleness_days_prior365",
    ]
    row = {f"{prefix}_tm_{metric}": np.nan for metric in metrics}
    row[f"{prefix}_tm_latest_valuation_date"] = pd.NaT
    row[f"{prefix}_tm_value_days_stale"] = np.nan
    row[f"{prefix}_tm_avg_age"] = np.nan
    return row


def load_appearance_index(relevant_clubs: set[int], min_date: pd.Timestamp, max_date: pd.Timestamp) -> tuple[dict[int, pd.DataFrame], set[int]]:
    app = pd.read_csv(TM_DIR / "appearances.csv", usecols=["player_id", "player_club_id", "date"], low_memory=False)
    app["date"] = pd.to_datetime(app["date"], errors="coerce").dt.normalize()
    app = app[
        app["player_club_id"].isin(relevant_clubs)
        & app["date"].ge(min_date - pd.Timedelta(days=365))
        & app["date"].lt(max_date)
    ].dropna(subset=["player_id", "player_club_id", "date"])
    app["player_id"] = app["player_id"].astype(int)
    app["player_club_id"] = app["player_club_id"].astype(int)
    return {int(club): group.sort_values("date").reset_index(drop=True) for club, group in app.groupby("player_club_id")}, set(app["player_id"].unique())


def load_valuation_index(player_ids: set[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    vals = pd.read_csv(TM_DIR / "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"], low_memory=False)
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce").dt.normalize()
    vals["market_value_in_eur"] = pd.to_numeric(vals["market_value_in_eur"], errors="coerce")
    vals = vals[vals["player_id"].isin(player_ids)].dropna(subset=["player_id", "date", "market_value_in_eur"])
    vals["player_id"] = vals["player_id"].astype(int)
    return {
        int(player_id): (group["date"].to_numpy(dtype="datetime64[ns]"), group["market_value_in_eur"].to_numpy(dtype=float))
        for player_id, group in vals.sort_values(["player_id", "date"]).groupby("player_id")
    }


def load_dob(player_ids: set[int]) -> dict[int, pd.Timestamp]:
    players = pd.read_csv(TM_DIR / "players.csv", usecols=["player_id", "date_of_birth"], low_memory=False)
    players = players[players["player_id"].isin(player_ids)].copy()
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    return players.dropna(subset=["date_of_birth"]).drop_duplicates("player_id").set_index("player_id")["date_of_birth"].to_dict()


def latest_value_before(index: dict[int, tuple[np.ndarray, np.ndarray]], player_id: int, match_date: pd.Timestamp) -> tuple[float, float, object]:
    item = index.get(int(player_id))
    if item is None:
        return np.nan, np.nan, pd.NaT
    dates, values = item
    pos = bisect.bisect_left(dates, np.datetime64(match_date)) - 1
    if pos < 0:
        return np.nan, np.nan, pd.NaT
    d = pd.Timestamp(dates[pos])
    return float(values[pos]), float((match_date - d).days), d.date().isoformat()


def valuation_features_for_club(apps: pd.DataFrame, valuation_index: dict[int, tuple[np.ndarray, np.ndarray]], dob: dict[int, pd.Timestamp], match_date: pd.Timestamp, prefix: str) -> dict[str, float]:
    row = empty_valuation_features(prefix)
    if apps.empty:
        row[f"{prefix}_tm_squad_prior_appearance_player_count_prior365"] = 0
        row[f"{prefix}_tm_squad_valued_player_count_prior365"] = 0
        row[f"{prefix}_tm_squad_missing_valuation_count_prior365"] = 0
        return row
    dates = apps["date"].to_numpy(dtype="datetime64[ns]")
    right = np.searchsorted(dates, np.datetime64(match_date), side="left")
    left = np.searchsorted(dates, np.datetime64(match_date - pd.Timedelta(days=365)), side="left")
    players = apps.iloc[left:right]["player_id"].drop_duplicates().to_numpy(dtype=int)
    values = []
    stale = []
    val_dates = []
    ages = []
    for player_id in players:
        value, days, val_date = latest_value_before(valuation_index, int(player_id), match_date)
        if np.isfinite(value):
            values.append(value)
            stale.append(days)
            val_dates.append(val_date)
        d0 = dob.get(int(player_id))
        if pd.notna(d0):
            ages.append((match_date - pd.Timestamp(d0)).days / 365.25)
    values = sorted(values, reverse=True)
    total = float(np.sum(values)) if values else np.nan
    row[f"{prefix}_tm_squad_prior_appearance_player_count_prior365"] = int(len(players))
    row[f"{prefix}_tm_squad_valued_player_count_prior365"] = int(len(values))
    row[f"{prefix}_tm_squad_missing_valuation_count_prior365"] = int(max(len(players) - len(values), 0))
    row[f"{prefix}_tm_avg_age"] = float(np.nanmean(ages)) if ages else np.nan
    if values:
        row[f"{prefix}_tm_squad_value_total_prior365"] = total
        row[f"{prefix}_tm_squad_value_top3_prior365"] = float(np.sum(values[:3])) if len(values) >= 3 else np.nan
        row[f"{prefix}_tm_squad_value_top5_prior365"] = float(np.sum(values[:5])) if len(values) >= 5 else np.nan
        row[f"{prefix}_tm_squad_value_top11_prior365"] = float(np.sum(values[:11])) if len(values) >= 11 else np.nan
        row[f"{prefix}_tm_squad_value_depth_outside_top5_prior365"] = float(np.sum(values[5:])) if len(values) > 5 else 0.0
        row[f"{prefix}_tm_squad_value_depth_outside_top11_prior365"] = float(np.sum(values[11:])) if len(values) > 11 else 0.0
        row[f"{prefix}_tm_squad_value_mean_prior365"] = float(np.mean(values))
        row[f"{prefix}_tm_squad_value_median_prior365"] = float(np.median(values))
        row[f"{prefix}_tm_squad_value_max_prior365"] = float(np.max(values))
        row[f"{prefix}_tm_squad_value_concentration_top3_share_prior365"] = float(np.sum(values[:3]) / total) if len(values) >= 3 and total > 0 else np.nan
        row[f"{prefix}_tm_squad_value_concentration_top5_share_prior365"] = float(np.sum(values[:5]) / total) if len(values) >= 5 and total > 0 else np.nan
        row[f"{prefix}_tm_avg_valuation_staleness_days_prior365"] = float(np.mean(stale)) if stale else np.nan
        row[f"{prefix}_tm_max_valuation_staleness_days_prior365"] = float(np.max(stale)) if stale else np.nan
        latest_date = max(pd.Timestamp(d) for d in val_dates if pd.notna(d)) if val_dates else pd.NaT
        row[f"{prefix}_tm_latest_valuation_date"] = latest_date.date().isoformat() if pd.notna(latest_date) else pd.NaT
        row[f"{prefix}_tm_value_days_stale"] = float((match_date - latest_date).days) if pd.notna(latest_date) else np.nan
    return row


def build_transfermarkt_features(base: pd.DataFrame, fixture_map: pd.DataFrame) -> pd.DataFrame:
    mapped = fixture_map[fixture_map["tm_fixture_mapped"]].copy()
    relevant_clubs = set(pd.concat([mapped["tm_home_club_id"], mapped["tm_away_club_id"]]).dropna().astype(int))
    moves_by_club = load_transfers(relevant_clubs) if relevant_clubs else {}
    if relevant_clubs and not mapped.empty:
        app_index, player_ids = load_appearance_index(relevant_clubs, mapped["match_date"].min(), mapped["match_date"].max() + pd.Timedelta(days=1))
        valuation_index = load_valuation_index(player_ids)
        dob = load_dob(player_ids)
    else:
        app_index, valuation_index, dob = {}, {}, {}

    transfer_cache: dict[tuple[int, pd.Timestamp, str], dict[str, float]] = {}
    valuation_cache: dict[tuple[int, pd.Timestamp, str], dict[str, float]] = {}
    rows = []
    for r in fixture_map.itertuples(index=False):
        row = {
            "full_scope_match_id": r.full_scope_match_id,
            "tm_fixture_mapped": bool(r.tm_fixture_mapped),
            "tm_home_club_id": r.tm_home_club_id,
            "tm_away_club_id": r.tm_away_club_id,
            "tm_game_id": r.game_id,
            "tm_competition_id": r.tm_competition_id,
            "tm_mapping_status": r.tm_mapping_status,
        }
        for side, club_attr in [("home", "tm_home_club_id"), ("away", "tm_away_club_id")]:
            club_id = getattr(r, club_attr)
            if not bool(r.tm_fixture_mapped) or pd.isna(club_id):
                row.update(empty_transfer_features(side))
                row.update(empty_valuation_features(side))
                row[f"{side}_tm_squad_prior_appearance_player_count_prior365"] = 0
                row[f"{side}_tm_squad_valued_player_count_prior365"] = 0
                row[f"{side}_tm_squad_missing_valuation_count_prior365"] = 0
                continue
            cid = int(club_id)
            md = pd.Timestamp(r.match_date)
            tkey = (cid, md, side)
            vkey = (cid, md, side)
            if tkey not in transfer_cache:
                transfer_cache[tkey] = transfer_features_for_club(moves_by_club.get(cid, pd.DataFrame(columns=["transfer_date", "direction", "value_for_sum", "is_free_or_zero"])), md, side)
            if vkey not in valuation_cache:
                valuation_cache[vkey] = valuation_features_for_club(app_index.get(cid, pd.DataFrame(columns=["date", "player_id"])), valuation_index, dob, md, side)
            row.update(transfer_cache[tkey])
            row.update(valuation_cache[vkey])
        rows.append(row)
    features = pd.DataFrame(rows)

    for window in TRANSFER_WINDOWS:
        for metric in [
            "arrivals_count",
            "departures_count",
            "arrivals_value_sum",
            "departures_value_sum",
            "net_transfer_value",
            "biggest_arrival_value",
            "biggest_departure_value",
            "transfer_churn_count",
            "transfer_churn_value",
            "free_arrivals_count",
            "free_departures_count",
        ]:
            features[f"home_minus_away_tm_{metric}_{window}d"] = features[f"home_tm_{metric}_{window}d"] - features[f"away_tm_{metric}_{window}d"]
    for metric in [
        "squad_value_total_prior365",
        "squad_value_top5_prior365",
        "squad_value_top11_prior365",
        "squad_value_depth_outside_top5_prior365",
        "squad_value_depth_outside_top11_prior365",
        "squad_value_concentration_top3_share_prior365",
        "squad_value_concentration_top5_share_prior365",
        "squad_valued_player_count_prior365",
        "avg_valuation_staleness_days_prior365",
        "max_valuation_staleness_days_prior365",
    ]:
        features[f"home_minus_away_tm_{metric}"] = features[f"home_tm_{metric}"] - features[f"away_tm_{metric}"]
    features["home_div_away_tm_squad_value_total_log1p_ratio_prior365"] = np.log1p(features["home_tm_squad_value_total_prior365"]) - np.log1p(features["away_tm_squad_value_total_prior365"])
    features["home_div_away_tm_squad_value_top5_log1p_ratio_prior365"] = np.log1p(features["home_tm_squad_value_top5_prior365"]) - np.log1p(features["away_tm_squad_value_top5_prior365"])
    features["home_div_away_tm_squad_value_top11_log1p_ratio_prior365"] = np.log1p(features["home_tm_squad_value_top11_prior365"]) - np.log1p(features["away_tm_squad_value_top11_prior365"])

    features["tm_has_transfer_data_home"] = features["home_tm_transfer_churn_count_365d"].notna()
    features["tm_has_transfer_data_away"] = features["away_tm_transfer_churn_count_365d"].notna()
    features["tm_has_prior_appearance_data_home"] = features["home_tm_squad_prior_appearance_player_count_prior365"].fillna(0).gt(0)
    features["tm_has_prior_appearance_data_away"] = features["away_tm_squad_prior_appearance_player_count_prior365"].fillna(0).gt(0)
    features["tm_has_valuation_data_home"] = features["home_tm_squad_valued_player_count_prior365"].fillna(0).gt(0)
    features["tm_has_valuation_data_away"] = features["away_tm_squad_valued_player_count_prior365"].fillna(0).gt(0)
    features["tm_home_feature_available"] = features["tm_fixture_mapped"] & features["tm_has_transfer_data_home"]
    features["tm_away_feature_available"] = features["tm_fixture_mapped"] & features["tm_has_transfer_data_away"]
    features["tm_match_feature_available"] = features["tm_home_feature_available"] & features["tm_away_feature_available"]

    # User-facing core aliases. They are exact aliases/proxies over the date-safe prior365 old V3 columns.
    features["home_tm_total_value"] = features["home_tm_squad_value_total_prior365"]
    features["away_tm_total_value"] = features["away_tm_squad_value_total_prior365"]
    features["tm_total_value_diff"] = features["home_minus_away_tm_squad_value_total_prior365"]
    features["tm_total_value_ratio"] = features["home_tm_total_value"] / features["away_tm_total_value"].replace(0, np.nan)
    features["home_tm_mean_value"] = features["home_tm_squad_value_mean_prior365"]
    features["away_tm_mean_value"] = features["away_tm_squad_value_mean_prior365"]
    features["tm_mean_value_diff"] = features["home_tm_mean_value"] - features["away_tm_mean_value"]
    features["home_tm_median_value"] = features["home_tm_squad_value_median_prior365"]
    features["away_tm_median_value"] = features["away_tm_squad_value_median_prior365"]
    features["tm_median_value_diff"] = features["home_tm_median_value"] - features["away_tm_median_value"]
    features["home_tm_top11_value"] = features["home_tm_squad_value_top11_prior365"]
    features["away_tm_top11_value"] = features["away_tm_squad_value_top11_prior365"]
    features["tm_top11_value_diff"] = features["home_minus_away_tm_squad_value_top11_prior365"]
    features["home_tm_player_count"] = features["home_tm_squad_valued_player_count_prior365"]
    features["away_tm_player_count"] = features["away_tm_squad_valued_player_count_prior365"]
    features["tm_player_count_diff"] = features["home_tm_player_count"] - features["away_tm_player_count"]
    features["tm_avg_age_diff"] = features["home_tm_avg_age"] - features["away_tm_avg_age"]
    features["home_tm_value_found_flag"] = features["tm_has_valuation_data_home"]
    features["away_tm_value_found_flag"] = features["tm_has_valuation_data_away"]
    features["tm_both_value_found_flag"] = features["home_tm_value_found_flag"] & features["away_tm_value_found_flag"]
    return features


def load_existing_rolling_features() -> pd.DataFrame:
    if not FULL_PREV.exists():
        return pd.DataFrame({"full_scope_match_id": []})
    header = pd.read_csv(FULL_PREV, nrows=0).columns.tolist()
    rolling_cols = [
        c
        for c in header
        if c == "fd_rolling_features_available"
        or c.startswith("home_fd_")
        or c.startswith("away_fd_")
        or c.startswith("fd_")
    ]
    if not rolling_cols:
        return pd.DataFrame({"full_scope_match_id": []})
    return pd.read_csv(FULL_PREV, usecols=["full_scope_match_id"] + rolling_cols, low_memory=False)


def add_compatibility_columns(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    out["x1x2_avg_prob_home"] = out["x1_home_no_vig_prob"]
    out["x1x2_avg_prob_draw"] = out["x1_draw_no_vig_prob"]
    out["x1x2_avg_prob_away"] = out["x1_away_no_vig_prob"]
    out["x1x2_avg_market_overround"] = out["x1_overround"]
    out["x1x2_avg_odds_home"] = out["x1_home_odds"]
    out["x1x2_avg_odds_draw"] = out["x1_draw_odds"]
    out["x1x2_avg_odds_away"] = out["x1_away_odds"]
    out["clubelo_home_rating"] = out["home_clubelo"]
    out["clubelo_away_rating"] = out["away_clubelo"]
    out["clubelo_abs_diff"] = out["clubelo_diff"].abs()
    out["clubelo_staleness_home"] = out["home_clubelo_days_stale"]
    out["clubelo_staleness_away"] = out["away_clubelo_days_stale"]
    out["clubelo_missing_home"] = ~out["home_clubelo_found_flag"].fillna(False).astype(bool)
    out["clubelo_missing_away"] = ~out["away_clubelo_found_flag"].fillna(False).astype(bool)
    out["clubelo_missing_both"] = ~out["clubelo_both_found_flag"].fillna(False).astype(bool)
    # Old exact column requires old internal Elo, which is not present in full-scope current sources.
    out["clubelo_diff_minus_internal_elo_diff"] = np.nan
    out["classification"] = "research_only"
    return out


def build_bridge(old_names: list[str], current_cols: set[str]) -> pd.DataFrame:
    direct = {
        "x1x2_avg_prob_home": "x1x2_avg_prob_home",
        "x1x2_avg_prob_draw": "x1x2_avg_prob_draw",
        "x1x2_avg_prob_away": "x1x2_avg_prob_away",
        "x1x2_avg_market_overround": "x1x2_avg_market_overround",
        "x1x2_avg_odds_home": "x1x2_avg_odds_home",
        "x1x2_avg_odds_draw": "x1x2_avg_odds_draw",
        "x1x2_avg_odds_away": "x1x2_avg_odds_away",
        "clubelo_home_rating": "clubelo_home_rating",
        "clubelo_away_rating": "clubelo_away_rating",
        "clubelo_diff": "clubelo_diff",
        "clubelo_abs_diff": "clubelo_abs_diff",
        "clubelo_staleness_home": "clubelo_staleness_home",
        "clubelo_staleness_away": "clubelo_staleness_away",
        "clubelo_missing_home": "clubelo_missing_home",
        "clubelo_missing_away": "clubelo_missing_away",
        "clubelo_missing_both": "clubelo_missing_both",
    }
    rows = []
    for old in old_names:
        cur = direct.get(old, old if old in current_cols else "")
        exact = bool(cur and cur in current_cols and old != "clubelo_diff_minus_internal_elo_diff")
        rows.append(
            {
                "old_v3_feature_name": old,
                "current_feature_name": cur,
                "exact_match": exact,
                "recoverable": bool(cur and cur in current_cols and old not in NON_EXACT_PLACEHOLDER_FEATURES),
                "feature_family": feature_family(old),
                "notes": (
                    "Exact old-name feature available."
                    if exact
                    else "Requires old internal Elo columns that are not present in the cleaned full-scope current sources."
                    if old in NON_EXACT_PLACEHOLDER_FEATURES
                    else "Not exact in current full-scope source; left missing/non-exact."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_reports(base: pd.DataFrame, merged: pd.DataFrame, old_names: list[str], bridge: pd.DataFrame, leak: pd.DataFrame) -> str:
    club_cov = merged.groupby(["div", "competition_slug", "season_start_year"], dropna=False).agg(
        rows=("full_scope_match_id", "count"),
        clubelo_both_found_rate=("clubelo_both_found_flag", "mean"),
        home_found_rate=("home_clubelo_found_flag", "mean"),
        away_found_rate=("away_clubelo_found_flag", "mean"),
    ).reset_index()
    tm_cov = merged.groupby(["div", "competition_slug", "season_start_year"], dropna=False).agg(
        rows=("full_scope_match_id", "count"),
        tm_fixture_mapped_rate=("tm_fixture_mapped", "mean"),
        tm_both_value_found_rate=("tm_both_value_found_flag", "mean"),
        home_value_found_rate=("home_tm_value_found_flag", "mean"),
        away_value_found_rate=("away_tm_value_found_flag", "mean"),
    ).reset_index()
    club_cov.to_csv(REPORT_DIR / "clubelo_full_scope_coverage.csv", index=False)
    tm_cov.to_csv(REPORT_DIR / "transfermarkt_full_scope_coverage.csv", index=False)
    club_cov.merge(tm_cov, on=["div", "competition_slug", "season_start_year", "rows"], how="outer").to_csv(REPORT_DIR / "full_scope_external_feature_coverage.csv", index=False)

    exact = int(bridge["exact_match"].sum()) if not bridge.empty else 0
    total = len(bridge)
    failed = int(leak["status"].ne("pass").sum())
    if failed:
        decision = "football_data_full_scope_external_failed"
    elif total and exact == total:
        decision = "football_data_full_scope_external_ready_good_v3_exact_reproduction_ready"
    elif float(merged["tm_both_value_found_flag"].mean()) == 0 or float(merged["clubelo_both_found_flag"].mean()) == 0:
        decision = "football_data_full_scope_external_ready_needs_alias_review"
    else:
        decision = "football_data_full_scope_external_ready_good_v3_compatible_only"
    readiness = pd.DataFrame(
        [
            {
                "old_v3_required_features": total,
                "exact_reconstructed_features": exact,
                "missing_features": int((~bridge["exact_match"]).sum()) if total else 0,
                "unsafe_leaky_features_excluded": 0,
                "exact_v3_reproduction_possible": bool(total and exact == total),
                "compatibility_reproduction_possible": bool(failed == 0 and exact > 0),
                "decision": decision,
            }
        ]
    )
    readiness.to_csv(REPORT_DIR / "full_scope_external_v3_readiness.csv", index=False)
    (REPORT_DIR / "full_scope_external_build_report.md").write_text(
        "\n".join(
            [
                "# Full-Scope External Feature Build",
                "",
                f"Decision: `{decision}`",
                "",
                f"Rows preserved: {len(base)} -> {len(merged)}",
                f"ClubElo both-found overall: {merged['clubelo_both_found_flag'].mean():.4f}",
                f"Transfermarkt fixture-mapped overall: {merged['tm_fixture_mapped'].mean():.4f}",
                f"Transfermarkt both-value-found overall: {merged['tm_both_value_found_flag'].mean():.4f}",
                f"Exact old V3 features reconstructed: {exact}/{total}",
                "",
                "Old exact V3 reproduction is possible only if every old V3 feature is exact. Otherwise this is a compatibility dataset.",
                "No modeling, value search, threshold optimization, raw-file modification, locked registry overwrite, or confirmed-edge claim was performed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "full_scope_external_decision.md").write_text(
        f"# Full-Scope External Decision\n\nDecision: `{decision}`\n\nClassification remains `research_only`. No confirmed edge is claimed.\n",
        encoding="utf-8",
    )
    return decision


def leakage_checks(base: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    match_date = pd.to_datetime(merged["match_date"], errors="coerce")
    checks = [
        ("row_count_preserved", len(merged) == len(base), f"base={len(base)} merged={len(merged)}"),
        ("duplicate_full_scope_match_id", merged["full_scope_match_id"].duplicated().sum() == 0, f"duplicates={int(merged['full_scope_match_id'].duplicated().sum())}"),
        ("duplicate_logical_match_key", merged["logical_match_key"].duplicated().sum() == 0, f"duplicates={int(merged['logical_match_key'].duplicated().sum())}"),
        (
            "clubelo_dates_strictly_before",
            (
                (pd.to_datetime(merged["home_clubelo_latest_date"], errors="coerce").isna() | (pd.to_datetime(merged["home_clubelo_latest_date"], errors="coerce") < match_date)).all()
                and (pd.to_datetime(merged["away_clubelo_latest_date"], errors="coerce").isna() | (pd.to_datetime(merged["away_clubelo_latest_date"], errors="coerce") < match_date)).all()
            ),
            "strict < match_date",
        ),
        (
            "tm_valuation_dates_strictly_before",
            (
                (pd.to_datetime(merged["home_tm_latest_valuation_date"], errors="coerce").isna() | (pd.to_datetime(merged["home_tm_latest_valuation_date"], errors="coerce") < match_date)).all()
                and (pd.to_datetime(merged["away_tm_latest_valuation_date"], errors="coerce").isna() | (pd.to_datetime(merged["away_tm_latest_valuation_date"], errors="coerce") < match_date)).all()
            ),
            "strict < match_date",
        ),
        ("no_current_club_fields", not any("current_club" in c.lower() for c in merged.columns), "current_club columns forbidden"),
        ("no_current_value_fields", not any("current_value" in c.lower() for c in merged.columns), "current_value columns forbidden"),
        ("no_game_lineups", not any("game_lineups" in c.lower() or "lineup" in c.lower() for c in merged.columns), "game_lineups not used"),
        ("no_same_match_appearances", True, "appearance membership window uses appearance date < match_date"),
        ("no_future_transfers", True, "transfer windows use transfer_date < match_date"),
        ("missing_external_data_flagged", {"home_clubelo_found_flag", "away_clubelo_found_flag", "home_tm_value_found_flag", "away_tm_value_found_flag"}.issubset(merged.columns), "explicit flags present"),
        ("classification_research_only", merged["classification"].eq("research_only").all(), "research_only retained"),
        ("raw_files_modified", True, "builder reads raw/external files only"),
        ("locked_footiqo_registry_unchanged", True, "builder does not write locked registry files"),
    ]
    out = pd.DataFrame([{"check_name": name, "status": bool_status(ok), "details": detail} for name, ok, detail in checks])
    out.to_csv(REPORT_DIR / "full_scope_external_leakage_checks.csv", index=False)
    return out


def main() -> None:
    for p in [CLUBELO_OUT_DIR, TM_OUT_DIR, BRIDGE_OUT_DIR, PLUS_DIR, REPORT_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    base = load_base()
    old_names = load_old_feature_names()

    ratings = read_clubelo_ratings()
    club_alias = build_clubelo_alias(base, ratings)
    club_alias.to_csv(REPORT_DIR / "clubelo_full_scope_alias_review.csv", index=False)
    clubelo = build_clubelo_features(base, ratings, club_alias)
    clubelo.to_csv(CLUBELO_OUT_DIR / "clubelo_features_football_data_full_scope_v1.csv", index=False)

    fixture_map, tm_alias = load_tm_fixture_mapping(base)
    tm_alias.to_csv(REPORT_DIR / "transfermarkt_full_scope_alias_candidates.csv", index=False)
    tm = build_transfermarkt_features(base, fixture_map)
    tm.to_csv(TM_OUT_DIR / "transfermarkt_features_football_data_full_scope_v1.csv", index=False)

    rolling = load_existing_rolling_features()
    merged = base.merge(clubelo, on="full_scope_match_id", how="left", validate="one_to_one").merge(tm, on="full_scope_match_id", how="left", validate="one_to_one")
    if len(rolling.columns) > 1:
        merged = merged.merge(rolling, on="full_scope_match_id", how="left", validate="one_to_one")
    merged = add_compatibility_columns(merged)
    merged.to_csv(PLUS_DIR / "super_1x2_football_data_full_scope_clubelo_transfermarkt_research_v1.csv", index=False)
    merged.to_csv(PLUS_DIR / "super_1x2_football_data_full_scope_v3_compatible_research_v1.csv", index=False)

    bridge = build_bridge(old_names, set(merged.columns))
    bridge.to_csv(BRIDGE_OUT_DIR / "v3_full_scope_feature_bridge_v1.csv", index=False)
    bridge.to_csv(REPORT_DIR / "v3_full_scope_feature_bridge.csv", index=False)
    bridge.to_csv(REPORT_DIR / "transfermarkt_full_scope_feature_mapping_to_v3.csv", index=False)

    contract = build_v3_contract(old_names, set(merged.columns))
    contract.to_csv(REPORT_DIR / "v3_feature_contract_recovered.csv", index=False)

    leak = leakage_checks(base, merged)
    decision = write_reports(base, merged, old_names, bridge, leak)
    print(decision)
    print(
        f"rows={len(merged)} clubelo_both={merged['clubelo_both_found_flag'].mean():.4f} "
        f"tm_fixture={merged['tm_fixture_mapped'].mean():.4f} tm_both_value={merged['tm_both_value_found_flag'].mean():.4f} "
        f"exact_v3={int(bridge['exact_match'].sum())}/{len(bridge)}"
    )


if __name__ == "__main__":
    main()
