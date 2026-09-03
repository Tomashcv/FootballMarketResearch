from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


TM_DIR = Path("data/external/players/transfermarkt_raw/player_scores")
FEATURE_MATRIX = Path("data/processed/features/football_feature_matrix_v1_1.csv")
CLUBELO_ARCHIVE = Path("data/raw_external/clubelo_manual/clubelo_archive.zip")
REPORT_DIR = Path("outputs/reports")

LEAGUES = ["E0", "E1", "E2", "E3", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "SC0"]
LEAGUE_TO_TM_COMP = {
    "E0": "GB1",
    "E1": "GB2",
    "E2": "GB3",
    "E3": "GB4",
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
TM_COMP_TO_LEAGUE = {value: key for key, value in LEAGUE_TO_TM_COMP.items()}

TM_FILES = [
    "players.csv",
    "clubs.csv",
    "games.csv",
    "appearances.csv",
    "player_valuations.csv",
    "transfers.csv",
    "game_lineups.csv",
    "club_games.csv",
    "competitions.csv",
]

OUTPUTS = {
    "schema": REPORT_DIR / "transfermarkt_file_schema_audit.csv",
    "competition": REPORT_DIR / "transfermarkt_competition_mapping_audit.csv",
    "club_candidates": REPORT_DIR / "transfermarkt_club_mapping_candidates.csv",
    "match_coverage": REPORT_DIR / "transfermarkt_match_mapping_coverage.csv",
    "valuation": REPORT_DIR / "transfermarkt_valuation_date_safety.csv",
    "transfer": REPORT_DIR / "transfermarkt_transfer_date_safety.csv",
    "squad_policy": REPORT_DIR / "transfermarkt_squad_assignment_policy.csv",
    "feature_policy": REPORT_DIR / "transfermarkt_v1_recommended_feature_policy.csv",
    "markdown": REPORT_DIR / "transfermarkt_mapping_date_safety_audit.md",
}

LEGAL_WORDS = {
    "1",
    "1900",
    "1901",
    "1905",
    "1907",
    "1899",
    "1893",
    "1910",
    "1919",
    "1920",
    "1936",
    "2005",
    "2020",
    "ac",
    "afc",
    "as",
    "association",
    "associazione",
    "balompie",
    "calcio",
    "cf",
    "club",
    "de",
    "fc",
    "football",
    "futbol",
    "futebol",
    "fussball",
    "fußball",
    "real",
    "sad",
    "sc",
    "soccer",
    "societa",
    "sport",
    "sporting",
    "sportiva",
    "the",
    "town",
    "ud",
    "united",
    "verein",
}


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in LEGAL_WORDS]
    return " ".join(tokens)


def count_rows(path: Path) -> int:
    total = 0
    with path.open("rb") as handle:
        for total, _ in enumerate(handle, start=0):
            pass
    return max(total, 0)


def columns_matching(columns: list[str], patterns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if any(pattern in column.casefold() for pattern in patterns)]


def file_schema_audit() -> pd.DataFrame:
    rows = []
    for filename in TM_FILES:
        path = TM_DIR / filename
        if not path.exists():
            rows.append({"file": filename, "exists": False, "usable_for_historical_pre_match_features": "no_missing_file"})
            continue
        sample = pd.read_csv(path, nrows=20, low_memory=False)
        columns = list(sample.columns)
        leakage = []
        for column in columns:
            low = column.casefold()
            if "current_club" in low:
                leakage.append(column)
            if filename == "game_lineups.csv" and low in {"type", "position", "number", "team_captain"}:
                leakage.append(column)
            if filename == "players.csv" and low in {"market_value_in_eur", "highest_market_value_in_eur", "last_season"}:
                leakage.append(column)
        if filename in {"games.csv", "club_games.csv", "competitions.csv"}:
            usable = "yes_for_mapping_not_player_features"
        elif filename == "transfers.csv":
            usable = "yes_if_transfer_date_strictly_before_match_date"
        elif filename == "appearances.csv":
            usable = "partial_membership_evidence_only_prior_appearances"
        elif filename == "player_valuations.csv":
            usable = "yes_values_only_if_date_strictly_before_match_date_and_membership_safe"
        elif filename == "game_lineups.csv":
            usable = "no_for_pre_match_unless_publication_timestamp_proven"
        elif filename == "players.csv":
            usable = "static_bio_only_current_club_fields_not_historical"
        elif filename == "clubs.csv":
            usable = "yes_for_club_identity_mapping_only"
        else:
            usable = "review_required"
        rows.append(
            {
                "file": filename,
                "exists": True,
                "row_count": count_rows(path),
                "columns": "|".join(columns),
                "date_columns": "|".join(columns_matching(columns, ("date",))),
                "club_id_columns": "|".join(columns_matching(columns, ("club_id", "opponent_id"))),
                "player_id_columns": "|".join(columns_matching(columns, ("player_id",))),
                "competition_columns": "|".join(columns_matching(columns, ("competition", "league"))),
                "season_columns": "|".join(columns_matching(columns, ("season",))),
                "obvious_leakage_columns": "|".join(dict.fromkeys(leakage)),
                "usable_for_historical_pre_match_features": usable,
            }
        )
    return pd.DataFrame(rows)


def load_feature_matrix_ids() -> pd.DataFrame:
    usecols = ["match_id", "match_date", "league", "season_start_year", "season_end_year", "home_team", "away_team"]
    matches = pd.read_csv(FEATURE_MATRIX, usecols=usecols, low_memory=False)
    matches = matches[matches["league"].isin(LEAGUES)].copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce").dt.normalize()
    matches["season_start_year"] = pd.to_numeric(matches["season_start_year"], errors="coerce").astype("Int64")
    return matches.dropna(subset=["match_date", "league", "home_team", "away_team", "season_start_year"]).reset_index(drop=True)


def competition_audit(games: pd.DataFrame, competitions: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    comp_rows = []
    game_counts = games.groupby(["competition_id", "season"], dropna=False).size().rename("tm_games").reset_index()
    fd_counts = matches.groupby(["league", "season_start_year"], dropna=False).size().rename("football_data_fixtures").reset_index()
    comp_lookup = competitions.set_index("competition_id").to_dict("index")
    for league, comp_id in LEAGUE_TO_TM_COMP.items():
        rows = game_counts[game_counts["competition_id"].eq(comp_id)].copy()
        if rows.empty:
            comp_rows.append(
                {
                    "league": league,
                    "tm_competition_id": comp_id,
                    "tm_competition_name": comp_lookup.get(comp_id, {}).get("name", ""),
                    "mapping_status": "unmatched_no_games",
                    "season_start_year": pd.NA,
                    "tm_games": 0,
                    "football_data_fixtures": int(fd_counts.loc[fd_counts["league"].eq(league), "football_data_fixtures"].sum()),
                }
            )
            continue
        rows = rows.rename(columns={"season": "season_start_year"})
        rows["league"] = league
        rows["tm_competition_id"] = comp_id
        rows["tm_competition_name"] = comp_lookup.get(comp_id, {}).get("name", "")
        rows = rows.merge(fd_counts, on=["league", "season_start_year"], how="left")
        rows["football_data_fixtures"] = rows["football_data_fixtures"].fillna(0).astype(int)
        rows["mapping_status"] = np.where(rows["football_data_fixtures"].gt(0), "matched", "tm_season_no_football_data_fixture")
        comp_rows.extend(
            rows[
                [
                    "league",
                    "tm_competition_id",
                    "tm_competition_name",
                    "mapping_status",
                    "season_start_year",
                    "tm_games",
                    "football_data_fixtures",
                ]
            ].to_dict("records")
        )
    requested = set(LEAGUE_TO_TM_COMP.values())
    extra_domestic = competitions[
        competitions["type"].fillna("").eq("domestic_league") & ~competitions["competition_id"].isin(requested)
    ].copy()
    for row in extra_domestic.head(80).itertuples(index=False):
        comp_rows.append(
            {
                "league": "",
                "tm_competition_id": row.competition_id,
                "tm_competition_name": row.name,
                "mapping_status": "unmatched_not_requested",
                "season_start_year": pd.NA,
                "tm_games": int(game_counts.loc[game_counts["competition_id"].eq(row.competition_id), "tm_games"].sum()),
                "football_data_fixtures": 0,
            }
        )
    return pd.DataFrame(comp_rows).sort_values(["league", "tm_competition_id", "season_start_year"], na_position="last")


def club_name_pool(games: pd.DataFrame, clubs: pd.DataFrame) -> pd.DataFrame:
    home = games[["competition_id", "season", "home_club_id", "home_club_name"]].rename(
        columns={"home_club_id": "club_id", "home_club_name": "club_name"}
    )
    away = games[["competition_id", "season", "away_club_id", "away_club_name"]].rename(
        columns={"away_club_id": "club_id", "away_club_name": "club_name"}
    )
    pool = pd.concat([home, away], ignore_index=True)
    pool["league"] = pool["competition_id"].map(TM_COMP_TO_LEAGUE)
    pool = pool.dropna(subset=["league", "club_id", "club_name"]).copy()
    club_names = clubs[["club_id", "name", "domestic_competition_id"]].rename(columns={"name": "club_name"})
    club_names["league"] = club_names["domestic_competition_id"].map(TM_COMP_TO_LEAGUE)
    club_names["season"] = pd.NA
    pool = pd.concat([pool[["league", "competition_id", "season", "club_id", "club_name"]], club_names[["league", "season", "club_id", "club_name"]]], ignore_index=True)
    pool = pool.dropna(subset=["league", "club_id", "club_name"]).drop_duplicates()
    pool["club_id"] = pd.to_numeric(pool["club_id"], errors="coerce").astype("Int64")
    pool["norm_name"] = pool["club_name"].map(normalize_name)
    return pool[pool["norm_name"].ne("")].reset_index(drop=True)


def build_club_candidates(matches: pd.DataFrame, name_pool: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], int]]:
    teams = (
        pd.concat(
            [
                matches[["league", "season_start_year", "home_team"]].rename(columns={"home_team": "team"}),
                matches[["league", "season_start_year", "away_team"]].rename(columns={"away_team": "team"}),
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    pool_by_league = {league: group.copy() for league, group in name_pool.groupby("league")}
    rows = []
    exact_mapping: dict[tuple[str, str], int] = {}
    for team in teams.itertuples(index=False):
        league = str(team.league)
        team_name = str(team.team)
        norm = normalize_name(team_name)
        pool = pool_by_league.get(league, pd.DataFrame(columns=name_pool.columns))
        exact = pool[pool["norm_name"].eq(norm)].copy()
        exact_ids = sorted(pd.to_numeric(exact["club_id"], errors="coerce").dropna().astype(int).unique())
        if len(exact_ids) == 1:
            exact_mapping[(league, team_name)] = exact_ids[0]
            rows.append(
                {
                    "league": league,
                    "season_start_year": int(team.season_start_year),
                    "football_data_team": team_name,
                    "match_type": "exact_normalized",
                    "tm_club_id": exact_ids[0],
                    "tm_club_name": "; ".join(sorted(exact["club_name"].dropna().astype(str).unique())[:5]),
                    "score": 1.0,
                    "risk": "low",
                    "note": "Used for fixture mapping; exact after conservative normalization.",
                }
            )
            continue
        if len(exact_ids) > 1:
            rows.append(
                {
                    "league": league,
                    "season_start_year": int(team.season_start_year),
                    "football_data_team": team_name,
                    "match_type": "ambiguous_exact_normalized",
                    "tm_club_id": "|".join(map(str, exact_ids)),
                    "tm_club_name": "; ".join(sorted(exact["club_name"].dropna().astype(str).unique())[:8]),
                    "score": 1.0,
                    "risk": "high",
                    "note": "Not used automatically.",
                }
            )
            continue
        scored = []
        for row in pool[["club_id", "club_name", "norm_name"]].drop_duplicates().itertuples(index=False):
            score = SequenceMatcher(None, norm, row.norm_name).ratio() if norm and row.norm_name else 0.0
            if score >= 0.72:
                scored.append((score, int(row.club_id), str(row.club_name)))
        for score, club_id, club_name in sorted(scored, reverse=True)[:5]:
            rows.append(
                {
                    "league": league,
                    "season_start_year": int(team.season_start_year),
                    "football_data_team": team_name,
                    "match_type": "fuzzy_candidate",
                    "tm_club_id": club_id,
                    "tm_club_name": club_name,
                    "score": round(float(score), 4),
                    "risk": "medium" if score >= 0.88 else "high",
                    "note": "Manual alias suggestion only; not used automatically.",
                }
            )
        if not scored:
            rows.append(
                {
                    "league": league,
                    "season_start_year": int(team.season_start_year),
                    "football_data_team": team_name,
                    "match_type": "unmatched",
                    "tm_club_id": "",
                    "tm_club_name": "",
                    "score": 0.0,
                    "risk": "high",
                    "note": "No normalized exact match or useful fuzzy candidate.",
                }
            )
    candidates = pd.DataFrame(rows)
    return candidates.sort_values(["league", "season_start_year", "football_data_team", "match_type", "score"], ascending=[True, True, True, True, False]), exact_mapping


def fixture_mapping(matches: pd.DataFrame, games: pd.DataFrame, exact_mapping: dict[tuple[str, str], int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = matches.copy()
    mapped["tm_competition_id"] = mapped["league"].map(LEAGUE_TO_TM_COMP)
    mapped["home_tm_club_id"] = [exact_mapping.get((str(row.league), str(row.home_team)), np.nan) for row in mapped.itertuples(index=False)]
    mapped["away_tm_club_id"] = [exact_mapping.get((str(row.league), str(row.away_team)), np.nan) for row in mapped.itertuples(index=False)]
    mapped["mapping_key_available"] = mapped[["tm_competition_id", "home_tm_club_id", "away_tm_club_id"]].notna().all(axis=1)
    tm_games = games[["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id"]].copy()
    tm_games["date"] = pd.to_datetime(tm_games["date"], errors="coerce").dt.normalize()
    tm_games = tm_games.rename(
        columns={
            "competition_id": "tm_competition_id",
            "season": "season_start_year",
            "date": "match_date",
            "home_club_id": "home_tm_club_id",
            "away_club_id": "away_tm_club_id",
        }
    )
    tm_games["candidate_count"] = tm_games.groupby(
        ["tm_competition_id", "season_start_year", "match_date", "home_tm_club_id", "away_tm_club_id"]
    )["game_id"].transform("count")
    joined = mapped.merge(
        tm_games,
        on=["tm_competition_id", "season_start_year", "match_date", "home_tm_club_id", "away_tm_club_id"],
        how="left",
    )
    joined["match_mapping_status"] = np.select(
        [
            ~joined["mapping_key_available"],
            joined["game_id"].isna(),
            joined["candidate_count"].fillna(0).gt(1),
            joined["game_id"].notna(),
        ],
        ["club_mapping_missing", "no_transfermarkt_game_candidate", "ambiguous_duplicate_transfermarkt_games", "mapped"],
        default="unknown",
    )
    coverage = (
        joined.groupby(["league", "season_start_year"], dropna=False)
        .agg(
            football_data_fixtures=("match_id", "count"),
            mapped_fixture_count=("match_mapping_status", lambda s: int(s.eq("mapped").sum())),
            unmatched_fixture_count=("match_mapping_status", lambda s: int((~s.eq("mapped")).sum())),
            duplicate_candidate_count=("match_mapping_status", lambda s: int(s.eq("ambiguous_duplicate_transfermarkt_games").sum())),
            ambiguous_candidate_count=("match_mapping_status", lambda s: int(s.eq("ambiguous_duplicate_transfermarkt_games").sum())),
            missing_club_mapping_count=("match_mapping_status", lambda s: int(s.eq("club_mapping_missing").sum())),
            no_tm_game_candidate_count=("match_mapping_status", lambda s: int(s.eq("no_transfermarkt_game_candidate").sum())),
        )
        .reset_index()
    )
    coverage["mapping_coverage_rate"] = coverage["mapped_fixture_count"] / coverage["football_data_fixtures"].replace(0, np.nan)
    return joined, coverage.sort_values(["league", "season_start_year"])


def build_appearance_index(relevant_clubs: set[int], min_date: pd.Timestamp) -> dict[int, pd.DataFrame]:
    app = pd.read_csv(
        TM_DIR / "appearances.csv",
        usecols=["player_id", "player_club_id", "date"],
        parse_dates=["date"],
        low_memory=False,
    )
    app = app[app["player_club_id"].isin(relevant_clubs) & app["date"].lt(pd.Timestamp.today().normalize())]
    app = app[app["date"].ge(min_date - pd.Timedelta(days=370))]
    app["player_id"] = pd.to_numeric(app["player_id"], errors="coerce")
    app = app.dropna(subset=["player_id", "player_club_id", "date"]).copy()
    app["player_id"] = app["player_id"].astype(int)
    return {int(club_id): group[["date", "player_id"]].sort_values("date").reset_index(drop=True) for club_id, group in app.groupby("player_club_id")}


def build_valuation_index(player_ids: set[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    vals = pd.read_csv(
        TM_DIR / "player_valuations.csv",
        usecols=["player_id", "date", "market_value_in_eur"],
        parse_dates=["date"],
        low_memory=False,
    )
    vals = vals[vals["player_id"].isin(player_ids)].dropna(subset=["player_id", "date", "market_value_in_eur"]).copy()
    vals["player_id"] = vals["player_id"].astype(int)
    vals["market_value_in_eur"] = pd.to_numeric(vals["market_value_in_eur"], errors="coerce")
    index = {}
    for player_id, group in vals.groupby("player_id"):
        group = group.sort_values("date")
        dates = group["date"].to_numpy(dtype="datetime64[ns]")
        values = group["market_value_in_eur"].to_numpy(dtype=float)
        index[int(player_id)] = (dates, values)
    return index


def prior_players(club_apps: pd.DataFrame, match_date: pd.Timestamp, window_days: int = 365) -> np.ndarray:
    if club_apps.empty:
        return np.array([], dtype=int)
    dates = club_apps["date"].to_numpy(dtype="datetime64[ns]")
    right = np.searchsorted(dates, np.datetime64(match_date), side="left")
    left = np.searchsorted(dates, np.datetime64(match_date - pd.Timedelta(days=window_days)), side="left")
    if right <= left:
        return np.array([], dtype=int)
    return club_apps.iloc[left:right]["player_id"].drop_duplicates().to_numpy(dtype=int)


def latest_valuation_before(index: dict[int, tuple[np.ndarray, np.ndarray]], player_id: int, match_date: pd.Timestamp) -> tuple[float, float]:
    item = index.get(int(player_id))
    if item is None:
        return np.nan, np.nan
    dates, values = item
    pos = bisect_left(dates, np.datetime64(match_date)) - 1
    if pos < 0:
        return np.nan, np.nan
    staleness = (match_date - pd.Timestamp(dates[pos])).days
    return float(values[pos]), float(staleness)


def valuation_date_safety(mapped_fixtures: pd.DataFrame) -> pd.DataFrame:
    mapped = mapped_fixtures[mapped_fixtures["match_mapping_status"].eq("mapped")].copy()
    if mapped.empty:
        return pd.DataFrame()
    sides = pd.concat(
        [
            mapped[["match_id", "league", "season_start_year", "match_date", "home_tm_club_id"]].rename(columns={"home_tm_club_id": "club_id"}),
            mapped[["match_id", "league", "season_start_year", "match_date", "away_tm_club_id"]].rename(columns={"away_tm_club_id": "club_id"}),
        ],
        ignore_index=True,
    ).dropna(subset=["club_id"])
    sides["club_id"] = sides["club_id"].astype(int)
    app_index = build_appearance_index(set(sides["club_id"].unique()), sides["match_date"].min())
    player_ids = {int(pid) for frame in app_index.values() for pid in frame["player_id"].unique()}
    val_index = build_valuation_index(player_ids)

    rows = []
    cache = {}
    for side in sides.itertuples(index=False):
        key = (int(side.club_id), pd.Timestamp(side.match_date))
        if key not in cache:
            players = prior_players(app_index.get(int(side.club_id), pd.DataFrame(columns=["date", "player_id"])), pd.Timestamp(side.match_date))
            values = []
            staleness = []
            for player_id in players:
                value, stale = latest_valuation_before(val_index, int(player_id), pd.Timestamp(side.match_date))
                if np.isfinite(value):
                    values.append(value)
                    staleness.append(stale)
            values = sorted(values, reverse=True)
            total = float(np.sum(values)) if values else np.nan
            top3 = float(np.sum(values[:3])) if len(values) >= 3 else np.nan
            top5 = float(np.sum(values[:5])) if len(values) >= 5 else np.nan
            top11 = float(np.sum(values[:11])) if len(values) >= 11 else np.nan
            cache[key] = {
                "prior_membership_players_365d": int(len(players)),
                "valued_players_before_match": int(len(values)),
                "latest_known_player_valuation_feasible": len(values) > 0,
                "squad_total_value_feasible": len(values) >= 11,
                "top3_value_feasible": len(values) >= 3,
                "top5_value_feasible": len(values) >= 5,
                "top11_value_feasible": len(values) >= 11,
                "depth_value_outside_top5_feasible": len(values) >= 11,
                "concentration_metrics_feasible": len(values) >= 5 and np.isfinite(total) and total > 0,
                "valuation_staleness_days_median": float(np.median(staleness)) if staleness else np.nan,
                "missing_valuation_count": int(max(len(players) - len(values), 0)),
                "squad_total_value_eur_median_input": total,
                "top3_value_eur_median_input": top3,
                "top5_value_eur_median_input": top5,
                "top11_value_eur_median_input": top11,
                "depth_outside_top5_value_eur_median_input": float(np.sum(values[5:])) if len(values) > 5 else np.nan,
                "top5_concentration_median_input": float(top5 / total) if len(values) >= 5 and np.isfinite(total) and total > 0 else np.nan,
            }
        rows.append({"league": side.league, "season_start_year": int(side.season_start_year), **cache[key]})
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["league", "season_start_year"], dropna=False)
        .agg(
            club_match_sides=("league", "count"),
            median_prior_membership_players_365d=("prior_membership_players_365d", "median"),
            median_valued_players_before_match=("valued_players_before_match", "median"),
            latest_known_player_valuation_coverage=("latest_known_player_valuation_feasible", "mean"),
            squad_total_value_coverage=("squad_total_value_feasible", "mean"),
            top3_value_coverage=("top3_value_feasible", "mean"),
            top5_value_coverage=("top5_value_feasible", "mean"),
            top11_value_coverage=("top11_value_feasible", "mean"),
            depth_value_outside_top5_coverage=("depth_value_outside_top5_feasible", "mean"),
            concentration_metrics_coverage=("concentration_metrics_feasible", "mean"),
            median_valuation_staleness_days=("valuation_staleness_days_median", "median"),
            median_missing_valuation_count=("missing_valuation_count", "median"),
            median_squad_total_value_eur=("squad_total_value_eur_median_input", "median"),
            median_top3_value_eur=("top3_value_eur_median_input", "median"),
            median_top5_value_eur=("top5_value_eur_median_input", "median"),
            median_top11_value_eur=("top11_value_eur_median_input", "median"),
            median_depth_outside_top5_value_eur=("depth_outside_top5_value_eur_median_input", "median"),
            median_top5_concentration=("top5_concentration_median_input", "median"),
        )
        .reset_index()
    )
    summary["membership_rule"] = "player counted only with appearance for same club in prior 365 days; no current_club fields used"
    summary["date_rule"] = "valuation date strictly before match_date"
    return summary.sort_values(["league", "season_start_year"])


def transfer_date_safety(mapped_fixtures: pd.DataFrame) -> pd.DataFrame:
    mapped = mapped_fixtures[mapped_fixtures["match_mapping_status"].eq("mapped")].copy()
    if mapped.empty:
        return pd.DataFrame()
    sides = pd.concat(
        [
            mapped[["match_id", "league", "season_start_year", "match_date", "home_tm_club_id"]].rename(columns={"home_tm_club_id": "club_id"}),
            mapped[["match_id", "league", "season_start_year", "match_date", "away_tm_club_id"]].rename(columns={"away_tm_club_id": "club_id"}),
        ],
        ignore_index=True,
    ).dropna(subset=["club_id"])
    sides["club_id"] = sides["club_id"].astype(int)
    transfers = pd.read_csv(TM_DIR / "transfers.csv", parse_dates=["transfer_date"], low_memory=False)
    transfers["market_value_in_eur"] = pd.to_numeric(transfers["market_value_in_eur"], errors="coerce").fillna(0.0)
    relevant = set(sides["club_id"].unique())
    arrivals = transfers[transfers["to_club_id"].isin(relevant)].copy()
    arrivals["club_id"] = arrivals["to_club_id"].astype(int)
    arrivals["direction"] = "arrival"
    departures = transfers[transfers["from_club_id"].isin(relevant)].copy()
    departures["club_id"] = departures["from_club_id"].astype(int)
    departures["direction"] = "departure"
    moves = pd.concat([arrivals, departures], ignore_index=True)
    moves = moves.dropna(subset=["club_id", "transfer_date"])
    by_club = {int(club): group.sort_values("transfer_date").reset_index(drop=True) for club, group in moves.groupby("club_id")}
    windows = [30, 90, 180, 365]
    rows = []
    cache = {}
    for side in sides.itertuples(index=False):
        key = (int(side.club_id), pd.Timestamp(side.match_date))
        if key not in cache:
            club_moves = by_club.get(int(side.club_id), pd.DataFrame(columns=moves.columns))
            data = {}
            for window in windows:
                start = pd.Timestamp(side.match_date) - pd.Timedelta(days=window)
                prior = club_moves[club_moves["transfer_date"].lt(pd.Timestamp(side.match_date)) & club_moves["transfer_date"].ge(start)]
                arr = prior[prior["direction"].eq("arrival")]
                dep = prior[prior["direction"].eq("departure")]
                arr_value = float(arr["market_value_in_eur"].sum())
                dep_value = float(dep["market_value_in_eur"].sum())
                data[f"arrivals_{window}d_feasible"] = True
                data[f"departures_{window}d_feasible"] = True
                data[f"arrival_count_{window}d"] = int(len(arr))
                data[f"departure_count_{window}d"] = int(len(dep))
                data[f"arrival_value_eur_{window}d"] = arr_value
                data[f"departure_value_eur_{window}d"] = dep_value
                data[f"net_transfer_value_change_eur_{window}d"] = arr_value - dep_value
                data[f"squad_churn_count_{window}d"] = int(len(arr) + len(dep))
                data[f"biggest_arrival_value_eur_{window}d"] = float(arr["market_value_in_eur"].max()) if not arr.empty else 0.0
                data[f"biggest_departure_value_eur_{window}d"] = float(dep["market_value_in_eur"].max()) if not dep.empty else 0.0
            cache[key] = data
        rows.append({"league": side.league, "season_start_year": int(side.season_start_year), **cache[key]})
    detail = pd.DataFrame(rows)
    agg = {"club_match_sides": ("league", "count")}
    for window in windows:
        for col in [
            f"arrival_count_{window}d",
            f"departure_count_{window}d",
            f"arrival_value_eur_{window}d",
            f"departure_value_eur_{window}d",
            f"net_transfer_value_change_eur_{window}d",
            f"squad_churn_count_{window}d",
            f"biggest_arrival_value_eur_{window}d",
            f"biggest_departure_value_eur_{window}d",
        ]:
            agg[f"median_{col}"] = (col, "median")
            agg[f"coverage_nonzero_{col}"] = (col, lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).ne(0).mean()))
    summary = detail.groupby(["league", "season_start_year"], dropna=False).agg(**agg).reset_index()
    summary["date_rule"] = "transfer_date strictly before match_date"
    summary["assignment_rule"] = "arrivals use to_club_id; departures use from_club_id; no future transfer rows after match_date"
    return summary.sort_values(["league", "season_start_year"])


def squad_assignment_policy(match_coverage: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    mapped_rate = float(match_coverage["mapped_fixture_count"].sum() / match_coverage["football_data_fixtures"].sum())
    valuation_rate = float(valuation["top11_value_coverage"].mean()) if not valuation.empty and "top11_value_coverage" in valuation else 0.0
    return pd.DataFrame(
        [
            {
                "option": "A_transfers_based_roster_reconstruction",
                "leakage_risk": "medium",
                "coverage": "transfer rows cover dated arrivals/departures, but loans/free moves and initial roster state need validation",
                "implementation_complexity": "high",
                "suitability_for_pre_match_features": "suitable for churn features now; not sufficient alone for full roster valuation",
                "audit_observed_mapping_rate": mapped_rate,
            },
            {
                "option": "B_appearances_based_historical_club_membership",
                "leakage_risk": "low_if_only_prior_appearances",
                "coverage": "conservative and incomplete because new arrivals before debut are missing",
                "implementation_complexity": "medium",
                "suitability_for_pre_match_features": "suitable for partial valuation coverage and staleness diagnostics",
                "audit_observed_top11_valuation_rate": valuation_rate,
            },
            {
                "option": "C_game_lineups_based_recent_squad_proxy",
                "leakage_risk": "high_without_publication_timestamp",
                "coverage": "potentially strong, but lineups are match-event artifacts",
                "implementation_complexity": "medium",
                "suitability_for_pre_match_features": "not suitable until lineup timestamp/publication safety is proven",
            },
            {
                "option": "D_hybrid_transfers_plus_appearances",
                "leakage_risk": "medium_low_with_strict_dates",
                "coverage": "best candidate after manual club mapping review and transfer semantics validation",
                "implementation_complexity": "high",
                "suitability_for_pre_match_features": "recommended future path after mapping corrections; use warnings for partial valuations",
            },
        ]
    )


def recommended_policy(match_coverage: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    total = int(match_coverage["football_data_fixtures"].sum()) if not match_coverage.empty else 0
    mapped = int(match_coverage["mapped_fixture_count"].sum()) if not match_coverage.empty else 0
    mapping_rate = mapped / total if total else 0.0
    valuation_top11 = float(valuation["top11_value_coverage"].mean()) if not valuation.empty and "top11_value_coverage" in valuation else 0.0
    if mapping_rate < 0.25:
        status = "transfermarkt_mapping_ready_only"
        reason = "Competition mapping is explicit, but exact-only club and match coverage is too low for feature build."
    elif valuation_top11 < 0.50:
        status = "transfermarkt_transfers_ready"
        reason = "Dated transfer churn features are date-safe on mapped fixtures; valuation/squad features remain partial."
    else:
        status = "transfermarkt_valuations_ready_partial"
        reason = "Valuation features are date-safe under prior-appearance membership, but roster coverage is still conservative and incomplete."
    return pd.DataFrame(
        [
            {
                "recommended_status": status,
                "mapping_coverage_rate": mapping_rate,
                "mapped_fixture_count": mapped,
                "total_fixture_count": total,
                "mean_top11_valuation_coverage": valuation_top11,
                "confirmed_edge_claim": "no",
                "models_run": "no",
                "value_search_run": "no",
                "threshold_optimization_run": "no",
                "betting_rules_created": "no",
                "reason": reason,
                "next_gate": "manual alias review, fixture remap, and strict-date feature builder tests before any predictive use",
            }
        ]
    )


def write_markdown(
    schema: pd.DataFrame,
    competition: pd.DataFrame,
    candidates: pd.DataFrame,
    match_coverage: pd.DataFrame,
    valuation: pd.DataFrame,
    transfer: pd.DataFrame,
    squad_policy: pd.DataFrame,
    feature_policy: pd.DataFrame,
) -> None:
    mapped = int(match_coverage["mapped_fixture_count"].sum()) if not match_coverage.empty else 0
    total = int(match_coverage["football_data_fixtures"].sum()) if not match_coverage.empty else 0
    unmatched = total - mapped
    exact_teams = int(candidates["match_type"].eq("exact_normalized").sum()) if not candidates.empty else 0
    fuzzy_teams = int(candidates["match_type"].eq("fuzzy_candidate").sum()) if not candidates.empty else 0
    ambiguous = int(candidates["match_type"].str.contains("ambiguous", na=False).sum()) if not candidates.empty else 0
    policy = feature_policy.iloc[0].to_dict()
    lines = [
        "# Transfermarkt mapping and date-safety audit",
        "",
        "Scope: mapping, coverage, and date-safety only. No predictive models, value searches, threshold optimization, giant-model training, betting rules, or confirmed-edge claims were run or created.",
        "",
        "## Inputs",
        f"- Transfermarkt raw directory: `{TM_DIR}`",
        f"- Football feature matrix: `{FEATURE_MATRIX}`",
        f"- Optional ClubElo archive present: `{CLUBELO_ARCHIVE.exists()}`",
        "",
        "## File/schema audit",
        f"- Candidate files audited: {len(schema)}",
        "- Current-club fields are flagged as leakage risks and are not used as historical truth.",
        "- `game_lineups.csv` is classified as unsafe for pre-match features unless lineup publication timestamp safety is proven.",
        f"- Detailed output: `{OUTPUTS['schema']}`",
        "",
        "## Competition mapping audit",
        f"- Requested league mappings: {', '.join(f'{k}->{v}' for k, v in LEAGUE_TO_TM_COMP.items())}",
        f"- Matched competition-season rows: {int(competition['mapping_status'].eq('matched').sum())}",
        f"- Detailed output: `{OUTPUTS['competition']}`",
        "",
        "## Club/team mapping audit",
        f"- Exact normalized team-season mappings used automatically: {exact_teams}",
        f"- Fuzzy alias candidates reported only, not applied silently: {fuzzy_teams}",
        f"- Ambiguous normalized matches not used automatically: {ambiguous}",
        f"- Detailed output: `{OUTPUTS['club_candidates']}`",
        "",
        "## Match mapping audit",
        f"- Mapped fixtures: {mapped}",
        f"- Unmatched fixtures: {unmatched}",
        f"- Overall exact-only fixture coverage: {mapped / total:.3f}" if total else "- Overall exact-only fixture coverage: n/a",
        f"- Detailed output: `{OUTPUTS['match_coverage']}`",
        "",
        "## Valuation date-safety audit",
        "- Player valuations are used only when valuation date is strictly before match_date.",
        "- A player counts for a club only with prior appearance evidence for that same club in the prior 365 days.",
        "- `players.current_club_*` and `player_valuations.current_club_*` are not used as historical membership proof.",
        "- Feasible metrics under this conservative rule: latest known player valuation, top 3/top 5/top 11 when enough prior-valued players exist, squad total, depth outside top 5, concentration, staleness, and missing valuation count.",
        f"- Detailed output: `{OUTPUTS['valuation']}`",
        "",
        "## Transfer date-safety audit",
        "- Arrivals/departures are assigned with `to_club_id`/`from_club_id` only when `transfer_date < match_date`.",
        "- Feasible metrics: 30/90/180/365 day arrival/departure counts, transfer-value sums, net value change, churn, biggest arrival, and biggest departure.",
        f"- Detailed output: `{OUTPUTS['transfer']}`",
        "",
        "## Historical squad assignment options",
        f"- Detailed output: `{OUTPUTS['squad_policy']}`",
        squad_policy.to_markdown(index=False),
        "",
        "## Recommended Transfermarkt v1 policy",
        f"- Classification: `{policy['recommended_status']}`",
        f"- Reason: {policy['reason']}",
        "- Conservative next gate: manual alias review, fixture remap, and strict-date feature-builder tests before predictive use.",
        f"- Detailed output: `{OUTPUTS['feature_policy']}`",
    ]
    OUTPUTS["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    schema = file_schema_audit()
    schema.to_csv(OUTPUTS["schema"], index=False)

    matches = load_feature_matrix_ids()
    competitions = pd.read_csv(TM_DIR / "competitions.csv", low_memory=False)
    games = pd.read_csv(
        TM_DIR / "games.csv",
        usecols=["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id", "home_club_name", "away_club_name"],
        low_memory=False,
    )
    games = games[games["competition_id"].isin(TM_COMP_TO_LEAGUE)].copy()
    competition = competition_audit(games, competitions, matches)
    competition.to_csv(OUTPUTS["competition"], index=False)

    clubs = pd.read_csv(TM_DIR / "clubs.csv", usecols=["club_id", "name", "domestic_competition_id"], low_memory=False)
    name_pool = club_name_pool(games, clubs)
    candidates, exact_mapping = build_club_candidates(matches, name_pool)
    candidates.to_csv(OUTPUTS["club_candidates"], index=False)

    mapped_fixtures, match_coverage = fixture_mapping(matches, games, exact_mapping)
    match_coverage.to_csv(OUTPUTS["match_coverage"], index=False)

    valuation = valuation_date_safety(mapped_fixtures)
    valuation.to_csv(OUTPUTS["valuation"], index=False)

    transfer = transfer_date_safety(mapped_fixtures)
    transfer.to_csv(OUTPUTS["transfer"], index=False)

    squad_policy = squad_assignment_policy(match_coverage, valuation)
    squad_policy.to_csv(OUTPUTS["squad_policy"], index=False)

    feature_policy = recommended_policy(match_coverage, valuation)
    feature_policy.to_csv(OUTPUTS["feature_policy"], index=False)

    write_markdown(schema, competition, candidates, match_coverage, valuation, transfer, squad_policy, feature_policy)


if __name__ == "__main__":
    main()
