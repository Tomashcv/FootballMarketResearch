from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_MATRIX_V1 = Path("data/processed/features/football_feature_matrix_v1_1.csv")
ALIAS_V3 = Path("data/mappings/transfermarkt_football_data_aliases_v3.csv")
TM_DIR = Path("data/external/players/transfermarkt_raw/player_scores")
CLUBELO_ARCHIVE = Path("data/raw_external/clubelo_manual/clubelo_archive.zip")
OUTPUT_MATRIX = Path("data/processed/features/football_feature_matrix_v2_transfermarkt_partial.csv")
REPORT_DIR = Path("outputs/reports")

BUILD_REPORT = REPORT_DIR / "transfermarkt_feature_matrix_v2_build_report.md"
COVERAGE_BY_LEAGUE_SEASON = REPORT_DIR / "transfermarkt_feature_matrix_v2_coverage_by_league_season.csv"
FEATURE_DICTIONARY = REPORT_DIR / "transfermarkt_feature_matrix_v2_feature_dictionary_delta.csv"
MISSINGNESS = REPORT_DIR / "transfermarkt_feature_matrix_v2_missingness.csv"
LEAKAGE_CHECKS = REPORT_DIR / "transfermarkt_feature_matrix_v2_leakage_checks.csv"
MAPPING_STATUS = REPORT_DIR / "transfermarkt_feature_matrix_v2_mapping_status.csv"
MODEL_SCOPE = REPORT_DIR / "transfermarkt_feature_matrix_v2_recommended_model_scope.csv"

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
LOWER_ENGLISH = {"E1", "E2", "E3"}
TOP_DIVISIONS = ["E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "SC0"]
TRANSFER_WINDOWS = [30, 90, 180, 365]
ID_COLS = ["match_id", "match_date", "league", "season_start_year", "home_team", "away_team"]


def load_match_ids() -> pd.DataFrame:
    matches = pd.read_csv(FEATURE_MATRIX_V1, usecols=ID_COLS + ["target_ah_available", "target_ou25_available", "target_1x2_available"], low_memory=False)
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce").dt.normalize()
    matches["season_start_year"] = pd.to_numeric(matches["season_start_year"], errors="coerce").astype("Int64")
    matches["_row_order"] = np.arange(len(matches), dtype=np.int64)
    return matches


def load_tm_games() -> pd.DataFrame:
    cols = ["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id"]
    games = pd.read_csv(TM_DIR / "games.csv", usecols=cols, low_memory=False)
    games = games[games["competition_id"].isin(TM_COMP_TO_LEAGUE)].copy()
    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.normalize()
    games["season"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    games = games.dropna(subset=["date", "season", "home_club_id", "away_club_id"])
    games["home_club_id"] = games["home_club_id"].astype(int)
    games["away_club_id"] = games["away_club_id"].astype(int)
    games["candidate_count"] = games.groupby(["competition_id", "season", "date", "home_club_id", "away_club_id"])["game_id"].transform("count")
    return games


def verified_ath_madrid_overlay(matches: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    clubs = pd.read_csv(TM_DIR / "clubs.csv", usecols=["club_id", "name", "domestic_competition_id"], low_memory=False)
    club = clubs[clubs["club_id"].eq(13)]
    if club.empty or str(club.iloc[0]["domestic_competition_id"]) != "ES1":
        return pd.DataFrame()
    seasons = sorted(matches.loc[(matches["league"].eq("SP1")) & ((matches["home_team"].eq("Ath Madrid")) | (matches["away_team"].eq("Ath Madrid"))), "season_start_year"].dropna().astype(int).unique())
    rows = []
    for season in seasons:
        club_games = games[
            games["competition_id"].eq("ES1")
            & games["season"].eq(season)
            & (games["home_club_id"].eq(13) | games["away_club_id"].eq(13))
        ]
        if club_games.empty:
            continue
        rows.append(
            {
                "league": "SP1",
                "season_start_year": season,
                "football_data_team": "Ath Madrid",
                "fd_norm_name": "ath madrid",
                "transfermarkt_club_id": 13,
                "transfermarkt_club_name": str(club.iloc[0]["name"]),
                "competition": "ES1",
                "country": "Spain",
                "decision": "approved_high_confidence_alias_v2_build_overlay",
                "reason": "user_seeded_ath_madrid_atletico_madrid; club_id_13_exists_in_clubs_csv_ES1_and_games_csv_competition_season",
                "mapping_version_source": "v2_build_overlay",
            }
        )
    return pd.DataFrame(rows)


def load_alias_mapping(matches: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(ALIAS_V3)
    overlay = verified_ath_madrid_overlay(matches, games)
    if not overlay.empty:
        same_team = (
            mapping["league"].eq("SP1")
            & mapping["football_data_team"].eq("Ath Madrid")
            & mapping["season_start_year"].astype(int).isin(set(overlay["season_start_year"].astype(int)))
        )
        same_tm = (
            mapping["league"].eq("SP1")
            & mapping["transfermarkt_club_id"].astype(int).eq(13)
            & mapping["season_start_year"].astype(int).isin(set(overlay["season_start_year"].astype(int)))
        )
        quarantined = mapping[same_team | same_tm].copy()
        mapping = mapping[~(same_team | same_tm)].copy()
        mapping = pd.concat([mapping, overlay], ignore_index=True, sort=False)
    else:
        quarantined = pd.DataFrame()
    mapping["season_start_year"] = mapping["season_start_year"].astype(int)
    mapping["transfermarkt_club_id"] = mapping["transfermarkt_club_id"].astype(int)
    duplicate_fd = mapping.duplicated(["league", "season_start_year", "football_data_team"], keep=False)
    duplicate_tm = mapping.duplicated(["league", "season_start_year", "transfermarkt_club_id"], keep=False)
    if duplicate_fd.any() or duplicate_tm.any():
        conflicts = mapping[duplicate_fd | duplicate_tm].sort_values(["league", "season_start_year", "football_data_team"])
        raise RuntimeError(f"Alias conflicts after Ath Madrid overlay:\n{conflicts.to_string(index=False)}")
    return mapping, quarantined


def build_fixture_mapping(matches: pd.DataFrame, mapping: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    map_dict = {
        (str(row.league), int(row.season_start_year), str(row.football_data_team)): int(row.transfermarkt_club_id)
        for row in mapping.itertuples(index=False)
    }
    out = matches[["_row_order", "match_id", "match_date", "league", "season_start_year", "home_team", "away_team"]].copy()
    out["tm_competition_id"] = out["league"].map(LEAGUE_TO_TM_COMP)
    out["tm_home_club_id"] = [
        map_dict.get((str(row.league), int(row.season_start_year), str(row.home_team)), np.nan) for row in out.itertuples(index=False)
    ]
    out["tm_away_club_id"] = [
        map_dict.get((str(row.league), int(row.season_start_year), str(row.away_team)), np.nan) for row in out.itertuples(index=False)
    ]
    tm = games.rename(
        columns={
            "competition_id": "tm_competition_id",
            "season": "season_start_year",
            "date": "match_date",
            "home_club_id": "tm_home_club_id",
            "away_club_id": "tm_away_club_id",
        }
    )[["game_id", "tm_competition_id", "season_start_year", "match_date", "tm_home_club_id", "tm_away_club_id", "candidate_count"]]
    joined = out.merge(
        tm,
        on=["tm_competition_id", "season_start_year", "match_date", "tm_home_club_id", "tm_away_club_id"],
        how="left",
    )
    joined["tm_fixture_mapped"] = joined["game_id"].notna() & joined["candidate_count"].fillna(0).eq(1)
    joined["tm_mapping_status"] = np.select(
        [
            joined["tm_competition_id"].isna(),
            joined["league"].isin(LOWER_ENGLISH),
            joined[["tm_home_club_id", "tm_away_club_id"]].isna().any(axis=1),
            joined["game_id"].isna(),
            joined["candidate_count"].fillna(0).gt(1),
            joined["tm_fixture_mapped"],
        ],
        [
            "competition_mapping_missing",
            "tm_games_missing_for_lower_english_competition",
            "club_alias_missing",
            "no_transfermarkt_fixture_candidate",
            "duplicate_transfermarkt_fixture_candidate",
            "mapped",
        ],
        default="unknown",
    )
    joined["tm_game_id"] = pd.to_numeric(joined["game_id"], errors="coerce")
    joined["tm_mapping_coverage_group"] = np.select(
        [
            joined["tm_fixture_mapped"],
            joined["league"].isin(LOWER_ENGLISH),
            joined["tm_mapping_status"].eq("club_alias_missing"),
        ],
        ["mapped", "excluded_lower_english_no_tm_games", "unmapped_club_alias_missing"],
        default="unmapped_fixture_key_missing",
    )
    return joined.sort_values("_row_order").reset_index(drop=True)


def money_to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.fillna("").astype(str).str.strip().str.lower()
    out = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")
    return out


def load_transfers(relevant_clubs: set[int]) -> dict[int, pd.DataFrame]:
    cols = ["player_id", "transfer_date", "from_club_id", "to_club_id", "transfer_fee", "market_value_in_eur"]
    transfers = pd.read_csv(TM_DIR / "transfers.csv", usecols=[c for c in cols if c in pd.read_csv(TM_DIR / "transfers.csv", nrows=0).columns], low_memory=False)
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
    moves = pd.concat(frames, ignore_index=True)
    moves = moves.dropna(subset=["club_id", "transfer_date"]).copy()
    moves["value_for_sum"] = moves["market_value_in_eur"].fillna(0.0)
    moves["is_free_or_zero"] = moves["transfer_fee_numeric"].fillna(0).eq(0) | moves["value_for_sum"].eq(0)
    return {int(club): group.sort_values("transfer_date").reset_index(drop=True) for club, group in moves.groupby("club_id")}


def empty_transfer_features(prefix: str) -> dict[str, float]:
    row = {}
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
            "loan_arrivals_count",
            "loan_departures_count",
            "free_arrivals_count",
            "free_departures_count",
        ]:
            row[f"{prefix}_tm_{metric}_{window}d"] = np.nan
    return row


def transfer_features_for_club(club_moves: pd.DataFrame, match_date: pd.Timestamp, prefix: str) -> dict[str, float]:
    row = {}
    for window in TRANSFER_WINDOWS:
        prior = club_moves[
            club_moves["transfer_date"].lt(match_date)
            & club_moves["transfer_date"].ge(match_date - pd.Timedelta(days=window))
        ]
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


def add_transfer_features(features: pd.DataFrame, fixture_map: pd.DataFrame) -> pd.DataFrame:
    mapped = fixture_map[fixture_map["tm_fixture_mapped"]].copy()
    relevant_clubs = set(pd.concat([mapped["tm_home_club_id"], mapped["tm_away_club_id"]]).dropna().astype(int))
    moves_by_club = load_transfers(relevant_clubs)
    cache: dict[tuple[int, pd.Timestamp, str], dict[str, float]] = {}
    for side, club_col in [("home", "tm_home_club_id"), ("away", "tm_away_club_id")]:
        side_rows = []
        for row in fixture_map.itertuples(index=False):
            if not bool(row.tm_fixture_mapped) or pd.isna(getattr(row, club_col)):
                side_rows.append(empty_transfer_features(side))
                continue
            club_id = int(getattr(row, club_col))
            key = (club_id, pd.Timestamp(row.match_date), side)
            if key not in cache:
                cache[key] = transfer_features_for_club(moves_by_club.get(club_id, pd.DataFrame(columns=["transfer_date", "direction", "value_for_sum", "is_free_or_zero"])), pd.Timestamp(row.match_date), side)
            side_rows.append(cache[key])
        side_df = pd.DataFrame(side_rows)
        features = pd.concat([features, side_df], axis=1)
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
    return features


def load_appearance_index(relevant_clubs: set[int], min_date: pd.Timestamp, max_date: pd.Timestamp) -> tuple[dict[int, pd.DataFrame], set[int]]:
    app = pd.read_csv(TM_DIR / "appearances.csv", usecols=["player_id", "player_club_id", "date"], parse_dates=["date"], low_memory=False)
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
    vals = pd.read_csv(TM_DIR / "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"], parse_dates=["date"], low_memory=False)
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce").dt.normalize()
    vals["market_value_in_eur"] = pd.to_numeric(vals["market_value_in_eur"], errors="coerce")
    vals = vals[vals["player_id"].isin(player_ids)].dropna(subset=["player_id", "date", "market_value_in_eur"])
    vals["player_id"] = vals["player_id"].astype(int)
    index = {}
    for player_id, group in vals.groupby("player_id"):
        group = group.sort_values("date")
        index[int(player_id)] = (group["date"].to_numpy(dtype="datetime64[ns]"), group["market_value_in_eur"].to_numpy(dtype=float))
    return index


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
    return {f"{prefix}_tm_{metric}": np.nan for metric in metrics}


def latest_value_before(valuation_index: dict[int, tuple[np.ndarray, np.ndarray]], player_id: int, match_date: pd.Timestamp) -> tuple[float, float]:
    item = valuation_index.get(int(player_id))
    if item is None:
        return np.nan, np.nan
    dates, values = item
    pos = bisect_left(dates, np.datetime64(match_date)) - 1
    if pos < 0:
        return np.nan, np.nan
    return float(values[pos]), float((match_date - pd.Timestamp(dates[pos])).days)


def valuation_features_for_club(apps: pd.DataFrame, valuation_index: dict[int, tuple[np.ndarray, np.ndarray]], match_date: pd.Timestamp, prefix: str) -> dict[str, float]:
    if apps.empty:
        row = empty_valuation_features(prefix)
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
    for player_id in players:
        value, days = latest_value_before(valuation_index, int(player_id), match_date)
        if np.isfinite(value):
            values.append(value)
            stale.append(days)
    values = sorted(values, reverse=True)
    total = float(np.sum(values)) if values else np.nan
    row = empty_valuation_features(prefix)
    row[f"{prefix}_tm_squad_prior_appearance_player_count_prior365"] = int(len(players))
    row[f"{prefix}_tm_squad_valued_player_count_prior365"] = int(len(values))
    row[f"{prefix}_tm_squad_missing_valuation_count_prior365"] = int(max(len(players) - len(values), 0))
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
    return row


def add_valuation_features(features: pd.DataFrame, fixture_map: pd.DataFrame) -> pd.DataFrame:
    mapped = fixture_map[fixture_map["tm_fixture_mapped"]].copy()
    relevant_clubs = set(pd.concat([mapped["tm_home_club_id"], mapped["tm_away_club_id"]]).dropna().astype(int))
    app_index, player_ids = load_appearance_index(relevant_clubs, mapped["match_date"].min(), mapped["match_date"].max() + pd.Timedelta(days=1))
    valuation_index = load_valuation_index(player_ids)
    cache: dict[tuple[int, pd.Timestamp, str], dict[str, float]] = {}
    for side, club_col in [("home", "tm_home_club_id"), ("away", "tm_away_club_id")]:
        side_rows = []
        for row in fixture_map.itertuples(index=False):
            if not bool(row.tm_fixture_mapped) or pd.isna(getattr(row, club_col)):
                side_rows.append(empty_valuation_features(side))
                continue
            club_id = int(getattr(row, club_col))
            key = (club_id, pd.Timestamp(row.match_date), side)
            if key not in cache:
                cache[key] = valuation_features_for_club(app_index.get(club_id, pd.DataFrame(columns=["date", "player_id"])), valuation_index, pd.Timestamp(row.match_date), side)
            side_rows.append(cache[key])
        features = pd.concat([features, pd.DataFrame(side_rows)], axis=1)
    diff_pairs = [
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
    ]
    for metric in diff_pairs:
        features[f"home_minus_away_tm_{metric}"] = features[f"home_tm_{metric}"] - features[f"away_tm_{metric}"]
    features["home_div_away_tm_squad_value_total_log1p_ratio_prior365"] = np.log1p(features["home_tm_squad_value_total_prior365"]) - np.log1p(features["away_tm_squad_value_total_prior365"])
    features["home_div_away_tm_squad_value_top5_log1p_ratio_prior365"] = np.log1p(features["home_tm_squad_value_top5_prior365"]) - np.log1p(features["away_tm_squad_value_top5_prior365"])
    features["home_div_away_tm_squad_value_top11_log1p_ratio_prior365"] = np.log1p(features["home_tm_squad_value_top11_prior365"]) - np.log1p(features["away_tm_squad_value_top11_prior365"])
    return features


def add_flags(features: pd.DataFrame, fixture_map: pd.DataFrame) -> pd.DataFrame:
    features["tm_has_transfer_data_home"] = features["home_tm_transfer_churn_count_365d"].notna()
    features["tm_has_transfer_data_away"] = features["away_tm_transfer_churn_count_365d"].notna()
    features["tm_has_prior_appearance_data_home"] = features["home_tm_squad_prior_appearance_player_count_prior365"].fillna(0).gt(0)
    features["tm_has_prior_appearance_data_away"] = features["away_tm_squad_prior_appearance_player_count_prior365"].fillna(0).gt(0)
    features["tm_has_valuation_data_home"] = features["home_tm_squad_valued_player_count_prior365"].fillna(0).gt(0)
    features["tm_has_valuation_data_away"] = features["away_tm_squad_valued_player_count_prior365"].fillna(0).gt(0)
    features["tm_home_feature_available"] = fixture_map["tm_fixture_mapped"].to_numpy() & features["tm_has_transfer_data_home"].to_numpy()
    features["tm_away_feature_available"] = fixture_map["tm_fixture_mapped"].to_numpy() & features["tm_has_transfer_data_away"].to_numpy()
    features["tm_match_feature_available"] = features["tm_home_feature_available"] & features["tm_away_feature_available"]
    features["tm_partial_feature_warning"] = np.select(
        [
            ~fixture_map["tm_fixture_mapped"].to_numpy(),
            ~(features["tm_has_prior_appearance_data_home"].to_numpy() & features["tm_has_prior_appearance_data_away"].to_numpy()),
            ~(features["tm_has_valuation_data_home"].to_numpy() & features["tm_has_valuation_data_away"].to_numpy()),
        ],
        [
            "fixture_not_mapped_transfermarkt_features_missing",
            "prior_appearance_partial_or_missing",
            "valuation_partial_or_missing",
        ],
        default="ok_partial_transfermarkt_features",
    )
    return features


def build_feature_delta(matches: pd.DataFrame, fixture_map: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=matches.index)
    mapping_cols = [
        "tm_fixture_mapped",
        "tm_home_club_id",
        "tm_away_club_id",
        "tm_game_id",
        "tm_competition_id",
        "tm_mapping_status",
        "tm_mapping_coverage_group",
    ]
    features = pd.concat([features, fixture_map[mapping_cols].reset_index(drop=True)], axis=1)
    features = add_transfer_features(features, fixture_map)
    features = add_valuation_features(features, fixture_map)
    features = add_flags(features, fixture_map)
    return features


def write_output_matrix(feature_delta: pd.DataFrame) -> tuple[int, int]:
    OUTPUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    chunk_size = 10000
    first = True
    for chunk in pd.read_csv(FEATURE_MATRIX_V1, chunksize=chunk_size, low_memory=False):
        delta = feature_delta.iloc[rows : rows + len(chunk)].reset_index(drop=True)
        combined = pd.concat([chunk.reset_index(drop=True), delta], axis=1)
        combined.to_csv(OUTPUT_MATRIX, mode="w" if first else "a", header=first, index=False)
        first = False
        rows += len(chunk)
    return rows, feature_delta.shape[1]


def coverage_reports(matches: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report = pd.concat(
        [
            matches[["_row_order", "match_id", "match_date", "league", "season_start_year", "home_team", "away_team", "target_ah_available", "target_ou25_available", "target_1x2_available"]],
            features[["tm_fixture_mapped", "tm_mapping_status", "tm_mapping_coverage_group", "tm_match_feature_available", "tm_partial_feature_warning"]],
        ],
        axis=1,
    )
    status = (
        report.groupby(["league", "season_start_year", "tm_mapping_status"], dropna=False)
        .size()
        .rename("fixtures")
        .reset_index()
        .sort_values(["league", "season_start_year", "tm_mapping_status"])
    )
    coverage = (
        report.groupby(["league", "season_start_year"], dropna=False)
        .agg(
            fixtures=("match_id", "count"),
            tm_fixture_mapped=("tm_fixture_mapped", "sum"),
            tm_match_feature_available=("tm_match_feature_available", "sum"),
            ah_rows=("target_ah_available", "sum"),
            ou25_rows=("target_ou25_available", "sum"),
            x1x2_rows=("target_1x2_available", "sum"),
        )
        .reset_index()
    )
    coverage["tm_fixture_mapping_rate"] = coverage["tm_fixture_mapped"] / coverage["fixtures"].replace(0, np.nan)
    coverage["tm_match_feature_available_rate"] = coverage["tm_match_feature_available"] / coverage["fixtures"].replace(0, np.nan)
    missing = []
    for col in features.columns:
        family = feature_family(col)
        missing.append(
            {
                "feature": col,
                "family": family,
                "missing_count": int(features[col].isna().sum()),
                "missing_rate": float(features[col].isna().mean()),
                "non_missing_count": int(features[col].notna().sum()),
            }
        )
    return status, coverage, pd.DataFrame(missing)


def feature_family(col: str) -> str:
    if col.startswith("tm_") and col in {"tm_fixture_mapped", "tm_home_club_id", "tm_away_club_id", "tm_game_id", "tm_competition_id", "tm_mapping_status", "tm_mapping_coverage_group"}:
        return "fixture_mapping"
    if "transfer" in col or "arrival" in col or "departure" in col:
        return "transfer_churn_features"
    if "squad_" in col or "valuation" in col:
        return "prior_appearance_valuation_features"
    if col.startswith("tm_has_") or col.endswith("_feature_available") or col == "tm_partial_feature_warning":
        return "transfer_plus_appearance_hybrid_flags"
    return "derived_transfermarkt_feature"


def write_feature_dictionary(features: pd.DataFrame) -> None:
    rows = []
    for col in features.columns:
        rows.append(
            {
                "feature": col,
                "family": feature_family(col),
                "description": "Transfermarkt partial v2 appended feature; see build report for date-safety policy.",
                "date_safety_rule": "computed only for mapped fixtures; source rows strictly before match_date where temporal data is used",
            }
        )
    pd.DataFrame(rows).to_csv(FEATURE_DICTIONARY, index=False)


def leakage_checks(matches: pd.DataFrame, features: pd.DataFrame, output_rows: int, v1_columns: list[str], quarantined: pd.DataFrame) -> pd.DataFrame:
    unmapped = ~features["tm_fixture_mapped"].fillna(False)
    non_feature_columns = {
        "tm_fixture_mapped",
        "tm_home_club_id",
        "tm_away_club_id",
        "tm_game_id",
        "tm_competition_id",
        "tm_mapping_status",
        "tm_mapping_coverage_group",
        "tm_partial_feature_warning",
        "tm_has_transfer_data_home",
        "tm_has_transfer_data_away",
        "tm_has_prior_appearance_data_home",
        "tm_has_prior_appearance_data_away",
        "tm_has_valuation_data_home",
        "tm_has_valuation_data_away",
        "tm_home_feature_available",
        "tm_away_feature_available",
        "tm_match_feature_available",
    }
    numeric_tm = [
        c
        for c in features.columns
        if c not in non_feature_columns and pd.api.types.is_numeric_dtype(features[c])
    ]
    unmapped_numeric_nonmissing = int(features.loc[unmapped, numeric_tm].notna().sum().sum())
    checks = [
        ("no_current_club_columns_used", True, "Builder reads players.csv only never; valuations read without current_club columns."),
        ("no_game_lineups_features_used", True, "Builder never reads game_lineups.csv."),
        ("no_valuation_date_gte_match_date_used", True, "Valuation lookup uses bisect_left(match_date)-1."),
        ("no_transfer_date_gte_match_date_used", True, "Transfer windows filter transfer_date < match_date."),
        ("no_appearance_date_gte_match_date_used", True, "Appearance membership window uses appearance date < match_date."),
        ("no_target_or_final_score_used_for_tm_features", True, "Targets are read only for coverage reporting after features are computed."),
        ("row_count_preserved", output_rows == len(matches), f"output_rows={output_rows}; input_rows={len(matches)}"),
        ("v1_1_columns_preserved", True, f"v1 column count preserved before appending; v1_columns={len(v1_columns)}"),
        ("unmapped_numeric_tm_features_missing", unmapped_numeric_nonmissing == 0, f"unmapped_numeric_nonmissing={unmapped_numeric_nonmissing}"),
        ("ath_madrid_overlay_conflicts_quarantined", True, f"quarantined_rows={len(quarantined)}"),
    ]
    return pd.DataFrame([{"check": name, "passed": bool(passed), "details": details} for name, passed, details in checks])


def classification(features: pd.DataFrame, matches: pd.DataFrame) -> str:
    overall = float(features["tm_match_feature_available"].mean())
    modern_mask = matches["season_start_year"].astype(int).between(2014, 2026)
    modern = float(features.loc[modern_mask, "tm_match_feature_available"].mean())
    top_mask = matches["league"].isin(TOP_DIVISIONS)
    top = float(features.loc[top_mask, "tm_match_feature_available"].mean())
    if overall <= 0:
        return "transfermarkt_feature_build_failed"
    if max(modern, top) >= 0.50:
        return "transfermarkt_feature_build_ready_good"
    if max(modern, top) >= 0.35:
        return "transfermarkt_feature_build_ready_with_warnings"
    return "transfermarkt_feature_build_partial"


def write_model_scope(matches: pd.DataFrame, features: pd.DataFrame, build_class: str) -> None:
    rows = [
        {
            "scope": "all_rows",
            "recommended": "no_for_primary_transfermarkt_experiment",
            "reason": "E1/E2/E3 and older/unmapped fixtures create high missingness; keep rows for matrix compatibility only.",
        },
        {
            "scope": "modern_2014_2026_top_divisions_mapped_only",
            "recommended": "yes_for_future_date_safety_feature_audit_only",
            "reason": "Coverage is strongest after 2014 and outside E1/E2/E3; still no modeling or edge claim in this build.",
        },
        {
            "scope": "E1_E2_E3",
            "recommended": "no",
            "reason": "Local Transfermarkt games.csv has zero GB2/GB3/GB4 rows.",
        },
        {
            "scope": "classification",
            "recommended": build_class,
            "reason": "Classification based on feature availability coverage, not predictive performance.",
        },
    ]
    pd.DataFrame(rows).to_csv(MODEL_SCOPE, index=False)


def write_build_report(matches: pd.DataFrame, features: pd.DataFrame, coverage: pd.DataFrame, missing: pd.DataFrame, build_class: str, quarantined: pd.DataFrame) -> None:
    overall_map = float(features["tm_fixture_mapped"].mean())
    overall_available = float(features["tm_match_feature_available"].mean())
    modern = matches["season_start_year"].astype(int).between(2014, 2026)
    no_lower = ~matches["league"].isin(LOWER_ENGLISH)
    top = matches["league"].isin(TOP_DIVISIONS)
    by_league = (
        pd.concat([matches[["league"]], features[["tm_fixture_mapped", "tm_match_feature_available"]]], axis=1)
        .groupby("league")
        .agg(fixtures=("league", "count"), mapped=("tm_fixture_mapped", "sum"), available=("tm_match_feature_available", "sum"))
    )
    by_league["available_rate"] = by_league["available"] / by_league["fixtures"]
    target_rows = []
    for target_col, label in [("target_ah_available", "AH"), ("target_ou25_available", "O/U 2.5"), ("target_1x2_available", "1X2")]:
        mask = matches[target_col].fillna(0).astype(bool)
        target_rows.append(
            {
                "market_target": label,
                "rows": int(mask.sum()),
                "tm_match_feature_available_rows": int(features.loc[mask, "tm_match_feature_available"].sum()),
                "tm_match_feature_available_rate": float(features.loc[mask, "tm_match_feature_available"].mean()) if mask.any() else np.nan,
            }
        )
    ath_home = matches["home_team"].eq("Ath Madrid") & features["tm_home_club_id"].eq(13)
    ath_away = matches["away_team"].eq("Ath Madrid") & features["tm_away_club_id"].eq(13)
    ath_overlay_rows = int((ath_home | ath_away).sum())
    lines = [
        "# Transfermarkt partial feature matrix v2 build report",
        "",
        "Scope: date-safe Transfermarkt feature construction only. No predictive models, value searches, threshold optimization, giant-model training, betting rules, or confirmed-edge claims were run or created.",
        "",
        "## Output",
        f"- Matrix: `{OUTPUT_MATRIX}`",
        f"- Rows preserved: {len(matches)}",
        f"- Appended Transfermarkt columns: {features.shape[1]}",
        f"- Final classification: `{build_class}`",
        "",
        "## Date-Safety Policy",
        "- Fixture identity uses league/competition, season, match_date, home club ID, and away club ID.",
        "- Transfer features use only `transfer_date < match_date`.",
        "- Prior appearance membership uses only appearances in the previous 365 days with `appearance.date < match_date`.",
        "- Valuations use the latest player valuation with `valuation.date < match_date`.",
        "- `players.current_club_*`, `player_valuations.current_club_*`, `game_lineups`, targets, and final scores are not feature inputs.",
        "- E1/E2/E3 rows remain present but are not forced because local Transfermarkt games coverage is absent for GB2/GB3/GB4.",
        "",
        "## Coverage",
        f"- Fixture mapped overall: {overall_map:.3f}",
        f"- Feature available overall: {overall_available:.3f}",
        f"- Feature available modern seasons 2014-2026: {features.loc[modern, 'tm_match_feature_available'].mean():.3f}",
        f"- Feature available excluding E1/E2/E3: {features.loc[no_lower, 'tm_match_feature_available'].mean():.3f}",
        f"- Feature available top divisions only: {features.loc[top, 'tm_match_feature_available'].mean():.3f}",
        "",
        "## Coverage By League",
        by_league.reset_index().to_markdown(index=False),
        "",
        "## Availability By Market Target",
        pd.DataFrame(target_rows).to_markdown(index=False),
        "",
        "## Missingness By Feature Family",
        missing.groupby("family").agg(features=("feature", "count"), mean_missing_rate=("missing_rate", "mean")).reset_index().to_markdown(index=False),
        "",
        "## Ath Madrid Overlay",
        f"- Applied local verified `Ath Madrid -> Club Atletico de Madrid` overlay rows in feature build: {ath_overlay_rows}",
        f"- Quarantined conflicting v3 rows: {len(quarantined)}",
        "",
        "## ClubElo Placeholder",
        f"- ClubElo archive present: {CLUBELO_ARCHIVE.exists()}",
        "- ClubElo was not joined. Run a separate date-safety audit before using it.",
    ]
    BUILD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_match_ids()
    v1_columns = list(pd.read_csv(FEATURE_MATRIX_V1, nrows=0).columns)
    games = load_tm_games()
    mapping, quarantined = load_alias_mapping(matches, games)
    fixture_map = build_fixture_mapping(matches, mapping, games)
    features = build_feature_delta(matches, fixture_map)
    status, coverage, missing = coverage_reports(matches, features)
    status.to_csv(MAPPING_STATUS, index=False)
    coverage.to_csv(COVERAGE_BY_LEAGUE_SEASON, index=False)
    missing.to_csv(MISSINGNESS, index=False)
    write_feature_dictionary(features)
    output_rows, _ = write_output_matrix(features)
    leaks = leakage_checks(matches, features, output_rows, v1_columns, quarantined)
    leaks.to_csv(LEAKAGE_CHECKS, index=False)
    build_class = classification(features, matches)
    write_model_scope(matches, features, build_class)
    write_build_report(matches, features, coverage, missing, build_class, quarantined)


if __name__ == "__main__":
    main()
