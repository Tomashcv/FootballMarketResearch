from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


TM_DIR = Path("data/external/players/transfermarkt_raw/player_scores")
FEATURE_MATRIX = Path("data/processed/features/football_feature_matrix_v1_1.csv")
V1_MAPPING = Path("data/mappings/transfermarkt_football_data_aliases_v1.csv")
REPORT_DIR = Path("outputs/reports")
MAPPING_DIR = Path("data/mappings")

ALIASES_V2 = MAPPING_DIR / "transfermarkt_football_data_aliases_v2.csv"
TARGET_CANDIDATES = REPORT_DIR / "transfermarkt_targeted_alias_candidates_v2.csv"
DELTA_APPROVED = REPORT_DIR / "transfermarkt_aliases_approved_v2_delta.csv"
MANUAL_V2 = REPORT_DIR / "transfermarkt_aliases_manual_review_required_v2.csv"
CLUB_COVERAGE = REPORT_DIR / "transfermarkt_club_mapping_coverage_after_alias_v2.csv"
FIXTURE_COVERAGE = REPORT_DIR / "transfermarkt_fixture_mapping_coverage_after_alias_v2.csv"
UNMATCHED_V2 = REPORT_DIR / "transfermarkt_unmatched_teams_after_alias_v2.csv"
E123_DIAGNOSIS = REPORT_DIR / "transfermarkt_e1_e2_e3_zero_coverage_diagnosis_v2.csv"
MARKDOWN = REPORT_DIR / "transfermarkt_targeted_alias_review_v2.md"

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

TARGET_TEAMS = {
    "Club Brugge",
    "PSV Eindhoven",
    "Benfica",
    "Sp Lisbon",
    "Barcelona",
    "Real Madrid",
    "M'gladbach",
    "Guimaraes",
    "PAOK",
    "Hearts",
    "Espanol",
    "St Truiden",
    "Cardiff",
    "Derby",
    "QPR",
    "AEK",
    "Preston",
    "Buyuksehyr",
    "Bristol City",
    "Ipswich",
    "Sheffield Weds",
    "Birmingham",
    "Middlesbrough",
    "Millwall",
    "Reading",
    "Morecambe",
    "NAC Breda",
    "Hull",
    "Leeds",
    "Nott'm Forest",
    "Paris SG",
    "Bayern Munich",
    "Ein Frankfurt",
    "FC Koln",
}

TARGET_EXPANSIONS = {
    "sp lisbon": "sporting clube portugal sporting cp lisboa",
    "m gladbach": "borussia monchengladbach",
    "espanol": "espanyol barcelona",
    "paris sg": "paris saint germain",
    "bayern munich": "bayern munchen",
    "ein frankfurt": "eintracht frankfurt",
    "fc koln": "koln cologne",
    "nott m forest": "nottingham forest",
    "sheffield weds": "sheffield wednesday",
    "buyuksehyr": "istanbul basaksehir buyuksehir",
    "guimaraes": "vitoria guimaraes",
    "hearts": "heart midlothian",
    "aek": "aek athens athina",
    "club brugge": "club brugge",
    "psv eindhoven": "psv eindhoven philips",
    "benfica": "benfica sport lisboa",
    "st truiden": "sint truiden truidense",
    "nac breda": "nac breda",
    "qpr": "queens park rangers",
}

LEGAL_TOKENS = {
    "a",
    "ac",
    "afc",
    "ag",
    "as",
    "association",
    "associacao",
    "associazione",
    "balompie",
    "calcio",
    "cf",
    "club",
    "de",
    "del",
    "fc",
    "football",
    "futbol",
    "futebol",
    "fussball",
    "fußball",
    "koninklijke",
    "real",
    "royal",
    "royale",
    "sad",
    "sc",
    "soccer",
    "sociedade",
    "societa",
    "sport",
    "sporting",
    "sportiva",
    "the",
    "ud",
    "verein",
    "vereniging",
    "voetbal",
    "voetbalclub",
}


def strip_accents(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def raw_key(value: object) -> str:
    text = strip_accents(value).casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def tokens(value: object, keep_real: bool = False) -> list[str]:
    legal = LEGAL_TOKENS - ({"real"} if keep_real else set())
    out = []
    for token in raw_key(value).split():
        if len(token) == 4 and token.isdigit():
            continue
        if token in legal:
            continue
        out.append(token)
    return out


def expansion_for_team(team: str) -> str:
    return TARGET_EXPANSIONS.get(raw_key(team), "")


def similarity(fd_team: str, tm_name: str) -> tuple[float, str, bool]:
    keep_real = raw_key(fd_team) == "real madrid"
    fd_base = tokens(fd_team, keep_real=keep_real)
    fd_expanded = tokens(expansion_for_team(fd_team), keep_real=keep_real)
    tm_tokens = tokens(tm_name, keep_real=keep_real)
    fd_tokens = list(dict.fromkeys(fd_base + fd_expanded))
    if not fd_tokens or not tm_tokens:
        return 0.0, "empty_tokens", False
    fd_set = set(fd_tokens)
    tm_set = set(tm_tokens)
    seq = SequenceMatcher(None, " ".join(fd_tokens), " ".join(tm_tokens)).ratio()
    inter = len(fd_set & tm_set)
    union = len(fd_set | tm_set)
    jaccard = inter / union if union else 0.0
    containment = inter / len(fd_set) if fd_set else 0.0
    reverse = inter / len(tm_set) if tm_set else 0.0
    alias_hit = bool(fd_expanded and len(set(fd_expanded) & tm_set) >= min(2, len(set(fd_expanded))))
    if raw_key(fd_team) in {"aek", "paok", "qpr"}:
        alias_hit = bool(set(fd_base + fd_expanded) & tm_set)
    score = max(seq, jaccard, containment * 0.97, reverse * 0.92, 0.94 if alias_hit else 0.0)
    details = (
        f"seq={seq:.3f};jaccard={jaccard:.3f};fd_token_containment={containment:.3f};"
        f"tm_token_containment={reverse:.3f};targeted_expansion_hit={alias_hit}"
    )
    return round(float(score), 4), details, alias_hit


def load_matches() -> pd.DataFrame:
    cols = ["match_id", "match_date", "league", "season_start_year", "home_team", "away_team"]
    matches = pd.read_csv(FEATURE_MATRIX, usecols=cols, low_memory=False)
    matches = matches[matches["league"].isin(LEAGUES)].copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce").dt.normalize()
    matches["season_start_year"] = pd.to_numeric(matches["season_start_year"], errors="coerce").astype("Int64")
    return matches.dropna(subset=["match_date", "league", "season_start_year", "home_team", "away_team"]).reset_index(drop=True)


def load_tm_games() -> pd.DataFrame:
    cols = ["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id", "home_club_name", "away_club_name"]
    games = pd.read_csv(TM_DIR / "games.csv", usecols=cols, low_memory=False)
    games = games[games["competition_id"].isin(TM_COMP_TO_LEAGUE)].copy()
    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.normalize()
    games["league"] = games["competition_id"].map(TM_COMP_TO_LEAGUE)
    games["season"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    return games.dropna(subset=["date", "league", "season", "home_club_id", "away_club_id"])


def team_seasons(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["league", "season_start_year", "home_team"]].rename(columns={"home_team": "football_data_team"})
    away = matches[["league", "season_start_year", "away_team"]].rename(columns={"away_team": "football_data_team"})
    teams = pd.concat([home, away], ignore_index=True).drop_duplicates()
    teams["fd_key"] = teams["football_data_team"].map(raw_key)
    return teams.sort_values(["league", "season_start_year", "football_data_team"]).reset_index(drop=True)


def build_tm_pool(games: pd.DataFrame, clubs: pd.DataFrame, competitions: pd.DataFrame) -> pd.DataFrame:
    home = games[["league", "competition_id", "season", "home_club_id", "home_club_name"]].rename(
        columns={"home_club_id": "tm_club_id", "home_club_name": "tm_club_name"}
    )
    away = games[["league", "competition_id", "season", "away_club_id", "away_club_name"]].rename(
        columns={"away_club_id": "tm_club_id", "away_club_name": "tm_club_name"}
    )
    side_names = pd.concat([home, away], ignore_index=True)
    club_names = clubs[["club_id", "name", "domestic_competition_id"]].rename(
        columns={"club_id": "tm_club_id", "name": "tm_club_name", "domestic_competition_id": "competition_id"}
    )
    club_names["league"] = club_names["competition_id"].map(TM_COMP_TO_LEAGUE)
    club_names["season"] = pd.NA
    pool = pd.concat([side_names, club_names], ignore_index=True)
    pool = pool.dropna(subset=["league", "competition_id", "tm_club_id", "tm_club_name"]).copy()
    comp_meta = competitions[["competition_id", "country_name", "name"]].rename(
        columns={"country_name": "country", "name": "competition_name"}
    )
    pool = pool.merge(comp_meta, on="competition_id", how="left")
    pool["tm_club_id"] = pd.to_numeric(pool["tm_club_id"], errors="coerce").astype("Int64")
    return pool.dropna(subset=["tm_club_id"]).drop_duplicates().reset_index(drop=True)


def v1_mapping_dict(v1: pd.DataFrame) -> dict[tuple[str, int, str], int]:
    return {
        (str(row.league), int(row.season_start_year), str(row.football_data_team)): int(row.transfermarkt_club_id)
        for row in v1.itertuples(index=False)
    }


def fixture_evidence(matches: pd.DataFrame, games: pd.DataFrame, mapping: dict[tuple[str, int, str], int]) -> dict[tuple[str, int, str, int], int]:
    tm_by_side: dict[tuple[str, int, pd.Timestamp, int], set[int]] = defaultdict(set)
    candidates_by_opponent: dict[tuple[str, int, pd.Timestamp, int], set[int]] = defaultdict(set)
    for game in games.itertuples(index=False):
        league = str(game.league)
        season = int(game.season)
        date = pd.Timestamp(game.date)
        home_id = int(game.home_club_id)
        away_id = int(game.away_club_id)
        tm_by_side[(league, season, date, home_id)].add(away_id)
        tm_by_side[(league, season, date, away_id)].add(home_id)
        candidates_by_opponent[(league, season, date, away_id)].add(home_id)
        candidates_by_opponent[(league, season, date, home_id)].add(away_id)
    counts: dict[tuple[str, int, str, int], int] = defaultdict(int)
    for match in matches.itertuples(index=False):
        season = int(match.season_start_year)
        date = pd.Timestamp(match.match_date)
        home_key = (str(match.league), season, str(match.home_team))
        away_key = (str(match.league), season, str(match.away_team))
        home_id = mapping.get(home_key)
        away_id = mapping.get(away_key)
        if home_id is not None and away_id is not None:
            if away_id in tm_by_side.get((str(match.league), season, date, home_id), set()):
                counts[(str(match.league), season, str(match.home_team), home_id)] += 1
                counts[(str(match.league), season, str(match.away_team), away_id)] += 1
        # Candidate-side evidence for unmapped target paired with mapped opponent.
        if away_id is not None:
            for home_candidate in candidates_by_opponent.get((str(match.league), season, date, away_id), set()):
                counts[(str(match.league), season, str(match.home_team), int(home_candidate))] += 1
        if home_id is not None:
            for away_candidate in candidates_by_opponent.get((str(match.league), season, date, home_id), set()):
                counts[(str(match.league), season, str(match.away_team), int(away_candidate))] += 1
    return counts


def build_target_candidates(teams: pd.DataFrame, tm_pool: pd.DataFrame, evidence: dict[tuple[str, int, str, int], int]) -> pd.DataFrame:
    target_keys = {raw_key(team) for team in TARGET_TEAMS}
    target_team_seasons = teams[teams["fd_key"].isin(target_keys)].copy()
    rows = []
    for team in target_team_seasons.itertuples(index=False):
        pool = tm_pool[
            tm_pool["league"].eq(team.league)
            & (tm_pool["season"].isna() | tm_pool["season"].eq(team.season_start_year))
        ].copy()
        by_club = {}
        for cand in pool.itertuples(index=False):
            score, details, alias_hit = similarity(str(team.football_data_team), str(cand.tm_club_name))
            club_id = int(cand.tm_club_id)
            item = {
                "league": str(team.league),
                "season_start_year": int(team.season_start_year),
                "football_data_team": str(team.football_data_team),
                "candidate_transfermarkt_club_id": club_id,
                "candidate_transfermarkt_club_name": str(cand.tm_club_name),
                "country": cand.country,
                "competition": cand.competition_id,
                "competition_name": cand.competition_name,
                "fuzzy_score": score,
                "score_details": details,
                "targeted_expansion": expansion_for_team(str(team.football_data_team)),
                "targeted_expansion_hit": alias_hit,
                "fixture_evidence_count": int(evidence.get((str(team.league), int(team.season_start_year), str(team.football_data_team), club_id), 0)),
            }
            if club_id not in by_club or (score, item["fixture_evidence_count"]) > (
                by_club[club_id]["fuzzy_score"],
                by_club[club_id]["fixture_evidence_count"],
            ):
                by_club[club_id] = item
        scored = sorted(by_club.values(), key=lambda x: (x["fuzzy_score"], x["fixture_evidence_count"]), reverse=True)
        rows.extend(scored[:8])
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    top_by_season = candidates.sort_values(["fuzzy_score", "fixture_evidence_count"], ascending=False).groupby(
        ["league", "season_start_year", "football_data_team"], as_index=False
    ).head(1)
    stability = (
        top_by_season.groupby(["league", "football_data_team", "candidate_transfermarkt_club_id"])["season_start_year"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    out = []
    for key, group in candidates.groupby(["league", "season_start_year", "football_data_team"], dropna=False):
        group = group.sort_values(["fuzzy_score", "fixture_evidence_count"], ascending=False).copy()
        top = group.iloc[0].to_dict()
        second = float(group.iloc[1]["fuzzy_score"]) if len(group) > 1 else 0.0
        competing = int((group["fuzzy_score"].ge(max(0.86, float(top["fuzzy_score"]) - 0.03))).sum() - 1)
        seasons_supported = int(stability.get((top["league"], top["football_data_team"], top["candidate_transfermarkt_club_id"]), 0))
        support = int(top["fixture_evidence_count"])
        signals = []
        if top["competition"] == LEAGUE_TO_TM_COMP.get(str(top["league"])):
            signals.append("same_competition_country")
        if float(top["fuzzy_score"]) >= 0.90:
            signals.append("high_targeted_similarity")
        if bool(top["targeted_expansion_hit"]):
            signals.append("known_abbreviation_expansion")
        if support >= 2:
            signals.append("fixture_date_opponent_evidence")
        if competing == 0 and float(top["fuzzy_score"]) - second >= 0.03:
            signals.append("no_close_competing_candidate")
        if seasons_supported >= 2:
            signals.append("stable_across_multiple_seasons")

        if top["competition"] != LEAGUE_TO_TM_COMP.get(str(top["league"])):
            decision = "rejected_wrong_country_or_competition"
        elif competing > 0 and support < 2:
            decision = "manual_review_required"
        elif float(top["fuzzy_score"]) >= 0.90 and len(signals) >= 4 and competing == 0:
            decision = "approved_high_confidence_alias"
        elif bool(top["targeted_expansion_hit"]) and len(signals) >= 4 and competing == 0:
            decision = "approved_high_confidence_alias"
        elif support >= 3 and float(top["fuzzy_score"]) >= 0.78 and competing == 0:
            decision = "approved_high_confidence_alias"
        else:
            decision = "manual_review_required"
        top["seasons_supported"] = seasons_supported
        top["competing_candidate_count"] = competing
        top["proposed_decision"] = decision
        top["reason"] = "; ".join(signals) if signals else "insufficient independent targeted alias evidence"
        out.append(top)
    return pd.DataFrame(out).sort_values(["league", "season_start_year", "football_data_team"])


def combine_mappings(v1: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    v1_out = v1.copy()
    v1_out["mapping_version_source"] = "v1"
    delta_out = delta[
        [
            "league",
            "season_start_year",
            "football_data_team",
            "candidate_transfermarkt_club_id",
            "candidate_transfermarkt_club_name",
            "competition",
            "country",
            "proposed_decision",
            "reason",
        ]
    ].rename(
        columns={
            "candidate_transfermarkt_club_id": "transfermarkt_club_id",
            "candidate_transfermarkt_club_name": "transfermarkt_club_name",
            "proposed_decision": "decision",
        }
    )
    delta_out["fd_norm_name"] = delta_out["football_data_team"].map(raw_key)
    delta_out["mapping_version_source"] = "v2_delta"
    cols = [
        "league",
        "season_start_year",
        "football_data_team",
        "fd_norm_name",
        "transfermarkt_club_id",
        "transfermarkt_club_name",
        "competition",
        "country",
        "decision",
        "reason",
        "mapping_version_source",
    ]
    combined = pd.concat([v1_out.reindex(columns=cols), delta_out.reindex(columns=cols)], ignore_index=True)
    combined["season_start_year"] = combined["season_start_year"].astype(int)
    combined["transfermarkt_club_id"] = combined["transfermarkt_club_id"].astype(int)
    duplicate_fd = combined.duplicated(["league", "season_start_year", "football_data_team"], keep=False)
    duplicate_tm = combined.duplicated(["league", "season_start_year", "transfermarkt_club_id"], keep=False)
    if duplicate_fd.any() or duplicate_tm.any():
        conflicts = combined[duplicate_fd | duplicate_tm].sort_values(["league", "season_start_year", "football_data_team"])
        raise RuntimeError(f"Approved v2 mapping conflicts detected:\n{conflicts.to_string(index=False)}")
    return combined.sort_values(["league", "season_start_year", "football_data_team"])


def split_applyable_delta(v1: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    approved = candidates[candidates["proposed_decision"].eq("approved_high_confidence_alias")].copy()
    v1_fd_keys = set(zip(v1["league"], v1["season_start_year"].astype(int), v1["football_data_team"]))
    v1_tm_keys = set(zip(v1["league"], v1["season_start_year"].astype(int), v1["transfermarkt_club_id"].astype(int)))
    apply_rows = []
    manual_rows = []
    for row in approved.to_dict("records"):
        fd_key = (row["league"], int(row["season_start_year"]), row["football_data_team"])
        tm_key = (row["league"], int(row["season_start_year"]), int(row["candidate_transfermarkt_club_id"]))
        if fd_key in v1_fd_keys:
            row["proposed_decision"] = "already_approved_in_v1_not_reapplied"
            row["reason"] = str(row["reason"]) + "; already_present_in_v1_mapping"
            manual_rows.append(row)
        elif tm_key in v1_tm_keys:
            row["proposed_decision"] = "manual_review_required"
            row["reason"] = str(row["reason"]) + "; transfermarkt_club_id_already_used_by_v1_in_same_league_season"
            manual_rows.append(row)
        else:
            apply_rows.append(row)
    apply_df = pd.DataFrame(apply_rows) if apply_rows else candidates.iloc[0:0].copy()
    manual_df = pd.DataFrame(manual_rows) if manual_rows else candidates.iloc[0:0].copy()
    return apply_df, manual_df


def club_coverage(teams: pd.DataFrame, mapping: pd.DataFrame, manual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped_keys = set(zip(mapping["league"], mapping["season_start_year"], mapping["football_data_team"]))
    manual_keys = set(zip(manual["league"], manual["season_start_year"], manual["football_data_team"]))
    enriched = teams.copy()
    enriched["mapped"] = [key in mapped_keys for key in zip(enriched["league"], enriched["season_start_year"], enriched["football_data_team"])]
    enriched["manual_review_required"] = [
        key in manual_keys for key in zip(enriched["league"], enriched["season_start_year"], enriched["football_data_team"])
    ]
    coverage = (
        enriched.groupby(["league", "season_start_year"], dropna=False)
        .agg(
            total_football_data_team_season_rows=("football_data_team", "count"),
            mapped_team_season_rows=("mapped", "sum"),
            manual_review_required_rows=("manual_review_required", "sum"),
        )
        .reset_index()
    )
    coverage["unmapped_team_season_rows"] = coverage["total_football_data_team_season_rows"] - coverage["mapped_team_season_rows"]
    coverage["mapping_coverage"] = coverage["mapped_team_season_rows"] / coverage["total_football_data_team_season_rows"].replace(0, np.nan)
    unmatched = (
        enriched[~enriched["mapped"]]
        .groupby(["league", "football_data_team"], dropna=False)
        .agg(
            unmatched_team_seasons=("season_start_year", "nunique"),
            seasons=("season_start_year", lambda s: "|".join(str(item) for item in sorted(s.astype(int).unique())[:24])),
        )
        .reset_index()
        .sort_values(["unmatched_team_seasons", "league", "football_data_team"], ascending=[False, True, True])
    )
    return coverage.sort_values(["league", "season_start_year"]), unmatched


def fixture_mapping(matches: pd.DataFrame, games: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    map_dict = v1_mapping_dict(mapping.rename(columns={"transfermarkt_club_id": "transfermarkt_club_id"}))
    mapped = matches.copy()
    mapped["tm_competition_id"] = mapped["league"].map(LEAGUE_TO_TM_COMP)
    mapped["home_tm_club_id"] = [
        map_dict.get((str(row.league), int(row.season_start_year), str(row.home_team)), np.nan) for row in mapped.itertuples(index=False)
    ]
    mapped["away_tm_club_id"] = [
        map_dict.get((str(row.league), int(row.season_start_year), str(row.away_team)), np.nan) for row in mapped.itertuples(index=False)
    ]
    mapped["mapping_key_available"] = mapped[["tm_competition_id", "home_tm_club_id", "away_tm_club_id"]].notna().all(axis=1)
    tm = games.rename(
        columns={
            "competition_id": "tm_competition_id",
            "season": "season_start_year",
            "date": "match_date",
            "home_club_id": "home_tm_club_id",
            "away_club_id": "away_tm_club_id",
        }
    )[["game_id", "tm_competition_id", "season_start_year", "match_date", "home_tm_club_id", "away_tm_club_id"]].copy()
    tm["candidate_count"] = tm.groupby(
        ["tm_competition_id", "season_start_year", "match_date", "home_tm_club_id", "away_tm_club_id"]
    )["game_id"].transform("count")
    joined = mapped.merge(
        tm,
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
    coverage["fixture_mapping_coverage"] = coverage["mapped_fixture_count"] / coverage["football_data_fixtures"].replace(0, np.nan)
    return joined, coverage.sort_values(["league", "season_start_year"])


def diagnose_e123(matches: pd.DataFrame, games: pd.DataFrame, mapping: pd.DataFrame, fixture_cov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for league in ["E1", "E2", "E3"]:
        comp = LEAGUE_TO_TM_COMP[league]
        fd = matches[matches["league"].eq(league)]
        tm_comp_games = games[games["competition_id"].eq(comp)]
        mapped_team_rows = mapping[mapping["league"].eq(league)]
        cov = fixture_cov[fixture_cov["league"].eq(league)]
        if tm_comp_games.empty:
            cause = "competitions_missing_in_transfermarkt_games"
            detail = f"Transfermarkt games.csv has zero rows for {comp}; fixture remap cannot produce matches regardless of aliases."
        elif mapped_team_rows.empty:
            cause = "club_aliases_missing"
            detail = "Competition games exist, but no approved aliases are available."
        elif int(cov["no_tm_game_candidate_count"].sum()) > 0:
            cause = "fixture_date_or_season_mismatch"
            detail = "Approved aliases exist, but date/season/home/away keys do not match Transfermarkt games."
        else:
            cause = "other_mapping_issue"
            detail = "Review fixture keys and competition mapping."
        rows.append(
            {
                "league": league,
                "transfermarkt_competition_id": comp,
                "football_data_fixtures": int(len(fd)),
                "transfermarkt_games_for_competition": int(len(tm_comp_games)),
                "transfermarkt_seasons": "|".join(str(x) for x in sorted(tm_comp_games["season"].dropna().astype(int).unique())),
                "approved_alias_rows": int(len(mapped_team_rows)),
                "mapped_fixtures": int(cov["mapped_fixture_count"].sum()) if not cov.empty else 0,
                "diagnosis": cause,
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def coverage_class(rate: float) -> str:
    if rate < 0.25:
        return "alias_mapping_v2_failed"
    if rate < 0.50:
        return "fixture_mapping_ready_partial"
    if rate < 0.75:
        return "fixture_mapping_ready_good"
    return "fixture_mapping_ready_high_coverage"


def write_markdown(
    candidates: pd.DataFrame,
    delta: pd.DataFrame,
    manual: pd.DataFrame,
    club_cov: pd.DataFrame,
    fixture_cov: pd.DataFrame,
    unmatched: pd.DataFrame,
    diagnosis: pd.DataFrame,
) -> None:
    total_team = int(club_cov["total_football_data_team_season_rows"].sum())
    mapped_team = int(club_cov["mapped_team_season_rows"].sum())
    total_fix = int(fixture_cov["football_data_fixtures"].sum())
    mapped_fix = int(fixture_cov["mapped_fixture_count"].sum())
    unmatched_fix = int(fixture_cov["unmatched_fixture_count"].sum())
    rate = mapped_fix / total_fix if total_fix else 0.0
    modern = fixture_cov[fixture_cov["season_start_year"].between(2014, 2026)]
    modern_rate = modern["mapped_fixture_count"].sum() / modern["football_data_fixtures"].sum() if not modern.empty else 0.0
    no_lower_eng = fixture_cov[~fixture_cov["league"].isin(["E1", "E2", "E3"])]
    no_lower_rate = no_lower_eng["mapped_fixture_count"].sum() / no_lower_eng["football_data_fixtures"].sum() if not no_lower_eng.empty else 0.0
    top_divisions = fixture_cov[fixture_cov["league"].isin(["E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "SC0"])]
    top_rate = top_divisions["mapped_fixture_count"].sum() / top_divisions["football_data_fixtures"].sum() if not top_divisions.empty else 0.0
    by_league = fixture_cov.groupby("league")[["football_data_fixtures", "mapped_fixture_count"]].sum()
    by_league["fixture_mapping_coverage"] = by_league["mapped_fixture_count"] / by_league["football_data_fixtures"].replace(0, np.nan)
    next_step = (
        "Proceed only to Transfermarkt v2 feature-build design for mapped competitions, with strict date-safety tests and explicit missingness flags."
        if rate >= 0.25
        else "Continue manual alias work before feature-build design."
    )
    lines = [
        "# Transfermarkt targeted alias review v2",
        "",
        "Scope: targeted alias mapping and fixture remapping only. No predictive models, value searches, threshold optimization, final model features, betting rules, or confirmed-edge claims were run or created.",
        "",
        "## Targeted Alias Decisions",
        f"- Target candidate rows scored: {len(candidates)}",
        f"- Newly approved high-confidence alias rows: {len(delta)}",
        f"- Targeted manual-review rows: {len(manual)}",
        "- Applied v2 mapping starts from v1 approved aliases and adds only newly approved high-confidence aliases.",
        "",
        "## Team-Season Coverage",
        f"- Mapped team-season rows: {mapped_team}",
        f"- Unmapped team-season rows: {total_team - mapped_team}",
        f"- Manual-review rows: {int(club_cov['manual_review_required_rows'].sum())}",
        f"- Coverage overall: {mapped_team / total_team:.3f}" if total_team else "- Coverage overall: n/a",
        "",
        "## Fixture Coverage",
        f"- Mapped fixtures: {mapped_fix}",
        f"- Unmatched fixtures: {unmatched_fix}",
        f"- Duplicate candidates: {int(fixture_cov['duplicate_candidate_count'].sum())}",
        f"- Ambiguous candidates: {int(fixture_cov['ambiguous_candidate_count'].sum())}",
        f"- Coverage overall: {rate:.3f}",
        f"- Coverage modern seasons 2014-2026: {modern_rate:.3f}",
        f"- Coverage excluding E1/E2/E3: {no_lower_rate:.3f}",
        f"- Coverage top divisions only: {top_rate:.3f}",
        f"- Coverage classification: `{coverage_class(rate)}`",
        "",
        "## Coverage By League",
        by_league.reset_index().to_markdown(index=False),
        "",
        "## Remaining Top Unmatched Teams",
        unmatched.head(30).to_markdown(index=False),
        "",
        "## E1/E2/E3 Zero Coverage Diagnosis",
        diagnosis.to_markdown(index=False),
        "",
        "## Recommended Next Step",
        next_step,
        "",
        "## Outputs",
        f"- `{ALIASES_V2}`",
        f"- `{TARGET_CANDIDATES}`",
        f"- `{DELTA_APPROVED}`",
        f"- `{MANUAL_V2}`",
        f"- `{CLUB_COVERAGE}`",
        f"- `{FIXTURE_COVERAGE}`",
        f"- `{UNMATCHED_V2}`",
        f"- `{E123_DIAGNOSIS}`",
    ]
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    games = load_tm_games()
    clubs = pd.read_csv(TM_DIR / "clubs.csv", usecols=["club_id", "name", "domestic_competition_id"], low_memory=False)
    competitions = pd.read_csv(TM_DIR / "competitions.csv", low_memory=False)
    v1 = pd.read_csv(V1_MAPPING)
    teams = team_seasons(matches)
    tm_pool = build_tm_pool(games, clubs, competitions)
    initial_mapping = v1_mapping_dict(v1)
    evidence = fixture_evidence(matches, games, initial_mapping)
    candidates = build_target_candidates(teams, tm_pool, evidence)
    candidates.to_csv(TARGET_CANDIDATES, index=False)
    delta, withheld = split_applyable_delta(v1, candidates)
    delta.to_csv(DELTA_APPROVED, index=False)
    manual = pd.concat(
        [candidates[candidates["proposed_decision"].eq("manual_review_required")].copy(), withheld],
        ignore_index=True,
        sort=False,
    )
    manual.to_csv(MANUAL_V2, index=False)
    combined = combine_mappings(v1, delta)
    combined.to_csv(ALIASES_V2, index=False)
    club_cov, unmatched = club_coverage(teams, combined, manual)
    club_cov.to_csv(CLUB_COVERAGE, index=False)
    unmatched.to_csv(UNMATCHED_V2, index=False)
    fixture_detail, fixture_cov = fixture_mapping(matches, games, combined)
    fixture_cov.to_csv(FIXTURE_COVERAGE, index=False)
    diagnosis = diagnose_e123(matches, games, combined, fixture_cov)
    diagnosis.to_csv(E123_DIAGNOSIS, index=False)
    write_markdown(candidates, delta, manual, club_cov, fixture_cov, unmatched, diagnosis)


if __name__ == "__main__":
    main()
