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
REPORT_DIR = Path("outputs/reports")
MAPPING_DIR = Path("data/mappings")

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

ALIAS_MAPPING = MAPPING_DIR / "transfermarkt_football_data_aliases_v1.csv"
ALIAS_CANDIDATES = REPORT_DIR / "transfermarkt_alias_candidates_scored_v1.csv"
ALIASES_APPROVED = REPORT_DIR / "transfermarkt_aliases_approved_v1.csv"
ALIASES_MANUAL = REPORT_DIR / "transfermarkt_aliases_manual_review_required_v1.csv"
CLUB_COVERAGE = REPORT_DIR / "transfermarkt_club_mapping_coverage_after_alias_v1.csv"
FIXTURE_COVERAGE = REPORT_DIR / "transfermarkt_fixture_mapping_coverage_after_alias_v1.csv"
UNMATCHED_TEAMS = REPORT_DIR / "transfermarkt_unmatched_teams_after_alias_v1.csv"
MARKDOWN = REPORT_DIR / "transfermarkt_alias_review_v1.md"

LEGAL_TOKENS = {
    "1",
    "a",
    "ac",
    "afc",
    "ag",
    "as",
    "association",
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

TOKEN_REPLACEMENTS = {
    "ad": "adana",
    "az": "alkmaar zaanstreek",
    "ath": "athletic atletico",
    "boro": "middlesbrough",
    "braunschweig": "brunswick",
    "ein": "eintracht",
    "evian": "evian thonon gaillard",
    "fc": "",
    "fsv": "",
    "genclerbirligi": "genclerbirligi",
    "gladbach": "monchengladbach",
    "inter": "internazionale",
    "lisbon": "lisboa",
    "lyon": "lyonnais",
    "mgladbach": "monchengladbach",
    "man": "manchester",
    "munich": "munchen bayern",
    "nottm": "nottingham",
    "pauli": "st pauli",
    "psg": "paris saint germain",
    "psv": "philips eindhoven",
    "qpr": "queens park rangers",
    "rennes": "rennais",
    "sp": "sporting",
    "st": "saint sint",
}


def strip_accents(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def tokenize(value: object, remove_legal: bool = True) -> list[str]:
    text = strip_accents(value).casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens: list[str] = []
    for token in text.split():
        replacement = TOKEN_REPLACEMENTS.get(token, token)
        for item in replacement.split():
            if not item:
                continue
            if len(item) == 4 and item.isdigit():
                continue
            if remove_legal and item in LEGAL_TOKENS:
                continue
            tokens.append(item)
    return tokens


def norm(value: object) -> str:
    return " ".join(tokenize(value))


def similarity(fd_name: str, tm_name: str) -> tuple[float, str]:
    fd_tokens = tokenize(fd_name)
    tm_tokens = tokenize(tm_name)
    if not fd_tokens or not tm_tokens:
        return 0.0, "empty_tokens"
    fd_set = set(fd_tokens)
    tm_set = set(tm_tokens)
    seq = SequenceMatcher(None, " ".join(fd_tokens), " ".join(tm_tokens)).ratio()
    inter = len(fd_set & tm_set)
    union = len(fd_set | tm_set)
    jaccard = inter / union if union else 0.0
    containment = inter / len(fd_set) if fd_set else 0.0
    reverse_containment = inter / len(tm_set) if tm_set else 0.0
    score = max(seq, jaccard, containment * 0.97, reverse_containment * 0.92)
    reason = f"seq={seq:.3f};jaccard={jaccard:.3f};fd_token_containment={containment:.3f};tm_token_containment={reverse_containment:.3f}"
    return round(float(score), 4), reason


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


def build_tm_club_seasons(games: pd.DataFrame, clubs: pd.DataFrame, competitions: pd.DataFrame) -> pd.DataFrame:
    home = games[["league", "competition_id", "season", "home_club_id", "home_club_name"]].rename(
        columns={"home_club_id": "tm_club_id", "home_club_name": "tm_club_name"}
    )
    away = games[["league", "competition_id", "season", "away_club_id", "away_club_name"]].rename(
        columns={"away_club_id": "tm_club_id", "away_club_name": "tm_club_name"}
    )
    side_names = pd.concat([home, away], ignore_index=True)
    side_names["tm_club_id"] = pd.to_numeric(side_names["tm_club_id"], errors="coerce").astype("Int64")
    club_names = clubs[["club_id", "name", "domestic_competition_id"]].rename(
        columns={"club_id": "tm_club_id", "name": "tm_club_name", "domestic_competition_id": "competition_id"}
    )
    club_names["league"] = club_names["competition_id"].map(TM_COMP_TO_LEAGUE)
    club_names["season"] = pd.NA
    combined = pd.concat(
        [
            side_names[["league", "competition_id", "season", "tm_club_id", "tm_club_name"]],
            club_names[["league", "competition_id", "season", "tm_club_id", "tm_club_name"]],
        ],
        ignore_index=True,
    ).dropna(subset=["league", "competition_id", "tm_club_id", "tm_club_name"])
    comp_meta = competitions[["competition_id", "country_name", "name"]].rename(
        columns={"name": "tm_competition_name", "country_name": "country"}
    )
    combined = combined.merge(comp_meta, on="competition_id", how="left")
    combined["tm_club_id"] = combined["tm_club_id"].astype(int)
    combined["tm_norm_name"] = combined["tm_club_name"].map(norm)
    return combined.drop_duplicates().reset_index(drop=True)


def team_seasons(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["league", "season_start_year", "home_team"]].rename(columns={"home_team": "football_data_team"})
    away = matches[["league", "season_start_year", "away_team"]].rename(columns={"away_team": "football_data_team"})
    teams = pd.concat([home, away], ignore_index=True).drop_duplicates()
    teams["fd_norm_name"] = teams["football_data_team"].map(norm)
    return teams.sort_values(["league", "season_start_year", "football_data_team"]).reset_index(drop=True)


def build_initial_candidates(teams: pd.DataFrame, tm_clubs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    exact_lookup: dict[tuple[str, str, str], int] = {}
    for team in teams.itertuples(index=False):
        pool = tm_clubs[
            tm_clubs["league"].eq(team.league)
            & (tm_clubs["season"].isna() | tm_clubs["season"].eq(team.season_start_year))
        ].copy()
        if pool.empty:
            rows.append(
                {
                    "league": team.league,
                    "season_start_year": int(team.season_start_year),
                    "football_data_team": team.football_data_team,
                    "fd_norm_name": team.fd_norm_name,
                    "candidate_transfermarkt_club_id": pd.NA,
                    "candidate_transfermarkt_club_name": "",
                    "competition": LEAGUE_TO_TM_COMP.get(str(team.league), ""),
                    "country": "",
                    "fuzzy_score": 0.0,
                    "score_details": "no_transfermarkt_club_pool_for_competition_season",
                }
            )
            continue
        scored = []
        for cand in pool[["competition_id", "country", "tm_competition_name", "tm_club_id", "tm_club_name", "tm_norm_name"]].drop_duplicates().itertuples(index=False):
            score, details = similarity(str(team.football_data_team), str(cand.tm_club_name))
            exact = int(team.fd_norm_name == cand.tm_norm_name and team.fd_norm_name != "")
            scored.append(
                {
                    "league": team.league,
                    "season_start_year": int(team.season_start_year),
                    "football_data_team": team.football_data_team,
                    "fd_norm_name": team.fd_norm_name,
                    "candidate_transfermarkt_club_id": int(cand.tm_club_id),
                    "candidate_transfermarkt_club_name": cand.tm_club_name,
                    "competition": cand.competition_id,
                    "country": cand.country,
                    "fuzzy_score": 1.0 if exact else score,
                    "score_details": "normalized_exact" if exact else details,
                }
            )
        by_club: dict[int, dict] = {}
        for item in scored:
            club_id = int(item["candidate_transfermarkt_club_id"])
            if club_id not in by_club or float(item["fuzzy_score"]) > float(by_club[club_id]["fuzzy_score"]):
                by_club[club_id] = item
        scored = sorted(by_club.values(), key=lambda item: item["fuzzy_score"], reverse=True)
        rows.extend(scored[:8])
        exact_top = [item for item in scored if item["score_details"] == "normalized_exact"]
        if len({item["candidate_transfermarkt_club_id"] for item in exact_top}) == 1:
            exact_lookup[(str(team.league), str(team.season_start_year), str(team.football_data_team))] = exact_top[0][
                "candidate_transfermarkt_club_id"
            ]
    return pd.DataFrame(rows)


def best_similarity_map(candidates: pd.DataFrame) -> dict[tuple[str, int, str], int]:
    best = {}
    valid = candidates.dropna(subset=["candidate_transfermarkt_club_id"]).copy()
    for key, group in valid.groupby(["league", "season_start_year", "football_data_team"], dropna=False):
        group = group.sort_values("fuzzy_score", ascending=False)
        top = group.iloc[0]
        second = float(group.iloc[1]["fuzzy_score"]) if len(group) > 1 else 0.0
        if float(top["fuzzy_score"]) >= 0.80 and float(top["fuzzy_score"]) - second >= 0.04:
            best[(str(key[0]), int(key[1]), str(key[2]))] = int(top["candidate_transfermarkt_club_id"])
    return best


def fixture_evidence(matches: pd.DataFrame, games: pd.DataFrame, prelim: dict[tuple[str, int, str], int]) -> dict[tuple[str, int, str, int], int]:
    tm_by_key: dict[tuple[str, int, pd.Timestamp, int], set[int]] = defaultdict(set)
    for game in games.itertuples(index=False):
        tm_by_key[(str(game.league), int(game.season), pd.Timestamp(game.date), int(game.home_club_id))].add(int(game.away_club_id))
        tm_by_key[(str(game.league), int(game.season), pd.Timestamp(game.date), int(game.away_club_id))].add(int(game.home_club_id))
    counts: dict[tuple[str, int, str, int], int] = defaultdict(int)
    for match in matches.itertuples(index=False):
        season = int(match.season_start_year)
        date = pd.Timestamp(match.match_date)
        home_key = (str(match.league), season, str(match.home_team))
        away_key = (str(match.league), season, str(match.away_team))
        home_id = prelim.get(home_key)
        away_id = prelim.get(away_key)
        if home_id is not None and away_id is not None:
            if away_id in tm_by_key.get((str(match.league), season, date, home_id), set()):
                counts[(str(match.league), season, str(match.home_team), home_id)] += 1
                counts[(str(match.league), season, str(match.away_team), away_id)] += 1
    return counts


def season_stability(candidates: pd.DataFrame) -> dict[tuple[str, str, int], int]:
    valid = candidates.dropna(subset=["candidate_transfermarkt_club_id"]).copy()
    top = valid.sort_values("fuzzy_score", ascending=False).groupby(["league", "season_start_year", "football_data_team"], as_index=False).head(1)
    return (
        top.groupby(["league", "football_data_team", "candidate_transfermarkt_club_id"])["season_start_year"]
        .nunique()
        .astype(int)
        .to_dict()
    )


def decide_candidates(candidates: pd.DataFrame, evidence: dict[tuple[str, int, str, int], int], stability: dict[tuple[str, str, int], int]) -> pd.DataFrame:
    out = []
    for key, group in candidates.groupby(["league", "season_start_year", "football_data_team"], dropna=False):
        league, season, team = str(key[0]), int(key[1]), str(key[2])
        group = group.sort_values("fuzzy_score", ascending=False).copy()
        if group["candidate_transfermarkt_club_id"].isna().all():
            row = group.iloc[0].to_dict()
            row.update(
                {
                    "seasons_observed": 0,
                    "supporting_fixture_evidence_count": 0,
                    "competing_candidate_count": 0,
                    "decision": "manual_review_required",
                    "reason": "No Transfermarkt club pool for this competition-season.",
                }
            )
            out.append(row)
            continue
        top = group.iloc[0].to_dict()
        top_id = int(top["candidate_transfermarkt_club_id"])
        top_score = float(top["fuzzy_score"])
        second_score = float(group.iloc[1]["fuzzy_score"]) if len(group) > 1 else 0.0
        competing = int((group["fuzzy_score"].ge(max(0.80, top_score - 0.05))).sum() - 1)
        support = int(evidence.get((league, season, team, top_id), 0))
        seasons = int(stability.get((league, team, top_id), 0))
        signals = []
        if top["competition"] == LEAGUE_TO_TM_COMP.get(league):
            signals.append("same_competition")
        if top["score_details"] == "normalized_exact":
            signals.append("normalized_exact")
        if top_score >= 0.86:
            signals.append("high_fuzzy_similarity")
        if support >= 2:
            signals.append("fixture_date_opponent_evidence")
        if competing == 0 and top_score - second_score >= 0.04:
            signals.append("no_competing_close_candidate")
        if seasons >= 2:
            signals.append("stable_multiple_seasons")

        if top["competition"] != LEAGUE_TO_TM_COMP.get(league):
            decision = "rejected_wrong_country_or_competition"
        elif top["score_details"] == "normalized_exact" and competing == 0:
            decision = "approved_exact"
        elif competing > 0 and support < 2:
            decision = "rejected_ambiguous"
        elif top_score < 0.68 and support < 2:
            decision = "rejected_low_similarity"
        elif "high_fuzzy_similarity" in signals and len(signals) >= 3 and competing == 0:
            decision = "approved_high_confidence_alias"
        elif support >= 3 and top_score >= 0.72 and competing == 0:
            decision = "approved_high_confidence_alias"
        else:
            decision = "manual_review_required"
        top.update(
            {
                "seasons_observed": seasons,
                "supporting_fixture_evidence_count": support,
                "competing_candidate_count": competing,
                "decision": decision,
                "reason": "; ".join(signals) if signals else "insufficient independent alias evidence",
            }
        )
        out.append(top)
    decided = pd.DataFrame(out)
    approved = decided["decision"].isin(["approved_exact", "approved_high_confidence_alias"])
    duplicate_fd = decided[approved].duplicated(["league", "season_start_year", "football_data_team"], keep=False)
    duplicate_tm = decided[approved].duplicated(["league", "season_start_year", "candidate_transfermarkt_club_id"], keep=False)
    conflict = approved & (duplicate_fd | duplicate_tm)
    decided.loc[conflict, "decision"] = "manual_review_required"
    decided.loc[conflict, "reason"] = decided.loc[conflict, "reason"].astype(str) + "; rejected_from_application_due_to_league_season_mapping_conflict"
    return decided.sort_values(["league", "season_start_year", "football_data_team"])


def club_mapping_coverage(teams: pd.DataFrame, decided: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    applied = decided[decided["decision"].isin(["approved_exact", "approved_high_confidence_alias"])].copy()
    mapped_keys = set(zip(applied["league"], applied["season_start_year"], applied["football_data_team"]))
    manual_keys = set(
        zip(
            decided.loc[decided["decision"].eq("manual_review_required"), "league"],
            decided.loc[decided["decision"].eq("manual_review_required"), "season_start_year"],
            decided.loc[decided["decision"].eq("manual_review_required"), "football_data_team"],
        )
    )
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
    coverage["ambiguous_rows"] = coverage["manual_review_required_rows"]
    coverage["team_season_mapping_coverage"] = coverage["mapped_team_season_rows"] / coverage[
        "total_football_data_team_season_rows"
    ].replace(0, np.nan)
    unmatched = enriched[~enriched["mapped"]].groupby(["league", "football_data_team"], dropna=False).agg(
        unmatched_team_seasons=("season_start_year", "nunique"),
        seasons=("season_start_year", lambda s: "|".join(str(item) for item in sorted(s.astype(int).unique())[:20])),
    ).reset_index().sort_values(["unmatched_team_seasons", "league", "football_data_team"], ascending=[False, True, True])
    return coverage.sort_values(["league", "season_start_year"]), unmatched


def fixture_mapping(matches: pd.DataFrame, games: pd.DataFrame, approved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = {
        (str(row.league), int(row.season_start_year), str(row.football_data_team)): int(row.candidate_transfermarkt_club_id)
        for row in approved.itertuples(index=False)
    }
    mapped = matches.copy()
    mapped["tm_competition_id"] = mapped["league"].map(LEAGUE_TO_TM_COMP)
    mapped["home_tm_club_id"] = [
        mapping.get((str(row.league), int(row.season_start_year), str(row.home_team)), np.nan) for row in mapped.itertuples(index=False)
    ]
    mapped["away_tm_club_id"] = [
        mapping.get((str(row.league), int(row.season_start_year), str(row.away_team)), np.nan) for row in mapped.itertuples(index=False)
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


def coverage_classification(rate: float) -> str:
    if rate < 0.25:
        return "alias_mapping_partial"
    if rate < 0.50:
        return "fixture_mapping_ready_partial"
    if rate < 0.75:
        return "fixture_mapping_ready_good"
    return "fixture_mapping_ready_high_coverage"


def write_markdown(
    candidates: pd.DataFrame,
    approved: pd.DataFrame,
    manual: pd.DataFrame,
    club_cov: pd.DataFrame,
    fixture_cov: pd.DataFrame,
    unmatched: pd.DataFrame,
    fixture_detail: pd.DataFrame,
) -> None:
    total_team_rows = int(club_cov["total_football_data_team_season_rows"].sum())
    mapped_team_rows = int(club_cov["mapped_team_season_rows"].sum())
    manual_rows = int(club_cov["manual_review_required_rows"].sum())
    total_fixtures = int(fixture_cov["football_data_fixtures"].sum())
    mapped_fixtures = int(fixture_cov["mapped_fixture_count"].sum())
    unmatched_fixtures = int(fixture_cov["unmatched_fixture_count"].sum())
    duplicate = int(fixture_cov["duplicate_candidate_count"].sum())
    ambiguous = int(fixture_cov["ambiguous_candidate_count"].sum())
    fixture_rate = mapped_fixtures / total_fixtures if total_fixtures else 0.0
    classification = coverage_classification(fixture_rate)
    league_summary = fixture_cov.groupby("league")[["football_data_fixtures", "mapped_fixture_count"]].sum()
    league_summary["fixture_mapping_coverage"] = league_summary["mapped_fixture_count"] / league_summary["football_data_fixtures"].replace(0, np.nan)
    top_unmatched_league_season = fixture_cov.sort_values("unmatched_fixture_count", ascending=False).head(15)
    failed = fixture_detail[~fixture_detail["match_mapping_status"].eq("mapped")][
        ["match_id", "match_date", "league", "season_start_year", "home_team", "away_team", "match_mapping_status"]
    ].head(25)

    next_step = (
        "Manual alias review remains the next gate; prioritize the teams listed in `transfermarkt_unmatched_teams_after_alias_v1.csv`."
        if fixture_rate < 0.25
        else "Fixture coverage is usable enough for a Transfermarkt v2 feature-build design, still with strict date-safety tests before predictive use."
    )

    lines = [
        "# Transfermarkt club alias review and fixture remap v1",
        "",
        "Scope: conservative alias review and fixture remapping only. No predictive models, value searches, threshold optimization, final Transfermarkt model features, betting rules, or confirmed-edge claims were run or created.",
        "",
        "## Alias Decision Policy",
        "- Applied mappings include only `approved_exact` and `approved_high_confidence_alias`.",
        "- Manual-review and rejected mappings are not applied.",
        "- Auto-approval requires same competition plus independent support from normalized exactness, high fuzzy similarity, fixture date/opponent evidence, no close competing candidate, or stability across seasons.",
        "- Final scores are used only as diagnostic fixture-identity evidence.",
        "",
        "## Club Mapping Coverage",
        f"- Total football-data team-season rows: {total_team_rows}",
        f"- Mapped team-season rows: {mapped_team_rows}",
        f"- Unmapped team-season rows: {total_team_rows - mapped_team_rows}",
        f"- Manual-review rows: {manual_rows}",
        f"- Coverage: {mapped_team_rows / total_team_rows:.3f}" if total_team_rows else "- Coverage: n/a",
        f"- Approved aliases: {len(approved)}",
        f"- Manual-review aliases: {len(manual)}",
        "",
        "## Fixture Mapping Coverage",
        f"- Mapped fixture count: {mapped_fixtures}",
        f"- Unmatched fixture count: {unmatched_fixtures}",
        f"- Duplicate candidate count: {duplicate}",
        f"- Ambiguous candidate count: {ambiguous}",
        f"- Coverage overall: {fixture_rate:.3f}",
        f"- Coverage classification: `{classification}`",
        "",
        "## Coverage By League",
        league_summary.reset_index().to_markdown(index=False),
        "",
        "## Top Unmatched League-Seasons",
        top_unmatched_league_season[
            ["league", "season_start_year", "football_data_fixtures", "mapped_fixture_count", "unmatched_fixture_count", "fixture_mapping_coverage"]
        ].to_markdown(index=False),
        "",
        "## Top Unmatched Teams",
        unmatched.head(30).to_markdown(index=False),
        "",
        "## Failed Mapping Examples",
        failed.to_markdown(index=False),
        "",
        "## Recommended Next Step",
        next_step,
        "",
        "## Outputs",
        f"- `{ALIAS_MAPPING}`",
        f"- `{ALIAS_CANDIDATES}`",
        f"- `{ALIASES_APPROVED}`",
        f"- `{ALIASES_MANUAL}`",
        f"- `{CLUB_COVERAGE}`",
        f"- `{FIXTURE_COVERAGE}`",
        f"- `{UNMATCHED_TEAMS}`",
    ]
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    games = load_tm_games()
    clubs = pd.read_csv(TM_DIR / "clubs.csv", usecols=["club_id", "name", "domestic_competition_id"], low_memory=False)
    competitions = pd.read_csv(TM_DIR / "competitions.csv", low_memory=False)

    teams = team_seasons(matches)
    tm_clubs = build_tm_club_seasons(games, clubs, competitions)
    raw_candidates = build_initial_candidates(teams, tm_clubs)
    prelim = best_similarity_map(raw_candidates)
    evidence = fixture_evidence(matches, games, prelim)
    stability = season_stability(raw_candidates)
    candidates = decide_candidates(raw_candidates, evidence, stability)
    candidates.to_csv(ALIAS_CANDIDATES, index=False)

    approved = candidates[candidates["decision"].isin(["approved_exact", "approved_high_confidence_alias"])].copy()
    manual = candidates[candidates["decision"].eq("manual_review_required")].copy()
    approved.to_csv(ALIASES_APPROVED, index=False)
    manual.to_csv(ALIASES_MANUAL, index=False)
    approved[
        [
            "league",
            "season_start_year",
            "football_data_team",
            "fd_norm_name",
            "candidate_transfermarkt_club_id",
            "candidate_transfermarkt_club_name",
            "competition",
            "country",
            "decision",
            "reason",
        ]
    ].rename(
        columns={
            "candidate_transfermarkt_club_id": "transfermarkt_club_id",
            "candidate_transfermarkt_club_name": "transfermarkt_club_name",
        }
    ).to_csv(ALIAS_MAPPING, index=False)

    club_cov, unmatched = club_mapping_coverage(teams, candidates)
    club_cov.to_csv(CLUB_COVERAGE, index=False)
    unmatched.to_csv(UNMATCHED_TEAMS, index=False)

    fixture_detail, fixture_cov = fixture_mapping(matches, games, approved)
    fixture_cov.to_csv(FIXTURE_COVERAGE, index=False)

    write_markdown(candidates, approved, manual, club_cov, fixture_cov, unmatched, fixture_detail)


if __name__ == "__main__":
    main()
