from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


TM_DIR = Path("data/external/players/transfermarkt_raw/player_scores")
FEATURE_MATRIX = Path("data/processed/features/football_feature_matrix_v1_1.csv")
V2_MAPPING = Path("data/mappings/transfermarkt_football_data_aliases_v2.csv")
REPORT_DIR = Path("outputs/reports")
MAPPING_DIR = Path("data/mappings")

ALIASES_V3 = MAPPING_DIR / "transfermarkt_football_data_aliases_v3.csv"
MARKDOWN = REPORT_DIR / "transfermarkt_obvious_alias_verification_v3.md"
DELTA_APPROVED = REPORT_DIR / "transfermarkt_aliases_approved_v3_delta.csv"
MANUAL_V3 = REPORT_DIR / "transfermarkt_aliases_manual_review_required_v3.csv"
FIXTURE_COVERAGE = REPORT_DIR / "transfermarkt_fixture_mapping_coverage_after_alias_v3.csv"
UNMATCHED_V3 = REPORT_DIR / "transfermarkt_unmatched_teams_after_alias_v3.csv"

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

# Human-seeded targets resolved against local clubs.csv. IDs are used only as
# deterministic local identity anchors and are still verified against games.csv.
SEEDS = {
    ("SP1", "Barcelona"): {"club_id": 131, "aliases": "FC Barcelona / Barcelona"},
    ("SP1", "Real Madrid"): {"club_id": 418, "aliases": "Real Madrid"},
    ("B1", "Club Brugge"): {"club_id": 2282, "aliases": "Club Brugge KV / Club Brugge"},
    ("N1", "PSV Eindhoven"): {"club_id": 383, "aliases": "PSV Eindhoven / PSV"},
    ("P1", "Benfica"): {"club_id": 294, "aliases": "SL Benfica / Benfica"},
    ("P1", "Sp Lisbon"): {"club_id": 336, "aliases": "Sporting CP / Sporting Clube de Portugal"},
    ("P1", "Guimaraes"): {"club_id": 2420, "aliases": "Vitoria Guimaraes / Vitoria SC"},
    ("G1", "PAOK"): {"club_id": 1091, "aliases": "PAOK Thessaloniki / PAOK"},
    ("G1", "AEK"): {"club_id": 2441, "aliases": "AEK Athens / AEK"},
    ("SP1", "Espanol"): {"club_id": 714, "aliases": "RCD Espanyol / Espanyol"},
    ("SC0", "Hearts"): {"club_id": 43, "aliases": "Heart of Midlothian / Hearts"},
    ("B1", "St Truiden"): {"club_id": 475, "aliases": "Sint-Truidense VV / St Truiden"},
    ("N1", "NAC Breda"): {"club_id": 132, "aliases": "NAC Breda"},
    ("T1", "Buyuksehyr"): {"club_id": 6890, "aliases": "Istanbul Basaksehir / Basaksehir"},
    ("D1", "M'gladbach"): {"club_id": 18, "aliases": "Borussia Monchengladbach"},
    ("F1", "Paris SG"): {"club_id": 583, "aliases": "Paris Saint-Germain / PSG"},
    ("D1", "FC Koln"): {"club_id": 3, "aliases": "1. FC Koln / FC Koln"},
    ("D1", "Ein Frankfurt"): {"club_id": 24, "aliases": "Eintracht Frankfurt"},
    ("D1", "Bayern Munich"): {"club_id": 27, "aliases": "Bayern Munchen / Bayern Munich"},
    ("E0", "Nott'm Forest"): {"club_id": 703, "aliases": "Nottingham Forest"},
}


def load_matches() -> pd.DataFrame:
    cols = ["match_id", "match_date", "league", "season_start_year", "home_team", "away_team"]
    matches = pd.read_csv(FEATURE_MATRIX, usecols=cols, low_memory=False)
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce").dt.normalize()
    matches["season_start_year"] = pd.to_numeric(matches["season_start_year"], errors="coerce").astype("Int64")
    return matches.dropna(subset=["match_date", "league", "season_start_year", "home_team", "away_team"]).reset_index(drop=True)


def load_games() -> pd.DataFrame:
    cols = ["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id", "home_club_name", "away_club_name"]
    games = pd.read_csv(TM_DIR / "games.csv", usecols=cols, low_memory=False)
    games = games[games["competition_id"].isin(TM_COMP_TO_LEAGUE)].copy()
    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.normalize()
    games["season"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    games["league"] = games["competition_id"].map(TM_COMP_TO_LEAGUE)
    return games.dropna(subset=["date", "season", "league", "home_club_id", "away_club_id"])


def team_seasons(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["league", "season_start_year", "home_team"]].rename(columns={"home_team": "football_data_team"})
    away = matches[["league", "season_start_year", "away_team"]].rename(columns={"away_team": "football_data_team"})
    return pd.concat([home, away], ignore_index=True).drop_duplicates().reset_index(drop=True)


def v_mapping(mapping: pd.DataFrame) -> dict[tuple[str, int, str], int]:
    return {
        (str(row.league), int(row.season_start_year), str(row.football_data_team)): int(row.transfermarkt_club_id)
        for row in mapping.itertuples(index=False)
    }


def candidate_fixture_evidence(matches: pd.DataFrame, games: pd.DataFrame, base_mapping: dict[tuple[str, int, str], int]) -> dict[tuple[str, int, str, int], int]:
    candidates_by_opponent: dict[tuple[str, int, pd.Timestamp, int], set[int]] = defaultdict(set)
    for game in games.itertuples(index=False):
        league = str(game.league)
        season = int(game.season)
        date = pd.Timestamp(game.date)
        home_id = int(game.home_club_id)
        away_id = int(game.away_club_id)
        candidates_by_opponent[(league, season, date, away_id)].add(home_id)
        candidates_by_opponent[(league, season, date, home_id)].add(away_id)
    counts: dict[tuple[str, int, str, int], int] = defaultdict(int)
    for match in matches.itertuples(index=False):
        league = str(match.league)
        season = int(match.season_start_year)
        date = pd.Timestamp(match.match_date)
        home_key = (league, season, str(match.home_team))
        away_key = (league, season, str(match.away_team))
        home_id = base_mapping.get(home_key)
        away_id = base_mapping.get(away_key)
        if away_id is not None:
            for candidate in candidates_by_opponent.get((league, season, date, away_id), set()):
                counts[(league, season, str(match.home_team), int(candidate))] += 1
        if home_id is not None:
            for candidate in candidates_by_opponent.get((league, season, date, home_id), set()):
                counts[(league, season, str(match.away_team), int(candidate))] += 1
    return counts


def seed_verification_rows(matches: pd.DataFrame, games: pd.DataFrame, clubs: pd.DataFrame, competitions: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    teams = team_seasons(matches)
    v2_map = v_mapping(v2)
    evidence = candidate_fixture_evidence(matches, games, v2_map)
    games_by_club = {}
    for (league, season), group in games.groupby(["league", "season"], dropna=False):
        counts = defaultdict(int)
        for row in group.itertuples(index=False):
            counts[int(row.home_club_id)] += 1
            counts[int(row.away_club_id)] += 1
        games_by_club[(str(league), int(season))] = counts
    v2_tm_owner = {
        (str(row.league), int(row.season_start_year), int(row.transfermarkt_club_id)): str(row.football_data_team)
        for row in v2.itertuples(index=False)
    }
    club_lookup = clubs.set_index("club_id").to_dict("index")
    comp_lookup = competitions.set_index("competition_id").to_dict("index")
    rows = []
    for team in teams.itertuples(index=False):
        seed = SEEDS.get((str(team.league), str(team.football_data_team)))
        if not seed:
            continue
        league = str(team.league)
        if league in LOWER_ENGLISH:
            continue
        season = int(team.season_start_year)
        club_id = int(seed["club_id"])
        club = club_lookup.get(club_id)
        expected_comp = LEAGUE_TO_TM_COMP.get(league)
        if not club:
            rows.append(
                {
                    "league": league,
                    "season_start_year": season,
                    "football_data_team": team.football_data_team,
                    "candidate_transfermarkt_club_id": club_id,
                    "candidate_transfermarkt_club_name": "",
                    "seeded_aliases": seed["aliases"],
                    "decision": "manual_review_required",
                    "reason": "seeded_candidate_missing_from_clubs_csv",
                }
            )
            continue
        club_comp = str(club.get("domestic_competition_id", ""))
        comp = comp_lookup.get(club_comp, {})
        correct_comp = club_comp == expected_comp
        club_game_count = int(games_by_club.get((league, season), {}).get(club_id, 0))
        fixture_count = int(evidence.get((league, season, str(team.football_data_team), club_id), 0))
        current_id = v2_map.get((league, season, str(team.football_data_team)))
        existing_owner = v2_tm_owner.get((league, season, club_id))
        signals = []
        if correct_comp:
            signals.append("same_country_correct_competition")
        if club_game_count > 0:
            signals.append("candidate_appears_in_games_competition_season")
        if fixture_count >= 2:
            signals.append("repeated_match_date_opponent_evidence")
        if current_id == club_id:
            signals.append("already_verified_in_v2")
        no_conflict = existing_owner in {None, str(team.football_data_team)}
        if no_conflict:
            signals.append("no_existing_v2_club_id_conflict")
        if current_id is not None and current_id != club_id:
            signals.append(f"v2_same_team_conflict_current_id={current_id}")
        if existing_owner and existing_owner != str(team.football_data_team):
            signals.append(f"v2_transfermarkt_id_owned_by={existing_owner}")

        if not correct_comp:
            decision = "manual_review_required"
            reason = "wrong_competition_for_seeded_candidate"
        elif club_game_count <= 0:
            decision = "manual_review_required"
            reason = "no_games_csv_competition_season_evidence"
        elif current_id == club_id:
            decision = "already_approved_in_v2"
            reason = "; ".join(signals)
        elif no_conflict and (fixture_count >= 2 or club_game_count >= 10):
            decision = "approved_high_confidence_alias"
            reason = "; ".join(signals)
        elif not no_conflict and fixture_count >= 2 and club_game_count >= 10:
            decision = "approved_high_confidence_alias_with_v2_conflict_quarantine"
            reason = "; ".join(signals)
        elif current_id is not None and current_id != club_id and club_game_count >= 10:
            decision = "approved_high_confidence_alias_with_v2_same_team_correction"
            reason = "; ".join(signals)
        else:
            decision = "manual_review_required"
            reason = "; ".join(signals) or "insufficient_local_fixture_identity_evidence"
        rows.append(
            {
                "league": league,
                "season_start_year": season,
                "football_data_team": team.football_data_team,
                "candidate_transfermarkt_club_id": club_id,
                "candidate_transfermarkt_club_name": club["name"],
                "seeded_aliases": seed["aliases"],
                "candidate_competition": club_comp,
                "expected_competition": expected_comp,
                "country": comp.get("country_name", ""),
                "club_games_in_competition_season": club_game_count,
                "fixture_evidence_count": fixture_count,
                "current_v2_transfermarkt_club_id": current_id if current_id is not None else pd.NA,
                "existing_v2_owner_of_candidate_id": existing_owner or "",
                "decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["league", "season_start_year", "football_data_team"])


def build_v3_mapping(v2: pd.DataFrame, verifications: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    approved_decisions = {
        "approved_high_confidence_alias",
        "approved_high_confidence_alias_with_v2_conflict_quarantine",
        "approved_high_confidence_alias_with_v2_same_team_correction",
    }
    approved = verifications[verifications["decision"].isin(approved_decisions)].copy()
    v3 = v2.copy()
    v3["mapping_version_source"] = v3.get("mapping_version_source", "v2")
    quarantined_rows = []
    for row in approved.itertuples(index=False):
        same_team = (
            v3["league"].eq(row.league)
            & v3["season_start_year"].astype(int).eq(int(row.season_start_year))
            & v3["football_data_team"].eq(row.football_data_team)
        )
        conflicting_owner = (
            v3["league"].eq(row.league)
            & v3["season_start_year"].astype(int).eq(int(row.season_start_year))
            & v3["transfermarkt_club_id"].astype(int).eq(int(row.candidate_transfermarkt_club_id))
            & ~v3["football_data_team"].eq(row.football_data_team)
        )
        if same_team.any():
            old = v3[same_team].copy()
            old["decision"] = "quarantined_v2_same_team_conflict"
            old["reason"] = f"replaced_by_verified_seeded_alias_id={int(row.candidate_transfermarkt_club_id)}"
            quarantined_rows.append(old)
            v3 = v3[~same_team].copy()
        if conflicting_owner.any():
            old = v3[conflicting_owner].copy()
            old["decision"] = "quarantined_v2_transfermarkt_id_conflict"
            old["reason"] = f"candidate_id_verified_for_seeded_team={row.football_data_team}"
            quarantined_rows.append(old)
            v3 = v3[~conflicting_owner].copy()
    delta = approved[
        [
            "league",
            "season_start_year",
            "football_data_team",
            "candidate_transfermarkt_club_id",
            "candidate_transfermarkt_club_name",
            "candidate_competition",
            "country",
            "decision",
            "reason",
        ]
    ].rename(
        columns={
            "candidate_transfermarkt_club_id": "transfermarkt_club_id",
            "candidate_transfermarkt_club_name": "transfermarkt_club_name",
            "candidate_competition": "competition",
        }
    )
    delta["fd_norm_name"] = delta["football_data_team"].astype(str).str.casefold()
    delta["mapping_version_source"] = "v3_delta"
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
    v3 = pd.concat([v3.reindex(columns=cols), delta.reindex(columns=cols)], ignore_index=True)
    v3["season_start_year"] = v3["season_start_year"].astype(int)
    v3["transfermarkt_club_id"] = v3["transfermarkt_club_id"].astype(int)
    duplicate_fd = v3.duplicated(["league", "season_start_year", "football_data_team"], keep=False)
    duplicate_tm = v3.duplicated(["league", "season_start_year", "transfermarkt_club_id"], keep=False)
    if duplicate_fd.any() or duplicate_tm.any():
        conflicts = v3[duplicate_fd | duplicate_tm].sort_values(["league", "season_start_year", "football_data_team"])
        raise RuntimeError(f"v3 mapping conflicts remain:\n{conflicts.to_string(index=False)}")
    quarantine = pd.concat(quarantined_rows, ignore_index=True, sort=False) if quarantined_rows else pd.DataFrame()
    manual = pd.concat(
        [
            verifications[~verifications["decision"].isin(approved_decisions | {"already_approved_in_v2"})].copy(),
            quarantine,
        ],
        ignore_index=True,
        sort=False,
    )
    return v3.sort_values(["league", "season_start_year", "football_data_team"]), delta, manual


def fixture_mapping(matches: pd.DataFrame, games: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    map_dict = v_mapping(mapping)
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


def unmatched_teams(matches: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    teams = team_seasons(matches)
    mapped_keys = set(zip(mapping["league"], mapping["season_start_year"].astype(int), mapping["football_data_team"]))
    teams["mapped"] = [key in mapped_keys for key in zip(teams["league"], teams["season_start_year"].astype(int), teams["football_data_team"])]
    return (
        teams[~teams["mapped"]]
        .groupby(["league", "football_data_team"], dropna=False)
        .agg(
            unmatched_team_seasons=("season_start_year", "nunique"),
            seasons=("season_start_year", lambda s: "|".join(str(item) for item in sorted(s.astype(int).unique())[:24])),
        )
        .reset_index()
        .sort_values(["unmatched_team_seasons", "league", "football_data_team"], ascending=[False, True, True])
    )


def coverage_gate(overall: float, modern: float, top: float) -> str:
    best = max(modern, top)
    if best >= 0.75:
        return "fixture_mapping_ready_high_coverage"
    if best >= 0.50:
        return "fixture_mapping_ready_good"
    return "fixture_mapping_ready_partial"


def write_markdown(delta: pd.DataFrame, manual: pd.DataFrame, coverage: pd.DataFrame, unmatched: pd.DataFrame) -> None:
    total = int(coverage["football_data_fixtures"].sum())
    mapped = int(coverage["mapped_fixture_count"].sum())
    unmatched_count = int(coverage["unmatched_fixture_count"].sum())
    overall = mapped / total if total else 0.0
    modern = coverage[coverage["season_start_year"].between(2014, 2026)]
    modern_rate = modern["mapped_fixture_count"].sum() / modern["football_data_fixtures"].sum() if not modern.empty else 0.0
    no_lower = coverage[~coverage["league"].isin(LOWER_ENGLISH)]
    no_lower_rate = no_lower["mapped_fixture_count"].sum() / no_lower["football_data_fixtures"].sum() if not no_lower.empty else 0.0
    top = coverage[coverage["league"].isin(TOP_DIVISIONS)]
    top_rate = top["mapped_fixture_count"].sum() / top["football_data_fixtures"].sum() if not top.empty else 0.0
    by_league = coverage.groupby("league")[["football_data_fixtures", "mapped_fixture_count"]].sum()
    by_league["fixture_mapping_coverage"] = by_league["mapped_fixture_count"] / by_league["football_data_fixtures"].replace(0, np.nan)
    lines = [
        "# Transfermarkt obvious-club alias verification v3",
        "",
        "Scope: seeded alias verification and fixture remapping only. No predictive models, value searches, threshold optimization, final Transfermarkt model features, betting rules, or confirmed-edge claims were run or created.",
        "",
        "## Verification Policy",
        "- Seeded candidates were resolved against local `clubs.csv`.",
        "- Approval required the correct Transfermarkt competition and local `games.csv` evidence for the same competition-season.",
        "- E1/E2/E3 were not forced because local Transfermarkt `games.csv` has zero GB2/GB3/GB4 rows.",
        "- Existing v2 rows conflicting with verified seeded aliases were quarantined rather than silently retained.",
        "",
        "## Aliases Approved In V3",
        f"- Approved v3 delta rows: {len(delta)}",
        delta.groupby(["league", "football_data_team"])["season_start_year"].nunique().reset_index(name="approved_team_seasons").to_markdown(index=False)
        if not delta.empty
        else "- None",
        "",
        "## Manual Or Quarantined Rows",
        f"- Manual/quarantined rows: {len(manual)}",
        manual[["league", "season_start_year", "football_data_team", "decision", "reason"]].head(40).to_markdown(index=False)
        if not manual.empty
        else "- None",
        "",
        "## Fixture Coverage",
        f"- Mapped fixtures: {mapped}",
        f"- Unmatched fixtures: {unmatched_count}",
        f"- Duplicate candidates: {int(coverage['duplicate_candidate_count'].sum())}",
        f"- Ambiguous candidates: {int(coverage['ambiguous_candidate_count'].sum())}",
        f"- Coverage overall: {overall:.3f}",
        f"- Coverage modern seasons 2014-2026: {modern_rate:.3f}",
        f"- Coverage excluding E1/E2/E3: {no_lower_rate:.3f}",
        f"- Coverage top divisions only: {top_rate:.3f}",
        f"- Coverage gate: `{coverage_gate(overall, modern_rate, top_rate)}`",
        "",
        "## Coverage By League",
        by_league.reset_index().to_markdown(index=False),
        "",
        "## Remaining Top Unmatched Clubs",
        unmatched.head(30).to_markdown(index=False),
        "",
        "## Outputs",
        f"- `{ALIASES_V3}`",
        f"- `{DELTA_APPROVED}`",
        f"- `{MANUAL_V3}`",
        f"- `{FIXTURE_COVERAGE}`",
        f"- `{UNMATCHED_V3}`",
    ]
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    games = load_games()
    clubs = pd.read_csv(TM_DIR / "clubs.csv", usecols=["club_id", "name", "domestic_competition_id"], low_memory=False)
    competitions = pd.read_csv(TM_DIR / "competitions.csv", low_memory=False)
    v2 = pd.read_csv(V2_MAPPING)
    verifications = seed_verification_rows(matches, games, clubs, competitions, v2)
    v3, delta, manual = build_v3_mapping(v2, verifications)
    v3.to_csv(ALIASES_V3, index=False)
    delta.to_csv(DELTA_APPROVED, index=False)
    manual.to_csv(MANUAL_V3, index=False)
    _, coverage = fixture_mapping(matches, games, v3)
    coverage.to_csv(FIXTURE_COVERAGE, index=False)
    unmatched = unmatched_teams(matches, v3)
    unmatched.to_csv(UNMATCHED_V3, index=False)
    write_markdown(delta, manual, coverage, unmatched)


if __name__ == "__main__":
    main()
