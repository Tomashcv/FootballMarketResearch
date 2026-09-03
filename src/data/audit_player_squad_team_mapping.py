from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.features.player_squad_strength import normalize_name


LEAGUES = ["E0", "I1", "SP1", "D1", "F1", "P1"]
LEAGUE_COMPETITIONS = {
    "E0": "GB1",
    "I1": "IT1",
    "SP1": "ES1",
    "D1": "L1",
    "F1": "FR1",
    "P1": "PO1",
}
LEAGUE_NAMES = {
    "E0": "Premier League",
    "I1": "Serie A",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "P1": "Liga Portugal",
}
TM_DIR = Path("data/external/players/transfermarkt_raw/player_scores")
MAPPING_PATH = Path("data/manual/player_squad_team_name_mapping.csv")
REPORT_DIR = Path("outputs/reports")
CANDIDATES_PATH = REPORT_DIR / "player_squad_team_mapping_candidates.csv"
UNMATCHED_PATH = REPORT_DIR / "player_squad_team_mapping_unmatched.csv"
AUDIT_PATH = REPORT_DIR / "player_squad_team_mapping_audit.md"

LEGAL_TOKENS = {
    "1900",
    "1907",
    "1899",
    "1893",
    "1846",
    "1910",
    "1919",
    "1936",
    "2020",
    "club",
    "football",
    "futbol",
    "futebol",
    "fußball",
    "fussball",
    "calcio",
    "vereniging",
    "verein",
    "association",
    "associazione",
    "associacao",
    "societa",
    "sportiva",
    "sport",
    "sporting",
    "sportive",
    "sports",
    "squadra",
    "de",
    "do",
    "da",
    "del",
    "di",
    "the",
    "and",
    "team",
    "sad",
    "spa",
    "s",
    "a",
    "d",
}

KNOWN_ALIASES = {
    ("E0", "Arsenal"): "Arsenal Football Club",
    ("E0", "Aston Villa"): "Aston Villa Football Club",
    ("E0", "Birmingham"): "Birmingham City",
    ("E0", "Blackburn"): "Blackburn Rovers",
    ("E0", "Blackpool"): "Blackpool FC",
    ("E0", "Bolton"): "Bolton Wanderers",
    ("E0", "Bournemouth"): "Association Football Club Bournemouth",
    ("E0", "Brentford"): "Brentford Football Club",
    ("E0", "Brighton"): "Brighton and Hove Albion Football Club",
    ("E0", "Burnley"): "Burnley Football Club",
    ("E0", "Cardiff"): "Cardiff City",
    ("E0", "Chelsea"): "Chelsea Football Club",
    ("E0", "Crystal Palace"): "Crystal Palace Football Club",
    ("E0", "Everton"): "Everton Football Club",
    ("E0", "Fulham"): "Fulham Football Club",
    ("E0", "Huddersfield"): "Huddersfield Town",
    ("E0", "Hull"): "Hull City",
    ("E0", "Ipswich"): "Ipswich Town",
    ("E0", "Leeds"): "Leeds United Association Football Club",
    ("E0", "Leicester"): "Leicester City",
    ("E0", "Liverpool"): "Liverpool Football Club",
    ("E0", "Luton"): "Luton Town",
    ("E0", "Man City"): "Manchester City Football Club",
    ("E0", "Man United"): "Manchester United Football Club",
    ("E0", "Middlesbrough"): "Middlesbrough FC",
    ("E0", "Newcastle"): "Newcastle United Football Club",
    ("E0", "Norwich"): "Norwich City",
    ("E0", "Nott'm Forest"): "Nottingham Forest Football Club",
    ("E0", "QPR"): "Queens Park Rangers",
    ("E0", "Reading"): "Reading FC",
    ("E0", "Sheffield United"): "Sheffield United",
    ("E0", "Southampton"): "Southampton FC",
    ("E0", "Stoke"): "Stoke City",
    ("E0", "Sunderland"): "Sunderland Association Football Club",
    ("E0", "Swansea"): "Swansea City",
    ("E0", "Tottenham"): "Tottenham Hotspur Football Club",
    ("E0", "Watford"): "Watford FC",
    ("E0", "West Brom"): "West Bromwich Albion",
    ("E0", "West Ham"): "West Ham United Football Club",
    ("E0", "Wigan"): "Wigan Athletic",
    ("E0", "Wolves"): "Wolverhampton Wanderers Football Club",
    ("I1", "Atalanta"): "Atalanta Bergamasca Calcio S.p.a.",
    ("I1", "Bari"): "AS Bari",
    ("I1", "Genoa"): "Genoa Cricket and Football Club",
    ("I1", "Inter"): "Football Club Internazionale Milano S.p.A.",
    ("I1", "Milan"): "Associazione Calcio Milan",
    ("I1", "Napoli"): "Società Sportiva Calcio Napoli",
    ("I1", "Novara"): "Novara Calcio 1908",
    ("I1", "Parma"): "Parma Calcio 1913",
    ("I1", "Verona"): "Verona Hellas Football Club",
    ("SP1", "Alaves"): "Deportivo Alavés S. A. D.",
    ("SP1", "Ath Madrid"): "Club Atlético de Madrid S.A.D.",
    ("SP1", "Ath Bilbao"): "Athletic Club Bilbao",
    ("SP1", "Betis"): "Real Betis Balompié S.A.D.",
    ("SP1", "Celta"): "Real Club Celta de Vigo S. A. D.",
    ("SP1", "Espanol"): "Reial Club Deportiu Espanyol de Barcelona S.A.D.",
    ("SP1", "Hercules"): "Hércules CF",
    ("SP1", "La Coruna"): "Deportivo de La Coruña",
    ("SP1", "Levante"): "Levante Unión Deportiva S.A.D.",
    ("SP1", "Mallorca"): "Real Club Deportivo Mallorca S.A.D.",
    ("SP1", "Santander"): "Racing Santander",
    ("SP1", "Sociedad"): "Real Sociedad de Fútbol S.A.D.",
    ("SP1", "Sp Gijon"): "Sporting Gijón",
    ("D1", "Bayern Munich"): "FC Bayern München",
    ("D1", "Dortmund"): "Borussia Dortmund",
    ("D1", "FC Koln"): "1. Fußball-Club Köln",
    ("D1", "M'gladbach"): "Borussia Verein für Leibesübungen 1900 Mönchengladbach",
    ("D1", "Ein Frankfurt"): "Eintracht Frankfurt Fußball AG",
    ("D1", "Hoffenheim"): "Turn- und Sportgemeinschaft 1899 Hoffenheim Fußball-Spielbetriebs",
    ("D1", "Kaiserslautern"): "1.FC Kaiserslautern",
    ("D1", "Leverkusen"): "Bayer 04 Leverkusen Fußball",
    ("D1", "Mainz"): "1. Fußball- und Sportverein Mainz 05",
    ("D1", "Nurnberg"): "1.FC Nuremberg",
    ("D1", "RB Leipzig"): "RasenBallsport Leipzig",
    ("D1", "St Pauli"): "Fußball-Club St. Pauli von 1910",
    ("D1", "Stuttgart"): "Verein für Bewegungsspiele Stuttgart 1893",
    ("D1", "Union Berlin"): "1. Fußballclub Union Berlin",
    ("D1", "Werder Bremen"): "Sportverein Werder Bremen von 1899",
    ("D1", "Wolfsburg"): "Verein für Leibesübungen Wolfsburg",
    ("F1", "Ajaccio"): "AC Ajaccio",
    ("F1", "Ajaccio GFCO"): "GFC Ajaccio",
    ("F1", "Arles"): "AC Arles-Avignon",
    ("F1", "Auxerre"): "Association de la Jeunesse auxerroise",
    ("F1", "Brest"): "Stade brestois 29",
    ("F1", "Evian Thonon Gaillard"): "FC Évian Thonon Gaillard",
    ("F1", "Le Havre"): "Le Havre Athletic Club",
    ("F1", "Lens"): "Racing Club de Lens",
    ("F1", "Lille"): "Lille Olympique Sporting Club",
    ("F1", "Lorient"): "Football Club Lorient-Bretagne Sud",
    ("F1", "Lyon"): "Olympique Lyonnais",
    ("F1", "Nancy"): "AS Nancy-Lorraine",
    ("F1", "Nice"): "Olympique Gymnaste Club Nice Côte d'Azur",
    ("F1", "Nimes"): "Nîmes Olympique",
    ("F1", "Paris SG"): "Paris Saint-Germain Football Club",
    ("F1", "Rennes"): "Stade Rennais Football Club",
    ("F1", "Sochaux"): "FC Sochaux-Montbéliard",
    ("F1", "St Etienne"): "AS Saint-Étienne",
    ("P1", "AVS"): "Avs Futebol",
    ("P1", "Aves"): "Desportivo Aves (- 2020)",
    ("P1", "Belenenses"): "B SAD",
    ("P1", "Benfica"): "Sport Lisboa e Benfica",
    ("P1", "Casa Pia"): "Casa Pia Atlético Clube",
    ("P1", "Estrela"): "Club Football Estrela da Amadora",
    ("P1", "Estoril"): "Grupo Desportivo Estoril Praia",
    ("P1", "Gil Vicente"): "Gil Vicente Futebol Clube",
    ("P1", "Guimaraes"): "Vitória Sport Clube",
    ("P1", "Nacional"): "Clube Desportivo Nacional",
    ("P1", "Pacos Ferreira"): "FC Paços de Ferreira",
    ("P1", "Porto"): "Futebol Clube do Porto",
    ("P1", "Rio Ave"): "Rio Ave Futebol Clube",
    ("P1", "Santa Clara"): "Clube Desportivo Santa Clara",
    ("P1", "Setubal"): "Vitória Setúbal FC",
    ("P1", "Sp Braga"): "Sporting Clube de Braga",
    ("P1", "Sp Lisbon"): "Sporting Clube de Portugal",
    ("P1", "Tondela"): "Clube Desportivo de Tondela",
}

VALUATION_NAME_OVERRIDES = {
    ("D1", "Ein Frankfurt"): "Eintracht Frankfurt",
    ("D1", "Leverkusen"): "Bayer 04 Leverkusen",
    ("D1", "M'gladbach"): "Borussia Mönchengladbach",
    ("E0", "Arsenal"): "Arsenal FC",
    ("E0", "Burnley"): "Burnley FC",
    ("E0", "Chelsea"): "Chelsea FC",
    ("E0", "Everton"): "Everton FC",
    ("E0", "Fulham"): "Fulham FC",
    ("F1", "Lens"): "RC Lens",
    ("F1", "Metz"): "FC Metz",
    ("F1", "Rennes"): "Stade Rennais FC",
    ("F1", "Toulouse"): "FC Toulouse",
    ("I1", "Atalanta"): "Atalanta BC",
    ("I1", "Inter"): "FC Internazionale",
    ("I1", "Juventus"): "Juventus FC",
    ("I1", "Lecce"): "US Lecce",
    ("I1", "Napoli"): "SSC Napoli",
    ("I1", "Roma"): "AS Roma",
    ("I1", "Siena"): "AC Siena",
    ("I1", "Verona"): "Hellas Verona",
    ("P1", "Estoril"): "GD Estoril Praia",
    ("P1", "Moreirense"): "Moreirense FC",
    ("P1", "Nacional"): "CD Nacional",
    ("P1", "Porto"): "FC Porto",
    ("P1", "Rio Ave"): "Rio Ave FC",
    ("P1", "Santa Clara"): "CD Santa Clara",
    ("P1", "Sp Lisbon"): "Sporting CP",
    ("P1", "Tondela"): "CD Tondela",
    ("SP1", "Osasuna"): "CA Osasuna",
    ("SP1", "Villarreal"): "Villarreal CF",
}


def ascii_normalize(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return normalize_name(text)


def compact_key(value: object) -> str:
    return re.sub(r"\s+", "", ascii_normalize(value))


def stripped_key(value: object) -> str:
    tokens = [token for token in ascii_normalize(value).split() if token not in LEGAL_TOKENS]
    return " ".join(tokens)


def club_aliases(row: pd.Series) -> set[str]:
    aliases = {
        ascii_normalize(row["name"]),
        compact_key(row["name"]),
        stripped_key(row["name"]),
        compact_key(stripped_key(row["name"])),
    }
    if pd.notna(row.get("club_code")):
        aliases.add(ascii_normalize(row["club_code"]))
        aliases.add(compact_key(row["club_code"]))
    return {alias for alias in aliases if alias}


def is_senior_valuation_name(value: object) -> bool:
    text = ascii_normalize(value)
    blocked_tokens = {"u18", "u19", "u20", "u21", "u23", "reserves", "reserve", "primavera", "ii", "b"}
    return not any(token in blocked_tokens for token in text.split())


def resolve_valuation_club_name(club_name: str, match_team: str, valuation_names: list[str]) -> str:
    if not club_name:
        return ""
    if club_name in valuation_names:
        return club_name

    team_key = ascii_normalize(match_team)
    club_key = ascii_normalize(club_name)
    club_stripped = stripped_key(club_name)
    candidates = []
    for value in valuation_names:
        if not is_senior_valuation_name(value):
            continue
        value_key = ascii_normalize(value)
        value_stripped = stripped_key(value)
        related = (
            team_key in value_key
            or value_key in club_key
            or value_stripped in club_stripped
            or club_stripped in value_stripped
        )
        score = max(
            SequenceMatcher(None, team_key, value_key).ratio(),
            SequenceMatcher(None, club_key, value_key).ratio(),
            SequenceMatcher(None, club_stripped, value_stripped).ratio(),
        )
        if related or score >= 0.74:
            candidates.append((score, value))

    candidates = sorted(candidates, reverse=True)
    if not candidates:
        return club_name
    best_score, best_name = candidates[0]
    if len(candidates) > 1 and best_score < 0.90 and best_score - candidates[1][0] < 0.08:
        return club_name
    return best_name


def load_match_teams() -> pd.DataFrame:
    rows = []
    for league in LEAGUES:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["HomeTeam", "AwayTeam"], low_memory=False)
        counts = pd.concat([frame["HomeTeam"], frame["AwayTeam"]], ignore_index=True).value_counts()
        for team, matches in counts.items():
            rows.append({"league": league, "match_team": team, "match_rows": int(matches)})
    return pd.DataFrame(rows).sort_values(["league", "match_team"]).reset_index(drop=True)


def load_clubs_with_competitions() -> pd.DataFrame:
    clubs = pd.read_csv(TM_DIR / "clubs.csv", low_memory=False)
    competitions = pd.read_csv(TM_DIR / "competitions.csv", low_memory=False)
    competitions = competitions.rename(
        columns={
            "competition_id": "domestic_competition_id",
            "name": "competition_name",
            "country_name": "competition_country",
        }
    )
    merged = clubs.merge(
        competitions[["domestic_competition_id", "competition_name", "competition_country", "type", "sub_type"]],
        on="domestic_competition_id",
        how="left",
    )
    valuation_path = Path("data/external/players/transfermarkt_market_values.csv")
    if valuation_path.exists():
        valuation_clubs = pd.read_csv(valuation_path, usecols=["club_name"], low_memory=False)
        existing_names = set(merged["name"].dropna().map(ascii_normalize))
        alias_targets = {ascii_normalize(name) for name in KNOWN_ALIASES.values()}
        valuation_names = sorted(
            {
                str(name)
                for name in valuation_clubs["club_name"].dropna().unique()
                if ascii_normalize(name)
                and ascii_normalize(name) in alias_targets
                and ascii_normalize(name) not in existing_names
            }
        )
        valuation_only = pd.DataFrame(
            {
                "club_id": pd.NA,
                "club_code": "",
                "name": valuation_names,
                "domestic_competition_id": "",
                "competition_name": "",
                "competition_country": "",
                "type": "",
                "sub_type": "",
            }
        )
        merged = pd.concat([merged, valuation_only], ignore_index=True, sort=False)
    merged["name_ascii_normalized"] = merged["name"].map(ascii_normalize)
    merged["stripped_key"] = merged["name"].map(stripped_key)
    return merged


def candidate_rows(team_row: pd.Series, clubs: pd.DataFrame) -> list[dict]:
    league = team_row["league"]
    team = team_row["match_team"]
    expected_comp = LEAGUE_COMPETITIONS[league]
    team_aliases = {ascii_normalize(team), compact_key(team), stripped_key(team), compact_key(stripped_key(team))}
    rows = []
    for _, club in clubs.iterrows():
        aliases = club_aliases(club)
        direct = bool(team_aliases & aliases)
        contains = any(
            len(alias.split()) >= 1 and alias in club["name_ascii_normalized"].split()
            for alias in [ascii_normalize(team), stripped_key(team)]
            if alias
        )
        fuzzy = max(
            SequenceMatcher(None, ascii_normalize(team), club["name_ascii_normalized"]).ratio(),
            SequenceMatcher(None, compact_key(team), compact_key(club["name"])).ratio(),
            SequenceMatcher(None, stripped_key(team), club["stripped_key"]).ratio(),
        )
        same_comp = club["domestic_competition_id"] == expected_comp
        score = fuzzy + (0.12 if same_comp else 0.0) + (0.18 if direct else 0.0) + (0.06 if contains else 0.0)
        if direct or contains or score >= 0.68:
            rows.append(
                {
                    "league": league,
                    "match_team": team,
                    "normalized_match_team": ascii_normalize(team),
                    "candidate_club_id": club["club_id"],
                    "candidate_club_name": club["name"],
                    "normalized_candidate_club": ascii_normalize(club["name"]),
                    "domestic_competition_id": club["domestic_competition_id"],
                    "competition_name": club.get("competition_name", ""),
                    "country": club.get("competition_country", ""),
                    "same_expected_competition": same_comp,
                    "direct_alias_match": direct,
                    "contains_match_token": contains,
                    "fuzzy_similarity": round(float(fuzzy), 4),
                    "score": round(float(score), 4),
                }
            )
    return sorted(rows, key=lambda row: (row["same_expected_competition"], row["score"]), reverse=True)[:8]


def select_mapping(team_row: pd.Series, candidates: list[dict], clubs_by_name: dict[str, pd.Series]) -> tuple[str, str]:
    league = team_row["league"]
    team = team_row["match_team"]
    alias = KNOWN_ALIASES.get((league, team))
    if alias:
        club = clubs_by_name.get(ascii_normalize(alias))
        if club is not None:
            return str(club["name"]), "high_known_alias"
        return "", "unmatched_alias_target_missing"

    same_comp = [candidate for candidate in candidates if candidate["same_expected_competition"]]
    direct = [candidate for candidate in same_comp if candidate["direct_alias_match"]]
    if len(direct) == 1:
        return direct[0]["candidate_club_name"], "high_exact_or_normalized_alias"

    contains = [candidate for candidate in same_comp if candidate["contains_match_token"] and candidate["score"] >= 0.76]
    if len(contains) == 1:
        return contains[0]["candidate_club_name"], "high_unique_same_competition_contains"

    strong = [candidate for candidate in same_comp if candidate["score"] >= 0.92]
    if len(strong) == 1:
        return strong[0]["candidate_club_name"], "high_unique_same_competition_fuzzy"

    return "", "ambiguous" if candidates else "unmatched"


def markdown_table(frame: pd.DataFrame, max_rows: int = 120) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(max_rows).fillna("").to_markdown(index=False)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    teams = load_match_teams()
    clubs = load_clubs_with_competitions()
    clubs_by_name = {ascii_normalize(row["name"]): row for _, row in clubs.iterrows()}
    valuation_path = Path("data/external/players/transfermarkt_market_values.csv")
    valuation_names = (
        sorted(pd.read_csv(valuation_path, usecols=["club_name"], low_memory=False)["club_name"].dropna().astype(str).unique())
        if valuation_path.exists()
        else []
    )

    mapping_rows = []
    candidate_output = []
    for _, team_row in teams.iterrows():
        candidates = candidate_rows(team_row, clubs)
        club_name, confidence = select_mapping(team_row, candidates, clubs_by_name)
        valuation_club_name = VALUATION_NAME_OVERRIDES.get(
            (team_row["league"], team_row["match_team"]),
            resolve_valuation_club_name(club_name, team_row["match_team"], valuation_names),
        )
        for rank, candidate in enumerate(candidates, start=1):
            candidate_output.append({"rank": rank, **candidate})
        mapping_rows.append(
            {
                "league": team_row["league"],
                "match_team": team_row["match_team"],
                "normalized_match_team": ascii_normalize(team_row["match_team"]),
                "player_data_source": "transfermarkt",
                "player_data_club_name": valuation_club_name,
                "normalized_player_data_club": ascii_normalize(valuation_club_name) if valuation_club_name else "",
                "valid_from": "",
                "valid_to": "",
                "confidence": confidence if club_name else "unmatched" if confidence.startswith("unmatched") else confidence,
            }
        )

    mapping = pd.DataFrame(mapping_rows)
    candidates = pd.DataFrame(candidate_output)
    unmatched = mapping[~mapping["confidence"].astype(str).str.startswith("high")].copy()
    mapping.to_csv(MAPPING_PATH, index=False)
    candidates.to_csv(CANDIDATES_PATH, index=False)
    unmatched.to_csv(UNMATCHED_PATH, index=False)

    competitions = pd.read_csv(TM_DIR / "competitions.csv", low_memory=False)
    competition_rows = []
    for league, competition_id in LEAGUE_COMPETITIONS.items():
        row = competitions[competitions["competition_id"].eq(competition_id)].iloc[0].to_dict()
        competition_rows.append(
            {
                "league": league,
                "league_name": LEAGUE_NAMES[league],
                "competition_id": competition_id,
                "competition_name": row.get("name", ""),
                "country": row.get("country_name", ""),
                "domestic_league_code": row.get("domestic_league_code", ""),
            }
        )
    comp_table = pd.DataFrame(competition_rows)
    summary = (
        mapping.assign(high_confidence=mapping["confidence"].astype(str).str.startswith("high"))
        .groupby("league")
        .agg(match_teams=("match_team", "count"), high_confidence_teams=("high_confidence", "sum"))
        .reset_index()
    )
    summary["unmatched_or_ambiguous_teams"] = summary["match_teams"] - summary["high_confidence_teams"]

    audit_lines = [
        "# Player Squad Team Mapping Audit",
        "",
        "No websites were scraped. Raw match data was not edited. Mapping was rebuilt from processed match team names and local Transfermarkt CSVs only.",
        "",
        "## Transfermarkt Competition Mapping",
        markdown_table(comp_table),
        "",
        "## Club Table Schema",
        "`clubs.csv` columns used: `club_id`, `club_code`, `name`, `domestic_competition_id`, joined to `competitions.csv` for `competition_name` and `country_name`.",
        "",
        "## Mapping Summary",
        markdown_table(summary),
        "",
        "## Conservative Acceptance Rules",
        "- Explicit known aliases are accepted only when the target club exists in local `clubs.csv`.",
        "- Unique exact/normalized aliases are accepted only inside the expected Transfermarkt domestic competition.",
        "- Unique same-competition containment/fuzzy matches are accepted only above conservative score thresholds.",
        "- Ambiguous cases remain in the candidate and unmatched reports for manual review.",
        "",
        f"Candidate report: `{CANDIDATES_PATH}`",
        f"Unmatched/ambiguous report: `{UNMATCHED_PATH}`",
    ]
    AUDIT_PATH.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print(f"mapping_rows: {len(mapping)}")
    print(f"high_confidence_rows: {int(mapping['confidence'].astype(str).str.startswith('high').sum())}")
    print(f"unmatched_or_ambiguous_rows: {len(unmatched)}")


if __name__ == "__main__":
    main()
