from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from zipfile import ZipFile
import io
import re
import unicodedata

import numpy as np
import pandas as pd
from difflib import SequenceMatcher


FEATURE_MATRIX = Path("data/processed/features/football_feature_matrix_v2_transfermarkt_partial.csv")
CLUBELO_ZIP = Path("data/raw_external/clubelo_manual/clubelo_archive.zip")
LOCKED_SELECTED_BETS = Path("outputs/reports/feature_matrix_v2_tm_1x2_locked_selected_bets.csv")
REPORT_DIR = Path("outputs/reports")

SCHEMA_MD = REPORT_DIR / "clubelo_manual_archive_schema_audit.md"
MAPPING_CSV = REPORT_DIR / "clubelo_team_mapping_candidates.csv"
DATE_COVERAGE_CSV = REPORT_DIR / "clubelo_date_safety_coverage.csv"
LOCKED_COVERAGE_CSV = REPORT_DIR / "clubelo_locked_candidate_coverage.csv"
FEATURE_POLICY_CSV = REPORT_DIR / "clubelo_candidate_feature_policy.csv"
RECOMMENDATION_MD = REPORT_DIR / "clubelo_v3_recommendation.md"

LOWER_ENGLISH = {"E1", "E2", "E3"}
TOP_DIVISIONS = {"E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "SC0"}
LEAGUE_COUNTRY = {
    "E0": "ENG",
    "E1": "ENG",
    "E2": "ENG",
    "E3": "ENG",
    "D1": "GER",
    "I1": "ITA",
    "SP1": "ESP",
    "F1": "FRA",
    "P1": "POR",
    "N1": "NED",
    "B1": "BEL",
    "T1": "TUR",
    "G1": "GRE",
    "SC0": "SCO",
}

LEGAL_WORDS = {
    "afc",
    "as",
    "athletic",
    "calcio",
    "cf",
    "club",
    "de",
    "fc",
    "football",
    "futbol",
    "futebol",
    "fk",
    "if",
    "real",
    "sc",
    "sport",
    "sporting",
    "the",
    "u",
    "ud",
}


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [tok for tok in text.split() if tok not in LEGAL_WORDS]
    return " ".join(tokens)


def token_sort_ratio(left: str, right: str) -> float:
    a = " ".join(sorted(left.split()))
    b = " ".join(sorted(right.split()))
    return SequenceMatcher(None, a, b).ratio()


def name_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return max(SequenceMatcher(None, left, right).ratio(), token_sort_ratio(left, right))


def count_zip_rows(zf: ZipFile, name: str) -> int:
    with zf.open(name) as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def read_archive() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with ZipFile(CLUBELO_ZIP) as zf:
        file_rows = []
        for info in zf.infolist():
            file_rows.append({"file": info.filename, "compressed_size": info.compress_size, "uncompressed_size": info.file_size, "rows": count_zip_rows(zf, info.filename)})
        files = pd.DataFrame(file_rows)
        with zf.open("EloRatings.csv") as handle:
            ratings = pd.read_csv(handle, low_memory=False)
        with zf.open("Matches.csv") as handle:
            matches = pd.read_csv(handle, low_memory=False)
    return files, ratings, matches


def archive_schema(files: pd.DataFrame, ratings: pd.DataFrame, matches: pd.DataFrame) -> dict[str, object]:
    ratings_source_columns = list(ratings.columns)
    matches_source_columns = list(matches.columns)
    ratings["date_parsed"] = pd.to_datetime(ratings["date"], errors="coerce")
    matches["match_date_parsed"] = pd.to_datetime(matches["MatchDate"], errors="coerce")
    dup_rating = int(ratings.duplicated(["club", "date_parsed"]).sum())
    return {
        "files": files,
        "ratings_rows": int(len(ratings)),
        "ratings_columns": ratings_source_columns,
        "ratings_min_date": ratings["date_parsed"].min(),
        "ratings_max_date": ratings["date_parsed"].max(),
        "ratings_club_count": int(ratings["club"].nunique(dropna=True)),
        "ratings_country_count": int(ratings["country"].nunique(dropna=True)),
        "ratings_date_parse_failures": int(ratings["date_parsed"].isna().sum()),
        "ratings_duplicate_club_date": dup_rating,
        "ratings_missing_values": ratings.drop(columns=["date_parsed"]).isna().sum().to_dict(),
        "matches_rows": int(len(matches)),
        "matches_columns": matches_source_columns,
        "matches_min_date": matches["match_date_parsed"].min(),
        "matches_max_date": matches["match_date_parsed"].max(),
        "matches_home_team_count": int(matches["HomeTeam"].nunique(dropna=True)),
        "matches_away_team_count": int(matches["AwayTeam"].nunique(dropna=True)),
        "matches_date_parse_failures": int(matches["match_date_parsed"].isna().sum()),
        "matches_missing_values": matches.drop(columns=["match_date_parsed"]).isna().sum().to_dict(),
    }


def load_fixtures() -> pd.DataFrame:
    cols = [
        "match_id",
        "match_date",
        "league",
        "season_start_year",
        "season_end_year",
        "home_team",
        "away_team",
        "home_elo",
        "away_elo",
        "elo_diff",
    ]
    df = pd.read_csv(FEATURE_MATRIX, usecols=cols, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce")
    df["season_end_year"] = pd.to_numeric(df["season_end_year"], errors="coerce")
    return df.dropna(subset=["match_date", "league", "home_team", "away_team", "season_start_year"]).copy()


def build_mapping(fixtures: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    club_rows = ratings[["club", "country"]].dropna().drop_duplicates().copy()
    club_rows["club_norm"] = club_rows["club"].map(normalize_name)
    club_by_country = {country: g.reset_index(drop=True) for country, g in club_rows.groupby("country")}
    teams = pd.concat(
        [
            fixtures[["league", "home_team"]].rename(columns={"home_team": "team"}),
            fixtures[["league", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    teams["country"] = teams["league"].map(LEAGUE_COUNTRY)
    teams["team_norm"] = teams["team"].map(normalize_name)
    rows = []
    accepted = []
    for team in teams.sort_values(["league", "team"]).itertuples(index=False):
        candidates = club_by_country.get(team.country, pd.DataFrame(columns=["club", "country", "club_norm"]))
        exact = candidates[candidates["club_norm"].eq(team.team_norm)].copy()
        if len(exact) == 1:
            status = "accepted_exact_normalized"
            mapped = str(exact["club"].iloc[0])
            score = 1.0
            top = exact.assign(score=1.0).head(1)
        else:
            scored = candidates.copy()
            scored["score"] = scored["club_norm"].map(lambda v: name_score(team.team_norm, v))
            top = scored.sort_values(["score", "club"], ascending=[False, True]).head(5)
            top_score = float(top["score"].iloc[0]) if len(top) else 0.0
            second_score = float(top["score"].iloc[1]) if len(top) > 1 else 0.0
            if len(exact) > 1:
                status = "ambiguous_exact_normalized"
                mapped = ""
                score = top_score
            elif top_score >= 0.94 and (top_score - second_score) >= 0.03:
                status = "accepted_high_confidence_fuzzy"
                mapped = str(top["club"].iloc[0])
                score = top_score
            elif top_score >= 0.82:
                status = "fuzzy_candidates_review_required"
                mapped = ""
                score = top_score
            else:
                status = "unmatched"
                mapped = ""
                score = top_score
        if mapped:
            accepted.append({"league": team.league, "team": team.team, "country": team.country, "clubelo_club": mapped, "mapping_status": status, "mapping_score": score})
        if len(top) == 0:
            rows.append(
                {
                    "league": team.league,
                    "football_team": team.team,
                    "country": team.country,
                    "team_norm": team.team_norm,
                    "mapping_status": status,
                    "accepted_clubelo_club": mapped,
                    "candidate_rank": np.nan,
                    "candidate_club": "",
                    "candidate_country": "",
                    "candidate_score": np.nan,
                }
            )
        else:
            for rank, cand in enumerate(top.itertuples(index=False), start=1):
                rows.append(
                    {
                        "league": team.league,
                        "football_team": team.team,
                        "country": team.country,
                        "team_norm": team.team_norm,
                        "mapping_status": status,
                        "accepted_clubelo_club": mapped,
                        "candidate_rank": rank,
                        "candidate_club": cand.club,
                        "candidate_country": cand.country,
                        "candidate_score": float(cand.score),
                    }
                )
    mapping = pd.DataFrame(rows)
    accepted_map = pd.DataFrame(accepted)
    mapping.to_csv(MAPPING_CSV, index=False)
    return mapping, accepted_map


def attach_mapping(fixtures: pd.DataFrame, accepted_map: pd.DataFrame) -> pd.DataFrame:
    home_map = accepted_map.rename(columns={"team": "home_team", "clubelo_club": "clubelo_home_club", "mapping_status": "home_mapping_status", "mapping_score": "home_mapping_score"})[
        ["league", "home_team", "clubelo_home_club", "home_mapping_status", "home_mapping_score"]
    ]
    away_map = accepted_map.rename(columns={"team": "away_team", "clubelo_club": "clubelo_away_club", "mapping_status": "away_mapping_status", "mapping_score": "away_mapping_score"})[
        ["league", "away_team", "clubelo_away_club", "away_mapping_status", "away_mapping_score"]
    ]
    out = fixtures.merge(home_map, on=["league", "home_team"], how="left").merge(away_map, on=["league", "away_team"], how="left")
    out["clubelo_home_mapped"] = out["clubelo_home_club"].notna()
    out["clubelo_away_mapped"] = out["clubelo_away_club"].notna()
    out["clubelo_both_mapped"] = out["clubelo_home_mapped"] & out["clubelo_away_mapped"]
    return out


def build_rating_index(ratings: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    clean = ratings.dropna(subset=["club", "date_parsed", "elo"]).copy()
    clean = clean.sort_values(["club", "date_parsed"])
    index = {}
    for club, group in clean.groupby("club"):
        index[str(club)] = (group["date_parsed"].to_numpy(dtype="datetime64[ns]"), pd.to_numeric(group["elo"], errors="coerce").to_numpy(dtype=float))
    return index


def latest_before(index: dict[str, tuple[np.ndarray, np.ndarray]], club: object, match_date: pd.Timestamp) -> tuple[float, float]:
    if pd.isna(club) or str(club) not in index or pd.isna(match_date):
        return np.nan, np.nan
    dates, values = index[str(club)]
    pos = bisect_left(dates, np.datetime64(match_date)) - 1
    if pos < 0:
        return np.nan, np.nan
    return float(values[pos]), float((match_date - pd.Timestamp(dates[pos])).days)


def add_date_safe_ratings(mapped: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    index = build_rating_index(ratings)
    home_vals = []
    home_stale = []
    away_vals = []
    away_stale = []
    for row in mapped.itertuples(index=False):
        h, hs = latest_before(index, row.clubelo_home_club, row.match_date)
        a, aas = latest_before(index, row.clubelo_away_club, row.match_date)
        home_vals.append(h)
        home_stale.append(hs)
        away_vals.append(a)
        away_stale.append(aas)
    out = mapped.copy()
    out["clubelo_home_rating"] = home_vals
    out["clubelo_away_rating"] = away_vals
    out["clubelo_staleness_home"] = home_stale
    out["clubelo_staleness_away"] = away_stale
    out["clubelo_home_rating_available"] = out["clubelo_home_rating"].notna()
    out["clubelo_away_rating_available"] = out["clubelo_away_rating"].notna()
    out["clubelo_both_rating_available"] = out["clubelo_home_rating_available"] & out["clubelo_away_rating_available"]
    out["clubelo_diff"] = out["clubelo_home_rating"] - out["clubelo_away_rating"]
    out["clubelo_abs_diff"] = out["clubelo_diff"].abs()
    out["clubelo_home_minus_internal_elo"] = out["clubelo_home_rating"] - out["home_elo"]
    out["clubelo_away_minus_internal_elo"] = out["clubelo_away_rating"] - out["away_elo"]
    out["clubelo_diff_minus_internal_elo_diff"] = out["clubelo_diff"] - out["elo_diff"]
    return out


def summarize_segment(frame: pd.DataFrame, segment: str, group_col: str | None = None) -> list[dict[str, object]]:
    rows = []
    groups = [(segment, frame)] if group_col is None else [(str(k), g) for k, g in frame.groupby(group_col, dropna=False)]
    for label, g in groups:
        staleness = pd.concat([g.loc[g["clubelo_home_rating_available"], "clubelo_staleness_home"], g.loc[g["clubelo_away_rating_available"], "clubelo_staleness_away"]])
        rows.append(
            {
                "segment": segment,
                "group": label,
                "fixtures": int(len(g)),
                "home_team_mapped": int(g["clubelo_home_mapped"].sum()),
                "away_team_mapped": int(g["clubelo_away_mapped"].sum()),
                "both_teams_mapped": int(g["clubelo_both_mapped"].sum()),
                "home_rating_available": int(g["clubelo_home_rating_available"].sum()),
                "away_rating_available": int(g["clubelo_away_rating_available"].sum()),
                "both_rating_available": int(g["clubelo_both_rating_available"].sum()),
                "both_rating_coverage": float(g["clubelo_both_rating_available"].mean()) if len(g) else np.nan,
                "min_staleness_days": float(staleness.min()) if len(staleness) else np.nan,
                "median_staleness_days": float(staleness.median()) if len(staleness) else np.nan,
                "p95_staleness_days": float(staleness.quantile(0.95)) if len(staleness) else np.nan,
                "max_staleness_days": float(staleness.max()) if len(staleness) else np.nan,
            }
        )
    return rows


def coverage_tables(safe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(summarize_segment(safe, "all_fixtures"))
    rows.extend(summarize_segment(safe, "by_league", "league"))
    rows.extend(summarize_segment(safe, "by_season_start_year", "season_start_year"))
    rows.extend(summarize_segment(safe[safe["season_start_year"].between(2014, 2026)], "modern_2014_2026"))
    rows.extend(summarize_segment(safe[~safe["league"].isin(LOWER_ENGLISH)], "excluding_E1_E2_E3"))
    rows.extend(summarize_segment(safe[safe["league"].isin(TOP_DIVISIONS)], "top_divisions_only"))
    coverage = pd.DataFrame(rows)
    coverage.to_csv(DATE_COVERAGE_CSV, index=False)
    return coverage


def locked_candidate_coverage(safe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    locked_universe = safe[~safe["league"].isin(LOWER_ENGLISH)].copy()
    rows.extend(summarize_segment(locked_universe, "locked_v2_scope_C_top_divisions_ex_e1_e2_e3"))
    if LOCKED_SELECTED_BETS.exists():
        bets = pd.read_csv(LOCKED_SELECTED_BETS, usecols=["match_id"])
        selected = safe[safe["match_id"].isin(set(bets["match_id"]))].copy()
        rows.extend(summarize_segment(selected, "locked_v2_selected_bets"))
        rows.extend(summarize_segment(selected, "locked_v2_selected_bets_by_league", "league"))
        rows.extend(summarize_segment(selected, "locked_v2_selected_bets_by_season_start_year", "season_start_year"))
    out = pd.DataFrame(rows)
    out.to_csv(LOCKED_COVERAGE_CSV, index=False)
    return out


def feature_policy() -> pd.DataFrame:
    rows = [
        ("clubelo_home_rating", "numeric", "latest EloRatings.csv rating strictly before match_date for mapped home club", "allowed"),
        ("clubelo_away_rating", "numeric", "latest EloRatings.csv rating strictly before match_date for mapped away club", "allowed"),
        ("clubelo_diff", "numeric", "clubelo_home_rating - clubelo_away_rating", "allowed"),
        ("clubelo_abs_diff", "numeric", "absolute value of clubelo_diff", "allowed"),
        ("clubelo_home_minus_internal_elo", "numeric", "clubelo_home_rating - existing internal home_elo", "allowed"),
        ("clubelo_away_minus_internal_elo", "numeric", "clubelo_away_rating - existing internal away_elo", "allowed"),
        ("clubelo_diff_minus_internal_elo_diff", "numeric", "clubelo_diff - existing internal elo_diff", "allowed"),
        ("clubelo_staleness_home", "numeric", "days between match_date and home rating date", "allowed"),
        ("clubelo_staleness_away", "numeric", "days between match_date and away rating date", "allowed"),
        ("clubelo_missing_home", "boolean", "home rating unavailable after date-safe lookup", "allowed"),
        ("clubelo_missing_away", "boolean", "away rating unavailable after date-safe lookup", "allowed"),
        ("clubelo_missing_both", "boolean", "home or away rating unavailable after date-safe lookup", "allowed"),
        ("clubelo_team_name", "string", "team names or ClubElo club strings", "forbidden_model_feature"),
        ("clubelo_matches_results_scores_post_match_stats", "mixed", "Matches.csv final scores/results/shots/post-match columns", "forbidden_model_feature"),
    ]
    out = pd.DataFrame(rows, columns=["feature", "type", "definition", "policy"])
    out.to_csv(FEATURE_POLICY_CSV, index=False)
    return out


def format_missing_values(missing: dict[str, int], max_items: int = 20) -> str:
    items = [(k, v) for k, v in missing.items() if v]
    if not items:
        return "none"
    return ", ".join(f"{k}={v}" for k, v in items[:max_items])


def write_schema_md(schema: dict[str, object]) -> None:
    lines = [
        "# ClubElo Manual Archive Schema Audit",
        "",
        "## Files",
        schema["files"].to_markdown(index=False),
        "",
        "## EloRatings.csv",
        f"- Rows: {schema['ratings_rows']}",
        f"- Columns: {', '.join(schema['ratings_columns'])}",
        f"- Date range: {schema['ratings_min_date'].date()} to {schema['ratings_max_date'].date()}",
        f"- Club count: {schema['ratings_club_count']}",
        f"- Country count: {schema['ratings_country_count']}",
        f"- Date parse failures: {schema['ratings_date_parse_failures']}",
        f"- Duplicate club/date ratings: {schema['ratings_duplicate_club_date']}",
        f"- Obvious missing values: {format_missing_values(schema['ratings_missing_values'])}",
        "",
        "## Matches.csv",
        f"- Rows: {schema['matches_rows']}",
        f"- Columns: {', '.join(schema['matches_columns'])}",
        f"- Date range: {schema['matches_min_date'].date()} to {schema['matches_max_date'].date()}",
        f"- Home team count: {schema['matches_home_team_count']}",
        f"- Away team count: {schema['matches_away_team_count']}",
        f"- Date parse failures: {schema['matches_date_parse_failures']}",
        f"- Obvious missing values: {format_missing_values(schema['matches_missing_values'])}",
        "",
        "Matches.csv is diagnostic only for this audit. Its final scores, results, shots, and other post-match columns are not approved as v3 features.",
        "",
    ]
    SCHEMA_MD.write_text("\n".join(lines), encoding="utf-8")


def write_recommendation_md(
    mapping: pd.DataFrame,
    coverage: pd.DataFrame,
    locked: pd.DataFrame,
    policy: pd.DataFrame,
) -> str:
    statuses = mapping.drop_duplicates(["league", "football_team"])["mapping_status"].value_counts()
    modern = coverage[(coverage["segment"].eq("modern_2014_2026")) & (coverage["group"].eq("modern_2014_2026"))]
    top = coverage[(coverage["segment"].eq("top_divisions_only")) & (coverage["group"].eq("top_divisions_only"))]
    excl = coverage[(coverage["segment"].eq("excluding_E1_E2_E3")) & (coverage["group"].eq("excluding_E1_E2_E3"))]
    selected = locked[(locked["segment"].eq("locked_v2_selected_bets")) & (locked["group"].eq("locked_v2_selected_bets"))]
    modern_cov = float(modern["both_rating_coverage"].iloc[0]) if len(modern) else 0.0
    top_cov = float(top["both_rating_coverage"].iloc[0]) if len(top) else 0.0
    selected_cov = float(selected["both_rating_coverage"].iloc[0]) if len(selected) else np.nan
    review_required = int(statuses.get("fuzzy_candidates_review_required", 0) + statuses.get("ambiguous_exact_normalized", 0) + statuses.get("unmatched", 0))
    if modern_cov >= 0.95 and top_cov >= 0.95 and review_required <= 30:
        decision = "clubelo_feature_build_ready_good"
    elif top_cov >= 0.90 and modern_cov >= 0.75:
        decision = "clubelo_feature_build_ready_partial"
    elif review_required < len(mapping.drop_duplicates(["league", "football_team"])):
        decision = "clubelo_mapping_ready_only"
    else:
        decision = "clubelo_not_ready"
    lines = [
        "# ClubElo V3 Recommendation",
        "",
        f"Decision: `{decision}`",
        "",
        "## Key Coverage",
        f"- Modern 2014-2026 both-team date-safe rating coverage: {modern_cov:.2%}",
        f"- Top divisions both-team date-safe rating coverage: {top_cov:.2%}",
        f"- Excluding E1/E2/E3 both-team date-safe rating coverage: {float(excl['both_rating_coverage'].iloc[0]) if len(excl) else np.nan:.2%}",
        f"- Locked v2 selected bets both-team date-safe rating coverage: {selected_cov:.2%}",
        "",
        "## Mapping Status Counts",
        statuses.to_frame("teams").to_markdown(),
        "",
        "## Leakage Policy",
        "- Use `EloRatings.csv` only as the candidate rating source.",
        "- Join the latest rating strictly before `match_date`; same-day and future ratings are forbidden.",
        "- `Matches.csv` is schema/mapping/coverage diagnostic only unless explicitly approved later.",
        "- Do not use ClubElo final scores, match results, shots, or post-match stats as features.",
        "- Do not use team names or ClubElo club names as direct model features.",
        "- Missing ratings stay missing with explicit missing flags; no fabricated ratings.",
        "",
        "## Feature Policy",
        policy.to_markdown(index=False),
        "",
        "No predictive modeling or value search was run. No confirmed edge is claimed.",
        "",
    ]
    RECOMMENDATION_MD.write_text("\n".join(lines), encoding="utf-8")
    return decision


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    files, ratings, matches = read_archive()
    schema = archive_schema(files, ratings, matches)
    write_schema_md(schema)
    fixtures = load_fixtures()
    mapping, accepted = build_mapping(fixtures, ratings)
    mapped = attach_mapping(fixtures, accepted)
    safe = add_date_safe_ratings(mapped, ratings)
    coverage = coverage_tables(safe)
    locked = locked_candidate_coverage(safe)
    policy = feature_policy()
    decision = write_recommendation_md(mapping, coverage, locked, policy)

    team_status = mapping.drop_duplicates(["league", "football_team"])["mapping_status"].value_counts().to_dict()
    modern_row = coverage[(coverage["segment"].eq("modern_2014_2026")) & (coverage["group"].eq("modern_2014_2026"))].iloc[0]
    top_row = coverage[(coverage["segment"].eq("top_divisions_only")) & (coverage["group"].eq("top_divisions_only"))].iloc[0]
    locked_row = locked[(locked["segment"].eq("locked_v2_selected_bets")) & (locked["group"].eq("locked_v2_selected_bets"))]
    print(
        {
            "decision": decision,
            "rating_rows": int(len(ratings)),
            "club_count": int(ratings["club"].nunique()),
            "team_mapping_status": team_status,
            "modern_both_rating_coverage": round(float(modern_row["both_rating_coverage"]), 6),
            "top_both_rating_coverage": round(float(top_row["both_rating_coverage"]), 6),
            "locked_selected_bets_coverage": round(float(locked_row["both_rating_coverage"].iloc[0]), 6) if len(locked_row) else None,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
