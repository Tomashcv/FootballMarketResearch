from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


RAW_P1_DIR = Path("data/raw/P1")
TM_DIR = Path("data/external/players/transfermarkt_raw/player_scores")
PROCESSED_P1 = Path("data/processed/P1/P1_matches.csv")
PLAYERS_DIR = Path("data/external/players")
REPORT_DIR = Path("outputs/reports")
MAPPING_PATH = Path("data/manual/player_squad_team_name_mapping.csv")
MARKET_VALUES_PATH = PLAYERS_DIR / "transfermarkt_market_values.csv"
CLUB_HISTORY_PATH = PLAYERS_DIR / "player_club_history.csv"
CLUB_HISTORY_DIAGNOSTIC_PATH = PLAYERS_DIR / "player_club_history_diagnostic_only.csv"

LEAGUES = ["E0", "I1", "SP1", "D1", "F1", "P1"]
CORE_MATCH_COLUMNS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ONE_X_TWO_COLUMNS = ["AvgH", "AvgD", "AvgA"]
AH_COLUMNS = ["AHh", "AvgAHH", "AvgAHA"]
CLOSING_AH_COLUMNS = ["AHCh", "AvgCAHH", "AvgCAHA"]
REQUIRED_MAPPING_COLUMNS = [
    "league",
    "match_team",
    "normalized_match_team",
    "player_data_source",
    "player_data_club_name",
    "normalized_player_data_club",
    "valid_from",
    "valid_to",
    "confidence",
]

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.casefold()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[columns].head(max_rows).fillna("").to_markdown(index=False)


def read_csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0, low_memory=False).columns)


def row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def season_years(dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    parsed = pd.to_datetime(dates, errors="coerce").dt.normalize()
    start = np.where(parsed.dt.month >= 7, parsed.dt.year, parsed.dt.year - 1)
    start = pd.Series(start, index=dates.index, dtype="Int64")
    end = start + 1
    return start, end


def existing_match_schema() -> list[str]:
    columns: list[str] = []
    for league in ["E0", "I1", "SP1", "D1", "F1"]:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if not path.exists():
            continue
        for column in read_csv_header(path):
            if column not in columns:
                columns.append(column)
    return columns


def import_p1_matches() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict]]:
    files = sorted(RAW_P1_DIR.glob("*.csv"))
    frames = []
    source_rows = []
    all_columns: list[str] = []
    for path in files:
        columns = read_csv_header(path)
        is_football_data = set(CORE_MATCH_COLUMNS).issubset(columns)
        source_rows.append(
            {
                "file": str(path),
                "size_bytes": path.stat().st_size,
                "columns": len(columns),
                "rows": row_count(path),
                "football_data_style": is_football_data,
                "missing_core_columns": ",".join(column for column in CORE_MATCH_COLUMNS if column not in columns),
            }
        )
        if not is_football_data:
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["source_file"] = path.name
        frames.append(frame)
        for column in frame.columns:
            if column not in all_columns:
                all_columns.append(column)
    if not frames:
        return pd.DataFrame(), pd.DataFrame(source_rows), [], []

    output = pd.concat(frames, ignore_index=True, sort=False)
    output["Date"] = pd.to_datetime(output["Date"], errors="coerce", dayfirst=True).dt.normalize()
    output = output.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    output["league"] = "P1"
    output["season_start_year"], output["season_end_year"] = season_years(output["Date"])
    output["season_start_year"] = output["season_start_year"].astype("Int64")
    output["season_end_year"] = output["season_end_year"].astype("Int64")

    dedupe_keys = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    before = len(output)
    output = output.sort_values(["Date", "HomeTeam", "AwayTeam", "source_file"]).drop_duplicates(dedupe_keys, keep="last")
    duplicates_removed = before - len(output)

    schema = existing_match_schema()
    for column in all_columns + ["league", "season_start_year", "season_end_year", "source_file"]:
        if column not in schema:
            schema.append(column)
    for column in schema:
        if column not in output.columns:
            output[column] = np.nan
    output = output[schema].sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    PROCESSED_P1.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(PROCESSED_P1, index=False)

    coverage_rows = []
    for season, group in output.groupby("season_end_year", dropna=False):
        missing_columns = [
            column
            for column in ONE_X_TWO_COLUMNS + AH_COLUMNS + CLOSING_AH_COLUMNS
            if column not in group.columns or group[column].isna().all()
        ]
        coverage_rows.append(
            {
                "league": "P1",
                "season_end_year": int(season) if pd.notna(season) else pd.NA,
                "matches": len(group),
                "one_x_two_odds_rows": int(group[ONE_X_TWO_COLUMNS].notna().all(axis=1).sum()) if set(ONE_X_TWO_COLUMNS).issubset(group.columns) else 0,
                "ah_odds_rows": int(group[AH_COLUMNS].notna().all(axis=1).sum()) if set(AH_COLUMNS).issubset(group.columns) else 0,
                "closing_ah_rows": int(group[CLOSING_AH_COLUMNS].notna().all(axis=1).sum()) if set(CLOSING_AH_COLUMNS).issubset(group.columns) else 0,
                "missing_columns": ",".join(missing_columns),
            }
        )
    coverage = pd.DataFrame(coverage_rows).sort_values("season_end_year")
    coverage["duplicates_removed_total"] = duplicates_removed
    coverage.to_csv(REPORT_DIR / "p1_match_data_coverage.csv", index=False)
    return output, pd.DataFrame(source_rows), schema, [{"duplicates_removed": duplicates_removed}]


def audit_transfermarkt_files() -> pd.DataFrame:
    rows = []
    for path in sorted(TM_DIR.glob("*.csv")):
        columns = read_csv_header(path)
        lower = {column.casefold(): column for column in columns}
        rows.append(
            {
                "file": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "rows": row_count(path),
                "columns_count": len(columns),
                "columns": ",".join(columns),
                "has_player_id": "player_id" in lower,
                "has_date": bool({"date", "transfer_date"} & set(lower)),
                "has_club_id": any(column in lower for column in ["club_id", "current_club_id", "from_club_id", "to_club_id"]),
                "has_market_value": any("market_value" in column for column in lower),
            }
        )
    return pd.DataFrame(rows)


def build_market_values() -> tuple[pd.DataFrame, dict]:
    valuations_path = TM_DIR / "player_valuations.csv"
    players_path = TM_DIR / "players.csv"
    valuations = pd.read_csv(valuations_path, low_memory=False)
    players = pd.read_csv(players_path, usecols=["player_id", "name"], low_memory=False)
    required = {"player_id", "date", "market_value_in_eur", "current_club_name"}
    missing = sorted(required - set(valuations.columns))
    if missing:
        return pd.DataFrame(), {"status": "missing_required_columns", "missing": ",".join(missing)}

    output = valuations[["date", "player_id", "current_club_name", "market_value_in_eur"]].copy()
    output["valuation_date"] = pd.to_datetime(output.pop("date"), errors="coerce").dt.normalize()
    output["market_value_eur"] = pd.to_numeric(output.pop("market_value_in_eur"), errors="coerce")
    output["club_name"] = output.pop("current_club_name")
    output = output.merge(players.rename(columns={"name": "player_name"}), on="player_id", how="left")
    output = output[["valuation_date", "player_id", "player_name", "club_name", "market_value_eur"]]
    output = output.dropna(subset=["valuation_date", "player_id", "market_value_eur"]).copy()
    output = output.sort_values(["valuation_date", "player_id", "club_name"]).reset_index(drop=True)
    PLAYERS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(MARKET_VALUES_PATH, index=False)
    return output, {
        "status": "ready_time_dated",
        "rows": len(output),
        "players": output["player_id"].nunique(),
        "clubs": output["club_name"].nunique(dropna=True),
        "date_min": output["valuation_date"].min().date().isoformat() if len(output) else "",
        "date_max": output["valuation_date"].max().date().isoformat() if len(output) else "",
    }


def build_club_history() -> tuple[pd.DataFrame, str, dict]:
    transfers = pd.read_csv(TM_DIR / "transfers.csv", low_memory=False)
    required = {"player_id", "player_name", "transfer_date", "to_club_name"}
    missing = sorted(required - set(transfers.columns))
    if missing:
        return pd.DataFrame(), "diagnostic_only", {"status": "missing_required_columns", "missing": ",".join(missing)}

    output = transfers[["player_id", "player_name", "transfer_date", "to_club_name"]].copy()
    output["valid_from"] = pd.to_datetime(output.pop("transfer_date"), errors="coerce").dt.normalize()
    output["club_name"] = output.pop("to_club_name")
    output = output.dropna(subset=["player_id", "valid_from", "club_name"]).copy()
    output = output.sort_values(["player_id", "valid_from", "club_name"]).drop_duplicates(["player_id", "valid_from", "club_name"])
    output["valid_to"] = output.groupby("player_id")["valid_from"].shift(-1)
    consistent = bool((output["valid_to"].isna() | (output["valid_to"] > output["valid_from"])).all())
    output = output[["player_id", "player_name", "club_name", "valid_from", "valid_to"]].reset_index(drop=True)
    future_rows = int((output["valid_from"] > pd.Timestamp.today().normalize()).sum())
    if consistent:
        output.to_csv(CLUB_HISTORY_PATH, index=False)
        status = "time_safe_partial_from_transfers"
    else:
        output.to_csv(CLUB_HISTORY_DIAGNOSTIC_PATH, index=False)
        status = "diagnostic_only"
    return output, status, {
        "status": status,
        "rows": len(output),
        "players": output["player_id"].nunique(),
        "clubs": output["club_name"].nunique(dropna=True),
        "future_valid_from_rows": future_rows,
        "logical_intervals": consistent,
        "limitation": "Intervals start only at observed to-club transfer dates; pre-first-transfer membership is not backfilled.",
    }


def load_processed_matches() -> dict[str, pd.DataFrame]:
    matches = {}
    for league in LEAGUES:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if not path.exists():
            matches[league] = pd.DataFrame()
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["league"] = league
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        frame["season_end_year"] = pd.to_numeric(frame["season_end_year"], errors="coerce").astype("Int64")
        matches[league] = frame.dropna(subset=["Date", "HomeTeam", "AwayTeam", "season_end_year"]).copy()
    return matches


def build_mapping(matches_by_league: dict[str, pd.DataFrame], clubs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    club_names = sorted(clubs["name"].dropna().astype(str).unique())
    club_by_norm = {}
    for name in club_names:
        club_by_norm.setdefault(normalize_name(name), name)
    normalized_clubs = sorted(club_by_norm)

    rows = []
    candidate_rows = []
    for league, frame in matches_by_league.items():
        if frame.empty:
            continue
        teams = pd.concat([frame["HomeTeam"], frame["AwayTeam"]], ignore_index=True).dropna().astype(str).drop_duplicates().sort_values()
        for team in teams:
            normalized = normalize_name(team)
            exact = club_by_norm.get(normalized, "")
            rows.append(
                {
                    "league": league,
                    "match_team": team,
                    "normalized_match_team": normalized,
                    "player_data_source": "transfermarkt",
                    "player_data_club_name": exact,
                    "normalized_player_data_club": normalize_name(exact) if exact else "",
                    "valid_from": "",
                    "valid_to": "",
                    "confidence": "high_exact_normalized" if exact else "unmatched",
                }
            )
            if not exact:
                for candidate in get_close_matches(normalized, normalized_clubs, n=5, cutoff=0.78):
                    candidate_rows.append(
                        {
                            "league": league,
                            "match_team": team,
                            "normalized_match_team": normalized,
                            "candidate_club_name": club_by_norm[candidate],
                            "normalized_candidate_club": candidate,
                            "candidate_only": True,
                        }
                    )

    mapping = pd.DataFrame(rows, columns=REQUIRED_MAPPING_COLUMNS)
    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(MAPPING_PATH, index=False)
    return mapping, pd.DataFrame(candidate_rows)


def coverage_preview(
    matches_by_league: dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
    market_values: pd.DataFrame,
    club_history: pd.DataFrame,
    club_history_status: str,
) -> pd.DataFrame:
    mapped = mapping[mapping["confidence"].eq("high_exact_normalized")].copy()
    mapped_by_team = {(row.league, row.match_team): row.player_data_club_name for row in mapped.itertuples(index=False)}
    valuation_clubs = set(market_values["club_name"].dropna().map(normalize_name)) if len(market_values) else set()
    safe_history = club_history_status.startswith("time_safe")
    history_clubs = set(club_history["club_name"].dropna().map(normalize_name)) if len(club_history) and safe_history else set()
    rows = []
    for league, frame in matches_by_league.items():
        if frame.empty:
            continue
        for season, group in frame.groupby("season_end_year"):
            home_mapped = group["HomeTeam"].map(lambda team: (league, team) in mapped_by_team)
            away_mapped = group["AwayTeam"].map(lambda team: (league, team) in mapped_by_team)
            home_club = group["HomeTeam"].map(lambda team: mapped_by_team.get((league, team), ""))
            away_club = group["AwayTeam"].map(lambda team: mapped_by_team.get((league, team), ""))
            valuation_home = home_club.map(lambda club: normalize_name(club) in valuation_clubs)
            valuation_away = away_club.map(lambda club: normalize_name(club) in valuation_clubs)
            history_home = home_club.map(lambda club: normalize_name(club) in history_clubs)
            history_away = away_club.map(lambda club: normalize_name(club) in history_clubs)
            both_time_safe = home_mapped & away_mapped & valuation_home & valuation_away & history_home & history_away
            rows.append(
                {
                    "league": league,
                    "season_end_year": int(season),
                    "matches": len(group),
                    "home_club_mapped": int(home_mapped.sum()),
                    "away_club_mapped": int(away_mapped.sum()),
                    "both_clubs_mapped": int((home_mapped & away_mapped).sum()),
                    "valuation_coverage": float((valuation_home & valuation_away).mean()) if len(group) else 0.0,
                    "club_history_coverage": float((history_home & history_away).mean()) if len(group) else 0.0,
                    "both_side_time_safe_coverage": float(both_time_safe.mean()) if len(group) else 0.0,
                    "safe_for_feature_build": bool(safe_history),
                }
            )
    preview = pd.DataFrame(rows).sort_values(["league", "season_end_year"])
    preview.to_csv(REPORT_DIR / "player_squad_feature_coverage_preview.csv", index=False)
    return preview


def write_reports(
    p1_matches: pd.DataFrame,
    p1_sources: pd.DataFrame,
    tm_audit: pd.DataFrame,
    market_values: pd.DataFrame,
    market_status: dict,
    club_history: pd.DataFrame,
    club_history_status: str,
    club_history_meta: dict,
    mapping: pd.DataFrame,
    candidates: pd.DataFrame,
    preview: pd.DataFrame,
) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p1_coverage = pd.read_csv(REPORT_DIR / "p1_match_data_coverage.csv") if (REPORT_DIR / "p1_match_data_coverage.csv").exists() else pd.DataFrame()
    p1_lines = [
        "# P1 Match Data Import Audit",
        "",
        f"Processed file: `{PROCESSED_P1}`",
        f"Rows written: {len(p1_matches)}",
        f"Raw files inspected: {len(p1_sources)}",
        "",
        "## Raw Files",
        markdown_table(p1_sources, ["file", "size_bytes", "rows", "columns", "football_data_style", "missing_core_columns"]),
        "",
        "## Season Coverage",
        markdown_table(p1_coverage, ["season_end_year", "matches", "one_x_two_odds_rows", "ah_odds_rows", "closing_ah_rows", "missing_columns"]),
        "",
        "Closing odds are retained only as raw/diagnostic columns and are not designated as bet-time-safe features.",
    ]
    (REPORT_DIR / "p1_match_data_import_audit.md").write_text("\n".join(p1_lines) + "\n", encoding="utf-8")

    useful = tm_audit[tm_audit["file"].isin(["players.csv", "clubs.csv", "competitions.csv", "games.csv", "appearances.csv", "player_valuations.csv", "transfers.csv", "game_lineups.csv"])]
    tm_lines = [
        "# Transfermarkt Player Dataset Audit",
        "",
        "No website scraping or external API calls were made. Local CSVs only.",
        "",
        "## Files",
        markdown_table(tm_audit, ["file", "size_bytes", "rows", "columns_count", "has_player_id", "has_date", "has_club_id", "has_market_value"]),
        "",
        "## Useful Tables",
        markdown_table(useful, ["file", "rows", "columns_count", "has_player_id", "has_date", "has_club_id", "has_market_value"]),
        "",
        "## Valuation Schema",
        f"Status: `{market_status.get('status')}`. Required columns found: player_id, date, market_value_in_eur, current_club_name.",
        f"Rows written to `{MARKET_VALUES_PATH}`: {len(market_values)}",
    ]
    (REPORT_DIR / "transfermarkt_player_dataset_audit.md").write_text("\n".join(tm_lines) + "\n", encoding="utf-8")

    mv_cov = pd.DataFrame(
        [
            {
                "rows": len(market_values),
                "players": market_values["player_id"].nunique() if len(market_values) else 0,
                "clubs": market_values["club_name"].nunique(dropna=True) if len(market_values) else 0,
                "date_min": market_values["valuation_date"].min() if len(market_values) else pd.NaT,
                "date_max": market_values["valuation_date"].max() if len(market_values) else pd.NaT,
                "numeric_market_value_rows": int(pd.to_numeric(market_values["market_value_eur"], errors="coerce").notna().sum()) if len(market_values) else 0,
            }
        ]
    )
    mv_cov.to_csv(REPORT_DIR / "transfermarkt_player_market_values_coverage.csv", index=False)
    history_cov = pd.DataFrame([club_history_meta])
    history_cov.to_csv(REPORT_DIR / "transfermarkt_player_club_history_coverage.csv", index=False)

    unmatched = mapping[mapping["confidence"].eq("unmatched")].groupby("league").size().reset_index(name="unmatched_teams")
    matched = mapping[mapping["confidence"].ne("unmatched")].groupby("league").size().reset_index(name="matched_teams")
    mapping_status = matched.merge(unmatched, on="league", how="outer").fillna(0)
    readiness = bool(len(market_values) and club_history_status.startswith("time_safe") and preview["both_side_time_safe_coverage"].gt(0).any())
    if not len(p1_matches):
        classification = "insufficient_data"
    elif not len(market_values):
        classification = "p1_ready_players_club_history_missing"
    elif not club_history_status.startswith("time_safe"):
        classification = "p1_ready_players_club_history_missing"
    elif readiness:
        classification = "p1_ready_players_partially_ready"
    else:
        classification = "p1_ready_players_partially_ready"

    readiness_lines = [
        "# Player Squad Readiness Status",
        "",
        f"Market values: `{market_status.get('status')}`",
        f"Club history: `{club_history_status}`",
        f"Feature coverage preview rows: {len(preview)}",
        f"Ready for feature build: {'yes' if readiness else 'partial only'}",
        "",
        "## Mapping Status",
        markdown_table(mapping_status, ["league", "matched_teams", "unmatched_teams"]),
        "",
        "## Leakage Controls",
        "- Current club fields from `players.csv` were not used for historical membership.",
        "- Valuation rows use their own dated `current_club_name`; dates must strictly precede match dates in downstream features.",
        "- Transfer-based club history starts only at observed transfer dates and does not backfill pre-first-transfer membership.",
        "- Lineups were audited as local tables only and remain diagnostic unless a reliable pre-kickoff timestamp is established.",
    ]
    (REPORT_DIR / "player_squad_readiness_status.md").write_text("\n".join(readiness_lines) + "\n", encoding="utf-8")

    candidate_path = REPORT_DIR / "player_squad_team_name_mapping_candidates.csv"
    candidates.to_csv(candidate_path, index=False)
    final_lines = [
        "# Player Squad Data Prep Final Summary",
        "",
        f"P1 status: `{ 'ready' if len(p1_matches) else 'missing_or_failed' }`",
        f"Transfermarkt valuation status: `{market_status.get('status')}`",
        f"Club history status: `{club_history_status}`",
        "Mapping status: exact normalized mappings only were written; uncertain candidates are report-only.",
        "",
        "## Leakage Risks",
        "- Club history is partial because it only covers observed to-club transfer intervals.",
        "- Transfer effective dates are safe as lower bounds, but they are not announcement timestamps.",
        "- Valuation club names are dated in valuation rows; downstream feature code must require `valuation_date < match Date`.",
        "- Do not use `players.current_club_*` for historical matches.",
        "- Treat lineups as diagnostic only.",
        "",
        f"Player/squad features ready: `{ 'partial_ready_for_preview_only' if readiness else 'not_ready_for_final_feature_build' }`",
        f"Final classification: **{classification}**",
        "",
        "Exact next command/prompt needed:",
        "`Run a leakage-closed player/squad feature build using only transfermarkt_market_values.csv rows with valuation_date < match Date, player_club_history.csv intervals active on match Date, and reviewed high-confidence mappings; do not train models.`",
    ]
    (REPORT_DIR / "player_squad_data_prep_final_summary.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    return classification


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLAYERS_DIR.mkdir(parents=True, exist_ok=True)
    p1_matches, p1_sources, _, _ = import_p1_matches()
    tm_audit = audit_transfermarkt_files()
    market_values, market_status = build_market_values()
    club_history, club_history_status, club_history_meta = build_club_history()
    clubs = pd.read_csv(TM_DIR / "clubs.csv", usecols=["club_id", "name", "domestic_competition_id"], low_memory=False)
    matches_by_league = load_processed_matches()
    mapping, candidates = build_mapping(matches_by_league, clubs)
    preview = coverage_preview(matches_by_league, mapping, market_values, club_history, club_history_status)
    classification = write_reports(
        p1_matches,
        p1_sources,
        tm_audit,
        market_values,
        market_status,
        club_history,
        club_history_status,
        club_history_meta,
        mapping,
        candidates,
        preview,
    )
    print(f"P1 rows: {len(p1_matches)}")
    print(f"Market value rows: {len(market_values)}")
    print(f"Club history rows: {len(club_history)} ({club_history_status})")
    print(f"Mapping rows: {len(mapping)}")
    print(f"Final classification: {classification}")


if __name__ == "__main__":
    main()
