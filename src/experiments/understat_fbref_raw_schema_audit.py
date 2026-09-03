from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import shutil
import sqlite3

import numpy as np
import pandas as pd


REPORT_DIR = Path("outputs/reports")
INTERIM_DIR = Path("data/interim/external_audits")

UNDERSTAT_ZIP = Path("data/raw_external/understat_manual/understat_archive.zip")
FBREF_MATCH_ZIP = Path("data/raw_external/fbref_2023_2024_manual/fbref_2023_2024_archive.zip")
FBREF_SEASONS_ZIP = Path("data/raw_external/fbref_player_seasons_manual/fbref_player_seasons_2017_2024.zip")
FBREF_2425_ZIP = Path("data/raw_external/fbref_player_seasons_manual/fbref_player_seasons_2024_2025.zip")

SCHEMA_MD = REPORT_DIR / "understat_fbref_raw_schema_audit.md"
UNDERSTAT_COVERAGE_CSV = REPORT_DIR / "understat_raw_coverage.csv"
FBREF_COVERAGE_CSV = REPORT_DIR / "fbref_raw_coverage.csv"
FBREF_SCHEMA_CSV = REPORT_DIR / "fbref_player_season_schema_audit.csv"
RECOMMENDATION_MD = REPORT_DIR / "v4_external_source_recommendation.md"


POSTMATCH_UNDERSTAT = {
    "xG",
    "xGA",
    "npxG",
    "npxGA",
    "ppda",
    "ppda_allowed",
    "deep",
    "deep_allowed",
    "scored",
    "missed",
    "xpts",
    "result",
    "wins",
    "draws",
    "loses",
    "pts",
    "npxGD",
}
FBREF_POSTMATCH_TOKENS = [
    "Gls",
    "Ast",
    "xG",
    "npxG",
    "xAG",
    "GA",
    "Save",
    "W",
    "D",
    "L",
    "CS",
    "onG",
    "onGA",
    "onxG",
    "onxGA",
    "PPM",
    "+/-",
]


def count_zip_rows(zf: ZipFile, member: str) -> int:
    with zf.open(member) as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def read_csv_member(zip_path: Path, member: str, **kwargs) -> pd.DataFrame:
    with ZipFile(zip_path) as zf:
        with zf.open(member) as handle:
            return pd.read_csv(handle, low_memory=False, **kwargs)


def zip_inventory(zip_path: Path) -> pd.DataFrame:
    rows = []
    with ZipFile(zip_path) as zf:
        for info in zf.infolist():
            row_count = count_zip_rows(zf, info.filename) if info.filename.lower().endswith(".csv") else np.nan
            rows.append(
                {
                    "archive": str(zip_path),
                    "member": info.filename,
                    "compressed_size": info.compress_size,
                    "uncompressed_size": info.file_size,
                    "csv_rows": row_count,
                }
            )
    return pd.DataFrame(rows)


def inspect_understat_db() -> pd.DataFrame:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    db_path = INTERIM_DIR / "understat.db"
    with ZipFile(UNDERSTAT_ZIP) as zf:
        with zf.open("understat.db") as src, db_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    rows = []
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql_query("select name from sqlite_master where type='table' order by name", conn)
        for table in tables["name"].tolist():
            count = pd.read_sql_query(f'select count(*) as rows from "{table}"', conn)["rows"].iloc[0]
            cols = pd.read_sql_query(f'pragma table_info("{table}")', conn)
            rows.append(
                {
                    "archive": str(UNDERSTAT_ZIP),
                    "member": "understat.db",
                    "table": table,
                    "rows": int(count),
                    "columns": "|".join(cols["name"].astype(str)),
                }
            )
    return pd.DataFrame(rows)


def understat_coverage() -> tuple[pd.DataFrame, dict[str, object]]:
    game = read_csv_member(UNDERSTAT_ZIP, "game_stats.csv")
    player = read_csv_member(UNDERSTAT_ZIP, "player_stats.csv")
    game["date_parsed"] = pd.to_datetime(game["date"], errors="coerce")
    rows = []
    dup_key = ["league", "season", "club_name", "home_away", "date"]
    rows.append(
        {
            "source": "understat_game_stats",
            "segment": "all",
            "group": "all",
            "rows": len(game),
            "leagues": game["league"].nunique(),
            "seasons": game["season"].nunique(),
            "teams": game["club_name"].nunique(),
            "min_date": game["date_parsed"].min(),
            "max_date": game["date_parsed"].max(),
            "date_parse_failures": int(game["date_parsed"].isna().sum()),
            "duplicate_team_date_rows": int(game.duplicated(dup_key).sum()),
            "missing_key_values": int(game[["id", "league", "season", "club_name", "home_away", "date"]].isna().any(axis=1).sum()),
        }
    )
    for league, g in game.groupby("league", dropna=False):
        rows.append(
            {
                "source": "understat_game_stats",
                "segment": "by_league",
                "group": league,
                "rows": len(g),
                "leagues": 1,
                "seasons": g["season"].nunique(),
                "teams": g["club_name"].nunique(),
                "min_date": g["date_parsed"].min(),
                "max_date": g["date_parsed"].max(),
                "date_parse_failures": int(g["date_parsed"].isna().sum()),
                "duplicate_team_date_rows": int(g.duplicated(dup_key).sum()),
                "missing_key_values": int(g[["id", "league", "season", "club_name", "home_away", "date"]].isna().any(axis=1).sum()),
            }
        )
    for season, g in game.groupby("season", dropna=False):
        rows.append(
            {
                "source": "understat_game_stats",
                "segment": "by_season",
                "group": season,
                "rows": len(g),
                "leagues": g["league"].nunique(),
                "seasons": 1,
                "teams": g["club_name"].nunique(),
                "min_date": g["date_parsed"].min(),
                "max_date": g["date_parsed"].max(),
                "date_parse_failures": int(g["date_parsed"].isna().sum()),
                "duplicate_team_date_rows": int(g.duplicated(dup_key).sum()),
                "missing_key_values": int(g[["id", "league", "season", "club_name", "home_away", "date"]].isna().any(axis=1).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(UNDERSTAT_COVERAGE_CSV, index=False)
    info = {
        "game_rows": len(game),
        "player_rows": len(player),
        "game_columns": list(game.drop(columns=["date_parsed"]).columns),
        "player_columns": list(player.columns),
        "leagues": sorted(game["league"].dropna().unique().tolist()),
        "seasons": sorted(game["season"].dropna().unique().tolist()),
        "teams": int(game["club_name"].nunique()),
        "date_min": game["date_parsed"].min(),
        "date_max": game["date_parsed"].max(),
        "date_parse_failures": int(game["date_parsed"].isna().sum()),
        "postmatch_columns": [c for c in game.columns if c in POSTMATCH_UNDERSTAT],
    }
    return out, info


def fbref_match_like_coverage() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    info: dict[str, object] = {"archive_label_issue": False}
    with ZipFile(FBREF_MATCH_ZIP) as zf:
        members = zf.namelist()
        info["members"] = members
    if set(members) == {"EloRatings.csv", "Matches.csv"}:
        info["archive_label_issue"] = True
    for member in members:
        if not member.lower().endswith(".csv"):
            continue
        df = read_csv_member(FBREF_MATCH_ZIP, member)
        date_col = "MatchDate" if "MatchDate" in df.columns else ("date" if "date" in df.columns else None)
        dates = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.Series(pd.NaT, index=df.index)
        team_cols = [c for c in ["HomeTeam", "AwayTeam", "club", "Squad", "squad"] if c in df.columns]
        teams = pd.concat([df[c].dropna().astype(str) for c in team_cols], ignore_index=True).nunique() if team_cols else np.nan
        league_col = "Division" if "Division" in df.columns else ("Comp" if "Comp" in df.columns else None)
        rows.append(
            {
                "archive": str(FBREF_MATCH_ZIP),
                "member": member,
                "source_family": "fbref_2023_2024_manual_label",
                "rows": len(df),
                "columns": len(df.columns),
                "league_count": int(df[league_col].nunique()) if league_col else np.nan,
                "leagues": "|".join(sorted(df[league_col].dropna().astype(str).unique())) if league_col else "",
                "season": "unknown",
                "team_count": teams,
                "player_count": np.nan,
                "min_date": dates.min(),
                "max_date": dates.max(),
                "date_parse_failures": int(dates.isna().sum()) if date_col else np.nan,
                "schema_warning": "archive_appears_to_contain_clubelo_files_not_fbref" if info["archive_label_issue"] else "",
                "safe_use": "diagnostic_only_until_source_identity_resolved",
            }
        )
    return rows, info


def fbref_player_season_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    coverage_rows = []
    schema_rows = []
    for zip_path in [FBREF_SEASONS_ZIP, FBREF_2425_ZIP]:
        with ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                df = read_csv_member(zip_path, member)
                season = ""
                if "season" in df.columns:
                    vals = df["season"].dropna().astype(str).unique()
                    season = "|".join(sorted(vals[:5]))
                elif "2024_2025" in member or "2024-2025" in member:
                    season = "2024-2025_inferred_from_filename"
                else:
                    season = member.replace("cleaned_", "").replace(".csv", "")
                comp_col = "comp" if "comp" in df.columns else ("Comp" if "Comp" in df.columns else None)
                squad_col = "squad" if "squad" in df.columns else ("Squad" if "Squad" in df.columns else None)
                player_col = "player" if "player" in df.columns else ("Player" if "Player" in df.columns else None)
                post_cols = [c for c in df.columns if any(tok == c or tok in c for tok in FBREF_POSTMATCH_TOKENS)]
                coverage_rows.append(
                    {
                        "archive": str(zip_path),
                        "member": member,
                        "source_family": "fbref_player_seasons",
                        "rows": len(df),
                        "columns": len(df.columns),
                        "league_count": int(df[comp_col].nunique()) if comp_col else np.nan,
                        "leagues": "|".join(sorted(df[comp_col].dropna().astype(str).unique())) if comp_col else "",
                        "season": season,
                        "team_count": int(df[squad_col].nunique()) if squad_col else np.nan,
                        "player_count": int(df[player_col].nunique()) if player_col else np.nan,
                        "min_date": "",
                        "max_date": "",
                        "date_parse_failures": np.nan,
                        "schema_warning": "season_aggregate_no_match_dates",
                        "safe_use": "prior_completed_season_features_only",
                    }
                )
                schema_rows.append(
                    {
                        "archive": str(zip_path),
                        "member": member,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": "|".join(map(str, df.columns)),
                        "identity_columns": "|".join([c for c in [player_col, squad_col, comp_col, "season"] if c]),
                        "post_or_same_season_aggregate_columns": "|".join(post_cols[:120]),
                        "leakage_policy": "do_not_use_for_same_season_pre_match_prediction; prior_completed_season_only",
                    }
                )
    coverage = pd.DataFrame(coverage_rows)
    schema = pd.DataFrame(schema_rows)
    schema.to_csv(FBREF_SCHEMA_CSV, index=False)
    info = {
        "files": len(schema),
        "rows": int(coverage["rows"].sum()),
        "seasons": sorted(coverage["season"].astype(str).unique().tolist()),
    }
    return coverage, schema, info


def write_reports(
    inventory: pd.DataFrame,
    db_schema: pd.DataFrame,
    under_info: dict[str, object],
    fbref_match_info: dict[str, object],
    fbref_coverage: pd.DataFrame,
    fbref_schema: pd.DataFrame,
) -> None:
    under_ready = "v4.1_ready_for_mapping_and_rolling_past_only_prototype"
    fbref_match_ready = "not_ready_source_identity_mismatch"
    fbref_seasons_ready = "v4.2_ready_for_prior_completed_season_mapping_prototype"
    fbref_2425_ready = "v4.3_hold_until_completed_or_use_only_for_future_2025_26_after_season_completion"
    lines = [
        "# Understat / FBref Raw Schema Audit",
        "",
        "No features were built, no models were trained, no joins to the main matrix were made, and raw ZIP files were not modified.",
        "",
        "## Archive Inventory",
        inventory.to_markdown(index=False),
        "",
        "## Understat",
        f"- `game_stats.csv` rows: {under_info['game_rows']}",
        f"- `player_stats.csv` rows: {under_info['player_rows']}",
        f"- Leagues: {', '.join(under_info['leagues'])}",
        f"- Seasons: {min(under_info['seasons'])} to {max(under_info['seasons'])}",
        f"- Teams: {under_info['teams']}",
        f"- Date range: {under_info['date_min']} to {under_info['date_max']}",
        f"- Date parse failures: {under_info['date_parse_failures']}",
        f"- Current-match/post-match columns requiring lagging: {', '.join(under_info['postmatch_columns'])}",
        f"- Recommendation: `{under_ready}`",
        "",
        "Understat `game_stats` can only be used after team mapping as rolling past-only match features. Current-match xG/result/scored/missed must never be used as pre-match features.",
        "",
        "## Understat SQLite",
        db_schema.to_markdown(index=False),
        "",
        "## FBref 2023/2024 Manual Archive",
        f"- Members: {', '.join(fbref_match_info['members'])}",
        f"- Source-label issue: `{fbref_match_info['archive_label_issue']}`",
        f"- Recommendation: `{fbref_match_ready}`",
        "",
        "The archive labeled FBref 2023/2024 appears to contain ClubElo-style `EloRatings.csv` and `Matches.csv`, including match scores/results/post-match stats. Treat as diagnostic only until source identity is resolved.",
        "",
        "## FBref Player Seasons",
        f"- Audited files: {len(fbref_schema)}",
        f"- Total rows across player-season files: {int(fbref_coverage[fbref_coverage['source_family'].eq('fbref_player_seasons')]['rows'].sum())}",
        f"- 2017/18-2023/24 recommendation: `{fbref_seasons_ready}`",
        f"- 2024/25 recommendation: `{fbref_2425_ready}`",
        "",
        "FBref player-season aggregates can only become prior-completed-season features. Do not use same-season aggregate stats for same-season pre-match prediction.",
        "",
    ]
    SCHEMA_MD.write_text("\n".join(lines), encoding="utf-8")

    rec = [
        "# V4 External Source Recommendation",
        "",
        "| Source | Recommendation | Rationale | Safe Next Step |",
        "|---|---|---|---|",
        f"| Understat game_stats | `{under_ready}` | Match-level coverage is structured with dates/leagues/seasons, but all useful match stats are current-match values and must be lagged. | Build a separate mapping/date-safety audit, then rolling past-only team form features. |",
        f"| FBref 2023/2024 manual archive | `{fbref_match_ready}` | Archive contents look like ClubElo files, not FBref; Matches.csv includes result/score/post-match stats. | Resolve provenance or quarantine as diagnostic only. |",
        f"| FBref player seasons 2017/18-2023/24 | `{fbref_seasons_ready}` | Season aggregate player data is structured across completed seasons. | Map squads and aggregate only prior completed season profiles for future-season fixtures. |",
        f"| FBref player seasons 2024/25 | `{fbref_2425_ready}` | 2024/25 season aggregates are unsafe for 2024/25 pre-match predictions. | Use only after season completion for later seasons, or exclude from v4.2. |",
        "",
        "No confirmed edge is claimed.",
        "",
    ]
    RECOMMENDATION_MD.write_text("\n".join(rec), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    inventories = [zip_inventory(p) for p in [UNDERSTAT_ZIP, FBREF_MATCH_ZIP, FBREF_SEASONS_ZIP, FBREF_2425_ZIP]]
    inventory = pd.concat(inventories, ignore_index=True)
    db_schema = inspect_understat_db()
    under_coverage, under_info = understat_coverage()
    fbref_match_rows, fbref_match_info = fbref_match_like_coverage()
    fbref_player_cov, fbref_schema, fbref_player_info = fbref_player_season_audit()
    fbref_coverage = pd.concat([pd.DataFrame(fbref_match_rows), fbref_player_cov], ignore_index=True, sort=False)
    fbref_coverage.to_csv(FBREF_COVERAGE_CSV, index=False)
    write_reports(inventory, db_schema, under_info, fbref_match_info, fbref_coverage, fbref_schema)
    print(
        {
            "understat_game_rows": under_info["game_rows"],
            "understat_leagues": len(under_info["leagues"]),
            "fbref_label_issue": fbref_match_info["archive_label_issue"],
            "fbref_player_files": fbref_player_info["files"],
            "fbref_player_rows": fbref_player_info["rows"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
