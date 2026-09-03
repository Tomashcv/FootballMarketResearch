from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


MANIFEST_PATH = Path("outputs/reports/player_squad_strength_missing_data_manifest.md")
SCHEMA_PATH = Path("outputs/reports/player_squad_strength_required_schema.csv")


SCHEMA_ROWS = [
    {
        "dataset_group": "Liga Portugal match/odds data",
        "expected_filename": "data/processed/P1/P1_matches.csv",
        "required_columns": "Date;HomeTeam;AwayTeam;FTHG;FTAG;FTR;season_start_year;season_end_year;AvgH;AvgD;AvgA;AHh;AvgAHH;AvgAHA",
        "optional_columns": "MaxH;MaxD;MaxA;B365H;B365D;B365A;PSH;PSD;PSA;AvgCH;AvgCD;AvgCA;AHCh;AvgCAHH;AvgCAHA;source_file",
        "required_coverage": "2011-2025 seasons if Portugal is included in backtests; future seasons for paper tracking",
        "use": "backtesting_and_future_paper_tracking",
        "join_to_match_data": "Native match table keyed by league=P1, Date, HomeTeam, AwayTeam; same processed schema as E0/I1/SP1/D1/F1",
        "leakage_risks": "Do not use closing odds as features; closing columns only for diagnostics. Preserve match dates and season boundaries.",
        "minimum_viable": False,
        "ideal": True,
        "diagnostic_only_if_missing_time": False,
    },
    {
        "dataset_group": "FIFA/EA/SoFIFA historical player ratings by season",
        "expected_filename": "data/external/players/sofifa_historical_player_ratings.csv",
        "required_columns": "snapshot_date;season_start_year;player_id;player_name;club_name;league_name;overall;potential;position",
        "optional_columns": "dob;age;nationality;height_cm;weight_kg;preferred_foot;weak_foot;skill_moves;work_rate;player_url;sofifa_version;source_file",
        "required_coverage": "At least 2020-2025 for initial Away AH research; 2011-2025 preferred for full match-data span; each row must be available on or before snapshot_date",
        "use": "backtesting_if_snapshot_date_precedes_match;future_paper_tracking",
        "join_to_match_data": "Normalize club_name through data/manual/player_squad_team_name_mapping.csv, then latest snapshot_date strictly before match Date for each HomeTeam/AwayTeam",
        "leakage_risks": "Current FIFA/EA ratings for historical matches are leakage unless the release/snapshot date is before the match. Season files without release dates should be treated as season-start only unless proven otherwise.",
        "minimum_viable": True,
        "ideal": True,
        "diagnostic_only_if_missing_time": True,
    },
    {
        "dataset_group": "Transfermarkt-style player market values by date or season",
        "expected_filename": "data/external/players/transfermarkt_market_values.csv",
        "required_columns": "valuation_date;player_id;player_name;club_name;market_value_eur",
        "optional_columns": "season_start_year;league_name;position;age;nationality;contract_until;source_url;currency;source_file",
        "required_coverage": "At least 2020-2025 for initial research; 2011-2025 preferred. Multiple dated valuations per player are ideal.",
        "use": "backtesting_if_valuation_date_precedes_match;future_paper_tracking",
        "join_to_match_data": "Normalize club_name through team mapping; for each club and match Date, use latest valuation_date strictly before Date, aggregate active players by club snapshot",
        "leakage_risks": "Latest/current market values cannot be backfilled into past seasons. End-of-season or post-transfer-window values must not be used before their valuation date.",
        "minimum_viable": True,
        "ideal": True,
        "diagnostic_only_if_missing_time": True,
    },
    {
        "dataset_group": "Player-club history or squad membership by date/season",
        "expected_filename": "data/external/players/player_club_history.csv",
        "required_columns": "player_id;player_name;club_name;valid_from;valid_to",
        "optional_columns": "season_start_year;league_name;loan_flag;transfer_type;from_club;to_club;position;squad_number;source_file",
        "required_coverage": "Must cover every player-season used by ratings/market values; at least 2020-2025 for initial research and 2011-2025 for full history",
        "use": "required_for_backtesting_when ratings/value rows do not already encode dated club membership",
        "join_to_match_data": "Player is eligible for club snapshot only when valid_from <= match Date and valid_to is null or valid_to > match Date; club_name then maps to match HomeTeam/AwayTeam",
        "leakage_risks": "Season-level squad lists can leak winter transfers if treated as valid from season start. Must represent valid_from/valid_to or be restricted to known season-start snapshots.",
        "minimum_viable": True,
        "ideal": True,
        "diagnostic_only_if_missing_time": True,
    },
    {
        "dataset_group": "Team name mapping between match data, FIFA/SoFIFA, and Transfermarkt",
        "expected_filename": "data/manual/player_squad_team_name_mapping.csv",
        "required_columns": "league;match_team;normalized_match_team;player_data_source;player_data_club_name;normalized_player_data_club;valid_from;valid_to;confidence",
        "optional_columns": "country;notes;source_url;reviewed_by;reviewed_at",
        "required_coverage": "Every club appearing in E0/I1/SP1/D1/F1 2011-2025; P1 if added; source-specific aliases for SoFIFA and Transfermarkt",
        "use": "backtesting_and_future_paper_tracking",
        "join_to_match_data": "Match HomeTeam/AwayTeam -> normalized_match_team -> source-specific player_data_club_name; valid date window must include match Date",
        "leakage_risks": "Ambiguous aliases, renamed clubs, B teams, shared city names, and historical name changes can create false joins. Low-confidence mappings must stay diagnostic.",
        "minimum_viable": True,
        "ideal": True,
        "diagnostic_only_if_missing_time": False,
    },
    {
        "dataset_group": "Optional lineups",
        "expected_filename": "data/external/players/match_lineups.csv",
        "required_columns": "match_date;league;home_team;away_team;player_id;player_name;club_name;starter;position",
        "optional_columns": "minutes;sub_on_minute;sub_off_minute;formation;captain;source_file;announced_at",
        "required_coverage": "Useful for future paper tracking immediately; historical backtests only if lineup announcement time is before match kickoff",
        "use": "future_paper_tracking_preferred;historical_backtesting_only_with announced_at_before_match",
        "join_to_match_data": "Date/league/home_team/away_team plus mapped club and player_id; only rows with announced_at before kickoff are feature-safe",
        "leakage_risks": "Final lineups scraped after kickoff or after match completion are leakage for backtests unless announcement timestamp is stored and pre-match.",
        "minimum_viable": False,
        "ideal": True,
        "diagnostic_only_if_missing_time": True,
    },
    {
        "dataset_group": "Optional injuries/suspensions",
        "expected_filename": "data/external/players/player_availability.csv",
        "required_columns": "as_of_date;player_id;player_name;club_name;status",
        "optional_columns": "reason;expected_return_date;source;source_url;severity;competition;updated_at",
        "required_coverage": "Future paper tracking if collected prospectively; historical backtests only with dated as_of_date/update timestamps",
        "use": "future_paper_tracking_preferred;historical_backtesting_only_if dated before match",
        "join_to_match_data": "Latest availability as_of_date strictly before match Date for players in club snapshot",
        "leakage_risks": "Retrospective injury databases often encode post-match knowledge. Missing update timestamps make them diagnostic only.",
        "minimum_viable": False,
        "ideal": True,
        "diagnostic_only_if_missing_time": True,
    },
]


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[columns].to_markdown(index=False, headers=headers)


def write_manifest(schema: pd.DataFrame) -> None:
    backtest = schema[schema["use"].str.contains("backtesting", na=False)].copy()
    minimum = schema[schema["minimum_viable"].eq(True)].copy()
    ideal = schema[schema["ideal"].eq(True)].copy()
    diagnostic = schema[schema["diagnostic_only_if_missing_time"].eq(True)].copy()
    lines = [
        "# Player/Squad Strength Missing-Data Manifest",
        "",
        "Context: the current audit found no local player ratings, market-value, squad, transfer-history, lineup, injury, or suspension datasets. Time-safe player observations loaded: 0. E0/I1/SP1/D1/F1 match data exists from 2011-2025; P1/Liga Portugal match data is missing.",
        "",
        "This manifest defines the local datasets required before the player/squad strength layer can be used in backtests. It does not train models, create betting strategies, scrape websites, edit raw match data, or permit current ratings for historical matches.",
        "",
        "## Required Dataset Checklist",
        "",
        markdown_table(
            schema,
            [
                "dataset_group",
                "expected_filename",
                "required_columns",
                "required_coverage",
                "use",
                "join_to_match_data",
                "leakage_risks",
            ],
            [
                "Dataset",
                "Expected filename",
                "Required columns",
                "Required coverage",
                "Use",
                "Join",
                "Leakage risks",
            ],
        ),
        "",
        "## Optional Columns",
        "",
        markdown_table(
            schema,
            ["dataset_group", "optional_columns"],
            ["Dataset", "Optional columns"],
        ),
        "",
        "## Backtest-Critical Inputs",
        "",
        markdown_table(
            backtest,
            ["dataset_group", "expected_filename", "required_coverage", "leakage_risks"],
            ["Dataset", "Expected filename", "Coverage", "Leakage risks"],
        ),
        "",
        "## Minimum Viable Dataset To Start",
        "",
        "Minimum viable means enough to run a time-safe feature coverage pass and a simple squad-strength baseline on at least the 2020-2025 top-five-league window. The minimum set is:",
        "",
        markdown_table(
            minimum,
            ["dataset_group", "expected_filename", "required_columns"],
            ["Dataset", "Expected filename", "Required columns"],
        ),
        "",
        "Practical minimum: dated FIFA/SoFIFA ratings or dated Transfermarkt-style values, dated club membership if not embedded in those rows, and a reviewed team-name mapping. P1 is not required for the first top-five-league squad layer, but is required before Portugal can be included.",
        "",
        "## Ideal Dataset",
        "",
        "The ideal dataset covers 2011-2025 for E0/I1/SP1/D1/F1 plus P1, has multiple dated snapshots per season, explicit player IDs, club validity windows, positions, age/date-of-birth, and source metadata.",
        "",
        markdown_table(
            ideal,
            ["dataset_group", "expected_filename", "required_coverage"],
            ["Dataset", "Expected filename", "Coverage"],
        ),
        "",
        "## Datasets That Are Not Time-Safe And Must Stay Diagnostic Only",
        "",
        markdown_table(
            diagnostic,
            ["dataset_group", "expected_filename", "leakage_risks"],
            ["Dataset", "Expected filename", "Why diagnostic only without timestamps"],
        ),
        "",
        "Specific diagnostic-only examples:",
        "",
        "- A single current FIFA/EA/SoFIFA export used for 2011-2025 historical matches.",
        "- Current Transfermarkt market values without valuation dates.",
        "- Season squad lists that include winter arrivals but no valid_from/valid_to dates.",
        "- Retrospective lineups without announcement timestamps.",
        "- Injury/suspension tables with post-match updates but no as-of timestamp.",
        "",
        "## Acceptance Gate Before Modeling",
        "",
        "Before any model or betting research uses the layer, rerun `src/experiments/player_squad_strength_data_layer.py` and require non-zero time-safe coverage, clean entity-resolution diagnostics, and no leakage warning. Closing odds remain diagnostic only.",
    ]
    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = pd.DataFrame(SCHEMA_ROWS)
    schema.to_csv(SCHEMA_PATH, index=False)
    write_manifest(schema)
    print(MANIFEST_PATH)
    print(SCHEMA_PATH)


if __name__ == "__main__":
    main()
