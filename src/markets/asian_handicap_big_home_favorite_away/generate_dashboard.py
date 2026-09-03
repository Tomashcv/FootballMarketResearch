import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.paths import get_league_matches_path
from src.common.paths import get_market_output_dir
from src.common.paths import get_global_dashboard_dir


MARKET_NAME = "asian_handicap_big_home_favorite_away"


def format_float(value, digits=3):
    if pd.isna(value):
        return "-"

    return f"{float(value):.{digits}f}"


def format_percent(value):
    if pd.isna(value):
        return "-"

    return f"{float(value) * 100:.2f}%"


def read_csv_if_exists(path):
    if path.exists():
        return pd.read_csv(path, low_memory=False)

    return pd.DataFrame()


def ensure_paper_files(output_dir):
    paper_dir = output_dir / "paper_tracking"
    paper_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = paper_dir / "paper_ledger.csv"
    upcoming_path = paper_dir / "upcoming_candidates.csv"

    if not ledger_path.exists():
        ledger_columns = [
            "created_at",
            "match_date",
            "home_team",
            "away_team",
            "market",
            "variant",
            "ah_line",
            "away_handicap",
            "away_odds",
            "bookmaker",
            "rule",
            "stake",
            "status",
            "settled_profit",
            "notes",
        ]

        pd.DataFrame(columns=ledger_columns).to_csv(ledger_path, index=False)

    if not upcoming_path.exists():
        upcoming_columns = [
            "match_date",
            "home_team",
            "away_team",
            "market",
            "variant",
            "ah_line",
            "away_handicap",
            "away_odds",
            "bookmaker",
            "rule",
            "candidate_status",
            "notes",
        ]

        pd.DataFrame(columns=upcoming_columns).to_csv(upcoming_path, index=False)

    return ledger_path, upcoming_path


def calculate_overall(bets):
    if len(bets) == 0:
        return {
            "bets": 0,
            "profit": 0.0,
            "roi": 0.0,
            "avg_odds": 0.0,
            "avg_line": 0.0,
            "max_drawdown": 0.0,
        }

    ordered = bets.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"], errors="coerce")
    ordered = ordered.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    cumulative_profit = ordered["profit"].astype(float).cumsum()
    running_max = cumulative_profit.cummax()
    drawdown = running_max - cumulative_profit

    return {
        "bets": int(len(ordered)),
        "profit": float(ordered["profit"].sum()),
        "roi": float(ordered["profit"].mean()),
        "avg_odds": float(ordered["away_ah_odds"].mean()),
        "avg_line": float(ordered["ah_line"].mean()),
        "max_drawdown": float(drawdown.max()),
    }


def dataframe_to_html_table(dataframe, columns=None, max_rows=None):
    if dataframe is None or len(dataframe) == 0:
        return "<p class='empty'>Sem dados ainda.</p>"

    display = dataframe.copy()

    if columns is not None:
        existing_columns = []

        for column in columns:
            if column in display.columns:
                existing_columns.append(column)

        display = display[existing_columns].copy()

    if max_rows is not None:
        display = display.head(max_rows).copy()

    return display.to_html(index=False, classes="data-table", border=0)


def build_by_line_table(bets):
    if len(bets) == 0:
        return pd.DataFrame()

    grouped = bets.groupby("ah_line").agg(
        bets=("profit", "count"),
        profit=("profit", "sum"),
        roi=("profit", "mean"),
        avg_odds=("away_ah_odds", "mean"),
    ).reset_index()

    grouped = grouped.sort_values("ah_line").copy()

    grouped["profit"] = grouped["profit"].round(3)
    grouped["roi"] = grouped["roi"].apply(format_percent)
    grouped["avg_odds"] = grouped["avg_odds"].round(3)

    return grouped


def build_by_year_table(bets):
    if len(bets) == 0:
        return pd.DataFrame()

    grouped = bets.groupby("season_end_year").agg(
        bets=("profit", "count"),
        profit=("profit", "sum"),
        roi=("profit", "mean"),
        avg_line=("ah_line", "mean"),
        avg_odds=("away_ah_odds", "mean"),
    ).reset_index()

    grouped = grouped.sort_values("season_end_year").copy()

    grouped["profit"] = grouped["profit"].round(3)
    grouped["roi"] = grouped["roi"].apply(format_percent)
    grouped["avg_line"] = grouped["avg_line"].round(3)
    grouped["avg_odds"] = grouped["avg_odds"].round(3)

    return grouped


def build_new_team_table(league_code, bets):
    if len(bets) == 0:
        return pd.DataFrame()

    matches_path = get_league_matches_path(league_code)

    if not matches_path.exists():
        return pd.DataFrame()

    matches = pd.read_csv(matches_path, low_memory=False)

    teams_by_season = {}

    for season, group in matches.groupby("season_end_year"):
        teams = set(group["HomeTeam"].dropna().unique().tolist())
        teams.update(group["AwayTeam"].dropna().unique().tolist())
        teams_by_season[int(season)] = teams

    rows = []

    for _, row in bets.iterrows():
        season = int(row["season_end_year"])
        away_team = row["AwayTeam"]
        previous_season = season - 1

        if previous_season in teams_by_season:
            away_is_new = away_team not in teams_by_season[previous_season]
        else:
            away_is_new = False

        new_row = row.to_dict()
        new_row["away_is_new_to_league"] = away_is_new
        rows.append(new_row)

    data = pd.DataFrame(rows)

    grouped = data.groupby("away_is_new_to_league").agg(
        bets=("profit", "count"),
        profit=("profit", "sum"),
        roi=("profit", "mean"),
        avg_line=("ah_line", "mean"),
        avg_odds=("away_ah_odds", "mean"),
    ).reset_index()

    grouped["away_is_new_to_league"] = grouped["away_is_new_to_league"].map({
        False: "No",
        True: "Yes",
    })

    grouped["profit"] = grouped["profit"].round(3)
    grouped["roi"] = grouped["roi"].apply(format_percent)
    grouped["avg_line"] = grouped["avg_line"].round(3)
    grouped["avg_odds"] = grouped["avg_odds"].round(3)

    return grouped


def build_recent_bets_table(bets):
    if len(bets) == 0:
        return pd.DataFrame()

    display = bets.copy()
    display["Date"] = pd.to_datetime(display["Date"], errors="coerce")
    display = display.sort_values(["Date", "HomeTeam", "AwayTeam"], ascending=[False, True, True])

    columns = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "ah_line",
        "away_handicap",
        "away_ah_odds",
        "FTHG",
        "FTAG",
        "profit",
        "selected_threshold",
    ]

    display = display[columns].copy()

    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    display["ah_line"] = display["ah_line"].round(2)
    display["away_handicap"] = display["away_handicap"].round(2)
    display["away_ah_odds"] = display["away_ah_odds"].round(3)
    display["profit"] = display["profit"].round(3)

    return display


def build_variant_section(variant_name, variant_dir, league_code):
    by_year_path = variant_dir / "baseline" / "nested_baseline_by_year.csv"
    bets_path = variant_dir / "baseline" / "nested_baseline_bets.csv"

    by_year = read_csv_if_exists(by_year_path)
    bets = read_csv_if_exists(bets_path)

    overall = calculate_overall(bets)

    by_year_table = build_by_year_table(bets)
    by_line_table = build_by_line_table(bets)
    new_team_table = build_new_team_table(league_code, bets)
    recent_bets = build_recent_bets_table(bets)

    html = f"""
    <section class="panel">
        <div class="panel-header">
            <div>
                <h2>{variant_name.upper()} Asian Handicap</h2>
                <p>Nested baseline: Away AH when home team is a big favourite.</p>
            </div>
            <span class="tag">{variant_name}</span>
        </div>

        <div class="cards">
            <div class="card">
                <div class="label">Bets</div>
                <div class="value">{overall["bets"]}</div>
            </div>
            <div class="card">
                <div class="label">Profit</div>
                <div class="value">{format_float(overall["profit"], 3)}</div>
            </div>
            <div class="card">
                <div class="label">ROI</div>
                <div class="value">{format_percent(overall["roi"])}</div>
            </div>
            <div class="card">
                <div class="label">Avg odds</div>
                <div class="value">{format_float(overall["avg_odds"], 3)}</div>
            </div>
            <div class="card">
                <div class="label">Avg line</div>
                <div class="value">{format_float(overall["avg_line"], 3)}</div>
            </div>
            <div class="card">
                <div class="label">Max drawdown</div>
                <div class="value">{format_float(overall["max_drawdown"], 3)}</div>
            </div>
        </div>

        <h3>By test year</h3>
        {dataframe_to_html_table(by_year_table)}

        <h3>By AH line</h3>
        {dataframe_to_html_table(by_line_table)}

        <h3>Away team new to league</h3>
        {dataframe_to_html_table(new_team_table)}

        <h3>Recent historical bets</h3>
        {dataframe_to_html_table(recent_bets, max_rows=30)}
    </section>
    """

    return html


def build_upcoming_section(upcoming_path):
    best_path = upcoming_path.parent / "upcoming_best_candidates.csv"

    if best_path.exists():
        upcoming = read_csv_if_exists(best_path)
    else:
        upcoming = read_csv_if_exists(upcoming_path)

    if len(upcoming) == 0:
        upcoming_html = """
        <div class="empty-box">
            <h3>No upcoming odds yet</h3>
            <p>
                The infrastructure is ready. When Premier League Asian Handicap odds become available,
                candidates can be written to <code>paper_tracking/upcoming_candidates.csv</code>.
            </p>
        </div>
        """
    else:
        upcoming_html = dataframe_to_html_table(upcoming)

    return f"""
    <section class="panel">
        <div class="panel-header">
            <div>
                <h2>Upcoming Candidates</h2>
                <p>Prepared for live/paper trading once future Asian Handicap odds are available.</p>
            </div>
            <span class="tag waiting">waiting for odds</span>
        </div>
        {upcoming_html}
    </section>
    """


def build_paper_section(ledger_path):
    ledger = read_csv_if_exists(ledger_path)

    if len(ledger) == 0:
        ledger_html = """
        <div class="empty-box">
            <h3>No paper bets yet</h3>
            <p>
                Paper ledger exists and is ready. Future candidates can be manually or automatically
                added before settlement.
            </p>
        </div>
        """
    else:
        ledger_html = dataframe_to_html_table(ledger)

    return f"""
    <section class="panel">
        <div class="panel-header">
            <div>
                <h2>Paper Ledger</h2>
                <p>1 unit paper tracking for the selected strategy.</p>
            </div>
            <span class="tag paper">paper only</span>
        </div>
        {ledger_html}
    </section>
    """


def build_html(league_code, market_output_dir, ledger_path, upcoming_path):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    main_section = build_variant_section(
        "main",
        market_output_dir / "main",
        league_code
    )

    closing_section = build_variant_section(
        "closing",
        market_output_dir / "closing",
        league_code
    )

    upcoming_section = build_upcoming_section(upcoming_path)
    paper_section = build_paper_section(ledger_path)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{league_code} AH Big Home Favourite Dashboard</title>
    <style>
        :root {{
            --bg: #0f172a;
            --panel: #111827;
            --panel-soft: #1f2937;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --accent: #38bdf8;
            --good: #22c55e;
            --warn: #f59e0b;
            --bad: #ef4444;
            --border: #334155;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #020617, #0f172a);
            color: var(--text);
        }}

        header {{
            padding: 36px 42px 22px 42px;
            border-bottom: 1px solid var(--border);
        }}

        header h1 {{
            margin: 0;
            font-size: 34px;
        }}

        header p {{
            margin: 8px 0 0 0;
            color: var(--muted);
            max-width: 900px;
            line-height: 1.5;
        }}

        .strategy-box {{
            margin-top: 18px;
            padding: 16px;
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 14px;
            max-width: 1100px;
        }}

        .strategy-box strong {{
            color: var(--accent);
        }}

        main {{
            padding: 28px 42px 60px 42px;
        }}

        .panel {{
            background: rgba(17, 24, 39, 0.92);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 28px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.22);
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            gap: 20px;
            align-items: flex-start;
            margin-bottom: 20px;
        }}

        h2 {{
            margin: 0;
            font-size: 24px;
        }}

        h3 {{
            margin-top: 28px;
            margin-bottom: 10px;
            font-size: 18px;
        }}

        .panel p {{
            margin: 6px 0 0 0;
            color: var(--muted);
        }}

        .tag {{
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.14);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #bbf7d0;
            font-size: 13px;
            white-space: nowrap;
        }}

        .tag.waiting {{
            background: rgba(245, 158, 11, 0.14);
            border-color: rgba(245, 158, 11, 0.4);
            color: #fde68a;
        }}

        .tag.paper {{
            background: rgba(56, 189, 248, 0.14);
            border-color: rgba(56, 189, 248, 0.4);
            color: #bae6fd;
        }}

        .cards {{
            display: grid;
            grid-template-columns: repeat(6, minmax(120px, 1fr));
            gap: 14px;
            margin: 18px 0 24px 0;
        }}

        .card {{
            background: var(--panel-soft);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px;
        }}

        .label {{
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 6px;
        }}

        .value {{
            font-size: 22px;
            font-weight: bold;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            border-radius: 12px;
            margin-top: 8px;
            font-size: 14px;
        }}

        .data-table th {{
            background: #0b1220;
            color: #cbd5e1;
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid var(--border);
        }}

        .data-table td {{
            padding: 9px 10px;
            border-bottom: 1px solid rgba(51, 65, 85, 0.65);
            color: #e5e7eb;
        }}

        .data-table tr:hover td {{
            background: rgba(56, 189, 248, 0.06);
        }}

        .empty, .empty-box p {{
            color: var(--muted);
        }}

        .empty-box {{
            padding: 18px;
            background: rgba(31, 41, 55, 0.85);
            border: 1px dashed var(--border);
            border-radius: 14px;
        }}

        code {{
            color: #bae6fd;
        }}

        footer {{
            color: var(--muted);
            font-size: 13px;
            padding: 0 42px 30px 42px;
        }}

        @media (max-width: 1100px) {{
            .cards {{
                grid-template-columns: repeat(3, minmax(120px, 1fr));
            }}
        }}

        @media (max-width: 700px) {{
            header, main, footer {{
                padding-left: 18px;
                padding-right: 18px;
            }}

            .cards {{
                grid-template-columns: repeat(2, minmax(120px, 1fr));
            }}

            .panel-header {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>{league_code} Asian Handicap Dashboard</h1>
        <p>
            Baseline strategy dashboard for Away Asian Handicap bets against big home favourites.
            This is a paper-trading candidate, not a real-money recommendation.
        </p>

        <div class="strategy-box">
            <strong>Live candidate rule:</strong>
            Premier League only · Away AH · home handicap line AHh ≤ -1.25 · 1 unit paper stake.
            Upcoming candidates will appear once future Asian Handicap odds are available.
        </div>
    </header>

    <main>
        {upcoming_section}
        {paper_section}
        {main_section}
        {closing_section}
    </main>

    <footer>
        Generated at {generated_at}. Market: {MARKET_NAME}. League: {league_code}.
    </footer>
</body>
</html>
"""

    return html


def update_global_dashboard(league_code, market_dashboard_path):
    global_dir = get_global_dashboard_dir()
    global_dir.mkdir(parents=True, exist_ok=True)

    index_path = global_dir / "index.html"

    relative_path = Path("..") / league_code / MARKET_NAME / "dashboard" / "index.html"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FootballMLV2 Dashboard</title>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #020617;
            color: #e5e7eb;
            padding: 42px;
        }}

        h1 {{
            margin-top: 0;
        }}

        .card {{
            display: block;
            max-width: 700px;
            padding: 22px;
            background: #111827;
            border: 1px solid #334155;
            border-radius: 16px;
            color: #e5e7eb;
            text-decoration: none;
            margin-top: 20px;
        }}

        .card:hover {{
            border-color: #38bdf8;
        }}

        .muted {{
            color: #9ca3af;
        }}
    </style>
</head>
<body>
    <h1>FootballMLV2 Dashboard</h1>
    <p class="muted">Available strategy dashboards.</p>

    <a class="card" href="{relative_path}">
        <h2>{league_code} Asian Handicap Big Home Favourite Away</h2>
        <p class="muted">Away AH against big home favourites. Paper-trading candidate.</p>
    </a>
</body>
</html>
"""

    index_path.write_text(html, encoding="utf-8")

    return index_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    args = parser.parse_args()

    league_code = args.league.upper()

    market_output_dir = get_market_output_dir(league_code, MARKET_NAME)
    dashboard_dir = market_output_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    ledger_path, upcoming_path = ensure_paper_files(market_output_dir)

    html = build_html(
        league_code=league_code,
        market_output_dir=market_output_dir,
        ledger_path=ledger_path,
        upcoming_path=upcoming_path
    )

    dashboard_path = dashboard_dir / "index.html"
    dashboard_path.write_text(html, encoding="utf-8")

    global_dashboard_path = update_global_dashboard(
        league_code=league_code,
        market_dashboard_path=dashboard_path
    )

    print("")
    print("Dashboard criado:")
    print(dashboard_path)
    print("")
    print("Global dashboard:")
    print(global_dashboard_path)
    print("")
    print("Paper ledger:")
    print(ledger_path)
    print("")
    print("Upcoming candidates:")
    print(upcoming_path)


if __name__ == "__main__":
    main()
