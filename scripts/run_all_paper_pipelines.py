from __future__ import annotations

import html
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if __package__ is None or __package__ == "":
    sys.path.append(str(ROOT))

from src.paper_trading.ah_e0_pipeline import COMBINED_INDEX, PAPER_ROOT, read_latest_manifest_timestamp  # noqa: E402
from src.paper_trading.ah_e0_pipeline import performance_summary as ah_summary  # noqa: E402
from src.paper_trading.ah_e0_pipeline import load_ledger as load_ah_ledger  # noqa: E402
from src.paper_trading.ah_e0_pipeline import LATEST_AH_COLUMNS, LATEST_CANDIDATE_PICKS as AH_LATEST_CANDIDATE_PICKS, LATEST_SKIPPED as AH_LATEST_SKIPPED  # noqa: E402
from src.paper_trading.v3_pipeline import HTML_REPORT as V3_HTML_REPORT  # noqa: E402
from src.paper_trading.v3_pipeline import PAPER_DIR as V3_PAPER_DIR  # noqa: E402
from src.paper_trading.v3_pipeline import performance_summary as v3_summary  # noqa: E402
from src.paper_trading.v3_pipeline import load_ledger as load_v3_ledger  # noqa: E402
from src.paper_trading.ah_e0_pipeline import HTML_REPORT as AH_HTML_REPORT  # noqa: E402
from src.paper_trading.ah_e0_pipeline import PAPER_DIR as AH_PAPER_DIR  # noqa: E402


def run_step(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / script)], check=True)


def rel(path: Path) -> str:
    return str(path.relative_to(PAPER_ROOT))


def row(name: str, report: Path, summary: dict[str, object], latest: str) -> str:
    profit = float(summary.get("profit_units", 0.0))
    roi = float(summary.get("roi", 0.0))
    return (
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td><a href='{html.escape(rel(report))}'>{html.escape(rel(report))}</a></td>"
        f"<td>{summary.get('open_bets', 0)}</td>"
        f"<td>{summary.get('settled_bets', 0)}</td>"
        f"<td>{profit:.4f}</td>"
        f"<td>{roi:.4%}</td>"
        f"<td>{html.escape(latest)}</td>"
        "</tr>"
    )


def html_table(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 80) -> str:
    if frame.empty:
        return "<p><em>No rows.</em></p>"
    show = frame.copy()
    if columns:
        show = show[[c for c in columns if c in show.columns]]
    show = show.head(max_rows)
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in show.columns)
    rows = []
    for _idx, item in show.iterrows():
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in item.tolist()) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def build_index() -> None:
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    v3 = v3_summary(load_v3_ledger())
    ah = ah_summary(load_ah_ledger())
    ah_columns = read_csv(LATEST_AH_COLUMNS)
    ah_required = ah_columns[ah_columns.get("column", pd.Series(dtype=str)).isin(["AHh", "AvgAHH", "AvgAHA"])] if not ah_columns.empty else pd.DataFrame()
    ah_candidates = read_csv(AH_LATEST_CANDIDATE_PICKS)
    ah_skipped = read_csv(AH_LATEST_SKIPPED)
    ah_skip_summary = ah_skipped["skip_status"].value_counts().rename_axis("skip_status").reset_index(name="rows") if "skip_status" in ah_skipped.columns else pd.DataFrame()
    css = "body{font-family:Arial,sans-serif;margin:24px;color:#1f2933}.warn{border:1px solid #c2410c;background:#fff7ed;padding:12px;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0 24px}th,td{border:1px solid #d9e2ec;padding:8px;text-align:left}th{background:#f5f7fa}h2{margin-top:28px}code{background:#f5f7fa;padding:2px 4px}"
    rows = [
        row("Frozen V3 1X2", V3_HTML_REPORT, v3, read_latest_manifest_timestamp(V3_PAPER_DIR)),
        row("Frozen AH E0", AH_HTML_REPORT, ah, read_latest_manifest_timestamp(AH_PAPER_DIR)),
    ]
    COMBINED_INDEX.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>Paper Trading Index</title><style>{css}</style></head><body>"
        "<h1>Paper Trading Index</h1>"
        "<div class='warn'><strong>research_only.</strong> Paper trading only. No live betting, no real-money staking, and no confirmed edge claim.</div>"
        "<h2>Strategies</h2>"
        "<table><thead><tr><th>Strategy name</th><th>Report</th><th>Open picks</th><th>Settled picks</th><th>Paper profit</th><th>ROI</th><th>Latest run timestamp</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<h2>Frozen AH E0</h2>"
        "<p>E0 Asian Handicap, away side only. Rule: <code>AHh &lt;= -1.25</code>; selected away handicap line is <code>-AHh</code>; selected odds are <code>AvgAHA</code>. Flat 1u paper stake.</p>"
        "<h3>AH Columns</h3>"
        f"{html_table(ah_required, ['source_file', 'column', 'available', 'non_null_rows', 'rows'])}"
        "<h3>AH Candidate Picks</h3>"
        f"{html_table(ah_candidates, ['match_date', 'league', 'home_team', 'away_team', 'selected_handicap_line', 'selected_odds', 'rule_name'])}"
        "<h3>AH Skipped Summary</h3>"
        f"{html_table(ah_skip_summary)}"
        "<h3>AH Recent Skipped Rows</h3>"
        f"{html_table(ah_skipped, ['match_date', 'home_team', 'away_team', 'home_ah_line', 'selected_handicap_line', 'selected_odds', 'skip_status'], max_rows=30)}"
        "</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    run_step("scripts/run_v3_paper_pipeline.py")
    run_step("scripts/run_ah_e0_paper_pipeline.py")
    build_index()
    print("all_paper_pipelines_ready_research_only")
    print(f"combined_index={COMBINED_INDEX}")


if __name__ == "__main__":
    main()
