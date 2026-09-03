from __future__ import annotations

import hashlib
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/ah_e0_frozen_candidate.yaml"
PAPER_ROOT = ROOT / "outputs/paper_trading"
PAPER_DIR = PAPER_ROOT / "ah_e0"
SNAPSHOT_DIR = PAPER_DIR / "snapshots"
HTML_DIR = PAPER_DIR / "html"
LOG_DIR = PAPER_DIR / "logs"
LEDGER_PATH = PAPER_DIR / "ah_e0_paper_ledger.csv"
LATEST_CANDIDATE_PICKS = PAPER_DIR / "ah_e0_latest_candidate_picks.csv"
LATEST_SKIPPED = PAPER_DIR / "ah_e0_latest_skipped_rows.csv"
LATEST_WARNINGS = PAPER_DIR / "ah_e0_latest_warnings.csv"
LATEST_AH_COLUMNS = PAPER_DIR / "ah_e0_latest_ah_columns_available.csv"
LATEST_SNAPSHOT_POINTER = SNAPSHOT_DIR / "latest_manifest_path.txt"
HTML_REPORT = HTML_DIR / "ah_e0_paper_latest.html"
PIPELINE_REPORT = PAPER_DIR / "ah_e0_paper_pipeline_report.md"
COMBINED_INDEX = PAPER_ROOT / "index.html"

LEAGUE = "E0"
MARKET = "Asian Handicap"
SELECTED_SIDE = "away"
RULE_NAME = "home_ah_line_le_-1.25_select_away"
HOME_AH_LINE_MAX = -1.25
STAKE_UNITS = 1.0
RESEARCH_ONLY = "research_only"
OPEN_STATUS = "OPEN"
SETTLED_STATUSES = {
    "SETTLED_WIN",
    "SETTLED_HALF_WIN",
    "SETTLED_PUSH",
    "SETTLED_HALF_LOSS",
    "SETTLED_LOSS",
}

LEDGER_COLUMNS = [
    "paper_bet_id",
    "created_at_utc",
    "run_id",
    "source_snapshot_id",
    "match_date",
    "league",
    "season_start_year",
    "home_team",
    "away_team",
    "market",
    "selected_side",
    "selected_handicap_line",
    "selected_odds",
    "rule_name",
    "stake_units",
    "status",
    "home_goals",
    "away_goals",
    "settled_at_utc",
    "profit_units",
    "closing_notes",
]

AH_COLUMN_CANDIDATES = [
    "AHh",
    "AvgAHH",
    "AvgAHA",
    "MaxAHH",
    "MaxAHA",
    "B365AHH",
    "B365AHA",
    "BFEAHH",
    "BFEAHA",
    "BbAHh",
    "BbAvAHH",
    "BbAvAHA",
    "AHCh",
    "AvgCAHH",
    "AvgCAHA",
    "B365CAHH",
    "B365CAHA",
]


def ensure_dirs() -> None:
    for path in [PAPER_DIR, SNAPSHOT_DIR, HTML_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def stable_decimal(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    return f"{float(value):.6f}"


def parse_date_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    first = pd.to_datetime(text, errors="coerce", dayfirst=True)
    second = pd.to_datetime(text, errors="coerce", dayfirst=False)
    return first.fillna(second)


def season_from_path(path: Path) -> int | None:
    text = path.name
    m = re.search(r"_(\d{4})_(\d{4})", text)
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d{2})(\d{2})", text)
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy < 50 else 1900 + yy
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def load_ledger(path: Path = LEDGER_PATH) -> pd.DataFrame:
    if not path.exists():
        return empty_ledger()
    ledger = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in LEDGER_COLUMNS:
        if col not in ledger.columns:
            ledger[col] = ""
    return ledger[LEDGER_COLUMNS].copy()


def write_ledger(ledger: pd.DataFrame, path: Path = LEDGER_PATH) -> None:
    ensure_dirs()
    out = ledger.copy()
    for col in LEDGER_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out[LEDGER_COLUMNS].to_csv(path, index=False)


def deterministic_paper_bet_id(
    match_date: object,
    league: object,
    home_team: object,
    away_team: object,
    market: object,
    selected_side: object,
    selected_handicap_line: object,
    selected_odds: object,
    rule_name: object,
) -> str:
    key = "|".join(
        [
            str(match_date)[:10],
            str(league),
            str(home_team),
            str(away_team),
            str(market),
            str(selected_side).lower(),
            stable_decimal(selected_handicap_line),
            stable_decimal(selected_odds),
            str(rule_name),
        ]
    )
    return "ah_e0_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def latest_e0_raw_paths() -> list[Path]:
    root = ROOT / "data/raw/E0/seasons"
    paths = sorted(root.glob("E0_*.csv"))
    with_seasons = [(p, season_from_path(p)) for p in paths]
    with_seasons = [(p, s) for p, s in with_seasons if s is not None]
    if not with_seasons:
        return []
    latest = max(s for _p, s in with_seasons)
    return [p for p, s in with_seasons if s == latest]


def read_latest_e0_raw() -> tuple[pd.DataFrame, list[Path], pd.DataFrame]:
    frames = []
    paths = latest_e0_raw_paths()
    column_rows = []
    for path in paths:
        raw = pd.read_csv(path, low_memory=False)
        raw = raw[raw.get("Div", LEAGUE).astype(str).eq(LEAGUE)].copy() if "Div" in raw.columns else raw.copy()
        raw["source_file"] = str(path.relative_to(ROOT))
        raw["season_start_year"] = season_from_path(path)
        raw["match_date"] = parse_date_series(raw["Date"]) if "Date" in raw.columns else pd.NaT
        frames.append(raw)
        for col in AH_COLUMN_CANDIDATES:
            column_rows.append(
                {
                    "source_file": str(path.relative_to(ROOT)),
                    "column": col,
                    "available": col in raw.columns,
                    "non_null_rows": int(raw[col].notna().sum()) if col in raw.columns else 0,
                    "rows": int(len(raw)),
                }
            )
    return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(), paths, pd.DataFrame(column_rows))


def build_snapshot_manifest(run_id: str, raw: pd.DataFrame, paths: list[Path], warnings: list[str]) -> dict[str, object]:
    ensure_dirs()
    files = []
    for path in paths:
        rows = raw[raw["source_file"].eq(str(path.relative_to(ROOT)))].copy() if not raw.empty else pd.DataFrame()
        dates = pd.to_datetime(rows.get("match_date", pd.Series(dtype=str)), errors="coerce")
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_count": int(len(pd.read_csv(path, low_memory=False))),
                "normalized_row_count": int(len(rows)),
                "min_date": dates.min().date().isoformat() if dates.notna().any() else "",
                "max_date": dates.max().date().isoformat() if dates.notna().any() else "",
                "leagues_detected": [LEAGUE],
            }
        )
    manifest = {
        "run_id": run_id,
        "timestamp_utc": iso_utc(),
        "input_files": files,
        "file_count": len(files),
        "row_count": int(len(raw)),
        "leagues_detected": [LEAGUE] if not raw.empty else [],
        "warnings": warnings,
    }
    path = SNAPSHOT_DIR / f"{run_id}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LATEST_SNAPSHOT_POINTER.write_text(str(path.relative_to(ROOT)) + "\n", encoding="utf-8")
    return manifest


def ah_columns_available(columns: pd.DataFrame) -> bool:
    if columns.empty:
        return False
    available = set(columns.loc[columns["available"], "column"].astype(str))
    return {"AHh", "AvgAHA"}.issubset(available)


def normalize_prediction_rows(raw: pd.DataFrame, run_id: str, snapshot_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    out = pd.DataFrame(
        {
            "run_id": run_id,
            "source_snapshot_id": snapshot_id,
            "match_date": pd.to_datetime(raw["match_date"], errors="coerce").dt.date.astype(str),
            "league": LEAGUE,
            "season_start_year": pd.to_numeric(raw["season_start_year"], errors="coerce").astype("Int64"),
            "home_team": raw.get("HomeTeam", pd.Series("", index=raw.index)).astype(str),
            "away_team": raw.get("AwayTeam", pd.Series("", index=raw.index)).astype(str),
            "market": MARKET,
            "selected_side": SELECTED_SIDE,
            "home_ah_line": pd.to_numeric(raw.get("AHh"), errors="coerce"),
            "selected_handicap_line": -pd.to_numeric(raw.get("AHh"), errors="coerce"),
            "selected_odds": pd.to_numeric(raw.get("AvgAHA"), errors="coerce"),
            "rule_name": RULE_NAME,
            "stake_units": STAKE_UNITS,
            "home_goals": pd.to_numeric(raw.get("FTHG"), errors="coerce"),
            "away_goals": pd.to_numeric(raw.get("FTAG"), errors="coerce"),
            "result": raw.get("FTR", pd.Series("", index=raw.index)).fillna("").astype(str),
            "source_file": raw.get("source_file", pd.Series("", index=raw.index)).astype(str),
        }
    )
    odds_ok = out["selected_odds"].gt(1.0) & out["selected_handicap_line"].notna()
    rule_ok = out["home_ah_line"].le(HOME_AH_LINE_MAX)
    unresolved = out["result"].str.strip().eq("") | out[["home_goals", "away_goals"]].isna().any(axis=1)
    picks = out[odds_ok & rule_ok & unresolved].copy()
    skipped = out[~odds_ok | (rule_ok & ~unresolved)].copy()
    if not picks.empty:
        picks["paper_bet_id"] = picks.apply(
            lambda r: deterministic_paper_bet_id(
                r["match_date"],
                r["league"],
                r["home_team"],
                r["away_team"],
                r["market"],
                r["selected_side"],
                r["selected_handicap_line"],
                r["selected_odds"],
                r["rule_name"],
            ),
            axis=1,
        )
    if not skipped.empty:
        skipped["skip_status"] = np.where(~odds_ok.reindex(skipped.index).fillna(False), "SKIPPED_NO_AH_ODDS", "SKIPPED_RESULT_ALREADY_AVAILABLE")
        skipped["skip_reason"] = np.where(skipped["skip_status"].eq("SKIPPED_NO_AH_ODDS"), "missing/invalid AH line or away AH odds", "final score already available; not a new paper pick")
    return picks, skipped


def append_new_picks_to_ledger(ledger: pd.DataFrame, picks: pd.DataFrame, created_at_utc: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    created_at_utc = created_at_utc or iso_utc()
    if picks.empty:
        return ledger.copy(), pd.DataFrame(columns=LEDGER_COLUMNS)
    existing = set(ledger.get("paper_bet_id", pd.Series(dtype=str)).dropna().astype(str))
    rows = []
    for r in picks.itertuples(index=False):
        bet_id = str(getattr(r, "paper_bet_id"))
        if bet_id in existing:
            continue
        rows.append(
            {
                "paper_bet_id": bet_id,
                "created_at_utc": created_at_utc,
                "run_id": getattr(r, "run_id"),
                "source_snapshot_id": getattr(r, "source_snapshot_id"),
                "match_date": getattr(r, "match_date"),
                "league": getattr(r, "league"),
                "season_start_year": getattr(r, "season_start_year"),
                "home_team": getattr(r, "home_team"),
                "away_team": getattr(r, "away_team"),
                "market": MARKET,
                "selected_side": SELECTED_SIDE,
                "selected_handicap_line": stable_decimal(getattr(r, "selected_handicap_line")),
                "selected_odds": stable_decimal(getattr(r, "selected_odds")),
                "rule_name": RULE_NAME,
                "stake_units": stable_decimal(STAKE_UNITS),
                "status": OPEN_STATUS,
                "home_goals": "",
                "away_goals": "",
                "settled_at_utc": "",
                "profit_units": "",
                "closing_notes": "paper_only; research_only; no_confirmed_edge",
            }
        )
        existing.add(bet_id)
    new_rows = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    out = pd.concat([ledger, new_rows], ignore_index=True, sort=False) if rows else ledger.copy()
    return out[LEDGER_COLUMNS].copy(), new_rows


def split_handicap(handicap: float) -> tuple[float, ...]:
    value = float(handicap)
    scaled = value * 4.0
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-8:
        return (value,)
    if rounded % 2 == 0:
        return (value,)
    lower = math.floor(value * 2.0) / 2.0
    upper = math.ceil(value * 2.0) / 2.0
    return (lower, upper)


def single_part_profit(adjusted_margin: float, odds: float) -> float:
    if adjusted_margin > 0:
        return float(odds) - 1.0
    if adjusted_margin == 0:
        return 0.0
    return -1.0


def status_from_profit_parts(parts: list[float]) -> str:
    if all(p > 0 for p in parts):
        return "SETTLED_WIN"
    if any(p > 0 for p in parts) and any(p == 0 for p in parts) and not any(p < 0 for p in parts):
        return "SETTLED_HALF_WIN"
    if all(p == 0 for p in parts):
        return "SETTLED_PUSH"
    if any(p < 0 for p in parts) and any(p == 0 for p in parts) and not any(p > 0 for p in parts):
        return "SETTLED_HALF_LOSS"
    if all(p < 0 for p in parts):
        return "SETTLED_LOSS"
    return "BLOCKED_DATA_QUALITY"


def settle_away_handicap(home_goals: float, away_goals: float, selected_handicap_line: float, selected_odds: float) -> tuple[str, float]:
    if any(pd.isna(v) for v in [home_goals, away_goals, selected_handicap_line, selected_odds]) or float(selected_odds) <= 1.0:
        return "BLOCKED_DATA_QUALITY", np.nan
    away_margin = float(away_goals) - float(home_goals)
    parts = split_handicap(float(selected_handicap_line))
    profits = [single_part_profit(away_margin + part, float(selected_odds)) for part in parts]
    return status_from_profit_parts(profits), float(np.mean(profits))


def result_lookup(raw: pd.DataFrame) -> dict[tuple[str, str, str, str], tuple[float, float]]:
    lookup = {}
    if raw.empty:
        return lookup
    for r in raw.itertuples(index=False):
        home_goals = getattr(r, "FTHG", np.nan)
        away_goals = getattr(r, "FTAG", np.nan)
        if pd.isna(home_goals) or pd.isna(away_goals):
            continue
        date = pd.to_datetime(getattr(r, "match_date", pd.NaT), errors="coerce")
        key = (date.date().isoformat() if pd.notna(date) else "", LEAGUE, str(getattr(r, "HomeTeam", "")), str(getattr(r, "AwayTeam", "")))
        lookup[key] = (float(home_goals), float(away_goals))
    return lookup


def settle_open_ledger(ledger: pd.DataFrame, lookup: dict[tuple[str, str, str, str], tuple[float, float]], settled_at_utc: str | None = None) -> pd.DataFrame:
    if ledger.empty:
        return ledger.copy()
    settled_at_utc = settled_at_utc or iso_utc()
    out = ledger.copy()
    for idx, row in out[out["status"].eq(OPEN_STATUS)].iterrows():
        key = (str(row["match_date"])[:10], str(row["league"]), str(row["home_team"]), str(row["away_team"]))
        if key not in lookup:
            continue
        home_goals, away_goals = lookup[key]
        status, profit = settle_away_handicap(home_goals, away_goals, float(row["selected_handicap_line"]), float(row["selected_odds"]))
        out.at[idx, "status"] = status
        out.at[idx, "home_goals"] = stable_decimal(home_goals)
        out.at[idx, "away_goals"] = stable_decimal(away_goals)
        out.at[idx, "settled_at_utc"] = settled_at_utc
        out.at[idx, "profit_units"] = "" if pd.isna(profit) else stable_decimal(profit)
    return out[LEDGER_COLUMNS].copy()


def performance_summary(ledger: pd.DataFrame) -> dict[str, object]:
    if ledger.empty:
        return {"total_paper_bets": 0, "open_bets": 0, "settled_bets": 0, "profit_units": 0.0, "roi": 0.0}
    settled = ledger[ledger["status"].isin(SETTLED_STATUSES)].copy()
    profit = float(pd.to_numeric(settled["profit_units"], errors="coerce").fillna(0.0).sum())
    settled_bets = int(len(settled))
    return {
        "total_paper_bets": int(len(ledger)),
        "open_bets": int(ledger["status"].eq(OPEN_STATUS).sum()),
        "settled_bets": settled_bets,
        "profit_units": profit,
        "roi": profit / settled_bets if settled_bets else 0.0,
    }


def html_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df.empty:
        return "<p><em>No rows.</em></p>"
    show = df[[c for c in columns if c in df.columns]].copy() if columns else df.copy()
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in show.columns)
    body = []
    for _idx, row in show.head(300).iterrows():
        body.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row.tolist()) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html_report(ledger: pd.DataFrame, columns: pd.DataFrame, warnings: pd.DataFrame | None = None) -> None:
    ensure_dirs()
    warnings = warnings if warnings is not None else pd.DataFrame()
    summary = performance_summary(ledger)
    open_picks = ledger[ledger["status"].eq(OPEN_STATUS)].copy() if not ledger.empty else ledger
    settled = ledger[ledger["status"].isin(SETTLED_STATUSES)].copy() if not ledger.empty else ledger
    metric_html = "".join(f"<div class='metric'><div class='label'>{html.escape(k)}</div><div class='value'>{html.escape(f'{v:.4f}' if isinstance(v, float) else str(v))}</div></div>" for k, v in summary.items())
    css = "body{font-family:Arial,sans-serif;margin:24px;color:#1f2933}.warn{border:1px solid #c2410c;background:#fff7ed;padding:12px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px}.metric{border:1px solid #d9e2ec;padding:10px}.label{font-size:12px;color:#52606d}.value{font-size:20px;font-weight:700}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d9e2ec;padding:6px;text-align:left}th{background:#f5f7fa}code{background:#f5f7fa;padding:2px 4px}"
    HTML_REPORT.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Frozen AH E0 Paper Trading Report</title><style>{css}</style></head><body>
<h1>Frozen AH E0 Paper Trading Report</h1><p>Generated UTC: {html.escape(iso_utc())}</p>
<div class="warn"><strong>research_only.</strong> Paper trading only. No live betting, no real-money staking, and no confirmed edge claim.</div>
<h2>Frozen Rule</h2><p>E0 only, Asian Handicap, select away side when <code>AHh &lt;= -1.25</code>. Selected away handicap line is <code>-AHh</code>; odds are <code>AvgAHA</code>. Flat 1u paper stake.</p>
<h2>Performance Summary</h2><div class="grid">{metric_html}</div>
<h2>Open Picks</h2>{html_table(open_picks)}
<h2>Settled Picks</h2>{html_table(settled)}
<h2>AH Columns Available</h2>{html_table(columns)}
<h2>Warnings</h2>{html_table(warnings)}
</body></html>""",
        encoding="utf-8",
    )


def write_pipeline_report(decision: str, columns: pd.DataFrame, warnings: list[str]) -> None:
    available = columns[columns["available"]].copy() if not columns.empty else pd.DataFrame()
    lines = [
        "# AH E0 Paper Trading Pipeline",
        "",
        f"Decision: `{decision}`",
        "",
        "Frozen E0 Asian Handicap away-side paper pipeline. No model, retraining, threshold optimization, API keys, live betting, or confirmed edge claim.",
        "",
        "## How To Run",
        "`python scripts/run_ah_e0_paper_pipeline.py`",
        "",
        "## AH Columns Available",
        available.to_markdown(index=False) if not available.empty else "_No AH columns found._",
        "",
        "## Outputs",
        "- `outputs/paper_trading/ah_e0/ah_e0_latest_candidate_picks.csv`",
        "- `outputs/paper_trading/ah_e0/ah_e0_paper_ledger.csv`",
        "- `outputs/paper_trading/ah_e0/html/ah_e0_paper_latest.html`",
        "",
        "## Limitations",
        "- Uses only local `data/raw/E0/seasons/*.csv` files.",
        "- Does not invent AH odds and does not use 1X2 odds as AH odds.",
        "- Completed matches are not added as new paper picks.",
        "- Paper-only, `research_only`, no confirmed edge.",
        "",
        "## Warnings",
    ]
    lines.extend([f"- {w}" for w in warnings] or ["- none"])
    PIPELINE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_thresholds() -> dict[str, float]:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    line = re.search(r"home_ah_line_max:\s*(-?[0-9.]+)", text)
    stake = re.search(r"stake_units:\s*([0-9.]+)", text)
    if not line or not stake:
        raise ValueError("missing AH config thresholds")
    return {"home_ah_line_max": float(line.group(1)), "stake_units": float(stake.group(1))}


def read_latest_manifest_timestamp(path: Path) -> str:
    pointer = path / "snapshots/latest_manifest_path.txt"
    if not pointer.exists():
        return ""
    manifest_path = ROOT / pointer.read_text(encoding="utf-8").strip()
    if not manifest_path.exists():
        return ""
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("timestamp_utc", "")
    except Exception:
        return ""
