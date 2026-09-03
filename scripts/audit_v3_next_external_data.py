from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope/super_1x2_football_data_full_scope_v3_exact_research_v1.csv"
DEFAULT_OUT = ROOT / "outputs/reports/v3_next_external_data"

CATEGORIES = {
    "market_1x2": [r"x1x2", r"odds", r"overround"],
    "clubelo_internal_elo": [r"clubelo", r"internal_elo", r"(^|_)elo(_|$)"],
    "transfermarkt_squad_value": [r"^tm_", r"valuation", r"squad", r"transfer", r"churn", r"staleness"],
    "understat_xg": [r"understat", r"(^|_)xg(_|$)", r"expected_goals"],
    "form_schedule": [r"form", r"days_since", r"matches_last", r"rest", r"congest", r"venue"],
    "travel_weather": [r"travel", r"distance", r"weather", r"temperature", r"wind", r"rain"],
    "availability_lineup": [r"injury", r"suspension", r"lineup", r"player", r"availability", r"minutes"],
    "bookmaker_disagreement": [r"b365", r"pinnacle", r"(^|_)ps[had]", r"max[had]", r"avg[had]", r"dispersion", r"disagreement"],
}


def category_for(column: str) -> list[str]:
    matches = []
    for category, patterns in CATEGORIES.items():
        if any(re.search(pattern, column, flags=re.IGNORECASE) for pattern in patterns):
            matches.append(category)
    return matches or ["other"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit timing-safe external data already present in the V3 exact matrix.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input, low_memory=False)
    rows = []
    for column in frame.columns:
        for category in category_for(column):
            series = frame[column]
            numeric = pd.to_numeric(series, errors="coerce")
            rows.append(
                {
                    "category": category,
                    "feature": column,
                    "rows": len(frame),
                    "non_null_rate": float(series.notna().mean()),
                    "numeric_non_null_rate": float(numeric.notna().mean()),
                    "unique_values": int(series.nunique(dropna=True)),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("category", as_index=False)
        .agg(
            feature_count=("feature", "nunique"),
            median_non_null_rate=("non_null_rate", "median"),
            max_non_null_rate=("non_null_rate", "max"),
        )
        .sort_values(["feature_count", "median_non_null_rate"], ascending=[False, False])
    )
    detail.to_csv(args.out / "v3_next_external_feature_detail.csv", index=False)
    summary.to_csv(args.out / "v3_next_external_feature_summary.csv", index=False)

    raw_schema_rows = []
    for league in ["E0", "SP1", "D1", "I1", "F1", "B1", "G1", "N1", "P1", "SC0", "T1"]:
        season_dir = ROOT / f"data/raw/{league}/seasons"
        files = sorted(season_dir.glob("*.csv")) if season_dir.exists() else []
        for path in files[-2:]:
            try:
                sample = pd.read_csv(path, nrows=5)
                columns = set(map(str, sample.columns))
                raw_schema_rows.append(
                    {
                        "league": league,
                        "path": str(path.relative_to(ROOT)),
                        "has_b365_1x2": {"B365H", "B365D", "B365A"}.issubset(columns),
                        "has_pinnacle_1x2": {"PSH", "PSD", "PSA"}.issubset(columns),
                        "has_avg_1x2": {"AvgH", "AvgD", "AvgA"}.issubset(columns),
                        "has_max_1x2": {"MaxH", "MaxD", "MaxA"}.issubset(columns),
                        "has_open_ah": any(c in columns for c in ["AHh", "B365AHH", "B365AHA", "AvgAHH", "AvgAHA"]),
                        "has_close_ah": any(c in columns for c in ["AHCh", "B365CAHH", "B365CAHA", "AvgCAHH", "AvgCAHA"]),
                        "column_count": len(columns),
                    }
                )
            except Exception as exc:
                raw_schema_rows.append({"league": league, "path": str(path.relative_to(ROOT)), "error": f"{type(exc).__name__}: {exc}"})
    raw_schema = pd.DataFrame(raw_schema_rows)
    raw_schema.to_csv(args.out / "v3_next_raw_market_schema_audit.csv", index=False)

    priority = [
        ("1", "Opening/pre-match bookmaker disagreement", "Build no-vig probabilities per bookmaker and dispersion/consensus features. Keep closing prices diagnostic-only."),
        ("2", "Player availability and expected lineup strength", "Use only timestamped pre-match availability; aggregate expected minutes and market value/Elo replacement cost."),
        ("3", "Rolling xG/xGA and shot quality", "Use strictly prior-match team rolling windows, opponent adjusted where possible, with source-date audits."),
        ("4", "Dynamic team-strength state", "Tune Elo update parameters only inside nested validation; consider attack/defence states rather than one scalar rating."),
        ("5", "Schedule, travel and weather interactions", "Retain only when coverage is broad and incremental predictive gain survives league/season exclusions."),
    ]
    report = [
        "# V3 Next External Data Audit",
        "",
        "This audit inventories data already available locally. It does not fetch sources, change the frozen V3, or claim an edge.",
        "",
        "## Existing feature categories",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Recommended priority",
    ]
    report += [f"{rank}. **{name}** — {detail_text}" for rank, name, detail_text in priority]
    report += [
        "",
        "## Guardrails",
        "- Do not use final scores, closing odds, confirmed lineups published after the prediction timestamp, or future valuations as features.",
        "- Add one source block at a time and require same-row comparison against the market-anchored challenger.",
        "- New data is useful only if it improves out-of-sample log loss/Brier before any value scan.",
    ]
    (args.out / "v3_next_external_data_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("v3_next_external_data_audit_ready")


if __name__ == "__main__":
    main()
