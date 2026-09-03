from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.features.player_squad_strength import TM_WINDOW_DAYS
from src.features.player_squad_strength import load_player_squad_team_mapping
from src.features.player_squad_strength import load_transfermarkt_market_values
from src.features.player_squad_strength import normalize_name


LEAGUES = ["E0", "I1", "SP1", "D1", "F1", "P1"]
PROXY_PATH = Path("data/processed/players/transfermarkt_valuation_only_club_strength_proxy.csv")
REPORT_DIR = Path("outputs/reports")
COVERAGE_CSV = REPORT_DIR / "transfermarkt_valuation_only_proxy_coverage.csv"
COVERAGE_MD = REPORT_DIR / "transfermarkt_valuation_only_proxy_coverage.md"
LEAKAGE_MD = REPORT_DIR / "transfermarkt_valuation_only_proxy_leakage_audit.md"
READINESS_MD = REPORT_DIR / "transfermarkt_valuation_only_proxy_readiness.md"

IDENTIFIER_COLUMNS = [
    "league",
    "season_end_year",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
]
CLOSING_PREFIXES = (
    "B365C",
    "BWC",
    "IWC",
    "WHC",
    "VCC",
    "MaxC",
    "AvgC",
    "PSC",
    "AHC",
    "BFECH",
    "BFEC",
    "1XBC",
    "PCAH",
)


def load_matches() -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["league"] = league
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        if "season_end_year" not in frame.columns:
            frame["season_end_year"] = np.where(frame["Date"].dt.month >= 7, frame["Date"].dt.year + 1, frame["Date"].dt.year)
        frame["season_end_year"] = pd.to_numeric(frame["season_end_year"], errors="coerce").astype("Int64")
        frames.append(frame.dropna(subset=["Date", "HomeTeam", "AwayTeam", "season_end_year"]).copy())
    if not frames:
        return pd.DataFrame(columns=IDENTIFIER_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(["league", "Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def high_confidence_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    confidence = mapping["confidence"].fillna("").astype(str).str.casefold()
    source = mapping.get("player_data_source", pd.Series("transfermarkt", index=mapping.index)).fillna("").astype(str).str.casefold()
    return mapping[source.eq("transfermarkt") & confidence.str.startswith("high")].copy()


def feature_columns() -> list[str]:
    columns = ["home_tm_mapped_club_name", "away_tm_mapped_club_name"]
    for side in ["home", "away", "home_minus_away"]:
        for days in TM_WINDOW_DAYS:
            columns.extend(
                [
                    f"{side}_tm_value_total_{days}d",
                    f"{side}_tm_value_top11_{days}d",
                    f"{side}_tm_value_top5_{days}d",
                    f"{side}_tm_value_median_{days}d",
                    f"{side}_tm_players_count_{days}d",
                ]
            )
    return columns


def transfermarkt_features_for_group(club_values: pd.DataFrame, match_date: pd.Timestamp) -> dict:
    row = {}
    date = pd.Timestamp(match_date).normalize()
    before_match = club_values[club_values["valuation_date"] < date]
    for days in TM_WINDOW_DAYS:
        prefix = f"{days}d"
        row[f"tm_value_total_{prefix}"] = np.nan
        row[f"tm_value_top11_{prefix}"] = np.nan
        row[f"tm_value_top5_{prefix}"] = np.nan
        row[f"tm_value_median_{prefix}"] = np.nan
        row[f"tm_players_count_{prefix}"] = np.nan
        row[f"tm_latest_valuation_date_{prefix}"] = pd.NaT
        window = before_match[before_match["valuation_date"] >= date - pd.Timedelta(days=int(days))]
        if window.empty:
            continue
        latest = window.sort_values(["player_id", "valuation_date"]).drop_duplicates("player_id", keep="last")
        values = pd.to_numeric(latest["market_value_eur"], errors="coerce").dropna().sort_values(ascending=False)
        if values.empty:
            continue
        row[f"tm_value_total_{prefix}"] = float(values.sum())
        row[f"tm_value_top11_{prefix}"] = float(values.head(11).sum())
        row[f"tm_value_top5_{prefix}"] = float(values.head(5).sum())
        row[f"tm_value_median_{prefix}"] = float(values.median())
        row[f"tm_players_count_{prefix}"] = int(values.size)
        row[f"tm_latest_valuation_date_{prefix}"] = latest["valuation_date"].max()
    return row


def build_proxy_dataset(matches: pd.DataFrame, market_values: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    mapped_club = {
        (str(row.league), str(row.match_team)): str(row.player_data_club_name)
        for row in mapping.itertuples(index=False)
        if pd.notna(row.player_data_club_name) and str(row.player_data_club_name)
    }
    market_by_club = {club_key: group.copy() for club_key, group in market_values.groupby("club_key", sort=False)}
    feature_cache: dict[tuple[str, pd.Timestamp], dict] = {}

    rows = []
    for idx, match in matches[IDENTIFIER_COLUMNS].iterrows():
        base = match.to_dict()
        base["match_id"] = int(idx)
        match_date = pd.Timestamp(match["Date"]).normalize()
        for side, team_column in [("home", "HomeTeam"), ("away", "AwayTeam")]:
            club_name = mapped_club.get((str(match["league"]), str(match[team_column])), "")
            base[f"{side}_tm_mapped_club_name"] = club_name
            club_key = normalize_name(club_name)
            cache_key = (club_key, match_date)
            if club_key and cache_key not in feature_cache:
                club_values = market_by_club.get(club_key, pd.DataFrame(columns=market_values.columns))
                feature_cache[cache_key] = transfermarkt_features_for_group(club_values, match_date)
            features = feature_cache.get(cache_key, transfermarkt_features_for_group(pd.DataFrame(columns=market_values.columns), match_date))
            for column, value in features.items():
                base[f"{side}_{column}"] = value
        for days in TM_WINDOW_DAYS:
            for metric in ["tm_value_total", "tm_value_top11", "tm_value_top5", "tm_value_median", "tm_players_count"]:
                home_column = f"home_{metric}_{days}d"
                away_column = f"away_{metric}_{days}d"
                base[f"home_minus_away_{metric}_{days}d"] = (
                    pd.to_numeric(base.get(home_column), errors="coerce") - pd.to_numeric(base.get(away_column), errors="coerce")
                )
        rows.append(base)

    output = pd.DataFrame(rows)
    ordered = ["match_id"] + [column for column in IDENTIFIER_COLUMNS + feature_columns() if column in output.columns]
    latest_columns = [column for column in output.columns if column.endswith(tuple(f"tm_latest_valuation_date_{days}d" for days in TM_WINDOW_DAYS))]
    return output[ordered + latest_columns]


def coverage_summary(proxy: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping_by_league = {
        league: sorted(group.loc[group["player_data_club_name"].fillna("").astype(str).eq(""), "match_team"].dropna().astype(str).unique())
        for league, group in mapping.groupby("league")
    }
    all_mapping_by_league = {
        league: sorted(group.loc[~group["confidence"].fillna("").astype(str).str.casefold().str.startswith("high"), "match_team"].dropna().astype(str).unique())
        for league, group in load_player_squad_team_mapping().groupby("league")
    }
    for (league, season), group in proxy.groupby(["league", "season_end_year"], dropna=False):
        home_mapped = group["home_tm_mapped_club_name"].fillna("").astype(str).ne("")
        away_mapped = group["away_tm_mapped_club_name"].fillna("").astype(str).ne("")
        row = {
            "league": league,
            "season_end_year": int(season) if pd.notna(season) else pd.NA,
            "matches": len(group),
            "home_mapped": int(home_mapped.sum()),
            "away_mapped": int(away_mapped.sum()),
            "both_mapped": int((home_mapped & away_mapped).sum()),
            "unmatched_clubs": "; ".join(all_mapping_by_league.get(league, mapping_by_league.get(league, []))),
        }
        for days in TM_WINDOW_DAYS:
            home_count = pd.to_numeric(group[f"home_tm_players_count_{days}d"], errors="coerce")
            away_count = pd.to_numeric(group[f"away_tm_players_count_{days}d"], errors="coerce")
            home_has = home_count.notna()
            away_has = away_count.notna()
            both_has = home_has & away_has
            both_counts = pd.concat([home_count, away_count], ignore_index=True).dropna()
            row[f"home_valuation_coverage_{days}d"] = float(home_has.mean()) if len(group) else 0.0
            row[f"away_valuation_coverage_{days}d"] = float(away_has.mean()) if len(group) else 0.0
            row[f"both_valuation_coverage_{days}d"] = float(both_has.mean()) if len(group) else 0.0
            row[f"median_players_per_home_snapshot_{days}d"] = float(home_count.dropna().median()) if home_count.notna().any() else np.nan
            row[f"median_players_per_away_snapshot_{days}d"] = float(away_count.dropna().median()) if away_count.notna().any() else np.nan
            row[f"pct_snapshots_at_least_5_players_{days}d"] = float((both_counts >= 5).mean()) if len(both_counts) else 0.0
            row[f"pct_snapshots_at_least_11_players_{days}d"] = float((both_counts >= 11).mean()) if len(both_counts) else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["league", "season_end_year"]).reset_index(drop=True)


def leakage_audit(proxy: pd.DataFrame) -> dict:
    warnings = []
    date_checks = []
    for side in ["home", "away"]:
        for days in TM_WINDOW_DAYS:
            used_column = f"{side}_tm_latest_valuation_date_{days}d"
            used = pd.to_datetime(proxy[used_column], errors="coerce")
            match_date = pd.to_datetime(proxy["Date"], errors="coerce")
            violations = used.notna() & (used >= match_date)
            date_checks.append(
                {
                    "check": f"{used_column} strictly before Date",
                    "violations": int(violations.sum()),
                    "checked_rows": int(used.notna().sum()),
                }
            )
            if violations.any():
                warnings.append(f"{used_column} has {int(violations.sum())} rows with valuation_date >= match Date.")

    closing_columns = [column for column in proxy.columns if column.startswith(CLOSING_PREFIXES) or column in {"AHCh"}]
    current_club_columns = [column for column in proxy.columns if "current_club" in column.casefold()]
    if closing_columns:
        warnings.append(f"Closing odds columns present in proxy output: {', '.join(closing_columns)}")
    if current_club_columns:
        warnings.append(f"current_club columns present in proxy output: {', '.join(current_club_columns)}")

    return {
        "date_checks": pd.DataFrame(date_checks),
        "closing_columns": closing_columns,
        "current_club_columns": current_club_columns,
        "diagnostic_club_history_used": False,
        "future_valuation_leakage": bool(warnings),
        "warnings": warnings,
    }


def classification(coverage: pd.DataFrame, audit: dict) -> tuple[str, list[str]]:
    reasons = []
    if audit["warnings"]:
        return "leakage_risk_detected", audit["warnings"]

    e0 = coverage[(coverage["league"].eq("E0")) & (coverage["season_end_year"].between(2020, 2025))]
    coverage_gate = bool(len(e0) and e0["both_valuation_coverage_365d"].ge(0.70).all())
    home_median_gate = bool(len(e0) and e0["median_players_per_home_snapshot_365d"].ge(11).all())
    away_median_gate = bool(len(e0) and e0["median_players_per_away_snapshot_365d"].ge(11).all())
    mapping_gate = bool(len(e0) and e0["both_mapped"].sum() / max(e0["matches"].sum(), 1) >= 0.70)

    if not coverage_gate:
        reasons.append("E0 2020-2025 both-side 365d valuation coverage is below 70%.")
    if not (home_median_gate and away_median_gate):
        reasons.append("E0 2020-2025 median players per 365d snapshot is below 11 or unavailable.")
    if not mapping_gate:
        reasons.append("High-confidence E0 mapping coverage is below the readiness threshold.")

    any_both_365 = bool(coverage["both_valuation_coverage_365d"].gt(0).any())
    if coverage_gate and home_median_gate and away_median_gate and mapping_gate:
        return "valuation_proxy_ready_for_predictive_audit", reasons
    if any_both_365:
        return "valuation_proxy_partially_usable", reasons
    return "valuation_proxy_too_sparse", reasons


def markdown_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(max_rows).fillna("").to_markdown(index=False)


def write_reports(proxy: pd.DataFrame, coverage: pd.DataFrame, audit: dict, final_classification: str, reasons: list[str]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROXY_PATH.parent.mkdir(parents=True, exist_ok=True)
    proxy.to_csv(PROXY_PATH, index=False)
    coverage.to_csv(COVERAGE_CSV, index=False)

    coverage_lines = [
        "# Transfermarkt Valuation-Only Proxy Coverage",
        "",
        f"Proxy dataset: `{PROXY_PATH}`",
        f"Rows: {len(proxy)}",
        "Scope: valuation-only club strength proxy; no betting strategies, value searches, model training, or closing-odds features.",
        "`home_mapped`, `away_mapped`, and `both_mapped` are match-row counts, not unique-team counts.",
        "",
        markdown_table(coverage),
    ]
    COVERAGE_MD.write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")

    leakage_lines = [
        "# Transfermarkt Valuation-Only Proxy Leakage Audit",
        "",
        "Checks run:",
        "- Used valuation dates must be strictly before match `Date`.",
        "- Proxy output must not contain closing odds columns.",
        "- Proxy output must not contain `current_club` columns.",
        "- Diagnostic-only club history is not loaded or used.",
        "- `players.current_club_*` is not loaded by this review.",
        "",
        "## Date Checks",
        markdown_table(audit["date_checks"]),
        "",
        f"Closing odds columns in proxy: `{', '.join(audit['closing_columns']) if audit['closing_columns'] else 'none'}`",
        f"current_club columns in proxy: `{', '.join(audit['current_club_columns']) if audit['current_club_columns'] else 'none'}`",
        "Diagnostic-only club history used: `no`",
        "",
        "## Warnings",
        "\n".join(f"- {warning}" for warning in audit["warnings"]) if audit["warnings"] else "_No leakage warnings detected._",
    ]
    LEAKAGE_MD.write_text("\n".join(leakage_lines) + "\n", encoding="utf-8")

    readiness_lines = [
        "# Transfermarkt Valuation-Only Proxy Readiness",
        "",
        f"Final classification: **{final_classification}**",
        "",
        "Predictive audit gate:",
        "- E0 2020-2025 both valuation coverage 365d >= 70%.",
        "- Median players per 365d snapshot >= 11.",
        "- No leakage warnings.",
        "- High-confidence mapping coverage for the tested league.",
        "",
        "Outcome: no models trained and no betting strategies or value searches run.",
        "",
        "## Gate Notes",
        "\n".join(f"- {reason}" for reason in reasons) if reasons else "_All gates passed._",
    ]
    READINESS_MD.write_text("\n".join(readiness_lines) + "\n", encoding="utf-8")


def main() -> None:
    matches = load_matches()
    mapping = high_confidence_mapping(load_player_squad_team_mapping())
    mapped_club_keys = set(mapping["player_data_club_name"].dropna().map(normalize_name))
    market_values = load_transfermarkt_market_values()
    market_values = market_values[market_values["club_key"].isin(mapped_club_keys)].copy()
    proxy = build_proxy_dataset(matches, market_values, mapping)
    coverage = coverage_summary(proxy, mapping)
    audit = leakage_audit(proxy)
    final_classification, reasons = classification(coverage, audit)
    write_reports(proxy, coverage, audit, final_classification, reasons)
    print(f"matches: {len(matches)}")
    print(f"proxy_rows: {len(proxy)}")
    print(f"coverage_rows: {len(coverage)}")
    print(f"classification: {final_classification}")


if __name__ == "__main__":
    main()
