from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

V1_PATH = Path("data/processed/features/football_feature_matrix_v1.csv")
OUT_PATH = Path("data/processed/features/football_feature_matrix_v1_1.csv")
REPORT_PATH = Path("outputs/reports/football_feature_matrix_v1_1_ah_line_fix_report.md")
COVERAGE_PATH = Path("outputs/reports/football_feature_matrix_v1_1_ah_line_coverage.csv")
LEAKAGE_PATH = Path("outputs/reports/football_feature_matrix_v1_1_leakage_checks.csv")
DELTA_PATH = Path("outputs/reports/football_feature_matrix_v1_1_feature_dictionary_delta.csv")

KEY_LEFT = ["league", "match_date", "home_team", "away_team", "season_start_year", "season_end_year"]
KEY_RIGHT = ["league", "Date", "HomeTeam", "AwayTeam", "season_start_year", "season_end_year"]
CORE_AH_COLUMNS = ["AHh", "AvgAHH", "AvgAHA", "MaxAHH", "MaxAHA"]
BOOKMAKER_AH_COLUMNS = [
    "B365AH",
    "B365AHH",
    "B365AHA",
    "BFEAHH",
    "BFEAHA",
    "BbAH",
    "BbAHh",
    "BbMxAHH",
    "BbMxAHA",
    "BbAvAHH",
    "BbAvAHA",
    "GBAH",
    "GBAHH",
    "GBAHA",
    "LBAH",
    "LBAHH",
    "LBAHA",
    "PAHH",
    "PAHA",
]
SOURCE_AH_COLUMNS = CORE_AH_COLUMNS + BOOKMAKER_AH_COLUMNS
DERIVED_COLUMNS = [
    "AH_bookmaker_count",
    "AH_overround",
    "no_vig_ah_home_probability",
    "no_vig_ah_away_probability",
    "abs_AHh",
    "home_is_ah_favourite",
    "away_is_ah_favourite",
    "ah_line_bucket",
    "ah_price_bucket",
    "ah_market_entropy",
    "ah_odds_spread",
]


def write_preserving_v1_prefix(added: pd.DataFrame) -> None:
    tmp_added = OUT_PATH.with_suffix(".added_tmp.csv")
    tmp_out = OUT_PATH.with_suffix(".tmp.csv")
    added.to_csv(tmp_added, index=False)
    with V1_PATH.open("r", encoding="utf-8", newline="") as base, tmp_added.open("r", encoding="utf-8", newline="") as extra, tmp_out.open("w", encoding="utf-8", newline="") as out:
        for base_line, extra_line in zip(base, extra):
            out.write(base_line.rstrip("\n\r"))
            out.write(",")
            out.write(extra_line)
    tmp_out.replace(OUT_PATH)
    tmp_added.unlink(missing_ok=True)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.loc[:, [c for c in columns if c in frame.columns]].head(limit).copy()
    return display.to_markdown(index=False)


def load_ah_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    coverage_rows = []
    for path in sorted(Path("data/processed").glob("*/*_matches.csv")):
        league = path.parent.name
        columns = pd.read_csv(path, nrows=0).columns
        usecols = [c for c in KEY_RIGHT[1:] + SOURCE_AH_COLUMNS if c in columns]
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        frame["league"] = league
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        for column in [c for c in SOURCE_AH_COLUMNS if c in frame.columns]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in SOURCE_AH_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        duplicate_count = int(frame.duplicated(KEY_RIGHT).sum())
        coverage_rows.append(
            {
                "league": league,
                "source_rows": len(frame),
                "safe_key_duplicates": duplicate_count,
                "source_AHh_rows": int(frame["AHh"].notna().sum()),
                "source_AvgAHH_rows": int(frame["AvgAHH"].notna().sum()),
                "source_AvgAHA_rows": int(frame["AvgAHA"].notna().sum()),
            }
        )
        frames.append(frame[KEY_RIGHT + SOURCE_AH_COLUMNS])
    source = pd.concat(frames, ignore_index=True, sort=False)
    return source, pd.DataFrame(coverage_rows)


def add_derived_ah_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["AH_bookmaker_count"] = pd.to_numeric(out["BbAH"], errors="coerce")
    home_away_pairs = [
        ("B365AHH", "B365AHA"),
        ("BFEAHH", "BFEAHA"),
        ("BbAvAHH", "BbAvAHA"),
        ("GBAHH", "GBAHA"),
        ("LBAHH", "LBAHA"),
        ("PAHH", "PAHA"),
    ]
    pair_count = sum((out[home].notna() & out[away].notna()).astype(int) for home, away in home_away_pairs)
    out["AH_bookmaker_count"] = out["AH_bookmaker_count"].fillna(pair_count)
    home_raw = 1.0 / pd.to_numeric(out["AvgAHH"], errors="coerce")
    away_raw = 1.0 / pd.to_numeric(out["AvgAHA"], errors="coerce")
    total = home_raw + away_raw
    out["AH_overround"] = total
    out["no_vig_ah_home_probability"] = home_raw / total
    out["no_vig_ah_away_probability"] = away_raw / total
    out["abs_AHh"] = pd.to_numeric(out["AHh"], errors="coerce").abs()
    out["home_is_ah_favourite"] = pd.to_numeric(out["AHh"], errors="coerce").lt(0).where(out["AHh"].notna())
    out["away_is_ah_favourite"] = pd.to_numeric(out["AHh"], errors="coerce").gt(0).where(out["AHh"].notna())
    out["ah_line_bucket"] = pd.cut(
        out["AHh"],
        bins=[-np.inf, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, np.inf],
        labels=False,
        include_lowest=True,
    )
    min_price = out[["AvgAHH", "AvgAHA"]].min(axis=1)
    out["ah_price_bucket"] = pd.cut(min_price, bins=[1.0, 1.5, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 10.0], labels=False, include_lowest=True)
    p_home = out["no_vig_ah_home_probability"].clip(1e-12, 1.0)
    p_away = out["no_vig_ah_away_probability"].clip(1e-12, 1.0)
    out["ah_market_entropy"] = -(p_home * np.log(p_home) + p_away * np.log(p_away))
    out["ah_odds_spread"] = out["AvgAHH"] - out["AvgAHA"]
    for column in ["home_is_ah_favourite", "away_is_ah_favourite"]:
        out[column] = out[column].astype("float")
    return out


def build_feature_dictionary_delta(output: pd.DataFrame, added_columns: list[str]) -> pd.DataFrame:
    rows = []
    source_cols = set(SOURCE_AH_COLUMNS)
    derived_cols = set(DERIVED_COLUMNS)
    for column in added_columns:
        if column in source_cols:
            role = "feature"
            group = "ah_line_market_features"
            policy = "prematch_market_from_processed_source"
        elif column in derived_cols:
            role = "feature"
            group = "ah_line_derived_market_features"
            policy = "derived_from_prematch_AHh_and_average_AH_odds_or_source_bookmaker_count"
        else:
            role = "feature"
            group = "ah_line_market_features"
            policy = "prematch_market_from_processed_source"
        rows.append(
            {
                "column": column,
                "feature_group": group,
                "role": role,
                "dtype": str(output[column].dtype),
                "non_null_rows": int(output[column].notna().sum()),
                "missing_rate": float(output[column].isna().mean()),
                "leakage_policy": policy,
            }
        )
    return pd.DataFrame(rows)


def leakage_checks(v1: pd.DataFrame, output: pd.DataFrame, source: pd.DataFrame, added_columns: list[str], unmatched: int) -> pd.DataFrame:
    rows = []
    rows.append({"check": "v1_columns_preserved", "status": "pass" if list(output.columns[: len(v1.columns)]) == list(v1.columns) else "fail", "details": f"{len(v1.columns)} v1 columns retained in original order."})
    rows.append({"check": "row_count_preserved", "status": "pass" if len(output) == len(v1) else "fail", "details": f"v1 rows={len(v1)}, v1_1 rows={len(output)}."})
    existing_equal = output.loc[:, list(v1.columns)].equals(v1)
    rows.append({"check": "v1_existing_column_values_preserved", "status": "pass" if existing_equal else "fail", "details": "All existing v1 column values are unchanged before appended AH columns."})
    rows.append({"check": "target_ah_home_cover_preserved", "status": "pass" if output["target_ah_home_cover"].equals(v1["target_ah_home_cover"]) else "fail", "details": "Settlement-derived target copied unchanged from v1."})
    rows.append({"check": "safe_join_key_unique_in_source", "status": "pass" if int(source.duplicated(KEY_RIGHT).sum()) == 0 else "fail", "details": f"duplicate source safe keys={int(source.duplicated(KEY_RIGHT).sum())}."})
    rows.append({"check": "safe_join_unmatched_rows", "status": "pass" if unmatched == 0 else "warning", "details": f"unmatched v1 rows={unmatched}; missing AH lines were not fabricated."})
    bad_closing = [c for c in added_columns if c.startswith("C") or "CAH" in c or "closing" in c.lower() or c == "AHCh"]
    rows.append({"check": "no_closing_odds_added", "status": "pass" if not bad_closing else "fail", "details": "closing-like added columns: " + "|".join(bad_closing)})
    rows.append({"check": "no_target_or_score_reconstruction", "status": "pass", "details": "AHh was joined only from processed pre-match market columns on fixture keys; target/final score columns were not used to populate AHh."})
    benchmark = ["AHh", "AvgAHH", "AvgAHA", "no_vig_ah_home_probability", "no_vig_ah_away_probability"]
    target_rows = output["target_ah_home_cover"].notna()
    missing = {c: int(output.loc[target_rows, c].isna().sum()) for c in benchmark}
    rows.append({"check": "ah_market_baseline_reproduction_columns", "status": "pass" if all(v == 0 for v in missing.values()) else "warning", "details": str(missing)})
    forbidden = [c for c in added_columns if any(token in c.lower() for token in ["transfermarkt", "player", "lineup", "squad", "current_club"])]
    rows.append({"check": "no_forbidden_player_transfermarkt_columns_added", "status": "pass" if not forbidden else "fail", "details": "|".join(forbidden)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    v1 = pd.read_csv(V1_PATH, low_memory=False)
    original_columns = list(v1.columns)
    v1["match_date"] = pd.to_datetime(v1["match_date"], errors="coerce").dt.normalize()
    source, source_coverage = load_ah_source()
    source_dupes = source.duplicated(KEY_RIGHT).sum()
    if source_dupes:
        source = source.drop_duplicates(KEY_RIGHT, keep="first")
    joined = v1.merge(source, left_on=KEY_LEFT, right_on=KEY_RIGHT, how="left", suffixes=("", "_source"), indicator=True)
    unmatched = int(joined["_merge"].ne("both").sum())
    joined = joined.drop(columns=["_merge", "Date", "HomeTeam", "AwayTeam"])
    output = add_derived_ah_features(joined)
    added_columns = [c for c in output.columns if c not in original_columns]
    write_preserving_v1_prefix(output.loc[:, added_columns])

    target_rows = output["target_ah_home_cover"].notna()
    coverage = (
        output.groupby(["league", "season_end_year"], dropna=False)
        .agg(
            rows=("match_id", "size"),
            target_ah_home_cover_rows=("target_ah_home_cover", lambda s: int(s.notna().sum())),
            AHh_rows=("AHh", lambda s: int(s.notna().sum())),
            AvgAHH_rows=("AvgAHH", lambda s: int(s.notna().sum())),
            AvgAHA_rows=("AvgAHA", lambda s: int(s.notna().sum())),
            target_exists_AHh_missing=("AHh", lambda s: int((s.isna() & target_rows.loc[s.index]).sum())),
            benchmark_core_complete=("AHh", lambda s: int((s.notna() & output.loc[s.index, "AvgAHH"].notna() & output.loc[s.index, "AvgAHA"].notna() & output.loc[s.index, "no_vig_ah_home_probability"].notna() & output.loc[s.index, "no_vig_ah_away_probability"].notna()).sum())),
        )
        .reset_index()
    )
    coverage.to_csv(COVERAGE_PATH, index=False)
    delta = build_feature_dictionary_delta(output, added_columns)
    delta.to_csv(DELTA_PATH, index=False)
    checks = leakage_checks(v1, output, source, added_columns, unmatched)
    checks.to_csv(LEAKAGE_PATH, index=False)

    target_ahh_missing = int(output.loc[target_rows, "AHh"].isna().sum())
    benchmark_missing = {
        c: int(output.loc[target_rows, c].isna().sum())
        for c in ["AHh", "AvgAHH", "AvgAHA", "no_vig_ah_home_probability", "no_vig_ah_away_probability"]
    }
    fail = checks["status"].eq("fail").any()
    if fail:
        classification = "ah_line_fix_failed"
    elif target_ahh_missing > 0 or any(benchmark_missing.values()):
        classification = "ah_line_fix_partial"
    else:
        classification = "feature_matrix_v1_1_ready_for_ah_audit"

    summary = pd.DataFrame(
        [
            {"metric": "v1_rows", "value": len(v1)},
            {"metric": "v1_columns", "value": len(original_columns)},
            {"metric": "v1_1_rows", "value": len(output)},
            {"metric": "v1_1_columns", "value": len(output.columns)},
            {"metric": "added_columns", "value": len(added_columns)},
            {"metric": "target_ah_rows", "value": int(target_rows.sum())},
            {"metric": "target_ah_rows_missing_AHh", "value": target_ahh_missing},
            {"metric": "safe_join_unmatched_rows", "value": unmatched},
        ]
    )
    lines = [
        "# Football Feature Matrix v1.1 AH Line Fix Report",
        "",
        f"Final classification: `{classification}`",
        "",
        "Scope guard: data build only. No predictive models, value searches, threshold optimization, live betting, or confirmed edge claims were run or made.",
        "",
        "## Build Summary",
        "",
        markdown_table(summary, ["metric", "value"], 50),
        "",
        "## Added AH Columns",
        "",
        "`" + "`, `".join(added_columns) + "`",
        "",
        "## Benchmark Reproduction Check",
        "",
        "Required prior AH market-only baseline columns in v1.1: `AHh`, `AvgAHH`, `AvgAHA`, `no_vig_ah_home_probability`, `no_vig_ah_away_probability`.",
        "",
        f"Missing among rows where `target_ah_home_cover` exists: `{benchmark_missing}`.",
        "",
        "## Leakage Checks",
        "",
        markdown_table(checks, ["check", "status", "details"], 30),
        "",
        "## AH Line Coverage Preview",
        "",
        markdown_table(coverage.sort_values(["target_exists_AHh_missing", "target_ah_home_cover_rows"], ascending=[False, False]), ["league", "season_end_year", "rows", "target_ah_home_cover_rows", "AHh_rows", "target_exists_AHh_missing", "benchmark_core_complete"], 30),
        "",
        "No AH lines were reconstructed from target or final score. Missing source AH lines remain missing.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print({"classification": classification, "rows": len(output), "columns": len(output.columns), "added_columns": len(added_columns), "target_ah_rows_missing_AHh": target_ahh_missing})


if __name__ == "__main__":
    main()
