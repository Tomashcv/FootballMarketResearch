from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.features.contextual_features import assert_no_closing_columns
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import asian_profit

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)


REPORT_PATH = Path("outputs/reports/market_disagreement_1x2_feature_audit.md")
COVERAGE_PATH = Path("outputs/reports/market_disagreement_1x2_feature_coverage.csv")
PREVIEW_PATH = Path("outputs/reports/market_disagreement_1x2_feature_preview.csv")
DETAIL_DIR = Path("outputs/market_disagreement_1x2")

LEAGUES = ["E0", "I1", "SP1", "D1", "F1"]
OPEN_SOURCES = {
    "avg": ("AvgH", "AvgD", "AvgA"),
    "max": ("MaxH", "MaxD", "MaxA"),
    "b365": ("B365H", "B365D", "B365A"),
    "ps": ("PSH", "PSD", "PSA"),
    "bw": ("BWH", "BWD", "BWA"),
    "iw": ("IWH", "IWD", "IWA"),
    "wh": ("WHH", "WHD", "WHA"),
    "vc": ("VCH", "VCD", "VCA"),
    "1xbet": ("1XBH", "1XBD", "1XBA"),
    "bfe": ("BFEH", "BFED", "BFEA"),
}
CLOSE_SOURCES = {
    "avg_close": ("AvgCH", "AvgCD", "AvgCA"),
    "max_close": ("MaxCH", "MaxCD", "MaxCA"),
    "b365_close": ("B365CH", "B365CD", "B365CA"),
    "ps_close": ("PSCH", "PSCD", "PSCA"),
    "1xbet_close": ("1XBCH", "1XBCD", "1XBCA"),
    "bfe_close": ("BFECH", "BFECD", "BFECA"),
}
SIDES = {"home": "H", "draw": "D", "away": "A"}
FEATURE_PREFIXES = (
    "prob_",
    "source_minus_",
    "max_minus_avg_",
    "bookmaker_probability_",
    "sharp_soft_",
    "draw_disagreement_index",
    "away_disagreement_index",
    "home_disagreement_index",
    "away_1x2_market_strength_minus_ah_market_strength",
    "home_1x2_market_strength_minus_ah_market_strength",
    "away_ps_prob_minus_avg_prob",
    "away_bfe_prob_minus_avg_prob",
    "away_max_prob_minus_avg_prob",
    "draw_pressure_index",
    "favourite_strength_disagreement",
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
        frame["season_end_year"] = pd.to_numeric(frame["season_end_year"], errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def available_sources(columns: list[str], candidates: dict[str, tuple[str, str, str]]) -> dict[str, tuple[str, str, str]]:
    present = set(columns)
    return {name: triplet for name, triplet in candidates.items() if set(triplet).issubset(present)}


def add_source_probabilities(frame: pd.DataFrame, sources: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    odds_columns = [column for triplet in sources.values() for column in triplet]
    assert_no_closing_columns(odds_columns)
    output = frame.copy()
    for source, (home_col, draw_col, away_col) in sources.items():
        raw_home = 1.0 / pd.to_numeric(output[home_col], errors="coerce")
        raw_draw = 1.0 / pd.to_numeric(output[draw_col], errors="coerce")
        raw_away = 1.0 / pd.to_numeric(output[away_col], errors="coerce")
        total = raw_home + raw_draw + raw_away
        output[f"{source}_1x2_overround"] = total
        output[f"prob_{source}_home"] = raw_home / total
        output[f"prob_{source}_draw"] = raw_draw / total
        output[f"prob_{source}_away"] = raw_away / total
    return output


def add_diagnostic_source_probabilities(frame: pd.DataFrame, sources: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    output = frame.copy()
    for source, (home_col, draw_col, away_col) in sources.items():
        raw_home = 1.0 / pd.to_numeric(output[home_col], errors="coerce")
        raw_draw = 1.0 / pd.to_numeric(output[draw_col], errors="coerce")
        raw_away = 1.0 / pd.to_numeric(output[away_col], errors="coerce")
        total = raw_home + raw_draw + raw_away
        output[f"prob_{source}_home"] = raw_home / total
        output[f"prob_{source}_draw"] = raw_draw / total
        output[f"prob_{source}_away"] = raw_away / total
    return output


def add_ah_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if {"AvgAHH", "AvgAHA"}.issubset(output.columns):
        home_raw = 1.0 / pd.to_numeric(output["AvgAHH"], errors="coerce")
        away_raw = 1.0 / pd.to_numeric(output["AvgAHA"], errors="coerce")
        total = home_raw + away_raw
        output["prob_avg_ah_home"] = home_raw / total
        output["prob_avg_ah_away"] = away_raw / total
    else:
        output["prob_avg_ah_home"] = np.nan
        output["prob_avg_ah_away"] = np.nan
    return output


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["home_win_outcome"] = output["FTR"].eq("H").astype(float)
    output["draw_outcome"] = output["FTR"].eq("D").astype(float)
    output["away_win_outcome"] = output["FTR"].eq("A").astype(float)
    required = {"AHh", "AvgAHA", "FTHG", "FTAG"}
    if required.issubset(output.columns):
        ah_line = pd.to_numeric(output["AHh"], errors="coerce")
        away_odds = pd.to_numeric(output["AvgAHA"], errors="coerce")
        away_margin = pd.to_numeric(output["FTAG"], errors="coerce") - pd.to_numeric(output["FTHG"], errors="coerce")
        output["away_ah_cover_outcome"] = [
            float(asian_profit(margin, -line, odds) > 0.0) if pd.notna(margin) and pd.notna(line) and pd.notna(odds) else np.nan
            for margin, line, odds in zip(away_margin, ah_line, away_odds)
        ]
    else:
        output["away_ah_cover_outcome"] = np.nan
    return output


def add_disagreement_features(frame: pd.DataFrame, sources: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    output = frame.copy()
    source_names = list(sources)
    for side in SIDES:
        for source in source_names:
            source_col = f"prob_{source}_{side}"
            if source_col not in output.columns:
                continue
            for anchor in ["avg", "ps", "b365"]:
                anchor_col = f"prob_{anchor}_{side}"
                if anchor_col in output.columns and source != anchor:
                    output[f"source_minus_{anchor}_{source}_{side}"] = output[source_col] - output[anchor_col]
        if f"prob_max_{side}" in output.columns and f"prob_avg_{side}" in output.columns:
            output[f"max_minus_avg_{side}"] = output[f"prob_max_{side}"] - output[f"prob_avg_{side}"]

        prob_cols = [f"prob_{source}_{side}" for source in source_names if f"prob_{source}_{side}" in output.columns]
        output[f"bookmaker_probability_std_{side}"] = output[prob_cols].std(axis=1, skipna=True)
        output[f"bookmaker_probability_range_{side}"] = output[prob_cols].max(axis=1, skipna=True) - output[prob_cols].min(axis=1, skipna=True)

        sharp_cols = [column for column in [f"prob_ps_{side}", f"prob_bfe_{side}"] if column in output.columns]
        soft_cols = [
            column
            for column in [f"prob_b365_{side}", f"prob_bw_{side}", f"prob_wh_{side}", f"prob_vc_{side}", f"prob_iw_{side}"]
            if column in output.columns
        ]
        output[f"sharp_soft_disagreement_{side}"] = output[sharp_cols].mean(axis=1, skipna=True) - output[soft_cols].mean(axis=1, skipna=True)

    output["home_disagreement_index"] = output[["bookmaker_probability_std_home", "bookmaker_probability_range_home"]].mean(axis=1, skipna=True)
    output["draw_disagreement_index"] = output[["bookmaker_probability_std_draw", "bookmaker_probability_range_draw"]].mean(axis=1, skipna=True)
    output["away_disagreement_index"] = output[["bookmaker_probability_std_away", "bookmaker_probability_range_away"]].mean(axis=1, skipna=True)

    fav_home = output.get("prob_avg_home") >= output.get("prob_avg_away")
    derived = pd.DataFrame(index=output.index)
    derived["away_1x2_market_strength_minus_ah_market_strength"] = output.get("prob_avg_away") - output.get("prob_avg_ah_away")
    derived["home_1x2_market_strength_minus_ah_market_strength"] = output.get("prob_avg_home") - output.get("prob_avg_ah_home")
    derived["away_ps_prob_minus_avg_prob"] = output.get("prob_ps_away") - output.get("prob_avg_away")
    derived["away_bfe_prob_minus_avg_prob"] = output.get("prob_bfe_away") - output.get("prob_avg_away")
    derived["away_max_prob_minus_avg_prob"] = output.get("prob_max_away") - output.get("prob_avg_away")
    derived["draw_pressure_index"] = output[["draw_disagreement_index", "sharp_soft_disagreement_draw"]].abs().mean(axis=1, skipna=True)
    derived["favourite_strength_disagreement"] = np.where(
        fav_home,
        derived["home_1x2_market_strength_minus_ah_market_strength"],
        derived["away_1x2_market_strength_minus_ah_market_strength"],
    )
    return pd.concat([output, derived], axis=1)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith(FEATURE_PREFIXES)]
    assert_no_closing_columns(columns)
    return columns


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, str, str]], dict[str, tuple[str, str, str]]]:
    open_sources = available_sources(list(matches.columns), OPEN_SOURCES)
    close_sources = available_sources(list(matches.columns), CLOSE_SOURCES)
    output = add_source_probabilities(matches, open_sources)
    output = add_ah_probabilities(output)
    output = add_targets(output)
    output = add_disagreement_features(output, open_sources)
    return output, open_sources, close_sources


def coverage_table(features: pd.DataFrame, open_sources: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    fcols = feature_columns(features)
    for (league, season), group in features.groupby(["league", "season_end_year"], dropna=False):
        row = {
            "league": league,
            "season_end_year": int(season) if pd.notna(season) else pd.NA,
            "matches": len(group),
            "feature_rows_any": int(group[fcols].notna().any(axis=1).sum()) if fcols else 0,
            "feature_rows_core_avg_ps_b365": int(
                group[["prob_avg_home", "prob_ps_home", "prob_b365_home"]].notna().all(axis=1).sum()
            ),
            "ah_rows": int(group[["prob_avg_ah_home", "prob_avg_ah_away"]].notna().all(axis=1).sum()),
        }
        for source in open_sources:
            row[f"{source}_rows"] = int(group[[f"prob_{source}_home", f"prob_{source}_draw", f"prob_{source}_away"]].notna().all(axis=1).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["league", "season_end_year"]).reset_index(drop=True)


def distribution_table(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in feature_columns(features):
        values = pd.to_numeric(features[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "feature": column,
                "rows": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=0),
                "p05": values.quantile(0.05),
                "p50": values.quantile(0.50),
                "p95": values.quantile(0.95),
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def correlation_table(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = ["home_win_outcome", "draw_outcome", "away_win_outcome", "away_ah_cover_outcome"]
    for league, group in features.groupby("league"):
        for feature in feature_columns(features):
            values = pd.to_numeric(group[feature], errors="coerce")
            for target in targets:
                clean = pd.DataFrame({"feature": values, "target": pd.to_numeric(group[target], errors="coerce")}).dropna()
                if len(clean) < 100 or clean["feature"].nunique() < 2 or clean["target"].nunique() < 2:
                    continue
                rows.append({"league": league, "feature": feature, "target": target, "rows": len(clean), "corr": clean["feature"].corr(clean["target"])})
    return pd.DataFrame(rows)


def stability_table(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = [
        "home_disagreement_index",
        "draw_disagreement_index",
        "away_disagreement_index",
        "draw_pressure_index",
        "favourite_strength_disagreement",
        "away_1x2_market_strength_minus_ah_market_strength",
    ]
    for (league, season), group in features.groupby(["league", "season_end_year"]):
        for feature in selected:
            values = pd.to_numeric(group.get(feature), errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({"league": league, "season_end_year": int(season), "feature": feature, "rows": len(values), "mean": values.mean(), "std": values.std(ddof=0)})
    return pd.DataFrame(rows)


def closing_movement_table(features: pd.DataFrame, matches: pd.DataFrame, close_sources: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    if "avg_close" not in close_sources:
        return pd.DataFrame()
    close = add_diagnostic_source_probabilities(matches[["league", "Date", "HomeTeam", "AwayTeam", *close_sources["avg_close"]]].copy(), {"avg_close": close_sources["avg_close"]})
    close = close[["league", "Date", "HomeTeam", "AwayTeam", "prob_avg_close_home", "prob_avg_close_draw", "prob_avg_close_away"]]
    merged = features.merge(close, on=["league", "Date", "HomeTeam", "AwayTeam"], how="left")
    merged["close_move_home"] = merged["prob_avg_close_home"] - merged["prob_avg_home"]
    merged["close_move_draw"] = merged["prob_avg_close_draw"] - merged["prob_avg_draw"]
    merged["close_move_away"] = merged["prob_avg_close_away"] - merged["prob_avg_away"]
    rows = []
    selected = [
        "home_disagreement_index",
        "draw_disagreement_index",
        "away_disagreement_index",
        "draw_pressure_index",
        "favourite_strength_disagreement",
        "away_ps_prob_minus_avg_prob",
        "away_bfe_prob_minus_avg_prob",
    ]
    for league, group in merged.groupby("league"):
        for feature in selected:
            values = pd.to_numeric(group.get(feature), errors="coerce")
            for movement in ["close_move_home", "close_move_draw", "close_move_away"]:
                clean = pd.DataFrame({"feature": values, "movement": pd.to_numeric(group[movement], errors="coerce")}).dropna()
                if len(clean) < 100 or clean["feature"].nunique() < 2 or clean["movement"].nunique() < 2:
                    continue
                rows.append({"league": league, "feature": feature, "closing_movement": movement, "rows": len(clean), "corr": clean["feature"].corr(clean["movement"])})
    return pd.DataFrame(rows)


def source_missingness_table(features: pd.DataFrame, open_sources: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for league, group in features.groupby("league"):
        for source in open_sources:
            mask = group[[f"prob_{source}_home", f"prob_{source}_draw", f"prob_{source}_away"]].notna().all(axis=1)
            rows.append({"league": league, "source": source, "matches": len(group), "available_rows": int(mask.sum()), "missing_rate": float(1.0 - mask.mean())})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[columns].head(max_rows).to_markdown(index=False, headers=headers, floatfmt=".4f")


def classify(coverage: pd.DataFrame) -> str:
    if coverage.empty or coverage["feature_rows_any"].sum() == 0:
        return "insufficient coverage"
    core_rate = coverage["feature_rows_core_avg_ps_b365"].sum() / coverage["matches"].sum()
    ah_rate = coverage["ah_rows"].sum() / coverage["matches"].sum()
    if core_rate >= 0.70 and ah_rate >= 0.30:
        return "usable"
    if core_rate >= 0.30:
        return "partially usable"
    return "diagnostic only"


def write_report(
    features: pd.DataFrame,
    open_sources: dict[str, tuple[str, str, str]],
    close_sources: dict[str, tuple[str, str, str]],
    coverage: pd.DataFrame,
    missingness: pd.DataFrame,
    distributions: pd.DataFrame,
    correlations: pd.DataFrame,
    closing: pd.DataFrame,
    stability: pd.DataFrame,
    classification: str,
) -> None:
    source_rows = []
    for source, triplet in open_sources.items():
        source_rows.append({"source": source, "home": triplet[0], "draw": triplet[1], "away": triplet[2], "use": "feature"})
    for source, triplet in close_sources.items():
        source_rows.append({"source": source, "home": triplet[0], "draw": triplet[1], "away": triplet[2], "use": "diagnostic only"})
    sources = pd.DataFrame(source_rows)
    top_corr = correlations.assign(abs_corr=correlations["corr"].abs()).sort_values("abs_corr", ascending=False)
    top_closing = closing.assign(abs_corr=closing["corr"].abs()).sort_values("abs_corr", ascending=False) if len(closing) else closing
    lines = [
        "# 1X2 Bookmaker-Disagreement Feature Audit",
        "",
        "No betting strategies or value models were run. Raw match data was not edited. Opening/pre-match 1X2 and AH odds were used for feature diagnostics; closing odds were used only for movement diagnostics.",
        "",
        "## Source Mapping",
        "",
        markdown_table(sources, ["source", "home", "draw", "away", "use"], ["Source", "Home", "Draw", "Away", "Use"], max_rows=30),
        "",
        "## Coverage By League And Season",
        "",
        markdown_table(coverage, ["league", "season_end_year", "matches", "feature_rows_any", "feature_rows_core_avg_ps_b365", "ah_rows"], ["League", "Season", "Matches", "Any feature", "Avg/PS/B365", "AH rows"], max_rows=80),
        "",
        "## Source Missingness",
        "",
        markdown_table(missingness, ["league", "source", "matches", "available_rows", "missing_rate"], ["League", "Source", "Matches", "Available", "Missing rate"], max_rows=80),
        "",
        "## Feature Distributions",
        "",
        markdown_table(distributions, ["feature", "rows", "mean", "std", "p05", "p50", "p95"], ["Feature", "Rows", "Mean", "Std", "P05", "P50", "P95"], max_rows=80),
        "",
        "## Strongest Outcome/AH Diagnostic Correlations",
        "",
        markdown_table(top_corr, ["league", "feature", "target", "rows", "corr"], ["League", "Feature", "Target", "Rows", "Corr"], max_rows=80),
        "",
        "## Closing Movement Diagnostics",
        "",
        markdown_table(top_closing, ["league", "feature", "closing_movement", "rows", "corr"], ["League", "Feature", "Closing movement", "Rows", "Corr"], max_rows=80),
        "",
        "## Feature Stability By Season",
        "",
        markdown_table(stability, ["league", "season_end_year", "feature", "rows", "mean", "std"], ["League", "Season", "Feature", "Rows", "Mean", "Std"], max_rows=100),
        "",
        "## Leakage Audit",
        "",
        "- Feature columns were built only from opening/pre-match 1X2 sources and opening/pre-match AH odds.",
        "- Closing columns were not included in the feature matrix and were used only in `closing_movement_diagnostics.csv`.",
        "- Outcome columns and AH cover are diagnostic targets only.",
        "- `1XBH/1XBD/1XBA` are treated as 1XBet 1X2 odds, not double-chance odds.",
        f"- Feature column count: {len(feature_columns(features))}.",
        "",
        f"Final classification: **{classification}**",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    if matches.empty:
        raise SystemExit("No processed match data found")
    features, open_sources, close_sources = build_features(matches)
    fcols = feature_columns(features)
    preview_columns = [
        "league",
        "Date",
        "season_end_year",
        "HomeTeam",
        "AwayTeam",
        "FTR",
        "away_ah_cover_outcome",
        *fcols,
    ]
    coverage = coverage_table(features, open_sources)
    missingness = source_missingness_table(features, open_sources)
    distributions = distribution_table(features)
    correlations = correlation_table(features)
    closing = closing_movement_table(features, matches, close_sources)
    stability = stability_table(features)
    classification = classify(coverage)
    coverage.to_csv(COVERAGE_PATH, index=False)
    preview = features[preview_columns].dropna(subset=fcols, how="all").copy()
    core_preview = preview.dropna(subset=["prob_avg_home", "prob_ps_home", "prob_b365_home"], how="any")
    if len(core_preview):
        preview = core_preview
    preview.head(1000).to_csv(PREVIEW_PATH, index=False)
    missingness.to_csv(DETAIL_DIR / "source_missingness.csv", index=False)
    distributions.to_csv(DETAIL_DIR / "feature_distributions.csv", index=False)
    correlations.to_csv(DETAIL_DIR / "feature_correlations.csv", index=False)
    closing.to_csv(DETAIL_DIR / "closing_movement_diagnostics.csv", index=False)
    stability.to_csv(DETAIL_DIR / "feature_stability_by_season.csv", index=False)
    write_report(features, open_sources, close_sources, coverage, missingness, distributions, correlations, closing, stability, classification)
    print(REPORT_PATH)
    print(COVERAGE_PATH)
    print(PREVIEW_PATH)
    print(f"classification={classification}")


if __name__ == "__main__":
    main()
