from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.metrics import expected_calibration_error


REPORT_PATH = Path("outputs/reports/market_consistency_1x2_double_chance_audit.md")
PREVIEW_PATH = Path("outputs/reports/market_consistency_1x2_double_chance_features_preview.csv")
DETAIL_DIR = Path("outputs/market_consistency_1x2_double_chance")

LEAGUES = ["E0", "I1", "SP1", "D1", "F1"]
OPEN_1X2_SOURCES = {
    "Avg": ("AvgH", "AvgD", "AvgA"),
    "Max": ("MaxH", "MaxD", "MaxA"),
    "B365": ("B365H", "B365D", "B365A"),
    "PS": ("PSH", "PSD", "PSA"),
    "BW": ("BWH", "BWD", "BWA"),
    "IW": ("IWH", "IWD", "IWA"),
    "WH": ("WHH", "WHD", "WHA"),
    "VC": ("VCH", "VCD", "VCA"),
    "1XBet": ("1XBH", "1XBD", "1XBA"),
    "BFE": ("BFEH", "BFED", "BFEA"),
}
CLOSING_1X2_SOURCES = {
    "AvgClose": ("AvgCH", "AvgCD", "AvgCA"),
    "MaxClose": ("MaxCH", "MaxCD", "MaxCA"),
    "B365Close": ("B365CH", "B365CD", "B365CA"),
    "PSClose": ("PSCH", "PSCD", "PSCA"),
    "1XBetClose": ("1XBCH", "1XBCD", "1XBCA"),
    "BFEClose": ("BFECH", "BFECD", "BFECA"),
}
DOUBLE_CHANCE_CANDIDATES = {
    "1x": ("1X", "DC1X", "DoubleChance1X", "DoubleChance_1X", "Avg1X", "B3651X", "PS1X"),
    "12": ("12", "DC12", "DoubleChance12", "DoubleChance_12", "Avg12", "B36512", "PS12"),
    "x2": ("X2", "DCX2", "DoubleChanceX2", "DoubleChance_X2", "AvgX2", "B365X2", "PSX2"),
}
BOOKMAKER_PREFIX_NOT_DOUBLE_CHANCE = ("1XBH", "1XBD", "1XBA", "1XBCH", "1XBCD", "1XBCA")


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


def available_triplets(columns: list[str], candidates: dict[str, tuple[str, str, str]]) -> dict[str, tuple[str, str, str]]:
    present = set(columns)
    return {name: triplet for name, triplet in candidates.items() if set(triplet).issubset(present)}


def find_double_chance_columns(columns: list[str]) -> dict[str, str | None]:
    present = {column.casefold(): column for column in columns}
    found: dict[str, str | None] = {}
    for key, candidates in DOUBLE_CHANCE_CANDIDATES.items():
        match = None
        for candidate in candidates:
            if candidate.casefold() in present and present[candidate.casefold()] not in BOOKMAKER_PREFIX_NOT_DOUBLE_CHANCE:
                match = present[candidate.casefold()]
                break
        found[key] = match
    return found


def outcome_arrays(frame: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    mapping = {"H": 0, "D": 1, "A": 2}
    y = frame["FTR"].map(mapping)
    one_hot = np.zeros((len(frame), 3), dtype=float)
    valid = y.notna()
    one_hot[np.arange(len(frame))[valid.to_numpy()], y[valid].astype(int).to_numpy()] = 1.0
    return y, one_hot


def add_1x2_probabilities(frame: pd.DataFrame, source: str, triplet: tuple[str, str, str]) -> pd.DataFrame:
    output = frame[["league", "Date", "season_end_year", "HomeTeam", "AwayTeam", "FTR"]].copy()
    h, d, a = triplet
    output["source"] = source
    output["home_odds_col"] = h
    output["draw_odds_col"] = d
    output["away_odds_col"] = a
    output["home_odds"] = pd.to_numeric(frame[h], errors="coerce")
    output["draw_odds"] = pd.to_numeric(frame[d], errors="coerce")
    output["away_odds"] = pd.to_numeric(frame[a], errors="coerce")
    output["p_home_raw"] = 1.0 / output["home_odds"]
    output["p_draw_raw"] = 1.0 / output["draw_odds"]
    output["p_away_raw"] = 1.0 / output["away_odds"]
    output["one_x_two_total_margin"] = output[["p_home_raw", "p_draw_raw", "p_away_raw"]].sum(axis=1)
    output["p_home_no_vig"] = output["p_home_raw"] / output["one_x_two_total_margin"]
    output["p_draw_no_vig"] = output["p_draw_raw"] / output["one_x_two_total_margin"]
    output["p_away_no_vig"] = output["p_away_raw"] / output["one_x_two_total_margin"]
    return output


def add_double_chance_features(probabilities: pd.DataFrame, original: pd.DataFrame, dc_columns: dict[str, str | None]) -> pd.DataFrame:
    output = probabilities.copy()
    one_x, one_two, x_two = dc_columns["1x"], dc_columns["12"], dc_columns["x2"]
    output["odds_1x_col"] = one_x or ""
    output["odds_12_col"] = one_two or ""
    output["odds_x2_col"] = x_two or ""
    output["p_1x_raw"] = 1.0 / pd.to_numeric(original[one_x], errors="coerce") if one_x else np.nan
    output["p_12_raw"] = 1.0 / pd.to_numeric(original[one_two], errors="coerce") if one_two else np.nan
    output["p_x2_raw"] = 1.0 / pd.to_numeric(original[x_two], errors="coerce") if x_two else np.nan
    output["draw_from_1_and_1x"] = output["p_1x_raw"] - output["p_home_raw"]
    output["draw_from_2_and_x2"] = output["p_x2_raw"] - output["p_away_raw"]
    output["home_from_x_and_1x"] = output["p_1x_raw"] - output["p_draw_raw"]
    output["away_from_x_and_x2"] = output["p_x2_raw"] - output["p_draw_raw"]
    output["p_home_from_dc"] = (output["p_1x_raw"] + output["p_12_raw"] - output["p_x2_raw"]) / 2.0
    output["p_draw_from_dc"] = (output["p_1x_raw"] + output["p_x2_raw"] - output["p_12_raw"]) / 2.0
    output["p_away_from_dc"] = (output["p_12_raw"] + output["p_x2_raw"] - output["p_1x_raw"]) / 2.0
    output["draw_real_minus_draw_from_1x"] = output["p_draw_raw"] - output["draw_from_1_and_1x"]
    output["draw_real_minus_draw_from_x2"] = output["p_draw_raw"] - output["draw_from_2_and_x2"]
    output["home_real_minus_home_from_dc"] = output["p_home_raw"] - output["p_home_from_dc"]
    output["away_real_minus_away_from_dc"] = output["p_away_raw"] - output["p_away_from_dc"]
    output["dc_total_margin"] = output[["p_1x_raw", "p_12_raw", "p_x2_raw"]].sum(axis=1)
    output["margin_difference_1x2_vs_dc"] = output["one_x_two_total_margin"] - output["dc_total_margin"]
    synth = output[["p_home_from_dc", "p_draw_from_dc", "p_away_from_dc"]]
    output["invalid_synthetic_probability"] = synth.lt(0).any(axis=1) | synth.gt(1).any(axis=1)
    return output


def multiclass_ece(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    return expected_calibration_error(correct, confidence)


def probability_metrics(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (league, source), group in features.groupby(["league", "source"]):
        clean = group.dropna(subset=["FTR", "p_home_no_vig", "p_draw_no_vig", "p_away_no_vig"]).copy()
        y, _ = outcome_arrays(clean)
        clean = clean[y.notna()].copy()
        y = y[y.notna()].astype(int)
        if clean.empty:
            continue
        probabilities = clean[["p_home_no_vig", "p_draw_no_vig", "p_away_no_vig"]].to_numpy(dtype=float)
        rows.append(
            {
                "league": league,
                "source": source,
                "matches": len(clean),
                "log_loss": log_loss(y, probabilities, labels=[0, 1, 2]),
                "brier": np.mean([brier_score_loss((y == klass).astype(int), probabilities[:, klass]) for klass in range(3)]),
                "ece": multiclass_ece(y.to_numpy(), probabilities),
                "home_outcome_corr": clean["p_home_no_vig"].corr((clean["FTR"] == "H").astype(float)),
                "draw_outcome_corr": clean["p_draw_no_vig"].corr((clean["FTR"] == "D").astype(float)),
                "away_outcome_corr": clean["p_away_no_vig"].corr((clean["FTR"] == "A").astype(float)),
            }
        )
    return pd.DataFrame(rows)


def coverage_rows(features: pd.DataFrame, dc_columns: dict[str, str | None]) -> pd.DataFrame:
    rows = []
    for (league, season, source), group in features.groupby(["league", "season_end_year", "source"], dropna=False):
        rows.append(
            {
                "league": league,
                "season_end_year": int(season) if pd.notna(season) else pd.NA,
                "source": source,
                "matches": len(group),
                "one_x_two_rows": int(group[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1).sum()),
                "double_chance_rows": int(group[["p_1x_raw", "p_12_raw", "p_x2_raw"]].notna().all(axis=1).sum()),
                "invalid_synthetic_probabilities": int(group["invalid_synthetic_probability"].fillna(False).sum()),
                "dc_1x_column": dc_columns["1x"] or "",
                "dc_12_column": dc_columns["12"] or "",
                "dc_x2_column": dc_columns["x2"] or "",
            }
        )
    return pd.DataFrame(rows)


def closing_diagnostics(features: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    close_sources = available_triplets(list(matches.columns), CLOSING_1X2_SOURCES)
    rows = []
    if "AvgClose" not in close_sources:
        return pd.DataFrame(rows)
    close = add_1x2_probabilities(matches, "AvgClose", close_sources["AvgClose"])
    key_cols = ["league", "Date", "HomeTeam", "AwayTeam"]
    close = close[key_cols + ["p_home_no_vig", "p_draw_no_vig", "p_away_no_vig"]].rename(
        columns={
            "p_home_no_vig": "close_home_prob",
            "p_draw_no_vig": "close_draw_prob",
            "p_away_no_vig": "close_away_prob",
        }
    )
    merged = features.merge(close, on=key_cols, how="left")
    for (league, source), group in merged.groupby(["league", "source"]):
        rows.append(
            {
                "league": league,
                "source": source,
                "rows": int(group["close_home_prob"].notna().sum()),
                "home_prob_move_corr": group["p_home_no_vig"].corr(group["close_home_prob"] - group["p_home_no_vig"]),
                "draw_prob_move_corr": group["p_draw_no_vig"].corr(group["close_draw_prob"] - group["p_draw_no_vig"]),
                "away_prob_move_corr": group["p_away_no_vig"].corr(group["close_away_prob"] - group["p_away_no_vig"]),
                "draw_disagreement_move_corr": group["draw_real_minus_draw_from_1x"].corr(group["close_draw_prob"] - group["p_draw_no_vig"]),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[columns].to_markdown(index=False, headers=headers, floatfmt=".4f")


def write_report(
    matches: pd.DataFrame,
    open_sources: dict[str, tuple[str, str, str]],
    close_sources: dict[str, tuple[str, str, str]],
    dc_columns: dict[str, str | None],
    coverage: pd.DataFrame,
    metrics: pd.DataFrame,
    closing: pd.DataFrame,
    classification: str,
) -> None:
    source_rows = []
    for source, (h, d, a) in open_sources.items():
        source_rows.append({"source": source, "home_win_odds": h, "draw_odds": d, "away_win_odds": a, "feature_use": "opening/pre-match"})
    for source, (h, d, a) in close_sources.items():
        source_rows.append({"source": source, "home_win_odds": h, "draw_odds": d, "away_win_odds": a, "feature_use": "closing diagnostic only"})
    sources = pd.DataFrame(source_rows)
    dc = pd.DataFrame(
        [
            {"double_chance_leg": "1X", "column": dc_columns["1x"] or "", "available": bool(dc_columns["1x"])},
            {"double_chance_leg": "12", "column": dc_columns["12"] or "", "available": bool(dc_columns["12"])},
            {"double_chance_leg": "X2", "column": dc_columns["x2"] or "", "available": bool(dc_columns["x2"])},
        ]
    )
    misleading = [column for column in BOOKMAKER_PREFIX_NOT_DOUBLE_CHANCE if column in matches.columns]
    lines = [
        "# Market Consistency Audit: 1X2 and Double Chance",
        "",
        "No betting strategies or value models were run. Raw data was not edited. Closing odds are diagnostic only and are not included as features.",
        "",
        "## Column Mapping",
        "",
        markdown_table(sources, ["source", "home_win_odds", "draw_odds", "away_win_odds", "feature_use"], ["Source", "Home", "Draw", "Away", "Use"]),
        "",
        "## Double Chance Columns",
        "",
        markdown_table(dc, ["double_chance_leg", "column", "available"], ["Leg", "Column", "Available"]),
        "",
        f"Bookmaker-prefix columns present but not double chance: {', '.join(misleading) if misleading else 'none'}.",
        "",
        "## Coverage",
        "",
        markdown_table(
            coverage.groupby(["league", "source"], as_index=False)[["matches", "one_x_two_rows", "double_chance_rows", "invalid_synthetic_probabilities"]].sum(),
            ["league", "source", "matches", "one_x_two_rows", "double_chance_rows", "invalid_synthetic_probabilities"],
            ["League", "Source", "Rows", "1X2 rows", "DC rows", "Invalid synthetic"],
        ),
        "",
        "## 1X2 Probability Diagnostics",
        "",
        markdown_table(metrics, ["league", "source", "matches", "log_loss", "brier", "ece", "home_outcome_corr", "draw_outcome_corr", "away_outcome_corr"], ["League", "Source", "Rows", "Log loss", "Brier", "ECE", "Home corr", "Draw corr", "Away corr"]),
        "",
        "## Closing-Move Diagnostics",
        "",
        markdown_table(closing, ["league", "source", "rows", "home_prob_move_corr", "draw_prob_move_corr", "away_prob_move_corr", "draw_disagreement_move_corr"], ["League", "Source", "Rows", "Home move corr", "Draw move corr", "Away move corr", "Draw disagreement corr"]),
        "",
        "## Leakage Audit",
        "",
        "- Opening/pre-match 1X2 odds were used for feature diagnostics.",
        "- Closing 1X2 odds were used only for closing-move diagnostics.",
        "- No double chance synthetic features were usable because no true 1X/12/X2 columns were found.",
        "- Columns `1XBH`, `1XBD`, and `1XBA` are treated as 1XBet home/draw/away odds, not double chance.",
        "",
        f"Final classification: **{classification}**",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    if matches.empty:
        raise SystemExit("No processed match data found")
    open_sources = available_triplets(list(matches.columns), OPEN_1X2_SOURCES)
    close_sources = available_triplets(list(matches.columns), CLOSING_1X2_SOURCES)
    dc_columns = find_double_chance_columns(list(matches.columns))
    feature_frames = []
    for source, triplet in open_sources.items():
        probs = add_1x2_probabilities(matches, source, triplet)
        feature_frames.append(add_double_chance_features(probs, matches, dc_columns))
    features = pd.concat(feature_frames, ignore_index=True, sort=False) if feature_frames else pd.DataFrame()
    coverage = coverage_rows(features, dc_columns) if len(features) else pd.DataFrame()
    metrics = probability_metrics(features) if len(features) else pd.DataFrame()
    closing = closing_diagnostics(features, matches) if len(features) else pd.DataFrame()
    has_dc = all(dc_columns.values()) and coverage["double_chance_rows"].sum() > 0 if len(coverage) else False
    classification = "data available" if has_dc else ("partially available" if len(open_sources) else "not available")
    if len(open_sources) and not has_dc:
        classification = "useful diagnostic only"
    preview = features[features[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1)].copy()
    if preview.empty:
        preview = features.copy()
    preview.head(500).to_csv(PREVIEW_PATH, index=False)
    coverage.to_csv(DETAIL_DIR / "coverage_by_league_season_source.csv", index=False)
    metrics.to_csv(DETAIL_DIR / "probability_metrics.csv", index=False)
    closing.to_csv(DETAIL_DIR / "closing_move_diagnostics.csv", index=False)
    write_report(matches, open_sources, close_sources, dc_columns, coverage, metrics, closing, classification)
    print(REPORT_PATH)
    print(PREVIEW_PATH)
    print(f"classification={classification}")


if __name__ == "__main__":
    main()
