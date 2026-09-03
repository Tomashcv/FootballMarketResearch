from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "E1", "E2", "E3"]

REPORT_PATH = Path("outputs/reports/ah_settlement_engine_audit.md")
UNIT_TEST_PATH = Path("outputs/reports/ah_settlement_unit_tests.csv")
REAL_DATA_PATH = Path("outputs/reports/ah_settlement_real_data_by_league_season.csv")
COMPATIBILITY_PATH = Path("outputs/reports/ah_settlement_binary_target_compatibility.csv")

LABELS = ["full_win", "half_win", "push", "half_loss", "full_loss"]


@dataclass(frozen=True)
class Settlement:
    label: str
    profit: float
    adjusted_margin: float
    parts: tuple[float, ...]


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


def label_from_part_profits(part_profits: Iterable[float]) -> str:
    values = list(part_profits)
    wins = sum(value > 0 for value in values)
    pushes = sum(value == 0 for value in values)
    losses = sum(value < 0 for value in values)
    if wins == len(values):
        return "full_win"
    if wins and pushes and not losses:
        return "half_win"
    if pushes == len(values):
        return "push"
    if losses and pushes and not wins:
        return "half_loss"
    if losses == len(values):
        return "full_loss"
    return "impossible"


def settle_side(team_margin: float, handicap: float, odds: float) -> Settlement:
    if pd.isna(team_margin) or pd.isna(handicap) or pd.isna(odds) or float(odds) <= 1.0:
        return Settlement("invalid", np.nan, np.nan, tuple())
    parts = split_handicap(float(handicap))
    part_profits = [single_part_profit(float(team_margin) + part, float(odds)) for part in parts]
    return Settlement(
        label=label_from_part_profits(part_profits),
        profit=float(np.mean(part_profits)),
        adjusted_margin=float(team_margin) + float(handicap),
        parts=parts,
    )


def no_vig_overround(home_odds: pd.Series, away_odds: pd.Series) -> pd.Series:
    home = 1.0 / pd.to_numeric(home_odds, errors="coerce")
    away = 1.0 / pd.to_numeric(away_odds, errors="coerce")
    return home + away


def load_matches() -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["league"] = league
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No processed match files found for AH settlement audit")
    output = pd.concat(frames, ignore_index=True, sort=False)
    for column in ["Date", "season_end_year", "FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA"]:
        if column == "Date" and column in output.columns:
            output[column] = pd.to_datetime(output[column], errors="coerce")
        elif column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def add_settlement_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["home_goal_margin"] = output["FTHG"] - output["FTAG"]
    output["away_goal_margin"] = -output["home_goal_margin"]
    home_settlements = [
        settle_side(margin, line, odds)
        for margin, line, odds in zip(output["home_goal_margin"], output["AHh"], output["AvgAHH"])
    ]
    away_settlements = [
        settle_side(margin, -line if pd.notna(line) else np.nan, odds)
        for margin, line, odds in zip(output["away_goal_margin"], output["AHh"], output["AvgAHA"])
    ]
    output["home_ah_profit_at_avg_odds"] = [item.profit for item in home_settlements]
    output["away_ah_profit_at_avg_odds"] = [item.profit for item in away_settlements]
    output["home_ah_settlement_label"] = [item.label for item in home_settlements]
    output["away_ah_settlement_label"] = [item.label for item in away_settlements]
    output["home_ah_adjusted_margin"] = [item.adjusted_margin for item in home_settlements]
    output["away_ah_adjusted_margin"] = [item.adjusted_margin for item in away_settlements]
    output["ah_overround_estimate"] = no_vig_overround(output["AvgAHH"], output["AvgAHA"])
    output["valid_ah_settlement"] = output["home_ah_settlement_label"].isin(LABELS) & output["away_ah_settlement_label"].isin(LABELS)
    output["home_binary_target_previous"] = np.where(
        output["home_ah_adjusted_margin"] > 0,
        1.0,
        np.where(output["home_ah_adjusted_margin"] < 0, 0.0, np.nan),
    )
    output["home_cover_from_settlement"] = np.where(
        output["home_ah_settlement_label"].isin(["full_win", "half_win"]),
        1.0,
        np.where(output["home_ah_settlement_label"].isin(["half_loss", "full_loss"]), 0.0, np.nan),
    )
    return output


def run_unit_tests() -> pd.DataFrame:
    rows = [
        (1, 0.0, 0, "push", 0.0, "push", 0.0),
        (2, -0.5, 0, "full_loss", -1.0, "full_win", 1.0),
        (3, 0.5, 0, "full_win", 1.0, "full_loss", -1.0),
        (4, -1.0, 1, "push", 0.0, "push", 0.0),
        (5, -1.0, 2, "full_win", 1.0, "full_loss", -1.0),
        (6, 1.0, -1, "push", 0.0, "push", 0.0),
        (7, -0.25, 0, "half_loss", -0.5, "half_win", 0.5),
        (8, 0.25, 0, "half_win", 0.5, "half_loss", -0.5),
        (9, -0.75, 1, "half_win", 0.5, "half_loss", -0.5),
        (10, 0.75, -1, "half_loss", -0.5, "half_win", 0.5),
        (11, -1.25, 1, "half_loss", -0.5, "half_win", 0.5),
        (12, -1.25, 2, "full_win", 1.0, "full_loss", -1.0),
        (13, 1.25, -1, "half_win", 0.5, "half_loss", -0.5),
        (14, 1.25, -2, "full_loss", -1.0, "full_win", 1.0),
        (15, -1.5, 1, "full_loss", -1.0, "full_win", 1.0),
        (16, -1.5, 2, "full_win", 1.0, "full_loss", -1.0),
    ]
    output = []
    odds = 2.0
    for case_id, ahh, margin, expected_home_label, expected_home_profit, expected_away_label, expected_away_profit in rows:
        home = settle_side(margin, ahh, odds)
        away = settle_side(-margin, -ahh, odds)
        output.append(
            {
                "case_id": case_id,
                "AHh": ahh,
                "home_margin": margin,
                "odds": odds,
                "home_ah_settlement_label": home.label,
                "expected_home_label": expected_home_label,
                "home_ah_profit_at_avg_odds": home.profit,
                "expected_home_profit": expected_home_profit,
                "away_ah_settlement_label": away.label,
                "expected_away_label": expected_away_label,
                "away_ah_profit_at_avg_odds": away.profit,
                "expected_away_profit": expected_away_profit,
                "home_passed": home.label == expected_home_label and abs(home.profit - expected_home_profit) < 1e-9,
                "away_passed": away.label == expected_away_label and abs(away.profit - expected_away_profit) < 1e-9,
                "home_split_parts": ",".join(f"{part:g}" for part in home.parts),
                "away_split_parts": ",".join(f"{part:g}" for part in away.parts),
            }
        )
    return pd.DataFrame(output)


def real_data_audit(settled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (league, season), group in settled.groupby(["league", "season_end_year"], dropna=False):
        home_counts = group["home_ah_settlement_label"].value_counts(normalize=True)
        away_counts = group["away_ah_settlement_label"].value_counts(normalize=True)
        valid = group[group["valid_ah_settlement"]].copy()
        same_match_profit = valid["home_ah_profit_at_avg_odds"] + valid["away_ah_profit_at_avg_odds"]
        impossible = group[
            ~group["home_ah_settlement_label"].isin(LABELS + ["invalid"])
            | ~group["away_ah_settlement_label"].isin(LABELS + ["invalid"])
            | (group["valid_ah_settlement"] & ~np.isclose(group["home_ah_adjusted_margin"], -group["away_ah_adjusted_margin"]))
        ]
        rows.append(
            {
                "league": league,
                "season_end_year": int(season) if pd.notna(season) else np.nan,
                "matches": int(len(group)),
                "rows_with_valid_ah_settlement": int(group["valid_ah_settlement"].sum()),
                "missing_score_rows": int(group[["FTHG", "FTAG"]].isna().any(axis=1).sum()),
                "missing_ah_line_rows": int(group["AHh"].isna().sum()),
                "missing_odds_rows": int(group[["AvgAHH", "AvgAHA"]].isna().any(axis=1).sum()),
                "home_full_win_rate": float(home_counts.get("full_win", 0.0)),
                "home_half_win_rate": float(home_counts.get("half_win", 0.0)),
                "home_push_rate": float(home_counts.get("push", 0.0)),
                "home_half_loss_rate": float(home_counts.get("half_loss", 0.0)),
                "home_full_loss_rate": float(home_counts.get("full_loss", 0.0)),
                "away_full_win_rate": float(away_counts.get("full_win", 0.0)),
                "away_half_win_rate": float(away_counts.get("half_win", 0.0)),
                "away_push_rate": float(away_counts.get("push", 0.0)),
                "away_half_loss_rate": float(away_counts.get("half_loss", 0.0)),
                "away_full_loss_rate": float(away_counts.get("full_loss", 0.0)),
                "average_home_ah_profit_using_AvgAHH": float(valid["home_ah_profit_at_avg_odds"].mean()) if len(valid) else np.nan,
                "average_away_ah_profit_using_AvgAHA": float(valid["away_ah_profit_at_avg_odds"].mean()) if len(valid) else np.nan,
                "bookmaker_overround_estimate": float(valid["ah_overround_estimate"].mean()) if len(valid) else np.nan,
                "mean_home_plus_away_profit_same_match": float(same_match_profit.mean()) if len(same_match_profit) else np.nan,
                "min_home_plus_away_profit_same_match": float(same_match_profit.min()) if len(same_match_profit) else np.nan,
                "max_home_plus_away_profit_same_match": float(same_match_profit.max()) if len(same_match_profit) else np.nan,
                "impossible_settlement_rows": int(len(impossible)),
            }
        )
    return pd.DataFrame(rows).sort_values(["league", "season_end_year"]).reset_index(drop=True)


def compatibility_audit(settled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (league, season), group in settled.groupby(["league", "season_end_year"], dropna=False):
        valid = group[group["valid_ah_settlement"]].copy()
        comparable = valid[valid["home_binary_target_previous"].notna() & valid["home_cover_from_settlement"].notna()].copy()
        mismatches = comparable[comparable["home_binary_target_previous"] != comparable["home_cover_from_settlement"]]
        pushes = valid[valid["home_ah_settlement_label"].eq("push")]
        rows.append(
            {
                "league": league,
                "season_end_year": int(season) if pd.notna(season) else np.nan,
                "valid_ah_settlement_rows": int(len(valid)),
                "binary_comparable_rows": int(len(comparable)),
                "push_rows_excluded_from_binary_target": int(len(pushes)),
                "cover_rows_from_settlement": int(valid["home_ah_settlement_label"].isin(["full_win", "half_win"]).sum()),
                "no_cover_rows_from_settlement": int(valid["home_ah_settlement_label"].isin(["half_loss", "full_loss"]).sum()),
                "binary_target_mismatch_rows": int(len(mismatches)),
                "binary_target_compatible": bool(len(mismatches) == 0),
                "binary_probability_value_use": "not_direct; use settlement engine and historical settled ROI for value review",
            }
        )
    return pd.DataFrame(rows).sort_values(["league", "season_end_year"]).reset_index(drop=True)


def classify(unit_tests: pd.DataFrame, real: pd.DataFrame, compatibility: pd.DataFrame) -> str:
    if not bool((unit_tests["home_passed"] & unit_tests["away_passed"]).all()):
        return "settlement_failed"
    if int(real["impossible_settlement_rows"].sum()) > 0:
        return "settlement_failed"
    if int(compatibility["binary_target_mismatch_rows"].sum()) > 0:
        return "settlement_failed"
    return "settlement_passed_ready_for_locked_value_review"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def write_outputs(unit_tests: pd.DataFrame, real: pd.DataFrame, compatibility: pd.DataFrame, classification: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unit_tests.to_csv(UNIT_TEST_PATH, index=False)
    real.to_csv(REAL_DATA_PATH, index=False)
    compatibility.to_csv(COMPATIBILITY_PATH, index=False)
    league_real = (
        real.groupby("league")
        .agg(
            matches=("matches", "sum"),
            rows_with_valid_ah_settlement=("rows_with_valid_ah_settlement", "sum"),
            missing_score_rows=("missing_score_rows", "sum"),
            missing_ah_line_rows=("missing_ah_line_rows", "sum"),
            missing_odds_rows=("missing_odds_rows", "sum"),
            average_home_ah_profit_using_AvgAHH=("average_home_ah_profit_using_AvgAHH", "mean"),
            average_away_ah_profit_using_AvgAHA=("average_away_ah_profit_using_AvgAHA", "mean"),
            bookmaker_overround_estimate=("bookmaker_overround_estimate", "mean"),
            impossible_settlement_rows=("impossible_settlement_rows", "sum"),
        )
        .reset_index()
    )
    comp_summary = (
        compatibility.groupby("league")
        .agg(
            valid_ah_settlement_rows=("valid_ah_settlement_rows", "sum"),
            push_rows_excluded_from_binary_target=("push_rows_excluded_from_binary_target", "sum"),
            binary_target_mismatch_rows=("binary_target_mismatch_rows", "sum"),
        )
        .reset_index()
    )
    lines = [
        "# Asian Handicap Settlement Engine Audit",
        "",
        f"Final classification: `{classification}`",
        "",
        "Scope: deterministic AH settlement for home and away sides using `FTHG`, `FTAG`, `AHh`, `AvgAHH`, and `AvgAHA`. No predictive models, Transfermarkt features, player features, lineups, team-name features, closing-odds features, value search, threshold optimization, betting strategies, or live betting were used.",
        "",
        "## Settlement Rules",
        "",
        "- Home margin is `FTHG - FTAG`; home adjusted margin is `home_margin + AHh`.",
        "- Away line is `-AHh`; away adjusted margin is `-home_margin - AHh`.",
        "- Quarter lines split into the adjacent half/integer lines and average the two unit outcomes.",
        "- Full win profit is `odds - 1`; half win `(odds - 1) / 2`; push `0`; half loss `-0.5`; full loss `-1`.",
        "",
        "## Synthetic Unit Tests",
        "",
        markdown_table(unit_tests, ["case_id", "AHh", "home_margin", "home_ah_settlement_label", "expected_home_label", "home_ah_profit_at_avg_odds", "expected_home_profit", "away_ah_settlement_label", "expected_away_label", "away_ah_profit_at_avg_odds", "expected_away_profit", "home_passed", "away_passed"], max_rows=40),
        "",
        "## Real Data Summary By League",
        "",
        markdown_table(league_real, ["league", "matches", "rows_with_valid_ah_settlement", "missing_score_rows", "missing_ah_line_rows", "missing_odds_rows", "average_home_ah_profit_using_AvgAHH", "average_away_ah_profit_using_AvgAHA", "bookmaker_overround_estimate", "impossible_settlement_rows"], max_rows=30),
        "",
        "## Binary Target Compatibility",
        "",
        markdown_table(comp_summary, ["league", "valid_ah_settlement_rows", "push_rows_excluded_from_binary_target", "binary_target_mismatch_rows"], max_rows=30),
        "",
        "Compatibility conclusion: the previous binary AH home-cover target is compatible with settlement labels for probability diagnostics: `full_win` and `half_win` map to cover, `half_loss` and `full_loss` map to no-cover, and pushes are excluded. Binary predictive probabilities should not be used as direct payout returns; any later value review must use this settlement engine and settled historical ROI.",
        "",
        "No confirmed edge is claimed. No value review was run.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    unit_tests = run_unit_tests()
    raw = load_matches()
    settled = add_settlement_columns(raw)
    real = real_data_audit(settled)
    compatibility = compatibility_audit(settled)
    classification = classify(unit_tests, real, compatibility)
    write_outputs(unit_tests, real, compatibility, classification)
    print(
        {
            "unit_test_rows": len(unit_tests),
            "real_data_rows": len(real),
            "compatibility_rows": len(compatibility),
            "classification": classification,
        }
    )


if __name__ == "__main__":
    main()
