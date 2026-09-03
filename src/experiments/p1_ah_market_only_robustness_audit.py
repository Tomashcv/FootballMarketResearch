from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import asian_profit


INPUT_PATH = Path("data/processed/P1/P1_matches.csv")
REPORT_PATH = Path("outputs/reports/p1_ah_market_only_robustness_audit.md")
RULE_RESULTS_PATH = Path("outputs/reports/p1_ah_market_only_rule_results.csv")
NESTED_RESULTS_PATH = Path("outputs/reports/p1_ah_market_only_nested_results.csv")
ROBUSTNESS_PATH = Path("outputs/reports/p1_ah_market_only_robustness_checks.csv")

BIG_THREE = {"Benfica", "Porto", "Sp Lisbon"}
BIG_FOUR = {"Benfica", "Porto", "Sp Lisbon", "Sp Braga"}
RANDOM_ITERATIONS = 500


@dataclass(frozen=True)
class Rule:
    name: str
    max_home_line: float
    min_away_odds: float | None = None


RULES = [
    Rule("away_ah_big_home_favourite_all", -1.00, None),
    Rule("away_ah_big_home_favourite_away_odds_ge_1_80", -1.00, 1.80),
    Rule("away_ah_big_home_favourite_away_odds_ge_1_85", -1.00, 1.85),
    Rule("away_ah_big_home_favourite_away_odds_ge_1_90", -1.00, 1.90),
    Rule("away_ah_big_home_favourite_home_line_le_minus_1_25", -1.25, None),
    Rule("away_ah_big_home_favourite_home_line_le_minus_1_5", -1.50, None),
    Rule("frozen_e0_rule_away_odds_ge_1_85", -1.00, 1.85),
]


def clean_team(value: object) -> str:
    return str(value).strip()


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_PATH, low_memory=False)
    required = ["Date", "HomeTeam", "AwayTeam", "season_end_year", "FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required P1 columns: {missing}")
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["season_end_year"] = pd.to_numeric(frame["season_end_year"], errors="coerce")
    numeric = ["FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA", "AHCh", "AvgCAHH", "AvgCAHA"]
    for column in numeric:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["HomeTeam"] = frame["HomeTeam"].map(clean_team)
    frame["AwayTeam"] = frame["AwayTeam"].map(clean_team)
    frame = frame.dropna(subset=required).copy()
    frame = frame[(frame["AvgAHH"] > 1.0) & (frame["AvgAHA"] > 1.0)].copy()
    frame["season_end_year"] = frame["season_end_year"].astype(int)
    frame["home_margin"] = frame["FTHG"] - frame["FTAG"]
    frame["away_margin"] = frame["FTAG"] - frame["FTHG"]
    frame["away_handicap"] = -frame["AHh"]
    frame["home_handicap"] = frame["AHh"]
    frame["away_ah_odds"] = frame["AvgAHA"]
    frame["home_ah_odds"] = frame["AvgAHH"]
    frame["profit"] = [
        asian_profit(margin, handicap, odds)
        for margin, handicap, odds in zip(frame["away_margin"], frame["away_handicap"], frame["away_ah_odds"])
    ]
    frame["opposite_profit"] = [
        asian_profit(margin, handicap, odds)
        for margin, handicap, odds in zip(frame["home_margin"], frame["home_handicap"], frame["home_ah_odds"])
    ]
    frame["open_away_no_vig_probability"] = no_vig_probability(frame["AvgAHA"], frame["AvgAHH"])
    frame["open_home_no_vig_probability"] = no_vig_probability(frame["AvgAHH"], frame["AvgAHA"])
    if {"AvgCAHA", "AvgCAHH", "AHCh"}.issubset(frame.columns):
        has_close = frame[["AvgCAHA", "AvgCAHH", "AHCh"]].notna().all(axis=1)
        frame["close_away_no_vig_probability"] = np.where(
            has_close,
            no_vig_probability(frame["AvgCAHA"], frame["AvgCAHH"]),
            np.nan,
        )
        frame["close_home_no_vig_probability"] = np.where(
            has_close,
            no_vig_probability(frame["AvgCAHH"], frame["AvgCAHA"]),
            np.nan,
        )
        frame["clv_pp"] = frame["close_away_no_vig_probability"] - frame["open_away_no_vig_probability"]
        frame["clv_positive"] = frame["clv_pp"] > 0
        frame["home_clv_pp"] = frame["close_home_no_vig_probability"] - frame["open_home_no_vig_probability"]
        frame["home_clv_positive"] = frame["home_clv_pp"] > 0
        frame["closing_line_move_to_away"] = np.where(has_close, -frame["AHCh"] - frame["away_handicap"], np.nan)
    else:
        frame["close_away_no_vig_probability"] = np.nan
        frame["close_home_no_vig_probability"] = np.nan
        frame["clv_pp"] = np.nan
        frame["clv_positive"] = np.nan
        frame["home_clv_pp"] = np.nan
        frame["home_clv_positive"] = np.nan
        frame["closing_line_move_to_away"] = np.nan
    frame = frame.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    frame["match_id"] = np.arange(len(frame))
    return frame


def no_vig_probability(side_odds: pd.Series, other_odds: pd.Series) -> pd.Series:
    side = 1.0 / side_odds.astype(float)
    other = 1.0 / other_odds.astype(float)
    return side / (side + other)


def select_rule(frame: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    mask = frame["AHh"] <= rule.max_home_line
    if rule.min_away_odds is not None:
        mask &= frame["AvgAHA"] >= rule.min_away_odds
    selected = frame[mask].copy()
    selected["rule"] = rule.name
    selected["rule_max_home_line"] = rule.max_home_line
    selected["rule_min_away_odds"] = rule.min_away_odds if rule.min_away_odds is not None else np.nan
    return selected


def z_score(profits: pd.Series) -> float:
    profits = profits.astype(float)
    if len(profits) < 2:
        return 0.0
    std = float(profits.std(ddof=1))
    if std == 0.0 or math.isnan(std):
        return 0.0
    return float(profits.mean() / (std / math.sqrt(len(profits))))


def max_drawdown(frame: pd.DataFrame, profit_column: str = "profit") -> float:
    if frame.empty:
        return 0.0
    ordered = frame.sort_values(["Date", "HomeTeam", "AwayTeam"])
    cumulative = ordered[profit_column].astype(float).cumsum()
    running_max = cumulative.cummax()
    return float((running_max - cumulative).max())


def hhi(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    shares = series.value_counts(normalize=True)
    return float((shares * shares).sum())


def summarize(
    frame: pd.DataFrame,
    profit_column: str = "profit",
    odds_column: str = "away_ah_odds",
    clv_column: str = "clv_pp",
    clv_positive_column: str = "clv_positive",
) -> dict[str, object]:
    if frame.empty:
        return {
            "bets": 0,
            "profit": 0.0,
            "roi": 0.0,
            "z_score": 0.0,
            "max_drawdown": 0.0,
            "average_odds": np.nan,
            "wins": 0,
            "pushes": 0,
            "losses": 0,
            "half_wins": 0,
            "half_losses": 0,
            "average_clv": np.nan,
            "clv_positive_pct": np.nan,
            "home_team_hhi": 0.0,
            "away_team_hhi": 0.0,
            "top_home_team_share": np.nan,
            "top_away_team_share": np.nan,
        }
    profits = frame[profit_column].astype(float)
    home_counts = frame["HomeTeam"].value_counts(normalize=True)
    away_counts = frame["AwayTeam"].value_counts(normalize=True)
    return {
        "bets": int(len(frame)),
        "profit": float(profits.sum()),
        "roi": float(profits.mean()),
        "z_score": z_score(profits),
        "max_drawdown": max_drawdown(frame, profit_column),
        "average_odds": float(frame[odds_column].mean()),
        "wins": int((profits > 0.51).sum()),
        "pushes": int((profits == 0.0).sum()),
        "losses": int((profits == -1.0).sum()),
        "half_wins": int(((profits > 0.0) & (profits <= 0.51)).sum()),
        "half_losses": int((profits == -0.5).sum()),
        "average_clv": float(frame[clv_column].mean()) if frame[clv_column].notna().any() else np.nan,
        "clv_positive_pct": float(frame[clv_positive_column].mean()) if frame[clv_positive_column].notna().any() else np.nan,
        "home_team_hhi": hhi(frame["HomeTeam"]),
        "away_team_hhi": hhi(frame["AwayTeam"]),
        "top_home_team_share": float(home_counts.iloc[0]) if len(home_counts) else np.nan,
        "top_away_team_share": float(away_counts.iloc[0]) if len(away_counts) else np.nan,
    }


def fixed_rule_results(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        selected = select_rule(frame, rule)
        for season, group in selected.groupby("season_end_year"):
            row = {
                "scope": "fixed_rule_by_season",
                "rule": rule.name,
                "season_end_year": int(season),
            }
            row.update(summarize(group))
            rows.append(row)
        overall = {"scope": "fixed_rule_overall", "rule": rule.name, "season_end_year": "all"}
        overall.update(summarize(selected))
        rows.append(overall)
    return pd.DataFrame(rows)


def select_nested_rule(validation: pd.DataFrame) -> tuple[Rule | None, pd.DataFrame]:
    candidates = []
    for rule in RULES:
        selected = select_rule(validation, rule)
        summary = summarize(selected)
        seasons = selected.groupby("season_end_year")["profit"].mean() if len(selected) else pd.Series(dtype=float)
        row = {
            "rule": rule.name,
            "validation_bets": summary["bets"],
            "validation_profit": summary["profit"],
            "validation_roi": summary["roi"],
            "validation_z_score": summary["z_score"],
            "validation_positive_seasons": int((seasons > 0).sum()),
            "validation_min_season_roi": float(seasons.min()) if len(seasons) else np.nan,
            "validation_average_clv": summary["average_clv"],
            "validation_clv_positive_pct": summary["clv_positive_pct"],
        }
        candidates.append(row)
    candidate_frame = pd.DataFrame(candidates)
    eligible = candidate_frame[
        (candidate_frame["validation_bets"] >= 50)
        & (candidate_frame["validation_roi"] > 0)
    ].copy()
    if eligible.empty:
        return None, candidate_frame
    eligible = eligible.sort_values(
        ["validation_positive_seasons", "validation_roi", "validation_z_score", "validation_bets"],
        ascending=[False, False, False, False],
    )
    selected_name = str(eligible.iloc[0]["rule"])
    return next(rule for rule in RULES if rule.name == selected_name), candidate_frame


def nested_results(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = sorted(frame["season_end_year"].unique())
    for test_year in years:
        validation = frame[frame["season_end_year"] < test_year].copy()
        test = frame[frame["season_end_year"] == test_year].copy()
        if validation["season_end_year"].nunique() < 2:
            continue
        selected_rule, candidates = select_nested_rule(validation)
        if selected_rule is None:
            rows.append(
                {
                    "test_year": int(test_year),
                    "selected_rule": "none",
                    "validation_years": ",".join(map(str, sorted(validation["season_end_year"].unique()))),
                    "selection_status": "no_prior_positive_rule_with_min_50_bets",
                    "test_bets": 0,
                }
            )
            continue
        selected_test = select_rule(test, selected_rule)
        summary = summarize(selected_test)
        selected_validation = candidates[candidates["rule"].eq(selected_rule.name)].iloc[0].to_dict()
        row = {
            "test_year": int(test_year),
            "selected_rule": selected_rule.name,
            "validation_years": ",".join(map(str, sorted(validation["season_end_year"].unique()))),
            "selection_status": "selected_on_prior_seasons_only",
            "validation_bets": selected_validation["validation_bets"],
            "validation_profit": selected_validation["validation_profit"],
            "validation_roi": selected_validation["validation_roi"],
            "validation_z_score": selected_validation["validation_z_score"],
            "validation_positive_seasons": selected_validation["validation_positive_seasons"],
            "test_bets": summary["bets"],
            "test_profit": summary["profit"],
            "test_roi": summary["roi"],
            "test_z_score": summary["z_score"],
            "test_max_drawdown": summary["max_drawdown"],
            "test_average_odds": summary["average_odds"],
            "test_wins": summary["wins"],
            "test_pushes": summary["pushes"],
            "test_losses": summary["losses"],
            "test_half_wins": summary["half_wins"],
            "test_half_losses": summary["half_losses"],
            "test_average_clv": summary["average_clv"],
            "test_clv_positive_pct": summary["clv_positive_pct"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_rule_subset(frame: pd.DataFrame, rule: Rule, check: str, subset: pd.DataFrame) -> dict[str, object]:
    selected = select_rule(subset, rule)
    row = {
        "rule": rule.name,
        "check": check,
    }
    row.update(summarize(selected))
    return row


def random_same_size_check(frame: pd.DataFrame, selected: pd.DataFrame, seed: int) -> dict[str, object]:
    if selected.empty:
        return {
            "random_iterations": RANDOM_ITERATIONS,
            "random_mean_roi": np.nan,
            "random_p95_roi": np.nan,
            "random_prob_profit_ge_actual": np.nan,
            "random_prob_roi_ge_actual": np.nan,
        }
    rng = np.random.default_rng(seed)
    profits = []
    universe_by_season = {season: group for season, group in frame.groupby("season_end_year")}
    selected_sizes = selected.groupby("season_end_year").size().to_dict()
    actual_profit = float(selected["profit"].sum())
    actual_roi = float(selected["profit"].mean())
    for _ in range(RANDOM_ITERATIONS):
        pieces = []
        for season, size in selected_sizes.items():
            universe = universe_by_season.get(season)
            if universe is None or len(universe) < size:
                continue
            sampled_index = rng.choice(universe.index.to_numpy(), size=size, replace=False)
            pieces.append(frame.loc[sampled_index, "profit"])
        if pieces:
            sample_profits = pd.concat(pieces).astype(float)
            profits.append((float(sample_profits.sum()), float(sample_profits.mean())))
    if not profits:
        return {
            "random_iterations": RANDOM_ITERATIONS,
            "random_mean_roi": np.nan,
            "random_p95_roi": np.nan,
            "random_prob_profit_ge_actual": np.nan,
            "random_prob_roi_ge_actual": np.nan,
        }
    random_frame = pd.DataFrame(profits, columns=["profit", "roi"])
    return {
        "random_iterations": RANDOM_ITERATIONS,
        "random_mean_roi": float(random_frame["roi"].mean()),
        "random_p95_roi": float(random_frame["roi"].quantile(0.95)),
        "random_prob_profit_ge_actual": float((random_frame["profit"] >= actual_profit).mean()),
        "random_prob_roi_ge_actual": float((random_frame["roi"] >= actual_roi).mean()),
    }


def robustness_checks(frame: pd.DataFrame, rule_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    best_profit_season_by_rule = {}
    seasonal = rule_results[rule_results["scope"].eq("fixed_rule_by_season")].copy()
    for rule_name, group in seasonal.groupby("rule"):
        if not group.empty:
            best_profit_season_by_rule[rule_name] = int(group.sort_values("profit", ascending=False).iloc[0]["season_end_year"])
    for idx, rule in enumerate(RULES):
        selected = select_rule(frame, rule)
        rows.append(summarize_rule_subset(frame, rule, "overall", frame))
        rows.append(summarize_rule_subset(frame, rule, "exclude_benfica_porto_sporting_braga", frame[~frame["HomeTeam"].isin(BIG_FOUR) & ~frame["AwayTeam"].isin(BIG_FOUR)]))
        rows.append(summarize_rule_subset(frame, rule, "exclude_benfica_porto_sporting", frame[~frame["HomeTeam"].isin(BIG_THREE) & ~frame["AwayTeam"].isin(BIG_THREE)]))
        rows.append(summarize_rule_subset(frame, rule, "exclude_2026", frame[frame["season_end_year"] != 2026]))
        best_season = best_profit_season_by_rule.get(rule.name)
        if best_season is not None:
            rows.append(summarize_rule_subset(frame, rule, f"exclude_best_profit_season_{best_season}", frame[frame["season_end_year"] != best_season]))
        opposite = {
            "rule": rule.name,
            "check": "opposite_home_ah_same_matches",
        }
        opposite.update(
            summarize(
                selected,
                profit_column="opposite_profit",
                odds_column="home_ah_odds",
                clv_column="home_clv_pp",
                clv_positive_column="home_clv_positive",
            )
        )
        rows.append(opposite)
        random_row = {
            "rule": rule.name,
            "check": "random_same_size_ah_selections",
        }
        random_row.update(summarize(selected))
        random_row.update(random_same_size_check(frame, selected, seed=20260630 + idx))
        rows.append(random_row)
        concentration = {
            "rule": rule.name,
            "check": "team_concentration",
            "bets": len(selected),
            "home_team_hhi": hhi(selected["HomeTeam"]),
            "away_team_hhi": hhi(selected["AwayTeam"]),
            "top_home_teams": "; ".join(f"{team}:{count}" for team, count in selected["HomeTeam"].value_counts().head(5).items()),
            "top_away_teams": "; ".join(f"{team}:{count}" for team, count in selected["AwayTeam"].value_counts().head(5).items()),
        }
        rows.append(concentration)
    return pd.DataFrame(rows)


def classify(rule_results: pd.DataFrame, nested: pd.DataFrame, robustness: pd.DataFrame) -> tuple[str, list[str]]:
    failures = []
    overall = rule_results[rule_results["scope"].eq("fixed_rule_overall")].copy()
    viable = overall[
        (overall["bets"] >= 50)
        & (overall["roi"] > 0)
        & (overall["z_score"] >= 1.5)
        & (overall["average_clv"] > 0)
        & (overall["clv_positive_pct"] >= 0.52)
    ].copy()
    if viable.empty:
        failures.append("No fixed rule cleared bets>=50, ROI>0, z>=1.5, Avg CLV>0, and CLV+>=52%.")
    nested_tests = nested[nested.get("selection_status", pd.Series(dtype=str)).eq("selected_on_prior_seasons_only")].copy()
    if nested_tests.empty or float(nested_tests.get("test_profit", pd.Series(dtype=float)).sum()) <= 0:
        failures.append("Nested prior-season rule selection did not produce positive aggregate test profit.")
    elif z_score(nested_tests["test_profit"].astype(float)) < 1.5:
        failures.append("Nested selected-rule season profits did not clear z>=1.5.")
    robust_overall = robustness[robustness["check"].isin(["exclude_benfica_porto_sporting_braga", "exclude_benfica_porto_sporting", "exclude_2026"])].copy()
    if not viable.empty:
        viable_rules = set(viable["rule"])
        troubled = robust_overall[robust_overall["rule"].isin(viable_rules) & (robust_overall["roi"] <= 0)]
        if not troubled.empty:
            failures.append("At least one viable fixed rule failed a core exclusion robustness check.")
    if failures:
        if not overall[(overall["bets"] >= 50) & (overall["roi"] > 0)].empty:
            return "research_only", failures
        return "reject", failures
    if nested_tests["test_year"].nunique() >= 2:
        return "shadow_candidate", ["Historical checks passed, but this is still not a confirmed edge or value review."]
    return "research_only", ["Insufficient nested test seasons for shadow promotion."]


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.loc[:, [column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def write_report(
    frame: pd.DataFrame,
    rule_results: pd.DataFrame,
    nested: pd.DataFrame,
    robustness: pd.DataFrame,
    classification: str,
    failures: list[str],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    coverage = frame.groupby("season_end_year").agg(
        matches=("match_id", "size"),
        big_home_fav_rows=("AHh", lambda s: int((s <= -1.0).sum())),
        avg_away_odds=("AvgAHA", "mean"),
        closing_ah_coverage=("clv_pp", lambda s: float(s.notna().mean())),
    ).reset_index()
    overall = rule_results[rule_results["scope"].eq("fixed_rule_overall")].sort_values("profit", ascending=False)
    seasonal = rule_results[rule_results["scope"].eq("fixed_rule_by_season")]
    best_by_rule = seasonal.sort_values(["rule", "profit"], ascending=[True, False]).groupby("rule").head(1)
    lines = [
        "# P1 AH Market-Only Robustness Audit",
        "",
        "Scope: P1/Liga Portugal away Asian Handicap in big home-favourite spots. Transfermarkt, player features, lineups, and live betting are not used.",
        "",
        "Bet-time-safe selection uses only `AHh`, `AvgAHH`, and `AvgAHA`. Closing AH columns are diagnostic only after selection.",
        "",
        f"Final classification: `{classification}`",
        "",
        "## Coverage",
        "",
        markdown_table(coverage, ["season_end_year", "matches", "big_home_fav_rows", "avg_away_odds", "closing_ah_coverage"], max_rows=20),
        "",
        "## Fixed Rules Overall",
        "",
        markdown_table(
            overall,
            ["rule", "bets", "profit", "roi", "z_score", "max_drawdown", "average_odds", "wins", "pushes", "losses", "average_clv", "clv_positive_pct", "home_team_hhi", "away_team_hhi"],
            max_rows=20,
        ),
        "",
        "## Best Single Season By Rule",
        "",
        markdown_table(best_by_rule, ["rule", "season_end_year", "bets", "profit", "roi", "z_score", "average_clv", "clv_positive_pct"], max_rows=20),
        "",
        "## Nested Prior-Season Selection",
        "",
        markdown_table(
            nested,
            ["test_year", "selected_rule", "selection_status", "validation_bets", "validation_roi", "validation_z_score", "test_bets", "test_profit", "test_roi", "test_z_score", "test_average_clv", "test_clv_positive_pct"],
            max_rows=30,
        ),
        "",
        "## Robustness Highlights",
        "",
        markdown_table(
            robustness[robustness["check"].isin(["overall", "exclude_benfica_porto_sporting_braga", "exclude_benfica_porto_sporting", "exclude_2026", "opposite_home_ah_same_matches", "random_same_size_ah_selections"])],
            ["rule", "check", "bets", "profit", "roi", "z_score", "average_clv", "clv_positive_pct", "random_mean_roi", "random_p95_roi", "random_prob_profit_ge_actual"],
            max_rows=80,
        ),
        "",
        "## Gate Assessment",
        "",
    ]
    lines.extend(f"- {failure}" for failure in failures)
    lines.extend(
        [
            "",
            "No confirmed edge is claimed. No value review or betting strategy deployment was run.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame = load_data()
    rule_results = fixed_rule_results(frame)
    nested = nested_results(frame)
    robustness = robustness_checks(frame, rule_results)
    classification, failures = classify(rule_results, nested, robustness)
    RULE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rule_results.to_csv(RULE_RESULTS_PATH, index=False)
    nested.to_csv(NESTED_RESULTS_PATH, index=False)
    robustness.to_csv(ROBUSTNESS_PATH, index=False)
    write_report(frame, rule_results, nested, robustness, classification, failures)
    print(
        {
            "rules": len(RULES),
            "rule_result_rows": len(rule_results),
            "nested_rows": len(nested),
            "robustness_rows": len(robustness),
            "classification": classification,
        }
    )


if __name__ == "__main__":
    main()
