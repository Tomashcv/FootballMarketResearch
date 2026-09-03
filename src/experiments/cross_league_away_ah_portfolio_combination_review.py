from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


REPORT_PATH = Path("outputs/reports/cross_league_away_ah_portfolio_combination_review.md")
SUMMARY_PATH = Path("outputs/reports/cross_league_away_ah_portfolio_combination_summary.csv")
OVERLAP_PATH = Path("outputs/reports/cross_league_away_ah_portfolio_combination_overlap.csv")
FAILURES_PATH = Path("outputs/reports/cross_league_away_ah_portfolio_combination_failures.csv")
DETAIL_DIR = Path("outputs/cross_league_away_ah_portfolio_combination")

SWEEP_DIR = Path("outputs/cross_league_away_ah_advanced_model_sweep")
SELECTED_BETS_PATH = SWEEP_DIR / "selected_bets.csv"
SWEEP_NEGATIVE_CONTROLS_PATH = SWEEP_DIR / "negative_controls.csv"

LEAGUES = ["I1", "SP1", "D1", "F1"]
BASE_CANDIDATES = [
    "away_odds_ge_1_85",
    "away_odds_ge_1_85_plus_memory_knn_profit",
    "logistic_market_residual",
    "xgboost_market_residual",
    "numpy_mlp_market_residual",
    "ft_transformer_market_residual",
    "deep_cross_network_market_residual",
    "wide_deep_market_residual",
    "bag_last5_pooled_market_residual",
    "sequence_transformer_n5_no_seq_odds_market_residual",
]
MODEL_CANDIDATES = [
    candidate
    for candidate in BASE_CANDIDATES
    if candidate not in {"away_odds_ge_1_85", "away_odds_ge_1_85_plus_memory_knn_profit"}
]
SEQUENCE_RECENT_CANDIDATES = [
    "bag_last5_pooled_market_residual",
    "sequence_transformer_n5_no_seq_odds_market_residual",
]
COMBINATIONS = [
    "union_all_positive",
    "intersection_2plus",
    "intersection_3plus",
    "rule_plus_model_confirm",
    "memory_plus_model_confirm",
    "diversified_bucket_portfolio",
]


def match_key(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        + "|"
        + frame["league"].astype(str)
        + "|"
        + frame["HomeTeam"].astype(str)
        + "|"
        + frame["AwayTeam"].astype(str)
    )


def load_base_selected_bets() -> pd.DataFrame:
    frame = pd.read_csv(SELECTED_BETS_PATH, low_memory=False)
    frame = frame[frame["league"].isin(LEAGUES) & frame["strategy"].isin(BASE_CANDIDATES)].copy()
    frame["match_key"] = match_key(frame)
    frame["season_end_year"] = pd.to_numeric(frame["season_end_year"], errors="coerce").astype(int)
    return frame


def build_signal_panel(selected: pd.DataFrame) -> pd.DataFrame:
    base = (
        selected.sort_values(["league", "Date", "HomeTeam", "AwayTeam", "strategy"])
        .drop_duplicates("match_key")
        .copy()
    )
    for candidate in BASE_CANDIDATES:
        keys = set(selected.loc[selected["strategy"].eq(candidate), "match_key"])
        base[f"signal_{candidate}"] = base["match_key"].isin(keys)
    base["candidate_confirmations"] = base[[f"signal_{candidate}" for candidate in BASE_CANDIDATES]].sum(axis=1).astype(int)
    base["model_confirmations"] = base[[f"signal_{candidate}" for candidate in MODEL_CANDIDATES]].sum(axis=1).astype(int)
    return base.reset_index(drop=True)


def select_combination(panel: pd.DataFrame, combination: str, *, permute_signals: bool = False) -> pd.DataFrame:
    working = panel.copy()
    signal_columns = [f"signal_{candidate}" for candidate in BASE_CANDIDATES]
    if permute_signals:
        rng = np.random.default_rng(118733)
        group_columns: list[str] | str = ["league", "season_end_year"] if working["league"].nunique() > 1 else "season_end_year"
        for _, idx in working.groupby(group_columns).groups.items():
            for column in signal_columns:
                values = working.loc[idx, column].to_numpy(copy=True)
                working.loc[idx, column] = values[rng.permutation(len(values))]
        working["candidate_confirmations"] = working[signal_columns].sum(axis=1).astype(int)
        working["model_confirmations"] = working[[f"signal_{candidate}" for candidate in MODEL_CANDIDATES]].sum(axis=1).astype(int)

    if combination == "union_all_positive":
        selected = working[working["candidate_confirmations"] >= 1].copy()
    elif combination == "intersection_2plus":
        selected = working[working["candidate_confirmations"] >= 2].copy()
    elif combination == "intersection_3plus":
        selected = working[working["candidate_confirmations"] >= 3].copy()
    elif combination == "rule_plus_model_confirm":
        selected = working[working["signal_away_odds_ge_1_85"] & (working["model_confirmations"] >= 1)].copy()
    elif combination == "memory_plus_model_confirm":
        selected = working[
            working["signal_away_odds_ge_1_85_plus_memory_knn_profit"] & (working["model_confirmations"] >= 1)
        ].copy()
    elif combination == "diversified_bucket_portfolio":
        sequence_recent = working[[f"signal_{candidate}" for candidate in SEQUENCE_RECENT_CANDIDATES]].sum(axis=1)
        model_count = working[[f"signal_{candidate}" for candidate in MODEL_CANDIDATES]].sum(axis=1)
        mask = (
            working["signal_away_odds_ge_1_85"]
            | working["signal_away_odds_ge_1_85_plus_memory_knn_profit"]
            | (model_count >= 2)
            | (sequence_recent >= 1)
        )
        selected = working[mask].copy()
        selected["portfolio_bucket"] = np.select(
            [
                selected["signal_away_odds_ge_1_85_plus_memory_knn_profit"],
                selected[[f"signal_{candidate}" for candidate in SEQUENCE_RECENT_CANDIDATES]].sum(axis=1) >= 1,
                selected[[f"signal_{candidate}" for candidate in MODEL_CANDIDATES]].sum(axis=1) >= 2,
                selected["signal_away_odds_ge_1_85"],
            ],
            ["memory_confirmation", "sequence_recent_form_confirmation", "model_confirmation", "pure_market_rule"],
            default="unbucketed",
        )
    else:
        raise ValueError(f"Unknown combination: {combination}")
    selected["strategy"] = combination
    return selected.drop_duplicates("match_key").copy()


def random_portfolio_like(panel: pd.DataFrame, selected: pd.DataFrame, strategy: str) -> pd.DataFrame:
    rng = np.random.default_rng(82231)
    frames = []
    group_columns: list[str] | str = ["league", "season_end_year"] if panel["league"].nunique() > 1 else "season_end_year"
    for group_key, group in panel.groupby(group_columns):
        if isinstance(group_key, tuple):
            league, season = group_key
            count = len(selected[(selected["league"].eq(league)) & (selected["season_end_year"].eq(season))])
        else:
            count = len(selected[selected["season_end_year"].eq(group_key)])
        if count <= 0:
            continue
        frames.append(group.sample(n=min(count, len(group)), random_state=int(rng.integers(0, 1_000_000))).copy())
    output = pd.concat(frames, ignore_index=True, sort=False) if frames else panel.iloc[0:0].copy()
    output["strategy"] = strategy
    return output


def row_for_bets(scope: str, strategy: str, bets: pd.DataFrame, kind: str) -> dict:
    row = advanced.overall_row(strategy, bets, "portfolio", "portfolio")
    row["league"] = scope
    row["kind"] = kind
    row["unique_matches"] = int(bets["match_key"].nunique()) if len(bets) else 0
    row["avg_candidate_confirmations"] = (
        float(bets["candidate_confirmations"].mean()) if len(bets) and "candidate_confirmations" in bets.columns else pd.NA
    )
    row["max_candidate_confirmations"] = (
        int(bets["candidate_confirmations"].max()) if len(bets) and "candidate_confirmations" in bets.columns else pd.NA
    )
    return row


def season_rows(scope: str, strategy: str, bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in bets.groupby("season_end_year"):
        row = row_for_bets(scope, strategy, group, "season")
        row["season"] = int(season)
        rows.append(row)
    return pd.DataFrame(rows)


def league_contribution_rows(strategy: str, bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for league, group in bets.groupby("league"):
        row = row_for_bets("GLOBAL", strategy, group, "global_league_contribution")
        row["contribution_league"] = league
        rows.append(row)
    return pd.DataFrame(rows)


def exclusion_rows(scope: str, strategy: str, bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bets.empty:
        return pd.DataFrame(rows)
    seasonal = season_rows(scope, strategy, bets)
    best_season = int(seasonal.sort_values("profit", ascending=False).iloc[0]["season"])
    for season in sorted(bets["season_end_year"].unique()):
        row = row_for_bets(scope, strategy, bets[bets["season_end_year"].ne(season)].copy(), "season_exclusion")
        row["exclusion_type"] = "exclude_each_season"
        row["excluded"] = int(season)
        rows.append(row)
    row = row_for_bets(scope, strategy, bets[bets["season_end_year"].ne(best_season)].copy(), "season_exclusion")
    row["exclusion_type"] = "exclude_best_profit_season"
    row["excluded"] = best_season
    rows.append(row)
    counts = bets["HomeTeam"].value_counts()
    for count in [1, 2, 3]:
        teams = list(counts.head(count).index)
        row = row_for_bets(scope, strategy, bets[~bets["HomeTeam"].isin(teams)].copy(), "home_exclusion")
        row["exclusion_type"] = f"exclude_top{count}_home"
        row["excluded"] = ", ".join(teams)
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_rows(scope: str, strategy: str, bets: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected_keys = set(bets["match_key"]) if len(bets) else set()
    for candidate in BASE_CANDIDATES:
        candidate_keys = set(panel.loc[panel[f"signal_{candidate}"], "match_key"])
        common = selected_keys & candidate_keys
        rows.append(
            {
                "league": scope,
                "strategy": strategy,
                "base_candidate": candidate,
                "portfolio_bets": len(selected_keys),
                "candidate_bets": len(candidate_keys),
                "overlap_bets": len(common),
                "share_of_portfolio": len(common) / len(selected_keys) if selected_keys else pd.NA,
                "share_of_candidate": len(common) / len(candidate_keys) if candidate_keys else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def control_success(controls: pd.DataFrame) -> bool:
    if controls.empty:
        return False
    return bool(((controls["profit"] > 0) & (controls["roi"] > 0) & (controls["avg_clv_pp"] > 0)).any())


def benchmark_row(scope: str, panel: pd.DataFrame) -> dict:
    memory = panel[panel["signal_away_odds_ge_1_85_plus_memory_knn_profit"]].copy()
    if len(memory):
        return row_for_bets(scope, "away_odds_ge_1_85_plus_memory_knn_profit", memory, "base")
    rule = panel[panel["signal_away_odds_ge_1_85"]].copy()
    return row_for_bets(scope, "away_odds_ge_1_85", rule, "base")


def classify(
    row: pd.Series,
    exclusions: pd.DataFrame,
    controls: pd.DataFrame,
    benchmark: dict,
    *,
    global_scope: bool,
    league_contribution: pd.DataFrame | None = None,
) -> tuple[str, str, list[str], int, int]:
    min_bets = 200 if global_scope else 80
    no_2025 = exclusions[(exclusions["exclusion_type"].eq("exclude_each_season")) & (exclusions["excluded"].astype(str).eq("2025"))]
    best_ex = exclusions[exclusions["exclusion_type"].eq("exclude_best_profit_season")]
    top1 = exclusions[exclusions["exclusion_type"].eq("exclude_top1_home")]
    top3 = exclusions[exclusions["exclusion_type"].eq("exclude_top3_home")]
    league_dependency = False
    if global_scope and league_contribution is not None and len(league_contribution):
        positive_profit = max(float(row["profit"]), 0.0)
        max_league_profit = pd.to_numeric(league_contribution["profit"], errors="coerce").max()
        league_dependency = bool(positive_profit > 0 and max_league_profit / positive_profit > 0.70)
    gates = {
        "min_bets": row["bets"] >= min_bets,
        "positive_roi": row["roi"] > 0,
        "z_ge_1_5": row["z_score"] >= 1.5,
        "positive_clv": row["avg_clv_pp"] > 0,
        "clv_plus_ge_52": row["clv_positive_rate"] >= 0.52,
        "positive_roi_without_2025": bool(len(no_2025) and float(no_2025.iloc[0]["roi"]) > 0),
        "positive_roi_excluding_best_profit_season": bool(len(best_ex) and float(best_ex.iloc[0]["roi"]) > 0),
        "positive_roi_excluding_top1_home": bool(len(top1) and float(top1.iloc[0]["roi"]) > 0),
        "not_destroyed_excluding_top3_home": bool(len(top3) and float(top3.iloc[0]["roi"]) > -0.05),
        "negative_controls_fail": not control_success(controls),
        "beats_simple_league_benchmark_robustness": (
            row["profit"] > benchmark["profit"]
            and row["z_score"] > benchmark["z_score"]
            and row["max_drawdown"] <= max(float(benchmark["max_drawdown"]), 1e-9)
        ),
        "global_not_dependent_on_one_league": (not global_scope) or (not league_dependency),
    }
    failed = [name for name, passed in gates.items() if not passed]
    passed = len(gates) - len(failed)
    if row["profit"] <= 0 or row["roi"] <= 0:
        return "reject", "Failed gates: " + "; ".join(failed), failed, passed, len(gates)
    if passed >= len(gates) - 2 and gates["negative_controls_fail"] and gates["positive_clv"]:
        return "paper challenger candidate pending locked falsification", "Passed most gates; needs locked falsification.", failed, passed, len(gates)
    return "research only", "Failed gates: " + "; ".join(failed), failed, passed, len(gates)


def stored_negative_controls(scope: str) -> pd.DataFrame:
    controls = pd.read_csv(SWEEP_NEGATIVE_CONTROLS_PATH)
    if scope != "GLOBAL":
        controls = controls[controls["league"].eq(scope)].copy()
    else:
        controls = controls[controls["league"].isin(LEAGUES)].copy()
        rows = []
        for strategy, group in controls.groupby("strategy"):
            row = {
                "league": "GLOBAL",
                "strategy": f"{strategy}__stored_sweep_control",
                "kind": "negative_control",
                "bets": group["bets"].sum(),
                "profit": group["profit"].sum(),
                "roi": group["profit"].sum() / group["bets"].sum() if group["bets"].sum() else 0.0,
                "z_score": np.nan,
                "max_drawdown": group["max_drawdown"].max(),
                "avg_clv_pp": np.average(group["avg_clv_pp"], weights=group["bets"]) if group["bets"].sum() else np.nan,
                "clv_positive_rate": np.average(group["clv_positive_rate"], weights=group["bets"]) if group["bets"].sum() else np.nan,
                "top3_home_bet_share": np.nan,
                "home_hhi_bets": np.nan,
                "unique_matches": group["bets"].sum(),
                "avg_candidate_confirmations": pd.NA,
                "max_candidate_confirmations": pd.NA,
            }
            rows.append(row)
        return pd.DataFrame(rows)
    controls["strategy"] = controls["strategy"] + "__stored_sweep_control"
    controls["unique_matches"] = controls["bets"]
    controls["avg_candidate_confirmations"] = pd.NA
    controls["max_candidate_confirmations"] = pd.NA
    return controls


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    return advanced.markdown_table(frame, columns, headers)


def run_scope(scope: str, panel: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    summary_rows: list[dict] = []
    seasonal_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []
    overlap_frames: list[pd.DataFrame] = []
    control_rows: list[dict] = []
    selected_frames: list[pd.DataFrame] = []
    contribution_frames: list[pd.DataFrame] = []
    benchmark = benchmark_row(scope, panel)
    stored_controls = stored_negative_controls(scope)

    for combination in COMBINATIONS:
        bets = select_combination(panel, combination)
        selected_frames.append(bets)
        row = row_for_bets(scope, combination, bets, "portfolio")
        season = season_rows(scope, combination, bets)
        exc = exclusion_rows(scope, combination, bets)
        signal_control = select_combination(panel, combination, permute_signals=True)
        signal_control["strategy"] = f"{combination}__signal_permute_control"
        random_control = random_portfolio_like(panel, bets, f"{combination}__random_portfolio_control")
        controls = pd.DataFrame(
            [
                row_for_bets(scope, signal_control["strategy"].iloc[0], signal_control, "negative_control"),
                row_for_bets(scope, random_control["strategy"].iloc[0], random_control, "negative_control"),
            ]
        )
        relevant_controls = pd.concat([controls, stored_controls], ignore_index=True, sort=False)
        contributions = league_contribution_rows(combination, bets) if scope == "GLOBAL" else pd.DataFrame()
        classification, rationale, failed, passed, total = classify(
            pd.Series(row),
            exc,
            relevant_controls,
            benchmark,
            global_scope=scope == "GLOBAL",
            league_contribution=contributions,
        )
        row["classification"] = classification
        row["rationale"] = rationale
        row["gate_failures"] = ";".join(failed)
        row["passed_gates"] = passed
        row["total_gates"] = total
        summary_rows.append(row)
        control_rows.extend(controls.to_dict("records"))
        if len(season):
            seasonal_frames.append(season)
        if len(exc):
            exclusion_frames.append(exc)
        if len(contributions):
            contribution_frames.append(contributions)
        overlap_frames.append(overlap_rows(scope, combination, bets, panel))

    control_rows.extend(stored_controls.to_dict("records"))
    return summary_rows, seasonal_frames, exclusion_frames, overlap_frames, control_rows, selected_frames, contribution_frames


def write_report(
    summary: pd.DataFrame,
    overlap: pd.DataFrame,
    failures: pd.DataFrame,
    seasonal: pd.DataFrame,
    exclusions: pd.DataFrame,
    controls: pd.DataFrame,
    contributions: pd.DataFrame,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    overlap.to_csv(OVERLAP_PATH, index=False)
    failures.to_csv(FAILURES_PATH, index=False)
    seasonal.to_csv(DETAIL_DIR / "season_by_season.csv", index=False)
    exclusions.to_csv(DETAIL_DIR / "exclusions.csv", index=False)
    controls.to_csv(DETAIL_DIR / "negative_controls.csv", index=False)
    contributions.to_csv(DETAIL_DIR / "global_league_contributions.csv", index=False)

    portfolio = summary[summary["kind"].eq("portfolio")].copy()
    candidates = portfolio[
        portfolio["classification"].eq("paper challenger candidate pending locked falsification")
        & ~portfolio["gate_failures"].str.contains("negative_controls_fail", na=False)
    ].copy()
    lines = [
        "# Cross-League Away AH Portfolio Combination Review",
        "",
        "Scope: I1, SP1, D1, and F1 Away Asian Handicap big home favourite spots.",
        "",
        "Inputs were fixed selected-bet signals from the stored cross-league advanced-model sweep. No new base models were trained, raw data was not edited, and closing odds were used only for CLV diagnostics.",
        "",
        "## Portfolio Results",
        "",
        markdown_table(
            portfolio,
            [
                "league",
                "strategy",
                "bets",
                "profit",
                "roi",
                "z_score",
                "max_drawdown",
                "avg_clv_pp",
                "clv_positive_rate",
                "top3_home_bet_share",
                "home_hhi_bets",
                "unique_matches",
                "avg_candidate_confirmations",
                "classification",
                "gate_failures",
            ],
            [
                "League",
                "Portfolio",
                "Bets",
                "Profit",
                "ROI",
                "z",
                "Max DD",
                "Avg CLV pp",
                "CLV+",
                "Top3 home",
                "Home HHI",
                "Unique",
                "Avg confirms",
                "Class",
                "Gate failures",
            ],
        ),
        "",
        "## Global League Contribution",
        "",
        markdown_table(
            contributions,
            ["strategy", "contribution_league", "bets", "profit", "roi", "z_score", "avg_clv_pp"],
            ["Portfolio", "League", "Bets", "Profit", "ROI", "z", "Avg CLV pp"],
        ),
        "",
        "## Negative Controls",
        "",
        markdown_table(
            controls,
            ["league", "strategy", "bets", "profit", "roi", "z_score", "avg_clv_pp", "clv_positive_rate"],
            ["League", "Control", "Bets", "Profit", "ROI", "z", "Avg CLV pp", "CLV+"],
        ),
        "",
        "## Season By Season",
        "",
        markdown_table(
            seasonal,
            ["league", "strategy", "season", "bets", "profit", "roi", "z_score", "avg_clv_pp"],
            ["League", "Portfolio", "Season", "Bets", "Profit", "ROI", "z", "Avg CLV pp"],
        ),
        "",
        "## Exclusions",
        "",
        markdown_table(
            exclusions,
            ["league", "strategy", "exclusion_type", "excluded", "bets", "profit", "roi", "z_score", "avg_clv_pp"],
            ["League", "Portfolio", "Exclusion", "Excluded", "Bets", "Profit", "ROI", "z", "Avg CLV pp"],
        ),
        "",
        "## Overlap",
        "",
        markdown_table(
            overlap,
            ["league", "strategy", "base_candidate", "portfolio_bets", "candidate_bets", "overlap_bets", "share_of_portfolio"],
            ["League", "Portfolio", "Base candidate", "Portfolio bets", "Base bets", "Overlap", "Share portfolio"],
        ),
        "",
        "## Candidates worth locked falsification",
        "",
        markdown_table(
            candidates,
            ["league", "strategy", "bets", "profit", "roi", "z_score", "avg_clv_pp", "clv_positive_rate", "gate_failures"],
            ["League", "Portfolio", "Bets", "Profit", "ROI", "z", "Avg CLV pp", "CLV+", "Remaining failures"],
        ),
        "",
        "No result is a confirmed edge, safe edge, or live-betting-ready result.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    selected = load_base_selected_bets()
    full_panel = build_signal_panel(selected)
    summary_rows: list[dict] = []
    seasonal_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []
    overlap_frames: list[pd.DataFrame] = []
    control_rows: list[dict] = []
    selected_frames: list[pd.DataFrame] = []
    contribution_frames: list[pd.DataFrame] = []

    for league in LEAGUES:
        scope_panel = full_panel[full_panel["league"].eq(league)].copy()
        result = run_scope(league, scope_panel)
        summary_rows.extend(result[0])
        seasonal_frames.extend(result[1])
        exclusion_frames.extend(result[2])
        overlap_frames.extend(result[3])
        control_rows.extend(result[4])
        selected_frames.extend(result[5])
        contribution_frames.extend(result[6])

    global_result = run_scope("GLOBAL", full_panel.copy())
    summary_rows.extend(global_result[0])
    seasonal_frames.extend(global_result[1])
    exclusion_frames.extend(global_result[2])
    overlap_frames.extend(global_result[3])
    control_rows.extend(global_result[4])
    selected_frames.extend(global_result[5])
    contribution_frames.extend(global_result[6])

    summary = pd.DataFrame(summary_rows)
    controls = pd.DataFrame(control_rows)
    seasonal = pd.concat(seasonal_frames, ignore_index=True, sort=False) if seasonal_frames else pd.DataFrame()
    exclusions = pd.concat(exclusion_frames, ignore_index=True, sort=False) if exclusion_frames else pd.DataFrame()
    overlap = pd.concat(overlap_frames, ignore_index=True, sort=False) if overlap_frames else pd.DataFrame()
    contributions = pd.concat(contribution_frames, ignore_index=True, sort=False) if contribution_frames else pd.DataFrame()
    selected_output = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    selected_output.to_csv(DETAIL_DIR / "selected_bets.csv", index=False)

    failures = summary[summary["kind"].eq("portfolio") & summary["gate_failures"].astype(str).ne("")].copy()
    write_report(summary, overlap, failures, seasonal, exclusions, controls, contributions)
    print(REPORT_PATH)
    print(SUMMARY_PATH)
    print(OVERLAP_PATH)
    print(FAILURES_PATH)


if __name__ == "__main__":
    main()
