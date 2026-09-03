from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


REPORT_PATH = Path("outputs/reports/e0_away_ah_portfolio_combination_review.md")
SUMMARY_PATH = Path("outputs/reports/e0_away_ah_portfolio_combination_summary.csv")
OVERLAP_PATH = Path("outputs/reports/e0_away_ah_portfolio_combination_bet_overlap.csv")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/portfolio_combination_review")

BASE_CANDIDATES = [
    "away_odds_ge_1_85",
    "away_odds_ge_1_85_plus_memory_knn_profit",
    "logistic_binary_cover",
    "logistic_market_residual",
    "xgboost_market_residual",
    "deep_cross_network_ensemble",
    "sequence_transformer_n5_no_sequence_odds_ah",
    "pooled_last5_market_residual",
]

MODEL_CANDIDATES = [
    "logistic_binary_cover",
    "logistic_market_residual",
    "xgboost_market_residual",
    "deep_cross_network_ensemble",
    "sequence_transformer_n5_no_sequence_odds_ah",
    "pooled_last5_market_residual",
]

SCORE_CANDIDATES = [
    "logistic_binary_cover",
    "logistic_market_residual",
    "xgboost_market_residual",
    "deep_cross_network_ensemble",
    "pooled_last5_market_residual",
]

SEQUENCE_RECENT_CANDIDATES = [
    "sequence_transformer_n5_no_sequence_odds_ah",
    "pooled_last5_market_residual",
]

COMBINATIONS = [
    "union_all_positive",
    "intersection_2plus",
    "intersection_3plus",
    "rule_plus_model_confirm",
    "memory_plus_model_confirm",
    "score_average_portfolio",
    "diversified_bucket_portfolio",
]


def _key(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        + "|"
        + frame["HomeTeam"].astype(str)
        + "|"
        + frame["AwayTeam"].astype(str)
    )


def load_strategy(path: str, source_strategy: str, candidate: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame["strategy"].eq(source_strategy)].copy()
    if frame.empty:
        return frame
    frame["match_key"] = _key(frame)
    frame["candidate"] = candidate
    return frame


def load_base_candidate_bets() -> dict[str, pd.DataFrame]:
    root = "outputs/E0/asian_handicap_big_home_favorite_away"
    return {
        "away_odds_ge_1_85": load_strategy(f"{root}/memory_odds_combo_review/nested_bets.csv", "away_odds_ge_1_85", "away_odds_ge_1_85"),
        "away_odds_ge_1_85_plus_memory_knn_profit": load_strategy(
            f"{root}/memory_odds_combo_review/nested_bets.csv",
            "away_odds_ge_1_85_plus_memory_knn_profit",
            "away_odds_ge_1_85_plus_memory_knn_profit",
        ),
        "logistic_binary_cover": load_strategy(f"{root}/advanced_tabular_neural_review/nested_bets.csv", "logistic_binary_cover", "logistic_binary_cover"),
        "logistic_market_residual": load_strategy(f"{root}/advanced_tabular_neural_review/nested_bets.csv", "logistic_market_residual", "logistic_market_residual"),
        "xgboost_market_residual": load_strategy(f"{root}/advanced_tabular_neural_review/nested_bets.csv", "xgboost_market_residual", "xgboost_market_residual"),
        "deep_cross_network_ensemble": load_strategy(
            f"{root}/deep_cross_wide_deep_falsification/selected_bets.csv",
            "deep_cross_network_ensemble",
            "deep_cross_network_ensemble",
        ),
        "sequence_transformer_n5_no_sequence_odds_ah": load_strategy(
            f"{root}/sequence_transformer_n5_no_seq_odds_falsification/selected_bets.csv",
            "locked_ensemble_ensemble",
            "sequence_transformer_n5_no_sequence_odds_ah",
        ),
        "pooled_last5_market_residual": load_strategy(
            f"{root}/bag_last5_pooled_form_falsification/selected_bets.csv",
            "pooled_logistic_market_residual",
            "pooled_last5_market_residual",
        ),
    }


def build_signal_panel(candidate_bets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for candidate, frame in candidate_bets.items():
        if frame.empty:
            continue
        rows.append(frame)
    all_bets = pd.concat(rows, ignore_index=True, sort=False)
    base = all_bets.sort_values(["Date", "HomeTeam", "AwayTeam"]).drop_duplicates("match_key").copy()
    for candidate in BASE_CANDIDATES:
        keys = set(candidate_bets[candidate]["match_key"]) if candidate in candidate_bets and len(candidate_bets[candidate]) else set()
        base[f"signal_{candidate}"] = base["match_key"].isin(keys)
        score = pd.Series(np.nan, index=base.index, dtype=float)
        if candidate in candidate_bets and "model_score" in candidate_bets[candidate].columns:
            score_map = candidate_bets[candidate].drop_duplicates("match_key").set_index("match_key")["model_score"]
            score = base["match_key"].map(score_map)
        base[f"score_{candidate}"] = pd.to_numeric(score, errors="coerce")
    base["candidate_confirmations"] = base[[f"signal_{candidate}" for candidate in BASE_CANDIDATES]].sum(axis=1).astype(int)
    base["model_confirmations"] = base[[f"signal_{candidate}" for candidate in MODEL_CANDIDATES]].sum(axis=1).astype(int)
    base["season_end_year"] = pd.to_numeric(base["season_end_year"], errors="coerce").astype(int)
    return base.reset_index(drop=True)


def normalized_score_average(panel: pd.DataFrame, permute: bool = False) -> pd.Series:
    score_frame = panel[[f"score_{candidate}" for candidate in SCORE_CANDIDATES]].copy()
    if permute:
        rng = np.random.default_rng(92217)
        for season, idx in panel.groupby("season_end_year").groups.items():
            for column in score_frame.columns:
                values = score_frame.loc[idx, column].to_numpy(copy=True)
                score_frame.loc[idx, column] = values[rng.permutation(len(values))]
    normalized = []
    for column in score_frame.columns:
        values = pd.to_numeric(score_frame[column], errors="coerce")
        mean = values.mean()
        std = values.std(ddof=0)
        if pd.isna(std) or std == 0:
            normalized.append(pd.Series(np.nan, index=panel.index))
        else:
            normalized.append((values - mean) / std)
    output = pd.concat(normalized, axis=1).mean(axis=1, skipna=True).fillna(-np.inf)
    return output


def score_threshold_selection(panel: pd.DataFrame, score: pd.Series, test_year: int) -> float | None:
    validation_year = test_year - 1
    validation = panel[panel["season_end_year"].eq(validation_year)].copy()
    validation_score = score.loc[validation.index]
    clean = validation_score.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    candidates = []
    for threshold in sorted(set(float(clean.quantile(q)) for q in [0.50, 0.60, 0.70, 0.80])):
        selected = validation[validation_score >= threshold].copy()
        if len(selected) < 8:
            continue
        result = summarize(selected)
        if result["profit"] <= 0 or result["roi"] <= 0:
            continue
        candidates.append({"threshold": threshold, **result})
    if not candidates:
        return None
    return float(pd.DataFrame(candidates).sort_values(["z_score", "roi", "bets"], ascending=[False, False, False]).iloc[0]["threshold"])


def select_combination(panel: pd.DataFrame, combination: str, *, permute_signals: bool = False, permute_scores: bool = False) -> pd.DataFrame:
    working = panel.copy()
    signal_columns = [f"signal_{candidate}" for candidate in BASE_CANDIDATES]
    if permute_signals:
        rng = np.random.default_rng(70631)
        for _, idx in working.groupby("season_end_year").groups.items():
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
        mask = (
            working["signal_away_odds_ge_1_85"]
            | working["signal_away_odds_ge_1_85_plus_memory_knn_profit"]
            | (working[[f"signal_{candidate}" for candidate in MODEL_CANDIDATES]].sum(axis=1) >= 2)
            | (working[[f"signal_{candidate}" for candidate in SEQUENCE_RECENT_CANDIDATES]].sum(axis=1) >= 1)
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
    elif combination == "score_average_portfolio":
        scores = normalized_score_average(working, permute=permute_scores)
        frames = []
        for test_year in sorted(working["season_end_year"].unique()):
            threshold = score_threshold_selection(working, scores, int(test_year))
            if threshold is None:
                continue
            year_frame = working[working["season_end_year"].eq(test_year) & (scores >= threshold)].copy()
            year_frame["portfolio_score"] = scores.loc[year_frame.index].to_numpy()
            year_frame["selected_score_threshold"] = threshold
            frames.append(year_frame)
        selected = pd.concat(frames, ignore_index=True, sort=False) if frames else working.iloc[0:0].copy()
    else:
        raise ValueError(f"Unknown combination: {combination}")
    selected["strategy"] = combination
    return selected.drop_duplicates("match_key").copy()


def random_portfolio_like(panel: pd.DataFrame, selected: pd.DataFrame, strategy: str) -> pd.DataFrame:
    rng = np.random.default_rng(314159)
    frames = []
    for season, group in panel.groupby("season_end_year"):
        count = int((selected["season_end_year"] == season).sum())
        if count <= 0:
            continue
        sample = group.sample(n=min(count, len(group)), random_state=int(rng.integers(0, 1_000_000))).copy()
        frames.append(sample)
    output = pd.concat(frames, ignore_index=True, sort=False) if frames else panel.iloc[0:0].copy()
    output["strategy"] = strategy
    return output


def row_for_bets(strategy: str, bets: pd.DataFrame, kind: str) -> dict:
    row = advanced.overall_row(strategy, bets, "portfolio", "portfolio")
    row["kind"] = kind
    row["unique_matches"] = int(bets["match_key"].nunique()) if len(bets) else 0
    if len(bets) and "candidate_confirmations" in bets.columns:
        row["avg_candidate_confirmations"] = float(bets["candidate_confirmations"].mean())
        row["max_candidate_confirmations"] = int(bets["candidate_confirmations"].max())
    else:
        row["avg_candidate_confirmations"] = pd.NA
        row["max_candidate_confirmations"] = pd.NA
    return row


def season_rows(strategy: str, bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in bets.groupby("season_end_year"):
        row = row_for_bets(strategy, group, "season")
        row["season"] = int(season)
        rows.append(row)
    return pd.DataFrame(rows)


def exclusion_rows(strategy: str, bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bets.empty:
        return pd.DataFrame(rows)
    seasonal = season_rows(strategy, bets)
    best_season = int(seasonal.sort_values("profit", ascending=False).iloc[0]["season"])
    for season in sorted(bets["season_end_year"].unique()):
        row = row_for_bets(strategy, bets[bets["season_end_year"].ne(season)].copy(), "season_exclusion")
        row["exclusion_type"] = "exclude_each_season"
        row["excluded"] = int(season)
        rows.append(row)
    row = row_for_bets(strategy, bets[bets["season_end_year"].ne(best_season)].copy(), "season_exclusion")
    row["exclusion_type"] = "exclude_best_profit_season"
    row["excluded"] = best_season
    rows.append(row)
    counts = bets["HomeTeam"].value_counts()
    for n in [1, 2, 3]:
        teams = list(counts.head(n).index)
        row = row_for_bets(strategy, bets[~bets["HomeTeam"].isin(teams)].copy(), "home_exclusion")
        row["exclusion_type"] = f"exclude_top{n}_home"
        row["excluded"] = ", ".join(teams)
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_rows(strategy: str, bets: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected_keys = set(bets["match_key"]) if len(bets) else set()
    for candidate in BASE_CANDIDATES:
        candidate_keys = set(panel.loc[panel[f"signal_{candidate}"], "match_key"])
        common = selected_keys & candidate_keys
        rows.append(
            {
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


def classify(row: pd.Series, exclusions: pd.DataFrame, controls: pd.DataFrame, memory_row: pd.Series) -> tuple[str, str]:
    no_2024 = exclusions[(exclusions["exclusion_type"].eq("exclude_each_season")) & (exclusions["excluded"].astype(str).eq("2024"))]
    no_2025 = exclusions[(exclusions["exclusion_type"].eq("exclude_each_season")) & (exclusions["excluded"].astype(str).eq("2025"))]
    best_ex = exclusions[exclusions["exclusion_type"].eq("exclude_best_profit_season")]
    top3 = exclusions[exclusions["exclusion_type"].eq("exclude_top3_home")]
    control_success = controls[(controls["profit"] > 0) & (controls["roi"] > 0) & (controls["avg_clv_pp"] > 0)]
    gates = {
        "at_least_80_bets": row["bets"] >= 80,
        "positive_roi": row["roi"] > 0,
        "z_ge_1_5": row["z_score"] >= 1.5,
        "positive_clv": row["avg_clv_pp"] > 0,
        "clv_plus_ge_52": row["clv_positive_rate"] >= 0.52,
        "positive_roi_without_2025": bool(len(no_2025) and float(no_2025.iloc[0]["roi"]) > 0),
        "positive_roi_without_2024": bool(len(no_2024) and float(no_2024.iloc[0]["roi"]) > 0),
        "positive_roi_excluding_best_profit_season": bool(len(best_ex) and float(best_ex.iloc[0]["roi"]) > 0),
        "not_destroyed_excluding_top3_home": bool(len(top3) and float(top3.iloc[0]["roi"]) > -0.05),
        "drawdown_lte_memory_rule": row["max_drawdown"] <= memory_row["max_drawdown"],
        "negative_controls_fail": control_success.empty,
        "beats_memory_rule_robustness": row["profit"] > memory_row["profit"] and row["z_score"] > memory_row["z_score"],
    }
    failed = [name for name, passed in gates.items() if not passed]
    if row["profit"] <= 0 or row["roi"] <= 0:
        return "reject", "Failed gates: " + ", ".join(failed)
    if not failed and row["z_score"] >= 2.0:
        return "main paper candidate", "Clears locked portfolio gates, but not confirmed without forward paper tracking."
    if not failed:
        return "paper challenger candidate", "Clears locked portfolio gates at z>=1.5, but needs locked follow-up/paper tracking."
    return "research only", "Failed gates: " + ", ".join(failed)


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    return advanced.markdown_table(frame, columns, headers)


def write_outputs(summary: pd.DataFrame, overlap: pd.DataFrame, seasonal: pd.DataFrame, exclusions: pd.DataFrame, controls: pd.DataFrame, selected: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    overlap.to_csv(OVERLAP_PATH, index=False)
    seasonal.to_csv(DETAIL_DIR / "season_by_season.csv", index=False)
    exclusions.to_csv(DETAIL_DIR / "exclusions.csv", index=False)
    controls.to_csv(DETAIL_DIR / "negative_controls.csv", index=False)
    selected.to_csv(DETAIL_DIR / "selected_bets.csv", index=False)
    combo_rows = summary[summary["kind"].eq("portfolio")].copy()
    control_rows = summary[summary["kind"].eq("negative_control")].copy()
    lines = [
        "# E0 Away AH Portfolio Combination Review",
        "",
        "Scope: locked E0 Away AH big home favourite portfolio combinations using previously stored candidate selections and scores only.",
        "",
        "No base model was retrained. Raw data was not edited. Closing odds were diagnostic only for CLV.",
        "",
        "## Portfolio Results",
        "",
        markdown_table(combo_rows, ["strategy", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "home_hhi_bets", "unique_matches", "avg_candidate_confirmations", "classification", "rationale"], ["Strategy", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate", "Top3 home", "Home HHI", "Unique", "Avg confirms", "Classification", "Rationale"]),
        "",
        "## Negative Controls",
        "",
        markdown_table(control_rows, ["strategy", "bets", "profit", "roi", "z_score", "avg_clv_pp", "clv_positive_rate"], ["Strategy", "Bets", "Profit", "ROI", "z", "Avg CLV pp", "CLV+ rate"]),
        "",
        "## Season By Season",
        "",
        markdown_table(seasonal, ["strategy", "season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp"], ["Strategy", "Season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp"]),
        "",
        "## Exclusions",
        "",
        markdown_table(exclusions, ["strategy", "exclusion_type", "excluded", "bets", "profit", "roi", "z_score", "avg_clv_pp"], ["Strategy", "Exclusion", "Excluded", "Bets", "Profit", "ROI", "z", "Avg CLV pp"]),
        "",
        "## Base Candidate Overlap",
        "",
        markdown_table(overlap, ["strategy", "base_candidate", "portfolio_bets", "candidate_bets", "overlap_bets", "share_of_portfolio", "share_of_candidate"], ["Strategy", "Base", "Portfolio bets", "Base bets", "Overlap", "Share portfolio", "Share base"]),
        "",
        "## Final Note",
        "",
        "No portfolio is a confirmed edge. Forward paper tracking would be required before any confirmed classification.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    candidate_bets = load_base_candidate_bets()
    panel = build_signal_panel(candidate_bets)
    selected_frames = []
    summary_rows = []
    seasonal_frames = []
    exclusion_frames = []
    overlap_frames = []
    control_rows = []

    base_memory = candidate_bets["away_odds_ge_1_85_plus_memory_knn_profit"].copy()
    base_memory["match_key"] = _key(base_memory)
    memory_row = row_for_bets("away_odds_ge_1_85_plus_memory_knn_profit", base_memory, "base")

    portfolio_cache: dict[str, pd.DataFrame] = {}
    for combination in COMBINATIONS:
        bets = select_combination(panel, combination)
        portfolio_cache[combination] = bets
        selected_frames.append(bets)
        row = row_for_bets(combination, bets, "portfolio")
        season = season_rows(combination, bets)
        exc = exclusion_rows(combination, bets)
        signal_control = select_combination(panel, combination, permute_signals=True)
        signal_control["strategy"] = f"{combination}__signal_permute_control"
        random_control = random_portfolio_like(panel, bets, f"{combination}__random_portfolio_control")
        controls = [row_for_bets(signal_control["strategy"].iloc[0], signal_control, "negative_control"), row_for_bets(random_control["strategy"].iloc[0], random_control, "negative_control")]
        if combination == "score_average_portfolio":
            score_control = select_combination(panel, combination, permute_scores=True)
            score_control["strategy"] = f"{combination}__score_permute_control"
            controls.append(row_for_bets(score_control["strategy"].iloc[0], score_control, "negative_control"))
        controls_frame = pd.DataFrame(controls)
        classification, rationale = classify(row, exc, controls_frame, memory_row)
        row["classification"] = classification
        row["rationale"] = rationale
        summary_rows.append(row)
        control_rows.extend(controls)
        if len(season):
            seasonal_frames.append(season)
        if len(exc):
            exclusion_frames.append(exc)
        overlap_frames.append(overlap_rows(combination, bets, panel))

    shuffled_sources = [
        ("deep_shuffled_label_model_candidates", "outputs/E0/asian_handicap_big_home_favorite_away/deep_cross_wide_deep_falsification/selected_bets.csv", "shuffled_training_labels_negative_control_ensemble"),
        ("pooled_shuffled_label_model_candidates", "outputs/E0/asian_handicap_big_home_favorite_away/bag_last5_pooled_form_falsification/selected_bets.csv", "shuffled_training_labels_negative_control"),
        ("sequence_shuffled_label_model_candidates", "outputs/E0/asian_handicap_big_home_favorite_away/sequence_transformer_n5_no_seq_odds_falsification/selected_bets.csv", "shuffled_train_labels_negative_control_ensemble"),
    ]
    for name, path, strategy in shuffled_sources:
        frame = load_strategy(path, strategy, name)
        control_rows.append(row_for_bets(name, frame, "negative_control"))

    summary = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame(control_rows)], ignore_index=True, sort=False)
    seasonal = pd.concat(seasonal_frames, ignore_index=True, sort=False) if seasonal_frames else pd.DataFrame()
    exclusions = pd.concat(exclusion_frames, ignore_index=True, sort=False) if exclusion_frames else pd.DataFrame()
    overlap = pd.concat(overlap_frames, ignore_index=True, sort=False)
    selected = pd.concat(selected_frames, ignore_index=True, sort=False)
    controls = summary[summary["kind"].eq("negative_control")].copy()
    write_outputs(summary, overlap, seasonal, exclusions, controls, selected)
    print(REPORT_PATH)
    print(SUMMARY_PATH)
    print(OVERLAP_PATH)


if __name__ == "__main__":
    main()
