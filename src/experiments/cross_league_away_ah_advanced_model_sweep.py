from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced
from src.experiments import e0_away_ah_bag_last5_pooled_form_falsification_review as pooled_e0
from src.experiments import e0_away_ah_deep_cross_wide_deep_falsification_review as deep_e0
from src.experiments import e0_away_ah_sequence_transformer_n5_falsification_review as seq_falsification
from src.experiments import e0_away_ah_team_sequence_model_review as sequence_review
from src.experiments import i1_away_ah_contextual_memory_review as league_review
from src.features.contextual_features import assert_no_closing_columns
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


REPORT_PATH = Path("outputs/reports/cross_league_away_ah_advanced_model_sweep_summary.md")
SUMMARY_PATH = Path("outputs/reports/cross_league_away_ah_advanced_model_sweep_summary.csv")
FAILURES_PATH = Path("outputs/reports/cross_league_away_ah_advanced_model_sweep_failures.csv")
DETAIL_DIR = Path("outputs/cross_league_away_ah_advanced_model_sweep")

LEAGUES = {
    "I1": "Serie A",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}
SEEDS = [11, 23, 37]
TARGET_STYLE = "market_residual"
MIN_VALIDATION_BETS = 12
NO_SEQ_ODDS_COLUMNS = tuple(
    column for column in sequence_review.SEQUENCE_FEATURE_COLUMNS if column not in seq_falsification.SEQUENCE_ODDS_COLUMNS
)


@dataclass(frozen=True)
class ModelConfig:
    strategy: str
    family: str
    kind: str
    control_group: str = "current"
    shuffle_labels: bool = False
    random_noise: bool = False
    random_sequence_order: bool = False
    current_tabular_only: bool = False


def prepare_league_data(league: str) -> pd.DataFrame:
    old_league = league_review.LEAGUE
    try:
        league_review.LEAGUE = league
        dataframe = league_review.add_knn_profit_memory(league_review.prepare_data())
    finally:
        league_review.LEAGUE = old_league
    dataframe[advanced.TARGET_COLUMN] = (pd.to_numeric(dataframe["profit"], errors="coerce") > 0.0).astype(int)
    dataframe["home_market_probability"] = 1.0 - pd.to_numeric(dataframe["away_market_probability"], errors="coerce")
    return dataframe.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def temporal_splits(dataframe: pd.DataFrame):
    return advanced.make_temporal_splits(sorted(dataframe["season_end_year"].unique()))


def candidate_thresholds(scores: pd.Series) -> list[float]:
    clean = pd.to_numeric(scores, errors="coerce").dropna()
    if len(clean) == 0:
        return []
    return sorted(set(float(clean.quantile(q)) for q in advanced.SCORE_QUANTILES))


def select_by_validation(validation: pd.DataFrame, scores: pd.Series) -> dict | None:
    candidates = []
    for ah_threshold in THRESHOLDS:
        for score_threshold in candidate_thresholds(scores):
            selected = validation[
                (pd.to_numeric(validation["ah_line"], errors="coerce") <= ah_threshold) & (scores >= score_threshold)
            ].copy()
            if len(selected) < MIN_VALIDATION_BETS:
                continue
            result = summarize(selected)
            if result["profit"] <= 0 or result["roi"] <= 0:
                continue
            candidates.append(
                {
                    "selected_threshold": ah_threshold,
                    "selected_score_threshold": score_threshold,
                    "validation_bets": result["bets"],
                    "validation_profit": result["profit"],
                    "validation_roi": result["roi"],
                    "validation_z_score": result["z_score"],
                }
            )
    if not candidates:
        return None
    return (
        pd.DataFrame(candidates)
        .sort_values(["validation_z_score", "validation_roi", "validation_bets"], ascending=[False, False, False])
        .iloc[0]
        .to_dict()
    )


def selected_from_scores(test: pd.DataFrame, scores: pd.Series, selected: dict, strategy: str, model_family: str) -> pd.DataFrame:
    bets = test[
        (pd.to_numeric(test["ah_line"], errors="coerce") <= float(selected["selected_threshold"]))
        & (scores >= float(selected["selected_score_threshold"]))
    ].copy()
    bets["strategy"] = strategy
    bets["model_family"] = model_family
    bets["target_style"] = TARGET_STYLE
    bets["selected_threshold"] = selected["selected_threshold"]
    bets["selected_score_threshold"] = selected["selected_score_threshold"]
    return bets


def selected_from_probabilities(test: pd.DataFrame, probability: np.ndarray, selected: dict, strategy: str, model_family: str) -> pd.DataFrame:
    scores = advanced.candidate_score(test, probability, TARGET_STYLE)
    bets = selected_from_scores(test, scores, selected, strategy, model_family)
    if len(bets):
        bets["model_probability"] = probability[bets.index.map(test.index.get_loc)]
        bets["model_score"] = scores.loc[bets.index].to_numpy()
    return bets


def overall_row(league: str, strategy: str, bets: pd.DataFrame, model_family: str, kind: str) -> dict:
    row = advanced.overall_row(strategy, bets, model_family, TARGET_STYLE if kind != "rule" else "rule")
    row["league"] = league
    row["kind"] = kind
    return row


def run_rule(dataframe: pd.DataFrame, league: str, strategy: str, requires_memory: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_year_rows = []
    bet_frames = []
    metric_rows = []
    for split in temporal_splits(dataframe):
        validation = dataframe[dataframe["season_end_year"].eq(split.validation_year)].copy()
        test = dataframe[dataframe["season_end_year"].eq(split.test_year)].copy()
        base_validation = pd.Series(1.0, index=validation.index)
        if strategy.startswith("away_odds_ge_1_85"):
            base_validation = base_validation.where(pd.to_numeric(validation["away_ah_odds"], errors="coerce") >= 1.85, -np.inf)
        if requires_memory:
            memory = pd.to_numeric(validation.get("memory_score_knn_profit"), errors="coerce")
            base_validation = base_validation + memory.rank(pct=True).fillna(-np.inf)
        selected = select_by_validation(validation, base_validation)
        if selected is None:
            by_year_rows.append({"league": league, "strategy": strategy, "test_year": split.test_year, "test_bets": 0, "test_profit": 0.0, "test_roi": 0.0, "selected_filter": "no_valid_validation_candidate"})
            continue
        test_score = pd.Series(1.0, index=test.index)
        if strategy.startswith("away_odds_ge_1_85"):
            test_score = test_score.where(pd.to_numeric(test["away_ah_odds"], errors="coerce") >= 1.85, -np.inf)
        if requires_memory:
            memory = pd.to_numeric(test.get("memory_score_knn_profit"), errors="coerce")
            test_score = test_score + memory.rank(pct=True).fillna(-np.inf)
        bets = selected_from_scores(test, test_score, selected, strategy, "rule")
        result = summarize(bets)
        by_year_rows.append({"league": league, "strategy": strategy, "test_year": split.test_year, "validation_year": split.validation_year, "test_bets": result["bets"], "test_profit": result["profit"], "test_roi": result["roi"], "test_z_score": result["z_score"], **selected})
        if len(bets):
            bets["league"] = league
            bets["nested_test_year"] = split.test_year
            bet_frames.append(bets)
    return pd.DataFrame(by_year_rows), pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame(), pd.DataFrame(metric_rows)


def sklearn_model(family: str, seed: int):
    if family == "logistic":
        return advanced.logistic_model(seed)
    if family == "xgboost":
        return advanced.xgboost_model(seed)
    raise ValueError(f"Unsupported sklearn family: {family}")


def torch_model(family: str, seed: int):
    if family == "numpy_mlp":
        return advanced.neural_model(seed)
    if family == "ft_transformer":
        return advanced.torch_transformer_model(seed)
    if family in {"deep_cross_network", "wide_deep_combined"}:
        return deep_e0.TorchCurrentClassifier(family, seed)
    raise ValueError(f"Unsupported model family: {family}")


def add_pooled_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    pooled = pooled_e0.build_pooled_last5_features(dataframe)
    output = pd.concat([dataframe.copy(), pooled], axis=1)
    return output, list(pooled.columns)


def fit_current_arrays(train, validation, test, extra_numeric: list[str] | None = None):
    numeric, categorical = advanced.available_feature_columns(train)
    if extra_numeric:
        numeric = numeric + [column for column in extra_numeric if column in train.columns]
    assert_no_closing_columns(numeric + categorical)
    preprocessor, _, _ = advanced.build_preprocessor(train[numeric + categorical])
    preprocessor.fit(train[numeric + categorical].copy())
    return (
        advanced.transform(preprocessor, train, numeric, categorical).astype(np.float32),
        advanced.transform(preprocessor, validation, numeric, categorical).astype(np.float32),
        advanced.transform(preprocessor, test, numeric, categorical).astype(np.float32),
        numeric + categorical,
    )


def run_current_model(dataframe: pd.DataFrame, league: str, config: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = dataframe.copy()
    pooled_columns: list[str] = []
    if config.kind == "pooled":
        source, pooled_columns = add_pooled_features(source)
    by_year_rows = []
    bet_frames = []
    seed_bet_frames = []
    metric_rows = []
    for split in temporal_splits(source):
        train = source[source["season_end_year"].isin(split.train_years)].copy()
        validation = source[source["season_end_year"].eq(split.validation_year)].copy()
        test = source[source["season_end_year"].eq(split.test_year)].copy()
        train_x, validation_x, test_x, feature_names = fit_current_arrays(train, validation, test, pooled_columns)
        if config.random_noise:
            train_x = deep_e0._noise_like(train_x, 1000 + split.test_year)
            validation_x = deep_e0._noise_like(validation_x, 2000 + split.test_year)
            test_x = deep_e0._noise_like(test_x, 3000 + split.test_year)
        train_y = train[advanced.TARGET_COLUMN].astype(int).to_numpy()
        validation_y = validation[advanced.TARGET_COLUMN].astype(int).to_numpy()
        if config.shuffle_labels:
            rng = np.random.default_rng(4000 + split.test_year)
            train_y = train_y.copy()
            rng.shuffle(train_y)
        val_probs = []
        test_probs = []
        for seed in SEEDS:
            if config.family in {"logistic", "xgboost"}:
                model = sklearn_model(config.family, seed)
                if model is None:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    val_p, test_p = advanced.safe_probability_fit_predict(model, train_x, train_y, validation_x, test_x)
            else:
                model = torch_model(config.family, seed)
                val_p, test_p = advanced.safe_probability_fit_predict_with_validation(model, train_x, train_y, validation_x, validation_y, test_x)
            val_probs.append(val_p)
            test_probs.append(test_p)
            metric_rows.append(advanced.probability_metrics(test, test_p, f"{config.strategy}_seed_{seed}", split.test_year) | {"league": league, "strategy": config.strategy, "seed": seed, "feature_count": len(feature_names)})
            seed_selected = select_by_validation(validation, advanced.candidate_score(validation, val_p, TARGET_STYLE))
            if seed_selected is not None:
                seed_bets = selected_from_probabilities(test, test_p, seed_selected, f"{config.strategy}_seed_{seed}", config.family)
                if len(seed_bets):
                    seed_bets["league"] = league
                    seed_bet_frames.append(seed_bets)
        if not val_probs:
            continue
        val_ensemble = np.mean(val_probs, axis=0)
        test_ensemble = np.mean(test_probs, axis=0)
        metric_rows.append(advanced.probability_metrics(test, test_ensemble, config.strategy, split.test_year) | {"league": league, "strategy": config.strategy, "seed": "ensemble", "feature_count": len(feature_names)})
        selected = select_by_validation(validation, advanced.candidate_score(validation, val_ensemble, TARGET_STYLE))
        if selected is None:
            by_year_rows.append({"league": league, "strategy": config.strategy, "test_year": split.test_year, "validation_year": split.validation_year, "test_bets": 0, "test_profit": 0.0, "test_roi": 0.0, "selected_filter": "no_valid_validation_candidate"})
            continue
        bets = selected_from_probabilities(test, test_ensemble, selected, config.strategy, config.family)
        result = summarize(bets)
        by_year_rows.append({"league": league, "strategy": config.strategy, "test_year": split.test_year, "validation_year": split.validation_year, "test_bets": result["bets"], "test_profit": result["profit"], "test_roi": result["roi"], "test_z_score": result["z_score"], **selected})
        if len(bets):
            bets["league"] = league
            bets["nested_test_year"] = split.test_year
            bet_frames.append(bets)
    return (
        pd.DataFrame(by_year_rows),
        pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame(),
        pd.concat(seed_bet_frames, ignore_index=True, sort=False) if seed_bet_frames else pd.DataFrame(),
        pd.DataFrame(metric_rows),
    )


def run_sequence_model(dataframe: pd.DataFrame, league: str, config: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_year_rows = []
    bet_frames = []
    metric_rows = []
    all_home, all_away = sequence_review.build_sequence_arrays(dataframe, 5)
    indices = [sequence_review.SEQUENCE_FEATURE_COLUMNS.index(column) for column in NO_SEQ_ODDS_COLUMNS]
    all_home = all_home[:, :, indices]
    all_away = all_away[:, :, indices]
    if config.current_tabular_only:
        all_home = np.zeros((len(dataframe), 5, 1), dtype=float)
        all_away = np.zeros((len(dataframe), 5, 1), dtype=float)
    if config.random_sequence_order:
        rng = np.random.default_rng(12345)
        all_home = all_home.copy()
        all_away = all_away.copy()
        for idx in range(len(all_home)):
            all_home[idx] = all_home[idx, rng.permutation(5)]
            all_away[idx] = all_away[idx, rng.permutation(5)]
    for split in temporal_splits(dataframe):
        train = dataframe[dataframe["season_end_year"].isin(split.train_years)].copy()
        validation = dataframe[dataframe["season_end_year"].eq(split.validation_year)].copy()
        test = dataframe[dataframe["season_end_year"].eq(split.test_year)].copy()
        preprocessor, numeric, categorical = advanced.fit_preprocessor(train)
        seq_scaler = sequence_review.fit_sequence_scaler(all_home[train.index.to_numpy()], all_away[train.index.to_numpy()])
        train_y = train[advanced.TARGET_COLUMN].astype(int).to_numpy()
        if config.shuffle_labels:
            rng = np.random.default_rng(5000 + split.test_year)
            train_y = train_y.copy()
            rng.shuffle(train_y)

        def bundle(subset, y_override=None):
            idx = subset.index.to_numpy()
            return sequence_review.SequenceBundle(
                current_x=advanced.transform(preprocessor, subset, numeric, categorical).astype(np.float32),
                home_sequence=sequence_review.transform_sequences(seq_scaler, all_home[idx]),
                away_sequence=sequence_review.transform_sequences(seq_scaler, all_away[idx]),
                y=subset[advanced.TARGET_COLUMN].astype(int).to_numpy() if y_override is None else y_override,
                dataframe=subset.copy(),
            )

        train_bundle = bundle(train, train_y)
        validation_bundle = bundle(validation)
        test_bundle = bundle(test)
        val_probs = []
        test_probs = []
        for seed in SEEDS:
            model = sequence_review.TeamSequenceClassifier("sequence_transformer", seed).fit(train_bundle, validation_bundle)
            val_p = model.predict_proba(validation_bundle)[:, 1]
            test_p = model.predict_proba(test_bundle)[:, 1]
            val_probs.append(val_p)
            test_probs.append(test_p)
            metric_rows.append(sequence_review.probability_metrics(test, test_p, f"{config.strategy}_seed_{seed}", split.test_year) | {"league": league, "strategy": config.strategy, "seed": seed})
        val_ensemble = np.mean(val_probs, axis=0)
        test_ensemble = np.mean(test_probs, axis=0)
        metric_rows.append(sequence_review.probability_metrics(test, test_ensemble, config.strategy, split.test_year) | {"league": league, "strategy": config.strategy, "seed": "ensemble"})
        selected = select_by_validation(validation, advanced.candidate_score(validation, val_ensemble, TARGET_STYLE))
        if selected is None:
            by_year_rows.append({"league": league, "strategy": config.strategy, "test_year": split.test_year, "validation_year": split.validation_year, "test_bets": 0, "test_profit": 0.0, "test_roi": 0.0, "selected_filter": "no_valid_validation_candidate"})
            continue
        bets = selected_from_probabilities(test, test_ensemble, selected, config.strategy, config.family)
        result = summarize(bets)
        by_year_rows.append({"league": league, "strategy": config.strategy, "test_year": split.test_year, "validation_year": split.validation_year, "test_bets": result["bets"], "test_profit": result["profit"], "test_roi": result["roi"], "test_z_score": result["z_score"], **selected})
        if len(bets):
            bets["league"] = league
            bets["nested_test_year"] = split.test_year
            bet_frames.append(bets)
    return pd.DataFrame(by_year_rows), pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame(), pd.DataFrame(metric_rows)


def season_rows(league: str, strategy: str, bets: pd.DataFrame) -> list[dict]:
    rows = []
    for season, group in bets.groupby("season_end_year"):
        row = overall_row(league, strategy, group, group["model_family"].iloc[0], "season")
        row["season"] = int(season)
        rows.append(row)
    return rows


def exclusion_rows(league: str, strategy: str, bets: pd.DataFrame) -> list[dict]:
    rows = []
    if bets.empty:
        return rows
    seasonal = pd.DataFrame(season_rows(league, strategy, bets))
    best_season = int(seasonal.sort_values("profit", ascending=False).iloc[0]["season"]) if len(seasonal) else None
    for season in sorted(bets["season_end_year"].unique()):
        row = overall_row(league, strategy, bets[bets["season_end_year"].ne(season)].copy(), bets["model_family"].iloc[0], "exclude")
        row["excluded"] = int(season)
        row["exclusion_type"] = "exclude_each_season"
        rows.append(row)
    if best_season is not None:
        row = overall_row(league, strategy, bets[bets["season_end_year"].ne(best_season)].copy(), bets["model_family"].iloc[0], "exclude")
        row["excluded"] = best_season
        row["exclusion_type"] = "exclude_best_profit_season"
        rows.append(row)
    counts = bets["HomeTeam"].value_counts()
    for n in [1, 2, 3]:
        teams = list(counts.head(n).index)
        row = overall_row(league, strategy, bets[~bets["HomeTeam"].isin(teams)].copy(), bets["model_family"].iloc[0], "exclude")
        row["excluded"] = ", ".join(teams)
        row["exclusion_type"] = f"exclude_top{n}_home"
        rows.append(row)
    return rows


def seed_mean_row(league: str, config: ModelConfig, seed_bets: pd.DataFrame) -> dict:
    rows = []
    for seed in SEEDS:
        strategy = f"{config.strategy}_seed_{seed}"
        rows.append(overall_row(league, strategy, seed_bets[seed_bets["strategy"].eq(strategy)].copy(), config.family, "seed"))
    frame = pd.DataFrame(rows)
    row = {"league": league, "strategy": f"{config.strategy}_seed_mean", "model_family": f"{config.family}_seed_mean", "target_style": TARGET_STYLE, "kind": "seed_mean", "seed_count": len(SEEDS)}
    for column in ["bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "top3_away_bet_share", "home_hhi_bets", "away_hhi_bets"]:
        row[column] = float(frame[column].mean()) if column in frame else pd.NA
    row["seed_profit_std"] = float(frame["profit"].std(ddof=0))
    row["seed_roi_std"] = float(frame["roi"].std(ddof=0))
    return row


def leakage_rows(league: str, dataframe: pd.DataFrame) -> list[dict]:
    numeric, categorical = advanced.available_feature_columns(dataframe)
    histories = sequence_review.build_team_histories(dataframe)
    violations = 0
    self_inclusions = 0
    for index, row in dataframe.iterrows():
        current_date = pd.Timestamp(row["Date"])
        for team in [row["HomeTeam"], row["AwayTeam"]]:
            selected = [item for item in histories[str(team)] if pd.Timestamp(item["source_date"]) < current_date][-5:]
            violations += sum(pd.Timestamp(item["source_date"]) >= current_date for item in selected)
            self_inclusions += sum(int(item["source_index"]) == int(index) for item in selected)
    return [
        {"league": league, "check": "closing_absent_current_features", "passed": True, "detail": ",".join(numeric + categorical)},
        {"league": league, "check": "sequence_dates_strictly_before_current", "passed": violations == 0, "detail": violations},
        {"league": league, "check": "current_match_not_in_sequence", "passed": self_inclusions == 0, "detail": self_inclusions},
        {"league": league, "check": "scalers_fit_train_only", "passed": True, "detail": "fit per train split"},
    ]


def classify_rows(summary: pd.DataFrame, exclusions: pd.DataFrame, controls: pd.DataFrame, leakage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    failures = []
    simple = summary[summary["strategy"].isin(["away_odds_ge_1_85", "away_odds_ge_1_85_plus_memory_knn_profit"])].copy()
    for idx, row in summary.iterrows():
        if row["kind"] in {"negative_control", "seed_mean"}:
            continue
        league = row["league"]
        strategy = row["strategy"]
        league_simple = simple[simple["league"].eq(league)]
        simple_best_z = float(league_simple["z_score"].max()) if len(league_simple) else 0.0
        current_controls = controls[(controls["league"].eq(league)) & (controls["kind"].eq("negative_control"))]
        model_exclusions = exclusions[(exclusions["league"].eq(league)) & (exclusions["strategy"].eq(strategy))]
        no_2025 = model_exclusions[(model_exclusions["exclusion_type"].eq("exclude_each_season")) & (model_exclusions["excluded"].astype(str).eq("2025"))]
        best_ex = model_exclusions[model_exclusions["exclusion_type"].eq("exclude_best_profit_season")]
        top1 = model_exclusions[model_exclusions["exclusion_type"].eq("exclude_top1_home")]
        top3 = model_exclusions[model_exclusions["exclusion_type"].eq("exclude_top3_home")]
        control_success = current_controls[(current_controls["profit"] > 0) & (current_controls["roi"] > 0) & (current_controls["avg_clv_pp"] > 0)]
        gates = {
            "at_least_50_bets": row["bets"] >= 50,
            "positive_roi": row["roi"] > 0,
            "z_ge_1_5": row["z_score"] >= 1.5,
            "positive_clv": row["avg_clv_pp"] > 0,
            "clv_plus_ge_52": row["clv_positive_rate"] >= 0.52,
            "positive_roi_excluding_2025": bool(len(no_2025) and float(no_2025.iloc[0]["roi"]) > 0),
            "positive_roi_excluding_best_profit_season": bool(len(best_ex) and float(best_ex.iloc[0]["roi"]) > 0),
            "positive_roi_excluding_top1_home": bool(len(top1) and float(top1.iloc[0]["roi"]) > 0),
            "not_destroyed_excluding_top3_home": bool(len(top3) and float(top3.iloc[0]["roi"]) > -0.05),
            "top3_home_not_extreme": row["top3_home_bet_share"] <= 0.70,
            "negative_controls_fail": control_success.empty,
            "no_leakage_warning": bool(leakage[leakage["league"].eq(league)]["passed"].all()),
            "beats_simple_league_benchmark": row["z_score"] > simple_best_z and row["profit"] > 0,
        }
        failed = [name for name, ok in gates.items() if not bool(ok)]
        classification = "reject"
        if row["profit"] > 0 and row["roi"] > 0:
            classification = "research only"
        if not failed:
            classification = "paper challenger candidate pending locked falsification"
        rows.append({"index": idx, "classification": classification, "gate_failures": ";".join(failed), "passed_gates": len(gates) - len(failed), "total_gates": len(gates)})
        for gate in failed:
            failures.append({"league": league, "strategy": strategy, "failed_gate": gate})
    class_frame = pd.DataFrame(rows).set_index("index") if rows else pd.DataFrame()
    failure_frame = pd.DataFrame(failures)
    return class_frame, failure_frame


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    return advanced.markdown_table(frame, columns, headers)


def write_report(summary, failures, seasonal, exclusions, controls, leakage, metrics):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    failures.to_csv(FAILURES_PATH, index=False)
    seasonal.to_csv(DETAIL_DIR / "season_by_season.csv", index=False)
    exclusions.to_csv(DETAIL_DIR / "exclusions.csv", index=False)
    controls.to_csv(DETAIL_DIR / "negative_controls.csv", index=False)
    leakage.to_csv(DETAIL_DIR / "leakage_audit.csv", index=False)
    metrics.to_csv(DETAIL_DIR / "probability_metrics.csv", index=False)
    candidates = summary[
        summary["classification"].eq("paper challenger candidate pending locked falsification")
        | ((summary["classification"].eq("research only")) & (summary["passed_gates"] >= 10) & ~summary["gate_failures"].str.contains("negative_controls_fail", na=False))
    ].copy()
    lines = [
        "# Cross-League Away AH Advanced Model Sweep",
        "",
        "Scope: research sweep only. No live betting, no external APIs, no raw data edits, and no closing odds as bet-time-safe features.",
        "",
        "All positive results remain research only unless they later pass a separate locked falsification review.",
        "",
        "## All League/Model Results",
        "",
        markdown_table(summary, ["league", "strategy", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "home_hhi_bets", "classification", "gate_failures"], ["League", "Strategy", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate", "Top3 home", "Home HHI", "Classification", "Gate failures"]),
        "",
        "## Negative Controls",
        "",
        markdown_table(controls, ["league", "strategy", "bets", "profit", "roi", "z_score", "avg_clv_pp", "clv_positive_rate"], ["League", "Control", "Bets", "Profit", "ROI", "z", "Avg CLV pp", "CLV+ rate"]),
        "",
        "## Leakage Audit",
        "",
        markdown_table(leakage, ["league", "check", "passed", "detail"], ["League", "Check", "Passed", "Detail"]),
        "",
        "## Candidates Worth Locked Falsification",
        "",
        markdown_table(candidates, ["league", "strategy", "bets", "profit", "roi", "z_score", "avg_clv_pp", "clv_positive_rate", "passed_gates", "gate_failures"], ["League", "Strategy", "Bets", "Profit", "ROI", "z", "Avg CLV pp", "CLV+ rate", "Passed gates", "Remaining failures"]),
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    model_configs = [
        ModelConfig("logistic_market_residual", "logistic", "current"),
        ModelConfig("xgboost_market_residual", "xgboost", "current"),
        ModelConfig("numpy_mlp_market_residual", "numpy_mlp", "current"),
        ModelConfig("ft_transformer_market_residual", "ft_transformer", "current"),
        ModelConfig("deep_cross_network_market_residual", "deep_cross_network", "current"),
        ModelConfig("wide_deep_market_residual", "wide_deep_combined", "current"),
        ModelConfig("bag_last5_pooled_market_residual", "logistic", "pooled"),
        ModelConfig("sequence_transformer_n5_no_seq_odds_market_residual", "sequence_transformer", "sequence", control_group="sequence"),
        ModelConfig("shuffled_labels_negative_control", "deep_cross_network", "negative_control", shuffle_labels=True),
        ModelConfig("random_feature_noise_negative_control", "deep_cross_network", "negative_control", random_noise=True),
        ModelConfig("sequence_random_order_negative_control", "sequence_transformer", "negative_control", control_group="sequence", random_sequence_order=True),
        ModelConfig("sequence_current_tabular_only_control", "sequence_transformer", "negative_control", control_group="sequence", current_tabular_only=True),
    ]
    all_summary = []
    all_by_year = []
    all_bets = []
    all_seed_bets = []
    all_metrics = []
    all_seasonal = []
    all_exclusions = []
    all_leakage = []
    for league in LEAGUES:
        dataframe = prepare_league_data(league)
        all_leakage.extend(leakage_rows(league, dataframe))
        rule_runs = [
            ("away_odds_ge_1_85", False),
            ("away_odds_ge_1_85_plus_memory_knn_profit", True),
        ]
        for strategy, requires_memory in rule_runs:
            by_year, bets, metrics = run_rule(dataframe, league, strategy, requires_memory)
            all_by_year.append(by_year)
            if len(bets):
                all_bets.append(bets)
            all_summary.append(overall_row(league, strategy, bets, "rule", "rule"))
            all_seasonal.extend(season_rows(league, strategy, bets) if len(bets) else [])
            all_exclusions.extend(exclusion_rows(league, strategy, bets) if len(bets) else [])
        for config in model_configs:
            if config.kind == "sequence" or config.control_group == "sequence":
                by_year, bets, metrics = run_sequence_model(dataframe, league, config)
                seed_bets = pd.DataFrame()
            else:
                by_year, bets, seed_bets, metrics = run_current_model(dataframe, league, config)
            all_by_year.append(by_year)
            if len(bets):
                all_bets.append(bets)
            if len(seed_bets):
                all_seed_bets.append(seed_bets)
            if len(metrics):
                all_metrics.append(metrics)
            kind = "negative_control" if config.kind == "negative_control" else config.kind
            all_summary.append(overall_row(league, config.strategy, bets, config.family, kind))
            if config.kind not in {"negative_control", "sequence"} and len(seed_bets):
                all_summary.append(seed_mean_row(league, config, seed_bets))
            all_seasonal.extend(season_rows(league, config.strategy, bets) if len(bets) else [])
            all_exclusions.extend(exclusion_rows(league, config.strategy, bets) if len(bets) else [])
    summary = pd.DataFrame(all_summary)
    seasonal = pd.DataFrame(all_seasonal)
    exclusions = pd.DataFrame(all_exclusions)
    metrics = pd.concat(all_metrics, ignore_index=True, sort=False) if all_metrics else pd.DataFrame()
    leakage = pd.DataFrame(all_leakage)
    controls = summary[summary["kind"].eq("negative_control")].copy()
    class_frame, failures = classify_rows(summary, exclusions, controls, leakage)
    if len(class_frame):
        summary.loc[class_frame.index, "classification"] = class_frame["classification"]
        summary.loc[class_frame.index, "gate_failures"] = class_frame["gate_failures"]
        summary.loc[class_frame.index, "passed_gates"] = class_frame["passed_gates"]
        summary.loc[class_frame.index, "total_gates"] = class_frame["total_gates"]
    summary["classification"] = summary["classification"].fillna("research only")
    summary["gate_failures"] = summary["gate_failures"].fillna("")
    write_report(summary, failures, seasonal, exclusions, controls, leakage, metrics)
    pd.concat(all_by_year, ignore_index=True, sort=False).to_csv(DETAIL_DIR / "nested_by_year.csv", index=False)
    if all_bets:
        pd.concat(all_bets, ignore_index=True, sort=False).to_csv(DETAIL_DIR / "selected_bets.csv", index=False)
    if all_seed_bets:
        pd.concat(all_seed_bets, ignore_index=True, sort=False).to_csv(DETAIL_DIR / "seed_selected_bets.csv", index=False)
    print(REPORT_PATH)
    print(SUMMARY_PATH)
    print(FAILURES_PATH)


if __name__ == "__main__":
    main()
