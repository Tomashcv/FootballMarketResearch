from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced
from src.experiments import e0_away_ah_sequence_transformer_n5_falsification_review as base
from src.experiments import e0_away_ah_team_sequence_model_review as sequence_review


REPORT_PATH = Path("outputs/reports/e0_away_ah_sequence_transformer_n5_no_seq_odds_falsification_review.md")
SUMMARY_PATH = Path("outputs/reports/e0_away_ah_sequence_transformer_n5_no_seq_odds_falsification_summary.csv")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/sequence_transformer_n5_no_seq_odds_falsification")

NO_SEQ_ODDS_COLUMNS = tuple(
    column for column in sequence_review.SEQUENCE_FEATURE_COLUMNS if column not in base.SEQUENCE_ODDS_COLUMNS
)


def no_seq_config(name: str, **kwargs) -> base.VariantConfig:
    kwargs.setdefault("sequence_columns", NO_SEQ_ODDS_COLUMNS)
    return base.VariantConfig(name, **kwargs)


def load_benchmark_rows_with_original() -> pd.DataFrame:
    benchmarks = base.load_benchmark_rows()
    original_path = Path("outputs/reports/e0_away_ah_sequence_transformer_n5_falsification_summary.csv")
    if original_path.exists():
        original = pd.read_csv(original_path)
        original = original[original["strategy"].isin(["locked_ensemble", "locked_individual_seed_mean"])].copy()
        original["strategy"] = original["strategy"].map(
            {
                "locked_ensemble": "original_sequence_transformer_n5_market_residual_ensemble",
                "locked_individual_seed_mean": "original_sequence_transformer_n5_market_residual_seed_mean",
            }
        )
        original["variant"] = "original_locked_benchmark"
        benchmarks = pd.concat([benchmarks, original], ignore_index=True, sort=False)
    return benchmarks


def main() -> None:
    base.REPORT_PATH = REPORT_PATH
    base.SUMMARY_PATH = SUMMARY_PATH
    base.DETAIL_DIR = DETAIL_DIR
    dataframe = advanced.prepare_e0_data()
    variants = [
        no_seq_config("locked_ensemble", description="locked no sequence odds/AH seed ensemble"),
        no_seq_config("sequence_only_current_market_feature_check", numeric_columns=tuple(base.MARKET_CURRENT_COLUMNS), categorical_columns=()),
        no_seq_config("no_team_categorical_embeddings_feature_check", numeric_columns=tuple(advanced.NUMERIC_FEATURE_COLUMNS), categorical_columns=()),
        no_seq_config(
            "no_internal_elo_feature_check",
            numeric_columns=tuple(c for c in advanced.NUMERIC_FEATURE_COLUMNS if c not in base.INTERNAL_ELO_COLUMNS),
            categorical_columns=tuple(advanced.CATEGORICAL_FEATURE_COLUMNS),
            sequence_columns=tuple(c for c in NO_SEQ_ODDS_COLUMNS if c not in base.SEQUENCE_ELO_COLUMNS),
        ),
        no_seq_config(
            "no_weather_travel_rest_feature_check",
            numeric_columns=tuple(c for c in advanced.NUMERIC_FEATURE_COLUMNS if c not in base.CONTEXT_COLUMNS),
            categorical_columns=tuple(advanced.CATEGORICAL_FEATURE_COLUMNS),
            sequence_columns=tuple(c for c in NO_SEQ_ODDS_COLUMNS if c not in base.SEQUENCE_CONTEXT_COLUMNS),
        ),
        no_seq_config(
            "no_memory_score_feature_check",
            numeric_columns=tuple(c for c in advanced.NUMERIC_FEATURE_COLUMNS if c != "memory_score_knn_profit"),
            categorical_columns=tuple(advanced.CATEGORICAL_FEATURE_COLUMNS),
        ),
        no_seq_config(
            "no_current_team_names_feature_check",
            numeric_columns=tuple(advanced.NUMERIC_FEATURE_COLUMNS),
            categorical_columns=(),
        ),
        no_seq_config("shuffled_train_labels_negative_control", shuffle_train_labels=True),
        no_seq_config("random_sequence_order_negative_control", random_sequence_order=True),
        no_seq_config("random_sequence_rows_same_team_negative_control", random_sequence_rows="same_team"),
        no_seq_config("random_sequence_rows_any_team_negative_control", random_sequence_rows="any_team"),
        base.VariantConfig("current_tabular_only_no_sequence_negative_control", sequence_columns=(), zero_sequences=True),
    ]

    by_year_frames = []
    bet_frames = []
    metric_frames = []
    overall_rows = []
    for config in variants:
        by_year, bets, metrics = base.run_variant(dataframe, config)
        by_year_frames.append(by_year)
        if len(bets):
            bet_frames.append(bets)
        overall_rows.append(base.row_for_bets(f"{config.name}_ensemble", bets, base.LOCKED_MODEL_TYPE, config.name))
        if len(metrics):
            metric_frames.append(metrics)

    benchmarks = load_benchmark_rows_with_original()
    benchmarks["variant"] = benchmarks["variant"].fillna("benchmark") if "variant" in benchmarks else "benchmark"
    overall_rows.extend(benchmarks.to_dict("records"))

    summary = pd.DataFrame(overall_rows)
    by_year = pd.concat(by_year_frames, ignore_index=True, sort=False)
    bets = pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame()
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()
    locked_bets = bets[bets["strategy"].eq("locked_ensemble_ensemble")].copy()
    locked_bets["strategy"] = "locked_ensemble"
    summary.loc[summary["strategy"].eq("locked_ensemble_ensemble"), "strategy"] = "locked_ensemble"
    seasonal = advanced.seasonal_rows(locked_bets)
    season_exclusion = base.season_exclusions("locked_ensemble", locked_bets)
    team_exclusion = base.home_team_exclusions("locked_ensemble", locked_bets)
    negative = summary[summary["variant"].astype(str).str.contains("negative_control", na=False)].copy()
    audit = base.leakage_audit(dataframe, variants[0])
    classification, rationale = base.classify(summary, season_exclusion, team_exclusion, negative, audit)
    base.write_outputs(summary, by_year, bets, metrics, seasonal, season_exclusion, team_exclusion, audit, classification, rationale)

    text = REPORT_PATH.read_text(encoding="utf-8")
    text = text.replace(
        "# E0 Sequence Transformer N=5 Market-Residual Falsification Review",
        "# E0 Sequence Transformer N=5 No-Sequence-Odds/AH Falsification Review",
    )
    text = text.replace(
        "Locked candidate: `sequence_transformer`, N=5, target=`market_residual`, seeds=[11, 23, 37].",
        "Locked candidate: `sequence_transformer`, N=5, target=`market_residual`, seeds=[11, 23, 37], sequence odds/AH features removed.",
    )
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(REPORT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
