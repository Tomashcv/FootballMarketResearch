from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.modeling.market_residual import CandidateSpec, FitMetadata, fit_candidate
from src.modeling.probability import (
    apply_blend_temperature,
    fit_blend_temperature,
    market_probs,
    probability_metrics,
)
from src.modeling.temporal import build_nested_year_folds, split_nested_fold
from src.modeling.uncertainty import cluster_bootstrap_profit
from src.modeling.v3_features import (
    add_research_derived_features,
    build_v3_adapter,
    feature_coverage,
    feature_list_sha256,
    load_feature_contract,
    resolve_feature_group,
)
from src.modeling.value_selection import (
    add_value_columns,
    apply_rule,
    build_rule_grid,
    max_drawdown,
    rule_metrics,
    select_rule_on_validation,
    z_score,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/v3_next_research.json"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_current_season_exact() -> pd.DataFrame:
    # Reuse the already-audited current-season build. This does not fit on current outcomes.
    from scripts.run_v3_2025_validation import (
        add_internal_elo_with_history,
        build_external_exact,
        build_market_dataset,
        load_raw_2025_norm,
    )

    _inventory, norm = load_raw_2025_norm()
    if norm.empty:
        return pd.DataFrame()
    market, _marketed, _valid_rows, _conflicts, _implausible = build_market_dataset(norm)
    if market.empty:
        return pd.DataFrame()
    external = build_external_exact(market)
    return add_internal_elo_with_history(external)


def load_research_data(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
    historical_path = ROOT / config["paths"]["historical_exact_input"]
    contract_path = ROOT / config["paths"]["feature_contract"]
    if not historical_path.exists():
        raise FileNotFoundError(f"Historical exact input missing: {historical_path}")
    historical = pd.read_csv(historical_path, low_memory=False)
    raw = historical
    warnings: list[str] = []
    if bool(config.get("include_current_season", True)):
        try:
            current = _load_current_season_exact()
            if not current.empty:
                raw = pd.concat([historical, current], ignore_index=True, sort=False)
                raw = raw.drop_duplicates("full_scope_match_id", keep="last")
            else:
                warnings.append("Current-season exact rows were unavailable; research uses historical exact rows only.")
        except Exception as exc:  # current data is optional; historical audit must remain runnable
            warnings.append(f"Current-season exact build failed: {type(exc).__name__}: {exc}")
    features = load_feature_contract(contract_path)
    expected_hash = str(config.get("feature_contract_sha256", "")).strip()
    actual_hash = feature_list_sha256(features)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(f"Feature contract hash mismatch: expected {expected_hash}, got {actual_hash}")
    adapter = build_v3_adapter(raw, features, require_target=True)
    adapter, derived = add_research_derived_features(adapter)
    return adapter, features + derived, warnings


def _candidate_specs(config: dict[str, Any]) -> list[CandidateSpec]:
    specs = []
    for row in config["model_grid"]:
        specs.append(
            CandidateSpec(
                name=str(row["name"]),
                family=str(row["family"]),
                feature_group=str(row["feature_group"]),
                params=dict(row.get("params", {})),
                recency_half_life_years=row.get("recency_half_life_years"),
                league_balance_strength=float(row.get("league_balance_strength", 0.0)),
            )
        )
    return specs


def _selection_gate(model_metrics: dict[str, float], market_metrics: dict[str, float], gate: dict[str, float]) -> tuple[bool, dict[str, float]]:
    delta = {
        "delta_log_loss": model_metrics["log_loss"] - market_metrics["log_loss"],
        "delta_brier": model_metrics["brier"] - market_metrics["brier"],
        "delta_ece": model_metrics["ece"] - market_metrics["ece"],
        "delta_accuracy": model_metrics["accuracy"] - market_metrics["accuracy"],
    }
    passed = (
        delta["delta_log_loss"] < float(gate.get("max_delta_log_loss", 0.0))
        and delta["delta_brier"] <= float(gate.get("max_delta_brier", 0.0005))
        and delta["delta_ece"] <= float(gate.get("max_delta_ece", 0.01))
    )
    return bool(passed), delta


def _prediction_frame(part: pd.DataFrame, model_prob: np.ndarray, candidate_name: str, alpha: float, temperature: float) -> pd.DataFrame:
    market = market_probs(part)
    out = part[
        [
            "match_id",
            "full_scope_match_id",
            "canonical_match_id",
            "logical_match_key",
            "source_file",
            "match_date",
            "league",
            "season_start_year",
            "season_end_year",
            "home_team",
            "away_team",
            "target_y",
            "target_outcome_1x2",
            "x1_odds_source",
        ]
    ].copy()
    out[["market_home_prob", "market_draw_prob", "market_away_prob"]] = market
    out[["model_home_prob", "model_draw_prob", "model_away_prob"]] = model_prob
    out["odds_home"] = pd.to_numeric(part["x1x2_avg_odds_home"], errors="coerce").to_numpy()
    out["odds_draw"] = pd.to_numeric(part["x1x2_avg_odds_draw"], errors="coerce").to_numpy()
    out["odds_away"] = pd.to_numeric(part["x1x2_avg_odds_away"], errors="coerce").to_numpy()
    out["candidate_name"] = candidate_name
    out["calibration_alpha"] = float(alpha)
    out["calibration_temperature"] = float(temperature)
    return add_value_columns(out)


def _summary_from_predictions(predictions: pd.DataFrame) -> dict[str, float | int]:
    if predictions.empty:
        return {}
    y = predictions["target_y"].to_numpy(dtype=int)
    model = predictions[["model_home_prob", "model_draw_prob", "model_away_prob"]].to_numpy(dtype=float)
    market = predictions[["market_home_prob", "market_draw_prob", "market_away_prob"]].to_numpy(dtype=float)
    mm = probability_metrics(y, model)
    bm = probability_metrics(y, market)
    return {
        "prediction_rows": int(len(predictions)),
        **{f"model_{k}": v for k, v in mm.items()},
        **{f"market_{k}": v for k, v in bm.items()},
        "delta_log_loss": mm["log_loss"] - bm["log_loss"],
        "delta_brier": mm["brier"] - bm["brier"],
        "delta_ece": mm["ece"] - bm["ece"],
        "delta_accuracy": mm["accuracy"] - bm["accuracy"],
    }


def _selected_summary(selected: pd.DataFrame) -> dict[str, float | int]:
    if selected.empty:
        return {
            "bets": 0,
            "profit": 0.0,
            "roi": 0.0,
            "z_score": 0.0,
            "max_drawdown": 0.0,
            "positive_test_years": 0,
            "positive_leagues": 0,
            "best_year_profit_share": 0.0,
            "best_league_profit_share": 0.0,
        }
    profit = pd.to_numeric(selected["profit"], errors="coerce").fillna(0.0)
    by_year = selected.assign(_profit=profit).groupby("season_start_year")["_profit"].sum()
    by_league = selected.assign(_profit=profit).groupby("league")["_profit"].sum()
    positive_year = by_year[by_year > 0]
    positive_league = by_league[by_league > 0]
    total = float(profit.sum())
    return {
        "bets": int(len(selected)),
        "profit": total,
        "roi": total / len(selected),
        "z_score": z_score(profit),
        "max_drawdown": max_drawdown(profit),
        "positive_test_years": int((by_year > 0).sum()),
        "positive_leagues": int((by_league > 0).sum()),
        "best_year_profit_share": float(positive_year.max() / positive_year.sum()) if len(positive_year) and positive_year.sum() > 0 else 0.0,
        "best_league_profit_share": float(positive_league.max() / positive_league.sum()) if len(positive_league) and positive_league.sum() > 0 else 0.0,
    }


def run(config_path: Path = DEFAULT_CONFIG, quick: bool = False) -> str:
    config = load_config(config_path)
    output_dir = ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        data, all_features, warnings = load_research_data(config)
    except Exception as exc:
        decision = "v3_next_data_not_ready"
        (output_dir / "v3_next_decision.md").write_text(f"# V3 Next Decision\n\n`{decision}`\n\n{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise

    data = data[~data["league"].isin(set(config.get("excluded_leagues", ["E1", "E2", "E3"])))].copy()
    specs = _candidate_specs(config)
    if quick:
        specs = specs[: min(3, len(specs))]
    feature_groups = {name: resolve_feature_group(all_features, name) for name in sorted({spec.feature_group for spec in specs})}
    coverage_parts = []
    for name, cols in feature_groups.items():
        part = feature_coverage(data, cols)
        part.insert(0, "feature_group", name)
        coverage_parts.append(part)
    coverage_df = pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()
    coverage_df.to_csv(output_dir / "v3_next_feature_coverage.csv", index=False)

    folds = build_nested_year_folds(
        data,
        [int(y) for y in config["outer_test_years"]],
        tune_lag_years=int(config.get("tune_lag_years", 2)),
        calibration_lag_years=int(config.get("calibration_lag_years", 1)),
        min_train_rows=int(config.get("min_train_rows", 1000)),
    )
    if not folds:
        decision = "v3_next_data_not_ready"
        (output_dir / "v3_next_decision.md").write_text(f"# V3 Next Decision\n\n`{decision}`\n\nNo valid nested temporal folds.\n", encoding="utf-8")
        return decision

    rules_cfg = config["value_rule_grid"]
    rules = build_rule_grid(
        sides=rules_cfg.get("sides", ["away"]),
        edge_thresholds=rules_cfg["edge_thresholds"],
        odds_minima=rules_cfg["odds_minima"],
        odds_maxima=rules_cfg["odds_maxima"],
    )

    model_selection_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    rule_rows: list[pd.DataFrame] = []
    test_predictions: list[pd.DataFrame] = []
    selected_bets: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for fold in folds:
        train, tune, calibration, test = split_nested_fold(data, fold)
        tune_market = market_probs(tune)
        tune_market_metrics = probability_metrics(tune["target_y"].to_numpy(dtype=int), tune_market)
        candidates = []
        for spec in specs:
            cols = feature_groups[spec.feature_group]
            try:
                fitted, tune_prob, metadata = fit_candidate(spec, train, tune, cols)
                tune_metrics = probability_metrics(tune["target_y"].to_numpy(dtype=int), tune_prob)
                passed, delta = _selection_gate(tune_metrics, tune_market_metrics, config["predictive_gate"])
                row = {
                    "test_year": fold.test_year,
                    "tune_year": fold.tune_year,
                    "calibration_year": fold.calibration_year,
                    "candidate_name": spec.name,
                    "family": spec.family,
                    "feature_group": spec.feature_group,
                    "passes_predictive_gate": passed,
                    **asdict(metadata),
                    **{f"tune_{k}": v for k, v in tune_metrics.items()},
                    **{f"tune_market_{k}": v for k, v in tune_market_metrics.items()},
                    **delta,
                }
                candidates.append((spec, metadata, row))
                model_selection_rows.append(row)
            except Exception as exc:
                model_selection_rows.append(
                    {
                        "test_year": fold.test_year,
                        "tune_year": fold.tune_year,
                        "calibration_year": fold.calibration_year,
                        "candidate_name": spec.name,
                        "family": spec.family,
                        "feature_group": spec.feature_group,
                        "passes_predictive_gate": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        passing = [item for item in candidates if item[2].get("passes_predictive_gate")]
        if passing:
            selected_spec, selected_meta, selected_row = min(
                passing,
                key=lambda item: (
                    item[2]["tune_log_loss"],
                    item[2]["tune_brier"],
                    item[2]["tune_ece"],
                    item[2]["feature_count"],
                ),
            )
            selection_reason = "passed_tune_predictive_gate"
        else:
            raw_specs = [spec for spec in specs if spec.family == "raw_market"]
            selected_spec = raw_specs[0] if raw_specs else CandidateSpec("raw_market_fallback", "raw_market", "market_only", {})
            selected_meta = FitMetadata(selected_spec.name, selected_spec.family, 0, len(train), len(tune), None, True, None, 0.0)
            selection_reason = "no_model_passed_tune_gate_market_fallback"

        train_tune = pd.concat([train, tune], ignore_index=True, sort=False)
        selected_features = feature_groups.get(selected_spec.feature_group, [])
        calibration_model, calibration_raw_prob, calibration_meta = fit_candidate(
            selected_spec,
            train_tune,
            calibration,
            selected_features,
            fixed_rounds=selected_meta.best_rounds,
        )
        calibration_market = market_probs(calibration)
        blend = fit_blend_temperature(
            calibration["target_y"].to_numpy(dtype=int),
            calibration_raw_prob,
            calibration_market,
            config["calibration_grid"]["alphas"],
            config["calibration_grid"]["temperatures"],
        )
        calibrated_prob = apply_blend_temperature(calibration_raw_prob, calibration_market, blend.alpha, blend.temperature)
        calibrated_metrics = probability_metrics(calibration["target_y"].to_numpy(dtype=int), calibrated_prob)
        calibration_market_metrics = probability_metrics(calibration["target_y"].to_numpy(dtype=int), calibration_market)
        if calibrated_metrics["log_loss"] >= calibration_market_metrics["log_loss"]:
            # No forced model use. A fold that cannot beat the market in calibration becomes market-only.
            alpha = 0.0
            temperature = 1.0
            calibrated_prob = calibration_market
            calibrated_metrics = calibration_market_metrics
            calibration_reason = "calibration_did_not_beat_market_market_fallback"
        else:
            alpha = blend.alpha
            temperature = blend.temperature
            calibration_reason = "calibrated_model_beats_market"
        calibration_rows.append(
            {
                "test_year": fold.test_year,
                "candidate_name": selected_spec.name,
                "selection_reason": selection_reason,
                "calibration_reason": calibration_reason,
                "alpha": alpha,
                "temperature": temperature,
                **{f"calibration_{k}": v for k, v in calibrated_metrics.items()},
                **{f"calibration_market_{k}": v for k, v in calibration_market_metrics.items()},
                "calibration_delta_log_loss": calibrated_metrics["log_loss"] - calibration_market_metrics["log_loss"],
                "calibration_delta_brier": calibrated_metrics["brier"] - calibration_market_metrics["brier"],
                "best_rounds": calibration_meta.best_rounds,
            }
        )

        calibration_frame = _prediction_frame(calibration, calibrated_prob, selected_spec.name, alpha, temperature)
        selection, rule_table = select_rule_on_validation(
            calibration_frame,
            rules,
            min_bets=int(rules_cfg.get("min_validation_bets", 40)),
            require_positive_lcb=bool(rules_cfg.get("require_positive_lcb", False)),
            max_positive_league_share=float(rules_cfg.get("max_positive_league_share", 0.70)),
            minimum_positive_leagues=int(rules_cfg.get("minimum_positive_leagues", 2)),
        )
        rule_table.insert(0, "test_year", fold.test_year)
        rule_rows.append(rule_table)

        pretest = pd.concat([train, tune, calibration], ignore_index=True, sort=False)
        final_model, _unused, final_meta = fit_candidate(
            selected_spec,
            pretest,
            None,
            selected_features,
            fixed_rounds=selected_meta.best_rounds,
        )
        test_raw_prob = final_model.predict_proba(test)
        test_market = market_probs(test)
        test_calibrated = apply_blend_temperature(test_raw_prob, test_market, alpha, temperature)
        test_frame = _prediction_frame(test, test_calibrated, selected_spec.name, alpha, temperature)
        test_frame["fold_test_year"] = fold.test_year
        test_frame["fold_tune_year"] = fold.tune_year
        test_frame["fold_calibration_year"] = fold.calibration_year
        test_frame["model_selection_reason"] = selection_reason
        test_frame["rule_selection_reason"] = selection.reason
        test_predictions.append(test_frame)

        fold_selected = apply_rule(test_frame, selection.rule) if selection.rule is not None else pd.DataFrame(columns=list(test_frame.columns) + ["profit"])
        if not fold_selected.empty:
            fold_selected["fold_test_year"] = fold.test_year
            selected_bets.append(fold_selected)
        fold_test_model_metrics = probability_metrics(test["target_y"].to_numpy(dtype=int), test_calibrated)
        fold_test_market_metrics = probability_metrics(test["target_y"].to_numpy(dtype=int), test_market)
        selected_metrics = rule_metrics(fold_selected)
        fold_rows.append(
            {
                "test_year": fold.test_year,
                "tune_year": fold.tune_year,
                "calibration_year": fold.calibration_year,
                "selected_candidate": selected_spec.name,
                "selected_family": selected_spec.family,
                "selected_feature_group": selected_spec.feature_group,
                "selection_reason": selection_reason,
                "calibration_alpha": alpha,
                "calibration_temperature": temperature,
                "selected_rule": selection.rule.name if selection.rule else "NO_BET",
                "rule_selection_reason": selection.reason,
                "test_rows": len(test),
                "final_feature_count": final_meta.feature_count,
                "final_rounds": final_meta.best_rounds,
                **{f"test_{k}": v for k, v in fold_test_model_metrics.items()},
                **{f"test_market_{k}": v for k, v in fold_test_market_metrics.items()},
                "test_delta_log_loss": fold_test_model_metrics["log_loss"] - fold_test_market_metrics["log_loss"],
                "test_delta_brier": fold_test_model_metrics["brier"] - fold_test_market_metrics["brier"],
                "test_delta_ece": fold_test_model_metrics["ece"] - fold_test_market_metrics["ece"],
                **{f"test_value_{k}": v for k, v in selected_metrics.items()},
            }
        )

    model_selection = pd.DataFrame(model_selection_rows)
    calibrations = pd.DataFrame(calibration_rows)
    fold_summary = pd.DataFrame(fold_rows)
    predictions = pd.concat(test_predictions, ignore_index=True, sort=False) if test_predictions else pd.DataFrame()
    bets = pd.concat(selected_bets, ignore_index=True, sort=False) if selected_bets else pd.DataFrame()
    rule_grid = pd.concat(rule_rows, ignore_index=True, sort=False) if rule_rows else pd.DataFrame()

    model_selection.to_csv(output_dir / "v3_next_model_selection.csv", index=False)
    calibrations.to_csv(output_dir / "v3_next_calibration.csv", index=False)
    fold_summary.to_csv(output_dir / "v3_next_fold_summary.csv", index=False)
    predictions.to_csv(output_dir / "v3_next_test_predictions.csv", index=False)
    bets.to_csv(output_dir / "v3_next_selected_bets.csv", index=False)
    rule_grid.to_csv(output_dir / "v3_next_rule_grid_validation.csv", index=False)

    predictive_summary = _summary_from_predictions(predictions)
    value_summary = _selected_summary(bets)
    bootstrap = cluster_bootstrap_profit(
        bets,
        cluster_col="season_start_year",
        iterations=int(config.get("bootstrap_iterations", 2000)),
        seed=int(config.get("seed", 17)),
    )
    overall = {**predictive_summary, **value_summary, **{f"bootstrap_{k}": v for k, v in bootstrap.items()}}
    pd.DataFrame([overall]).to_csv(output_dir / "v3_next_overall_summary.csv", index=False)

    if not bets.empty:
        by_year = []
        for year, group in bets.groupby("season_start_year"):
            by_year.append({"season_start_year": year, **rule_metrics(group)})
        by_league = []
        for league, group in bets.groupby("league"):
            by_league.append({"league": league, **rule_metrics(group)})
        pd.DataFrame(by_year).to_csv(output_dir / "v3_next_by_year.csv", index=False)
        pd.DataFrame(by_league).to_csv(output_dir / "v3_next_by_league.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "v3_next_by_year.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "v3_next_by_league.csv", index=False)

    improved_folds = int((fold_summary.get("test_delta_log_loss", pd.Series(dtype=float)) < 0).sum()) if not fold_summary.empty else 0
    if not predictive_summary or predictive_summary.get("delta_log_loss", 0.0) >= 0 or predictive_summary.get("delta_brier", 0.0) > 0:
        decision = "v3_next_no_predictive_gain"
    elif value_summary["bets"] == 0 or value_summary["profit"] <= 0:
        decision = "v3_next_predictive_gain_no_value"
    elif (
        value_summary["z_score"] >= 1.0
        and value_summary["positive_test_years"] >= 2
        and value_summary["positive_leagues"] >= 3
        and bootstrap["prob_profit_positive"] >= 0.80
    ):
        decision = "v3_next_value_candidate_research_only"
    else:
        decision = "v3_next_predictive_gain_no_value"
    if (
        decision == "v3_next_value_candidate_research_only"
        and value_summary["best_year_profit_share"] <= 0.45
        and value_summary["best_league_profit_share"] <= 0.45
        and improved_folds >= max(2, math.ceil(len(fold_summary) / 2))
        and bootstrap["prob_profit_positive"] >= 0.90
    ):
        decision = "v3_next_challenger_for_frozen_forward_test_research_only"

    (output_dir / "v3_next_decision.md").write_text(
        "# V3 Next Decision\n\n"
        f"`{decision}`\n\n"
        "Research only. The frozen V3 paper candidate was not modified. No confirmed edge is claimed.\n",
        encoding="utf-8",
    )

    report_lines = [
        "# V3 Next Nested Research",
        "",
        f"Decision: `{decision}`",
        "",
        "The frozen V3 and its paper pipeline remain untouched. This challenger uses train -> tune -> calibration/rule-selection -> test chronology for every outer fold.",
        "",
        "## Predictive summary",
        "```",
        pd.DataFrame([predictive_summary]).to_string(index=False) if predictive_summary else "No predictions",
        "```",
        "",
        "## Value summary",
        "```",
        pd.DataFrame([value_summary]).to_string(index=False),
        "```",
        "",
        "## Cluster bootstrap",
        "```",
        pd.DataFrame([bootstrap]).to_string(index=False),
        "```",
        "",
        "## Fold summary",
        "```",
        fold_summary.to_string(index=False) if not fold_summary.empty else "No folds",
        "```",
        "",
        "## Warnings",
    ]
    report_lines += [f"- {warning}" for warning in warnings] if warnings else ["- none"]
    report_lines += [
        "",
        "## Guardrails",
        "- No test-year tuning.",
        "- No current-season outcomes are used for fitting that same season.",
        "- Model use is optional per fold; failure to beat the market triggers market fallback.",
        "- Probability shrinkage/temperature and value rule are selected only on the calibration year.",
        "- No automatic promotion to paper or live betting.",
    ]
    (output_dir / "v3_next_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return decision
