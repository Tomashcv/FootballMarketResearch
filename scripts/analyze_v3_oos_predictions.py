from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HIST = ROOT / "outputs/reports/v3_exact_reproduction/v3_exact_row_predictions.csv"
DEFAULT_CURRENT = ROOT / "outputs/reports/v3_2025_validation/v3_2025_row_predictions.csv"
DEFAULT_OUT = ROOT / "outputs/reports/v3_oos_diagnostic"


def normalize(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), 1e-8, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def brier(y: np.ndarray, prob: np.ndarray) -> float:
    p = normalize(prob)
    one = np.zeros_like(p)
    one[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one) ** 2, axis=1)))


def ece(y: np.ndarray, prob: np.ndarray, bins: int = 15) -> float:
    p = normalize(prob)
    confidence = p.max(axis=1)
    prediction = p.argmax(axis=1)
    correct = (prediction == y).astype(float)
    result = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(result)


def predictive_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["target_y"].to_numpy(dtype=int)
    market = frame[["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]].to_numpy(dtype=float)
    model = frame[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
    market_ll = float(log_loss(y, normalize(market), labels=[0, 1, 2]))
    model_ll = float(log_loss(y, normalize(model), labels=[0, 1, 2]))
    return {
        "rows": int(len(frame)),
        "market_log_loss": market_ll,
        "model_log_loss": model_ll,
        "delta_log_loss": model_ll - market_ll,
        "market_brier": brier(y, market),
        "model_brier": brier(y, model),
        "delta_brier": brier(y, model) - brier(y, market),
        "market_ece": ece(y, market),
        "model_ece": ece(y, model),
        "delta_ece": ece(y, model) - ece(y, market),
    }


def z_score(profit: pd.Series) -> float:
    values = pd.to_numeric(profit, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    sd = float(values.std(ddof=1))
    return float(values.sum() / (sd * np.sqrt(len(values)))) if sd > 0 else 0.0


def bet_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"bets": 0, "profit": 0.0, "roi": 0.0, "z_score": 0.0, "average_odds": np.nan, "average_edge": np.nan}
    profit = pd.to_numeric(frame["away_profit"], errors="coerce").fillna(0.0)
    total = float(profit.sum())
    return {
        "bets": int(len(frame)),
        "profit": total,
        "roi": total / len(frame),
        "z_score": z_score(profit),
        "average_odds": float(frame["x1x2_avg_odds_away"].mean()),
        "average_edge": float(frame["away_edge"].mean()),
    }


def fixed_v3_selection(frame: pd.DataFrame) -> pd.DataFrame:
    threshold = np.where(frame["season_start_year"].le(2022), 0.01, 0.015)
    return frame[frame["away_edge"].ge(threshold) & frame["x1x2_avg_odds_away"].ge(1.5)].copy()


def apply_rule(frame: pd.DataFrame, edge: float, odds_min: float, odds_max: float | None) -> pd.DataFrame:
    mask = frame["away_edge"].ge(edge) & frame["x1x2_avg_odds_away"].ge(odds_min)
    if odds_max is not None:
        mask &= frame["x1x2_avg_odds_away"].le(odds_max)
    return frame[mask].copy()


def conservative_rule_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    metrics = bet_metrics(frame)
    if frame.empty or len(frame) <= 1:
        return {**metrics, "lcb_mean_profit": -np.inf, "positive_years": 0, "positive_leagues": 0, "max_positive_league_share": 1.0}
    profit = frame["away_profit"]
    se = float(profit.std(ddof=1) / np.sqrt(len(frame)))
    by_year = frame.groupby("season_start_year")["away_profit"].sum()
    by_league = frame.groupby("league")["away_profit"].sum()
    positive = by_league[by_league > 0]
    share = float(positive.max() / positive.sum()) if len(positive) and positive.sum() > 0 else 1.0
    return {
        **metrics,
        "lcb_mean_profit": float(profit.mean() - se),
        "positive_years": int((by_year > 0).sum()),
        "positive_leagues": int((by_league > 0).sum()),
        "max_positive_league_share": share,
    }


def nested_meta_rule(frame: pd.DataFrame) -> pd.DataFrame:
    grid = [
        (edge, odds_min, odds_max)
        for edge in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        for odds_min in [1.5, 1.7, 1.9]
        for odds_max in [None, 2.25, 2.75, 3.5]
        if odds_max is None or odds_max >= odds_min
    ]
    rows = []
    for test_year in [2023, 2024, 2025]:
        validation = frame[frame["season_start_year"].lt(test_year)]
        candidates = []
        for edge, odds_min, odds_max in grid:
            selected = apply_rule(validation, edge, odds_min, odds_max)
            stats = conservative_rule_stats(selected)
            if (
                stats["bets"] >= 150
                and stats["positive_years"] >= 2
                and stats["positive_leagues"] >= 3
                and stats["max_positive_league_share"] <= 0.60
            ):
                candidates.append((stats["lcb_mean_profit"], stats["roi"], stats["bets"], edge, odds_min, odds_max, stats))
        if not candidates:
            rows.append({"test_year": test_year, "selected_rule": "NO_RULE", "reason": "no_prior_oos_rule_passed_gate"})
            continue
        best = max(candidates, key=lambda row: (row[0], row[1], row[2]))
        _, _, _, edge, odds_min, odds_max, validation_stats = best
        test_selected = apply_rule(frame[frame["season_start_year"].eq(test_year)], edge, odds_min, odds_max)
        test_stats = conservative_rule_stats(test_selected)
        rows.append(
            {
                "test_year": test_year,
                "selected_rule": f"away_edge_{edge}_odds_{odds_min}_{odds_max}",
                "edge_min": edge,
                "odds_min": odds_min,
                "odds_max": odds_max,
                **{f"validation_{key}": value for key, value in validation_stats.items()},
                **{f"test_{key}": value for key, value in test_stats.items()},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose existing out-of-sample V3 probabilities without retraining or threshold mining on test years.")
    parser.add_argument("--historical", type=Path, default=DEFAULT_HIST)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.historical.exists():
        raise FileNotFoundError(args.historical)
    frames = [pd.read_csv(args.historical)]
    if args.current.exists():
        frames.append(pd.read_csv(args.current))
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["season_start_year"] = pd.to_numeric(data["season_start_year"], errors="coerce").astype(int)
    data["away_profit"] = np.where(data["target_y"].eq(2), data["x1x2_avg_odds_away"] - 1.0, -1.0)
    args.out.mkdir(parents=True, exist_ok=True)

    overall_predictive = pd.DataFrame([predictive_metrics(data)])
    by_year_predictive = pd.DataFrame([{"season_start_year": year, **predictive_metrics(group)} for year, group in data.groupby("season_start_year")])
    by_league_predictive = pd.DataFrame([{"league": league, **predictive_metrics(group)} for league, group in data.groupby("league")]).sort_values("delta_log_loss")
    fixed = fixed_v3_selection(data)
    fixed_overall = pd.DataFrame([bet_metrics(fixed)])
    fixed_by_year = pd.DataFrame([{"season_start_year": year, **bet_metrics(group)} for year, group in fixed.groupby("season_start_year")])
    fixed_by_league = pd.DataFrame([{"league": league, **bet_metrics(group)} for league, group in fixed.groupby("league")]).sort_values("profit", ascending=False)
    nested = nested_meta_rule(data)

    overall_predictive.to_csv(args.out / "v3_oos_predictive_overall.csv", index=False)
    by_year_predictive.to_csv(args.out / "v3_oos_predictive_by_year.csv", index=False)
    by_league_predictive.to_csv(args.out / "v3_oos_predictive_by_league.csv", index=False)
    fixed_overall.to_csv(args.out / "v3_oos_fixed_rule_overall.csv", index=False)
    fixed_by_year.to_csv(args.out / "v3_oos_fixed_rule_by_year.csv", index=False)
    fixed_by_league.to_csv(args.out / "v3_oos_fixed_rule_by_league.csv", index=False)
    nested.to_csv(args.out / "v3_oos_nested_meta_rule.csv", index=False)

    o = overall_predictive.iloc[0]
    b = fixed_overall.iloc[0]
    report = [
        "# V3 Existing OOS Diagnostic",
        "",
        "This report uses only already out-of-sample yearly V3 predictions. It does not retrain the model or claim a confirmed edge.",
        "",
        "## Main finding",
        f"- Predictive improvement exists across the combined sample: delta log loss {o['delta_log_loss']:.6f}, delta Brier {o['delta_brier']:.6f}, delta ECE {o['delta_ece']:.6f} on {int(o['rows'])} rows.",
        f"- The historical/current fixed away rule has {int(b['bets'])} bets, {b['profit']:.2f}u profit, ROI {b['roi']:.2%}, z {b['z_score']:.3f}.",
        "- Therefore the bottleneck is no longer only prediction. It is converting a small, consistently better probability forecast into a stable decision rule without overfitting thresholds.",
        "",
        "## Predictive performance by season",
        "```",
        by_year_predictive.to_string(index=False),
        "```",
        "",
        "## Predictive performance by league",
        "```",
        by_league_predictive.to_string(index=False),
        "```",
        "",
        "## Fixed rule by season",
        "```",
        fixed_by_year.to_string(index=False),
        "```",
        "",
        "## Fixed rule by league",
        "```",
        fixed_by_league.to_string(index=False),
        "```",
        "",
        "## Prior-OOS-only threshold meta-test",
        "```",
        nested.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "- Simple post-hoc odds caps/threshold changes are not a reliable solution: rules that look strong on prior OOS seasons can still fail in the next season.",
        "- The next research should focus on model shrinkage/calibration, recency weighting, feature-block ablation and uncertainty-aware selection, all nested before the test season.",
        "- League-specific models should not be promoted merely because I1/F1/P1 were profitable; league-specific predictive and value gates must be passed prospectively.",
    ]
    (args.out / "v3_oos_diagnostic.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("v3_oos_diagnostic_ready")


if __name__ == "__main__":
    main()
