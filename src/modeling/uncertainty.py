from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_bootstrap_profit(
    selected: pd.DataFrame,
    cluster_col: str = "season_start_year",
    iterations: int = 2000,
    seed: int = 17,
) -> dict[str, float | int]:
    if selected.empty:
        return {
            "iterations": int(iterations),
            "clusters": 0,
            "profit_mean": 0.0,
            "profit_ci_low": 0.0,
            "profit_ci_high": 0.0,
            "roi_mean": 0.0,
            "roi_ci_low": 0.0,
            "roi_ci_high": 0.0,
            "prob_profit_positive": 0.0,
        }
    if cluster_col not in selected.columns:
        raise KeyError(f"Missing bootstrap cluster column: {cluster_col}")
    groups = {key: group.copy() for key, group in selected.groupby(cluster_col, dropna=False)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    profits = np.empty(iterations, dtype=float)
    rois = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([groups[key] for key in sampled_keys], ignore_index=True)
        profit = pd.to_numeric(sample["profit"], errors="coerce").fillna(0.0)
        profits[i] = float(profit.sum())
        rois[i] = float(profit.mean()) if len(profit) else 0.0
    return {
        "iterations": int(iterations),
        "clusters": int(len(keys)),
        "profit_mean": float(profits.mean()),
        "profit_ci_low": float(np.quantile(profits, 0.025)),
        "profit_ci_high": float(np.quantile(profits, 0.975)),
        "roi_mean": float(rois.mean()),
        "roi_ci_low": float(np.quantile(rois, 0.025)),
        "roi_ci_high": float(np.quantile(rois, 0.975)),
        "prob_profit_positive": float((profits > 0).mean()),
    }
