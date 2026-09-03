from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SIDE_TO_CLASS = {"home": 0, "draw": 1, "away": 2}


@dataclass(frozen=True)
class RuleSpec:
    name: str
    side: str
    edge_min: float
    odds_min: float
    odds_max: float | None = None


@dataclass(frozen=True)
class RuleSelection:
    rule: RuleSpec | None
    reason: str
    validation_bets: int
    validation_profit: float
    validation_roi: float
    validation_z: float
    validation_lcb_mean_profit: float
    validation_positive_leagues: int
    validation_max_positive_league_share: float


def add_value_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for side, target_class in SIDE_TO_CLASS.items():
        market = pd.to_numeric(out[f"market_{side}_prob"], errors="coerce")
        model = pd.to_numeric(out[f"model_{side}_prob"], errors="coerce")
        odds = pd.to_numeric(out[f"odds_{side}"], errors="coerce")
        out[f"{side}_edge"] = model - market
        out[f"{side}_profit"] = np.where(out["target_y"].eq(target_class), odds - 1.0, -1.0)
    return out


def apply_rule(frame: pd.DataFrame, rule: RuleSpec) -> pd.DataFrame:
    if rule.side not in SIDE_TO_CLASS:
        raise ValueError(f"Unsupported side: {rule.side}")
    edge = pd.to_numeric(frame[f"{rule.side}_edge"], errors="coerce")
    odds = pd.to_numeric(frame[f"odds_{rule.side}"], errors="coerce")
    mask = edge.ge(rule.edge_min) & odds.ge(rule.odds_min)
    if rule.odds_max is not None:
        mask &= odds.le(rule.odds_max)
    selected = frame.loc[mask].copy()
    selected["selected_side"] = rule.side
    selected["selected_rule"] = rule.name
    selected["selected_edge"] = selected[f"{rule.side}_edge"]
    selected["selected_odds"] = selected[f"odds_{rule.side}"]
    selected["profit"] = selected[f"{rule.side}_profit"]
    return selected


def z_score(profit: pd.Series) -> float:
    values = pd.to_numeric(profit, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    sd = float(values.std(ddof=1))
    return float(values.sum() / (sd * np.sqrt(len(values)))) if sd > 0 else 0.0


def max_drawdown(profit: pd.Series) -> float:
    values = pd.to_numeric(profit, errors="coerce").fillna(0.0)
    if values.empty:
        return 0.0
    curve = values.cumsum()
    return float((curve - curve.cummax()).min())


def rule_metrics(selected: pd.DataFrame) -> dict[str, float | int]:
    bets = int(len(selected))
    if bets == 0:
        return {
            "bets": 0,
            "profit": 0.0,
            "roi": 0.0,
            "z_score": 0.0,
            "max_drawdown": 0.0,
            "mean_profit": 0.0,
            "se_mean_profit": np.nan,
            "lcb_mean_profit": -np.inf,
            "positive_leagues": 0,
            "max_positive_league_share": 1.0,
        }
    profit = pd.to_numeric(selected["profit"], errors="coerce").fillna(0.0)
    total = float(profit.sum())
    sd = float(profit.std(ddof=1)) if bets > 1 else np.nan
    se = float(sd / np.sqrt(bets)) if bets > 1 and np.isfinite(sd) else np.nan
    lcb = float(profit.mean() - se) if np.isfinite(se) else -np.inf
    by_league = selected.assign(_profit=profit).groupby("league", dropna=False)["_profit"].sum()
    positive = by_league[by_league > 0]
    max_share = float(positive.max() / positive.sum()) if len(positive) and positive.sum() > 0 else 1.0
    return {
        "bets": bets,
        "profit": total,
        "roi": total / bets,
        "z_score": z_score(profit),
        "max_drawdown": max_drawdown(profit),
        "mean_profit": float(profit.mean()),
        "se_mean_profit": se,
        "lcb_mean_profit": lcb,
        "positive_leagues": int((by_league > 0).sum()),
        "max_positive_league_share": max_share,
    }


def build_rule_grid(
    sides: Iterable[str],
    edge_thresholds: Iterable[float],
    odds_minima: Iterable[float],
    odds_maxima: Iterable[float | None],
) -> list[RuleSpec]:
    rules: list[RuleSpec] = []
    for side in sides:
        for edge in edge_thresholds:
            for odds_min in odds_minima:
                for odds_max in odds_maxima:
                    if odds_max is not None and odds_max < odds_min:
                        continue
                    max_label = "none" if odds_max is None else f"{odds_max:.2f}"
                    name = f"{side}_edge_{edge:.3f}_odds_{odds_min:.2f}_{max_label}"
                    rules.append(RuleSpec(name, side, float(edge), float(odds_min), None if odds_max is None else float(odds_max)))
    return rules


def evaluate_rule_grid(frame: pd.DataFrame, rules: list[RuleSpec]) -> pd.DataFrame:
    rows = []
    for rule in rules:
        metrics = rule_metrics(apply_rule(frame, rule))
        rows.append({**asdict(rule), **metrics})
    return pd.DataFrame(rows)


def select_rule_on_validation(
    frame: pd.DataFrame,
    rules: list[RuleSpec],
    min_bets: int,
    require_positive_lcb: bool,
    max_positive_league_share: float,
    minimum_positive_leagues: int,
) -> tuple[RuleSelection, pd.DataFrame]:
    table = evaluate_rule_grid(frame, rules)
    if table.empty:
        return RuleSelection(None, "no_rules", 0, 0.0, 0.0, 0.0, -np.inf, 0, 1.0), table
    eligible = table[
        table["bets"].ge(int(min_bets))
        & table["positive_leagues"].ge(int(minimum_positive_leagues))
        & table["max_positive_league_share"].le(float(max_positive_league_share))
    ].copy()
    if require_positive_lcb:
        eligible = eligible[eligible["lcb_mean_profit"].gt(0.0)]
    if eligible.empty:
        diagnostic = table.sort_values(["lcb_mean_profit", "roi", "bets"], ascending=[False, False, False]).iloc[0]
        return (
            RuleSelection(
                None,
                "no_validation_rule_passed_conservative_gate",
                int(diagnostic["bets"]),
                float(diagnostic["profit"]),
                float(diagnostic["roi"]),
                float(diagnostic["z_score"]),
                float(diagnostic["lcb_mean_profit"]),
                int(diagnostic["positive_leagues"]),
                float(diagnostic["max_positive_league_share"]),
            ),
            table,
        )
    best = eligible.sort_values(
        ["lcb_mean_profit", "roi", "positive_leagues", "bets", "edge_min"],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    rule = RuleSpec(
        name=str(best["name"]),
        side=str(best["side"]),
        edge_min=float(best["edge_min"]),
        odds_min=float(best["odds_min"]),
        odds_max=None if pd.isna(best["odds_max"]) else float(best["odds_max"]),
    )
    return (
        RuleSelection(
            rule,
            "selected_on_calibration_year_only",
            int(best["bets"]),
            float(best["profit"]),
            float(best["roi"]),
            float(best["z_score"]),
            float(best["lcb_mean_profit"]),
            int(best["positive_leagues"]),
            float(best["max_positive_league_share"]),
        ),
        table,
    )
