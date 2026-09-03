from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments.ah_settlement_engine_audit import LABELS
from src.experiments.ah_settlement_engine_audit import settle_side


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "E1", "E2", "E3", "SC0"]
LAYER1 = {"E0", "D1", "I1", "SP1", "F1", "P1"}
LAYER2 = {"N1", "B1", "T1", "G1", "E1", "E2", "E3"}
ENGLISH_LOWER = {"E1", "E2", "E3"}
FEATURES = ["AHh", "AvgAHH", "AvgAHA", "market_home_probability", "market_away_probability"]
TARGET = "target_ah_home_cover"

REPORT_PATH = Path("outputs/reports/post_backfill_locked_ah_value_review.md")
FIXED_RULES_PATH = Path("outputs/reports/post_backfill_locked_ah_value_fixed_rules.csv")
NESTED_PATH = Path("outputs/reports/post_backfill_locked_ah_value_nested_selection.csv")
BY_YEAR_PATH = Path("outputs/reports/post_backfill_locked_ah_value_by_year.csv")
BY_LEAGUE_PATH = Path("outputs/reports/post_backfill_locked_ah_value_by_league.csv")
CONTROLS_PATH = Path("outputs/reports/post_backfill_locked_ah_value_controls.csv")
ROBUSTNESS_PATH = Path("outputs/reports/post_backfill_locked_ah_value_robustness.csv")


@dataclass(frozen=True)
class Rule:
    side: str
    edge: float
    min_odds: float

    @property
    def name(self) -> str:
        return f"{self.side}_edge_ge_{self.edge:.3f}_odds_ge_{self.min_odds:.2f}".replace(".", "_")


def rule_grid() -> list[Rule]:
    specs = [
        (0.01, 1.80),
        (0.015, 1.80),
        (0.02, 1.80),
        (0.03, 1.80),
        (0.04, 1.80),
        (0.05, 1.80),
        (0.02, 1.85),
        (0.03, 1.85),
        (0.04, 1.85),
        (0.05, 1.85),
        (0.02, 1.90),
        (0.03, 1.90),
        (0.04, 1.90),
        (0.05, 1.90),
    ]
    return [Rule(side, edge, odds) for side in ["home", "away"] for edge, odds in specs]


def load_matches() -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frame["league"] = league
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No processed match files found")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    for column in ["season_end_year", "FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA", "AvgCAHH", "AvgCAHA"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    home_raw = 1.0 / data["AvgAHH"]
    away_raw = 1.0 / data["AvgAHA"]
    total = home_raw + away_raw
    data["market_home_probability"] = home_raw / total
    data["market_away_probability"] = away_raw / total
    if {"AvgCAHH", "AvgCAHA"}.issubset(data.columns):
        close_home_raw = 1.0 / data["AvgCAHH"]
        close_away_raw = 1.0 / data["AvgCAHA"]
        close_total = close_home_raw + close_away_raw
        data["closing_home_probability"] = close_home_raw / close_total
        data["closing_away_probability"] = close_away_raw / close_total
    else:
        data["closing_home_probability"] = np.nan
        data["closing_away_probability"] = np.nan
    margin = data["FTHG"] - data["FTAG"]
    adjusted = margin + data["AHh"]
    data[TARGET] = np.where(adjusted > 0, 1.0, np.where(adjusted < 0, 0.0, np.nan))
    home_settled = [settle_side(m, line, odds) for m, line, odds in zip(margin, data["AHh"], data["AvgAHH"])]
    away_settled = [settle_side(-m, -line if pd.notna(line) else np.nan, odds) for m, line, odds in zip(margin, data["AHh"], data["AvgAHA"])]
    data["home_profit"] = [item.profit for item in home_settled]
    data["away_profit"] = [item.profit for item in away_settled]
    data["home_label"] = [item.label for item in home_settled]
    data["away_label"] = [item.label for item in away_settled]
    data["valid_settlement"] = data["home_label"].isin(LABELS) & data["away_label"].isin(LABELS)
    required = ["league", "Date", "season_end_year", "FTHG", "FTAG"] + FEATURES
    data = data.dropna(subset=required).copy()
    data = data[data["AvgAHH"].gt(1.0) & data["AvgAHA"].gt(1.0) & data["valid_settlement"]].copy()
    data["season_end_year"] = data["season_end_year"].astype(int)
    data["row_id"] = np.arange(len(data))
    return data.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def xgb_params(seed: int) -> dict[str, object]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 2,
        "eta": 0.03,
        "lambda": 8.0,
        "alpha": 2.0,
        "seed": seed,
        "verbosity": 0,
    }


def fit_predict(train: pd.DataFrame, validation: pd.DataFrame, target: pd.DataFrame, seed: int) -> np.ndarray:
    train = train.dropna(subset=[TARGET] + FEATURES).copy()
    validation_train = validation.dropna(subset=[TARGET] + FEATURES).copy()
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURES])
    x_validation = imputer.transform(validation_train[FEATURES])
    x_target = imputer.transform(target[FEATURES])
    model = xgb.train(
        xgb_params(seed),
        xgb.DMatrix(x_train, label=train[TARGET].astype(int).to_numpy(), feature_names=FEATURES),
        num_boost_round=250,
        evals=[(xgb.DMatrix(x_validation, label=validation_train[TARGET].astype(int).to_numpy(), feature_names=FEATURES), "validation")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return np.clip(model.predict(xgb.DMatrix(x_target, feature_names=FEATURES)), 1e-6, 1 - 1e-6)


def add_predictions(frame: pd.DataFrame, probabilities: np.ndarray, regime: str, fold_test_year: int, fold_role: str) -> pd.DataFrame:
    output = frame.copy()
    output["regime"] = regime
    output["fold_test_year"] = int(fold_test_year)
    output["fold_role"] = fold_role
    output["model_home_probability"] = probabilities
    output["model_away_probability"] = 1.0 - output["model_home_probability"]
    output["home_edge"] = output["model_home_probability"] - output["market_home_probability"]
    output["away_edge"] = output["model_away_probability"] - output["market_away_probability"]
    output["home_clv"] = output["closing_home_probability"] - output["market_home_probability"]
    output["away_clv"] = output["closing_away_probability"] - output["market_away_probability"]
    return output


def generate_predictions(data: pd.DataFrame, regime: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if regime == "A_recent_only":
        base = data[data["season_end_year"].between(2020, 2026)].copy()
        test_years = [2022, 2023, 2024, 2025, 2026]
    elif regime == "B_historical_training_modern_test":
        base = data.copy()
        test_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    else:
        raise ValueError(regime)
    tests = []
    validations = []
    for test_year in test_years:
        validation_year = test_year - 1
        train = base[base["season_end_year"].lt(validation_year)].copy()
        validation = base[base["season_end_year"].eq(validation_year)].copy()
        test = base[base["season_end_year"].eq(test_year)].copy()
        train_model = train.dropna(subset=[TARGET])
        validation_model = validation.dropna(subset=[TARGET])
        if len(train_model) == 0 or len(validation_model) == 0 or len(test) == 0 or train_model[TARGET].nunique() < 2:
            continue
        validations.append(add_predictions(validation, fit_predict(train_model, validation_model, validation, 100 + test_year), regime, test_year, "validation"))
        tests.append(add_predictions(test, fit_predict(train_model, validation_model, test, 100 + test_year), regime, test_year, "test"))
    return (
        pd.concat(tests, ignore_index=True, sort=False) if tests else pd.DataFrame(),
        pd.concat(validations, ignore_index=True, sort=False) if validations else pd.DataFrame(),
    )


def z_score(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(arr) < 2:
        return 0.0
    std = arr.std(ddof=1)
    return 0.0 if std == 0 or not np.isfinite(std) else float(arr.sum() / (std * np.sqrt(len(arr))))


def max_drawdown(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(float)
    cumulative = np.cumsum(arr)
    if len(cumulative) == 0:
        return 0.0
    peaks = np.maximum.accumulate(np.insert(cumulative, 0, 0.0))[1:]
    return float((peaks - cumulative).max())


def hhi(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    shares = series.value_counts(normalize=True)
    return float((shares * shares).sum())


def select_rule(frame: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if rule.side == "home":
        selected = frame[(frame["home_edge"].ge(rule.edge)) & (frame["AvgAHH"].ge(rule.min_odds))].copy()
        selected["selected_side"] = "home"
    else:
        selected = frame[(frame["away_edge"].ge(rule.edge)) & (frame["AvgAHA"].ge(rule.min_odds))].copy()
        selected["selected_side"] = "away"
    if selected.empty:
        return selected
    selected["rule_name"] = rule.name
    selected["rule_side"] = rule.side
    selected["rule_edge_threshold"] = rule.edge
    selected["rule_min_odds"] = rule.min_odds
    selected["selected_edge"] = np.where(selected["selected_side"].eq("home"), selected["home_edge"], selected["away_edge"])
    selected["selected_odds"] = np.where(selected["selected_side"].eq("home"), selected["AvgAHH"], selected["AvgAHA"])
    selected["selected_model_probability"] = np.where(selected["selected_side"].eq("home"), selected["model_home_probability"], selected["model_away_probability"])
    selected["selected_market_probability"] = np.where(selected["selected_side"].eq("home"), selected["market_home_probability"], selected["market_away_probability"])
    selected["selected_profit"] = np.where(selected["selected_side"].eq("home"), selected["home_profit"], selected["away_profit"])
    selected["selected_label"] = np.where(selected["selected_side"].eq("home"), selected["home_label"], selected["away_label"])
    selected["selected_clv"] = np.where(selected["selected_side"].eq("home"), selected["home_clv"], selected["away_clv"])
    return selected


def summarize(selected: pd.DataFrame, scope: str, rule: Rule | None = None, label: str = "") -> dict[str, object]:
    row = {
        "scope": scope,
        "label": label,
        "side": rule.side if rule else "",
        "rule_name": rule.name if rule else "",
        "edge_threshold": rule.edge if rule else np.nan,
        "min_odds": rule.min_odds if rule else np.nan,
        "bets": int(len(selected)),
    }
    if selected.empty:
        row.update({k: 0.0 for k in ["profit", "roi", "z_score", "max_drawdown", "league_concentration_hhi", "top_league_share", "push_rate", "half_win_loss_rate"]})
        row.update({k: np.nan for k in ["average_odds", "average_edge", "average_model_probability", "average_market_probability", "average_clv", "clv_positive_rate", "average_line"]})
        return row
    profit = selected["selected_profit"]
    league_shares = selected["league"].value_counts(normalize=True)
    labels = selected["selected_label"].astype(str)
    row.update(
        {
            "profit": float(profit.sum()),
            "roi": float(profit.mean()),
            "z_score": z_score(profit),
            "max_drawdown": max_drawdown(selected.sort_values("Date")["selected_profit"]),
            "average_odds": float(selected["selected_odds"].mean()),
            "average_edge": float(selected["selected_edge"].mean()),
            "average_model_probability": float(selected["selected_model_probability"].mean()),
            "average_market_probability": float(selected["selected_market_probability"].mean()),
            "average_clv": float(selected["selected_clv"].mean(skipna=True)),
            "clv_positive_rate": float(selected["selected_clv"].gt(0).mean()) if selected["selected_clv"].notna().any() else np.nan,
            "league_concentration_hhi": hhi(selected["league"]),
            "top_league_share": float(league_shares.iloc[0]) if len(league_shares) else np.nan,
            "average_line": float(selected["AHh"].mean()),
            "push_rate": float(labels.eq("push").mean()),
            "half_win_loss_rate": float(labels.isin(["half_win", "half_loss"]).mean()),
            "leagues": int(selected["league"].nunique()),
            "years": int(selected["season_end_year"].nunique()),
        }
    )
    return row


def fixed_rules(predictions: pd.DataFrame, regime: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    selections = {}
    for rule in rule_grid():
        selected = select_rule(predictions, rule)
        key = f"{regime}:{rule.name}"
        selections[key] = selected
        row = summarize(selected, "fixed_rule", rule)
        row["regime"] = regime
        rows.append(row)
    return pd.DataFrame(rows), selections


def nested_selection(test_predictions: pd.DataFrame, validation_predictions: pd.DataFrame, regime: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selected_tests = []
    for test_year in sorted(test_predictions["fold_test_year"].unique()):
        validation = validation_predictions[validation_predictions["fold_test_year"].eq(test_year)].copy()
        test = test_predictions[test_predictions["fold_test_year"].eq(test_year)].copy()
        candidates = []
        for rule in rule_grid():
            selected_validation = select_rule(validation, rule)
            stats = summarize(selected_validation, "validation", rule)
            max_league_share = stats.get("top_league_share", np.nan)
            if (
                stats["bets"] >= 150
                and stats["roi"] > 0
                and stats["z_score"] > 0.75
                and stats["profit"] > 0
                and pd.notna(max_league_share)
                and max_league_share <= 0.35
                and stats.get("leagues", 0) >= 4
            ):
                candidates.append(stats)
        if not candidates:
            rows.append(
                {
                    "regime": regime,
                    "test_year": int(test_year),
                    "selected_rule": "",
                    "selection_status": "no_validation_rule_passed",
                    "validation_bets": 0,
                    "validation_profit": 0.0,
                    "validation_roi": 0.0,
                    "validation_z": 0.0,
                    "validation_top_league_share": np.nan,
                    "validation_leagues": 0,
                    "test_bets": 0,
                    "test_profit": 0.0,
                    "test_roi": 0.0,
                    "test_z": 0.0,
                }
            )
            continue
        chosen = pd.DataFrame(candidates).sort_values(["z_score", "roi", "bets"], ascending=[False, False, False]).iloc[0]
        rule = next(r for r in rule_grid() if r.name == chosen["rule_name"])
        selected_test = select_rule(test, rule)
        selected_test["nested_test_year"] = int(test_year)
        selected_tests.append(selected_test)
        test_stats = summarize(selected_test, "nested_test", rule)
        rows.append(
            {
                "regime": regime,
                "test_year": int(test_year),
                "selected_rule": rule.name,
                "selection_status": "selected",
                "validation_bets": int(chosen["bets"]),
                "validation_profit": float(chosen["profit"]),
                "validation_roi": float(chosen["roi"]),
                "validation_z": float(chosen["z_score"]),
                "validation_top_league_share": float(chosen["top_league_share"]),
                "validation_leagues": int(chosen["leagues"]),
                "test_bets": int(test_stats["bets"]),
                "test_profit": float(test_stats["profit"]),
                "test_roi": float(test_stats["roi"]),
                "test_z": float(test_stats["z_score"]),
            }
        )
    nested_bets = pd.concat(selected_tests, ignore_index=True, sort=False) if selected_tests else pd.DataFrame()
    return pd.DataFrame(rows), nested_bets


def by_group(selected: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    rows = []
    if selected.empty:
        return pd.DataFrame()
    for key, group in selected.groupby(group_col):
        row = summarize(group, label)
        row[group_col] = key
        rows.append(row)
    return pd.DataFrame(rows)


def same_size_random(pool: pd.DataFrame, selected: pd.DataFrame, same_side: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    for year, group in selected.groupby("season_end_year"):
        year_pool = pool[pool["season_end_year"].eq(year)].copy()
        n = min(len(group), len(year_pool))
        if n <= 0:
            continue
        sample = year_pool.sample(n=n, random_state=int(rng.integers(0, 1_000_000))).copy()
        if same_side:
            side_values = group["selected_side"].sample(n=n, replace=True, random_state=int(rng.integers(0, 1_000_000))).to_numpy()
        else:
            side_values = rng.choice(["home", "away"], size=n)
        sample["selected_side"] = side_values
        sample = fill_selected_columns(sample)
        pieces.append(sample)
    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()


def fill_selected_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["selected_edge"] = np.where(out["selected_side"].eq("home"), out["home_edge"], out["away_edge"])
    out["selected_odds"] = np.where(out["selected_side"].eq("home"), out["AvgAHH"], out["AvgAHA"])
    out["selected_model_probability"] = np.where(out["selected_side"].eq("home"), out["model_home_probability"], out["model_away_probability"])
    out["selected_market_probability"] = np.where(out["selected_side"].eq("home"), out["market_home_probability"], out["market_away_probability"])
    out["selected_profit"] = np.where(out["selected_side"].eq("home"), out["home_profit"], out["away_profit"])
    out["selected_label"] = np.where(out["selected_side"].eq("home"), out["home_label"], out["away_label"])
    out["selected_clv"] = np.where(out["selected_side"].eq("home"), out["home_clv"], out["away_clv"])
    return out


def opposite_side(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out
    out["selected_side"] = np.where(out["selected_side"].eq("home"), "away", "home")
    return fill_selected_columns(out)


def always_side(pool: pd.DataFrame, side: str) -> pd.DataFrame:
    out = pool.copy()
    out["selected_side"] = side
    return fill_selected_columns(out)


def controls(pool: pd.DataFrame, selected: pd.DataFrame, best_rule: Rule | None, label: str) -> pd.DataFrame:
    rows = []
    rows.append({**summarize(same_size_random(pool, selected, False, 11), "control", label=f"{label}_random_same_size_same_year"), "target": label})
    rows.append({**summarize(same_size_random(pool, selected, True, 12), "control", label=f"{label}_random_same_size_same_year_same_side"), "target": label})
    if best_rule is not None:
        shuffled = pool.copy()
        rng = np.random.default_rng(13)
        shuffled["home_edge"] = rng.permutation(shuffled["home_edge"].to_numpy())
        shuffled["away_edge"] = -shuffled["home_edge"]
        rows.append({**summarize(select_rule(shuffled, best_rule), "control", label=f"{label}_shuffled_model_edge"), "target": label})
        permuted = pool.copy()
        for _, idx in permuted.groupby(["league", "season_end_year"]).groups.items():
            values = permuted.loc[idx, "home_edge"].to_numpy(copy=True)
            rng.shuffle(values)
            permuted.loc[idx, "home_edge"] = values
            permuted.loc[idx, "away_edge"] = -values
        rows.append({**summarize(select_rule(permuted, best_rule), "control", label=f"{label}_permuted_edge_within_league_season"), "target": label})
    market = pool.copy()
    market["apparent_edge"] = np.maximum(market["market_home_probability"] - 0.5, market["market_away_probability"] - 0.5)
    top = market.nlargest(len(selected), "apparent_edge").copy() if len(selected) else market.iloc[0:0].copy()
    if len(top):
        top["selected_side"] = np.where(top["market_home_probability"].ge(top["market_away_probability"]), "home", "away")
        top = fill_selected_columns(top)
    rows.append({**summarize(top, "control", label=f"{label}_market_no_vig_top_apparent_edge"), "target": label})
    rows.append({**summarize(opposite_side(selected), "control", label=f"{label}_opposite_side_same_matches"), "target": label})
    rows.append({**summarize(always_side(pool, "home"), "control", label=f"{label}_always_home_ah"), "target": label})
    rows.append({**summarize(always_side(pool, "away"), "control", label=f"{label}_always_away_ah"), "target": label})
    # Calibration-only baseline uses prior-fold model probabilities when available only as a control score.
    calibrated = pool.copy()
    trainable = calibrated.dropna(subset=[TARGET]).copy()
    cal_parts = []
    for year in sorted(calibrated["season_end_year"].unique()):
        train = trainable[trainable["season_end_year"].lt(year)].copy()
        test = calibrated[calibrated["season_end_year"].eq(year)].copy()
        if len(train) and len(test) and train[TARGET].nunique() == 2:
            model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))])
            model.fit(train[["market_home_probability"]], train[TARGET].astype(int))
            proba = model.predict_proba(test[["market_home_probability"]])[:, list(model.named_steps["model"].classes_).index(1)]
            test["model_home_probability"] = proba
            test["model_away_probability"] = 1.0 - test["model_home_probability"]
            test["home_edge"] = test["model_home_probability"] - test["market_home_probability"]
            test["away_edge"] = test["model_away_probability"] - test["market_away_probability"]
            cal_parts.append(test)
    if cal_parts and best_rule is not None:
        rows.append({**summarize(select_rule(pd.concat(cal_parts, ignore_index=True), best_rule), "control", label=f"{label}_calibration_only_market_baseline"), "target": label})
    return pd.DataFrame(rows)


def robustness(selected: pd.DataFrame, label: str) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    season_profit = selected.groupby("season_end_year")["selected_profit"].sum()
    league_profit = selected.groupby("league")["selected_profit"].sum()
    masks = [
        ("exclude_best_profit_season", selected["season_end_year"].ne(int(season_profit.idxmax()))),
        ("exclude_worst_profit_season", selected["season_end_year"].ne(int(season_profit.idxmin()))),
        ("exclude_best_profit_league", selected["league"].ne(str(league_profit.idxmax()))),
        ("exclude_SC0", selected["league"].ne("SC0")),
        ("exclude_english_lower_leagues", ~selected["league"].isin(ENGLISH_LOWER)),
        ("exclude_layer1", ~selected["league"].isin(LAYER1)),
        ("exclude_layer2", ~selected["league"].isin(LAYER2)),
        ("exclude_2026", selected["season_end_year"].ne(2026)),
    ]
    masks.extend((f"exclude_{league}", selected["league"].ne(league)) for league in LEAGUES if league != "SC0")
    rows = []
    for name, mask in masks:
        rows.append({**summarize(selected[mask].copy(), "robustness", label=name), "target": label})
    return pd.DataFrame(rows)


def classify(nested_bets: pd.DataFrame, control_rows: pd.DataFrame, robustness_rows: pd.DataFrame) -> str:
    if nested_bets.empty:
        return "predictive_only_no_value"
    stats = summarize(nested_bets, "nested")
    years_with_bets = nested_bets["season_end_year"].nunique()
    season_profit = nested_bets.groupby("season_end_year")["selected_profit"].sum()
    league_profit = nested_bets.groupby("league")["selected_profit"].sum()
    total_profit = float(stats["profit"])
    if total_profit <= 0 or stats["roi"] <= 0.02 or stats["z_score"] < 1.0 or years_with_bets < 4:
        return "predictive_only_no_value"
    if season_profit.max() > 0.5 * total_profit or league_profit.max() > 0.5 * total_profit:
        return "research_only"
    best_season = robustness_rows[(robustness_rows["target"].eq("best_fixed_rule")) & (robustness_rows["label"].eq("exclude_best_profit_season"))]
    if len(best_season) and float(best_season["profit"].iloc[0]) <= 0:
        return "research_only"
    nested_controls = control_rows[control_rows["target"].eq("nested_portfolio")]
    if len(nested_controls) and float(nested_controls["profit"].max()) >= total_profit:
        return "research_only"
    return "forward_paper_candidate"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def write_report(fixed: pd.DataFrame, nested: pd.DataFrame, nested_bets: pd.DataFrame, controls_df: pd.DataFrame, robustness_df: pd.DataFrame, classification: str) -> None:
    best = fixed[fixed["regime"].eq("B_historical_training_modern_test")].sort_values(["profit", "z_score"], ascending=[False, False]).head(15)
    nested_summary = pd.DataFrame([{**summarize(nested_bets, "nested_portfolio"), "target": "nested_portfolio"}])
    lines = [
        "# Post-Backfill Locked AH Value Review",
        "",
        f"Final classification: `{classification}`",
        "",
        "Scope: locked AH value review using frozen post-backfill XGBoost market-only predictions and existing AH settlement. No new models, model families, features, hyperparameter search, Transfermarkt, player features, lineups, team-name features, closing odds for selection, live betting, or confirmed edge claims were used.",
        "",
        "## Best Fixed Rules, Regime B",
        "",
        markdown_table(best, ["side", "rule_name", "bets", "profit", "roi", "z_score", "max_drawdown", "average_odds", "average_edge", "average_model_probability", "average_market_probability", "average_clv", "clv_positive_rate", "top_league_share"], 20),
        "",
        "## Nested Selection",
        "",
        markdown_table(nested, ["regime", "test_year", "selected_rule", "selection_status", "validation_bets", "validation_roi", "validation_z", "validation_top_league_share", "validation_leagues", "test_bets", "test_profit", "test_roi", "test_z"], 40),
        "",
        "## Nested Portfolio",
        "",
        markdown_table(nested_summary, ["bets", "profit", "roi", "z_score", "max_drawdown", "average_odds", "average_edge", "average_clv", "clv_positive_rate", "top_league_share"], 10),
        "",
        "## Controls",
        "",
        markdown_table(controls_df, ["target", "label", "bets", "profit", "roi", "z_score", "max_drawdown"], 80),
        "",
        "## Robustness",
        "",
        markdown_table(robustness_df, ["target", "label", "bets", "profit", "roi", "z_score"], 80),
        "",
        "No live betting was run. No confirmed edge is claimed.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = load_matches()
    test_a, val_a = generate_predictions(data, "A_recent_only")
    test_b, val_b = generate_predictions(data, "B_historical_training_modern_test")
    fixed_a, selections_a = fixed_rules(test_a, "A_recent_only")
    fixed_b, selections_b = fixed_rules(test_b, "B_historical_training_modern_test")
    fixed = pd.concat([fixed_a, fixed_b], ignore_index=True, sort=False)
    nested_a, nested_bets_a = nested_selection(test_a, val_a, "A_recent_only")
    nested_b, nested_bets_b = nested_selection(test_b, val_b, "B_historical_training_modern_test")
    nested = pd.concat([nested_a, nested_b], ignore_index=True, sort=False)
    best_row = fixed_b.sort_values(["profit", "z_score"], ascending=[False, False]).iloc[0]
    best_key = f"B_historical_training_modern_test:{best_row['rule_name']}"
    best_fixed_bets = selections_b[best_key]
    best_rule = next(rule for rule in rule_grid() if rule.name == best_row["rule_name"])
    by_year = pd.concat(
        [
            by_group(best_fixed_bets.assign(portfolio="best_fixed_rule"), "season_end_year", "best_fixed_rule"),
            by_group(nested_bets_b.assign(portfolio="nested_portfolio"), "season_end_year", "nested_portfolio"),
        ],
        ignore_index=True,
        sort=False,
    )
    by_league = pd.concat(
        [
            by_group(best_fixed_bets.assign(portfolio="best_fixed_rule"), "league", "best_fixed_rule"),
            by_group(nested_bets_b.assign(portfolio="nested_portfolio"), "league", "nested_portfolio"),
        ],
        ignore_index=True,
        sort=False,
    )
    control_rows = pd.concat(
        [
            controls(test_b, best_fixed_bets, best_rule, "best_fixed_rule"),
            controls(test_b, nested_bets_b, best_rule if len(nested_bets_b) else None, "nested_portfolio"),
        ],
        ignore_index=True,
        sort=False,
    )
    robustness_rows = pd.concat(
        [
            robustness(best_fixed_bets, "best_fixed_rule"),
            robustness(nested_bets_b, "nested_portfolio"),
        ],
        ignore_index=True,
        sort=False,
    )
    pre2020_rows = [
        {
            **summarize(select_rule(test_a, best_rule), "robustness", label="exclude_pre_2020_training_data"),
            "target": "best_fixed_rule",
        },
        {
            **summarize(nested_bets_a, "robustness", label="exclude_pre_2020_training_data"),
            "target": "nested_portfolio",
        },
    ]
    robustness_rows = pd.concat([robustness_rows, pd.DataFrame(pre2020_rows)], ignore_index=True, sort=False)
    classification = classify(nested_bets_b, control_rows, robustness_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(FIXED_RULES_PATH, index=False)
    nested.to_csv(NESTED_PATH, index=False)
    by_year.to_csv(BY_YEAR_PATH, index=False)
    by_league.to_csv(BY_LEAGUE_PATH, index=False)
    control_rows.to_csv(CONTROLS_PATH, index=False)
    robustness_rows.to_csv(ROBUSTNESS_PATH, index=False)
    write_report(fixed, nested, nested_bets_b, control_rows, robustness_rows, classification)
    print(
        {
            "data_rows": len(data),
            "regime_a_test_rows": len(test_a),
            "regime_b_test_rows": len(test_b),
            "fixed_rule_rows": len(fixed),
            "nested_rows": len(nested),
            "nested_b_bets": len(nested_bets_b),
            "controls": len(control_rows),
            "robustness": len(robustness_rows),
            "classification": classification,
        }
    )


if __name__ == "__main__":
    main()
