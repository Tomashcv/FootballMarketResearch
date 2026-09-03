from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "P1", "N1", "B1", "T1", "G1", "E1", "E2", "E3", "SC0"]
MARKETS = ["over_2_5", "btts_yes", "outcome_1x2"]

REPORT_PATH = Path("outputs/reports/other_markets_availability_baseline_audit.md")
COVERAGE_PATH = Path("outputs/reports/other_markets_coverage_by_league_season.csv")
DIAGNOSTICS_PATH = Path("outputs/reports/other_markets_baseline_diagnostics.csv")
RECOMMENDATIONS_PATH = Path("outputs/reports/other_markets_recommended_next_markets.csv")

OU25_SOURCES = {
    "AvgOU25": ("Avg>2.5", "Avg<2.5"),
    "BbAvOU25": ("BbAv>2.5", "BbAv<2.5"),
    "B365OU25": ("B365>2.5", "B365<2.5"),
    "MaxOU25": ("Max>2.5", "Max<2.5"),
    "BbMxOU25": ("BbMx>2.5", "BbMx<2.5"),
    "PinnacleOU25": ("P>2.5", "P<2.5"),
    "BetfairExchangeOU25": ("BFE>2.5", "BFE<2.5"),
    "GamebookersOU25": ("GB>2.5", "GB<2.5"),
}
BTTS_SOURCES = {
    "AvgBTTS": ("AvgGG", "AvgNG"),
    "MaxBTTS": ("MaxGG", "MaxNG"),
    "B365BTTS": ("B365GG", "B365NG"),
    "GenericBTTS": ("GG", "NG"),
    "BbAvBTTS": ("BbAvGG", "BbAvNG"),
    "BbMxBTTS": ("BbMxGG", "BbMxNG"),
}
ONE_X_TWO_SOURCES = {
    "Avg1X2": ("AvgH", "AvgD", "AvgA"),
    "BbAv1X2": ("BbAvH", "BbAvD", "BbAvA"),
    "B3651X2": ("B365H", "B365D", "B365A"),
    "Max1X2": ("MaxH", "MaxD", "MaxA"),
    "BbMx1X2": ("BbMxH", "BbMxD", "BbMxA"),
    "Pinnacle1X2": ("PSH", "PSD", "PSA"),
    "BetfairExchange1X2": ("BFEH", "BFED", "BFEA"),
    "Gamebookers1X2": ("GBH", "GBD", "GBA"),
}


def ece_binary(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (probabilities >= left) & (probabilities <= right if idx == bins - 1 else probabilities < right)
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(probabilities[mask].mean()))
    return total


def ece_multiclass(y: np.ndarray, probabilities: np.ndarray, labels: list[int], bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = np.array(labels)[probabilities.argmax(axis=1)]
    correct = (predicted == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (confidence >= left) & (confidence <= right if idx == bins - 1 else confidence < right)
        if mask.any():
            total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return total


def load_matches() -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        path = Path("data/processed") / league / f"{league}_matches.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["league"] = league
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No processed match files found")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    for column in ["season_start_year", "season_end_year", "FTHG", "FTAG"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["league", "season_end_year"]).copy()
    data["season_end_year"] = data["season_end_year"].astype(int)
    return data


def present_sources(columns: set[str], sources: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    return {name: cols for name, cols in sources.items() if set(cols).issubset(columns)}


def choose_source(frame: pd.DataFrame, sources: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["source"] = ""
    for idx in range(max(len(cols) for cols in sources.values())):
        out[f"odds_{idx}"] = np.nan
    for source, columns in sources.items():
        if not set(columns).issubset(frame.columns):
            continue
        odds = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        valid = odds.gt(1.0).all(axis=1) & out["source"].eq("")
        if not valid.any():
            continue
        out.loc[valid, "source"] = source
        for idx, column in enumerate(columns):
            out.loc[valid, f"odds_{idx}"] = odds.loc[valid, column]
    return out


def add_targets(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    scores = out[["FTHG", "FTAG"]].notna().all(axis=1)
    total = out["FTHG"] + out["FTAG"]
    out["valid_score"] = scores
    out["over_2_5_target"] = np.where(scores, (total > 2.5).astype(int), np.nan)
    out["btts_yes_target"] = np.where(scores, ((out["FTHG"] > 0) & (out["FTAG"] > 0)).astype(int), np.nan)
    out["outcome_1x2_target"] = np.select(
        [scores & out["FTHG"].gt(out["FTAG"]), scores & out["FTHG"].eq(out["FTAG"]), scores & out["FTHG"].lt(out["FTAG"])],
        [0, 1, 2],
        default=np.nan,
    )
    return out


def add_market_probabilities(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    ou = choose_source(out, OU25_SOURCES)
    btts = choose_source(out, BTTS_SOURCES)
    one_x_two = choose_source(out, ONE_X_TWO_SOURCES)

    for prefix, chosen, labels in [
        ("ou25", ou, ["over", "under"]),
        ("btts", btts, ["yes", "no"]),
        ("one_x_two", one_x_two, ["home", "draw", "away"]),
    ]:
        out[f"{prefix}_odds_source"] = chosen["source"]
        raw_cols = []
        for idx, label in enumerate(labels):
            out[f"{prefix}_{label}_odds"] = chosen[f"odds_{idx}"]
            out[f"{prefix}_{label}_raw_probability"] = 1.0 / out[f"{prefix}_{label}_odds"]
            raw_cols.append(f"{prefix}_{label}_raw_probability")
        out[f"{prefix}_overround"] = out[raw_cols].sum(axis=1)
        for label in labels:
            out[f"{prefix}_{label}_no_vig_probability"] = out[f"{prefix}_{label}_raw_probability"] / out[f"{prefix}_overround"]
    return out


def usable_mask(frame: pd.DataFrame, market: str) -> pd.Series:
    if market == "over_2_5":
        return frame["valid_score"] & frame["ou25_odds_source"].ne("") & frame["ou25_over_no_vig_probability"].notna()
    if market == "btts_yes":
        return frame["valid_score"] & frame["btts_odds_source"].ne("") & frame["btts_yes_no_vig_probability"].notna()
    if market == "outcome_1x2":
        return frame["valid_score"] & frame["one_x_two_odds_source"].ne("") & frame["one_x_two_home_no_vig_probability"].notna()
    raise ValueError(market)


def market_sources_for_frame(frame: pd.DataFrame, market: str) -> str:
    columns = set(frame.columns)
    if market == "over_2_5":
        found = present_sources(columns, OU25_SOURCES)
    elif market == "btts_yes":
        found = present_sources(columns, BTTS_SOURCES)
    elif market == "outcome_1x2":
        found = present_sources(columns, ONE_X_TWO_SOURCES)
    else:
        raise ValueError(market)
    return ";".join(f"{name}:{'/'.join(cols)}" for name, cols in found.items())


def coverage_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    first_last = {}
    for market in MARKETS:
        usable = data[usable_mask(data, market)]
        first_last[market] = usable.groupby("league")["season_end_year"].agg(["min", "max"]).to_dict("index") if len(usable) else {}
    for (league, season), group in data.groupby(["league", "season_end_year"], dropna=False):
        valid_score_rows = int(group["valid_score"].sum())
        for market in MARKETS:
            usable_rows = int(usable_mask(group, market).sum())
            found = market_sources_for_frame(group, market)
            league_bounds = first_last[market].get(league, {})
            rows.append(
                {
                    "league": league,
                    "season_end_year": int(season),
                    "market": market,
                    "matches": int(len(group)),
                    "valid_score_rows": valid_score_rows,
                    "odds_columns_found": found,
                    "usable_market_rows": usable_rows,
                    "market_coverage_pct": float(usable_rows / valid_score_rows) if valid_score_rows else 0.0,
                    "missing_odds_rows": int(valid_score_rows - usable_rows),
                    "first_season_with_usable_odds": league_bounds.get("min", np.nan),
                    "last_season_with_usable_odds": league_bounds.get("max", np.nan),
                }
            )
    return pd.DataFrame(rows).sort_values(["market", "league", "season_end_year"])


def binary_metrics(group: pd.DataFrame, target: str, probability: str) -> dict[str, float]:
    y = group[target].astype(int).to_numpy()
    p = np.clip(group[probability].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    if len(group) == 0 or len(np.unique(y)) < 2:
        return {"raw_market_log_loss": np.nan, "raw_market_brier": np.nan, "raw_market_ece": np.nan}
    return {
        "raw_market_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "raw_market_brier": float(brier_score_loss(y, p)),
        "raw_market_ece": float(ece_binary(y, p)),
    }


def multiclass_brier(y: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def diagnostics_for_group(market: str, group: pd.DataFrame, scope: str, key: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {"market": market, "scope": scope, **key, "rows": int(len(group))}
    row["seasons"] = int(group["season_end_year"].nunique()) if len(group) else 0
    row["leagues"] = int(group["league"].nunique()) if len(group) else 0
    if len(group) == 0:
        return row
    if market == "over_2_5":
        row.update(binary_metrics(group, "over_2_5_target", "ou25_over_no_vig_probability"))
        row.update(
            {
                "implied_probability_average": float(group["ou25_over_no_vig_probability"].mean()),
                "realised_target_rate": float(group["over_2_5_target"].mean()),
                "bookmaker_overround_estimate": float(group["ou25_overround"].mean()),
                "no_vig_over_probability": float(group["ou25_over_no_vig_probability"].mean()),
                "no_vig_under_probability": float(group["ou25_under_no_vig_probability"].mean()),
                "over_hit_rate": float(group["over_2_5_target"].mean()),
                "under_hit_rate": float(1.0 - group["over_2_5_target"].mean()),
            }
        )
    elif market == "btts_yes":
        row.update(binary_metrics(group, "btts_yes_target", "btts_yes_no_vig_probability"))
        row.update(
            {
                "implied_probability_average": float(group["btts_yes_no_vig_probability"].mean()),
                "realised_target_rate": float(group["btts_yes_target"].mean()),
                "bookmaker_overround_estimate": float(group["btts_overround"].mean()),
                "no_vig_btts_yes_probability": float(group["btts_yes_no_vig_probability"].mean()),
                "no_vig_btts_no_probability": float(group["btts_no_no_vig_probability"].mean()),
                "btts_yes_hit_rate": float(group["btts_yes_target"].mean()),
                "btts_no_hit_rate": float(1.0 - group["btts_yes_target"].mean()),
            }
        )
    elif market == "outcome_1x2":
        y = group["outcome_1x2_target"].astype(int).to_numpy()
        probabilities = np.clip(
            group[["one_x_two_home_no_vig_probability", "one_x_two_draw_no_vig_probability", "one_x_two_away_no_vig_probability"]].astype(float).to_numpy(),
            1e-6,
            1 - 1e-6,
        )
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        if len(np.unique(y)) >= 2:
            row["raw_market_log_loss"] = float(log_loss(y, probabilities, labels=[0, 1, 2]))
            row["raw_market_brier"] = multiclass_brier(y, probabilities)
            row["raw_market_ece"] = float(ece_multiclass(y, probabilities, [0, 1, 2]))
            row["multiclass_log_loss"] = row["raw_market_log_loss"]
        else:
            row["raw_market_log_loss"] = np.nan
            row["raw_market_brier"] = np.nan
            row["raw_market_ece"] = np.nan
            row["multiclass_log_loss"] = np.nan
        row.update(
            {
                "implied_probability_average": float(probabilities.max(axis=1).mean()),
                "realised_target_rate": np.nan,
                "bookmaker_overround_estimate": float(group["one_x_two_overround"].mean()),
                "no_vig_home_probability": float(group["one_x_two_home_no_vig_probability"].mean()),
                "no_vig_draw_probability": float(group["one_x_two_draw_no_vig_probability"].mean()),
                "no_vig_away_probability": float(group["one_x_two_away_no_vig_probability"].mean()),
                "realised_home_rate": float((y == 0).mean()),
                "realised_draw_rate": float((y == 1).mean()),
                "realised_away_rate": float((y == 2).mean()),
            }
        )
    return row


def baseline_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market in MARKETS:
        usable = data[usable_mask(data, market)].copy()
        rows.append(diagnostics_for_group(market, usable, "overall", {}))
        for season, group in usable.groupby("season_end_year"):
            rows.append(diagnostics_for_group(market, group, "by_season", {"season_end_year": int(season)}))
        for league, group in usable.groupby("league"):
            rows.append(diagnostics_for_group(market, group, "by_league", {"league": league}))
        for (league, season), group in usable.groupby(["league", "season_end_year"]):
            rows.append(diagnostics_for_group(market, group, "by_league_season", {"league": league, "season_end_year": int(season)}))
    return pd.DataFrame(rows)


def recommendation_rows(coverage: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market in MARKETS:
        cov = coverage[coverage["market"].eq(market)]
        total_rows = int(cov["usable_market_rows"].sum())
        usable_seasons = int(cov.loc[cov["usable_market_rows"].gt(0), "season_end_year"].nunique())
        usable_leagues = int(cov.loc[cov["usable_market_rows"].gt(0), "league"].nunique())
        overall = diagnostics[(diagnostics["market"].eq(market)) & (diagnostics["scope"].eq("overall"))]
        diagnostics_ok = len(overall) == 1 and pd.notna(overall["raw_market_log_loss"].iloc[0]) and pd.notna(overall["raw_market_brier"].iloc[0])
        warnings = []
        if total_rows == 0:
            warnings.append("no_usable_odds_detected")
        if not diagnostics_ok:
            warnings.append("baseline_diagnostics_not_computable")
        if diagnostics_ok:
            overround = float(overall["bookmaker_overround_estimate"].iloc[0])
            if not 1.0 <= overround <= 1.25:
                warnings.append("bookmaker_overround_outside_expected_range")
        if market == "btts_yes" and total_rows == 0:
            status = "insufficient_odds_coverage"
        elif usable_seasons >= 5 and total_rows >= 5000 and usable_leagues >= 5 and diagnostics_ok and not warnings:
            status = "ready_for_predictive_audit"
        elif total_rows > 0 and diagnostics_ok:
            status = "partial_coverage_only"
        elif total_rows > 0:
            status = "diagnostic_only"
        else:
            status = "insufficient_odds_coverage"
        rows.append(
            {
                "market": market,
                "recommendation": status,
                "usable_rows_total": total_rows,
                "usable_seasons": usable_seasons,
                "usable_leagues": usable_leagues,
                "baseline_diagnostics_computable": bool(diagnostics_ok),
                "major_parsing_warnings": ";".join(warnings),
            }
        )
    return pd.DataFrame(rows)


def final_classification(recommendations: pd.DataFrame) -> str:
    ready = set(recommendations.loc[recommendations["recommendation"].eq("ready_for_predictive_audit"), "market"])
    if not ready:
        return "no_other_markets_ready"
    if len(ready) > 1:
        return "multiple_other_markets_ready"
    if "over_2_5" in ready:
        return "over_under_ready"
    if "btts_yes" in ready:
        return "btts_ready"
    if "outcome_1x2" in ready:
        return "one_x_two_ready"
    return "partial_other_markets_ready"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def write_report(coverage: pd.DataFrame, diagnostics: pd.DataFrame, recommendations: pd.DataFrame, classification: str) -> None:
    overall = diagnostics[diagnostics["scope"].eq("overall")].copy()
    coverage_summary = (
        coverage.groupby("market", as_index=False)
        .agg(
            usable_rows_total=("usable_market_rows", "sum"),
            usable_leagues=("league", lambda s: int(coverage.loc[s.index][coverage.loc[s.index, "usable_market_rows"].gt(0)]["league"].nunique())),
            usable_seasons=("season_end_year", lambda s: int(coverage.loc[s.index][coverage.loc[s.index, "usable_market_rows"].gt(0)]["season_end_year"].nunique())),
            mean_coverage_pct=("market_coverage_pct", "mean"),
        )
        .sort_values("market")
    )
    lines = [
        "# Other Football Markets Availability and Baseline Audit",
        "",
        f"Final classification: `{classification}`",
        "",
        "Scope: availability and raw market baseline diagnostics only. No predictive models, betting strategies, value searches, threshold optimization, Transfermarkt data, player features, lineups, team-name features, closing-odds selection, scraping, external APIs, or confirmed edge claims were used.",
        "",
        "## Coverage Summary",
        "",
        markdown_table(coverage_summary, ["market", "usable_rows_total", "usable_leagues", "usable_seasons", "mean_coverage_pct"], 20),
        "",
        "## Baseline Diagnostics, Overall",
        "",
        markdown_table(
            overall,
            [
                "market",
                "rows",
                "seasons",
                "leagues",
                "raw_market_log_loss",
                "raw_market_brier",
                "raw_market_ece",
                "implied_probability_average",
                "realised_target_rate",
                "bookmaker_overround_estimate",
                "multiclass_log_loss",
            ],
            20,
        ),
        "",
        "## Recommended Next Markets",
        "",
        markdown_table(recommendations, ["market", "recommendation", "usable_rows_total", "usable_seasons", "usable_leagues", "baseline_diagnostics_computable", "major_parsing_warnings"], 20),
        "",
        "No confirmed edge is claimed.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = add_market_probabilities(add_targets(load_matches()))
    coverage = coverage_rows(data)
    diagnostics = baseline_diagnostics(data)
    recommendations = recommendation_rows(coverage, diagnostics)
    classification = final_classification(recommendations)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(COVERAGE_PATH, index=False)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)
    recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)
    write_report(coverage, diagnostics, recommendations, classification)
    print(
        {
            "classification": classification,
            "coverage_rows": len(coverage),
            "diagnostic_rows": len(diagnostics),
            "recommendations": recommendations.set_index("market")["recommendation"].to_dict(),
        }
    )


if __name__ == "__main__":
    main()
