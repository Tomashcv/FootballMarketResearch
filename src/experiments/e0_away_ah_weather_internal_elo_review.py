from pathlib import Path

import pandas as pd

from src.common.paths import get_league_matches_path
from src.features.contextual_features import build_contextual_features
from src.features.travel_features import build_travel_features
from src.features.weather_features import add_weather_features
from src.features.weather_features import add_weather_shock_features
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import asian_profit
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


LEAGUE = "E0"
REPORT_PATH = Path("outputs/reports/e0_away_ah_weather_internal_elo_review.md")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/weather_internal_elo_review")

MIN_VALIDATION_YEARS = 2
MIN_VALIDATION_BETS = 50
MIN_POSITIVE_VALIDATION_YEARS = 2


def fmt(value, digits=3):
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def pct(value):
    if value is None or pd.isna(value):
        return ""
    return f"{100.0 * float(value):.2f}%"


def load_coordinates():
    coordinates = pd.read_csv("data/external/stadiums/stadiums_with_gps_coordinates.csv")
    overrides = pd.read_csv("data/manual/team_stadium_overrides.csv")
    return coordinates, overrides


def load_weather():
    weather = pd.read_csv("data/external/weather/historical_match_weather.csv")
    normals = pd.read_csv("data/external/weather/monthly_climate_normals.csv")
    return weather, normals


def prepare_data():
    matches = pd.read_csv(get_league_matches_path(LEAGUE), low_memory=False)
    weather, normals = load_weather()

    # Restrict to locally cached match-weather scope. This avoids silently mixing
    # weather-covered and uncovered historical periods.
    weather_keys = weather[["Date", "HomeTeam", "AwayTeam"]].copy()
    weather_keys["Date"] = pd.to_datetime(weather_keys["Date"], errors="coerce").dt.normalize()
    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce").dt.normalize()
    matches = matches.merge(weather_keys.drop_duplicates(), on=["Date", "HomeTeam", "AwayTeam"], how="inner")

    dataframe = build_contextual_features(matches)
    coordinates, overrides = load_coordinates()
    dataframe = build_travel_features(dataframe, coordinates, overrides)
    dataframe = add_weather_features(dataframe, weather)
    dataframe = add_weather_shock_features(dataframe, normals)

    required = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "season_end_year",
        "AHh",
        "AvgAHH",
        "AvgAHA",
    ]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dataframe["ah_line"] = pd.to_numeric(dataframe["AHh"], errors="coerce")
    dataframe["home_ah_odds"] = pd.to_numeric(dataframe["AvgAHH"], errors="coerce")
    dataframe["away_ah_odds"] = pd.to_numeric(dataframe["AvgAHA"], errors="coerce")
    dataframe["season_end_year"] = pd.to_numeric(dataframe["season_end_year"], errors="coerce")
    dataframe = dataframe.dropna(
        subset=[
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "season_end_year",
            "ah_line",
            "home_ah_odds",
            "away_ah_odds",
        ]
    ).copy()
    dataframe = dataframe[(dataframe["home_ah_odds"] > 1.0) & (dataframe["away_ah_odds"] > 1.0)].copy()
    dataframe["season_end_year"] = dataframe["season_end_year"].astype(int)
    dataframe["away_margin"] = dataframe["FTAG"].astype(float) - dataframe["FTHG"].astype(float)
    dataframe["away_handicap"] = -dataframe["ah_line"]
    dataframe["profit"] = dataframe.apply(
        lambda row: asian_profit(row["away_margin"], row["away_handicap"], row["away_ah_odds"]),
        axis=1,
    )

    dataframe["home_raw_implied"] = 1.0 / dataframe["home_ah_odds"]
    dataframe["away_raw_implied"] = 1.0 / dataframe["away_ah_odds"]
    dataframe["overround"] = dataframe["home_raw_implied"] + dataframe["away_raw_implied"]
    dataframe["away_market_probability"] = dataframe["away_raw_implied"] / dataframe["overround"]

    if {"AvgCAHH", "AvgCAHA"}.issubset(dataframe.columns):
        close_home = 1.0 / pd.to_numeric(dataframe["AvgCAHH"], errors="coerce")
        close_away = 1.0 / pd.to_numeric(dataframe["AvgCAHA"], errors="coerce")
        dataframe["clv_probability_pp"] = ((close_away / (close_home + close_away)) - dataframe["away_market_probability"]) * 100.0
    else:
        dataframe["clv_probability_pp"] = pd.NA

    if "AHCh" in dataframe.columns:
        dataframe["line_move_to_away"] = dataframe["ah_line"] - pd.to_numeric(dataframe["AHCh"], errors="coerce")
    else:
        dataframe["line_move_to_away"] = pd.NA

    return dataframe.reset_index(drop=True)


def make_filter(name, description, func):
    return {"name": name, "description": description, "func": func}


def true_filter(dataframe):
    return pd.Series(True, index=dataframe.index)


def strategy_filter_sets():
    return {
        "original_nested": [make_filter("none", "Original nested AH threshold only", true_filter)],
        "away_odds_ge_1_85": [
            make_filter("away_odds_ge_1_85", "Away AH odds >= 1.85", lambda df: df["away_ah_odds"] >= 1.85)
        ],
        "travel_burden": [
            make_filter(
                f"travel_lt_{cutoff}km",
                f"Exclude away travel >= {cutoff} km",
                lambda df, cutoff=cutoff: df["travel_distance_km"].isna() | (df["travel_distance_km"] < cutoff),
            )
            for cutoff in [200, 275, 350, 425]
        ],
        "rest_disadvantage": [
            make_filter(
                f"no_away_rest_le_{rest}_diff_ge_{diff}",
                f"Exclude away rest <= {rest} days with home rest advantage >= {diff}",
                lambda df, rest=rest, diff=diff: ~(
                    (df["away_rest_days"] <= rest) & ((df["home_rest_days"] - df["away_rest_days"]) >= diff)
                ),
            )
            for rest, diff in [(3, 1), (4, 1), (4, 2), (5, 2)]
        ],
        "weather_match_features": [
            make_filter(
                f"wind_lt_{wind}_precip_lt_{precip}",
                f"Require weather present, wind < {wind} kph and precip < {precip} mm",
                lambda df, wind=wind, precip=precip: (
                    df["has_weather"]
                    & (pd.to_numeric(df["weather_wind_speed_kph"], errors="coerce") < wind)
                    & (pd.to_numeric(df["weather_precipitation_mm"], errors="coerce") < precip)
                ),
            )
            for wind, precip in [(35, 8), (40, 10), (45, 12), (50, 15)]
        ],
        "climate_shock_features": [
            make_filter(
                f"temp_shock_abs_le_{temp}_wind_shock_le_{wind}",
                f"Require climate shock present, abs temp shock <= {temp} C and wind shock <= {wind} kph",
                lambda df, temp=temp, wind=wind: (
                    df["has_weather_shock"]
                    & (pd.to_numeric(df["away_temperature_shock_c"], errors="coerce").abs() <= temp)
                    & (pd.to_numeric(df["away_wind_speed_shock_kph"], errors="coerce") <= wind)
                ),
            )
            for temp, wind in [(6, 18), (8, 22), (10, 25), (12, 30)]
        ],
        "internal_elo_features": [
            make_filter(
                f"home_internal_elo_diff_le_{cutoff}",
                f"Exclude away teams rated too far below home: home-away internal Elo <= {cutoff}",
                lambda df, cutoff=cutoff: pd.to_numeric(df["internal_elo_diff_home_minus_away"], errors="coerce") <= cutoff,
            )
            for cutoff in [100, 150, 200, 250]
        ],
        "market_internal_elo_disagreement": [
            make_filter(
                f"market_home_minus_elo_le_{cutoff}",
                f"Require market home probability minus internal Elo home probability <= {cutoff}",
                lambda df, cutoff=cutoff: pd.to_numeric(df["market_home_prob_minus_internal_elo_prob"], errors="coerce") <= cutoff,
            )
            for cutoff in [-0.05, 0.0, 0.05, 0.10]
        ],
        "combined_local_context": [
            make_filter(
                f"odds_travel_rest_elo_weather_{travel}_{elo}_{wind}",
                "Away odds >= 1.85 plus no severe travel/rest, no extreme weather, and not extreme internal Elo mismatch",
                lambda df, travel=travel, elo=elo, wind=wind: (
                    (df["away_ah_odds"] >= 1.85)
                    & (df["travel_distance_km"].isna() | (df["travel_distance_km"] < travel))
                    & ~((df["away_rest_days"] <= 4) & ((df["home_rest_days"] - df["away_rest_days"]) >= 2))
                    & (pd.to_numeric(df["internal_elo_diff_home_minus_away"], errors="coerce") <= elo)
                    & (pd.to_numeric(df["weather_wind_speed_kph"], errors="coerce") < wind)
                ),
            )
            for travel, elo, wind in [(350, 200, 45), (425, 200, 50), (425, 250, 45), (500, 250, 50)]
        ],
    }


def apply_spec(dataframe, threshold, spec):
    return dataframe[(dataframe["ah_line"] <= threshold) & spec["func"](dataframe)].copy()


def evaluate_validation(validation, threshold, spec):
    selected = apply_spec(validation, threshold, spec)
    if len(selected) == 0:
        return None
    summary = summarize(selected)
    by_year = selected.groupby("season_end_year")["profit"].mean()
    return {
        "threshold": threshold,
        "filter_name": spec["name"],
        "filter_description": spec["description"],
        "validation_bets": summary["bets"],
        "validation_profit": summary["profit"],
        "validation_roi": summary["roi"],
        "validation_z_score": summary["z_score"],
        "validation_positive_years": int((by_year > 0).sum()),
        "validation_min_year_roi": float(by_year.min()),
    }


def select_spec(validation, specs):
    candidates = []
    for threshold in THRESHOLDS:
        for spec in specs:
            result = evaluate_validation(validation, threshold, spec)
            if result is None:
                continue
            if result["validation_bets"] < MIN_VALIDATION_BETS:
                continue
            if result["validation_roi"] <= 0:
                continue
            if result["validation_positive_years"] < MIN_POSITIVE_VALIDATION_YEARS:
                continue
            if result["validation_min_year_roi"] <= 0:
                continue
            candidates.append(result)
    if not candidates:
        return None, pd.DataFrame()
    frame = pd.DataFrame(candidates).sort_values(
        ["validation_positive_years", "validation_min_year_roi", "validation_roi", "validation_bets"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    frame["validation_rank"] = frame.index + 1
    selected = frame.iloc[0].to_dict()
    spec = next(item for item in specs if item["name"] == selected["filter_name"])
    return {**selected, "filter_spec": spec}, frame


def run_nested(dataframe, strategy, specs):
    years = sorted(dataframe["season_end_year"].unique())
    by_year_rows = []
    bets = []
    candidates = []
    for test_year in years:
        validation_years = [year for year in years if year < test_year]
        if len(validation_years) < MIN_VALIDATION_YEARS:
            continue
        validation = dataframe[dataframe["season_end_year"].isin(validation_years)]
        test = dataframe[dataframe["season_end_year"] == test_year]
        selected, candidate_frame = select_spec(validation, specs)
        if selected is None:
            by_year_rows.append(
                {
                    "strategy": strategy,
                    "test_year": test_year,
                    "selected_threshold": pd.NA,
                    "selected_filter": "no_valid_validation_candidate",
                    "test_bets": 0,
                    "test_profit": 0.0,
                    "test_roi": 0.0,
                    "test_z_score": 0.0,
                    "test_max_drawdown": 0.0,
                }
            )
            continue
        selected_test = apply_spec(test, float(selected["threshold"]), selected["filter_spec"])
        summary = summarize(selected_test)
        by_year_rows.append(
            {
                "strategy": strategy,
                "test_year": test_year,
                "selected_threshold": selected["threshold"],
                "selected_filter": selected["filter_name"],
                "validation_bets": selected["validation_bets"],
                "validation_roi": selected["validation_roi"],
                "validation_min_year_roi": selected["validation_min_year_roi"],
                "test_bets": summary["bets"],
                "test_profit": summary["profit"],
                "test_roi": summary["roi"],
                "test_z_score": summary["z_score"],
                "test_max_drawdown": summary["max_drawdown"],
            }
        )
        if len(selected_test) > 0:
            selected_test = selected_test.copy()
            selected_test["strategy"] = strategy
            selected_test["nested_test_year"] = test_year
            selected_test["selected_threshold"] = selected["threshold"]
            selected_test["selected_filter"] = selected["filter_name"]
            bets.append(selected_test)
        if len(candidate_frame) > 0:
            candidate_frame = candidate_frame.copy()
            candidate_frame["strategy"] = strategy
            candidate_frame["test_year"] = test_year
            candidates.append(candidate_frame)
    return (
        pd.DataFrame(by_year_rows),
        pd.concat(bets, ignore_index=True) if bets else pd.DataFrame(),
        pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame(),
    )


def clv_summary(dataframe):
    if len(dataframe) == 0:
        return {"avg_clv_pp": pd.NA, "clv_positive_rate": pd.NA, "avg_line_move_to_away": pd.NA}
    clv = pd.to_numeric(dataframe["clv_probability_pp"], errors="coerce")
    line_move = pd.to_numeric(dataframe["line_move_to_away"], errors="coerce")
    return {
        "avg_clv_pp": float(clv.mean()) if clv.notna().any() else pd.NA,
        "clv_positive_rate": float((clv > 0).mean()) if clv.notna().any() else pd.NA,
        "avg_line_move_to_away": float(line_move.mean()) if line_move.notna().any() else pd.NA,
    }


def concentration(dataframe):
    if len(dataframe) == 0:
        return {"top3_home_bet_share": pd.NA, "top3_away_bet_share": pd.NA, "home_hhi_bets": pd.NA, "away_hhi_bets": pd.NA}

    def top3(column):
        return float(dataframe[column].value_counts(normalize=True).head(3).sum())

    def hhi(column):
        shares = dataframe[column].value_counts(normalize=True)
        return float((shares * shares).sum())

    return {
        "top3_home_bet_share": top3("HomeTeam"),
        "top3_away_bet_share": top3("AwayTeam"),
        "home_hhi_bets": hhi("HomeTeam"),
        "away_hhi_bets": hhi("AwayTeam"),
    }


def overall_row(strategy, bets):
    summary = summarize(bets)
    row = {
        "strategy": strategy,
        "bets": summary["bets"],
        "profit": summary["profit"],
        "roi": summary["roi"],
        "z_score": summary["z_score"],
        "max_drawdown": summary["max_drawdown"],
    }
    row.update(clv_summary(bets))
    row.update(concentration(bets))
    return row


def seasonal_rows(strategy, bets):
    rows = []
    for season, group in bets.groupby("season_end_year"):
        summary = summarize(group)
        row = {
            "strategy": strategy,
            "season": int(season),
            "bets": summary["bets"],
            "profit": summary["profit"],
            "roi": summary["roi"],
            "z_score": summary["z_score"],
            "max_drawdown": summary["max_drawdown"],
        }
        row.update(clv_summary(group))
        row.update(concentration(group))
        rows.append(row)
    return rows


def markdown_table(dataframe, columns, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in dataframe.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if column == "roi" or column.endswith("_rate") or column.endswith("_share"):
                values.append(pct(value))
            elif isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def final_decision(overall):
    original = overall[overall["strategy"] == "original_nested"].iloc[0]
    challengers = overall[overall["strategy"] != "original_nested"].copy()
    viable = challengers[(challengers["bets"] >= 150) & (challengers["profit"] > 0) & (challengers["roi"] > 0)]
    if len(viable) == 0:
        return "reject"
    best = viable.sort_values(["z_score", "roi"], ascending=[False, False]).iloc[0]
    clv_improves = best["avg_clv_pp"] > original["avg_clv_pp"] and best["avg_clv_pp"] > 0
    robust_improves = best["z_score"] > original["z_score"] and best["max_drawdown"] <= original["max_drawdown"]
    concentration_improves = best["top3_home_bet_share"] <= original["top3_home_bet_share"]
    if clv_improves and robust_improves and concentration_improves:
        return "live shadow candidate"
    return "paper trade only"


def write_report(dataframe, by_year_frames, bet_frames, candidate_frames):
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    by_year = pd.concat(by_year_frames, ignore_index=True)
    bets = pd.concat(bet_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    by_year.to_csv(DETAIL_DIR / "nested_by_year.csv", index=False)
    bets.to_csv(DETAIL_DIR / "nested_bets.csv", index=False)
    candidates.to_csv(DETAIL_DIR / "nested_candidates.csv", index=False)

    grouped_bets = [(name, group.copy()) for name, group in bets.groupby("strategy", sort=False)]
    overall = pd.DataFrame([overall_row(name, group) for name, group in grouped_bets])
    seasonal = pd.DataFrame([row for name, group in grouped_bets for row in seasonal_rows(name, group)])
    overall.to_csv(DETAIL_DIR / "overall.csv", index=False)
    seasonal.to_csv(DETAIL_DIR / "seasonal.csv", index=False)
    decision = final_decision(overall)

    selected = by_year[["strategy", "test_year", "selected_threshold", "selected_filter", "validation_roi", "validation_min_year_roi"]].copy()

    lines = [
        "# E0 Away AH Weather + Internal Elo Review",
        "",
        "Scope: E0 / Premier League Away Asian Handicap when home AH line indicates a large home favourite.",
        "",
        "Method: nested temporal validation only. AH thresholds and contextual filters are selected from prior validation seasons only. Bet-time-safe inputs use main/open odds, local travel/rest/weather/climate caches, and internal pre-match Elo from past processed matches. External ClubElo is not used. Closing AH columns are used only after selection for CLV diagnostics.",
        "",
        f"Local feature scope after cached weather join: {len(dataframe)} AH rows across seasons {int(dataframe['season_end_year'].min())}-{int(dataframe['season_end_year'].max())}.",
        "",
        "## Overall Results",
        "",
        markdown_table(
            overall,
            ["strategy", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "top3_away_bet_share", "home_hhi_bets", "away_hhi_bets"],
            ["Strategy", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV + rate", "Top3 home share", "Top3 away share", "Home HHI", "Away HHI"],
        ),
        "",
        "## Season By Season",
        "",
        markdown_table(
            seasonal,
            ["strategy", "season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "top3_away_bet_share"],
            ["Strategy", "Season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV + rate", "Top3 home share", "Top3 away share"],
        ),
        "",
        "## Selected Filters",
        "",
        markdown_table(
            selected,
            ["strategy", "test_year", "selected_threshold", "selected_filter", "validation_roi", "validation_min_year_roi"],
            ["Strategy", "Test season", "AH threshold", "Selected prior-validation filter", "Validation ROI", "Min validation season ROI"],
        ),
        "",
        "## Interpretation",
        "",
        "- The experiment is controlled to named local contextual filters; it is not a broad model search.",
        "- Weather and climate-shock tests use only locally cached weather/normals. Missing normals are not backfilled.",
        "- Internal Elo is computed chronologically from processed match results; ratings are recorded before each match result updates the teams.",
        "- Closing odds are not used as selection features; CLV is diagnostic only.",
        "",
        "## Final Decision",
        "",
        f"**{decision}**",
        "",
    ]
    if decision == "paper trade only":
        lines.append("Rationale: at least one challenger remains historically positive, but CLV/robustness/concentration do not jointly improve enough for live shadow or confirmed status.")
    elif decision == "reject":
        lines.append("Rationale: local contextual challengers did not preserve enough out-of-sample performance under nested validation.")
    else:
        lines.append("Rationale: the best challenger improves CLV and robustness versus the original, but prospective paper tracking is still required before confirmation.")
    lines.append("")
    lines.append("Do not call this a confirmed edge.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    dataframe = prepare_data()
    by_year_frames = []
    bet_frames = []
    candidate_frames = []
    for strategy, specs in strategy_filter_sets().items():
        by_year, bets, candidates = run_nested(dataframe, strategy, specs)
        by_year_frames.append(by_year)
        if len(bets) > 0:
            bet_frames.append(bets)
        if len(candidates) > 0:
            candidate_frames.append(candidates)
    write_report(dataframe, by_year_frames, bet_frames, candidate_frames)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
