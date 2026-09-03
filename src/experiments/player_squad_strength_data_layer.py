from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.features.player_squad_strength import FEATURE_COLUMNS
from src.features.player_squad_strength import discover_player_data_files
from src.features.player_squad_strength import inspect_player_file
from src.features.player_squad_strength import latest_squad_features
from src.features.player_squad_strength import load_time_safe_observations
from src.features.player_squad_strength import normalize_name


AUDIT_REPORT = Path("outputs/reports/player_squad_strength_data_audit.md")
ENTITY_RESOLUTION = Path("outputs/reports/player_squad_strength_entity_resolution.csv")
FEATURE_COVERAGE = Path("outputs/reports/player_squad_strength_feature_coverage.csv")
BASELINE_REVIEW = Path("outputs/reports/player_squad_strength_baseline_review.md")
DETAIL_DIR = Path("outputs/player_squad_strength")

LEAGUES = ["E0", "I1", "SP1", "D1", "F1", "P1"]
LEAGUE_NAMES = {
    "E0": "Premier League",
    "I1": "Serie A",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "P1": "Liga Portugal",
}
MARKET_COLUMNS = ["AvgH", "AvgD", "AvgA"]
CLOSING_PREFIXES = ("B365C", "BWCH", "IWCH", "WHCH", "VCCH", "MaxCH", "AvgCH", "PSCH", "AHCh", "AvgCAH")


def league_match_path(league: str) -> Path:
    return Path("data/processed") / league / f"{league}_matches.csv"


def load_matches(league: str) -> pd.DataFrame:
    path = league_match_path(league)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    frame["league"] = league
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame["season_end_year"] = pd.to_numeric(frame["season_end_year"], errors="coerce")
    return frame.dropna(subset=["Date", "HomeTeam", "AwayTeam", "season_end_year"]).copy()


def load_all_matches() -> dict[str, pd.DataFrame]:
    return {league: load_matches(league) for league in LEAGUES}


def audit_sources() -> pd.DataFrame:
    files = discover_player_data_files()
    rows = [inspect_player_file(path) for path in files]
    if not rows:
        return pd.DataFrame(
            columns=[
                "path",
                "rows_sampled",
                "columns",
                "date_column",
                "club_column",
                "player_column",
                "market_value_column",
                "overall_column",
                "potential_column",
                "date_min",
                "date_max",
                "time_safe_candidate",
                "time_safety_reason",
                "read_error",
            ]
        )
    return pd.DataFrame(rows)


def entity_resolution(matches_by_league: dict[str, pd.DataFrame], observations: pd.DataFrame) -> pd.DataFrame:
    observed_clubs = set(observations["club_key"]) if len(observations) and "club_key" in observations.columns else set()
    observed_names = (
        observations.drop_duplicates("club_key").set_index("club_key")["club_name"].to_dict()
        if len(observations) and "club_key" in observations.columns
        else {}
    )
    rows = []
    for league, frame in matches_by_league.items():
        if frame.empty:
            rows.append(
                {
                    "league": league,
                    "match_team": "",
                    "normalized_match_team": "",
                    "mapped_club": "",
                    "status": "league_match_data_missing",
                    "source_file": "",
                    "matches": 0,
                }
            )
            continue
        teams = pd.concat([frame["HomeTeam"], frame["AwayTeam"]], ignore_index=True)
        counts = teams.value_counts()
        for team, count in counts.items():
            key = normalize_name(team)
            rows.append(
                {
                    "league": league,
                    "match_team": team,
                    "normalized_match_team": key,
                    "mapped_club": observed_names.get(key, ""),
                    "status": "matched_exact_normalized" if key in observed_clubs else "unmatched_no_player_dataset",
                    "source_file": "local_time_safe_player_observations" if key in observed_clubs else "",
                    "matches": int(count),
                }
            )
    return pd.DataFrame(rows)


def add_squad_features(frame: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if observations.empty:
        for side in ["home", "away"]:
            for column in FEATURE_COLUMNS:
                output[f"{side}_{column}"] = np.nan
        return add_derived_features(output)

    home_rows = []
    away_rows = []
    for _, row in output.iterrows():
        date = pd.Timestamp(row["Date"])
        home_rows.append(latest_squad_features(observations, normalize_name(row["HomeTeam"]), date))
        away_rows.append(latest_squad_features(observations, normalize_name(row["AwayTeam"]), date))
    home = pd.DataFrame(home_rows).add_prefix("home_")
    away = pd.DataFrame(away_rows).add_prefix("away_")
    output = pd.concat([output.reset_index(drop=True), home, away], axis=1)
    return add_derived_features(output)


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in FEATURE_COLUMNS:
        home = f"home_{column}"
        away = f"away_{column}"
        if home in output.columns and away in output.columns:
            output[f"home_minus_away_{column}"] = pd.to_numeric(output[home], errors="coerce") - pd.to_numeric(output[away], errors="coerce")

    strength = pd.to_numeric(output.get("home_minus_away_fifa_overall_top11"), errors="coerce")
    if strength.isna().all():
        strength = pd.to_numeric(output.get("home_minus_away_squad_market_value_top11"), errors="coerce")
        strength = np.log1p(strength.clip(lower=-0.99)) if isinstance(strength, pd.Series) else strength
    if isinstance(strength, pd.Series) and strength.notna().any():
        scaled = (strength - strength.mean()) / (strength.std(ddof=0) or 1.0)
        output["squad_strength_home_probability"] = 1.0 / (1.0 + np.exp(-scaled))
    else:
        output["squad_strength_home_probability"] = np.nan

    if all(column in output.columns for column in MARKET_COLUMNS):
        raw_h = 1.0 / pd.to_numeric(output["AvgH"], errors="coerce")
        raw_d = 1.0 / pd.to_numeric(output["AvgD"], errors="coerce")
        raw_a = 1.0 / pd.to_numeric(output["AvgA"], errors="coerce")
        total = raw_h + raw_d + raw_a
        output["market_home_probability"] = raw_h / total
        output["market_draw_probability"] = raw_d / total
        output["market_away_probability"] = raw_a / total
        output["market_probability_minus_squad_strength_probability"] = (
            output["market_home_probability"] - output["squad_strength_home_probability"]
        )
        output["odds_disagreement_with_squad_strength"] = output["market_probability_minus_squad_strength_probability"].abs()
    return output


def feature_coverage(matches_by_league: dict[str, pd.DataFrame], observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for league, frame in matches_by_league.items():
        if frame.empty:
            rows.append(
                {
                    "league": league,
                    "league_name": LEAGUE_NAMES.get(league, league),
                    "matches": 0,
                    "seasons": "",
                    "time_safe": False,
                    "home_feature_rows": 0,
                    "away_feature_rows": 0,
                    "both_feature_rows": 0,
                    "home_coverage": 0.0,
                    "away_coverage": 0.0,
                    "both_coverage": 0.0,
                    "status": "league_match_data_missing",
                }
            )
            continue
        featured = add_squad_features(frame, observations)
        key_feature = "squad_market_value_top11"
        if featured[f"home_{key_feature}"].notna().sum() == 0 and featured["home_fifa_overall_top11"].notna().sum() > 0:
            key_feature = "fifa_overall_top11"
        home_has = featured[f"home_{key_feature}"].notna()
        away_has = featured[f"away_{key_feature}"].notna()
        row = {
            "league": league,
            "league_name": LEAGUE_NAMES.get(league, league),
            "matches": len(frame),
            "seasons": ";".join(map(str, sorted(frame["season_end_year"].dropna().astype(int).unique()))),
            "time_safe": bool(len(observations) and (home_has & away_has).any()),
            "home_feature_rows": int(home_has.sum()),
            "away_feature_rows": int(away_has.sum()),
            "both_feature_rows": int((home_has & away_has).sum()),
            "home_coverage": float(home_has.mean()) if len(home_has) else 0.0,
            "away_coverage": float(away_has.mean()) if len(away_has) else 0.0,
            "both_coverage": float((home_has & away_has).mean()) if len(home_has) else 0.0,
            "status": "ready_for_baseline" if (home_has & away_has).sum() >= 200 else "insufficient_time_safe_coverage",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def multiclass_ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (confidence >= left) & (confidence <= right if idx == bins - 1 else confidence < right)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return ece


def make_target(frame: pd.DataFrame) -> pd.Series:
    mapping = {"H": 0, "D": 1, "A": 2}
    return frame["FTR"].map(mapping)


def squad_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if (
            column.startswith("home_squad_")
            or column.startswith("away_squad_")
            or column.startswith("home_fifa_")
            or column.startswith("away_fifa_")
            or column.startswith("home_minus_away_")
            or column in {"rating_depth_gap", "squad_strength_home_probability"}
        )
    ]
    return [column for column in columns if not any(closing in column for closing in CLOSING_PREFIXES)]


def baseline_rows(matches_by_league: dict[str, pd.DataFrame], observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if observations.empty:
        return pd.DataFrame(rows)
    for league, frame in matches_by_league.items():
        if frame.empty:
            continue
        featured = add_squad_features(frame, observations)
        columns = squad_feature_columns(featured)
        usable = featured.dropna(subset=["FTR"]).copy()
        usable = usable[make_target(usable).notna()].copy()
        coverage = usable[columns].notna().any(axis=1) if columns else pd.Series(False, index=usable.index)
        usable = usable[coverage].copy()
        if len(usable) < 200 or len(columns) == 0:
            continue
        seasons = sorted(usable["season_end_year"].dropna().astype(int).unique())
        for idx in range(2, len(seasons)):
            train_years = seasons[: idx - 1]
            validation_year = seasons[idx - 1]
            test_year = seasons[idx]
            train = usable[usable["season_end_year"].isin(train_years)].copy()
            test = usable[usable["season_end_year"].eq(test_year)].copy()
            if len(train) < 100 or len(test) == 0:
                continue
            y_train = make_target(train).astype(int)
            y_test = make_target(test).astype(int)
            preprocessor = ColumnTransformer(
                [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), columns)],
                remainder="drop",
            )
            model = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000, multi_class="auto"))])
            model.fit(train[columns], y_train)
            proba = model.predict_proba(test[columns])
            market = test[["market_home_probability", "market_draw_probability", "market_away_probability"]].to_numpy(dtype=float)
            rows.append(
                {
                    "league": league,
                    "test_year": int(test_year),
                    "train_years": ";".join(map(str, train_years)),
                    "validation_year": int(validation_year),
                    "test_matches": len(test),
                    "squad_log_loss": log_loss(y_test, proba, labels=[0, 1, 2]),
                    "market_log_loss": log_loss(y_test, market, labels=[0, 1, 2]),
                    "squad_brier": np.mean([brier_score_loss((y_test == klass).astype(int), proba[:, klass]) for klass in range(3)]),
                    "market_brier": np.mean([brier_score_loss((y_test == klass).astype(int), market[:, klass]) for klass in range(3)]),
                    "squad_ece": multiclass_ece(y_test.to_numpy(), proba),
                    "market_ece": multiclass_ece(y_test.to_numpy(), market),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[columns].copy()
    return view.to_markdown(index=False, headers=headers, floatfmt=".4f")


def write_audit_report(audit: pd.DataFrame, observations: pd.DataFrame, coverage: pd.DataFrame, classification: str) -> None:
    lines = [
        "# Player/Squad Strength Data Audit",
        "",
        "Scope: E0, I1, SP1, D1, F1, and Portugal/Liga Portugal when local match data exists.",
        "",
        "No external APIs were called. Raw match data was not edited. Closing odds are excluded from feature definitions.",
        "",
        "## Local Player Data Files",
        "",
        markdown_table(
            audit,
            ["path", "rows_sampled", "date_column", "club_column", "player_column", "market_value_column", "overall_column", "date_min", "date_max", "time_safe_candidate", "time_safety_reason"],
            ["Path", "Rows sampled", "Date", "Club", "Player", "Market value", "Overall", "Date min", "Date max", "Time-safe?", "Reason"],
        ),
        "",
        "## Time-Safe Observation Status",
        "",
        f"Loaded time-safe player observations: {len(observations)}",
        "",
        "A dataset is considered usable only when it has dated player-club observations and at least one pre-match strength field such as market value or FIFA overall. Current-only squads or undated ratings are diagnostic only and are not allowed into backtests.",
        "",
        "## Feature Coverage",
        "",
        markdown_table(
            coverage,
            ["league", "matches", "seasons", "both_feature_rows", "both_coverage", "status"],
            ["League", "Matches", "Seasons", "Both rows", "Both coverage", "Status"],
        ),
        "",
        f"Final classification: **{classification}**",
    ]
    AUDIT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_baseline_review(baseline: pd.DataFrame, classification: str) -> None:
    lines = [
        "# Player/Squad Strength Baseline Review",
        "",
        "This is a baseline strength-model gate only, not a betting strategy search.",
        "",
        "Validation rule: nested temporal splits only; scalers are fit inside the train fold. Closing odds are absent from squad feature matrices.",
        "",
    ]
    if baseline.empty:
        lines.extend(
            [
                "No baseline model was fit because no local time-safe player/squad strength dataset has sufficient match coverage.",
                "",
                f"Final classification: **{classification}**",
            ]
        )
    else:
        lines.extend(
            [
                markdown_table(
                    baseline,
                    ["league", "test_year", "test_matches", "squad_log_loss", "market_log_loss", "squad_brier", "market_brier", "squad_ece", "market_ece"],
                    ["League", "Test year", "Matches", "Squad log loss", "Market log loss", "Squad Brier", "Market Brier", "Squad ECE", "Market ECE"],
                ),
                "",
                f"Final classification: **{classification}**",
            ]
        )
    BASELINE_REVIEW.write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify(audit: pd.DataFrame, coverage: pd.DataFrame, baseline: pd.DataFrame) -> str:
    if audit.empty:
        return "insufficient coverage"
    if not bool(audit["time_safe_candidate"].fillna(False).any()):
        return "not time-safe"
    if coverage.empty or coverage["both_feature_rows"].fillna(0).sum() < 200:
        return "data partially ready"
    if baseline.empty:
        return "data partially ready"
    return "data ready"


def main() -> None:
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    matches_by_league = load_all_matches()
    audit = audit_sources()
    audit.to_csv(DETAIL_DIR / "player_squad_source_audit.csv", index=False)
    observations = load_time_safe_observations(audit)
    observations.to_csv(DETAIL_DIR / "time_safe_player_observations.csv", index=False)
    resolution = entity_resolution(matches_by_league, observations)
    coverage = feature_coverage(matches_by_league, observations)
    baseline = baseline_rows(matches_by_league, observations)
    classification = classify(audit, coverage, baseline)
    resolution.to_csv(ENTITY_RESOLUTION, index=False)
    coverage.to_csv(FEATURE_COVERAGE, index=False)
    baseline.to_csv(DETAIL_DIR / "player_squad_strength_baseline_metrics.csv", index=False)
    write_audit_report(audit, observations, coverage, classification)
    write_baseline_review(baseline, classification)
    print(AUDIT_REPORT)
    print(ENTITY_RESOLUTION)
    print(FEATURE_COVERAGE)
    print(BASELINE_REVIEW)
    print(f"classification={classification}")


if __name__ == "__main__":
    main()
