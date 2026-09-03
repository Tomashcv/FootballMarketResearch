from __future__ import annotations

from pathlib import Path
import math
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments import transfermarkt_proxy_predictive_audit as base


LAYER1 = {"E0", "D1", "I1", "SP1", "F1", "P1"}
CANDIDATES = ["N1", "B1", "T1", "SC0", "G1", "E1"]
AH_REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA"]
ONE_X_TWO = ["AvgH", "AvgD", "AvgA", "FTR"]
CLOSING_AH = ["AHCh", "AvgCAHH", "AvgCAHA"]
REPORT_PATH = Path("outputs/reports/layer2_league_availability_audit.md")
COVERAGE_PATH = Path("outputs/reports/layer2_league_coverage_by_season.csv")
MARKET_PATH = Path("outputs/reports/layer2_market_baseline_diagnostics.csv")
TM_PATH = Path("outputs/reports/layer2_transfermarkt_mapping_coverage.csv")
INCLUSION_PATH = Path("outputs/reports/layer2_recommended_inclusion_set.csv")
EXTRA_PATH = Path("outputs/reports/layer2_discovered_extra_leagues.csv")
PROXY_PATH = Path("data/processed/players/transfermarkt_valuation_only_club_strength_proxy.csv")
MAPPING_PATH = Path("data/manual/player_squad_team_name_mapping.csv")


def normalize(value: object) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def processed_path(league: str) -> Path:
    return Path("data/processed") / league / f"{league}_matches.csv"


def load_matches(league: str) -> pd.DataFrame | None:
    path = processed_path(league)
    if not path.exists():
        return None
    frame = pd.read_csv(path, low_memory=False)
    frame["league"] = league
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if "season_end_year" in frame.columns:
        frame["season_end_year"] = pd.to_numeric(frame["season_end_year"], errors="coerce")
    return frame


def add_ah_target(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    margin = pd.to_numeric(output["FTHG"], errors="coerce") - pd.to_numeric(output["FTAG"], errors="coerce")
    line = pd.to_numeric(output["AHh"], errors="coerce")
    adjusted = margin + line
    output["target_ah_home_cover"] = np.where(adjusted > 0, 1, np.where(adjusted < 0, 0, np.nan))
    output["target_ah_push"] = adjusted == 0
    output["target_1x2"] = output["FTR"].map({"H": 0, "D": 1, "A": 2}) if "FTR" in output.columns else np.nan
    return output


def ah_market_probability(frame: pd.DataFrame) -> pd.Series:
    home = 1.0 / pd.to_numeric(frame["AvgAHH"], errors="coerce")
    away = 1.0 / pd.to_numeric(frame["AvgAHA"], errors="coerce")
    return home / (home + away)


def ece_binary(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (probabilities >= left) & (probabilities <= right if idx == bins - 1 else probabilities < right)
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(probabilities[mask].mean()))
    return float(total)


def coverage_rows(league: str, frame: pd.DataFrame | None) -> list[dict[str, object]]:
    if frame is None:
        return []
    rows = []
    missing_required = [column for column in AH_REQUIRED if column not in frame.columns]
    years = sorted(pd.to_numeric(frame.get("season_end_year", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique())
    if not years:
        years = [pd.NA]
    for season in years:
        group = frame[frame["season_end_year"].eq(season)].copy() if season is not pd.NA else frame.copy()
        ah_present = [column for column in ["AHh", "AvgAHH", "AvgAHA"] if column in group.columns]
        one_x_two_present = [column for column in ["AvgH", "AvgD", "AvgA"] if column in group.columns]
        target_frame = add_ah_target(group) if not missing_required else group.copy()
        row = {
            "league": league,
            "season_end_year": int(season) if pd.notna(season) else pd.NA,
            "matches": int(len(group)),
            "one_x_two_odds_coverage": float(group[one_x_two_present].notna().all(axis=1).mean()) if len(one_x_two_present) == 3 else 0.0,
            "ah_odds_coverage": float(group[ah_present].notna().all(axis=1).mean()) if len(ah_present) == 3 else 0.0,
            "usable_ah_target_rows": int(target_frame.dropna(subset=AH_REQUIRED + ["target_ah_home_cover"]).shape[0]) if not missing_required else 0,
            "usable_1x2_target_rows": int(group.dropna(subset=[column for column in ONE_X_TWO if column in group.columns]).shape[0]) if set(ONE_X_TWO).issubset(group.columns) else 0,
            "big_home_favourite_ah_rows": int((pd.to_numeric(group["AHh"], errors="coerce") <= -1.0).sum()) if "AHh" in group.columns else 0,
            "average_ah_home_line": float(pd.to_numeric(group["AHh"], errors="coerce").mean()) if "AHh" in group.columns else np.nan,
            "average_home_ah_odds": float(pd.to_numeric(group["AvgAHH"], errors="coerce").mean()) if "AvgAHH" in group.columns else np.nan,
            "average_away_ah_odds": float(pd.to_numeric(group["AvgAHA"], errors="coerce").mean()) if "AvgAHA" in group.columns else np.nan,
            "closing_ah_coverage_diagnostic_only": float(group[CLOSING_AH].notna().all(axis=1).mean()) if set(CLOSING_AH).issubset(group.columns) else 0.0,
            "missing_required_columns": ",".join(missing_required),
        }
        rows.append(row)
    return rows


def market_diagnostics(league: str, frame: pd.DataFrame | None) -> list[dict[str, object]]:
    if frame is None or not set(AH_REQUIRED).issubset(frame.columns):
        return [
            {
                "league": league,
                "season_end_year": "all",
                "rows": 0,
                "raw_ah_market_log_loss": np.nan,
                "raw_ah_brier": np.nan,
                "raw_ah_ece": np.nan,
                "ah_home_cover_rate": np.nan,
                "average_ah_odds": np.nan,
                "push_rate": np.nan,
                "home_covers": 0,
                "away_covers": 0,
                "status": "not_computable",
            }
        ]
    target_frame = add_ah_target(frame)
    target_frame["market_p"] = ah_market_probability(target_frame)
    ah_rows = target_frame.dropna(subset=["target_ah_home_cover", "market_p", "AvgAHH", "AvgAHA"]).copy()
    rows = []
    for season_key, group in [("all", ah_rows)] + [(int(season), g) for season, g in ah_rows.groupby("season_end_year")]:
        if len(group) == 0:
            continue
        y = group["target_ah_home_cover"].astype(int).to_numpy()
        p = np.clip(group["market_p"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
        all_ah = target_frame[target_frame["season_end_year"].eq(season_key)] if season_key != "all" else target_frame
        rows.append(
            {
                "league": league,
                "season_end_year": season_key,
                "rows": int(len(group)),
                "raw_ah_market_log_loss": float(log_loss(y, p, labels=[0, 1])),
                "raw_ah_brier": float(brier_score_loss(y, p)),
                "raw_ah_ece": ece_binary(y, p),
                "ah_home_cover_rate": float(y.mean()),
                "average_ah_odds": float(pd.concat([group["AvgAHH"], group["AvgAHA"]]).astype(float).mean()),
                "push_rate": float(all_ah["target_ah_push"].mean()) if "target_ah_push" in all_ah.columns else np.nan,
                "home_covers": int((y == 1).sum()),
                "away_covers": int((y == 0).sum()),
                "status": "computed",
            }
        )
    return rows


def simple_candidates(match_team: str, mapped_names: list[str]) -> str:
    norm = normalize(match_team)
    candidates = []
    for name in mapped_names:
        mapped_norm = normalize(name)
        if not mapped_norm:
            continue
        if norm == mapped_norm or norm in mapped_norm or mapped_norm in norm:
            candidates.append(name)
    return "; ".join(sorted(set(candidates))[:5])


def tm_mapping_coverage(league: str, frame: pd.DataFrame | None) -> dict[str, object]:
    if frame is None:
        return {
            "league": league,
            "unique_match_teams": 0,
            "teams_mapped_to_transfermarkt": 0,
            "unmatched_teams": "",
            "ambiguous_candidates": "",
            "valuation_coverage_365d": np.nan,
            "valuation_coverage_180d": np.nan,
            "median_players_per_club_365d": np.nan,
            "tm_proxy_usable_for_predictive_diagnostics": False,
            "mapping_needs_manual_review": True,
            "status": "missing_processed_file",
        }
    teams = sorted(set(frame.get("HomeTeam", pd.Series(dtype=str)).dropna().astype(str)) | set(frame.get("AwayTeam", pd.Series(dtype=str)).dropna().astype(str)))
    if MAPPING_PATH.exists():
        mapping = pd.read_csv(MAPPING_PATH)
        league_mapping = mapping[mapping["league"].astype(str).eq(league)].copy()
    else:
        mapping = pd.DataFrame()
        league_mapping = pd.DataFrame()
    mapped_teams = set(league_mapping["match_team"].dropna().astype(str)) if len(league_mapping) else set()
    unmatched = [team for team in teams if team not in mapped_teams]
    mapped_names = mapping["match_team"].dropna().astype(str).unique().tolist() if len(mapping) else []
    candidate_text = " | ".join(
        f"{team}: {simple_candidates(team, mapped_names)}" for team in unmatched if simple_candidates(team, mapped_names)
    )
    if PROXY_PATH.exists():
        proxy = pd.read_csv(PROXY_PATH, low_memory=False)
        proxy_league = proxy[proxy["league"].astype(str).eq(league)].copy()
    else:
        proxy_league = pd.DataFrame()
    if len(proxy_league):
        cov365 = float(proxy_league[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean())
        cov180 = float(proxy_league[["home_tm_value_total_180d", "away_tm_value_total_180d"]].notna().all(axis=1).mean())
        players = pd.concat(
            [
                pd.to_numeric(proxy_league["home_tm_players_count_365d"], errors="coerce"),
                pd.to_numeric(proxy_league["away_tm_players_count_365d"], errors="coerce"),
            ],
            ignore_index=True,
        )
        median_players = float(players.median()) if players.notna().any() else np.nan
    else:
        cov365 = np.nan
        cov180 = np.nan
        median_players = np.nan
    usable = bool(len(proxy_league) and cov365 >= 0.7 and pd.notna(median_players) and median_players >= 11 and not unmatched)
    needs_review = bool(unmatched or not len(proxy_league))
    return {
        "league": league,
        "unique_match_teams": int(len(teams)),
        "teams_mapped_to_transfermarkt": int(len(mapped_teams & set(teams))),
        "unmatched_teams": "; ".join(unmatched),
        "ambiguous_candidates": candidate_text,
        "valuation_coverage_365d": cov365,
        "valuation_coverage_180d": cov180,
        "median_players_per_club_365d": median_players,
        "tm_proxy_usable_for_predictive_diagnostics": usable,
        "mapping_needs_manual_review": needs_review,
        "status": "computed",
    }


def discovered_extra_leagues() -> pd.DataFrame:
    rows = []
    for path in sorted(Path("data/processed").glob("*/*_matches.csv")):
        league = path.parent.name
        if league in LAYER1 or league in CANDIDATES:
            continue
        frame = pd.read_csv(path, low_memory=False)
        season = pd.to_numeric(frame.get("season_end_year", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "league": league,
                "path": str(path),
                "rows": int(len(frame)),
                "season_min": int(season.min()) if season.notna().any() else pd.NA,
                "season_max": int(season.max()) if season.notna().any() else pd.NA,
                "status": "discovered_extra_league_not_auto_included",
            }
        )
    return pd.DataFrame(rows)


def classify_league(league: str, frame: pd.DataFrame | None, coverage: pd.DataFrame, market: pd.DataFrame, tm: dict[str, object]) -> str:
    if frame is None:
        return "layer2_missing_processed_file"
    missing = [column for column in AH_REQUIRED if column not in frame.columns]
    if missing:
        if set(ONE_X_TWO).issubset(frame.columns):
            return "layer2_ready_1x2_only"
        return "reject_for_now"
    league_cov = coverage[coverage["league"].eq(league)]
    usable_total = int(league_cov["usable_ah_target_rows"].sum()) if len(league_cov) else 0
    usable_seasons = int((league_cov["usable_ah_target_rows"] > 0).sum()) if len(league_cov) else 0
    market_ok = len(market[(market["league"].eq(league)) & (market["season_end_year"].astype(str).eq("all")) & market["status"].eq("computed")]) > 0
    if usable_seasons >= 3 and usable_total >= 500 and market_ok:
        return "layer2_ready_ah"
    if usable_seasons >= 1 and market_ok:
        return "layer2_insufficient_ah_coverage"
    return "reject_for_now"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def write_report(
    coverage: pd.DataFrame,
    market: pd.DataFrame,
    tm: pd.DataFrame,
    inclusion: pd.DataFrame,
    extras: pd.DataFrame,
    final_classification: str,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    availability = inclusion[[
        "league",
        "processed_file_exists",
        "rows",
        "season_min",
        "season_max",
        "missing_required_columns",
        "classification",
    ]]
    market_all = market[market["season_end_year"].astype(str).eq("all")].copy()
    lines = [
        "# Layer 2 League Availability And Coverage Audit",
        "",
        f"Final classification: `{final_classification}`",
        "",
        "Scope: report-only Layer 2 availability, AH/1X2 coverage, raw AH market diagnostics, and Transfermarkt mapping coverage. No predictive models, betting strategies, value searches, threshold optimization, lineups, or `players.current_club_*` fields were used.",
        "",
        "Closing AH columns are diagnostic-only coverage fields. Transfermarkt proxy coverage is read from existing generated proxy output only; missing mappings are not auto-accepted.",
        "",
        "## Candidate Availability",
        "",
        markdown_table(availability, availability.columns.tolist(), max_rows=20),
        "",
        "## Recommended Inclusion",
        "",
        markdown_table(inclusion, ["league", "classification", "inclusion_reason"], max_rows=20),
        "",
        "## Coverage By Season",
        "",
        markdown_table(coverage, ["league", "season_end_year", "matches", "one_x_two_odds_coverage", "ah_odds_coverage", "usable_ah_target_rows", "usable_1x2_target_rows", "big_home_favourite_ah_rows", "closing_ah_coverage_diagnostic_only"], max_rows=80),
        "",
        "## Raw AH Market Diagnostics",
        "",
        markdown_table(market_all, ["league", "rows", "raw_ah_market_log_loss", "raw_ah_brier", "raw_ah_ece", "ah_home_cover_rate", "average_ah_odds", "push_rate", "home_covers", "away_covers", "status"], max_rows=20),
        "",
        "## Transfermarkt Mapping Coverage",
        "",
        markdown_table(tm, ["league", "unique_match_teams", "teams_mapped_to_transfermarkt", "valuation_coverage_365d", "valuation_coverage_180d", "median_players_per_club_365d", "tm_proxy_usable_for_predictive_diagnostics", "mapping_needs_manual_review", "status"], max_rows=20),
        "",
        "## Discovered Extra Leagues",
        "",
        markdown_table(extras, ["league", "rows", "season_min", "season_max", "status"], max_rows=20),
        "",
        "No pooled Layer 2 model was run.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frames = {league: load_matches(league) for league in CANDIDATES}
    coverage = pd.DataFrame([row for league, frame in frames.items() for row in coverage_rows(league, frame)])
    market = pd.DataFrame([row for league, frame in frames.items() for row in market_diagnostics(league, frame)])
    tm_rows = [tm_mapping_coverage(league, frame) for league, frame in frames.items()]
    tm = pd.DataFrame(tm_rows)
    extras = discovered_extra_leagues()
    inclusion_rows = []
    for league, frame in frames.items():
        path = processed_path(league)
        seasons = pd.to_numeric(frame.get("season_end_year", pd.Series(dtype=float)), errors="coerce") if frame is not None else pd.Series(dtype=float)
        missing_required = [column for column in AH_REQUIRED if frame is None or column not in frame.columns]
        classification = classify_league(league, frame, coverage, market, tm[tm["league"].eq(league)].iloc[0].to_dict())
        reason_parts = []
        if frame is None:
            reason_parts.append("processed file missing")
        elif missing_required:
            reason_parts.append(f"missing required AH columns: {', '.join(missing_required)}")
        else:
            league_cov = coverage[coverage["league"].eq(league)]
            reason_parts.append(f"usable AH rows={int(league_cov['usable_ah_target_rows'].sum())}")
            reason_parts.append(f"usable AH seasons={int((league_cov['usable_ah_target_rows'] > 0).sum())}")
        if classification == "layer2_ready_ah" and bool(tm[tm["league"].eq(league)]["mapping_needs_manual_review"].iloc[0]):
            reason_parts.append("AH-ready; Transfermarkt mapping still needs review")
        inclusion_rows.append(
            {
                "league": league,
                "processed_file_exists": bool(path.exists()),
                "path": str(path),
                "rows": int(len(frame)) if frame is not None else 0,
                "season_min": int(seasons.min()) if seasons.notna().any() else pd.NA,
                "season_max": int(seasons.max()) if seasons.notna().any() else pd.NA,
                "missing_required_columns": ",".join(missing_required),
                "classification": classification,
                "inclusion_reason": "; ".join(reason_parts),
            }
        )
    inclusion = pd.DataFrame(inclusion_rows)
    ready_count = int(inclusion["classification"].eq("layer2_ready_ah").sum())
    if ready_count >= 2:
        final = "layer2_ready_for_pooled_market_correction"
    elif ready_count >= 1 or inclusion["classification"].eq("layer2_ready_1x2_only").any():
        final = "layer2_partial_ready"
    else:
        final = "layer2_not_ready"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(COVERAGE_PATH, index=False)
    market.to_csv(MARKET_PATH, index=False)
    tm.to_csv(TM_PATH, index=False)
    inclusion.to_csv(INCLUSION_PATH, index=False)
    extras.to_csv(EXTRA_PATH, index=False)
    write_report(coverage, market, tm, inclusion, extras, final)
    print(
        {
            "coverage_rows": len(coverage),
            "market_rows": len(market),
            "tm_rows": len(tm),
            "inclusion_rows": len(inclusion),
            "extra_rows": len(extras),
            "classification": final,
        }
    )


if __name__ == "__main__":
    main()
