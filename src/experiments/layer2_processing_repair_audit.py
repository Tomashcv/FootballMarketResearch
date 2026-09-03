from __future__ import annotations

from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from pandas.errors import ParserError
from sklearn.metrics import brier_score_loss, log_loss

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.build_league_dataset import clean_column_names, parse_match_date, read_raw_csv


CANDIDATES = ["N1", "B1", "T1", "SC0", "G1"]
EXTENDED = ["E1", "E2", "E3"]
ALL_AUDIT = CANDIDATES + EXTENDED
AH_REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "AHh", "AvgAHH", "AvgAHA"]
MATCH_REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ONE_X_TWO = ["AvgH", "AvgD", "AvgA", "FTR"]
CLOSING_AH = ["AHCh", "AvgCAHH", "AvgCAHA"]
REPORT_PATH = Path("outputs/reports/layer2_processing_repair_audit.md")
DISCOVERY_PATH = Path("outputs/reports/layer2_processing_file_discovery.csv")
COVERAGE_PATH = Path("outputs/reports/layer2_extended_coverage_by_season.csv")
MARKET_PATH = Path("outputs/reports/layer2_extended_market_baseline_diagnostics.csv")
TM_CANDIDATES_PATH = Path("outputs/reports/layer2_extended_transfermarkt_mapping_candidates.csv")
TM_COVERAGE_PATH = Path("outputs/reports/layer2_extended_transfermarkt_mapping_coverage.csv")
INCLUSION_PATH = Path("outputs/reports/layer2_extended_recommended_inclusion_set.csv")
PROXY_PATH = Path("data/processed/players/transfermarkt_valuation_only_club_strength_proxy.csv")
MAPPING_PATH = Path("data/manual/player_squad_team_name_mapping.csv")
CLUBS_PATH = Path("data/external/players/transfermarkt_raw/player_scores/clubs.csv")

TM_COMPETITIONS = {
    "N1": "NL1",
    "B1": "BE1",
    "T1": "TR1",
    "SC0": "SC1",
    "G1": "GR1",
    "E1": None,
    "E2": None,
    "E3": None,
}


def normalize(value: object) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def processed_path(league: str) -> Path:
    return Path("data/processed") / league / f"{league}_matches.csv"


def raw_files_for(league: str) -> list[Path]:
    root = Path("data/raw") / league
    files = sorted(root.glob("*.csv"))
    files.extend(sorted((root / "seasons").glob("*.csv")))
    return files


def infer_season(frame: pd.DataFrame) -> tuple[int | None, int | None]:
    if frame.empty or "Date" not in frame.columns:
        return None, None
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    if dates.notna().sum() == 0:
        return None, None
    min_date = dates.min()
    max_date = dates.max()
    return int(min_date.year), int(max_date.year)


def discover_files() -> pd.DataFrame:
    rows = []
    for league in ALL_AUDIT:
        paths = raw_files_for(league)
        if processed_path(league).exists():
            paths.append(processed_path(league))
        if not paths:
            rows.append(
                {
                    "league": league,
                    "path": "",
                    "storage_layer": "missing",
                    "rows": 0,
                    "columns": "",
                    "season_min": pd.NA,
                    "season_max": pd.NA,
                    "required_match_columns_present": False,
                    "ah_columns_present": False,
                    "can_process_to_standard_matches": False,
                    "status": "no_local_source_file_found",
                }
            )
            continue
        for path in paths:
            storage = "processed" if "/processed/" in str(path) else "raw"
            try:
                frame, _, _, _ = read_raw_csv(path) if storage == "raw" else (pd.read_csv(path, low_memory=False), "", "", 0)
                frame = clean_column_names(frame)
                if "Date" in frame.columns:
                    frame = parse_match_date(frame)
                season_start, season_end = infer_season(frame)
                rows.append(
                    {
                        "league": league,
                        "path": str(path),
                        "storage_layer": storage,
                        "rows": int(len(frame)),
                        "columns": ";".join(map(str, frame.columns)),
                        "season_min": season_start,
                        "season_max": season_end,
                        "required_match_columns_present": set(MATCH_REQUIRED).issubset(frame.columns),
                        "ah_columns_present": {"AHh", "AvgAHH", "AvgAHA"}.issubset(frame.columns),
                        "can_process_to_standard_matches": storage == "raw" and set(MATCH_REQUIRED).issubset(frame.columns),
                        "status": "compatible" if set(MATCH_REQUIRED).issubset(frame.columns) else "missing_required_match_columns",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "league": league,
                        "path": str(path),
                        "storage_layer": storage,
                        "rows": 0,
                        "columns": "",
                        "season_min": pd.NA,
                        "season_max": pd.NA,
                        "required_match_columns_present": False,
                        "ah_columns_present": False,
                        "can_process_to_standard_matches": False,
                        "status": f"read_failed:{type(exc).__name__}:{exc}",
                    }
                )
    return pd.DataFrame(rows)


def process_direct_raw(league: str) -> dict[str, object]:
    if processed_path(league).exists():
        return {"league": league, "processed_created": False, "status": "processed_file_already_exists", "rows": pd.read_csv(processed_path(league), low_memory=False).shape[0]}
    files = raw_files_for(league)
    if not files:
        return {"league": league, "processed_created": False, "status": "missing_source_file", "rows": 0}
    frames = []
    total_removed = 0
    for file_path in files:
        frame, _, _, _ = read_raw_csv(file_path)
        frame = clean_column_names(frame)
        frame = parse_match_date(frame)
        missing = [column for column in MATCH_REQUIRED if column not in frame.columns]
        if missing:
            return {"league": league, "processed_created": False, "status": f"incompatible_missing_columns:{','.join(missing)}", "rows": 0}
        before = len(frame)
        frame = frame.dropna(subset=MATCH_REQUIRED).copy()
        total_removed += before - len(frame)
        season_start, season_end = infer_season(frame)
        frame["league"] = league
        frame["season_start_year"] = season_start
        frame["season_end_year"] = season_end
        frame["source_file"] = file_path.name
        frames.append(frame)
    if not frames:
        return {"league": league, "processed_created": False, "status": "no_valid_rows", "rows": 0}
    output = pd.concat(frames, ignore_index=True, sort=False)
    output = output.sort_values(["Date", "HomeTeam", "AwayTeam"]).drop_duplicates(["Date", "HomeTeam", "AwayTeam"], keep="first").reset_index(drop=True)
    out = processed_path(league)
    out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out, index=False)
    return {
        "league": league,
        "processed_created": True,
        "status": "processed_file_created",
        "rows": int(len(output)),
        "required_rows_removed": int(total_removed),
        "season_min": int(output["season_end_year"].min()),
        "season_max": int(output["season_end_year"].max()),
    }


def load_processed(league: str) -> pd.DataFrame | None:
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


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    margin = pd.to_numeric(out["FTHG"], errors="coerce") - pd.to_numeric(out["FTAG"], errors="coerce")
    line = pd.to_numeric(out["AHh"], errors="coerce")
    adjusted = margin + line
    out["target_ah_home_cover"] = np.where(adjusted > 0, 1, np.where(adjusted < 0, 0, np.nan))
    out["target_ah_push"] = adjusted == 0
    out["target_1x2"] = out["FTR"].map({"H": 0, "D": 1, "A": 2}) if "FTR" in out.columns else np.nan
    return out


def ah_market_probability(frame: pd.DataFrame) -> pd.Series:
    home = 1.0 / pd.to_numeric(frame["AvgAHH"], errors="coerce")
    away = 1.0 / pd.to_numeric(frame["AvgAHA"], errors="coerce")
    return home / (home + away)


def ece_binary(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (probabilities >= left) & (probabilities <= right if idx == bins - 1 else probabilities < right)
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(probabilities[mask].mean()))
    return total


def coverage_rows(league: str, frame: pd.DataFrame | None) -> list[dict[str, object]]:
    if frame is None:
        return []
    missing = [column for column in AH_REQUIRED if column not in frame.columns]
    rows = []
    for season, group in frame.groupby("season_end_year", dropna=True):
        target = add_targets(group) if not missing else group.copy()
        rows.append(
            {
                "league": league,
                "season_end_year": int(season),
                "matches": int(len(group)),
                "one_x_two_odds_coverage": float(group[[c for c in ["AvgH", "AvgD", "AvgA"] if c in group.columns]].notna().all(axis=1).mean()) if {"AvgH", "AvgD", "AvgA"}.issubset(group.columns) else 0.0,
                "ah_odds_coverage": float(group[[c for c in ["AHh", "AvgAHH", "AvgAHA"] if c in group.columns]].notna().all(axis=1).mean()) if {"AHh", "AvgAHH", "AvgAHA"}.issubset(group.columns) else 0.0,
                "usable_ah_target_rows": int(target.dropna(subset=AH_REQUIRED + ["target_ah_home_cover"]).shape[0]) if not missing else 0,
                "usable_1x2_target_rows": int(group.dropna(subset=ONE_X_TWO).shape[0]) if set(ONE_X_TWO).issubset(group.columns) else 0,
                "big_home_favourite_ah_rows": int((pd.to_numeric(group["AHh"], errors="coerce") <= -1.0).sum()) if "AHh" in group.columns else 0,
                "average_ah_home_line": float(pd.to_numeric(group["AHh"], errors="coerce").mean()) if "AHh" in group.columns else np.nan,
                "average_home_ah_odds": float(pd.to_numeric(group["AvgAHH"], errors="coerce").mean()) if "AvgAHH" in group.columns else np.nan,
                "average_away_ah_odds": float(pd.to_numeric(group["AvgAHA"], errors="coerce").mean()) if "AvgAHA" in group.columns else np.nan,
                "closing_ah_coverage_diagnostic_only": float(group[CLOSING_AH].notna().all(axis=1).mean()) if set(CLOSING_AH).issubset(group.columns) else 0.0,
                "missing_required_columns": ",".join(missing),
            }
        )
    return rows


def market_rows(league: str, frame: pd.DataFrame | None) -> list[dict[str, object]]:
    if frame is None or not set(AH_REQUIRED).issubset(frame.columns):
        return [{"league": league, "season_end_year": "all", "rows": 0, "status": "not_computable"}]
    target = add_targets(frame)
    target["market_p"] = ah_market_probability(target)
    ah_rows = target.dropna(subset=["target_ah_home_cover", "market_p", "AvgAHH", "AvgAHA"]).copy()
    rows = []
    for season_key, group in [("all", ah_rows)] + [(int(season), g) for season, g in ah_rows.groupby("season_end_year")]:
        if group.empty:
            continue
        y = group["target_ah_home_cover"].astype(int).to_numpy()
        p = np.clip(group["market_p"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
        full = target if season_key == "all" else target[target["season_end_year"].eq(season_key)]
        rows.append(
            {
                "league": league,
                "season_end_year": season_key,
                "rows": int(len(group)),
                "raw_ah_market_log_loss": float(log_loss(y, p, labels=[0, 1])),
                "raw_ah_brier": float(brier_score_loss(y, p)),
                "raw_ah_ece": ece_binary(y, p),
                "home_cover_rate": float((y == 1).mean()),
                "away_cover_rate": float((y == 0).mean()),
                "push_rate": float(full["target_ah_push"].mean()),
                "average_ah_odds": float(pd.concat([group["AvgAHH"], group["AvgAHA"]]).astype(float).mean()),
                "home_covers": int((y == 1).sum()),
                "away_covers": int((y == 0).sum()),
                "status": "computed",
            }
        )
    return rows


def tm_club_reference(league: str) -> pd.DataFrame:
    if not CLUBS_PATH.exists():
        return pd.DataFrame(columns=["name", "normalized_name", "domestic_competition_id"])
    clubs = pd.read_csv(CLUBS_PATH)
    comp = TM_COMPETITIONS.get(league)
    if comp:
        clubs = clubs[clubs["domestic_competition_id"].astype(str).eq(comp)].copy()
    clubs["normalized_name"] = clubs["name"].map(normalize)
    return clubs[["club_id", "name", "normalized_name", "domestic_competition_id"]].drop_duplicates()


def mapping_candidates_and_coverage(leagues: list[str], frames: dict[str, pd.DataFrame | None]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(MAPPING_PATH) if MAPPING_PATH.exists() else pd.DataFrame()
    if len(mapping):
        mapping["normalized_match_team"] = mapping["match_team"].map(normalize)
    proxy = pd.read_csv(PROXY_PATH, low_memory=False) if PROXY_PATH.exists() else pd.DataFrame()
    candidate_rows = []
    coverage_rows_out = []
    for league in leagues:
        frame = frames.get(league)
        teams = []
        if frame is not None:
            teams = sorted(set(frame["HomeTeam"].dropna().astype(str)) | set(frame["AwayTeam"].dropna().astype(str)))
        league_mapping = mapping[mapping["league"].astype(str).eq(league)].copy() if len(mapping) else pd.DataFrame()
        mapped = set(league_mapping["match_team"].dropna().astype(str)) if len(league_mapping) else set()
        clubs = tm_club_reference(league)
        for team in teams:
            norm = normalize(team)
            exact = clubs[clubs["normalized_name"].eq(norm)] if len(clubs) else pd.DataFrame()
            contains = clubs[clubs["normalized_name"].str.contains(norm, regex=False, na=False) | clubs["normalized_name"].map(lambda x: norm in str(x) or str(x) in norm)] if len(clubs) else pd.DataFrame()
            status = "already_mapped" if team in mapped else "unmatched"
            if team not in mapped and len(exact) == 1:
                status = "exact_normalized_candidate_unaccepted"
            elif team not in mapped and len(contains) == 1:
                status = "single_contains_candidate_unaccepted"
            elif team not in mapped and len(contains) > 1:
                status = "ambiguous_candidates_manual_review"
            candidate_rows.append(
                {
                    "league": league,
                    "match_team": team,
                    "normalized_match_team": norm,
                    "tm_competition_filter": TM_COMPETITIONS.get(league) or "",
                    "existing_mapping": team in mapped,
                    "candidate_count": int(len(contains.drop_duplicates("club_id"))) if len(contains) else int(len(exact)),
                    "candidate_clubs": "; ".join(contains["name"].drop_duplicates().head(8).tolist()) if len(contains) else "; ".join(exact["name"].drop_duplicates().head(8).tolist()),
                    "candidate_status": status,
                }
            )
        proxy_league = proxy[proxy["league"].astype(str).eq(league)].copy() if len(proxy) else pd.DataFrame()
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
        coverage_rows_out.append(
            {
                "league": league,
                "unique_match_teams": int(len(teams)),
                "teams_mapped_to_transfermarkt": int(len(mapped & set(teams))),
                "unmatched_teams": "; ".join([team for team in teams if team not in mapped]),
                "exact_or_single_candidates": int(sum(1 for row in candidate_rows if row["league"] == league and row["candidate_status"] in {"exact_normalized_candidate_unaccepted", "single_contains_candidate_unaccepted"})),
                "ambiguous_candidates": int(sum(1 for row in candidate_rows if row["league"] == league and row["candidate_status"] == "ambiguous_candidates_manual_review")),
                "valuation_coverage_365d": cov365,
                "valuation_coverage_180d": cov180,
                "median_players_per_club_365d": median_players,
                "tm_proxy_usable_for_predictive_diagnostics": bool(len(proxy_league) and cov365 >= 0.7 and pd.notna(median_players) and median_players >= 11 and len(mapped & set(teams)) == len(teams)),
                "mapping_needs_manual_review": bool(len(mapped & set(teams)) < len(teams) or not len(proxy_league)),
            }
        )
    return pd.DataFrame(candidate_rows), pd.DataFrame(coverage_rows_out)


def classify_league(league: str, frame: pd.DataFrame | None, coverage: pd.DataFrame, market: pd.DataFrame, tm_cov: pd.DataFrame, process_status: str) -> tuple[str, str]:
    if frame is None:
        return ("layer2_missing_source_file" if process_status == "missing_source_file" else "reject_for_now", process_status)
    missing = [column for column in AH_REQUIRED if column not in frame.columns]
    if missing:
        if set(ONE_X_TWO).issubset(frame.columns):
            return "layer2_ready_1x2_only", f"missing AH columns: {','.join(missing)}"
        return "reject_for_now", f"missing required columns: {','.join(missing)}"
    cov = coverage[coverage["league"].eq(league)]
    usable_rows = int(cov["usable_ah_target_rows"].sum()) if len(cov) else 0
    usable_seasons = int((cov["usable_ah_target_rows"] > 0).sum()) if len(cov) else 0
    market_ok = len(market[(market["league"].eq(league)) & (market["season_end_year"].astype(str).eq("all")) & market["status"].eq("computed")]) > 0
    tm_row = tm_cov[tm_cov["league"].eq(league)]
    tm_usable = bool(len(tm_row) and bool(tm_row["tm_proxy_usable_for_predictive_diagnostics"].iloc[0]))
    if usable_rows >= 500 and usable_seasons >= 3 and market_ok and tm_usable:
        return "layer2_ready_ah_with_tm", f"usable AH rows={usable_rows}; usable seasons={usable_seasons}; TM usable"
    if usable_rows >= 500 and usable_seasons >= 3 and market_ok:
        return "layer2_ready_ah_market_only", f"usable AH rows={usable_rows}; usable seasons={usable_seasons}; TM mapping needed"
    if usable_rows > 0:
        return "layer2_insufficient_ah_coverage", f"usable AH rows={usable_rows}; usable seasons={usable_seasons}"
    return "reject_for_now", "market baseline not computable"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def write_report(discovery: pd.DataFrame, process_results: pd.DataFrame, coverage: pd.DataFrame, market: pd.DataFrame, tm_cov: pd.DataFrame, inclusion: pd.DataFrame, final: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    market_all = market[market["season_end_year"].astype(str).eq("all")].copy()
    lines = [
        "# Layer 2 Processing Repair And Extended Availability Audit",
        "",
        f"Final classification: `{final}`",
        "",
        "Scope: local processing repair and report-only availability diagnostics. No predictive models, betting strategies, value searches, threshold optimization, scraping, live betting, lineups, or `players.current_club_*` fields were used.",
        "",
        "Raw files were preserved unchanged. Closing AH columns are diagnostic only. Transfermarkt mappings are candidate/report-only unless already present in the manual mapping.",
        "",
        "## Processing Results",
        "",
        markdown_table(process_results, ["league", "processed_created", "status", "rows", "season_min", "season_max", "required_rows_removed"], max_rows=20),
        "",
        "## Recommended Inclusion",
        "",
        markdown_table(inclusion, ["league", "classification", "reason"], max_rows=20),
        "",
        "## File Discovery",
        "",
        markdown_table(discovery, ["league", "storage_layer", "path", "rows", "season_min", "season_max", "required_match_columns_present", "ah_columns_present", "can_process_to_standard_matches", "status"], max_rows=80),
        "",
        "## Extended Coverage",
        "",
        markdown_table(coverage, ["league", "season_end_year", "matches", "one_x_two_odds_coverage", "ah_odds_coverage", "usable_ah_target_rows", "usable_1x2_target_rows", "big_home_favourite_ah_rows", "closing_ah_coverage_diagnostic_only"], max_rows=100),
        "",
        "## Raw AH Market Diagnostics",
        "",
        markdown_table(market_all, ["league", "rows", "raw_ah_market_log_loss", "raw_ah_brier", "raw_ah_ece", "home_cover_rate", "away_cover_rate", "push_rate", "average_ah_odds", "status"], max_rows=20),
        "",
        "## Transfermarkt Mapping Coverage",
        "",
        markdown_table(tm_cov, ["league", "unique_match_teams", "teams_mapped_to_transfermarkt", "exact_or_single_candidates", "ambiguous_candidates", "valuation_coverage_365d", "median_players_per_club_365d", "tm_proxy_usable_for_predictive_diagnostics", "mapping_needs_manual_review"], max_rows=20),
        "",
        "Do not run pooled modeling from this report until the desired market-only inclusion set is confirmed.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    discovery_before = discover_files()
    process_rows = [process_direct_raw(league) for league in CANDIDATES]
    process_results = pd.DataFrame(process_rows)
    discovery_after = discover_files()
    frames = {league: load_processed(league) for league in ALL_AUDIT}
    coverage = pd.DataFrame([row for league, frame in frames.items() for row in coverage_rows(league, frame)])
    market = pd.DataFrame([row for league, frame in frames.items() for row in market_rows(league, frame)])
    tm_candidates, tm_cov = mapping_candidates_and_coverage(ALL_AUDIT, frames)
    inclusion_rows = []
    for league in ALL_AUDIT:
        process_status = process_results[process_results["league"].eq(league)]["status"].iloc[0] if league in CANDIDATES else "processed_file_already_exists"
        classification, reason = classify_league(league, frames[league], coverage, market, tm_cov, process_status)
        if league in CANDIDATES and process_status == "processed_file_created" and classification == "layer2_ready_ah_market_only":
            reason = f"processed file created; {reason}"
        inclusion_rows.append({"league": league, "classification": classification, "reason": reason})
    inclusion = pd.DataFrame(inclusion_rows)
    market_ready = int(inclusion["classification"].isin(["layer2_ready_ah_market_only", "layer2_ready_ah_with_tm"]).sum())
    tm_ready = int(inclusion["classification"].eq("layer2_ready_ah_with_tm").sum())
    if tm_ready >= 2:
        final = "layer2_ready_with_transfermarkt"
    elif market_ready >= 2:
        final = "layer2_ready_market_only"
    elif market_ready >= 1:
        final = "layer2_partial_ready_market_only"
    else:
        final = "layer2_not_ready"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    discovery_after.to_csv(DISCOVERY_PATH, index=False)
    coverage.to_csv(COVERAGE_PATH, index=False)
    market.to_csv(MARKET_PATH, index=False)
    tm_candidates.to_csv(TM_CANDIDATES_PATH, index=False)
    tm_cov.to_csv(TM_COVERAGE_PATH, index=False)
    inclusion.to_csv(INCLUSION_PATH, index=False)
    write_report(discovery_after, process_results, coverage, market, tm_cov, inclusion, final)
    print(
        {
            "processed_created": int(process_results["processed_created"].fillna(False).sum()),
            "discovery_rows": len(discovery_after),
            "coverage_rows": len(coverage),
            "market_rows": len(market),
            "tm_candidate_rows": len(tm_candidates),
            "tm_coverage_rows": len(tm_cov),
            "classification": final,
        }
    )


if __name__ == "__main__":
    main()
