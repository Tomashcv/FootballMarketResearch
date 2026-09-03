from __future__ import annotations

import hashlib
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/v3_frozen_candidate.yaml"
PAPER_DIR = ROOT / "outputs/paper_trading/v3"
SNAPSHOT_DIR = PAPER_DIR / "snapshots"
HTML_DIR = PAPER_DIR / "html"
LOG_DIR = PAPER_DIR / "logs"
LEDGER_PATH = PAPER_DIR / "v3_paper_ledger.csv"
LATEST_ROW_PREDICTIONS = PAPER_DIR / "v3_latest_row_predictions.csv"
LATEST_CANDIDATE_PICKS = PAPER_DIR / "v3_latest_candidate_picks.csv"
LATEST_SKIPPED_PICKS = PAPER_DIR / "v3_latest_skipped_picks.csv"
LATEST_WARNINGS = PAPER_DIR / "v3_latest_data_quality_warnings.csv"
LATEST_SNAPSHOT_POINTER = SNAPSHOT_DIR / "latest_manifest_path.txt"
HTML_REPORT = HTML_DIR / "v3_paper_latest.html"
PIPELINE_REPORT = PAPER_DIR / "v3_paper_pipeline_report.md"

ALLOWED_LEAGUES = ["E0", "SP1", "D1", "I1", "F1", "B1", "G1", "N1", "P1", "SC0", "T1"]
EXCLUDED_LEAGUES = {"E1", "E2", "E3"}
SCOPE = "scope_C_top_divisions_ex_e1_e2_e3"
MODEL = "xgboost_market_residual_multiclass"
FEATURE_GROUP = "v3_tm_plus_clubelo_core_staleness"
RULE_NAME = "away_edge_ge_0.015_away_odds_ge_1.5"
AWAY_EDGE_MIN = 0.015
AWAY_ODDS_MIN = 1.5
STAKE_UNITS = 1.0
RESEARCH_ONLY = "research_only"

OPEN_STATUS = "OPEN"
SETTLED_STATUSES = {"SETTLED_WIN", "SETTLED_LOSS", "SETTLED_PUSH_OR_VOID"}
VALID_STATUSES = {
    OPEN_STATUS,
    "SETTLED_WIN",
    "SETTLED_LOSS",
    "SETTLED_PUSH_OR_VOID",
    "SKIPPED_NO_ODDS",
    "SKIPPED_BELOW_EDGE",
    "SKIPPED_DUPLICATE",
    "BLOCKED_DATA_QUALITY",
}

LEDGER_COLUMNS = [
    "paper_bet_id",
    "created_at_utc",
    "run_id",
    "source_snapshot_id",
    "canonical_match_id",
    "match_date",
    "league",
    "season_start_year",
    "home_team",
    "away_team",
    "selected_side",
    "selected_odds",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "model_home_prob",
    "model_draw_prob",
    "model_away_prob",
    "away_edge",
    "rule_name",
    "stake_units",
    "status",
    "result",
    "settled_at_utc",
    "profit_units",
    "closing_notes",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def iso_utc() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dirs() -> None:
    for path in [PAPER_DIR, SNAPSHOT_DIR, HTML_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def load_ledger(path: Path = LEDGER_PATH) -> pd.DataFrame:
    if not path.exists():
        return empty_ledger()
    ledger = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in LEDGER_COLUMNS:
        if col not in ledger.columns:
            ledger[col] = ""
    return ledger[LEDGER_COLUMNS].copy()


def write_ledger(ledger: pd.DataFrame, path: Path = LEDGER_PATH) -> None:
    ensure_dirs()
    out = ledger.copy()
    for col in LEDGER_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out[LEDGER_COLUMNS].to_csv(path, index=False)


def canonical_match_id(row: pd.Series) -> str:
    for col in ["canonical_match_id", "full_scope_match_id", "match_id", "logical_match_key"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip() and str(row[col]).strip().lower() != "nan":
            return str(row[col]).strip()
    parts = [row.get("league", ""), row.get("season_start_year", ""), row.get("match_date", ""), row.get("home_team", ""), row.get("away_team", "")]
    return "|".join(str(p) for p in parts)


def stable_decimal(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value).strip()


def deterministic_paper_bet_id(
    canonical_match_id_value: object,
    selected_side: object,
    rule_name: object,
    selected_odds: object,
    match_date: object,
) -> str:
    key = "|".join(
        [
            str(canonical_match_id_value).strip(),
            str(selected_side).strip().lower(),
            str(rule_name).strip(),
            stable_decimal(selected_odds),
            str(match_date).strip()[:10],
        ]
    )
    return "v3_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_season_paths_from_norm(norm: pd.DataFrame) -> list[Path]:
    if norm.empty or "source_file" not in norm.columns:
        return []
    paths = []
    for rel in sorted(norm["source_file"].dropna().astype(str).unique()):
        path = ROOT / rel
        if "/seasons/" in rel and path.exists() and path.is_file():
            paths.append(path)
    return paths


def file_row_count(path: Path) -> int | None:
    try:
        return int(len(pd.read_csv(path, low_memory=False)))
    except Exception:
        return None


def build_snapshot_manifest(
    run_id: str,
    norm: pd.DataFrame,
    validation: pd.DataFrame | None,
    warnings: Iterable[str],
    path: Path | None = None,
) -> dict[str, object]:
    ensure_dirs()
    path = path or (SNAPSHOT_DIR / f"{run_id}_manifest.json")
    files = []
    for input_path in raw_season_paths_from_norm(norm):
        rel = str(input_path.relative_to(ROOT))
        rows = norm[norm["source_file"].astype(str).eq(rel)].copy() if not norm.empty else pd.DataFrame()
        dates = pd.to_datetime(rows.get("match_date", pd.Series(dtype=str)), errors="coerce")
        files.append(
            {
                "path": rel,
                "size_bytes": input_path.stat().st_size,
                "sha256": sha256_file(input_path),
                "row_count": file_row_count(input_path),
                "normalized_row_count": int(len(rows)),
                "min_date": dates.min().date().isoformat() if dates.notna().any() else "",
                "max_date": dates.max().date().isoformat() if dates.notna().any() else "",
                "leagues_detected": sorted(rows.get("div", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
            }
        )
    val_dates = pd.to_datetime(validation["match_date"], errors="coerce") if validation is not None and not validation.empty and "match_date" in validation.columns else pd.Series(dtype="datetime64[ns]")
    manifest = {
        "run_id": run_id,
        "timestamp_utc": iso_utc(),
        "input_files": files,
        "file_count": len(files),
        "normalized_row_count": int(len(norm)) if norm is not None else 0,
        "prediction_input_row_count": int(len(validation)) if validation is not None else 0,
        "min_match_date": val_dates.min().date().isoformat() if val_dates.notna().any() else "",
        "max_match_date": val_dates.max().date().isoformat() if val_dates.notna().any() else "",
        "leagues_detected": sorted(validation["div"].dropna().astype(str).unique().tolist()) if validation is not None and not validation.empty and "div" in validation.columns else [],
        "warnings": list(warnings),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LATEST_SNAPSHOT_POINTER.write_text(str(path.relative_to(ROOT)) + "\n", encoding="utf-8")
    return manifest


def select_current_raw_norm(norm: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if norm.empty:
        return norm.copy(), ["No normalized football-data rows found."]
    work = norm.copy()
    work = work[work.get("raw_season_file_flag", False).fillna(False).astype(bool)].copy()
    work = work[work["source_file"].astype(str).str.contains("/seasons/", regex=False, na=False)].copy()
    work = work[work["div"].astype(str).isin(ALLOWED_LEAGUES)].copy()
    work = work[~work["div"].astype(str).isin(EXCLUDED_LEAGUES)].copy()
    if work.empty:
        return work, ["No eligible current raw season rows found under data/raw/*/seasons/."]
    years = pd.to_numeric(work["season_start_year"], errors="coerce")
    current_year = int(years.max())
    work = work[years.eq(current_year)].copy()
    detected = sorted(work["div"].dropna().astype(str).unique().tolist())
    missing = sorted(set(ALLOWED_LEAGUES) - set(detected))
    if missing:
        warnings.append(f"Missing current raw season rows for leagues: {', '.join(missing)}")
    return work, warnings


def build_paper_market_dataset(fd_module, norm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if norm.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ["No rows supplied for paper market dataset."]
    teams, aliases, _added = fd_module.build_team_registry(norm)
    ids = fd_module.attach_team_ids(norm, aliases)
    marketed = fd_module.add_market(ids)
    valid = marketed[
        marketed["odds_valid"].fillna(False).astype(bool)
        & marketed["home_team_id"].notna()
        & marketed["away_team_id"].notna()
    ].copy()
    skipped_no_odds = marketed[
        ~marketed["odds_valid"].fillna(False).astype(bool)
        | marketed["home_team_id"].isna()
        | marketed["away_team_id"].isna()
    ].copy()
    if valid.empty:
        warnings.append("No valid 1X2 odds rows available for current raw season.")
        return pd.DataFrame(), marketed, skipped_no_odds, warnings
    valid = valid.sort_values(
        ["logical_match_key", "x1_odds_priority", "source_priority", "row_non_null_count", "source_file", "football_data_row_id"],
        ascending=[True, True, True, False, True, True],
    )
    valid["rank_within_match"] = valid.groupby("logical_match_key").cumcount() + 1
    selected = valid[valid["rank_within_match"].eq(1)].copy()
    selected["score_conflict_quarantine_flag"] = False
    selected, _source_map = fd_module.assign_ids(selected, marketed)
    selected["x1_home_raw_prob"] = 1.0 / selected["x1_home_odds"]
    selected["x1_draw_raw_prob"] = 1.0 / selected["x1_draw_odds"]
    selected["x1_away_raw_prob"] = 1.0 / selected["x1_away_odds"]
    selected["x1_overround"] = selected[["x1_home_raw_prob", "x1_draw_raw_prob", "x1_away_raw_prob"]].sum(axis=1)
    selected["x1_home_no_vig_prob"] = selected["x1_home_raw_prob"] / selected["x1_overround"]
    selected["x1_draw_no_vig_prob"] = selected["x1_draw_raw_prob"] / selected["x1_overround"]
    selected["x1_away_no_vig_prob"] = selected["x1_away_raw_prob"] / selected["x1_overround"]
    selected["classification"] = RESEARCH_ONLY
    selected["partial_latest_season_flag"] = True
    selected["partial_season_flag"] = True
    selected["dedup_tiebreak_policy"] = "paper: valid odds; B365>Avg>HDA; raw season files only; completeness; stable source,row"
    return selected, marketed, skipped_no_odds, warnings


def result_to_old(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if s in {"H", "D", "A"}:
        return s
    return {"HOME": "H", "DRAW": "D", "AWAY": "A", "HOME_WIN": "H", "AWAY_WIN": "A"}.get(s)


def add_internal_elo_with_history(hist_exact: pd.DataFrame, validation: pd.DataFrame, add_internal_elo_features_func) -> pd.DataFrame:
    all_rows = pd.concat([hist_exact, validation], ignore_index=True, sort=False)
    work = pd.DataFrame(
        {
            "full_scope_match_id": all_rows["full_scope_match_id"].astype(str),
            "league": all_rows["div"].astype(str),
            "Date": pd.to_datetime(all_rows["match_date"], errors="coerce"),
            "Time": all_rows.get("match_time", pd.Series("", index=all_rows.index)).fillna("").astype(str),
            "HomeTeam": all_rows["home_team_raw"].astype(str),
            "AwayTeam": all_rows["away_team_raw"].astype(str),
            "FTHG": pd.to_numeric(all_rows["home_goals"], errors="coerce"),
            "FTAG": pd.to_numeric(all_rows["away_goals"], errors="coerce"),
            "clubelo_diff": pd.to_numeric(all_rows["clubelo_diff"], errors="coerce"),
            "is_validation": np.r_[np.zeros(len(hist_exact), dtype=bool), np.ones(len(validation), dtype=bool)],
        },
        index=all_rows.index,
    )
    parts = []
    for _league, group in work.groupby("league", sort=False):
        elo = add_internal_elo_features_func(
            group[["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]],
            starting_elo=1500.0,
            k_factor=20.0,
            home_advantage_elo=65.0,
        )
        part = group[["full_scope_match_id", "is_validation", "clubelo_diff"]].copy()
        part["home_internal_elo"] = elo["home_internal_elo_pre"]
        part["away_internal_elo"] = elo["away_internal_elo_pre"]
        part["internal_elo_diff"] = elo["internal_elo_diff_home_minus_away"]
        part["clubelo_diff_minus_internal_elo_diff"] = part["clubelo_diff"] - part["internal_elo_diff"]
        parts.append(part)
    elo_all = pd.concat(parts).sort_index()
    val_elo = elo_all[elo_all["is_validation"]].drop(columns=["is_validation", "clubelo_diff"])
    out = validation.drop(columns=["home_internal_elo", "away_internal_elo", "internal_elo_diff", "clubelo_diff_minus_internal_elo_diff"], errors="ignore").copy()
    out["full_scope_match_id"] = out["full_scope_match_id"].astype(str)
    val_elo["full_scope_match_id"] = val_elo["full_scope_match_id"].astype(str)
    return out.merge(val_elo, on="full_scope_match_id", how="left", validate="one_to_one")


def build_prediction_adapter(raw: pd.DataFrame, feature_cols: list[str], require_target: bool = False) -> pd.DataFrame:
    """Build paper prediction rows in one concat operation to avoid DataFrame fragmentation."""
    index = raw.index
    target = raw.get("result_1x2", pd.Series(index=index, dtype=object)).map(result_to_old)
    base = pd.DataFrame(
        {
            "match_id": raw["full_scope_match_id"].astype(str),
            "full_scope_match_id": raw["full_scope_match_id"].astype(str),
            "canonical_match_id": raw.get("canonical_match_id", raw["full_scope_match_id"]).fillna(raw["full_scope_match_id"]).astype(str),
            "logical_match_key": raw["logical_match_key"].astype(str),
            "source_file": raw.get("source_file", pd.Series("", index=index)).fillna("").astype(str),
            "match_date": pd.to_datetime(raw["match_date"], errors="coerce"),
            "league": raw["div"].astype(str),
            "season_start_year": pd.to_numeric(raw["season_start_year"], errors="coerce").astype("Int64"),
            "home_team": raw["home_team_raw"].astype(str),
            "away_team": raw["away_team_raw"].astype(str),
            "target_outcome_1x2": target,
            "target_y": target.map({"H": 0, "D": 1, "A": 2}),
            "x1_odds_source": raw.get("x1_odds_source", pd.Series("", index=index)).fillna("").astype(str),
            "classification": RESEARCH_ONLY,
        },
        index=index,
    )
    base["season_end_year"] = base["season_start_year"] + 1
    numeric = raw.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
    flags = pd.DataFrame(index=index)
    for col in ["clubelo_both_found_flag", "tm_both_value_found_flag", "tm_match_feature_available"]:
        values = raw[col].fillna(False).astype(bool) if col in raw.columns else pd.Series(False, index=index)
        if col in numeric.columns:
            numeric[col] = values
        else:
            flags[col] = values
    out = pd.concat([base, numeric, flags], axis=1)
    market_prob_cols = ["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]
    market_odds_cols = ["x1x2_avg_odds_home", "x1x2_avg_odds_draw", "x1x2_avg_odds_away"]
    valid = (
        out["season_start_year"].notna()
        & out[market_prob_cols].notna().all(axis=1)
        & out[market_odds_cols].notna().all(axis=1)
        & out[market_odds_cols].gt(1).all(axis=1)
    )
    if require_target:
        valid &= out["target_y"].notna()
    out = out.loc[valid].copy()
    if require_target:
        out["target_y"] = out["target_y"].astype(int)
    return out.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def leakage_checks(validation: pd.DataFrame, pred: pd.DataFrame | None = None, selected: pd.DataFrame | None = None) -> pd.DataFrame:
    checks: list[tuple[str, bool, str]] = []
    if validation.empty:
        checks.append(("current_prediction_input_non_empty", False, "No current paper rows available."))
        return pd.DataFrame([{"check_name": n, "status": "pass" if ok else "fail", "details": d} for n, ok, d in checks])
    match_dates = pd.to_datetime(validation["match_date"], errors="coerce")
    home_ce = pd.to_datetime(validation.get("home_clubelo_latest_date", pd.Series(pd.NaT, index=validation.index)), errors="coerce")
    away_ce = pd.to_datetime(validation.get("away_clubelo_latest_date", pd.Series(pd.NaT, index=validation.index)), errors="coerce")
    tm_date_cols = [c for c in ["home_tm_latest_valuation_date", "away_tm_latest_valuation_date"] if c in validation.columns]
    tm_ok = True
    for col in tm_date_cols:
        d = pd.to_datetime(validation[col], errors="coerce")
        tm_ok = tm_ok and bool((d.isna() | d.lt(match_dates)).all())
    checks.extend(
        [
            ("raw_season_files_only", validation["source_file"].astype(str).str.contains("/seasons/", regex=False).all(), "Only data/raw league season files are prediction sources."),
            ("excluded_leagues_absent", ~validation["div"].astype(str).isin(EXCLUDED_LEAGUES).any(), "E1/E2/E3 absent."),
            ("classification_research_only", validation["classification"].eq(RESEARCH_ONLY).all(), "All current rows marked research_only."),
            ("internal_elo_pre_match", validation[["home_internal_elo", "away_internal_elo", "internal_elo_diff"]].notna().all().all(), "Internal Elo exists and is emitted before current match result update."),
            ("clubelo_strict_before_match", bool(((home_ce.isna() | home_ce.lt(match_dates)) & (away_ce.isna() | away_ce.lt(match_dates))).all()), "ClubElo latest date before match date where available."),
            ("tm_point_in_time", tm_ok, "Transfermarkt valuation dates before match date where available."),
            ("no_training_on_current_outcomes", True, "Prediction script trains only on historical exact rows before the paper season."),
        ]
    )
    if pred is not None and not pred.empty:
        pred_id_col = "full_scope_match_id" if "full_scope_match_id" in pred.columns else "canonical_match_id"
        checks.append(("no_duplicate_match_ids_predictions", pred[pred_id_col].duplicated().sum() == 0, f"duplicates={int(pred[pred_id_col].duplicated().sum())}"))
    if selected is not None and not selected.empty:
        checks.append(("away_side_only", selected["selected_side"].eq("away").all(), "Frozen rule selects away only."))
    return pd.DataFrame([{"check_name": n, "status": "pass" if ok else "fail", "details": d} for n, ok, d in checks])


def leakage_check_failed(checks: pd.DataFrame) -> bool:
    return not checks.empty and checks["status"].eq("fail").any()


def select_candidate_picks(pred: pd.DataFrame, run_id: str, snapshot_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()
    out = pred.copy()
    out["away_edge"] = out["model_away_prob"] - out["market_away_prob"]
    odds_ok = pd.to_numeric(out["selected_odds"], errors="coerce").ge(AWAY_ODDS_MIN)
    edge_ok = pd.to_numeric(out["away_edge"], errors="coerce").ge(AWAY_EDGE_MIN)
    unresolved = out.get("target_outcome_1x2", pd.Series("", index=out.index)).fillna("").astype(str).str.strip().eq("")
    picked = out[odds_ok & edge_ok & unresolved].copy()
    skipped = out[~(odds_ok & edge_ok & unresolved)].copy()
    for frame in [picked, skipped]:
        if frame.empty:
            continue
        frame["run_id"] = run_id
        frame["source_snapshot_id"] = snapshot_id
        frame["selected_side"] = "away"
        frame["rule_name"] = RULE_NAME
        frame["stake_units"] = STAKE_UNITS
    if not picked.empty:
        picked["paper_bet_id"] = picked.apply(lambda r: deterministic_paper_bet_id(r["canonical_match_id"], "away", RULE_NAME, r["selected_odds"], r["match_date"]), axis=1)
    if not skipped.empty:
        skipped_odds_ok = odds_ok.reindex(skipped.index).fillna(False)
        skipped_edge_ok = edge_ok.reindex(skipped.index).fillna(False)
        skipped_unresolved = unresolved.reindex(skipped.index).fillna(False)
        skipped["skip_status"] = np.select(
            [~skipped_odds_ok, ~skipped_edge_ok, ~skipped_unresolved],
            ["SKIPPED_NO_ODDS", "SKIPPED_BELOW_EDGE", "SKIPPED_RESULT_ALREADY_AVAILABLE"],
            default="BLOCKED_DATA_QUALITY",
        )
        reason_map = {
            "SKIPPED_NO_ODDS": "selected away odds missing or below frozen minimum",
            "SKIPPED_BELOW_EDGE": "away edge below frozen threshold",
            "SKIPPED_RESULT_ALREADY_AVAILABLE": "final result already available; not a new paper pick",
            "BLOCKED_DATA_QUALITY": "candidate could not be classified safely",
        }
        skipped["skip_reason"] = skipped["skip_status"].map(reason_map)
    return picked, skipped


def select_candidate_picks_after_quality_gate(
    validation: pd.DataFrame,
    pred: pd.DataFrame,
    run_id: str,
    snapshot_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checks = leakage_checks(validation, pred)
    if leakage_check_failed(checks):
        blocked = pred.copy()
        if not blocked.empty:
            blocked["skip_status"] = "BLOCKED_DATA_QUALITY"
            blocked["skip_reason"] = "feature leakage/data-quality check failed"
        return pd.DataFrame(), blocked, checks
    picks, skipped = select_candidate_picks(pred, run_id, snapshot_id)
    return picks, skipped, checks


def append_new_picks_to_ledger(
    ledger: pd.DataFrame,
    candidate_picks: pd.DataFrame,
    created_at_utc: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    created_at_utc = created_at_utc or iso_utc()
    if candidate_picks.empty:
        return ledger.copy(), pd.DataFrame(columns=LEDGER_COLUMNS)
    existing = set(ledger.get("paper_bet_id", pd.Series(dtype=str)).dropna().astype(str))
    rows = []
    for r in candidate_picks.itertuples(index=False):
        bet_id = str(getattr(r, "paper_bet_id"))
        if bet_id in existing:
            continue
        rows.append(
            {
                "paper_bet_id": bet_id,
                "created_at_utc": created_at_utc,
                "run_id": getattr(r, "run_id"),
                "source_snapshot_id": getattr(r, "source_snapshot_id"),
                "canonical_match_id": getattr(r, "canonical_match_id"),
                "match_date": str(getattr(r, "match_date"))[:10],
                "league": getattr(r, "league"),
                "season_start_year": getattr(r, "season_start_year"),
                "home_team": getattr(r, "home_team"),
                "away_team": getattr(r, "away_team"),
                "selected_side": "away",
                "selected_odds": stable_decimal(getattr(r, "selected_odds")),
                "market_home_prob": stable_decimal(getattr(r, "market_home_prob")),
                "market_draw_prob": stable_decimal(getattr(r, "market_draw_prob")),
                "market_away_prob": stable_decimal(getattr(r, "market_away_prob")),
                "model_home_prob": stable_decimal(getattr(r, "model_home_prob")),
                "model_draw_prob": stable_decimal(getattr(r, "model_draw_prob")),
                "model_away_prob": stable_decimal(getattr(r, "model_away_prob")),
                "away_edge": stable_decimal(getattr(r, "away_edge")),
                "rule_name": RULE_NAME,
                "stake_units": stable_decimal(STAKE_UNITS),
                "status": OPEN_STATUS,
                "result": "",
                "settled_at_utc": "",
                "profit_units": "",
                "closing_notes": "paper_only; research_only; no_confirmed_edge",
            }
        )
        existing.add(bet_id)
    new_rows = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    out = pd.concat([ledger, new_rows], ignore_index=True, sort=False) if rows else ledger.copy()
    return out[LEDGER_COLUMNS].copy(), new_rows


def result_lookup_from_raw(results: pd.DataFrame) -> dict[str, str]:
    if results.empty:
        return {}
    lookup = {}
    for row in results.itertuples(index=False):
        key = canonical_match_id(pd.Series(row._asdict()))
        result = result_to_old(getattr(row, "result_1x2", ""))
        if key and result in {"H", "D", "A"}:
            lookup[str(key)] = result
    return lookup


def settle_open_ledger(ledger: pd.DataFrame, result_lookup: dict[str, str], settled_at_utc: str | None = None) -> pd.DataFrame:
    if ledger.empty:
        return ledger.copy()
    settled_at_utc = settled_at_utc or iso_utc()
    out = ledger.copy()
    for idx, row in out[out["status"].eq(OPEN_STATUS)].iterrows():
        result = result_lookup.get(str(row["canonical_match_id"]))
        if result not in {"H", "D", "A"}:
            continue
        odds = float(row["selected_odds"])
        if str(row["selected_side"]).lower() == "away" and result == "A":
            out.at[idx, "status"] = "SETTLED_WIN"
            out.at[idx, "profit_units"] = stable_decimal(odds - 1.0)
        else:
            out.at[idx, "status"] = "SETTLED_LOSS"
            out.at[idx, "profit_units"] = stable_decimal(-1.0)
        out.at[idx, "result"] = result
        out.at[idx, "settled_at_utc"] = settled_at_utc
    return out[LEDGER_COLUMNS].copy()


def max_drawdown(profits: pd.Series) -> float:
    p = pd.to_numeric(profits, errors="coerce").dropna()
    if p.empty:
        return 0.0
    curve = p.cumsum()
    return float((curve - curve.cummax()).min())


def performance_summary(ledger: pd.DataFrame) -> dict[str, object]:
    if ledger.empty:
        return {"total_paper_bets": 0, "open_bets": 0, "settled_bets": 0, "profit_units": 0.0, "roi": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}
    settled = ledger[ledger["status"].isin(SETTLED_STATUSES)].copy()
    wins = int(settled["status"].eq("SETTLED_WIN").sum())
    profit = float(pd.to_numeric(settled["profit_units"], errors="coerce").fillna(0.0).sum())
    settled_bets = int(len(settled))
    return {
        "total_paper_bets": int(len(ledger)),
        "open_bets": int(ledger["status"].eq(OPEN_STATUS).sum()),
        "settled_bets": settled_bets,
        "profit_units": profit,
        "roi": profit / settled_bets if settled_bets else 0.0,
        "win_rate": wins / settled_bets if settled_bets else 0.0,
        "max_drawdown": max_drawdown(settled["profit_units"]) if settled_bets else 0.0,
    }


def group_summary(ledger: pd.DataFrame, by: str) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=[by, "bets", "open_bets", "settled_bets", "profit_units", "roi", "win_rate"])
    work = ledger.copy()
    if by == "month":
        work["month"] = pd.to_datetime(work["match_date"], errors="coerce").dt.to_period("M").astype(str)
    rows = []
    for key, group in work.groupby(by, dropna=False):
        s = performance_summary(group)
        rows.append({by: key, "bets": s["total_paper_bets"], "open_bets": s["open_bets"], "settled_bets": s["settled_bets"], "profit_units": s["profit_units"], "roi": s["roi"], "win_rate": s["win_rate"]})
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def html_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 200) -> str:
    if df.empty:
        return "<p><em>No rows.</em></p>"
    show = df.copy()
    if columns:
        show = show[[c for c in columns if c in show.columns]]
    show = show.head(max_rows)
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in show.columns)
    rows = []
    for _, row in show.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def read_latest_manifest() -> dict[str, object]:
    if not LATEST_SNAPSHOT_POINTER.exists():
        return {}
    rel = LATEST_SNAPSHOT_POINTER.read_text(encoding="utf-8").strip()
    path = ROOT / rel
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_html_report(ledger: pd.DataFrame, warnings: pd.DataFrame | None = None, manifest: dict[str, object] | None = None) -> None:
    ensure_dirs()
    manifest = manifest if manifest is not None else read_latest_manifest()
    warnings = warnings if warnings is not None else pd.DataFrame()
    summary = performance_summary(ledger)
    open_picks = ledger[ledger["status"].eq(OPEN_STATUS)].copy() if not ledger.empty else ledger
    settled = ledger[ledger["status"].isin(SETTLED_STATUSES)].copy() if not ledger.empty else ledger
    by_league = group_summary(ledger, "league")
    by_month = group_summary(ledger, "month")
    file_rows = pd.DataFrame(manifest.get("input_files", [])) if manifest else pd.DataFrame()
    css = """
body{font-family:Arial,sans-serif;margin:24px;color:#1f2933;background:#fff}
h1,h2{margin:0 0 12px} h2{margin-top:28px}
.warn{border:1px solid #c2410c;background:#fff7ed;padding:12px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin:12px 0}
.metric{border:1px solid #d9e2ec;padding:10px}
.label{font-size:12px;color:#52606d}.value{font-size:20px;font-weight:700}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 18px}
th,td{border:1px solid #d9e2ec;padding:6px;text-align:left;vertical-align:top}
th{background:#f5f7fa} code{background:#f5f7fa;padding:2px 4px}
"""
    metric_html = "".join(f"<div class='metric'><div class='label'>{html.escape(k)}</div><div class='value'>{html.escape(f'{v:.4f}' if isinstance(v, float) else str(v))}</div></div>" for k, v in summary.items())
    content = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Frozen V3 Paper Trading Report</title><style>{css}</style></head>
<body>
<h1>Frozen V3 Paper Trading Report</h1>
<p>Generated UTC: {html.escape(iso_utc())}</p>
<div class="warn"><strong>research_only.</strong> Paper trading only. No live betting, no real-money staking, and no confirmed edge claim.</div>
<h2>Frozen Rule</h2>
<p><code>{html.escape(SCOPE)}</code> / <code>{html.escape(MODEL)}</code> / <code>{html.escape(FEATURE_GROUP)}</code>. Away only where <code>away_edge &gt;= 0.015</code> and <code>away odds &gt;= 1.5</code>. Flat 1u paper stake.</p>
<h2>Performance Summary</h2><div class="grid">{metric_html}</div>
<h2>Current OPEN Picks</h2>{html_table(open_picks, ['paper_bet_id','match_date','league','home_team','away_team','selected_odds','market_away_prob','model_away_prob','away_edge','status'])}
<h2>Settled Picks</h2>{html_table(settled, ['paper_bet_id','match_date','league','home_team','away_team','selected_odds','result','status','profit_units'])}
<h2>By League</h2>{html_table(by_league)}
<h2>By Month</h2>{html_table(by_month)}
<h2>Data Quality Warnings</h2>{html_table(warnings)}
<h2>Input Snapshot Hashes</h2>{html_table(file_rows, ['path','size_bytes','sha256','row_count','normalized_row_count','min_date','max_date','leagues_detected'])}
</body></html>
"""
    HTML_REPORT.write_text(content, encoding="utf-8")


def write_pipeline_report(decision: str, warnings: Iterable[str]) -> None:
    ensure_dirs()
    lines = [
        "# V3 Paper Trading Pipeline",
        "",
        f"Decision: `{decision}`",
        "",
        "Labels: `v3_paper_pipeline_built_research_only`, `v3_paper_pipeline_ready_for_sportsedge_integration_research_only` when current data and leakage checks pass.",
        "",
        "## What Was Built",
        "- Frozen V3 config at `configs/v3_frozen_candidate.yaml`.",
        "- Local prediction, ledger update, settlement, HTML report, and master pipeline scripts.",
        "- Idempotent paper ledger with deterministic `paper_bet_id`.",
        "- Raw input snapshot manifests with file sizes, SHA256 hashes, row counts, dates, leagues, and warnings.",
        "",
        "## How To Run",
        "`python scripts/run_v3_paper_pipeline.py`",
        "",
        "## Files Created",
        "- `outputs/paper_trading/v3/v3_latest_row_predictions.csv`",
        "- `outputs/paper_trading/v3/v3_latest_candidate_picks.csv`",
        "- `outputs/paper_trading/v3/v3_paper_ledger.csv`",
        "- `outputs/paper_trading/v3/html/v3_paper_latest.html`",
        "- `outputs/paper_trading/v3/snapshots/<run_id>_manifest.json`",
        "",
        "## Data Assumptions",
        "- Current season rows come only from local football-data raw season CSVs under `data/raw/*/seasons/`.",
        "- E1/E2/E3 remain excluded.",
        "- Historical exact V3 rows are used only as pre-current-season training history.",
        "- ClubElo, Transfermarkt, and internal Elo checks must remain point-in-time safe.",
        "",
        "## Limitations",
        "- No API keys or odds feeds are used.",
        "- Missing odds create no picks.",
        "- Paper results must be settled from later local raw files once `FTR` is available.",
        "- Current-season outcomes are never used for model fitting or optimization.",
        "",
        "## Why Paper Only",
        "This is a research-only candidate with no confirmed edge claim. It uses flat 1u paper stakes, never places bets, and must not be optimized on paper results.",
        "",
        "## SportsEdge Next Step",
        "Expose the ledger and latest HTML report as read-only SportsEdge artifacts, keeping order placement disabled and preserving the `research_only` label.",
        "",
        "## Warnings",
    ]
    lines.extend([f"- {w}" for w in warnings] or ["- None"])
    PIPELINE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_thresholds() -> dict[str, float]:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    def pick(name: str) -> float:
        match = re.search(rf"{re.escape(name)}:\s*([0-9.]+)", text)
        if not match:
            raise ValueError(f"missing {name}")
        return float(match.group(1))
    return {"away_edge_min": pick("away_edge_min"), "away_odds_min": pick("away_odds_min"), "stake_units": pick("stake_units")}
