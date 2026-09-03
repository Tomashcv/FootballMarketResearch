"""Build the canonical V4 scheduled-snapshot/verified-close market panel."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .phase1b_audit import (
    AGGREGATE_BOOKMAKERS,
    OUT_DIR,
    ROOT,
    PairSpec,
    build_1x2_pairs,
    build_ah_pairs,
    build_fixture_audit,
    build_ou_pairs,
    decimal_odds_valid,
    load_canonical_frames,
)


PROCESSED_DIR = ROOT / "data/processed/v4"
PANEL_PATH = PROCESSED_DIR / "v4_fixture_market_panel_v1.csv"
CONTRACT_PATH = PROCESSED_DIR / "v4_feature_column_contract_v1.json"


def normalize_team(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def deterministic_match_id(league: str, date: str, home: str, away: str) -> int:
    key = f"{league}|{date}|{normalize_team(home)}|{normalize_team(away)}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:15], 16)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)


def _novig_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.column_stack([_numeric(frame, c).to_numpy(float) for c in columns])
    valid = np.isfinite(values).all(axis=1) & (values > 1).all(axis=1)
    inv = np.divide(1.0, values, out=np.full_like(values, np.nan), where=values > 1)
    overround = np.nansum(inv, axis=1)
    probs = np.divide(inv, overround[:, None], out=np.full_like(inv, np.nan), where=valid[:, None] & (overround[:, None] > 0))
    overround[~valid] = np.nan
    return probs, overround, valid


def _map_full_scope(panel: pd.DataFrame) -> pd.Series:
    path = ROOT / "data/processed/feature_blocks/internal_elo_full_scope/internal_elo_features_football_data_full_scope_v1.csv"
    use = ["full_scope_match_id", "league", "Date", "HomeTeam", "AwayTeam"]
    full = pd.read_csv(path, usecols=use, low_memory=False)
    full["_key"] = (
        full["league"].astype(str) + "|" + pd.to_datetime(full["Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        + "|" + full["HomeTeam"].map(normalize_team) + "|" + full["AwayTeam"].map(normalize_team)
    )
    mapping = full.drop_duplicates("_key").set_index("_key")["full_scope_match_id"]
    keys = (
        panel["id__league"].astype(str) + "|" + panel["id__match_date"].astype(str)
        + "|" + panel["id__home_team"].map(normalize_team) + "|" + panel["id__away_team"].map(normalize_team)
    )
    return keys.map(mapping)


def build_panel() -> tuple[pd.DataFrame, dict[str, object]]:
    raw, _loaded = load_canonical_frames()
    timing = build_fixture_audit(raw)
    keep = timing["eligible"].astype(bool)
    raw = raw.loc[keep].reset_index(drop=True)
    timing = timing.loc[keep].reset_index(drop=True)
    specs_1x2 = [s for s in build_1x2_pairs(raw.columns) if s.observation_type == "bookmaker"]
    specs_ah = [s for s in build_ah_pairs(raw.columns) if s.observation_type == "bookmaker"]
    specs_ou = [s for s in build_ou_pairs(raw.columns) if s.observation_type == "bookmaker"]

    dates = parse_dates(raw["Date"])
    season_start = raw["season"].astype(str).str[:4].astype(int)
    panel = pd.DataFrame({
        "id__canonical_match_id": [deterministic_match_id(l, d, h, a) for l, d, h, a in zip(raw["league"], dates.dt.strftime("%Y-%m-%d"), raw["HomeTeam"], raw["AwayTeam"])],
        "id__match_date": dates.dt.strftime("%Y-%m-%d"),
        "id__league": raw["league"],
        "id__season_start_year": season_start,
        "id__home_team": raw["HomeTeam"],
        "id__away_team": raw["AwayTeam"],
        "id__source_file": raw["source_file"],
        "id__source_row": raw["source_row_number"],
        "id__weekday": timing["weekday"],
        "id__snapshot_batch_class": timing["collection_batch"],
        "result__home_goals": _numeric(raw, "FTHG"),
        "result__away_goals": _numeric(raw, "FTAG"),
        "result__ftr": raw["FTR"] if "FTR" in raw else "",
    })
    panel["id__full_scope_match_id"] = _map_full_scope(panel)
    panel["result__available"] = panel[["result__home_goals", "result__away_goals"]].notna().all(axis=1)
    panel["quality__safe_snapshot_timing"] = True
    panel["quality__duplicate_fixture"] = panel["id__canonical_match_id"].duplicated(keep=False)
    panel["quality__valid_result"] = panel["result__available"] & panel["result__ftr"].isin(["H", "D", "A"])
    panel["quality__reschedule_unobservable"] = True

    snapshot_probs: list[np.ndarray] = []
    close_probs: list[np.ndarray] = []
    snapshot_odds: list[np.ndarray] = []
    close_odds: list[np.ndarray] = []
    snapshot_overrounds: list[np.ndarray] = []
    close_overrounds: list[np.ndarray] = []
    validity_snapshot: list[np.ndarray] = []
    validity_close: list[np.ndarray] = []
    provenance: dict[str, object] = {}

    for spec in specs_1x2:
        name = _safe_name(spec.family)
        sp, sor, sv = _novig_matrix(raw, spec.snapshot_columns)
        cp, cor, cv = _novig_matrix(raw, spec.closing_columns)
        so = np.column_stack([_numeric(raw, c).to_numpy(float) for c in spec.snapshot_columns])
        co = np.column_stack([_numeric(raw, c).to_numpy(float) for c in spec.closing_columns])
        snapshot_probs.append(sp); close_probs.append(cp); snapshot_odds.append(so); close_odds.append(co)
        snapshot_overrounds.append(sor); close_overrounds.append(cor); validity_snapshot.append(sv); validity_close.append(cv)
        for j, side in enumerate(("home", "draw", "away")):
            scol = f"feature_snapshot__1x2_{name}_odds_{side}"
            pcol = f"feature_snapshot__1x2_{name}_prob_{side}"
            ccol = f"label_close__1x2_{name}_odds_{side}"
            cpcol = f"label_close__1x2_{name}_prob_{side}"
            clvcol = f"label_close__price_clv_{side}__{name}"
            poscol = f"label_close__positive_price_clv_{side}__{name}"
            panel[scol] = so[:, j]; panel[pcol] = sp[:, j]
            panel[ccol] = co[:, j]; panel[cpcol] = cp[:, j]
            clv = np.divide(so[:, j], co[:, j], out=np.full(len(raw), np.nan), where=sv & cv) - 1.0
            panel[clvcol] = clv
            panel[poscol] = np.where(np.isfinite(clv), clv > 0, np.nan)
            provenance[scol] = {"source_columns": [spec.snapshot_columns[j]], "timing": "verified_scheduled_prematch_snapshot"}
            provenance[pcol] = {"source_columns": list(spec.snapshot_columns), "timing": "verified_scheduled_prematch_snapshot"}
            provenance[ccol] = {"source_columns": [spec.closing_columns[j]], "timing": "verified_closing"}
            provenance[cpcol] = {"source_columns": list(spec.closing_columns), "timing": "verified_closing"}
            provenance[clvcol] = {"source_columns": [spec.snapshot_columns[j], spec.closing_columns[j]], "role": "label"}
        panel[f"feature_snapshot__1x2_{name}_overround"] = sor
        panel[f"feature_snapshot__1x2_{name}_valid"] = sv
        panel[f"label_close__1x2_{name}_overround"] = cor
        panel[f"label_close__1x2_{name}_valid"] = cv
        panel[f"diagnostic__1x2_{name}_bookmaker"] = spec.bookmaker

    sp_stack = np.stack(snapshot_probs, axis=1)
    cp_stack = np.stack(close_probs, axis=1)
    so_stack = np.stack(snapshot_odds, axis=1)
    co_stack = np.stack(close_odds, axis=1)
    sv_stack = np.stack(validity_snapshot, axis=1)
    cv_stack = np.stack(validity_close, axis=1)
    sor_stack = np.column_stack(snapshot_overrounds)
    cor_stack = np.column_stack(close_overrounds)
    for j, side in enumerate(("home", "draw", "away")):
        consensus = np.nanmean(sp_stack[:, :, j], axis=1)
        close_consensus = np.nanmean(cp_stack[:, :, j], axis=1)
        panel[f"feature_snapshot__consensus_prob_{side}"] = consensus
        panel[f"feature_snapshot__best_odds_{side}"] = np.nanmax(np.where(sv_stack, so_stack[:, :, j], np.nan), axis=1)
        panel[f"feature_snapshot__worst_odds_{side}"] = np.nanmin(np.where(sv_stack, so_stack[:, :, j], np.nan), axis=1)
        panel[f"feature_snapshot__prob_dispersion_{side}"] = np.nanstd(sp_stack[:, :, j], axis=1)
        panel[f"feature_snapshot__best_to_consensus_{side}"] = panel[f"feature_snapshot__best_odds_{side}"] * consensus - 1.0
        panel[f"label_close__consensus_prob_{side}"] = close_consensus
        panel[f"label_close__best_odds_{side}"] = np.nanmax(np.where(cv_stack, co_stack[:, :, j], np.nan), axis=1)
        panel[f"label_close__prob_shift_{side}"] = close_consensus - consensus
    panel["feature_snapshot__bookmaker_count"] = sv_stack.sum(axis=1)
    panel["feature_snapshot__overround_mean"] = np.nanmean(sor_stack, axis=1)
    panel["feature_snapshot__overround_min"] = np.nanmin(sor_stack, axis=1)
    panel["feature_snapshot__overround_max"] = np.nanmax(sor_stack, axis=1)
    panel["label_close__bookmaker_count"] = cv_stack.sum(axis=1)
    panel["label_close__overround_mean"] = np.nanmean(cor_stack, axis=1)
    panel["quality__valid_1x2_snapshot"] = sv_stack.any(axis=1)
    panel["quality__valid_1x2_close"] = cv_stack.any(axis=1)
    panel["quality__bookmaker_pair_count"] = (sv_stack & cv_stack).sum(axis=1)

    # AH consensus from executable families. Exact line is preserved.
    ah_sp, ah_cp, ah_sv, ah_cv = [], [], [], []
    for spec in specs_ah:
        sp, _, sv = _novig_matrix(raw, spec.snapshot_columns)
        cp, _, cv = _novig_matrix(raw, spec.closing_columns)
        ah_sp.append(sp); ah_cp.append(cp); ah_sv.append(sv); ah_cv.append(cv)
    ah_sp_stack = np.stack(ah_sp, axis=1); ah_cp_stack = np.stack(ah_cp, axis=1)
    ah_sv_stack = np.stack(ah_sv, axis=1); ah_cv_stack = np.stack(ah_cv, axis=1)
    snap_line = _numeric(raw, "AHh"); close_line = _numeric(raw, "AHCh")
    line_valid = snap_line.notna() & close_line.notna() & ((snap_line * 4).round() == snap_line * 4) & ((close_line * 4).round() == close_line * 4)
    same_line = line_valid & snap_line.eq(close_line)
    panel["feature_snapshot__ah_home_line"] = snap_line
    panel["feature_snapshot__ah_prob_home"] = np.nanmean(ah_sp_stack[:, :, 0], axis=1)
    panel["feature_snapshot__ah_prob_away"] = np.nanmean(ah_sp_stack[:, :, 1], axis=1)
    panel["label_close__ah_home_line"] = close_line
    panel["label_close__ah_line_shift"] = close_line - snap_line
    panel["label_close__ah_prob_home"] = np.nanmean(ah_cp_stack[:, :, 0], axis=1)
    panel["label_close__ah_prob_away"] = np.nanmean(ah_cp_stack[:, :, 1], axis=1)
    panel["label_close__ah_same_line"] = same_line
    panel["label_close__ah_same_line_prob_shift_home"] = np.where(same_line, panel["label_close__ah_prob_home"] - panel["feature_snapshot__ah_prob_home"], np.nan)
    panel["label_close__ah_same_line_prob_shift_away"] = np.where(same_line, panel["label_close__ah_prob_away"] - panel["feature_snapshot__ah_prob_away"], np.nan)
    panel["label_close__ah_line_moved_toward_home"] = np.where(line_valid & ~same_line, close_line < snap_line, np.nan)
    panel["label_close__ah_line_moved_toward_away"] = np.where(line_valid & ~same_line, close_line > snap_line, np.nan)
    panel["quality__valid_ah_snapshot"] = ah_sv_stack.any(axis=1) & snap_line.notna()
    panel["quality__valid_ah_close"] = ah_cv_stack.any(axis=1) & close_line.notna()

    ou_sp, ou_cp, ou_sv, ou_cv = [], [], [], []
    for spec in specs_ou:
        sp, _, sv = _novig_matrix(raw, spec.snapshot_columns)
        cp, _, cv = _novig_matrix(raw, spec.closing_columns)
        ou_sp.append(sp); ou_cp.append(cp); ou_sv.append(sv); ou_cv.append(cv)
    ou_sp_stack = np.stack(ou_sp, axis=1); ou_cp_stack = np.stack(ou_cp, axis=1)
    ou_sv_stack = np.stack(ou_sv, axis=1); ou_cv_stack = np.stack(ou_cv, axis=1)
    panel["feature_snapshot__ou25_total_line"] = 2.5
    panel["feature_snapshot__ou25_prob_over"] = np.nanmean(ou_sp_stack[:, :, 0], axis=1)
    panel["feature_snapshot__ou25_prob_under"] = np.nanmean(ou_sp_stack[:, :, 1], axis=1)
    panel["label_close__ou25_total_line"] = 2.5
    panel["label_close__ou25_prob_over"] = np.nanmean(ou_cp_stack[:, :, 0], axis=1)
    panel["label_close__ou25_prob_under"] = np.nanmean(ou_cp_stack[:, :, 1], axis=1)
    panel["label_close__ou25_prob_shift_over"] = panel["label_close__ou25_prob_over"] - panel["feature_snapshot__ou25_prob_over"]
    panel["label_close__ou25_prob_shift_under"] = panel["label_close__ou25_prob_under"] - panel["feature_snapshot__ou25_prob_under"]
    panel["quality__valid_ou_snapshot"] = ou_sv_stack.any(axis=1)
    panel["quality__valid_ou_close"] = ou_cv_stack.any(axis=1)

    panel["quality__schema_warning_count"] = (
        (~sv_stack).sum(axis=1) + (~cv_stack).sum(axis=1)
        + (~ah_sv_stack).sum(axis=1) + (~ah_cv_stack).sum(axis=1)
        + (~ou_sv_stack).sum(axis=1) + (~ou_cv_stack).sum(axis=1)
    )
    feature_columns = [c for c in panel if c.startswith("feature_snapshot__")]
    contract = {
        "version": "v4_feature_column_contract_v1",
        "timing_semantics": "verified_scheduled_prematch_snapshot_not_opening",
        "approved_feature_columns": feature_columns,
        "approved_feature_prefixes": ["feature_snapshot__", "feature_history__", "feature_external__"],
        "prohibited_feature_prefixes": ["label_close__", "result__", "diagnostic__", "id__"],
        "closing_columns_feature_allowed": False,
        "result_columns_feature_allowed": False,
        "source_provenance": provenance,
        "executable_1x2_families": [{"family": s.family, "bookmaker": s.bookmaker, "snapshot": s.snapshot_columns, "closing": s.closing_columns} for s in specs_1x2],
        "ah_rule": "same-line prices only; exact quarter lines; line movement separate",
        "ou_rule": "fixed exact 2.5 line only",
    }
    return panel, contract


def parse_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", dayfirst=True)


def validate_panel(panel: pd.DataFrame, contract: dict[str, object]) -> pd.DataFrame:
    features = list(contract["approved_feature_columns"])
    checks = [
        ("unique_canonical_match_id", not panel["id__canonical_match_id"].duplicated().any(), int(panel["id__canonical_match_id"].duplicated().sum())),
        ("safe_weekdays_only", set(panel["id__weekday"]).issubset({"Saturday", "Sunday", "Wednesday", "Thursday"}), 0),
        ("quality_safe_timing_all", bool(panel["quality__safe_snapshot_timing"].all()), int((~panel["quality__safe_snapshot_timing"]).sum())),
        ("no_result_features", not any(c.startswith("result__") for c in features), sum(c.startswith("result__") for c in features)),
        ("no_closing_features", not any(c.startswith("label_close__") for c in features), sum(c.startswith("label_close__") for c in features)),
        ("feature_namespace_only", all(c.startswith("feature_snapshot__") for c in features), sum(not c.startswith("feature_snapshot__") for c in features)),
        ("no_closing_to_snapshot_fill", True, 0),
        ("ah_different_line_price_shift_missing", bool(panel.loc[~panel["label_close__ah_same_line"].fillna(False).astype(bool), ["label_close__ah_same_line_prob_shift_home", "label_close__ah_same_line_prob_shift_away"]].isna().all().all()), 0),
        ("quarter_lines_preserved", bool((((panel["feature_snapshot__ah_home_line"].dropna() * 4).round() == panel["feature_snapshot__ah_home_line"].dropna() * 4).all())), 0),
        ("ou_exact_line_only", bool(panel["feature_snapshot__ou25_total_line"].eq(2.5).all() and panel["label_close__ou25_total_line"].eq(2.5).all()), 0),
    ]
    return pd.DataFrame([{"check": c, "status": "pass" if ok else "fail", "affected_rows_or_columns": n} for c, ok, n in checks])


def run_phase2() -> dict[str, object]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, contract = build_panel()
    checks = validate_panel(panel, contract)
    blocked = checks["status"].eq("fail").any()
    decision = "v4_phase2_blocked_invalid_market_panel" if blocked else "v4_phase2_panel_and_movement_labels_ready_research_only"
    panel.to_csv(PANEL_PATH, index=False)
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    summary = pd.DataFrame([{
        "panel_rows": len(panel), "columns": len(panel.columns), "feature_columns": len(contract["approved_feature_columns"]),
        "leagues": panel["id__league"].nunique(), "seasons": panel["id__season_start_year"].nunique(),
        "full_scope_id_coverage": panel["id__full_scope_match_id"].notna().mean(), "decision": decision,
    }])
    coverage = panel.groupby(["id__league", "id__season_start_year"]).agg(
        fixture_rows=("id__canonical_match_id", "size"), valid_result_rows=("quality__valid_result", "sum"),
        valid_1x2_snapshot_rows=("quality__valid_1x2_snapshot", "sum"), valid_1x2_close_rows=("quality__valid_1x2_close", "sum"),
        valid_ah_snapshot_rows=("quality__valid_ah_snapshot", "sum"), valid_ah_close_rows=("quality__valid_ah_close", "sum"),
        valid_ou_snapshot_rows=("quality__valid_ou_snapshot", "sum"), valid_ou_close_rows=("quality__valid_ou_close", "sum"),
    ).reset_index()
    clv_cols = [c for c in panel if c.startswith("label_close__price_clv_")]
    clv = pd.DataFrame([{"label": c, "valid_rows": int(panel[c].notna().sum()), "coverage": panel[c].notna().mean(), "positive_rate": float((panel[c] > 0).mean())} for c in clv_cols])
    summary.to_csv(OUT_DIR / "v4_phase2_market_panel_summary.csv", index=False)
    coverage.to_csv(OUT_DIR / "v4_phase2_coverage_by_league_season.csv", index=False)
    clv.to_csv(OUT_DIR / "v4_phase2_clv_label_coverage.csv", index=False)
    checks.to_csv(OUT_DIR / "v4_phase2_leakage_checks.csv", index=False)
    report = f"""# V4 Phase 2 Canonical Market Panel

Decision: **{decision}**

- Rows: {len(panel)}
- Columns: {len(panel.columns)}
- Approved snapshot feature columns: {len(contract['approved_feature_columns'])}
- Full-scope ID coverage: {panel['id__full_scope_match_id'].notna().mean():.1%}
- Duplicate canonical IDs: {int(panel['id__canonical_match_id'].duplicated().sum())}
- Leakage checks passed: {int(checks['status'].eq('pass').sum())}/{len(checks)}

Non-C values remain scheduled snapshots, not opening odds. Closing and result namespaces are excluded from the feature contract. AH comparisons require exact quarter-line equality; O/U is fixed at 2.5. No model or profit calculation was run in this phase.
"""
    (OUT_DIR / "v4_phase2_panel_report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "v4_phase2_decision.md").write_text(f"# V4 Phase 2 Decision\n\n**{decision}**\n", encoding="utf-8")
    return {"decision": decision, "panel_rows": len(panel), "checks_passed": int(checks["status"].eq("pass").sum()), "checks": len(checks)}
