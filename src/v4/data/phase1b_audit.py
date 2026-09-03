"""V4 Phase 1B: fixture-safe scheduled-snapshot to verified-close audit.

This module never calls football-data non-C odds opening odds.  It promotes
them only on fixture weekdays for which the provider's documented batch day
must precede the fixture date by at least one full calendar day.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .market_contract import classify_column
from .phase1_audit import OUT_DIR, ROOT, SCOPE_LEAGUES, clean_columns, parse_dates, read_csv_resilient


PHASE1_COVERAGE = OUT_DIR / "v4_market_coverage_by_league_season.csv"

TIMING_CLASSES_1B = {
    "verified_opening",
    "verified_scheduled_prematch_snapshot",
    "scheduled_snapshot_ambiguous",
    "verified_closing",
    "timing_unknown",
    "not_applicable",
}

AGGREGATE_BOOKMAKERS = {
    "Market average", "Market maximum", "BetBrain average", "BetBrain maximum"
}

AH_CANDIDATES = {
    "B365": (("B365AHH", "B365AHA"), ("B365CAHH", "B365CAHA")),
    "P": (("PAHH", "PAHA"), ("PCAHH", "PCAHA")),
    "Max": (("MaxAHH", "MaxAHA"), ("MaxCAHH", "MaxCAHA")),
    "Avg": (("AvgAHH", "AvgAHA"), ("AvgCAHH", "AvgCAHA")),
    "BFE": (("BFEAHH", "BFEAHA"), ("BFECAHH", "BFECAHA")),
}


@dataclass(frozen=True)
class PairSpec:
    market: str
    family: str
    bookmaker: str
    observation_type: str
    snapshot_columns: tuple[str, ...]
    closing_columns: tuple[str, ...]
    snapshot_line_column: str = ""
    closing_line_column: str = ""
    fixed_line: float | None = None


def decimal_odds_valid(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    columns = list(columns)
    if not columns or not set(columns).issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.notna().all(axis=1) & numeric.gt(1).all(axis=1)


def fields_complete(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    columns = list(columns)
    if not columns or not set(columns).issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    return frame[columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)


def no_vig_probabilities(odds: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(odds), dtype=float)
    if values.size < 2 or not np.isfinite(values).all() or (values <= 1).any():
        return np.full(values.shape, np.nan)
    inverse = 1.0 / values
    return inverse / inverse.sum()


def probability_shift(snapshot_odds: Iterable[float], closing_odds: Iterable[float]) -> np.ndarray:
    """Closing no-vig probability minus snapshot no-vig probability."""

    return no_vig_probabilities(closing_odds) - no_vig_probabilities(snapshot_odds)


def price_clv(snapshot_odds: float, closing_odds: float) -> float:
    """Scheduled snapshot price divided by closing price, minus one."""

    if not np.isfinite(snapshot_odds) or not np.isfinite(closing_odds) or snapshot_odds <= 1 or closing_odds <= 1:
        return np.nan
    return float(snapshot_odds / closing_odds - 1.0)


def directly_comparable_prices(snapshot_line: float, closing_line: float) -> bool:
    return bool(np.isfinite(snapshot_line) and np.isfinite(closing_line) and snapshot_line == closing_line)


def ah_price_shift(snapshot_line: float, closing_line: float, snapshot_odds: Iterable[float], closing_odds: Iterable[float]) -> np.ndarray:
    """Return no-vig price shift only when the exact AH selections match."""

    if not directly_comparable_prices(snapshot_line, closing_line):
        return np.array([np.nan, np.nan])
    return probability_shift(snapshot_odds, closing_odds)


def ou_price_shift(snapshot_line: float, closing_line: float, snapshot_odds: Iterable[float], closing_odds: Iterable[float]) -> np.ndarray:
    """Return no-vig price shift only when the exact total selections match."""

    if not directly_comparable_prices(snapshot_line, closing_line):
        return np.array([np.nan, np.nan])
    return probability_shift(snapshot_odds, closing_odds)


def fixture_timing(date: pd.Timestamp, time_available: bool, duplicate_fixture: bool = False) -> dict[str, object]:
    """Classify fixture timing from documented batch day and fixture weekday."""

    if duplicate_fixture:
        return {
            "timing_classification": "timing_unknown",
            "eligible": False,
            "collection_batch": "unresolved",
            "documented_collection_weekday": "",
            "full_day_separation": False,
            "eligibility_reason": "excluded_duplicate_logical_fixture",
        }
    if pd.isna(date):
        return {
            "timing_classification": "timing_unknown",
            "eligible": False,
            "collection_batch": "unresolved",
            "documented_collection_weekday": "",
            "full_day_separation": False,
            "eligibility_reason": "excluded_missing_or_invalid_match_date",
        }
    weekday = int(date.weekday())
    if weekday in {5, 6}:  # Saturday / Sunday
        return {
            "timing_classification": "verified_scheduled_prematch_snapshot",
            "eligible": True,
            "collection_batch": "weekend_games_friday_afternoon",
            "documented_collection_weekday": "Friday",
            "full_day_separation": True,
            "eligibility_reason": "eligible_weekend_fixture_after_friday_batch",
        }
    if weekday in {2, 3}:  # Wednesday / Thursday
        return {
            "timing_classification": "verified_scheduled_prematch_snapshot",
            "eligible": True,
            "collection_batch": "midweek_games_tuesday_afternoon",
            "documented_collection_weekday": "Tuesday",
            "full_day_separation": True,
            "eligibility_reason": "eligible_midweek_fixture_after_tuesday_batch",
        }
    if weekday == 4:
        reason = "excluded_friday_same_day_batch_intraday_order_unproved"
        batch, day = "weekend_games_friday_afternoon", "Friday"
    elif weekday == 1:
        reason = "excluded_tuesday_same_day_batch_intraday_order_unproved"
        batch, day = "midweek_games_tuesday_afternoon", "Tuesday"
    elif weekday == 0:
        reason = "excluded_monday_not_clearly_covered_by_weekend_convention"
        batch, day = "weekend_scope_ambiguous", "Friday"
    else:
        reason = "excluded_no_documented_batch_rule_for_weekday"
        batch, day = "unresolved", ""
    return {
        "timing_classification": "scheduled_snapshot_ambiguous",
        "eligible": False,
        "collection_batch": batch,
        "documented_collection_weekday": day,
        "full_day_separation": False,
        "eligibility_reason": reason,
    }


def _schema_signature(columns: Iterable[str]) -> str:
    return hashlib.sha256("|".join(columns).encode("utf-8")).hexdigest()[:16]


def load_canonical_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = pd.read_csv(PHASE1_COVERAGE)
    chosen = coverage[
        coverage["source"].eq("football_data")
        & coverage["closing_one_x_two_coverage"].gt(0)
        & coverage["league"].isin(SCOPE_LEAGUES)
    ].copy()
    frames: list[pd.DataFrame] = []
    load_rows: list[dict[str, object]] = []
    for record in chosen.itertuples(index=False):
        path = ROOT / record.source_file
        frame, malformed = read_csv_resilient(path)
        if "Div" in frame:
            frame = frame[frame["Div"].astype("string").str.strip().eq(record.league)].copy()
        frame["league"] = record.league
        frame["season"] = record.season
        frame["source_file"] = record.source_file
        frame["source_row_number"] = np.arange(len(frame), dtype=int) + 2
        frame["schema_signature"] = _schema_signature([str(c) for c in frame.columns if c not in {"league", "season", "source_file", "source_row_number"}])
        frames.append(frame)
        load_rows.append({"league": record.league, "season": record.season, "source_file": record.source_file, "rows": len(frame), "malformed_rows": malformed})
    return pd.concat(frames, ignore_index=True, sort=False), pd.DataFrame(load_rows)


def build_fixture_audit(frame: pd.DataFrame) -> pd.DataFrame:
    dates = parse_dates(frame["Date"])
    time_text = frame["Time"].astype("string").str.strip() if "Time" in frame else pd.Series("", index=frame.index, dtype="string")
    time_available = time_text.str.match(r"^(?:[01]?\d|2[0-3]):[0-5]\d$", na=False)
    home = frame["HomeTeam"].astype("string").str.strip()
    away = frame["AwayTeam"].astype("string").str.strip()
    fixture_key = (
        frame["league"].astype(str) + "|" + frame["season"].astype(str) + "|"
        + dates.dt.strftime("%Y-%m-%d").fillna("invalid") + "|" + home.str.lower() + "|" + away.str.lower()
    )
    duplicate = fixture_key.duplicated(keep=False)
    timing = pd.DataFrame([
        fixture_timing(date, bool(has_time), bool(is_duplicate))
        for date, has_time, is_duplicate in zip(dates, time_available, duplicate)
    ])
    out = pd.DataFrame({
        "fixture_id": fixture_key,
        "league": frame["league"],
        "season": frame["season"],
        "source_file": frame["source_file"],
        "source_row_number": frame["source_row_number"],
        "schema_signature": frame["schema_signature"],
        "match_date": dates.dt.strftime("%Y-%m-%d").fillna(""),
        "match_time": time_text.fillna(""),
        "match_time_available": time_available,
        "weekday": dates.dt.day_name().fillna("Unknown"),
        "home_team": home,
        "away_team": away,
        "duplicate_logical_fixture": duplicate,
        "reschedule_status": "not_observable_no_original_schedule_or_status_column",
        "reschedule_exclusion": False,
    })
    out = pd.concat([out.reset_index(drop=True), timing.reset_index(drop=True)], axis=1)
    out["provider_evidence_id"] = "fd_archive_notes_weekend_friday_midweek_tuesday"
    out["exact_collection_timestamp_known"] = False
    out["non_c_is_opening"] = False
    return out


def _bookmaker_for(column: str) -> str:
    contract = classify_column("football_data", column)
    return contract.bookmaker if contract is not None else "unknown"


def build_1x2_pairs(columns: Iterable[str]) -> list[PairSpec]:
    cols = set(columns)
    prefixes: set[str] = set()
    for column in cols:
        contract = classify_column("football_data", column)
        if contract is None or contract.market != "1X2" or contract.role != "odds":
            continue
        base = re.sub(r"C(?=[HDA]$)", "", column) if contract.timing_classification == "verified_closing" else column
        if base.endswith(("H", "D", "A")):
            prefixes.add(base[:-1])
    output = []
    for prefix in sorted(prefixes):
        snapshot = tuple(prefix + side for side in "HDA")
        closing = tuple(prefix + "C" + side for side in "HDA")
        if not (set(snapshot) | set(closing)).intersection(cols):
            continue
        bookmaker = _bookmaker_for(snapshot[0])
        observation_type = "consensus_aggregate" if bookmaker in AGGREGATE_BOOKMAKERS else "bookmaker"
        output.append(PairSpec("1X2", prefix, bookmaker, observation_type, snapshot, closing))
    return output


def build_ah_pairs(columns: Iterable[str]) -> list[PairSpec]:
    cols = set(columns)
    output = []
    for family, (snapshot, closing) in AH_CANDIDATES.items():
        if not (set(snapshot) | set(closing)).intersection(cols):
            continue
        bookmaker = _bookmaker_for(snapshot[0])
        observation_type = "consensus_aggregate" if bookmaker in AGGREGATE_BOOKMAKERS else "bookmaker"
        output.append(PairSpec("Asian Handicap", family, bookmaker, observation_type, snapshot, closing, "AHh", "AHCh"))
    return output


def build_ou_pairs(columns: Iterable[str]) -> list[PairSpec]:
    cols = set(columns)
    families: set[tuple[str, float]] = set()
    for column in cols:
        contract = classify_column("football_data", column)
        if contract is None or contract.market != "Over/Under" or contract.role != "odds" or contract.selection != "over":
            continue
        if contract.timing_classification == "verified_closing":
            base = re.sub(r"C(?=>)", "", column)
        else:
            base = column
        match = re.fullmatch(r"(.+)>\s*(\d+(?:\.\d+)?)", base)
        if match:
            families.add((match.group(1), float(match.group(2))))
    output = []
    for prefix, line in sorted(families):
        text = f"{line:g}"
        snapshot = (f"{prefix}>{text}", f"{prefix}<{text}")
        closing = (f"{prefix}C>{text}", f"{prefix}C<{text}")
        bookmaker = _bookmaker_for(snapshot[0])
        observation_type = "consensus_aggregate" if bookmaker in AGGREGATE_BOOKMAKERS else "bookmaker"
        output.append(PairSpec("Over/Under", f"{prefix}_{text}", bookmaker, observation_type, snapshot, closing, fixed_line=line))
    return output


def _pair_masks(frame: pd.DataFrame, spec: PairSpec, eligible: pd.Series) -> dict[str, pd.Series]:
    snapshot_schema = set(spec.snapshot_columns).issubset(frame.columns)
    closing_schema = set(spec.closing_columns).issubset(frame.columns)
    snapshot_complete = fields_complete(frame, spec.snapshot_columns)
    closing_complete = fields_complete(frame, spec.closing_columns)
    snapshot_valid = decimal_odds_valid(frame, spec.snapshot_columns)
    closing_valid = decimal_odds_valid(frame, spec.closing_columns)
    line_complete = pd.Series(True, index=frame.index)
    line_valid = pd.Series(True, index=frame.index)
    if spec.market == "Asian Handicap":
        if {spec.snapshot_line_column, spec.closing_line_column}.issubset(frame.columns):
            lines = frame[[spec.snapshot_line_column, spec.closing_line_column]].apply(pd.to_numeric, errors="coerce")
            line_complete = lines.notna().all(axis=1)
            line_valid = line_complete & ((lines * 4).round().sub(lines * 4).abs().le(1e-8)).all(axis=1) & lines.abs().le(10).all(axis=1)
        else:
            line_complete &= False
            line_valid &= False
    pair_complete = snapshot_complete & closing_complete & line_complete
    pair_valid = eligible & snapshot_valid & closing_valid & line_valid
    return {
        "snapshot_schema": pd.Series(snapshot_schema, index=frame.index),
        "closing_schema": pd.Series(closing_schema, index=frame.index),
        "snapshot_complete": snapshot_complete,
        "closing_complete": closing_complete,
        "snapshot_valid": snapshot_valid,
        "closing_valid": closing_valid,
        "line_complete": line_complete,
        "line_valid": line_valid,
        "pair_complete": pair_complete,
        "pair_valid": pair_valid,
    }


def pair_audit(frame: pd.DataFrame, fixtures: pd.DataFrame, specs: list[PairSpec]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    eligible = fixtures["eligible"].astype(bool)
    rows: list[dict[str, object]] = []
    any_valid = {
        key: pd.Series(False, index=frame.index)
        for market in ("1X2", "Asian Handicap", "Over/Under")
        for key in (market, market + "_bookmaker")
    }
    for (league, season), indices in frame.groupby(["league", "season"], sort=True).groups.items():
        idx = pd.Index(indices)
        sub_eligible = eligible.loc[idx]
        for spec in specs:
            masks = _pair_masks(frame.loc[idx], spec, sub_eligible)
            valid_pair = masks["pair_valid"]
            any_valid[spec.market].loc[idx] |= valid_pair
            if spec.observation_type == "bookmaker":
                any_valid[spec.market + "_bookmaker"].loc[idx] |= valid_pair
            both_present = masks["snapshot_complete"] & masks["closing_complete"] & masks["line_complete"]
            required = set(spec.snapshot_columns) | set(spec.closing_columns) | ({spec.snapshot_line_column, spec.closing_line_column} if spec.market == "Asian Handicap" else set())
            schema_missing = sorted(column for column in required if column not in frame.columns or frame.loc[idx, column].notna().sum() == 0)
            safe_n = int(sub_eligible.sum())
            rows.append({
                "league": league,
                "season": season,
                "market": spec.market,
                "bookmaker": spec.bookmaker,
                "observation_type": spec.observation_type,
                "snapshot_column_family": "|".join(spec.snapshot_columns),
                "closing_column_family": "|".join(spec.closing_columns),
                "snapshot_line_column": spec.snapshot_line_column,
                "closing_line_column": spec.closing_line_column,
                "fixed_line": spec.fixed_line,
                "fixture_rows": len(idx),
                "safe_timing_fixture_rows": safe_n,
                "safe_timing_fixture_coverage": safe_n / len(idx) if len(idx) else 0.0,
                "snapshot_complete_rows": int((sub_eligible & masks["snapshot_complete"]).sum()),
                "closing_complete_rows": int((sub_eligible & masks["closing_complete"]).sum()),
                "both_sides_or_triplet_present_rows": int((sub_eligible & both_present).sum()),
                "valid_paired_rows": int(valid_pair.sum()),
                "valid_paired_coverage_of_safe_fixtures": float(valid_pair.sum() / safe_n) if safe_n else 0.0,
                "odds_gt_1_coverage_of_complete_pairs": float(valid_pair.sum() / (sub_eligible & both_present).sum()) if (sub_eligible & both_present).sum() else 0.0,
                "snapshot_present_closing_missing_rows": int((sub_eligible & masks["snapshot_complete"] & ~masks["closing_complete"]).sum()),
                "closing_present_snapshot_missing_rows": int((sub_eligible & masks["closing_complete"] & ~masks["snapshot_complete"]).sum()),
                "no_vig_labels_buildable_rows": int(valid_pair.sum()),
                "price_clv_buildable_rows": int(valid_pair.sum()) if spec.market == "1X2" and spec.observation_type == "bookmaker" else 0,
                "schema_transition_warning": "missing_required_columns:" + "|".join(schema_missing) if schema_missing else "",
            })
    return pd.DataFrame(rows), any_valid


def line_movement_audit(frame: pd.DataFrame, fixtures: pd.DataFrame, specs: list[PairSpec], market: str) -> pd.DataFrame:
    eligible = fixtures["eligible"].astype(bool)
    rows: list[dict[str, object]] = []
    for (league, season), indices in frame.groupby(["league", "season"], sort=True).groups.items():
        idx = pd.Index(indices)
        for spec in [x for x in specs if x.market == market]:
            masks = _pair_masks(frame.loc[idx], spec, eligible.loc[idx])
            valid_pair = masks["pair_valid"]
            if market == "Asian Handicap" and {spec.snapshot_line_column, spec.closing_line_column}.issubset(frame.columns):
                snapshot_line = pd.to_numeric(frame.loc[idx, spec.snapshot_line_column], errors="coerce")
                closing_line = pd.to_numeric(frame.loc[idx, spec.closing_line_column], errors="coerce")
            else:
                snapshot_line = pd.Series(spec.fixed_line, index=idx, dtype=float)
                closing_line = pd.Series(spec.fixed_line, index=idx, dtype=float)
            same = valid_pair & snapshot_line.eq(closing_line)
            different = valid_pair & ~snapshot_line.eq(closing_line)
            shift = closing_line - snapshot_line
            rows.append({
                "league": league,
                "season": season,
                "market": market,
                "bookmaker": spec.bookmaker,
                "observation_type": spec.observation_type,
                "snapshot_column_family": "|".join(spec.snapshot_columns),
                "closing_column_family": "|".join(spec.closing_columns),
                "safe_valid_paired_rows": int(valid_pair.sum()),
                "same_line_rows": int(same.sum()),
                "different_line_rows": int(different.sum()),
                "same_line_price_movement_rows": int(same.sum()),
                "different_line_price_comparison_prohibited_rows": int(different.sum()),
                "line_moved_toward_home_or_over_rows": int((different & shift.lt(0 if market == "Asian Handicap" else np.inf)).sum()) if market == "Asian Handicap" else int((different & shift.gt(0)).sum()),
                "line_moved_toward_away_or_under_rows": int((different & shift.gt(0)).sum()) if market == "Asian Handicap" else int((different & shift.lt(0)).sum()),
                "unchanged_line_rows": int(same.sum()),
                "quarter_line_rows": int((valid_pair & ((snapshot_line * 4).round().eq(snapshot_line * 4)) & ((closing_line * 4).round().eq(closing_line * 4))).sum()),
                "mean_closing_minus_snapshot_line": float(shift[valid_pair].mean()) if valid_pair.any() else np.nan,
                "combined_price_and_line_target_allowed": False,
            })
    return pd.DataFrame(rows)


def safe_coverage(frame: pd.DataFrame, fixtures: pd.DataFrame, pairs: pd.DataFrame, any_valid: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for (league, season), indices in frame.groupby(["league", "season"], sort=True).groups.items():
        idx = pd.Index(indices)
        eligible = fixtures.loc[idx, "eligible"].astype(bool)
        if {"AHh", "AHCh"}.issubset(frame.columns):
            snapshot_ah_line = pd.to_numeric(frame.loc[idx, "AHh"], errors="coerce")
            closing_ah_line = pd.to_numeric(frame.loc[idx, "AHCh"], errors="coerce")
            ah_same_line = any_valid["Asian Handicap_bookmaker"].loc[idx] & snapshot_ah_line.eq(closing_ah_line)
            ah_different_line = any_valid["Asian Handicap_bookmaker"].loc[idx] & ~snapshot_ah_line.eq(closing_ah_line)
            ah_shift = closing_ah_line - snapshot_ah_line
        else:
            ah_same_line = pd.Series(False, index=idx)
            ah_different_line = pd.Series(False, index=idx)
            ah_shift = pd.Series(np.nan, index=idx)
        market_pairs = pairs[(pairs["league"].eq(league)) & pairs["season"].eq(season)]
        executable = market_pairs[market_pairs["observation_type"].eq("bookmaker")]
        rows.append({
            "league": league,
            "season": season,
            "fixture_rows": len(idx),
            "safe_timing_fixture_rows": int(eligible.sum()),
            "safe_timing_fixture_coverage": float(eligible.mean()),
            "match_time_available_rows": int(fixtures.loc[idx, "match_time_available"].sum()),
            "match_time_available_coverage": float(fixtures.loc[idx, "match_time_available"].mean()),
            "verified_snapshot_close_1x2_fixture_rows": int(any_valid["1X2"].loc[idx].sum()),
            "verified_snapshot_close_1x2_bookmaker_fixture_rows": int(any_valid["1X2_bookmaker"].loc[idx].sum()),
            "verified_snapshot_close_ah_fixture_rows": int(any_valid["Asian Handicap"].loc[idx].sum()),
            "verified_snapshot_close_ah_bookmaker_fixture_rows": int(any_valid["Asian Handicap_bookmaker"].loc[idx].sum()),
            "verified_snapshot_close_ou_fixture_rows": int(any_valid["Over/Under"].loc[idx].sum()),
            "verified_snapshot_close_ou_bookmaker_fixture_rows": int(any_valid["Over/Under_bookmaker"].loc[idx].sum()),
            "verified_snapshot_close_cross_market_fixture_rows": int((any_valid["1X2"].loc[idx] & any_valid["Asian Handicap"].loc[idx] & any_valid["Over/Under"].loc[idx]).sum()),
            "verified_snapshot_close_cross_market_bookmaker_fixture_rows": int((any_valid["1X2_bookmaker"].loc[idx] & any_valid["Asian Handicap_bookmaker"].loc[idx] & any_valid["Over/Under_bookmaker"].loc[idx]).sum()),
            "ah_same_line_fixture_rows": int(ah_same_line.sum()),
            "ah_different_line_fixture_rows": int(ah_different_line.sum()),
            "ah_line_moved_toward_home_fixture_rows": int((ah_different_line & ah_shift.lt(0)).sum()),
            "ah_line_moved_toward_away_fixture_rows": int((ah_different_line & ah_shift.gt(0)).sum()),
            "ou_same_line_fixture_rows": int(any_valid["Over/Under_bookmaker"].loc[idx].sum()),
            "ou_different_line_fixture_rows": 0,
            "executable_1x2_bookmaker_pair_count": int(executable[(executable["market"].eq("1X2")) & executable["valid_paired_rows"].gt(0)].shape[0]),
            "executable_ah_bookmaker_pair_count": int(executable[(executable["market"].eq("Asian Handicap")) & executable["valid_paired_rows"].gt(0)].shape[0]),
            "executable_ou_bookmaker_pair_count": int(executable[(executable["market"].eq("Over/Under")) & executable["valid_paired_rows"].gt(0)].shape[0]),
            "excluded_friday_rows": int(fixtures.loc[idx, "eligibility_reason"].eq("excluded_friday_same_day_batch_intraday_order_unproved").sum()),
            "excluded_tuesday_rows": int(fixtures.loc[idx, "eligibility_reason"].eq("excluded_tuesday_same_day_batch_intraday_order_unproved").sum()),
            "excluded_monday_rows": int(fixtures.loc[idx, "eligibility_reason"].eq("excluded_monday_not_clearly_covered_by_weekend_convention").sum()),
            "excluded_duplicate_or_invalid_date_rows": int(fixtures.loc[idx, "eligibility_reason"].isin({"excluded_duplicate_logical_fixture", "excluded_missing_or_invalid_match_date"}).sum()),
        })
    return pd.DataFrame(rows)


def weekday_coverage(frame: pd.DataFrame, fixtures: pd.DataFrame, any_valid: dict[str, pd.Series]) -> pd.DataFrame:
    work = fixtures[["league", "season", "weekday", "eligible", "timing_classification", "eligibility_reason", "match_time_available"]].copy()
    work["valid_1x2_pair"] = any_valid["1X2"].values
    work["valid_ah_pair"] = any_valid["Asian Handicap"].values
    work["valid_ou_pair"] = any_valid["Over/Under"].values
    return work.groupby(["league", "season", "weekday", "timing_classification", "eligibility_reason"], dropna=False, sort=True).agg(
        fixture_rows=("eligible", "size"),
        eligible_rows=("eligible", "sum"),
        match_time_available_rows=("match_time_available", "sum"),
        valid_1x2_pair_rows=("valid_1x2_pair", "sum"),
        valid_ah_pair_rows=("valid_ah_pair", "sum"),
        valid_ou_pair_rows=("valid_ou_pair", "sum"),
    ).reset_index()


def timing_contract(specs: list[PairSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    snapshot_columns: dict[str, tuple[str, str, str]] = {}
    closing_columns: dict[str, tuple[str, str, str]] = {}
    for spec in specs:
        for column in spec.snapshot_columns + ((spec.snapshot_line_column,) if spec.snapshot_line_column else ()):
            snapshot_columns[column] = (spec.market, spec.family, "line" if column == spec.snapshot_line_column else "odds")
        for column in spec.closing_columns + ((spec.closing_line_column,) if spec.closing_line_column else ()):
            closing_columns[column] = (spec.market, spec.family, "line" if column == spec.closing_line_column else "odds")
    for column, (market, family, role) in sorted(snapshot_columns.items()):
        for timing, condition, policy in [
            ("verified_scheduled_prematch_snapshot", "fixture weekday is Saturday/Sunday after Friday batch or Wednesday/Thursday after Tuesday batch; no duplicate/invalid date", "allowed_scheduled_snapshot_feature_research_only"),
            ("scheduled_snapshot_ambiguous", "fixture is Friday, Tuesday, or Monday; exact intraday ordering/scope is unproved", "prohibited"),
            ("timing_unknown", "missing/invalid date or duplicate logical fixture", "prohibited"),
        ]:
            rows.append({"source": "football_data", "column": column, "market": market, "family": family, "role": role, "timing_classification": timing, "fixture_condition": condition, "feature_policy": policy, "evidence_id": "fd_archive_notes_weekend_friday_midweek_tuesday", "is_opening": False, "closing_field_can_be_snapshot_feature": False})
    for column, (market, family, role) in sorted(closing_columns.items()):
        rows.append({"source": "football_data", "column": column, "market": market, "family": family, "role": role, "timing_classification": "verified_closing", "fixture_condition": "recognized C field under explicit provider closing convention", "feature_policy": "closing_label_or_diagnostic_only", "evidence_id": "fd_archive_notes_closing_definition", "is_opening": False, "closing_field_can_be_snapshot_feature": False})
    return pd.DataFrame(rows)


def nested_feasibility(pairs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    executable = pairs[(pairs["market"].eq("1X2")) & pairs["observation_type"].eq("bookmaker") & pairs["valid_paired_rows"].ge(100)].copy()
    stable = executable.groupby(["league", "bookmaker", "snapshot_column_family", "closing_column_family"]).agg(
        usable_seasons=("season", "nunique"), first_season=("season", "min"), last_season=("season", "max"), total_rows=("valid_paired_rows", "sum")
    ).reset_index()
    stable["possible_outer_test_seasons_after_three_prior"] = (stable["usable_seasons"] - 3).clip(lower=0)
    return stable.sort_values(["possible_outer_test_seasons_after_three_prior", "total_rows"], ascending=False), int(stable["possible_outer_test_seasons_after_three_prior"].max()) if len(stable) else 0


def write_report(
    fixtures: pd.DataFrame,
    pairs: pd.DataFrame,
    coverage: pd.DataFrame,
    weekdays: pd.DataFrame,
    ah: pd.DataFrame,
    ou: pd.DataFrame,
    nested: pd.DataFrame,
    outer_seasons: int,
    loaded: pd.DataFrame,
) -> str:
    eligible = fixtures["eligible"].astype(bool)
    one_x_two = int(coverage["verified_snapshot_close_1x2_bookmaker_fixture_rows"].sum())
    ah_rows = int(coverage["verified_snapshot_close_ah_bookmaker_fixture_rows"].sum())
    ou_rows = int(coverage["verified_snapshot_close_ou_bookmaker_fixture_rows"].sum())
    cross = int(coverage["verified_snapshot_close_cross_market_bookmaker_fixture_rows"].sum())
    safe_rows = int(eligible.sum())
    total = len(fixtures)
    nonaggregate_1x2 = pairs[(pairs["market"].eq("1X2")) & pairs["observation_type"].eq("bookmaker")]
    same_bookmaker_rows = int(nonaggregate_1x2["valid_paired_rows"].sum())
    ah_diff = int(coverage["ah_different_line_fixture_rows"].sum())
    ah_same = int(coverage["ah_same_line_fixture_rows"].sum())
    ou_diff = int(coverage["ou_different_line_fixture_rows"].sum())
    ou_same = int(coverage["ou_same_line_fixture_rows"].sum())
    if cross > 0 and outer_seasons >= 2:
        decision = "v4_phase1b_safe_cross_market_snapshot_contract_research_only"
    elif ah_rows > 0 and outer_seasons >= 2:
        decision = "v4_phase1b_safe_1x2_ah_snapshot_contract_research_only"
    elif one_x_two > 0 and outer_seasons >= 2:
        decision = "v4_phase1b_safe_1x2_snapshot_contract_research_only"
    elif one_x_two > 0:
        decision = "v4_phase1b_snapshot_contract_too_partial_for_nested_validation"
    else:
        decision = "v4_phase1b_no_safe_snapshot_contract"

    by_day = fixtures.groupby(["weekday", "eligible"]).size().unstack(fill_value=0)
    day_lines = [f"- {day}: eligible={int(row.get(True, 0))}, excluded={int(row.get(False, 0))}" for day, row in by_day.iterrows()]
    top_nested = nested.head(15)
    nested_lines = [
        f"- {r.league} / {r.bookmaker} / `{r.snapshot_column_family}` -> `{r.closing_column_family}`: {r.usable_seasons} usable seasons, {r.total_rows} rows, {r.possible_outer_test_seasons_after_three_prior} possible outer seasons"
        for r in top_nested.itertuples(index=False)
    ] or ["- None"]
    usable = pairs[(pairs["market"].eq("1X2")) & pairs["observation_type"].eq("bookmaker") & pairs["valid_paired_rows"].ge(100)].copy()
    exact_nested_lines: list[str] = []
    for league, league_rows in usable.groupby("league", sort=True):
        family_scores = league_rows.groupby(["bookmaker", "snapshot_column_family", "closing_column_family"]).agg(
            seasons=("season", "nunique"), rows=("valid_paired_rows", "sum")
        ).sort_values(["seasons", "rows"], ascending=False)
        bookmaker, snapshot_family, closing_family = family_scores.index[0]
        selected = league_rows[
            league_rows["bookmaker"].eq(bookmaker)
            & league_rows["snapshot_column_family"].eq(snapshot_family)
            & league_rows["closing_column_family"].eq(closing_family)
        ]
        seasons = ", ".join(sorted(selected["season"].unique()))
        exact_nested_lines.append(f"- {league}: {bookmaker} `{snapshot_family}` -> `{closing_family}`; usable seasons: {seasons}")
    report = f"""# V4 Phase 1B Scheduled Snapshot Timing Audit

Decision: **{decision}**

Phase 1B only. Non-C football-data odds are not opening odds. No predictive model was built, no betting profit was calculated, no raw file was changed, and frozen V3/V3 Next/paper pipelines were not modified. No confirmed edge is claimed.

## Contract result

The provider documentation supports a fixture-conditional class named `verified_scheduled_prematch_snapshot`:

- Saturday and Sunday fixtures: documented Friday-afternoon batch precedes the fixture date.
- Wednesday and Thursday fixtures: documented Tuesday-afternoon batch precedes the fixture date.
- Friday and Tuesday: excluded because the batch and fixture share a date and the exact collection time is undocumented.
- Monday: excluded because the notes do not clearly say Monday is part of the weekend batch.
- Missing/invalid dates and duplicate logical fixtures: excluded.
- Kickoff time is not used to rescue same-day fixtures. It is not required when a full calendar-day separation exists.
- Reschedule status/original schedule is absent from the raw schema. The audit records this limitation per fixture and does not assert that a fixture was never rescheduled.

## Direct answers

1. **How many 1X2 fixtures have a verified scheduled snapshot and verified close?** {one_x_two} unique canonical fixtures have at least one valid same-bookmaker family pair; {same_bookmaker_rows} is the sum across all executable bookmaker-pair observations and therefore is not a unique-fixture count.
2. **Which seasons and leagues have enough coverage for nested temporal folds?** The machine-readable pair audit applies a conservative threshold of at least 100 valid safe rows per league-season and exact bookmaker family. Stable contracts are summarized below and fully recorded in the pair CSV.
3. **How many outer test seasons are possible?** Up to {outer_seasons} after requiring three earlier usable seasons for the same league/bookmaker/family. This is a feasibility count only; no folds or models were run.
4. **Do the same bookmakers exist at snapshot and close?** Yes for the explicitly mapped families reported as `observation_type=bookmaker`. Avg and Max are retained separately as `consensus_aggregate`, never bookmakers.
5. **Is the safe subset distorted?** Yes. It is structurally restricted to Saturday/Sunday and Wednesday/Thursday, so weekday composition is an explicit selection effect. League-season/day counts are in the weekday CSV.
6. **Can AH line movement be modelled?** Feasibly, as a future Phase 2 research label: {ah_diff} safe paired observations have different lines and {ah_same} have identical lines. Different-line prices are never directly compared.
7. **Can O/U line movement be modelled?** No with the audited football-data schema: {ou_same} safe paired observations are same-line and {ou_diff} are different-line. The local paired totals columns are fixed 2.5 prices, so price discovery can be studied but total-line movement cannot.
8. **Can snapshot-to-close price discovery be studied without calling it opening CLV?** Yes, on the safe subset, using the scheduled-snapshot terminology and exact conventions below.
9. **What fixtures remain excluded?** All Friday, Tuesday, Monday, invalid-date, and duplicate logical fixtures; any pair with absent/asymmetric fields, odds <=1, invalid AH lines, or missing required schema fields; and any source lacking the documented football-data convention.
10. **Is the sample large enough to justify Phase 2?** Yes for research-only scheduled-snapshot-to-close 1X2 and cross-market work under the stable contracts reported here. This does not establish an opening-odds or trading contract.

## Exact counts

- Canonical fixture rows inspected: {total}
- Source league-season files loaded: {len(loaded)}
- Safe timing fixtures: {safe_rows} ({safe_rows / total:.1%})
- Excluded timing fixtures: {total - safe_rows}
- Unique fixtures with valid 1X2 snapshot/close pair: {one_x_two}
- Unique fixtures with valid AH snapshot/close pair: {ah_rows}
- Unique fixtures with valid O/U snapshot/close pair: {ou_rows}
- Unique fixtures with all three markets: {cross}
- Pair audit rows: {len(pairs)}
- AH movement coverage rows: {len(ah)}
- O/U movement coverage rows: {len(ou)}

## Weekday composition

""" + "\n".join(day_lines) + """

## Movement conventions

On every valid eligible 1X2 pair, the contract can build `snapshot_no_vig_prob_home/draw/away`, `closing_no_vig_prob_home/draw/away`, and `probability_shift_home/draw/away`. Executable bookmaker pairs can additionally build selection-specific scheduled-snapshot price CLV, including `price_clv_away`. Aggregate Avg/Max observations support probability-shift diagnostics but are not executable bookmaker prices.

For 1X2, no-vig probabilities are computed independently at each snapshot. The sign convention is:

`probability_shift_away = closing_no_vig_prob_away - snapshot_no_vig_prob_away`

A positive value means the closing market assigned more probability to away. Executable-price CLV is:

`price_clv_away = snapshot_away_odds / closing_away_odds - 1`

A positive value means the scheduled snapshot offered the better away price. These are scheduled-snapshot-to-close labels, not opening CLV.

Synthetic checks: snapshot away odds 2.20 and closing away odds 2.00 give `price_clv_away = +0.10`. For snapshot H/D/A odds `[2.00, 3.50, 4.00]` and closing `[2.10, 3.60, 3.50]`, away no-vig probability moves from 0.241379 to 0.274809, so `probability_shift_away = +0.033430`; the positive sign correctly means movement toward away. These conventions are also locked by unit tests.

For AH, `handicap_line_shift = closing_home_handicap - snapshot_home_handicap`. Negative means the market moved toward the home side (the home handicap became more demanding); positive means movement toward away. Prices are compared only when both exact quarter lines are equal. Quarter lines are never rounded, and no combined line-plus-price scalar target is approved.

For O/U, prices are compared only for the exact same total. Different-line prices would be prohibited. The audited paired schema supplies fixed 2.5 totals and therefore does not support total-line movement labels.

## Nested temporal feasibility examples

""" + "\n".join(nested_lines) + """

## Exact best-family usable seasons by league

These seasons each have at least 100 safe valid same-bookmaker 1X2 pairs. The best-coverage stable family is selected separately per league:

""" + "\n".join(exact_nested_lines) + """

## Limitations

- The provider documentation gives batch weekday/part-of-day, not row-level timestamps.
- The safe class is conditional on fixture weekday and cannot be generalized to excluded weekdays.
- The source has no original schedule, postponement flag, reschedule history, or snapshot timestamp. Identified duplicate logical fixtures are excluded; otherwise reschedule status remains unobservable.
- `Avg` and `Max` are consensus aggregates. Only rows explicitly marked `bookmaker` represent same-bookmaker comparisons.
- This audit determines label/contract feasibility only. It does not calculate the labels into a processed panel and does not evaluate predictive or financial performance.
"""
    (OUT_DIR / "v4_phase1b_report.md").write_text(report, encoding="utf-8")
    decision_text = f"""# V4 Phase 1B Decision

**{decision}**

A conservative scheduled-prematch-snapshot contract exists only for Saturday/Sunday fixtures after the documented Friday batch and Wednesday/Thursday fixtures after the documented Tuesday batch. It is not an opening-odds contract.

Safe unique fixtures: 1X2={one_x_two}, AH={ah_rows}, O/U={ou_rows}, all three={cross}. Maximum feasible outer test seasons after three prior usable seasons: {outer_seasons}.

No model, betting-profit calculation, raw-data edit, or V3/paper-pipeline change was made. No confirmed edge is claimed. Stop after Phase 1B.
"""
    (OUT_DIR / "v4_phase1b_decision.md").write_text(decision_text, encoding="utf-8")
    return decision


def run() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, loaded = load_canonical_frames()
    fixtures = build_fixture_audit(frame)
    specs = build_1x2_pairs(frame.columns) + build_ah_pairs(frame.columns) + build_ou_pairs(frame.columns)
    pairs, any_valid = pair_audit(frame, fixtures, specs)
    coverage = safe_coverage(frame, fixtures, pairs, any_valid)
    weekdays = weekday_coverage(frame, fixtures, any_valid)
    ah = line_movement_audit(frame, fixtures, specs, "Asian Handicap")
    ou = line_movement_audit(frame, fixtures, specs, "Over/Under")
    contract = timing_contract(specs)
    nested, outer_seasons = nested_feasibility(pairs)

    fixtures.to_csv(OUT_DIR / "v4_phase1b_fixture_timing_audit.csv", index=False)
    pairs.to_csv(OUT_DIR / "v4_phase1b_snapshot_close_pairs.csv", index=False)
    coverage.to_csv(OUT_DIR / "v4_phase1b_safe_coverage_by_league_season.csv", index=False)
    weekdays.to_csv(OUT_DIR / "v4_phase1b_weekday_coverage.csv", index=False)
    ah.to_csv(OUT_DIR / "v4_phase1b_ah_line_movement_coverage.csv", index=False)
    ou.to_csv(OUT_DIR / "v4_phase1b_ou_line_movement_coverage.csv", index=False)
    contract.to_csv(OUT_DIR / "v4_phase1b_timing_contract.csv", index=False)
    decision = write_report(fixtures, pairs, coverage, weekdays, ah, ou, nested, outer_seasons, loaded)
    return {
        "decision": decision,
        "fixture_rows_inspected": len(fixtures),
        "safe_timing_fixture_rows": int(fixtures["eligible"].sum()),
        "pair_audit_rows": len(pairs),
        "one_x_two_safe_pair_fixtures": int(coverage["verified_snapshot_close_1x2_bookmaker_fixture_rows"].sum()),
        "ah_safe_pair_fixtures": int(coverage["verified_snapshot_close_ah_bookmaker_fixture_rows"].sum()),
        "ou_safe_pair_fixtures": int(coverage["verified_snapshot_close_ou_bookmaker_fixture_rows"].sum()),
        "cross_market_safe_pair_fixtures": int(coverage["verified_snapshot_close_cross_market_bookmaker_fixture_rows"].sum()),
        "maximum_outer_test_seasons": outer_seasons,
    }
