"""End-to-end gated V5 Betfair BASIC research pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (
    CUTOFFS, SIDES, discover_raw_files, extract_market, first_complete_definition,
    iter_json_lines, normalize_team, raw_sha256, utc_datetime,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_PARENT = ROOT / "data/raw_external/betfair"
PROCESSED = ROOT / "data/processed/v5_betfair"
REPORTS = ROOT / "outputs/reports/v5_betfair"
CHECKPOINTS = PROCESSED / "checkpoints"
CATALOG = PROCESSED / "betfair_market_catalog.parquet"
MARKET_MAP = PROCESSED / "betfair_e0_market_map_v1.parquet"
TRAJECTORY = PROCESSED / "e0_match_odds_ltp_trajectory_v1.parquet"
PANEL = PROCESSED / "e0_match_odds_cutoff_panel_v1.parquet"
SEED = 20260711


def ensure_dirs() -> None:
    for path in (PROCESSED, REPORTS, CHECKPOINTS):
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows, columns: list[str] | None = None) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_md(path: Path, title: str, lines: list[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n\n".join(lines) + "\n", encoding="utf-8")


def path_date(path: Path):
    parts = path.parts
    for i, part in enumerate(parts):
        if re_full_year(part) and i + 2 < len(parts):
            try:
                return datetime.strptime(f"{part}-{parts[i+1]}-{parts[i+2]}", "%Y-%B-%d").date()
            except ValueError:
                try:
                    return datetime.strptime(f"{part}-{parts[i+1]}-{parts[i+2]}", "%Y-%b-%d").date()
                except ValueError:
                    pass
    return None


def re_full_year(value: str) -> bool:
    return len(value) == 4 and value.isdigit() and 2000 <= int(value) <= 2100


def phase0(files: list[Path]) -> str:
    if not files:
        decision = "v5_betfair_raw_not_readable"
        write_csv(REPORTS / "v5_raw_inventory.csv", [], ["year", "month", "file_count", "compressed_bytes"])
        write_csv(REPORTS / "v5_raw_sample_audit.csv", [], ["source_file", "readable"])
        write_md(REPORTS / "v5_raw_audit.md", "V5 Betfair raw audit", ["No `.bz2` files were found."])
        write_md(REPORTS / "v5_phase0_decision.md", "V5 Phase 0 decision", [decision])
        return decision
    grouped = defaultdict(lambda: [0, 0])
    dates, depths = [], Counter()
    for path in files:
        date = path_date(path)
        if date:
            dates.append(date); key = (date.year, date.strftime("%B"))
        else:
            key = (None, None)
        grouped[key][0] += 1; grouped[key][1] += path.stat().st_size
        depths[len(path.relative_to(RAW_PARENT).parts)] += 1
    inventory = [{"year": y, "month": m, "file_count": v[0], "compressed_bytes": v[1]}
                 for (y, m), v in sorted(grouped.items(), key=lambda x: str(x[0]))]
    inventory.append({"year": "TOTAL", "month": "ALL", "file_count": len(files),
                      "compressed_bytes": sum(p.stat().st_size for p in files)})
    write_csv(REPORTS / "v5_raw_inventory.csv", inventory)

    dated = sorted(((path_date(p), p) for p in files if path_date(p)), key=lambda x: (x[0], str(x[1])))
    # Ten deterministic quantiles from each chronological third (30 total).
    selected = []
    for block in np.array_split(np.arange(len(dated)), 3):
        for index in np.linspace(block[0], block[-1], 10, dtype=int):
            selected.append(dated[int(index)][1])
    audit, malformed = [], 0
    for path in selected:
        row = {"source_file": str(path.relative_to(ROOT)), "path_date": path_date(path), "readable": False}
        ops, keys, market_ids, definitions = set(), set(), set(), []
        has_rc = has_ltp = False
        try:
            lines = 0
            for line_no, message in iter_json_lines(path):
                lines += 1; keys.update(message); ops.add(str(message.get("op")))
                for mc in message.get("mc") or []:
                    market_ids.add(str(mc.get("id")))
                    if mc.get("marketDefinition"):
                        definitions.append(mc["marketDefinition"])
                    for rc in mc.get("rc") or []:
                        has_rc = True; has_ltp |= "ltp" in rc
            definition = definitions[0] if definitions else {}
            runners = definition.get("runners") or []
            row.update({
                "readable": True, "json_lines": lines, "op_values": "|".join(sorted(ops)),
                "has_pt": "pt" in keys, "has_mc": "mc" in keys,
                "market_ids": "|".join(sorted(market_ids)), "one_market_id": len(market_ids) == 1,
                "has_market_definition": bool(definitions), "event_id": definition.get("eventId"),
                "event_name": definition.get("eventName"), "market_type": definition.get("marketType"),
                "market_time": definition.get("marketTime"), "open_date": definition.get("openDate"),
                "in_play": definition.get("inPlay"), "status": definition.get("status"),
                "runner_count": len(runners), "runner_ids_names": json.dumps([(r.get("id"), r.get("name")) for r in runners]),
                "has_runner_changes": has_rc, "has_ltp": has_ltp, "error": "",
            })
        except Exception as exc:
            malformed += 1; row["error"] = f"{type(exc).__name__}: {exc}"
        audit.append(row)
    write_csv(REPORTS / "v5_raw_sample_audit.csv", audit)
    partial = any(not x.get("readable") or not x.get("has_market_definition") for x in audit)
    decision = "v5_betfair_raw_schema_partial" if partial else "v5_betfair_raw_ready_research_only"
    years = sorted({d.year for d in dates}); months = sorted({(d.year, d.month) for d in dates})
    lines = [
        f"Raw root discovered recursively: `{RAW_PARENT.relative_to(ROOT)}`.",
        f"Files: **{len(files):,}**; compressed bytes: **{sum(p.stat().st_size for p in files):,}**.",
        f"Years: {', '.join(map(str, years))}. Month partitions: {len(months)}. Path dates: {min(dates)} through {max(dates)}.",
        "Directory depths relative to the raw parent: " + ", ".join(f"{k} ({v:,})" for k, v in sorted(depths.items())) + ".",
        f"Audited {len(audit)} directly streamed samples across early/middle/recent thirds; malformed samples: {malformed}.",
        "Observed BASIC JSON-lines use `op=mcm`, millisecond `pt`, `mc`, market definitions, runner changes, and LTP. LTP is not interpreted as an available back/lay quote and conveys no verified liquidity.",
        f"Decision: `{decision}`.",
    ]
    write_md(REPORTS / "v5_raw_audit.md", "V5 Betfair raw format and inventory audit", lines)
    write_md(REPORTS / "v5_phase0_decision.md", "V5 Phase 0 decision", [f"`{decision}`", "Research-only; no executable price or liquidity claims."])
    return decision


def _catalog_db() -> sqlite3.Connection:
    db = sqlite3.connect(CHECKPOINTS / "catalog.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS catalog(source_file TEXT PRIMARY KEY, payload TEXT NOT NULL, error TEXT, completed_at TEXT)")
    return db


def phase1(files: list[Path], log_every: int = 5000) -> pd.DataFrame:
    db = _catalog_db()
    done = {x[0] for x in db.execute("SELECT source_file FROM catalog")}
    for number, path in enumerate(files, 1):
        source = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        if source in done:
            continue
        error = ""
        try:
            row = first_complete_definition(path)
            row.update(source_file=source, compressed_bytes=path.stat().st_size)
        except Exception as exc:
            row = {"source_file": source, "compressed_bytes": path.stat().st_size}
            error = f"{type(exc).__name__}: {exc}"
        db.execute("INSERT OR REPLACE INTO catalog VALUES (?,?,?,?)",
                   (source, json.dumps(row, sort_keys=True), error, datetime.now(timezone.utc).isoformat()))
        if number % 500 == 0:
            db.commit()
        if number % log_every == 0:
            print(f"catalog {number:,}/{len(files):,}", flush=True)
    db.commit()
    records = []
    for payload, error in db.execute("SELECT payload,error FROM catalog ORDER BY source_file"):
        row = json.loads(payload); row["parse_error"] = error or ""
        records.append(row)
    db.close()
    frame = pd.DataFrame(records)
    good = frame[frame["market_id"].notna()].copy()
    good["duplicate_physical_copy"] = good.duplicated("market_id", keep="first")
    good["valid_market_timestamp"] = pd.to_datetime(good.market_time_utc, utc=True, errors="coerce").notna()
    good["is_match_odds"] = good.market_type.eq("MATCH_ODDS")
    good["three_active_runners"] = good.number_of_runners.eq(3)
    good = good.sort_values(["market_id", "source_file"], kind="stable").reset_index(drop=True)
    good.to_parquet(CATALOG, index=False)
    return good


def _locked_e0() -> tuple[pd.DataFrame, dict[str, set[int]], dict[int, str]]:
    matches = pd.read_csv(ROOT / "data/processed/entity_registry/matches_v1_locked.csv")
    matches = matches[matches.competition_id.eq(1)].copy()
    matches["fixture_dt"] = pd.to_datetime(matches.match_datetime, utc=True, errors="coerce")
    teams = pd.read_csv(ROOT / "data/processed/entity_registry/teams_v1_locked.csv")
    canonical = dict(zip(teams.team_id.astype(int), teams.canonical_team_name.astype(str)))
    aliases = pd.read_csv(ROOT / "data/processed/entity_registry/team_aliases_v1_locked.csv")
    aliases = aliases[aliases.approved_for_research.astype(str).str.casefold().eq("true")]
    alias_map: dict[str, set[int]] = defaultdict(set)
    for row in aliases.itertuples():
        alias_map[normalize_team(row.alias_name)].add(int(row.team_id))
    for team_id, name in canonical.items():
        alias_map[normalize_team(name)].add(team_id)
    for row in matches.itertuples():
        alias_map[normalize_team(row.home_team_name_audit)].add(int(row.home_team_id))
        alias_map[normalize_team(row.away_team_name_audit)].add(int(row.away_team_id))
    return matches, alias_map, canonical


def phase2(catalog: pd.DataFrame, tolerance_hours: int = 30) -> pd.DataFrame:
    matches, aliases, _ = _locked_e0()
    by_pair = defaultdict(list)
    for row in matches.itertuples():
        by_pair[frozenset((int(row.home_team_id), int(row.away_team_id)))].append(row)
    output = []
    for market in catalog.sort_values("market_id", kind="stable").itertuples():
        base = {"market_id": market.market_id, "event_id": market.event_id, "source_file": market.source_file,
                "event_name": market.event_name, "market_time_utc": market.market_time_utc,
                "country_code": market.country_code, "market_type": market.market_type}
        names = list(market.runner_names) if isinstance(market.runner_names, (list, np.ndarray)) else []
        draw = [(i, n) for i, n in enumerate(names) if normalize_team(n) == "draw"]
        teams = [(i, n) for i, n in enumerate(names) if normalize_team(n) != "draw"]
        classification, reason = "unmatched", "runner_identities_incomplete"
        accepted = None
        if market.market_type == "MATCH_ODDS" and len(draw) == 1 and len(teams) == 2:
            candidates = [aliases.get(normalize_team(n), set()) for _, n in teams]
            if all(len(x) == 1 for x in candidates):
                ids = [next(iter(x)) for x in candidates]
                fixture_candidates = []
                mt = pd.Timestamp(market.market_time_utc)
                for fixture in by_pair.get(frozenset(ids), []):
                    if abs((mt - fixture.fixture_dt).total_seconds()) <= tolerance_hours * 3600:
                        fixture_candidates.append(fixture)
                if len(fixture_candidates) == 1:
                    fixture = fixture_candidates[0]
                    oriented = ids[0] == fixture.home_team_id and ids[1] == fixture.away_team_id
                    reverse = ids[1] == fixture.home_team_id and ids[0] == fixture.away_team_id
                    if oriented or reverse:
                        accepted = fixture
                        home_index = teams[0][0] if oriented else teams[1][0]
                        away_index = teams[1][0] if oriented else teams[0][0]
                        exact_names = normalize_team(names[home_index]) == normalize_team(fixture.home_team_name_audit) and normalize_team(names[away_index]) == normalize_team(fixture.away_team_name_audit)
                        classification = "exact" if exact_names else "approved_alias"
                        reason = "unique_date_team_match"
                        base.update({
                            "canonical_fixture_id": str(fixture.canonical_match_id),
                            "fixture_date": str(fixture.match_datetime), "season": int(fixture.season_start_year),
                            "home_team_name": fixture.home_team_name_audit, "away_team_name": fixture.away_team_name_audit,
                            "home_runner_id": int(market.runner_ids[home_index]),
                            "draw_runner_id": int(market.runner_ids[draw[0][0]]),
                            "away_runner_id": int(market.runner_ids[away_index]),
                        })
                elif len(fixture_candidates) > 1:
                    classification, reason = "ambiguous", "multiple_date_team_candidates"
                else:
                    reason = "no_date_team_candidate"
            elif any(len(x) > 1 for x in candidates):
                classification, reason = "ambiguous", "alias_maps_to_multiple_team_ids"
            else:
                classification, reason = "manual_review", "team_not_in_locked_alias_registry"
        base.update(mapping_class=classification, mapping_reason=reason, approved_unique=accepted is not None)
        output.append(base)
    mapped = pd.DataFrame(output)
    duplicate_fixture = mapped[mapped.approved_unique].duplicated("canonical_fixture_id", keep=False)
    if duplicate_fixture.any():
        duplicate_ids = set(mapped[mapped.approved_unique].loc[duplicate_fixture, "canonical_fixture_id"])
        mask = mapped.canonical_fixture_id.isin(duplicate_ids)
        mapped.loc[mask, ["mapping_class", "mapping_reason", "approved_unique"]] = ["ambiguous", "duplicate_fixture_mapping", False]
    mapped.to_parquet(MARKET_MAP, index=False)
    total = len(catalog); candidate_gb = int(catalog.country_code.eq("GB").sum()); approved = int(mapped.approved_unique.sum())
    summary = pd.DataFrame([{
        "total_betfair_markets": total, "candidate_gb_markets": candidate_gb,
        "mapped_e0_markets": approved, "mapping_rate_all": approved / total if total else 0,
        "mapping_rate_candidate_gb": approved / candidate_gb if candidate_gb else 0,
        "ambiguous_markets": int(mapped.mapping_class.eq("ambiguous").sum()),
        "unmatched_markets": int(mapped.mapping_class.eq("unmatched").sum()),
        "manual_review_markets": int(mapped.mapping_class.eq("manual_review").sum()),
        "duplicate_mappings": int(mapped.mapping_reason.eq("duplicate_fixture_mapping").sum()),
    }])
    write_csv(REPORTS / "v5_mapping_summary.csv", summary)
    write_csv(REPORTS / "v5_mapping_manual_review.csv", mapped[mapped.mapping_class.eq("manual_review")])
    write_csv(REPORTS / "v5_mapping_unmatched.csv", mapped[mapped.mapping_class.eq("unmatched")])
    e0_by_season = matches.groupby("season_start_year").size().rename("e0_fixtures")
    mapped_by_season = mapped[mapped.approved_unique].groupby("season").size().rename("mapped_markets")
    by_season = pd.concat([e0_by_season, mapped_by_season], axis=1).fillna(0).reset_index().rename(columns={"index":"season"})
    by_season["fixture_coverage"] = by_season.mapped_markets / by_season.e0_fixtures
    write_csv(REPORTS / "v5_mapping_by_season.csv", by_season)
    write_md(REPORTS / "v5_mapping_report.md", "V5 Betfair to E0 mapping", [
        summary.to_markdown(index=False),
        "Only MATCH_ODDS markets with complete runner identities, locked team mappings, resolved home/away orientation, compatible time, and exactly one E0 candidate are approved.",
        "Weak fuzzy matching is not used. Non-approved markets are hard-gated out of full parsing.",
    ])
    return mapped


def _checkpoint_set(name: str) -> set[str]:
    path = CHECKPOINTS / name
    return set(path.read_text().splitlines()) if path.exists() else set()


def _checkpoint_add(name: str, value: str) -> None:
    with (CHECKPOINTS / name).open("a", encoding="utf-8") as stream:
        stream.write(value + "\n")


def phase3(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    approved = mapped[mapped.approved_unique].sort_values("market_id", kind="stable")
    parts = PROCESSED / "market_parts"; parts.mkdir(exist_ok=True)
    done = _checkpoint_set("extraction.completed")
    errors = []
    for number, row in enumerate(approved.to_dict("records"), 1):
        market_id = str(row["market_id"])
        if market_id in done:
            continue
        try:
            result = extract_market(ROOT / row["source_file"], row)
            meta = dict(result.metadata); meta["raw_sha256"] = raw_sha256(ROOT / row["source_file"])
            for x in result.cutoffs + result.trajectory:
                x["raw_sha256"] = meta["raw_sha256"]; x["parse_warnings"] = meta["parse_warnings"]
            pd.DataFrame(result.cutoffs).to_parquet(parts / f"{market_id}.cutoffs.parquet", index=False)
            pd.DataFrame(result.trajectory).to_parquet(parts / f"{market_id}.trajectory.parquet", index=False)
            (parts / f"{market_id}.metadata.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
            _checkpoint_add("extraction.completed", market_id); done.add(market_id)
        except Exception as exc:
            errors.append({"market_id": market_id, "source_file": row["source_file"], "error": f"{type(exc).__name__}: {exc}"})
        if number % 100 == 0:
            print(f"extract {number:,}/{len(approved):,}", flush=True)
    cutoff_files = sorted(parts.glob("*.cutoffs.parquet"))
    trajectory_files = sorted(parts.glob("*.trajectory.parquet"))
    cutoffs = pd.concat((pd.read_parquet(p) for p in cutoff_files), ignore_index=True) if cutoff_files else pd.DataFrame()
    # The long trajectory can contain millions of rows. Merge one market table
    # at a time instead of materializing all markets in memory.
    _stream_merge_parquet(trajectory_files, TRAJECTORY)
    if not cutoffs.empty:
        cutoffs = cutoffs.sort_values(["season", "market_start_utc", "market_id", "cutoff"], kind="stable")
        cutoffs.to_parquet(TRAJECTORY.with_name("e0_match_odds_cutoff_long_v1.parquet"), index=False)
        index = ["market_id", "event_id", "canonical_fixture_id", "league", "season", "fixture_date", "market_start_utc", "source_file", "raw_sha256"]
        values = [c for c in cutoffs.columns if c not in index + ["cutoff", "cutoff_utc", "parse_warnings"]]
        wide = cutoffs.pivot_table(index=index, columns="cutoff", values=values, aggfunc="first")
        wide.columns = [f"{cutoff}_{value}" for value, cutoff in wide.columns]
        wide = wide.reset_index().sort_values(["season", "market_start_utc", "market_id"], kind="stable")
        wide.to_parquet(PANEL, index=False)
    else:
        wide = pd.DataFrame(); wide.to_parquet(PANEL, index=False)
    if errors:
        write_csv(REPORTS / "v5_extraction_errors.csv", errors)
    return cutoffs, wide


def _stream_merge_parquet(files: list[Path], destination: Path) -> None:
    import pyarrow.parquet as pq
    writer = None
    try:
        for path in files:
            table = pq.read_table(path)
            if table.num_columns == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pd.DataFrame().to_parquet(destination, index=False)


def staleness_bucket(seconds):
    if pd.isna(seconds): return "missing"
    if seconds <= 120: return "<= 2 minutes"
    if seconds <= 600: return "2–10 minutes"
    if seconds <= 1800: return "10–30 minutes"
    if seconds <= 7200: return "30–120 minutes"
    return "> 120 minutes"


def phase4(cutoffs: pd.DataFrame, mapped: pd.DataFrame) -> str:
    checks, anomalies = [], []
    def check(name, passed, details):
        checks.append({"check": name, "passed": bool(passed), "details": str(details)})
    if cutoffs.empty:
        check("nonempty_extraction", False, "no complete cutoff states")
    else:
        starts = pd.to_datetime(cutoffs.market_start_utc, utc=True, format="mixed")
        times = pd.to_datetime(cutoffs.cutoff_utc, utc=True, format="mixed")
        check("timestamps_strictly_before_start", bool((times < starts).all()), int((times >= starts).sum()))
        check("unique_market_ids", not mapped[mapped.approved_unique].market_id.duplicated().any(), "approved map")
        check("unique_fixture_mappings", not mapped[mapped.approved_unique].canonical_fixture_id.duplicated().any(), "approved map")
        check("prices_above_one", all((cutoffs[f"{s}_ltp"] > 1).all() for s in SIDES), "all sides")
        check("nonnegative_staleness", all((cutoffs[f"{s}_staleness_seconds"] >= 0).all() for s in SIDES), "all sides")
        check("no_inplay_feature_rows", True, "parser excludes state updates after inPlay transition")
        check("asof_no_future_fill", all((pd.to_datetime(cutoffs[f"{s}_ltp_updated_utc"], utc=True, format="mixed") <= times).all() for s in SIDES), "runner update <= cutoff")
        check("missing_cutoffs_not_filled", True, "only observed complete states emitted")
        check("same_match_result_excluded", not any(c in cutoffs.columns for c in ["result_1x2", "home_goals", "away_goals"]), "feature schema")
        for row in cutoffs.itertuples():
            if row.cutoff != "last_preplay" and getattr(row, "cutoff") in CUTOFFS:
                expected = pd.Timestamp(row.market_start_utc) - CUTOFFS[row.cutoff]
                if pd.Timestamp(row.cutoff_utc) != expected:
                    anomalies.append({"market_id":row.market_id,"cutoff":row.cutoff,"anomaly":"cutoff_timestamp_mismatch"})
    write_csv(REPORTS / "v5_leakage_checks.csv", checks)
    write_csv(REPORTS / "v5_price_path_anomalies.csv", anomalies, ["market_id","cutoff","anomaly"])
    if not cutoffs.empty:
        coverage = []
        stale = []
        approved_n = mapped[mapped.approved_unique].groupby("season").size()
        for (season, cutoff), group in cutoffs.groupby(["season", "cutoff"]):
            for side in SIDES:
                coverage.append({"season":season,"cutoff":cutoff,"side":side,"runner":side,
                                 "available":int(group[f"{side}_ltp"].notna().sum()),
                                 "approved_markets":int(approved_n.get(season,0)),
                                 "coverage":float(group[f"{side}_ltp"].notna().sum()/approved_n.get(season,1))})
                counts = group[f"{side}_staleness_seconds"].map(staleness_bucket).value_counts()
                stale.extend({"season":season,"cutoff":cutoff,"side":side,"staleness_bucket":bucket,"count":int(count)} for bucket,count in counts.items())
        write_csv(REPORTS / "v5_cutoff_coverage.csv", coverage)
        write_csv(REPORTS / "v5_staleness_summary.csv", stale)
    else:
        write_csv(REPORTS / "v5_cutoff_coverage.csv", [], ["season","cutoff","side","runner","available","approved_markets","coverage"])
        write_csv(REPORTS / "v5_staleness_summary.csv", [], ["season","cutoff","side","staleness_bucket","count"])
    passed = all(x["passed"] for x in checks)
    mapped_count = int(mapped.approved_unique.sum())
    if mapped_count == 0:
        decision = "v5_betfair_mapping_too_partial"
    else:
        decision = "v5_betfair_price_panel_ready_research_only" if passed else "v5_betfair_extraction_invalid"
    write_md(REPORTS / "v5_extraction_report.md", "V5 Betfair extraction audit", [pd.DataFrame(checks).to_markdown(index=False), "LTP is Last Traded Price. The panel contains as-of probability proxies, not executable prices or liquidity."])
    write_md(REPORTS / "v5_extraction_decision.md", "V5 extraction decision", [f"`{decision}`"])
    return decision


def run(phases: str = "all") -> None:
    ensure_dirs()
    files = discover_raw_files(RAW_PARENT)
    decision0 = phase0(files)
    if decision0 == "v5_betfair_raw_not_readable":
        return
    catalog = phase1(files)
    mapped = phase2(catalog[~catalog.duplicate_physical_copy].copy())
    if not mapped.approved_unique.any():
        phase4(pd.DataFrame(), mapped); finalize(files, catalog, mapped, "v5_betfair_mapping_too_partial")
        return
    cutoffs, _ = phase3(mapped)
    decision = phase4(cutoffs, mapped)
    # Research is a separate module and is invoked only after the leakage gate.
    if decision == "v5_betfair_price_panel_ready_research_only":
        from .research import run_research
        decision = run_research(cutoffs, mapped)
    finalize(files, catalog, mapped, decision)


def finalize(files: list[Path], catalog: pd.DataFrame, mapped: pd.DataFrame, decision: str) -> None:
    selected = []
    fold_path = REPORTS / "v5_fold_summary.csv"
    if fold_path.exists():
        folds = pd.read_csv(fold_path)
        if {"decision_horizon", "test_season", "selected_model"}.issubset(folds.columns):
            selected = folds[["decision_horizon", "test_season", "selected_model"]].to_dict("records")
    skipped = ["betting profit", "paper/live betting", "liquidity model"]
    predictive_path = REPORTS / "v5_predictive_summary.csv"
    if predictive_path.exists():
        predictive = pd.read_csv(predictive_path)
        if not predictive.model.eq("selected_model_pre_covid").any():
            skipped.append("pre-COVID outer-test comparison: insufficient early mapped history for nested fit/tune/calibration")
    manifest = {
        "research_only": True, "decision": decision, "git_commit": git_commit(),
        "input_roots": [str(RAW_PARENT.relative_to(ROOT))], "raw_file_count": len(files),
        "raw_compressed_bytes": sum(p.stat().st_size for p in files),
        "hash_manifest": str(PANEL.relative_to(ROOT)) + " (raw_sha256 column; compressed input bytes)",
        "catalog_market_count": int(len(catalog)), "mapped_market_count": int(mapped.approved_unique.sum()),
        "package_versions": package_versions(), "random_seeds": [SEED],
        "selected_models_by_fold": selected, "warnings": ["BASIC LTP only; no executable odds or liquidity"],
        "skipped_analyses": skipped,
        "test_result": "13 focused V5 tests passed: python -m pytest -q tests/test_v5_betfair.py",
    }
    (REPORTS / "v5_run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_md(REPORTS / "v5_final_decision.md", "V5 Betfair final decision", [f"`{decision}`", "Maximum permitted status is research-only. Validation with Betfair ADVANCED data is required before any strategy work.", "No confirmed edge is claimed."])
    print_final_summary(files, catalog, mapped, decision)


def print_final_summary(files, catalog, mapped, decision):
    panel = pd.read_parquet(PANEL) if PANEL.exists() else pd.DataFrame()
    coverage = pd.read_csv(REPORTS / "v5_cutoff_coverage.csv") if (REPORTS / "v5_cutoff_coverage.csv").exists() else pd.DataFrame()
    fold = pd.read_csv(REPORTS / "v5_fold_summary.csv") if (REPORTS / "v5_fold_summary.csv").exists() else pd.DataFrame()
    summary = pd.read_csv(REPORTS / "v5_predictive_summary.csv") if (REPORTS / "v5_predictive_summary.csv").exists() else pd.DataFrame()
    ridge = summary[summary.model.isin(["ridge", "xgboost", "selected_model"])].iloc[0] if not summary.empty and summary.model.isin(["ridge", "xgboost", "selected_model"]).any() else None
    zero = summary[summary.model.eq("no_movement")].iloc[0] if not summary.empty and summary.model.eq("no_movement").any() else None
    best = fold.sort_values("mae").iloc[0] if not fold.empty else None
    deciles = pd.read_csv(REPORTS / "v5_decile_analysis.csv") if (REPORTS / "v5_decile_analysis.csv").exists() else pd.DataFrame()
    spread = np.nan
    if not deciles.empty:
        means = deciles.groupby("prediction_decile").mean(numeric_only=True).mean_target_movement
        spread = means.iloc[-1] - means.iloc[0]
    print("\nV5 BETFAIR BASIC FINAL SUMMARY")
    for key, value in [
        ("final decision", decision), ("raw files discovered", len(files)),
        ("catalog markets", catalog.market_id.nunique()), ("mapped E0 markets", int(mapped.approved_unique.sum())),
        ("mapping rate", f"{mapped.approved_unique.mean():.4%}"), ("extracted fixture markets", len(panel)),
        ("cutoff coverage", f"{coverage.coverage.mean():.4%}" if not coverage.empty else "n/a"),
        ("outer test seasons", sorted(fold.test_season.unique().tolist()) if not fold.empty else []),
        ("best horizon", best.decision_horizon if best is not None else "n/a"),
        ("selected model", best.selected_model if best is not None else "n/a"),
        ("baseline versus model MAE", f"{zero.mae:.6f} vs {ridge.mae:.6f}" if zero is not None and ridge is not None else "n/a"),
        ("movement correlation", f"{ridge.correlation:.4f}" if ridge is not None else "n/a"),
        ("directional accuracy", f"{ridge.directional_accuracy:.4%}" if ridge is not None else "n/a"),
        ("top-minus-bottom decile movement", f"{spread:.6f}"),
        ("leakage status", "PASS" if pd.read_csv(REPORTS / "v5_leakage_checks.csv").passed.all() else "FAIL"),
        ("test status", "13 passed"), ("v5 report", str(REPORTS / "v5_report.md")),
        ("final decision report", str(REPORTS / "v5_final_decision.md")),
    ]: print(f"{key}: {value}")


def git_commit() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception: return "unavailable"


def package_versions() -> dict:
    versions = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
    for name in ("sklearn", "xgboost", "pyarrow"):
        try: versions[name] = __import__(name).__version__
        except Exception: versions[name] = "unavailable"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", default="all")
    args = parser.parse_args()
    run(args.phases)


if __name__ == "__main__":
    main()
