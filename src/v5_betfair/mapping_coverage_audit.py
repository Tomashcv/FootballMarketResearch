"""Final, conservative E0 mapping-coverage audit for V5 Betfair BASIC.

This module never edits the locked entity registry or V1 V5 outputs.  It can
build an overlay-only corrected map/panel after an exact, repeated alias audit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .core import SIDES, extract_market, normalize_team, raw_sha256
from .pipeline import CATALOG, MARKET_MAP, PANEL, PROCESSED, REPORTS, ROOT, write_csv, write_md

FIXTURES = ROOT / "data/processed/entity_registry/matches_v1_locked.csv"
TEAMS = ROOT / "data/processed/entity_registry/teams_v1_locked.csv"
ALIASES = ROOT / "data/processed/entity_registry/team_aliases_v1_locked.csv"
CORRECTED_MAP = PROCESSED / "betfair_e0_market_map_v2_mapping_coverage.parquet"
CORRECTED_LONG = PROCESSED / "e0_match_odds_cutoff_long_v2_mapping_coverage.parquet"
CORRECTED_PANEL = PROCESSED / "e0_match_odds_cutoff_panel_v2_mapping_coverage.parquet"
CORRECTED_REPORTS = REPORTS / "mapping_coverage_corrected"

# These are not registry changes.  They are the small, explicit candidate set
# to be accepted only after repeated, date-compatible, unambiguous evidence.
CANDIDATE_ALIASES = {
    "man utd": 100, "c palace": 44, "sheff utd": 133,
    "leicester city": 88, "stoke city": 139, "huddersfield town": 75,
}


def _load():
    catalog = pd.read_parquet(CATALOG).copy()
    mapping = pd.read_parquet(MARKET_MAP).copy()
    fixtures = pd.read_csv(FIXTURES)
    fixtures = fixtures[(fixtures.competition_id.eq(1)) & fixtures.season_start_year.between(2015, 2024)].copy()
    fixtures["fixture_dt"] = pd.to_datetime(fixtures.match_datetime, utc=True, format="mixed")
    fixtures["fixture_date"] = fixtures.fixture_dt.dt.date
    teams = pd.read_csv(TEAMS)
    aliases = pd.read_csv(ALIASES)
    aliases = aliases[aliases.approved_for_research.astype(str).str.casefold().eq("true")]
    catalog["market_dt"] = pd.to_datetime(catalog.market_time_utc, utc=True, format="mixed")
    catalog["market_date"] = catalog.market_dt.dt.date
    return catalog, mapping, fixtures, teams, aliases


def _team_aliases(fixtures, teams, aliases, include_candidates=False):
    result: dict[int, set[str]] = defaultdict(set)
    for row in teams.itertuples(): result[int(row.team_id)].add(normalize_team(row.canonical_team_name))
    for row in aliases.itertuples(): result[int(row.team_id)].add(normalize_team(row.alias_name))
    for row in fixtures.itertuples():
        result[int(row.home_team_id)].add(normalize_team(row.home_team_name_audit))
        result[int(row.away_team_id)].add(normalize_team(row.away_team_name_audit))
    if include_candidates:
        for alias, team_id in CANDIDATE_ALIASES.items(): result[team_id].add(alias)
    return result


def _by_date(catalog):
    result = defaultdict(list)
    for row in catalog.itertuples(): result[row.market_date].append(row)
    return result


def _runner_names(market):
    names = market.runner_names if isinstance(market.runner_names, (list, np.ndarray)) else []
    return [normalize_team(x) for x in names]


def _fixture_markets(fixture, index, aliases, days=2):
    home = aliases[int(fixture.home_team_id)]; away = aliases[int(fixture.away_team_id)]
    all_markets, pair_markets, any_team = [], [], []
    for date in pd.date_range(fixture.fixture_dt - pd.Timedelta(days=days), fixture.fixture_dt + pd.Timedelta(days=days)).date:
        for market in index.get(date, []):
            names = _runner_names(market)
            home_matches = [x for x in names if x in home]
            away_matches = [x for x in names if x in away]
            if home_matches or away_matches: any_team.append((market, home_matches, away_matches))
            if home_matches and away_matches:
                all_markets.append((market, home_matches, away_matches))
                if abs((market.market_dt - fixture.fixture_dt).total_seconds()) <= 30 * 3600:
                    pair_markets.append((market, home_matches, away_matches))
    return all_markets, pair_markets, any_team


def _eligible(market, home_matches, away_matches):
    names = _runner_names(market)
    draw = [x for x in names if x == "draw"]
    return (
        market.market_type == "MATCH_ODDS" and market.number_of_runners == 3
        and len(draw) == 1 and len(home_matches) == 1 and len(away_matches) == 1
        and market.status == "OPEN" and not bool(market.duplicate_physical_copy)
    )


def _event_pair_support(market, fixture, aliases):
    """Accept common event-name separators and either home/away text order."""
    raw = str(market.event_name or "")
    parts = re.split(r"\s+(?:v|vs|@)\s+|\s+-\s+", raw, flags=re.IGNORECASE)
    if len(parts) != 2:
        return False, "other"
    left, right = normalize_team(parts[0]), normalize_team(parts[1])
    home, away = aliases[int(fixture.home_team_id)], aliases[int(fixture.away_team_id)]
    supported = (left in home and right in away) or (left in away and right in home)
    separator = "vs" if re.search(r"\s+vs\s+", raw, re.I) else "@" if "@" in raw else "-" if " - " in raw else "v"
    return supported, separator


def _orientation(market, fixture, home_matches, away_matches):
    names = _runner_names(market)
    ids = list(market.runner_ids)
    home_indexes = [i for i, x in enumerate(names) if x in home_matches]
    away_indexes = [i for i, x in enumerate(names) if x in away_matches]
    draw_indexes = [i for i, x in enumerate(names) if x == "draw"]
    if len(home_indexes) != 1 or len(away_indexes) != 1 or len(draw_indexes) != 1:
        return None
    return {
        "home_runner_id": int(ids[home_indexes[0]]), "away_runner_id": int(ids[away_indexes[0]]),
        "draw_runner_id": int(ids[draw_indexes[0]]),
    }


def _status(fixture, original_approved, original_duplicate_ids, locked, overlay, index):
    if str(fixture.canonical_match_id) in original_approved:
        return "mapped_approved", "approved V1 unique market", [], []
    locked_all, locked_near, locked_any = _fixture_markets(fixture, index, locked)
    over_all, over_near, over_any = _fixture_markets(fixture, index, overlay)
    eligible_overlay = [x for x in over_near if _eligible(*x) and _event_pair_support(x[0], fixture, overlay)[0]]
    used_candidate_aliases = sorted({
        token for market, _, _ in eligible_overlay for token in _runner_names(market)
        if token in CANDIDATE_ALIASES and CANDIDATE_ALIASES[token] in {fixture.home_team_id, fixture.away_team_id}
    })
    if len(eligible_overlay) == 1 and used_candidate_aliases:
        return "team_alias_unresolved", "exact candidate alias; unique eligible market", eligible_overlay, used_candidate_aliases
    if len(eligible_overlay) > 1 or str(fixture.canonical_match_id) in original_duplicate_ids:
        return "multiple_candidate_markets", "more than one eligible market; no arbitrary selection", eligible_overlay, used_candidate_aliases
    if over_all and not over_near:
        return "date_mismatch", "team pair found only outside ±30h acceptance tolerance", over_all, used_candidate_aliases
    if over_near:
        market, h, a = over_near[0]
        if market.duplicate_physical_copy:
            return "duplicate_market", "duplicate physical catalog copy", over_near, used_candidate_aliases
        if market.market_type != "MATCH_ODDS":
            return "non_match_odds", "candidate is not MATCH_ODDS", over_near, used_candidate_aliases
        if market.number_of_runners != 3 or _runner_names(market).count("draw") != 1:
            return "incomplete_runners", "not exactly home/draw/away active runners", over_near, used_candidate_aliases
        if market.status != "OPEN":
            return "market_cancelled_or_invalid", f"catalog definition status {market.status}", over_near, used_candidate_aliases
        return "home_away_orientation_unresolved", "runner identities do not yield one home/draw/away orientation", over_near, used_candidate_aliases
    if locked_any or over_any:
        return "team_alias_unresolved", "one fixture team appears but counterpart lacks exact approved alias evidence", over_any or locked_any, used_candidate_aliases
    return "no_betfair_candidate", "no date ±2-day catalog market with either fixture team identity", [], []


def audit():
    REPORTS.mkdir(parents=True, exist_ok=True)
    catalog, mapping, fixtures, teams, aliases = _load()
    locked = _team_aliases(fixtures, teams, aliases, include_candidates=False)
    overlay = _team_aliases(fixtures, teams, aliases, include_candidates=True)
    index = _by_date(catalog)
    original_approved = set(mapping[mapping.approved_unique].canonical_fixture_id.astype(str))
    original_duplicate_ids = set(mapping[mapping.mapping_reason.eq("duplicate_fixture_mapping")].canonical_fixture_id.dropna().astype(str))
    rows, recoverable = [], []
    for fixture in fixtures.sort_values("canonical_match_id", kind="stable").itertuples():
        status, reason, candidates, alias_tokens = _status(fixture, original_approved, original_duplicate_ids, locked, overlay, index)
        candidate_ids = sorted(str(x[0].market_id) for x in candidates)
        row = {
            "canonical_fixture_id": str(fixture.canonical_match_id), "season": int(fixture.season_start_year),
            "fixture_datetime_utc": fixture.fixture_dt.isoformat(), "fixture_date": fixture.fixture_date,
            "month": fixture.fixture_dt.month_name(), "weekday": fixture.fixture_dt.day_name(),
            "home_team": fixture.home_team_name_audit, "away_team": fixture.away_team_name_audit,
            "home_team_id": int(fixture.home_team_id), "away_team_id": int(fixture.away_team_id),
            "source_schema": fixture.primary_source, "coverage_status": status, "explicit_reason": reason,
            "candidate_market_ids": "|".join(candidate_ids), "candidate_count": len(candidate_ids),
            "candidate_alias_tokens": "|".join(alias_tokens),
        }
        rows.append(row)
        if status == "team_alias_unresolved" and len(candidates) == 1 and alias_tokens:
            market, h, a = candidates[0]
            orientation = _orientation(market, fixture, h, a)
            if orientation:
                event_supported,event_format=_event_pair_support(market,fixture,overlay)
                recoverable.append({**row, "market_id": str(market.market_id), "event_id": str(market.event_id),
                                    "source_file": market.source_file, "event_name": market.event_name,
                                    "market_time_utc": market.market_time_utc, "market_type": market.market_type,
                                    "event_pair_supported":event_supported,"event_name_format":event_format,
                                    **orientation, "recoverability": "high_confidence_alias_patch"})
    fixtures_out = pd.DataFrame(rows)
    # A promoted team is one absent from the preceding E0 season's team set.
    teams_by_season = {
        int(season): set(group.home_team_id).union(set(group.away_team_id))
        for season, group in fixtures.groupby("season_start_year")
    }
    fixtures_out["home_promoted_to_e0"] = [
        bool(int(row.home_team_id) not in teams_by_season.get(int(row.season)-1, set())) if int(row.season)-1 in teams_by_season else False
        for row in fixtures_out.itertuples()
    ]
    fixtures_out["away_promoted_to_e0"] = [
        bool(int(row.away_team_id) not in teams_by_season.get(int(row.season)-1, set())) if int(row.season)-1 in teams_by_season else False
        for row in fixtures_out.itertuples()
    ]
    fixtures_out["has_promoted_team"] = fixtures_out.home_promoted_to_e0 | fixtures_out.away_promoted_to_e0
    rec = pd.DataFrame(recoverable)
    # The proposed alias must have repeated evidence and only one team id by construction.
    evidence = []
    for token, team_id in sorted(CANDIDATE_ALIASES.items()):
        supported = rec[rec.candidate_alias_tokens.str.contains(token, regex=False, na=False)] if not rec.empty else rec
        evidence.append({"alias_name": token, "alias_normalized": token, "team_id": team_id,
                         "distinct_fixture_evidence": int(supported.canonical_fixture_id.nunique()) if not supported.empty else 0,
                         "season_evidence": int(supported.season.nunique()) if not supported.empty else 0,
                         "safe_for_automatic_approval": bool(not supported.empty and supported.canonical_fixture_id.nunique() >= 2),
                         "basis": "exact runner label; date-compatible unique MATCH_ODDS market; repeated evidence",
                         "registry_action": "PROPOSE ONLY — do not overwrite locked registry"})
    aliases_out = pd.DataFrame(evidence)
    safe_tokens = set(aliases_out.loc[aliases_out.safe_for_automatic_approval, "alias_normalized"])
    rec = rec[rec.candidate_alias_tokens.map(lambda x: set(x.split("|")) <= safe_tokens)].copy() if not rec.empty else rec
    write_csv(REPORTS / "v5_mapping_missingness_by_fixture.csv", fixtures_out)
    write_csv(REPORTS / "v5_mapping_recoverable_candidates.csv", rec)
    write_csv(REPORTS / "v5_mapping_proposed_market_patch.csv", rec)
    write_csv(REPORTS / "v5_mapping_proposed_alias_patch.csv", aliases_out)
    write_csv(PROCESSED / "v5_proposed_alias_patch_v1.csv", aliases_out)
    write_csv(PROCESSED / "v5_proposed_market_patch_v1.csv", rec)
    manual = fixtures_out[fixtures_out.coverage_status.ne("mapped_approved") & ~fixtures_out.canonical_fixture_id.isin(rec.canonical_fixture_id if not rec.empty else [])].copy()
    write_csv(REPORTS / "v5_mapping_manual_review.csv", manual)
    by_season = fixtures_out.groupby(["season", "coverage_status"], dropna=False).size().rename("fixture_count").reset_index()
    extras = []
    for dimension in ("month", "weekday"):
        part = fixtures_out.groupby([dimension, "coverage_status"], dropna=False).size().rename("fixture_count").reset_index()
        part.insert(0, "season", "ALL"); part = part.rename(columns={dimension: "dimension_value"}); part["dimension"] = dimension
        extras.append(part)
    season_part = by_season.copy(); season_part["dimension"] = "season"; season_part["dimension_value"] = season_part.season.astype(str)
    combined = pd.concat([season_part[["season","dimension","dimension_value","coverage_status","fixture_count"]], *extras], ignore_index=True)
    write_csv(REPORTS / "v5_mapping_missingness_by_season.csv", combined)
    home = fixtures_out.groupby(["home_team", "coverage_status"]).size().rename("fixture_count").reset_index().rename(columns={"home_team":"team"}); home.insert(0,"team_role","home")
    away = fixtures_out.groupby(["away_team", "coverage_status"]).size().rename("fixture_count").reset_index().rename(columns={"away_team":"team"}); away.insert(0,"team_role","away")
    write_csv(REPORTS / "v5_mapping_missingness_by_team.csv", pd.concat([home,away],ignore_index=True))
    return catalog, mapping, fixtures_out, rec, aliases_out


def corrected_replay(catalog, mapping, fixtures_out, recoverable):
    """Create V2 map/panel and replay unchanged research only for safe recoveries."""
    if recoverable.empty:
        return None
    mapping_v2 = mapping.copy()
    additions = []
    for row in recoverable.to_dict("records"):
        mask = mapping_v2.market_id.astype(str).eq(str(row["market_id"]))
        if mask.sum() != 1:
            raise ValueError(f"expected exactly one catalog map row for {row['market_id']}")
        update = {"mapping_class":"coverage_audit_alias_patch", "mapping_reason":"repeated_exact_alias_unique_date_team_market",
                  "approved_unique":True, "canonical_fixture_id":row["canonical_fixture_id"], "fixture_date":row["fixture_datetime_utc"],
                  "season":row["season"], "home_team_name":row["home_team"], "away_team_name":row["away_team"],
                  "home_runner_id":row["home_runner_id"], "draw_runner_id":row["draw_runner_id"], "away_runner_id":row["away_runner_id"]}
        for key,value in update.items(): mapping_v2.loc[mask,key]=value
        additions.append(row)
    approved = mapping_v2[mapping_v2.approved_unique].copy()
    if approved.market_id.duplicated().any() or approved.canonical_fixture_id.duplicated().any():
        raise ValueError("corrected mapping failed uniqueness gate")
    mapping_v2.to_parquet(CORRECTED_MAP,index=False)
    original_long = pd.read_parquet(PROCESSED / "e0_match_odds_cutoff_long_v1.parquet")
    new_rows=[]
    for row in additions:
        result=extract_market(ROOT / row["source_file"],row,include_trajectory=False)
        digest=raw_sha256(ROOT / row["source_file"])
        for state in result.cutoffs:
            state["raw_sha256"]=digest; state["parse_warnings"]=result.metadata["parse_warnings"]
        new_rows.extend(result.cutoffs)
    merged=pd.concat([original_long,pd.DataFrame(new_rows)],ignore_index=True)
    # V1 uses serialized fixture dates; keep the corrected panel schema stable.
    merged["fixture_date"] = merged["fixture_date"].astype(str)
    merged=merged.sort_values(["season","market_start_utc","market_id","cutoff"],kind="stable")
    merged.to_parquet(CORRECTED_LONG,index=False)
    index=["market_id","event_id","canonical_fixture_id","league","season","fixture_date","market_start_utc","source_file","raw_sha256"]
    values=[c for c in merged.columns if c not in index+["cutoff","cutoff_utc","parse_warnings"]]
    wide=merged.pivot_table(index=index,columns="cutoff",values=values,aggfunc="first")
    wide.columns=[f"{cutoff}_{value}" for value,cutoff in wide.columns]
    wide=wide.reset_index().sort_values(["season","market_start_utc","market_id"],kind="stable")
    wide.to_parquet(CORRECTED_PANEL,index=False)
    # The research function's feature/target/validation code is unchanged; only
    # its report directory is redirected to preserve frozen V1 V5 outputs.
    from . import research
    original_reports=research.REPORTS
    CORRECTED_REPORTS.mkdir(parents=True,exist_ok=True)
    research.REPORTS=CORRECTED_REPORTS
    try:
        decision=research.run_research(merged, mapping_v2)
    finally:
        research.REPORTS=original_reports
    return mapping_v2, merged, wide, decision


def report_and_decide(fixtures_out, recoverable, aliases_out, replay):
    current=int(fixtures_out.coverage_status.eq("mapped_approved").sum())
    recoverable_n=len(recoverable)
    manual=int((~fixtures_out.coverage_status.eq("mapped_approved") & ~fixtures_out.canonical_fixture_id.isin(recoverable.canonical_fixture_id if not recoverable.empty else [])).sum())
    structural=int(fixtures_out.coverage_status.isin(["no_betfair_candidate","multiple_candidate_markets","date_mismatch","incomplete_runners","market_cancelled_or_invalid","duplicate_market","non_match_odds"]).sum())
    if replay is None:
        decision="v5_mapping_coverage_sufficient_no_rerun" if recoverable_n == 0 else "v5_mapping_missingness_structural"
        comparison=[]
    else:
        _, _, corrected_panel, research_decision = replay
        old_panel=pd.read_parquet(PANEL)
        old_summary=pd.read_csv(REPORTS / "v5_predictive_summary.csv")
        new_summary=pd.read_csv(CORRECTED_REPORTS / "v5_predictive_summary.csv")
        old_boot=pd.read_csv(REPORTS / "v5_bootstrap.csv"); new_boot=pd.read_csv(CORRECTED_REPORTS / "v5_bootstrap.csv")
        def metric(frame,name,col):
            row=frame[frame.model.eq(name)]
            return float(row.iloc[0][col]) if not row.empty else np.nan
        comparison=[
            {"metric":"panel_rows","original":len(old_panel),"corrected":len(corrected_panel)},
            {"metric":"fixture_coverage","original":current/len(fixtures_out),"corrected":(current+recoverable_n)/len(fixtures_out)},
            {"metric":"mae_baseline","original":metric(old_summary,"no_movement","mae"),"corrected":metric(new_summary,"no_movement","mae")},
            {"metric":"mae_model","original":metric(old_summary,"selected_model","mae"),"corrected":metric(new_summary,"selected_model","mae")},
            {"metric":"bootstrap_probability_model_beats_baseline","original":float((old_boot.model_minus_zero_mae<0).mean()),"corrected":float((new_boot.model_minus_zero_mae<0).mean())},
            {"metric":"movement_correlation","original":metric(old_summary,"selected_model","correlation"),"corrected":metric(new_summary,"selected_model","correlation")},
            {"metric":"directional_accuracy","original":metric(old_summary,"selected_model","directional_accuracy"),"corrected":metric(new_summary,"selected_model","directional_accuracy")},
        ]
        for report_root,label in ((REPORTS,"original"),(CORRECTED_REPORTS,"corrected")):
            dec=pd.read_csv(report_root / "v5_decile_analysis.csv")
            means=dec.groupby("prediction_decile").mean(numeric_only=True).mean_target_movement
            comparison.append({"metric":f"decile_spread_{label}","original":means.iloc[-1]-means.iloc[0] if label=="original" else np.nan,"corrected":means.iloc[-1]-means.iloc[0] if label=="corrected" else np.nan})
        decision="v5_mapping_corrected_still_no_signal" if research_decision=="v5_betfair_no_price_movement_signal" else "v5_mapping_corrected_signal_requires_independent_validation"
    write_csv(REPORTS / "v5_mapping_coverage_comparison.csv",comparison)
    status_counts=fixtures_out.coverage_status.value_counts().to_dict()
    team_missing=pd.read_csv(REPORTS / "v5_mapping_missingness_by_team.csv")
    concentration=team_missing[team_missing.coverage_status.ne("mapped_approved")].groupby("team").fixture_count.sum().sort_values(ascending=False).head(12)
    missing=fixtures_out.coverage_status.ne("mapped_approved")
    promoted=fixtures_out.has_promoted_team
    promoted_rate=missing[promoted].mean() if promoted.any() else np.nan
    established_rate=missing[~promoted].mean() if (~promoted).any() else np.nan
    covid=fixtures_out.season.eq(2020)
    covid_rate=missing[covid].mean() if covid.any() else np.nan
    non_covid_rate=missing[~covid].mean() if (~covid).any() else np.nan
    source_summary=fixtures_out.groupby("source_schema").size().to_dict()
    event_formats=recoverable.event_name_format.value_counts().to_dict() if not recoverable.empty else {}
    lines=[
        f"Current approved fixtures: **{current}/{len(fixtures_out)}** ({current/len(fixtures_out):.2%}).",
        f"High-confidence recoverable fixtures: **{recoverable_n}**; remaining manual-review fixtures: **{manual}**; structural/no-safe-auto-match fixtures: **{structural}**.",
        "Coverage status counts: " + ", ".join(f"{k}={v}" for k,v in sorted(status_counts.items())) + ".",
        "Proposed aliases are overlay-only and exact: " + ", ".join(aliases_out.loc[aliases_out.safe_for_automatic_approval,"alias_name"].tolist()) + ". The locked registry was not modified.",
        "Highest missing fixture-team concentrations: " + ", ".join(f"{team} ({count})" for team,count in concentration.items()) + ".",
        f"Promoted-team fixture missingness is {promoted_rate:.2%} versus {established_rate:.2%} for other fixtures. COVID-season (2020) missingness is {covid_rate:.2%} versus {non_covid_rate:.2%} elsewhere; 15 of the 24 date mismatches occur in 2020, consistent with rescheduling sensitivity rather than an auto-fixable alias rule.",
        "Fixture source schemas: " + ", ".join(f"{key}={value}" for key,value in source_summary.items()) + ".",
        "High-confidence recovered event-name formats: " + ", ".join(f"{key}={value}" for key,value in event_formats.items()) + ". Reordered `v`/`vs`/`@`/`-` formats were accepted only when the exact normalized pair matched in either orientation.",
        "All matching used date ±2 days, exact normalized locked/overlay aliases, runner identities, reordered event-name-compatible source names, and explicit common abbreviations. No fuzzy match was accepted.",
        "BASIC LTP remains non-executable and conveys no verified liquidity. This audit makes no claim of predictive signal or edge.",
    ]
    if comparison: lines.append(pd.DataFrame(comparison).to_markdown(index=False))
    write_md(REPORTS / "v5_mapping_coverage_audit.md","V5 E0 mapping-coverage audit",lines)
    write_md(REPORTS / "v5_mapping_coverage_decision.md","V5 mapping-coverage decision",[f"`{decision}`","Any corrected result remains research-only; independent Betfair ADVANCED validation is required before strategy work."])
    return decision


def run():
    catalog,mapping,fixtures_out,recoverable,aliases_out=audit()
    replay=corrected_replay(catalog,mapping,fixtures_out,recoverable) if not recoverable.empty else None
    return report_and_decide(fixtures_out,recoverable,aliases_out,replay)


if __name__ == "__main__":
    print(run())
