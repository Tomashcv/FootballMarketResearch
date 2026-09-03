from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_football_data_source_layer import (
    ROOT,
    SUPER_DIR,
    OUT_DIR,
    REPORT_DIR,
    ALIASES_LOCKED,
    ALIASES_PLUS_TM,
    TEAMS,
    attach_team_ids,
    build_1x2,
    build_ah,
    build_source_map,
    normalize_name,
)


CANDIDATES = ROOT / "data/processed/football_data/football_data_team_alias_candidates_v1.csv"
NORMALIZED = ROOT / "data/processed/football_data/football_data_normalized_matches_v1.csv"
SOURCE_MAP = ROOT / "data/processed/football_data/football_data_source_match_map_v1.csv"
X1_PRE = ROOT / "data/processed/super_csvs/research_ready/football_data/super_1x2_football_data_top5_research_v1.csv"
AH_PRE = ROOT / "data/processed/super_csvs/research_ready/football_data/super_ah_football_data_top5_research_v1.csv"
AH_OPEN_PRE = ROOT / "data/processed/super_csvs/research_ready/football_data/super_ah_open_football_data_top5_research_v1.csv"
AH_CLOSE_PRE = ROOT / "data/processed/super_csvs/research_ready/football_data/super_ah_close_football_data_top5_research_v1.csv"

LOCKED_ALIAS_OUT = OUT_DIR / "football_data_team_alias_locked_v1.csv"
ENTITY_ALIAS_OUT = ROOT / "data/processed/entity_registry/team_aliases_v1_locked_plus_transfermarkt_football_data.csv"
SOURCE_MAP_LOCKED_OUT = OUT_DIR / "football_data_source_match_map_v1_locked.csv"
X1_LOCKED_OUT = SUPER_DIR / "super_1x2_football_data_top5_research_v1_locked.csv"
AH_LOCKED_OUT = SUPER_DIR / "super_ah_football_data_top5_research_v1_locked.csv"
AH_OPEN_LOCKED_OUT = SUPER_DIR / "super_ah_open_football_data_top5_research_v1_locked.csv"
AH_CLOSE_LOCKED_OUT = SUPER_DIR / "super_ah_close_football_data_top5_research_v1_locked.csv"


MANUAL_OBVIOUS = {
    "nott_m_forest": "nottingham",
    "m_gladbach": "b_monchengladbach",
}


def load_original_aliases() -> pd.DataFrame:
    path = ALIASES_PLUS_TM if ALIASES_PLUS_TM.exists() else ALIASES_LOCKED
    return pd.read_csv(path)


def build_locked_aliases() -> pd.DataFrame:
    candidates = pd.read_csv(CANDIDATES)
    teams = pd.read_csv(TEAMS)
    teams["canonical_normalized"] = teams["canonical_team_name"].map(normalize_name)
    team_by_norm = teams.drop_duplicates("canonical_normalized").set_index("canonical_normalized").to_dict("index")
    rows = []
    for _, row in candidates.iterrows():
        raw = row["football_data_team_raw"]
        norm = row["football_data_team_normalized"]
        league = row["league_hint"]
        target_norm = MANUAL_OBVIOUS.get(norm)
        status = ""
        approved = False
        manual = False
        notes = ""
        team = None
        if bool(row.get("approved_for_research", False)) and pd.notna(row.get("candidate_team_id")):
            team_id = int(row["candidate_team_id"])
            team_match = teams[teams["team_id"].eq(team_id)]
            if not team_match.empty:
                team = team_match.iloc[0].to_dict()
                status = "approved_exact" if str(row.get("match_type", "")).startswith("exact") else "approved_obvious_alias"
                approved = True
                notes = "Previously approved exact locked alias/canonical match retained."
        elif target_norm and target_norm in team_by_norm:
            team = team_by_norm[target_norm]
            status = "approved_obvious_alias"
            approved = True
            notes = f"Manual decision: {raw} maps to existing canonical team {target_norm}."
        elif norm in team_by_norm:
            team = team_by_norm[norm]
            status = "approved_exact"
            approved = True
            notes = "Exact normalized football-data team name exists in teams_v1_locked."
        else:
            status = "out_of_scope_current_registry"
            approved = False
            notes = "Real club but not present in current Footiqo top-5 canonical registry scope."
        rows.append(
            {
                "football_data_team_raw": raw,
                "football_data_team_normalized": norm,
                "team_id": int(team["team_id"]) if team is not None else np.nan,
                "canonical_team_name": team["canonical_team_name"] if team is not None else "",
                "country_hint": team["country"] if team is not None else row.get("country_hint", ""),
                "league_hint": league,
                "alias_status": status,
                "approved_for_research": approved,
                "manual_review_required": manual,
                "notes": notes,
            }
        )
    locked = pd.DataFrame(rows)
    conflicts = locked[locked["approved_for_research"].astype(bool)].groupby("football_data_team_normalized")["team_id"].nunique(dropna=True)
    conflict_norms = set(conflicts[conflicts.gt(1)].index)
    if conflict_norms:
        mask = locked["football_data_team_normalized"].isin(conflict_norms)
        locked.loc[mask, "alias_status"] = "rejected"
        locked.loc[mask, "approved_for_research"] = False
        locked.loc[mask, "manual_review_required"] = True
        locked.loc[mask, "notes"] = locked.loc[mask, "notes"].astype(str) + " Conflict: approved normalized alias maps to multiple team_id values."
    return locked.sort_values(["league_hint", "football_data_team_normalized"]).reset_index(drop=True)


def write_entity_aliases(locked: pd.DataFrame) -> pd.DataFrame:
    base = load_original_aliases()
    approved = locked[locked["approved_for_research"].astype(bool)].copy()
    next_id = int(pd.to_numeric(base["alias_id"], errors="coerce").max()) + 1 if not base.empty else 1
    rows = []
    for offset, (_, row) in enumerate(approved.iterrows()):
        rows.append(
            {
                "alias_id": next_id + offset,
                "team_id": int(row["team_id"]),
                "source": "football_data",
                "alias_name": row["football_data_team_raw"],
                "alias_normalized": row["football_data_team_normalized"],
                "source_team_name": row["football_data_team_raw"],
                "country_hint": row["country_hint"],
                "league_hint": row["league_hint"],
                "valid_from": "",
                "valid_to": "",
                "confidence": 1.0 if row["alias_status"] == "approved_exact" else 0.95,
                "alias_status": row["alias_status"],
                "approved_for_research": True,
                "manual_review_required": False,
                "notes": "Football-data locked alias table v1.",
            }
        )
    plus = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    plus.to_csv(ENTITY_ALIAS_OUT, index=False)
    return plus


def attach_team_ids_from_locked(norm: pd.DataFrame, locked: pd.DataFrame) -> pd.DataFrame:
    approved = locked[locked["approved_for_research"].astype(bool)].copy()
    tmp = approved.rename(
        columns={
            "football_data_team_raw": "football_data_team_raw",
            "football_data_team_normalized": "football_data_team_normalized",
            "team_id": "candidate_team_id",
            "canonical_team_name": "candidate_canonical_team_name",
        }
    )
    tmp["match_type"] = tmp["alias_status"]
    tmp["confidence"] = np.where(tmp["alias_status"].eq("approved_exact"), 1.0, 0.95)
    tmp["manual_review_required"] = False
    return attach_team_ids(norm, tmp)


def rebuild_market_csvs(norm_ids: pd.DataFrame, source_map: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs = {}
    x1 = build_1x2(norm_ids, source_map)
    if not x1.empty:
        x1.to_csv(X1_LOCKED_OUT, index=False)
    outputs["1x2"] = x1
    ah_primary = pd.DataFrame()
    if AH_OPEN_PRE.exists():
        ah_open = build_ah(norm_ids, source_map, "open")
        if not ah_open.empty:
            ah_open.to_csv(AH_OPEN_LOCKED_OUT, index=False)
        outputs["ah_open"] = ah_open
    else:
        outputs["ah_open"] = pd.DataFrame()
    if AH_CLOSE_PRE.exists():
        ah_close = build_ah(norm_ids, source_map, "close")
        if not ah_close.empty:
            ah_close.to_csv(AH_CLOSE_LOCKED_OUT, index=False)
        outputs["ah_close"] = ah_close
        ah_primary = ah_close
    else:
        outputs["ah_close"] = pd.DataFrame()
    if AH_PRE.exists():
        if ah_primary.empty:
            ah_primary = outputs["ah_open"]
        if not ah_primary.empty:
            ah_primary.to_csv(AH_LOCKED_OUT, index=False)
        outputs["ah_primary"] = ah_primary
    return outputs


def validate(locked: pd.DataFrame, source_map: pd.DataFrame, outputs: dict[str, pd.DataFrame], entity_aliases: pd.DataFrame) -> pd.DataFrame:
    teams = pd.read_csv(TEAMS)
    before_ids = set(teams["team_id"].astype(int))
    approved_ids = set(pd.to_numeric(locked.loc[locked["approved_for_research"].astype(bool), "team_id"], errors="coerce").dropna().astype(int))
    x1 = outputs.get("1x2", pd.DataFrame())
    ah = outputs.get("ah_primary", pd.DataFrame())
    nott = locked[locked["football_data_team_normalized"].eq("nott_m_forest")]
    glad = locked[locked["football_data_team_normalized"].eq("m_gladbach")]
    out_scope_approved = int(locked[locked["alias_status"].eq("out_of_scope_current_registry")]["approved_for_research"].astype(bool).sum())
    rows = [
        {"check_name": "no_remaining_manual_review_aliases", "status": "pass" if not locked["manual_review_required"].astype(bool).any() else "fail", "details": f"manual_review={int(locked['manual_review_required'].astype(bool).sum())}"},
        {"check_name": "no_new_team_id_created", "status": "pass" if approved_ids.issubset(before_ids) else "fail", "details": f"approved_team_ids={len(approved_ids)}"},
        {"check_name": "out_of_scope_not_approved", "status": "pass" if out_scope_approved == 0 else "fail", "details": f"approved_out_of_scope={out_scope_approved}"},
        {"check_name": "nottm_forest_maps_to_nottingham", "status": "pass" if not nott.empty and nott.iloc[0]['canonical_team_name'] == "nottingham" and bool(nott.iloc[0]['approved_for_research']) else "fail", "details": nott.to_dict('records') if not nott.empty else "missing"},
        {"check_name": "mgladbach_maps_to_b_monchengladbach", "status": "pass" if not glad.empty and glad.iloc[0]['canonical_team_name'] == "b_monchengladbach" and bool(glad.iloc[0]['approved_for_research']) else "fail", "details": glad.to_dict('records') if not glad.empty else "missing"},
        {"check_name": "x1_no_duplicate_canonical_match_id", "status": "pass" if not x1.empty and not x1["canonical_match_id"].duplicated().any() else "fail", "details": f"rows={len(x1)} duplicates={0 if x1.empty else int(x1['canonical_match_id'].duplicated().sum())}"},
        {"check_name": "x1_odds_gt_1", "status": "pass" if not x1.empty and x1[["x1_home_odds", "x1_draw_odds", "x1_away_odds"]].gt(1).all().all() else "fail", "details": "locked x1 odds filtered > 1"},
        {"check_name": "x1_targets_present", "status": "pass" if not x1.empty and x1[["target_home_win", "target_draw", "target_away_win"]].sum(axis=1).eq(1).all() else "fail", "details": "one target active per row"},
        {"check_name": "ah_no_duplicate_canonical_match_id", "status": "pass" if ah.empty or not ah["canonical_match_id"].duplicated().any() else "fail", "details": f"rows={len(ah)} duplicates={0 if ah.empty else int(ah['canonical_match_id'].duplicated().sum())}"},
        {"check_name": "ah_odds_gt_1", "status": "pass" if ah.empty or ah[["ah_home_odds", "ah_away_odds"]].gt(1).all().all() else "fail", "details": "locked ah odds filtered > 1"},
        {"check_name": "ah_settlement_columns_preserved", "status": "pass" if ah.empty or {"ah_home_unit_return", "ah_away_unit_return", "ah_home_settlement", "ah_away_settlement"}.issubset(ah.columns) else "fail", "details": "AH settlement columns present"},
        {"check_name": "classification_research_only", "status": "pass" if (x1.empty or x1["classification"].eq("research_only").all()) and (ah.empty or ah["classification"].eq("research_only").all()) else "fail", "details": "research_only retained"},
        {"check_name": "source_map_locked_written", "status": "pass" if SOURCE_MAP_LOCKED_OUT.exists() and len(source_map) > 0 else "fail", "details": f"rows={len(source_map)}"},
        {"check_name": "entity_alias_plus_file_written", "status": "pass" if ENTITY_ALIAS_OUT.exists() and len(entity_aliases) > 0 else "fail", "details": str(ENTITY_ALIAS_OUT.relative_to(ROOT))},
    ]
    return pd.DataFrame(rows)


def write_reports(
    locked: pd.DataFrame,
    source_map: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    validation: pd.DataFrame,
    decision: str,
    pre_counts: dict[str, int],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    locked.to_csv(REPORT_DIR / "football_data_alias_locked_table.csv", index=False)
    map_summary = source_map.groupby(["competition_slug", "season_start_year", "mapping_method"], dropna=False).size().reset_index(name="rows")
    unmatched_count = int(source_map["canonical_match_id"].isna().sum())
    mapped_unique = source_map.loc[source_map["canonical_match_id"].notna(), "canonical_match_id"].nunique()
    (REPORT_DIR / "football_data_source_map_locked_report.md").write_text(
        "\n".join(
            [
                "# Football-Data Locked Source Map Report",
                "",
                f"Mapped source rows: {int(source_map['canonical_match_id'].notna().sum())}",
                f"Unique mapped canonical_match_id: {mapped_unique}",
                f"Unmatched/out-of-scope rows: {unmatched_count}",
                f"Duplicate source-layer rows flagged: {int(source_map.get('duplicate_source_row_for_canonical_flag', pd.Series(False, index=source_map.index)).astype(bool).sum())}",
                "",
                map_summary.to_markdown(index=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    market_rows = [
        {"market_csv": "1x2", "rows_before_locking": pre_counts.get("1x2", 0), "rows_after_locking": len(outputs.get("1x2", pd.DataFrame()))},
        {"market_csv": "ah_primary", "rows_before_locking": pre_counts.get("ah_primary", 0), "rows_after_locking": len(outputs.get("ah_primary", pd.DataFrame()))},
        {"market_csv": "ah_open", "rows_before_locking": pre_counts.get("ah_open", 0), "rows_after_locking": len(outputs.get("ah_open", pd.DataFrame()))},
        {"market_csv": "ah_close", "rows_before_locking": pre_counts.get("ah_close", 0), "rows_after_locking": len(outputs.get("ah_close", pd.DataFrame()))},
    ]
    market_df = pd.DataFrame(market_rows)
    (REPORT_DIR / "football_data_market_csvs_locked_report.md").write_text(
        "\n".join(
            [
                "# Football-Data Locked Market CSV Report",
                "",
                market_df.to_markdown(index=False),
                "",
                "Research CSVs are rebuilt from locked aliases and locked source map. Classification remains research_only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    validation.to_csv(REPORT_DIR / "football_data_alias_locked_validation.csv", index=False)
    approved_count = int(locked["approved_for_research"].astype(bool).sum())
    out_scope_count = int(locked["alias_status"].eq("out_of_scope_current_registry").sum())
    lines = [
        "# Football-Data Alias Lock Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Locked alias rows: {len(locked)}",
        f"- Approved aliases: {approved_count}",
        f"- Out-of-scope current registry aliases: {out_scope_count}",
        f"- Remaining manual-review aliases: {int(locked['manual_review_required'].astype(bool).sum())}",
        f"- Nott'm Forest approved to nottingham: {bool((locked['football_data_team_normalized'].eq('nott_m_forest') & locked['canonical_team_name'].eq('nottingham') & locked['approved_for_research'].astype(bool)).any())}",
        f"- M'gladbach approved to b_monchengladbach: {bool((locked['football_data_team_normalized'].eq('m_gladbach') & locked['canonical_team_name'].eq('b_monchengladbach') & locked['approved_for_research'].astype(bool)).any())}",
        "",
        "No new team_id values were created. The canonical registry was not extended.",
        "",
        "## Validation",
        validation.to_markdown(index=False),
    ]
    (REPORT_DIR / "football_data_alias_lock_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT_DIR / "football_data_alias_lock_decision.md").write_text(
        "\n".join(["# Football-Data Alias Lock Decision", "", f"Decision: **{decision}**", "", "No modeling, value search, canonical extension, raw-file modification, or confirmed-edge claim was performed."]) + "\n",
        encoding="utf-8",
    )


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(pd.read_csv(path, usecols=[0]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUPER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    locked = build_locked_aliases()
    locked.to_csv(LOCKED_ALIAS_OUT, index=False)
    locked.to_csv(REPORT_DIR / "football_data_alias_locked_table.csv", index=False)
    entity_aliases = write_entity_aliases(locked)
    norm = pd.read_csv(NORMALIZED, low_memory=False)
    norm_ids = attach_team_ids_from_locked(norm, locked)
    source_map, _unmatched = build_source_map(norm_ids)
    source_map.to_csv(SOURCE_MAP_LOCKED_OUT, index=False)
    outputs = rebuild_market_csvs(norm_ids, source_map)
    validation = validate(locked, source_map, outputs, entity_aliases)
    decision = "football_data_alias_locked_ready_good"
    if validation["status"].eq("fail").any():
        decision = "football_data_alias_lock_failed"
    elif locked["manual_review_required"].astype(bool).any():
        decision = "football_data_alias_locked_ready_needs_review"
    pre_counts = {
        "1x2": count_rows(X1_PRE),
        "ah_primary": count_rows(AH_PRE),
        "ah_open": count_rows(AH_OPEN_PRE),
        "ah_close": count_rows(AH_CLOSE_PRE),
    }
    write_reports(locked, source_map, outputs, validation, decision, pre_counts)
    print(decision)
    print(
        f"aliases={len(locked)} approved={int(locked['approved_for_research'].astype(bool).sum())} "
        f"out_of_scope={int(locked['alias_status'].eq('out_of_scope_current_registry').sum())} "
        f"x1_rows={len(outputs.get('1x2', pd.DataFrame()))} ah_rows={len(outputs.get('ah_primary', pd.DataFrame()))}"
    )


if __name__ == "__main__":
    main()
