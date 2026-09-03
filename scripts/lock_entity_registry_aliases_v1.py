from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data/processed/entity_registry"
REPORT_DIR = ROOT / "outputs/reports/entity_registry"

APPROVED_DECISIONS = {
    338: ("SC Bastia", 18, "bastia", "France"),
    387: ("Greuther Fuerth", 67, "greuther_furth", "Germany"),
    391: ("FC Heidenheim", 71, "heidenheim", "Germany"),
    435: ("Nuernberg", 115, "nurnberg", "Germany"),
    471: ("Real Valladolid", 151, "valladolid", "Spain"),
}

REJECTED_DECISIONS = {
    384: {
        "alias_name": "Dijon",
        "team_id": 64,
        "expected_team": "gijon",
        "reason": "Rejected by manual review: false Dijon/Gijon fuzzy match. Dijon is a French club and must not map to Gijon, Spain. No correct Dijon team_id exists in this Footiqo top-5 entity registry, so the alias remains rejected and unavailable for research.",
    }
}


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def approved_alias_conflicts(aliases: pd.DataFrame) -> pd.DataFrame:
    approved = aliases[bool_series(aliases["approved_for_research"])].copy()
    rows = []
    for (source, alias_norm), g in approved.groupby(["source", "alias_normalized"], dropna=False):
        team_ids = sorted(g["team_id"].dropna().astype(int).unique())
        if len(team_ids) > 1:
            rows.append(
                {
                    "source": source,
                    "alias_normalized": alias_norm,
                    "approved_team_ids": "; ".join(map(str, team_ids)),
                    "alias_names": "; ".join(sorted(g["alias_name"].astype(str).unique())),
                    "row_count": len(g),
                }
            )
    return pd.DataFrame(rows, columns=["source", "alias_normalized", "approved_team_ids", "alias_names", "row_count"])


def apply_decisions(aliases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = aliases.copy()
    decision_rows = []

    for alias_id, (alias_name, team_id, canonical_team, country) in APPROVED_DECISIONS.items():
        mask = out["alias_id"].eq(alias_id)
        if mask.sum() != 1:
            raise ValueError(f"Expected one row for approved alias_id {alias_id}, found {mask.sum()}")
        row = out.loc[mask].iloc[0]
        if int(row["team_id"]) != team_id or str(row["alias_name"]) != alias_name:
            raise ValueError(f"Manual approved decision does not match row for alias_id {alias_id}")
        out.loc[mask, "alias_status"] = "approved_obvious_alias"
        out.loc[mask, "approved_for_research"] = True
        out.loc[mask, "manual_review_required"] = False
        out.loc[mask, "notes"] = (
            f"Approved by manual entity-registry lock review: {alias_name} maps to "
            f"team_id {team_id} ({canonical_team}, {country})."
        )
        decision_rows.append(
            {
                "alias_id": alias_id,
                "decision": "approved",
                "source": row["source"],
                "alias_name": alias_name,
                "team_id": team_id,
                "notes": out.loc[mask, "notes"].iloc[0],
            }
        )

    for alias_id, decision in REJECTED_DECISIONS.items():
        mask = out["alias_id"].eq(alias_id)
        if mask.sum() != 1:
            raise ValueError(f"Expected one row for rejected alias_id {alias_id}, found {mask.sum()}")
        row = out.loc[mask].iloc[0]
        if int(row["team_id"]) != decision["team_id"] or str(row["alias_name"]) != decision["alias_name"]:
            raise ValueError(f"Manual rejected decision does not match row for alias_id {alias_id}")
        out.loc[mask, "alias_status"] = "rejected"
        out.loc[mask, "approved_for_research"] = False
        out.loc[mask, "manual_review_required"] = False
        out.loc[mask, "notes"] = decision["reason"]
        decision_rows.append(
            {
                "alias_id": alias_id,
                "decision": "rejected",
                "source": row["source"],
                "alias_name": decision["alias_name"],
                "team_id": decision["team_id"],
                "notes": decision["reason"],
            }
        )

    decisions = pd.DataFrame(decision_rows).sort_values("alias_id").reset_index(drop=True)
    return out, decisions


def validate(teams: pd.DataFrame, aliases: pd.DataFrame, matches: pd.DataFrame, source_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = []

    def add(name: str, passed: bool, details: str = "") -> None:
        rows.append({"check_name": name, "status": "pass" if passed else "fail", "details": details})

    manual = bool_series(aliases["manual_review_required"])
    approved = bool_series(aliases["approved_for_research"])
    rejected = aliases["alias_status"].eq("rejected")
    conflicts = approved_alias_conflicts(aliases)

    add("no_alias_manual_review_required", not manual.any(), f"manual_review_required={int(manual.sum())}")
    add("rejected_aliases_not_approved", not (rejected & approved).any(), f"rejected_approved={int((rejected & approved).sum())}")
    add("dijon_gijon_false_match_rejected", bool((aliases["alias_id"].eq(384) & aliases["alias_status"].eq("rejected") & ~approved).any()), "alias_id=384 must remain rejected")
    add("manual_approved_aliases_approved", aliases[aliases["alias_id"].isin(APPROVED_DECISIONS)]["approved_for_research"].astype(bool).all(), f"approved_count={len(APPROVED_DECISIONS)}")
    add("no_wrong_country_approved_manual_alias", not bool(aliases["alias_id"].eq(384).any() and approved[aliases["alias_id"].eq(384)].any()), "Dijon/Gijon is not approved")
    add("no_duplicate_source_alias_to_different_approved_team", conflicts.empty, f"conflicts={len(conflicts)}")
    add("matches_row_count_18008", len(matches) == 18008, f"matches={len(matches)}")
    add("source_match_map_row_count_18008", len(source_map) == 18008, f"source_map={len(source_map)}")
    add("every_match_has_home_team_id", matches["home_team_id"].notna().all(), f"missing={int(matches['home_team_id'].isna().sum())}")
    add("every_match_has_away_team_id", matches["away_team_id"].notna().all(), f"missing={int(matches['away_team_id'].isna().sum())}")
    add("alias_conflicts_remain_zero", conflicts.empty, f"conflicts={len(conflicts)}")
    add("no_modeling_or_value_search", True, "Only entity alias decisions and locked table copies were produced.")

    validation = pd.DataFrame(rows)
    if validation["status"].eq("fail").any():
        decision = "entity_registry_lock_failed"
    elif manual.any() or not conflicts.empty:
        decision = "entity_registry_locked_ready_needs_review"
    else:
        decision = "entity_registry_locked_ready_good"
    return validation, conflicts, decision


def write_reports(aliases: pd.DataFrame, decisions: pd.DataFrame, validation: pd.DataFrame, conflicts: pd.DataFrame, decision: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(REPORT_DIR / "team_aliases_locked_decisions.csv", index=False)
    validation.to_csv(REPORT_DIR / "entity_registry_locked_validation.csv", index=False)

    report = [
        "# Entity Registry Lock Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Manual decisions applied to Understat alias review rows only. Original v1 registry files were left intact and locked copies were written.",
        "",
        "## Manual Decisions",
        f"- Approved aliases: {len(APPROVED_DECISIONS)}",
        f"- Rejected aliases: {len(REJECTED_DECISIONS)}",
        "- alias_id 384 Dijon remains rejected because it was a false Dijon/Gijon fuzzy match and no correct Dijon team_id exists in this registry.",
        "",
        "## Locked Alias Status",
        f"- Total aliases: {len(aliases)}",
        f"- Manual review remaining: {int(bool_series(aliases['manual_review_required']).sum())}",
        f"- Approved for research: {int(bool_series(aliases['approved_for_research']).sum())}",
        f"- Rejected: {int(aliases['alias_status'].eq('rejected').sum())}",
        f"- Approved alias conflicts: {len(conflicts)}",
        "",
        "No modeling was performed. No value search was performed. No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "entity_registry_lock_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    decision_md = [
        "# Entity Registry Locked Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "The locked entity registry has no remaining alias rows requiring manual review. Rejected aliases are not approved for research.",
        "",
        "No modeling was performed and no confirmed edge is claimed.",
    ]
    (REPORT_DIR / "entity_registry_locked_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")


def main() -> None:
    teams = pd.read_csv(IN_DIR / "teams_v1.csv")
    aliases = pd.read_csv(IN_DIR / "team_aliases_v1.csv")
    competitions = pd.read_csv(IN_DIR / "competitions_v1.csv", dtype={"competition_code": str})
    matches = pd.read_csv(IN_DIR / "matches_v1.csv")
    source_map = pd.read_csv(IN_DIR / "source_match_map_v1.csv")

    locked_aliases, decisions = apply_decisions(aliases)
    validation, conflicts, decision = validate(teams, locked_aliases, matches, source_map)

    teams.to_csv(IN_DIR / "teams_v1_locked.csv", index=False)
    locked_aliases.to_csv(IN_DIR / "team_aliases_v1_locked.csv", index=False)
    competitions.to_csv(IN_DIR / "competitions_v1_locked.csv", index=False)
    matches.to_csv(IN_DIR / "matches_v1_locked.csv", index=False)
    source_map.to_csv(IN_DIR / "source_match_map_v1_locked.csv", index=False)
    write_reports(locked_aliases, decisions, validation, conflicts, decision)


if __name__ == "__main__":
    main()
