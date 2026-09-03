from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

import build_feature_matrix_v4_2_fbref_prior_season_partial as base


V3_MATRIX = Path("data/processed/features/football_feature_matrix_v3_clubelo_partial.csv")
MAPPING_DIR = Path("data/mappings")
REPORT_DIR = Path("outputs/reports")

ALIAS_MAPPING_CSV = MAPPING_DIR / "fbref_football_data_aliases_v1.csv"
IMPACT_MD = REPORT_DIR / "fbref_alias_v1_impact_audit.md"
ACCEPTED_CSV = REPORT_DIR / "fbref_aliases_accepted_v1_improved.csv"
MANUAL_CSV = REPORT_DIR / "fbref_aliases_manual_review_required_v1_improved.csv"
REJECTED_CSV = REPORT_DIR / "fbref_aliases_rejected_or_ambiguous_v1.csv"
COVERAGE_AFTER_CSV = REPORT_DIR / "fbref_mapping_coverage_after_alias_v1.csv"

OUT_MATRIX_V2 = Path("data/processed/features/football_feature_matrix_v4_2_fbref_prior_season_partial_v2.csv")
BUILD_REPORT_V2 = REPORT_DIR / "feature_matrix_v4_2_fbref_v2_build_report.md"
COVERAGE_V2 = REPORT_DIR / "feature_matrix_v4_2_fbref_v2_coverage_by_league_season.csv"
MISSINGNESS_V2 = REPORT_DIR / "feature_matrix_v4_2_fbref_v2_missingness.csv"
LEAKAGE_V2 = REPORT_DIR / "feature_matrix_v4_2_fbref_v2_leakage_checks.csv"
SCOPE_V2 = REPORT_DIR / "feature_matrix_v4_2_fbref_v2_recommended_model_scope.md"

LOCKED_ROW_PREDICTIONS = Path("outputs/reports/feature_matrix_v2_tm_1x2_locked_row_predictions.csv")
LOCKED_SELECTED_BETS_CANDIDATES = [
    Path("outputs/reports/feature_matrix_v3_clubelo_locked_selected_bets.csv"),
    Path("outputs/reports/feature_matrix_v2_tm_1x2_locked_selected_bets.csv"),
]


IMPROVED_ALIASES = [
    ("E0", "Leicester", "Leicester City", "shortened_name_vs_full_name"),
    ("E0", "Leeds", "Leeds United", "shortened_name_vs_full_name"),
    ("E0", "Sheffield United", "Sheffield Utd", "full_name_vs_fbref_abbreviation"),
    ("E0", "Norwich", "Norwich City", "shortened_name_vs_full_name"),
    ("E0", "Ipswich", "Ipswich Town", "shortened_name_vs_full_name"),
    ("E0", "Luton", "Luton Town", "shortened_name_vs_full_name"),
    ("E0", "Stoke", "Stoke City", "shortened_name_vs_full_name"),
    ("E0", "Swansea", "Swansea City", "shortened_name_vs_full_name"),
    ("E0", "Cardiff", "Cardiff City", "shortened_name_vs_full_name"),
    ("D1", "Darmstadt", "Darmstadt 98", "shortened_name_vs_numbered_name"),
    ("D1", "Paderborn", "Paderborn 07", "shortened_name_vs_numbered_name"),
    ("D1", "Fortuna Dusseldorf", "Düsseldorf", "english_ascii_full_name_vs_fbref_short_local_name"),
    ("I1", "Verona", "Hellas Verona", "shortened_name_vs_full_name"),
]

MANUAL_REVIEW = [
    ("I1", "Pisa", "", "not present in available prior-season Serie A FBref profiles"),
    ("SP1", "Oviedo", "", "not present in available prior-season La Liga FBref profiles"),
    ("E0", "Hull", "", "could be Hull City historically, but no recent test-season impact; leave manual"),
    ("F1", "Ajaccio GFCO", "", "ambiguous Ajaccio/GFC Ajaccio naming; leave manual"),
]


def fingerprint(path: Path) -> tuple[int, int, str]:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return path.stat().st_size, int(path.stat().st_mtime), h.hexdigest()


def locked_ids(path: Path) -> tuple[set[object], str]:
    if not path.exists():
        return set(), "unavailable"
    return set(pd.read_csv(path, usecols=["match_id"])["match_id"]), str(path)


def selected_ids() -> tuple[set[object], str]:
    for path in LOCKED_SELECTED_BETS_CANDIDATES:
        if path.exists():
            return set(pd.read_csv(path, usecols=["match_id"])["match_id"]), str(path)
    return set(), "unavailable"


def make_alias_frames(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_pairs = set(zip(profiles["comp"], profiles["squad"]))
    existing = pd.DataFrame(base.ACCEPTED_ALIASES, columns=["league", "football_data_team", "fbref_team_name", "reason"])
    improved = pd.DataFrame(IMPROVED_ALIASES, columns=["league", "football_data_team", "fbref_team_name", "reason"])
    alias_all = pd.concat([existing, improved], ignore_index=True).drop_duplicates(["league", "football_data_team", "fbref_team_name"])
    alias_all["comp"] = alias_all["league"].map(base.LEAGUE_TO_FBREF)
    alias_all["alias_status"] = np.where(
        [pair in valid_pairs for pair in zip(alias_all["comp"], alias_all["fbref_team_name"])],
        "accepted_high_confidence_aliases_v1",
        "rejected_or_ambiguous",
    )
    alias_all["confidence"] = np.where(alias_all["alias_status"].eq("accepted_high_confidence_aliases_v1"), "high", "none")
    manual = pd.DataFrame(MANUAL_REVIEW, columns=["league", "football_data_team", "fbref_team_name", "reason"])
    manual["alias_status"] = "manual_review_required"
    manual["confidence"] = "manual_review"
    accepted = alias_all[alias_all["alias_status"].eq("accepted_high_confidence_aliases_v1")].copy()
    rejected = alias_all[alias_all["alias_status"].eq("rejected_or_ambiguous")].copy()
    return alias_all, accepted, manual, rejected


def impact_table(v3: pd.DataFrame, mapping: pd.DataFrame, locked: set[object], selected: set[object]) -> pd.DataFrame:
    teams = mapping.drop_duplicates(["league", "football_team"])
    missing = teams[
        teams["league"].isin(base.TOP5)
        & ~teams["mapping_status"].isin(["accepted_exact_normalized", "accepted_high_confidence_fuzzy", "accepted_high_confidence_aliases_v1"])
    ].copy()
    rows = []
    for rec in missing.to_dict("records"):
        mask = v3["league"].eq(rec["league"]) & (
            v3["home_team"].eq(rec["football_team"]) | v3["away_team"].eq(rec["football_team"])
        )
        rows.append(
            {
                "league": rec["league"],
                "football_data_team": rec["football_team"],
                "mapping_status_before": rec["mapping_status"],
                "top_candidate": rec.get("candidate_squad", ""),
                "top_candidate_score": rec.get("candidate_score", np.nan),
                "fixture_rows_impacted": int(mask.sum()),
                "test_2020_2025_rows_impacted": int((mask & v3["season_start_year"].between(2020, 2025)).sum()),
                "locked_prediction_rows_impacted": int((mask & v3["match_id"].isin(locked)).sum()) if locked else 0,
                "locked_selected_bets_impacted": int((mask & v3["match_id"].isin(selected)).sum()) if selected else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["test_2020_2025_rows_impacted", "fixture_rows_impacted"], ascending=[False, False])


def apply_aliases_to_base(aliases: list[tuple[str, str, str, str]]) -> None:
    existing = {(a, b, c) for a, b, c, _ in base.ACCEPTED_ALIASES}
    for row in aliases:
        key = row[:3]
        if key not in existing:
            base.ACCEPTED_ALIASES.append(row)


def make_coverage(v4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(base.segment_summary(v4, "all_rows"))
    rows.extend(base.segment_summary(v4[v4["league"].isin(base.TOP5)], "top5_only"))
    rows.extend(base.segment_summary(v4[v4["league"].isin(base.TOP5) & v4["season_start_year"].between(2020, 2025)], "top5_test_2020_2025"))
    if LOCKED_ROW_PREDICTIONS.exists():
        ids = set(pd.read_csv(LOCKED_ROW_PREDICTIONS, usecols=["match_id"])["match_id"])
        rows.extend(base.segment_summary(v4[v4["match_id"].isin(ids)], "locked_v3_prediction_row_universe"))
    selected, _ = selected_ids()
    if selected:
        rows.extend(base.segment_summary(v4[v4["match_id"].isin(selected)], "locked_v3_selected_bets"))
    rows.extend(base.segment_summary(v4, "by_league", "league"))
    rows.extend(base.segment_summary(v4, "by_season_start_year", "season_start_year"))
    return pd.DataFrame(rows)


def coverage_rate(cov: pd.DataFrame, segment: str) -> float:
    row = cov[cov["segment"].eq(segment)]
    return float(row.iloc[0]["both_available_rate"]) if len(row) else float("nan")


def write_outputs(
    decision: str,
    impact: pd.DataFrame,
    alias_all: pd.DataFrame,
    accepted: pd.DataFrame,
    manual: pd.DataFrame,
    rejected: pd.DataFrame,
    before_cov: pd.DataFrame,
    after_cov: pd.DataFrame,
    checks: pd.DataFrame,
    rebuilt: bool,
    locked_source: str,
    selected_source: str,
) -> None:
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    alias_file = pd.concat([accepted, manual, rejected], ignore_index=True, sort=False)[
        ["league", "football_data_team", "fbref_team_name", "alias_status", "confidence", "reason"]
    ]
    alias_file.to_csv(ALIAS_MAPPING_CSV, index=False)
    accepted[["league", "football_data_team", "fbref_team_name", "alias_status", "confidence", "reason"]].to_csv(ACCEPTED_CSV, index=False)
    manual[["league", "football_data_team", "fbref_team_name", "alias_status", "confidence", "reason"]].to_csv(MANUAL_CSV, index=False)
    rejected[["league", "football_data_team", "fbref_team_name", "alias_status", "confidence", "reason"]].to_csv(REJECTED_CSV, index=False)
    after_cov.to_csv(COVERAGE_AFTER_CSV, index=False)

    key_segments = ["all_rows", "top5_only", "top5_test_2020_2025", "locked_v3_prediction_row_universe", "locked_v3_selected_bets"]
    before_key = before_cov[before_cov["segment"].isin(key_segments)][["segment", "rows", "both_available", "both_available_rate"]]
    after_key = after_cov[after_cov["segment"].isin(key_segments)][["segment", "rows", "both_available", "both_available_rate"]]
    delta = after_key.merge(before_key, on="segment", suffixes=("_after", "_before"), how="left")
    delta["both_available_delta"] = delta["both_available_after"] - delta["both_available_before"]
    delta["both_available_rate_delta"] = delta["both_available_rate_after"] - delta["both_available_rate_before"]

    IMPACT_MD.write_text(
        "\n".join(
            [
                "# FBref Alias V1 Impact Audit",
                "",
                f"Decision: `{decision}`",
                "",
                "Aliases were same-league only and limited to high-confidence naming variants. No modeling, value search, same-season aggregate use, or raw ZIP modification was performed.",
                "",
                f"- Locked prediction source: `{locked_source}`",
                f"- Locked selected-bets source: `{selected_source}`",
                "",
                "## Highest-Impact Unmatched Teams Before Alias V1",
                base.md_table(impact.head(50), 50),
                "",
                "## Coverage Delta",
                base.md_table(delta, 20),
                "",
                "## Accepted Aliases",
                base.md_table(accepted[["league", "football_data_team", "fbref_team_name", "reason"]], 80),
                "",
            ]
        ),
        encoding="utf-8",
    )
    BUILD_REPORT_V2.write_text(
        "\n".join(
            [
                "# Feature Matrix V4.2 FBref V2 Alias Build Report",
                "",
                f"Decision: `{decision}`",
                "",
                "No predictive models, value searches, threshold optimization, Understat features, or locked v3 candidate changes were run. No confirmed edge is claimed.",
                "",
                f"- Rebuilt matrix: `{OUT_MATRIX_V2}`" if rebuilt else "- Rebuilt matrix: not written because coverage did not materially improve.",
                f"- Accepted high-confidence aliases: {len(accepted)}",
                f"- Manual-review aliases: {len(manual)}",
                f"- Rejected/ambiguous aliases: {len(rejected)}",
                "",
                "## Key Coverage Delta",
                base.md_table(delta, 20),
                "",
                "## Leakage Checks",
                base.md_table(checks, 60),
                "",
            ]
        ),
        encoding="utf-8",
    )
    SCOPE_V2.write_text(
        "\n".join(
            [
                "# V4.2 FBref V2 Recommended Model Scope",
                "",
                f"Decision: `{decision}`",
                "",
                "Recommended safe scope: top-five leagues with prior completed FBref profile coverage and explicit missing flags.",
                "",
                "Use only fixture season `Y` joined to FBref season `Y-1`. Do not use same-season aggregates, player/team/squad strings, Understat, or current-match data.",
                "",
                "No confirmed edge is claimed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    before = {p: fingerprint(p) for p in [base.FBREF_ZIP_2017_2024, base.FBREF_ZIP_2024_2025]}
    v3 = pd.read_csv(V3_MATRIX, low_memory=False)
    v3["match_date"] = pd.to_datetime(v3["match_date"], errors="coerce")
    players, _schema = base.standardize_player_seasons()
    profiles = base.aggregate_profiles(players)
    profiles["fbref_prev_profile_available_flag"] = True
    profiles["fbref_prev_profile_season_gap"] = 1
    profiles["fbref_prev_missing_flag"] = False

    mapping_before = pd.read_csv(base.MAPPING_CSV)
    locked, locked_source = locked_ids(LOCKED_ROW_PREDICTIONS)
    selected, selected_source = selected_ids()
    impact = impact_table(v3, mapping_before, locked, selected)

    alias_all, accepted_aliases, manual_aliases, rejected_aliases = make_alias_frames(profiles)
    apply_aliases_to_base(IMPROVED_ALIASES)
    base.ALIASES_ACCEPTED_CSV = ACCEPTED_CSV
    base.ALIASES_MANUAL_CSV = MANUAL_CSV
    base.MAPPING_CSV = REPORT_DIR / "_fbref_team_mapping_candidates_alias_v1_tmp.csv"
    _mapping_after, accepted, _, _ = base.build_mapping(v3, profiles)
    features, audit = base.attach_profiles(v3, accepted, profiles)
    v4 = pd.concat([v3.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    after_cov = make_coverage(v4)
    before_cov = pd.read_csv(base.COVERAGE_CSV)
    material = (
        coverage_rate(after_cov, "top5_test_2020_2025") - coverage_rate(before_cov, "top5_test_2020_2025") >= 0.03
        and coverage_rate(after_cov, "locked_v3_prediction_row_universe") - coverage_rate(before_cov, "locked_v3_prediction_row_universe") >= 0.03
    )
    after = {p: fingerprint(p) for p in [base.FBREF_ZIP_2017_2024, base.FBREF_ZIP_2024_2025]}
    checks = base.leakage_checks(v3.reset_index(drop=True), v4, features, audit, before, after)
    checks.to_csv(LEAKAGE_V2, index=False)
    rebuilt = False
    if checks["status"].ne("pass").any() or not material:
        decision = "v4_2_fbref_aliases_no_material_improvement"
        after_cov.to_csv(COVERAGE_V2, index=False)
        missing = pd.DataFrame(
            [{"column": c, "missing": int(features[c].isna().sum()), "missing_rate": float(features[c].isna().mean()), "non_null": int(features[c].notna().sum())} for c in features.columns]
        )
        missing.to_csv(MISSINGNESS_V2, index=False)
    else:
        rebuilt = True
        v4.to_csv(OUT_MATRIX_V2, index=False)
        after_cov.to_csv(COVERAGE_V2, index=False)
        missing = pd.DataFrame(
            [{"column": c, "missing": int(features[c].isna().sum()), "missing_rate": float(features[c].isna().mean()), "non_null": int(features[c].notna().sum())} for c in features.columns]
        )
        missing.to_csv(MISSINGNESS_V2, index=False)
        if coverage_rate(after_cov, "top5_only") >= 0.90 and coverage_rate(after_cov, "top5_test_2020_2025") >= 0.90:
            decision = "v4_2_fbref_feature_build_ready_good"
        else:
            decision = "v4_2_fbref_feature_build_ready_partial"

    write_outputs(
        decision=decision,
        impact=impact,
        alias_all=alias_all,
        accepted=accepted_aliases,
        manual=manual_aliases,
        rejected=rejected_aliases,
        before_cov=before_cov,
        after_cov=after_cov,
        checks=checks,
        rebuilt=rebuilt,
        locked_source=locked_source,
        selected_source=selected_source,
    )
    key = after_cov[after_cov["segment"].isin(["top5_only", "top5_test_2020_2025", "locked_v3_prediction_row_universe", "locked_v3_selected_bets"])]
    print(
        {
            "decision": decision,
            "rebuilt": rebuilt,
            "accepted_aliases": len(accepted_aliases),
            "coverage_after": {r.segment: round(float(r.both_available_rate), 6) for r in key.itertuples(index=False)},
            "failed_checks": int(checks["status"].ne("pass").sum()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
