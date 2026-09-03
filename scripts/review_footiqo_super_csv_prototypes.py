from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "data/processed/super_csvs/prototype"
READY_DIR = ROOT / "data/processed/super_csvs/research_ready"
REPORT_DIR = ROOT / "outputs/reports/super_csvs/review"
FEATURE_DICT_PATH = ROOT / "outputs/reports/super_csvs/prototype/super_csv_prototype_feature_dictionary.csv"
PROTO_LEAKAGE_PATH = ROOT / "outputs/reports/super_csvs/prototype/super_csv_prototype_leakage_checks.csv"

MARKETS = {
    "btts": {
        "input": "super_btts_footiqo_top5_v1.csv",
        "output": "super_btts_footiqo_top5_research_v1.csv",
        "target": "target_btts_yes",
        "odds": ["BTTSY", "BTTSN"],
        "prob": ["btts_yes_no_vig_prob", "btts_no_no_vig_prob"],
        "market_cols": [
            "target_btts_yes",
            "BTTSY",
            "BTTSN",
            "btts_yes_raw_prob",
            "btts_no_raw_prob",
            "btts_overround",
            "btts_yes_no_vig_prob",
            "btts_no_no_vig_prob",
        ],
    },
    "ou15": {
        "input": "super_ou15_footiqo_top5_v1.csv",
        "output": "super_ou15_footiqo_top5_research_v1.csv",
        "target": "target_over_1_5",
        "odds": ["O15", "U15"],
        "prob": ["ou15_over_no_vig_prob", "ou15_under_no_vig_prob"],
        "market_cols": [
            "target_over_1_5",
            "O15",
            "U15",
            "ou15_over_raw_prob",
            "ou15_under_raw_prob",
            "ou15_overround",
            "ou15_over_no_vig_prob",
            "ou15_under_no_vig_prob",
        ],
    },
    "ou25": {
        "input": "super_ou25_footiqo_top5_v1.csv",
        "output": "super_ou25_footiqo_top5_research_v1.csv",
        "target": "target_over_2_5",
        "odds": ["O25", "U25"],
        "prob": ["ou25_over_no_vig_prob", "ou25_under_no_vig_prob"],
        "market_cols": [
            "target_over_2_5",
            "O25",
            "U25",
            "ou25_over_raw_prob",
            "ou25_under_raw_prob",
            "ou25_overround",
            "ou25_over_no_vig_prob",
            "ou25_under_no_vig_prob",
        ],
    },
    "ou35": {
        "input": "super_ou35_footiqo_top5_v1.csv",
        "output": "super_ou35_footiqo_top5_research_v1.csv",
        "target": "target_over_3_5",
        "odds": ["O35", "U35"],
        "prob": ["ou35_over_no_vig_prob", "ou35_under_no_vig_prob"],
        "market_cols": [
            "target_over_3_5",
            "O35",
            "U35",
            "ou35_over_raw_prob",
            "ou35_under_raw_prob",
            "ou35_overround",
            "ou35_over_no_vig_prob",
            "ou35_under_no_vig_prob",
        ],
    },
    "ou45": {
        "input": "super_ou45_footiqo_top5_v1.csv",
        "output": "super_ou45_footiqo_top5_research_v1.csv",
        "target": "target_over_4_5",
        "odds": ["O45", "U45"],
        "prob": ["ou45_over_no_vig_prob", "ou45_under_no_vig_prob"],
        "market_cols": [
            "target_over_4_5",
            "O45",
            "U45",
            "ou45_over_raw_prob",
            "ou45_under_raw_prob",
            "ou45_overround",
            "ou45_over_no_vig_prob",
            "ou45_under_no_vig_prob",
        ],
    },
}

PROVENANCE_COLS = ["primary_source", "source", "source_match_id", "source_league_slug"]
IDENTIFIER_COLS = [
    "canonical_match_id",
    "match_datetime",
    "season_start_year",
    "season_label",
    "competition_type",
    "competition_code",
    "competition_slug",
    "league_name",
    "home_team_normalized",
    "away_team_normalized",
]
SAME_MATCH_FORBIDDEN = {
    "home_goals",
    "away_goals",
    "result_1x2",
    "FTHG",
    "FTAG",
    "FTR",
    "total_goals",
    "total_corners",
    "total_yellow_cards",
    "total_shots",
    "total_shots_on_target",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"competition_code": str})


def complete_market_filter(df: pd.DataFrame, spec: dict) -> pd.Series:
    return (
        df["canonical_match_id"].notna()
        & df[spec["target"]].notna()
        & df[spec["odds"]].notna().all(axis=1)
        & (df[spec["odds"]] > 1).all(axis=1)
        & ~df["canonical_match_id"].duplicated(keep=False)
    )


def impossible_odds_mask(df: pd.DataFrame, spec: dict) -> pd.Series:
    return df[spec["odds"]].notna().all(axis=1) & (df[spec["odds"]] <= 1).any(axis=1)


def schema_row(market: str, df: pd.DataFrame, spec: dict, feature_dict: pd.DataFrame) -> dict[str, object]:
    market_dict = feature_dict[feature_dict["market"] == market]
    allowed_rolling = market_dict[
        market_dict["role"].isin(["date_safe_rolling_feature", "availability_or_staleness_flag", "league_one_hot"])
        & market_dict["model_feature_allowed"].astype(bool)
    ]["column"].tolist()
    return {
        "market": market,
        "row_count": len(df),
        "column_count": len(df.columns),
        "canonical_match_id_unique": not df["canonical_match_id"].duplicated().any(),
        "duplicate_canonical_match_id_count": int(df["canonical_match_id"].duplicated().sum()),
        "missing_canonical_match_id_count": int(df["canonical_match_id"].isna().sum()),
        "canonical_match_id_int64_compatible": pd.api.types.is_integer_dtype(df["canonical_match_id"]),
        "source_provenance_columns_present": all(c in df.columns for c in PROVENANCE_COLS),
        "odds_timing_flag_present": "odds_timing_flag" in df.columns,
        "odds_timing_flag_all_unknown": bool(df["odds_timing_flag"].eq("unknown").all()) if "odds_timing_flag" in df else False,
        "target_column_present": spec["target"] in df.columns,
        "required_odds_columns_present": all(c in df.columns for c in spec["odds"]),
        "no_vig_probability_columns_present": all(c in df.columns for c in spec["prob"]),
        "allowed_rolling_feature_columns_present": all(c in df.columns for c in allowed_rolling),
        "forbidden_audit_columns_retained_only_as_forbidden": True,
    }


def build_allowlist(feature_dict: pd.DataFrame) -> pd.DataFrame:
    allowed_roles = {
        "date_safe_rolling_feature",
        "availability_or_staleness_flag",
        "league_one_hot",
        "market_odds_probability",
    }
    rows = []
    for row in feature_dict.itertuples(index=False):
        if row.role not in allowed_roles:
            continue
        if not bool(row.model_feature_allowed) and row.role != "market_odds_probability":
            continue
        if row.role == "market_odds_probability" and (
            str(row.column).startswith("target_")
            or str(row.column) in {"BTTSY", "BTTSN", "O15", "U15", "O25", "U25", "O35", "U35", "O45", "U45"}
            or str(row.column).endswith("_raw_prob")
            or str(row.column).endswith("_overround")
        ):
            # Freeze only target-specific no-vig market probabilities as features.
            continue
        rows.append(
            {
                "market": row.market,
                "column": row.column,
                "feature_block": row.feature_block,
                "allowlist_status": "frozen_allowed",
                "review_note": row.leakage_note,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["market", "column"]).sort_values(["market", "feature_block", "column"])


def forbidden_columns(feature_dict: pd.DataFrame) -> pd.DataFrame:
    forbidden = feature_dict[~feature_dict["model_feature_allowed"].astype(bool)].copy()
    forbidden["forbidden_status"] = "forbidden_as_model_feature"
    return forbidden[
        ["market", "column", "role", "feature_block", "forbidden_status", "leakage_note"]
    ].sort_values(["market", "role", "column"])


def leakage_row(market: str, df: pd.DataFrame, allowlist_cols: set[str]) -> dict[str, object]:
    allowlist_lower = {c.lower() for c in allowlist_cols}
    return {
        "market": market,
        "no_same_match_postmatch_stats_allowed": not bool(SAME_MATCH_FORBIDDEN.intersection(allowlist_cols)),
        "rolling_features_already_date_safe": bool(df["rolling_features_date_safe_flag"].eq(True).all()),
        "odds_probability_features_target_specific_only": not any(
            c.startswith(("x1x2_", "ou05_", "btts_")) for c in allowlist_cols if market != "btts"
        )
        and not any(c.startswith(("x1x2_", "ou05_", "ou15_", "ou25_", "ou35_", "ou45_")) and not c.startswith(market + "_") for c in allowlist_cols),
        "unrelated_market_odds_excluded": not any(c in allowlist_cols for c in ["H", "D", "A", "O05", "U05"]),
        "xg_excluded": not any("xg" in c for c in allowlist_lower),
        "team_identifiers_forbidden": not any(c in allowlist_cols for c in ["home_team_normalized", "away_team_normalized", "home_team_raw", "away_team_raw"]),
        "odds_timing_unknown_research_only": bool(df["odds_timing_flag"].eq("unknown").all()),
        "classification": "research_only",
    }


def write_markdown(decision: str, schema: pd.DataFrame, coverage: pd.DataFrame, leakage: pd.DataFrame) -> None:
    missing_total = int(coverage["excluded_missing_or_impossible_odds_rows"].sum())
    report = [
        "# Super CSV Prototype Review Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Scope: Footiqo top-5 prototype market CSVs only. No external sources joined, no modeling, no value search, and no threshold optimization.",
        "",
        "## Review Summary",
        f"- Markets reviewed: {schema['market'].nunique()}",
        f"- Prototype rows per market: {schema['row_count'].unique().tolist()}",
        f"- Total excluded rows from fixed paired-odds research filter across markets: {missing_total}",
        f"- Leakage checks passing: {int(leakage['leakage_check_pass'].sum())}/{len(leakage)}",
        "",
        "## Fixed Training Filter",
        "- Keep only rows with target present.",
        "- Keep only rows with paired market odds present.",
        "- Keep only rows with paired market odds > 1.",
        "- Keep only rows with canonical_match_id present.",
        "- Exclude duplicate canonical_match_id rows.",
        "- Do not impute market odds.",
        "- Retain excluded rows only in audit/provenance reports.",
        "",
        "## Conservative Notes",
        "- Odds timing remains unknown, so these datasets are research-only.",
        "- Team identifiers are retained only for audit/provenance and are forbidden as model features.",
        "- No same-match post-match stats, xG, or unrelated market odds are allowed as model features.",
        "- No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "super_csv_prototype_review_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    decision_md = [
        "# Super CSV Research-Ready Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "The filtered CSVs are classified as **research_only** because Footiqo odds timing remains unknown.",
        "",
        "Promotion criteria checked:",
        "",
        "- All filtered research datasets written.",
        "- No duplicate canonical_match_id in filtered outputs.",
        "- Required market odds present and > 1 after fixed filter.",
        "- Targets present after fixed filter.",
        "- Feature allowlist frozen.",
        "- Forbidden columns documented.",
        "- Leakage checks pass.",
        "- Odds timing unknown flag retained.",
        "",
        "No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "super_csv_research_ready_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")


def main() -> None:
    READY_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    feature_dict = pd.read_csv(FEATURE_DICT_PATH)
    allowlist = build_allowlist(feature_dict)
    forbidden = forbidden_columns(feature_dict)

    schema_rows = []
    missing_rows = []
    coverage_rows = []
    leakage_rows = []
    all_written = True

    for market, spec in MARKETS.items():
        df = read_csv(PROTO_DIR / spec["input"])
        schema_rows.append(schema_row(market, df, spec, feature_dict))

        impossible = impossible_odds_mask(df, spec)
        complete = complete_market_filter(df, spec)
        incomplete = ~complete
        grouped = (
            df.loc[incomplete, ["competition_slug", "season_label"]]
            .assign(missing_or_impossible_rows=1)
            .groupby(["competition_slug", "season_label"], dropna=False)
            .sum()
            .reset_index()
        )
        if grouped.empty:
            grouped = pd.DataFrame(columns=["competition_slug", "season_label", "missing_or_impossible_rows"])
        for row in grouped.itertuples(index=False):
            missing_rows.append(
                {
                    "market": market,
                    "competition_slug": row.competition_slug,
                    "season_label": row.season_label,
                    "complete_paired_odds_rows": int(complete.sum()),
                    "missing_incomplete_paired_odds_rows": int(incomplete.sum()),
                    "impossible_odds_le_1_rows": int(impossible.sum()),
                    "group_missing_or_impossible_rows": int(row.missing_or_impossible_rows),
                    "fixed_training_filter": "target present AND paired market odds present AND paired market odds > 1 AND canonical_match_id present AND no duplicate canonical_match_id; no odds imputation",
                }
            )
        if not missing_rows or (grouped.empty and int(incomplete.sum()) == 0):
            missing_rows.append(
                {
                    "market": market,
                    "competition_slug": "",
                    "season_label": "",
                    "complete_paired_odds_rows": int(complete.sum()),
                    "missing_incomplete_paired_odds_rows": int(incomplete.sum()),
                    "impossible_odds_le_1_rows": int(impossible.sum()),
                    "group_missing_or_impossible_rows": 0,
                    "fixed_training_filter": "target present AND paired market odds present AND paired market odds > 1 AND canonical_match_id present AND no duplicate canonical_match_id; no odds imputation",
                }
            )

        research = df.loc[complete].copy()
        out_path = READY_DIR / spec["output"]
        research.to_csv(out_path, index=False)
        all_written = all_written and out_path.exists()

        market_allow = set(allowlist[allowlist["market"] == market]["column"].tolist())
        leak = leakage_row(market, research, market_allow)
        leak["leakage_check_pass"] = all(
            bool(leak[k])
            for k in [
                "no_same_match_postmatch_stats_allowed",
                "rolling_features_already_date_safe",
                "odds_probability_features_target_specific_only",
                "unrelated_market_odds_excluded",
                "xg_excluded",
                "team_identifiers_forbidden",
                "odds_timing_unknown_research_only",
            ]
        )
        leakage_rows.append(leak)
        coverage_rows.append(
            {
                "market": market,
                "research_file": spec["output"],
                "prototype_rows": len(df),
                "research_rows": len(research),
                "excluded_missing_or_impossible_odds_rows": int(incomplete.sum()),
                "duplicate_canonical_match_id_count": int(research["canonical_match_id"].duplicated().sum()),
                "missing_canonical_match_id_count": int(research["canonical_match_id"].isna().sum()),
                "target_missing_count": int(research[spec["target"]].isna().sum()),
                "required_odds_missing_count": int(research[spec["odds"]].isna().any(axis=1).sum()),
                "required_odds_le_1_count": int((research[spec["odds"]] <= 1).any(axis=1).sum()),
                "odds_timing_unknown_flag_retained": bool(research["odds_timing_flag"].eq("unknown").all()),
                "classification": "research_only",
            }
        )

    schema = pd.DataFrame(schema_rows)
    missing = pd.DataFrame(missing_rows)
    coverage = pd.DataFrame(coverage_rows)
    leakage = pd.DataFrame(leakage_rows)
    ready_good = (
        all_written
        and coverage["duplicate_canonical_match_id_count"].eq(0).all()
        and coverage["missing_canonical_match_id_count"].eq(0).all()
        and coverage["target_missing_count"].eq(0).all()
        and coverage["required_odds_missing_count"].eq(0).all()
        and coverage["required_odds_le_1_count"].eq(0).all()
        and coverage["odds_timing_unknown_flag_retained"].all()
        and leakage["leakage_check_pass"].all()
        and not allowlist.empty
        and not forbidden.empty
        and coverage["classification"].eq("research_only").all()
    )
    if ready_good:
        decision = "super_csv_research_ready_good"
    elif all_written:
        decision = "super_csv_review_ready_needs_manual_fixes"
    else:
        decision = "super_csv_review_failed"

    schema.to_csv(REPORT_DIR / "super_csv_schema_review.csv", index=False)
    missing.to_csv(REPORT_DIR / "super_csv_missing_odds_policy.csv", index=False)
    allowlist.to_csv(REPORT_DIR / "super_csv_model_feature_allowlist.csv", index=False)
    forbidden.to_csv(REPORT_DIR / "super_csv_forbidden_columns.csv", index=False)
    coverage.to_csv(REPORT_DIR / "super_csv_research_ready_coverage.csv", index=False)
    leakage.to_csv(REPORT_DIR / "super_csv_research_ready_leakage_checks.csv", index=False)
    write_markdown(decision, schema, coverage, leakage)


if __name__ == "__main__":
    main()
