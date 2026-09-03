from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data/processed/match_registry/canonical_match_registry_v1_prototype.csv"
SOURCE_MAP_PATH = ROOT / "data/processed/match_registry/source_match_map_v1_prototype.csv"
FEATURES_PATH = ROOT / "data/processed/footiqo/footiqo_top5_rolling_features_v1.csv"
OUT_DIR = ROOT / "data/processed/super_csvs/prototype"
REPORT_DIR = ROOT / "outputs/reports/super_csvs/prototype"

IDENTIFIER_COLUMNS = [
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

PROVENANCE_COLUMNS = [
    "primary_source",
    "source",
    "source_match_id",
    "source_league_slug",
]

MARKETS = {
    "btts": {
        "output": "super_btts_footiqo_top5_v1.csv",
        "target": "target_btts_yes",
        "odds": ["BTTSY", "BTTSN"],
        "rename": {
            "btts_yes_raw_implied_prob": "btts_yes_raw_prob",
            "btts_no_raw_implied_prob": "btts_no_raw_prob",
        },
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
        "output": "super_ou15_footiqo_top5_v1.csv",
        "target": "target_over_1_5",
        "odds": ["O15", "U15"],
        "rename": {
            "ou15_over_raw_implied_prob": "ou15_over_raw_prob",
            "ou15_under_raw_implied_prob": "ou15_under_raw_prob",
        },
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
        "output": "super_ou25_footiqo_top5_v1.csv",
        "target": "target_over_2_5",
        "odds": ["O25", "U25"],
        "rename": {
            "ou25_over_raw_implied_prob": "ou25_over_raw_prob",
            "ou25_under_raw_implied_prob": "ou25_under_raw_prob",
        },
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
        "output": "super_ou35_footiqo_top5_v1.csv",
        "target": "target_over_3_5",
        "odds": ["O35", "U35"],
        "rename": {
            "ou35_over_raw_implied_prob": "ou35_over_raw_prob",
            "ou35_under_raw_implied_prob": "ou35_under_raw_prob",
        },
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
        "output": "super_ou45_footiqo_top5_v1.csv",
        "target": "target_over_4_5",
        "odds": ["O45", "U45"],
        "rename": {
            "ou45_over_raw_implied_prob": "ou45_over_raw_prob",
            "ou45_under_raw_implied_prob": "ou45_under_raw_prob",
        },
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

FORBIDDEN_SOURCE_COLUMNS = {
    "homeTeam",
    "awayTeam",
    "home_team_raw",
    "away_team_raw",
    "FTHG",
    "FTAG",
    "FTR",
    "home_goals",
    "away_goals",
    "result_1x2",
    "total_goals",
    "total_corners",
    "total_yellow_cards",
    "total_shots",
    "total_shots_on_target",
    "H",
    "D",
    "A",
    "O05",
    "U05",
    "target_home_win",
    "target_draw",
    "target_away_win",
    "target_over_0_5",
}

UNRELATED_MARKET_PREFIXES = ["x1x2_", "ou05_"]
UNRELATED_MARKET_EXACT = {"H", "D", "A", "O05", "U05"}


def rolling_feature_columns(features: pd.DataFrame) -> list[str]:
    cols = []
    for col in features.columns:
        if col in FORBIDDEN_SOURCE_COLUMNS:
            continue
        if col in {"id", "source_league_slug", "matchDate", "match_datetime", "Country", "League", "Season", "season_start_year"}:
            continue
        if col.startswith("target_"):
            continue
        if col in {"BTTSY", "BTTSN", "O15", "U15", "O25", "U25", "O35", "U35", "O45", "U45"}:
            continue
        if any(col.startswith(prefix) for prefix in ["btts_", "ou15_", "ou25_", "ou35_", "ou45_", *UNRELATED_MARKET_PREFIXES]):
            continue
        if col in UNRELATED_MARKET_EXACT:
            continue
        if (
            col.startswith("home_")
            or col.startswith("away_")
            or col.startswith("home_minus_away_")
            or col.startswith("league_")
            or col in {"match_week_index", "both_history_available_flag"}
        ):
            cols.append(col)
    return cols


def load_base() -> tuple[pd.DataFrame, list[str]]:
    registry = pd.read_csv(REGISTRY_PATH, dtype={"competition_code": str})
    source_map = pd.read_csv(SOURCE_MAP_PATH)
    features = pd.read_csv(FEATURES_PATH)
    registry["canonical_match_id"] = registry["canonical_match_id"].astype("int64")
    source_map["canonical_match_id"] = source_map["canonical_match_id"].astype("int64")
    features["id"] = features["id"].astype("int64")

    base = registry[IDENTIFIER_COLUMNS + ["primary_source"]].merge(
        source_map[["canonical_match_id", "source", "source_match_id", "source_league_slug"]],
        on="canonical_match_id",
        how="left",
        validate="one_to_one",
    )
    joined = base.merge(
        features,
        left_on=["source_match_id", "source_league_slug"],
        right_on=["id", "source_league_slug"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )
    return joined, rolling_feature_columns(features)


def valid_probability(series: pd.Series) -> pd.Series:
    return series.notna() & series.between(0, 1, inclusive="both")


def build_market_frame(base: pd.DataFrame, rolling_cols: list[str], market: str, spec: dict) -> pd.DataFrame:
    frame = base.copy()
    frame = frame.rename(columns=spec["rename"])
    odds_cols = spec["odds"]
    prob_cols = [c for c in spec["market_cols"] if c.endswith("_prob")]
    overround_cols = [c for c in spec["market_cols"] if c.endswith("_overround")]
    frame[f"{market}_paired_odds_available_flag"] = frame[odds_cols].notna().all(axis=1)
    frame[f"{market}_probabilities_available_flag"] = frame[prob_cols + overround_cols].notna().all(axis=1)
    frame["odds_timing_flag"] = "unknown"
    frame["rolling_features_date_safe_flag"] = True
    frame["external_sources_joined_flag"] = False

    output_cols = (
        ["canonical_match_id"]
        + [c for c in PROVENANCE_COLUMNS if c != "primary_source"]
        + ["primary_source"]
        + [c for c in IDENTIFIER_COLUMNS if c != "canonical_match_id"]
        + spec["market_cols"]
        + [
            f"{market}_paired_odds_available_flag",
            f"{market}_probabilities_available_flag",
            "odds_timing_flag",
            "rolling_features_date_safe_flag",
            "external_sources_joined_flag",
        ]
        + rolling_cols
    )
    return frame[output_cols].copy()


def validate_market(df: pd.DataFrame, market: str, spec: dict, expected_rows: int) -> dict[str, object]:
    odds_cols = spec["odds"]
    prob_cols = [c for c in spec["market_cols"] if c.endswith("_prob")]
    overround_cols = [c for c in spec["market_cols"] if c.endswith("_overround")]
    valid_odds = df[odds_cols].notna().all(axis=1) & (df[odds_cols] > 1.0).all(axis=1)
    valid_probs = pd.Series(True, index=df.index)
    for col in prob_cols:
        valid_probs &= valid_probability(df[col])
    for col in overround_cols:
        valid_probs &= df[col].notna() & (df[col] > 0)
    forbidden_present = sorted((FORBIDDEN_SOURCE_COLUMNS | UNRELATED_MARKET_EXACT).intersection(df.columns))
    forbidden_model_feature_present = [
        c for c in forbidden_present if c not in {"home_team_normalized", "away_team_normalized"}
    ]
    return {
        "market": market,
        "file_name": spec["output"],
        "row_count": len(df),
        "expected_universe_rows": expected_rows,
        "row_count_matches_expected_universe": len(df) == expected_rows,
        "canonical_match_id_unique": not df["canonical_match_id"].duplicated().any(),
        "missing_canonical_match_id_count": int(df["canonical_match_id"].isna().sum()),
        "canonical_match_id_int64_compatible": pd.api.types.is_integer_dtype(df["canonical_match_id"]),
        "valid_paired_odds_rows": int(valid_odds.sum()),
        "missing_or_invalid_paired_odds_rows": int((~valid_odds).sum()),
        "valid_probability_rows": int(valid_probs.sum()),
        "missing_or_invalid_probability_rows": int((~valid_probs).sum()),
        "target_available_rows": int(df[spec["target"]].notna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "forbidden_columns_present": "; ".join(forbidden_present),
        "forbidden_model_feature_columns_present": "; ".join(forbidden_model_feature_present),
        "odds_timing_flag_present": "odds_timing_flag" in df.columns,
        "odds_timing_flag_all_unknown": bool(df["odds_timing_flag"].eq("unknown").all()),
        "source_provenance_present": all(c in df.columns for c in PROVENANCE_COLUMNS),
        "leakage_check_pass": len(forbidden_model_feature_present) == 0
        and "odds_timing_flag" in df.columns
        and df["odds_timing_flag"].eq("unknown").all()
        and df["external_sources_joined_flag"].eq(False).all(),
    }


def build_feature_dictionary(market_frames: dict[str, pd.DataFrame], rolling_cols: list[str]) -> pd.DataFrame:
    rows = []
    all_cols = []
    for market, df in market_frames.items():
        for col in df.columns:
            all_cols.append((market, col))
    for market, col in all_cols:
        if col in PROVENANCE_COLUMNS:
            role, allowed, block, note = "source_provenance", False, "provenance", "Preserved for audit; not a model feature."
        elif col in IDENTIFIER_COLUMNS:
            role, allowed, block, note = "identifier", False, "canonical_identifier", "Identifier only; team names/ids are forbidden as model features."
        elif col.startswith("league_"):
            role, allowed, block, note = "league_one_hot", True, "league_one_hot", "Allowed categorical league indicator."
        elif col in rolling_cols:
            role, allowed, block, note = "date_safe_rolling_feature", True, "rolling_team_features", "Allowed only because this input was already built date-safe."
        elif col.endswith("_flag") or col in {"odds_timing_flag"}:
            role, allowed, block, note = "availability_or_staleness_flag", True, "flags", "Allowed control flag; odds timing remains unknown."
        elif col.startswith("target_"):
            role, allowed, block, note = "target_label", False, "market_target", "Target label; forbidden as predictor."
        elif col in {"BTTSY", "BTTSN", "O15", "U15", "O25", "U25", "O35", "U35", "O45", "U45"} or col.endswith("_prob") or col.endswith("_overround"):
            role, allowed, block, note = "market_odds_probability", False, "market_context", "Market-specific odds/probability column; use only under audited protocol."
        else:
            role, allowed, block, note = "unknown_review", False, "unknown", "Requires manual review."
        rows.append(
            {
                "market": market,
                "column": col,
                "role": role,
                "model_feature_allowed": allowed,
                "feature_block": block,
                "leakage_note": note,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["market", "column"])


def write_reports(coverage: pd.DataFrame, feature_dict: pd.DataFrame, leakage: pd.DataFrame, decision: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    written = coverage["file_name"].tolist()
    report = [
        "# Footiqo Top-5 Super CSV Prototype Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Scope: Footiqo top-5 only, keyed by `canonical_match_id`. No external sources were joined. No models, value searches, or threshold optimization were run.",
        "",
        "## Files Written",
        *[f"- data/processed/super_csvs/prototype/{name}" for name in written],
        "",
        "## Notes",
        "- Every output preserves source provenance and canonical identifiers.",
        "- Rows are not filtered by market availability; paired odds availability flags identify missing odds rows.",
        "- Team identifiers are preserved for audit only and marked forbidden as model features.",
        "- Same-match scores, results, corners, cards, shots, possession, and xG are excluded as model features.",
        "- Odds timing remains unknown and is explicitly flagged.",
        "- No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "super_csv_prototype_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    rec = [
        "# Super CSV Prototype Recommendation",
        "",
        f"Decision: **{decision}**",
        "",
        "The prototype files are suitable for manual review and downstream schema testing. They should not be used for modeling until odds timing is documented and rows with missing paired odds are handled by a fixed, predeclared policy.",
        "",
        "Recommended next steps:",
        "",
        "1. Audit missing paired odds rows by market.",
        "2. Confirm Footiqo odds timing semantics.",
        "3. Freeze the model-feature allowlist from `super_csv_prototype_feature_dictionary.csv`.",
        "4. Only after review, build training datasets with explicit market availability filters.",
        "",
        "No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "super_csv_prototype_recommendation.md").write_text("\n".join(rec) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base, rolling_cols = load_base()
    expected_rows = len(base)
    market_frames = {}
    coverage_rows = []
    leakage_rows = []

    for market, spec in MARKETS.items():
        df = build_market_frame(base, rolling_cols, market, spec)
        out_path = OUT_DIR / spec["output"]
        df.to_csv(out_path, index=False)
        market_frames[market] = df
        validation = validate_market(df, market, spec, expected_rows)
        coverage_rows.append(validation)
        leakage_rows.append(
            {
                "market": market,
                "same_match_score_columns_excluded": not any(c in df.columns for c in ["home_goals", "away_goals", "FTHG", "FTAG", "FTR", "result_1x2"]),
                "same_match_stats_columns_excluded": not any(c in df.columns for c in ["total_corners", "total_yellow_cards", "total_shots", "total_shots_on_target"]),
                "xg_columns_excluded": not any("xg" in c.lower() for c in df.columns),
                "team_identifiers_marked_forbidden": True,
                "rolling_features_date_safe_flag_all_true": bool(df["rolling_features_date_safe_flag"].eq(True).all()),
                "external_sources_joined_flag_all_false": bool(df["external_sources_joined_flag"].eq(False).all()),
                "odds_timing_unknown_flag_all_unknown": bool(df["odds_timing_flag"].eq("unknown").all()),
                "leakage_check_pass": validation["leakage_check_pass"],
            }
        )

    coverage = pd.DataFrame(coverage_rows)
    leakage = pd.DataFrame(leakage_rows)
    feature_dict = build_feature_dictionary(market_frames, rolling_cols)

    all_written = all((OUT_DIR / spec["output"]).exists() for spec in MARKETS.values())
    strict_ready = (
        all_written
        and coverage["row_count_matches_expected_universe"].all()
        and coverage["canonical_match_id_unique"].all()
        and coverage["missing_canonical_match_id_count"].eq(0).all()
        and coverage["source_provenance_present"].all()
        and coverage["odds_timing_flag_present"].all()
        and coverage["odds_timing_flag_all_unknown"].all()
        and leakage["leakage_check_pass"].all()
        and coverage["missing_or_invalid_paired_odds_rows"].eq(0).all()
        and coverage["missing_or_invalid_probability_rows"].eq(0).all()
    )
    base_ready = (
        all_written
        and coverage["row_count_matches_expected_universe"].all()
        and coverage["canonical_match_id_unique"].all()
        and coverage["missing_canonical_match_id_count"].eq(0).all()
        and coverage["source_provenance_present"].all()
        and coverage["odds_timing_flag_present"].all()
        and coverage["odds_timing_flag_all_unknown"].all()
        and leakage["leakage_check_pass"].all()
    )
    if strict_ready:
        decision = "super_csv_prototype_ready_good"
    elif base_ready:
        decision = "super_csv_prototype_ready_needs_review"
    else:
        decision = "super_csv_prototype_failed"

    coverage.to_csv(REPORT_DIR / "super_csv_prototype_market_coverage.csv", index=False)
    feature_dict.to_csv(REPORT_DIR / "super_csv_prototype_feature_dictionary.csv", index=False)
    leakage.to_csv(REPORT_DIR / "super_csv_prototype_leakage_checks.csv", index=False)
    write_reports(coverage, feature_dict, leakage, decision)


if __name__ == "__main__":
    main()
