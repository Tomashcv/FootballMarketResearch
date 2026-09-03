from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_TO_INT = {"H": 0, "D": 1, "A": 2}
MARKET_COLUMNS = [
    "x1x2_avg_prob_home",
    "x1x2_avg_prob_draw",
    "x1x2_avg_prob_away",
    "x1x2_avg_market_overround",
    "x1x2_avg_odds_home",
    "x1x2_avg_odds_draw",
    "x1x2_avg_odds_away",
]


def result_to_1x2(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in CLASS_TO_INT:
        return text
    return {
        "HOME": "H",
        "HOME_WIN": "H",
        "DRAW": "D",
        "AWAY": "A",
        "AWAY_WIN": "A",
    }.get(text)


def load_feature_contract(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature contract not found: {path}")
    contract = pd.read_csv(path)
    if "feature_name" not in contract.columns:
        raise ValueError(f"Feature contract has no feature_name column: {path}")
    features = contract["feature_name"].dropna().astype(str).tolist()
    if len(features) != len(set(features)):
        duplicates = pd.Series(features)[pd.Series(features).duplicated()].unique().tolist()
        raise ValueError(f"Feature contract contains duplicates: {duplicates[:10]}")
    return features


def feature_list_sha256(features: list[str]) -> str:
    payload = "\n".join(features).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_v3_adapter(raw: pd.DataFrame, feature_cols: list[str], require_target: bool = True) -> pd.DataFrame:
    required = [
        "full_scope_match_id",
        "logical_match_key",
        "match_date",
        "div",
        "season_start_year",
        "home_team_raw",
        "away_team_raw",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise KeyError(f"V3 adapter missing required columns: {missing}")

    index = raw.index
    result = raw.get("result_1x2", pd.Series(index=index, dtype=object)).map(result_to_1x2)
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
            "target_outcome_1x2": result,
            "target_y": result.map(CLASS_TO_INT),
            "x1_odds_source": raw.get("x1_odds_source", pd.Series("", index=index)).fillna("").astype(str),
            "classification": raw.get("classification", pd.Series("research_only", index=index)).fillna("research_only").astype(str),
        },
        index=index,
    )
    base["season_end_year"] = base["season_start_year"] + 1

    numeric_source = raw.reindex(columns=feature_cols)
    numeric = numeric_source.apply(pd.to_numeric, errors="coerce")
    flags = pd.DataFrame(index=index)
    for col in ["clubelo_both_found_flag", "tm_both_value_found_flag", "tm_match_feature_available"]:
        values = raw[col].fillna(False).astype(bool) if col in raw.columns else pd.Series(False, index=index)
        if col in numeric.columns:
            numeric[col] = values
        else:
            flags[col] = values

    out = pd.concat([base, numeric, flags], axis=1)
    valid = out["season_start_year"].notna()
    market_prob_cols = ["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]
    market_odds_cols = ["x1x2_avg_odds_home", "x1x2_avg_odds_draw", "x1x2_avg_odds_away"]
    missing_market = [c for c in market_prob_cols + market_odds_cols if c not in out.columns]
    if missing_market:
        raise KeyError(f"Feature contract/data is missing market columns: {missing_market}")
    valid &= out[market_prob_cols].notna().all(axis=1)
    valid &= out[market_odds_cols].notna().all(axis=1)
    valid &= out[market_odds_cols].gt(1.0).all(axis=1)
    if require_target:
        valid &= out["target_y"].notna()
    out = out.loc[valid].copy()
    if require_target:
        out["target_y"] = out["target_y"].astype(int)
    return out.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def add_research_derived_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    created: list[str] = []
    pcols = ["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]
    if all(c in out.columns for c in pcols):
        probs = out[pcols].clip(1e-8, 1.0)
        out["research_market_entropy"] = -(probs * np.log(probs)).sum(axis=1)
        out["research_market_top_probability"] = probs.max(axis=1)
        out["research_market_home_away_gap"] = probs["x1x2_avg_prob_home"] - probs["x1x2_avg_prob_away"]
        out["research_market_away_logit"] = np.log(probs["x1x2_avg_prob_away"]) - np.log(1.0 - probs["x1x2_avg_prob_away"].clip(upper=1.0 - 1e-8))
        created += [
            "research_market_entropy",
            "research_market_top_probability",
            "research_market_home_away_gap",
            "research_market_away_logit",
        ]
    for source in ["clubelo_diff", "internal_elo_diff", "clubelo_diff_minus_internal_elo_diff"]:
        if source in out.columns:
            col = f"research_abs_{source}"
            out[col] = pd.to_numeric(out[source], errors="coerce").abs()
            created.append(col)
    if {"clubelo_diff", "internal_elo_diff"}.issubset(out.columns):
        out["research_clubelo_internal_disagreement"] = pd.to_numeric(out["clubelo_diff"], errors="coerce") - pd.to_numeric(out["internal_elo_diff"], errors="coerce")
        out["research_clubelo_internal_sign_disagreement"] = (
            np.sign(pd.to_numeric(out["clubelo_diff"], errors="coerce"))
            != np.sign(pd.to_numeric(out["internal_elo_diff"], errors="coerce"))
        ).astype(float)
        created += ["research_clubelo_internal_disagreement", "research_clubelo_internal_sign_disagreement"]
    for flag in ["clubelo_both_found_flag", "tm_both_value_found_flag", "tm_match_feature_available"]:
        if flag in out.columns:
            col = f"research_{flag}_numeric"
            out[col] = out[flag].fillna(False).astype(bool).astype(float)
            created.append(col)
    return out, created


def resolve_feature_group(all_features: list[str], group_name: str) -> list[str]:
    features = list(dict.fromkeys(all_features))
    if group_name == "legacy_all":
        return features
    if group_name == "external_residual_only":
        return [
            c
            for c in features
            if not (
                c.startswith("x1x2_")
                or c.startswith("research_market_")
                or re.search(r"(^|_)odds(_|$)", c.lower())
                or "prob_" in c.lower()
            )
        ]
    if group_name == "elo_market_core":
        patterns = [
            r"^x1x2_",
            r"elo",
            r"clubelo",
            r"^league_",
            r"staleness",
            r"feature_available",
            r"both_found",
            r"^research_market_",
        ]
        return [c for c in features if any(re.search(p, c, flags=re.IGNORECASE) for p in patterns)]
    if group_name == "market_context_core":
        patterns = [
            r"^x1x2_",
            r"elo",
            r"clubelo",
            r"form",
            r"days_since",
            r"matches_last",
            r"rest",
            r"congest",
            r"venue",
            r"staleness",
            r"valuation",
            r"squad",
            r"feature_available",
            r"both_found",
            r"^league_",
            r"^research_",
        ]
        return [c for c in features if any(re.search(p, c, flags=re.IGNORECASE) for p in patterns)]
    if group_name == "market_only":
        return [c for c in features if c in MARKET_COLUMNS or c.startswith("research_market_")]
    raise ValueError(f"Unknown feature group: {group_name}")


def feature_coverage(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for col in features:
        if col not in frame.columns:
            rows.append({"feature": col, "present": False, "non_null_rate": 0.0, "unique_values": 0})
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "present": True,
                "non_null_rate": float(series.notna().mean()),
                "unique_values": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)
