from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from zipfile import ZipFile
import hashlib
import math
import re
import unicodedata

import numpy as np
import pandas as pd


V3_MATRIX = Path("data/processed/features/football_feature_matrix_v3_clubelo_partial.csv")
FBREF_ZIP_2017_2024 = Path("data/raw_external/fbref_player_seasons_manual/fbref_player_seasons_2017_2024.zip")
FBREF_ZIP_2024_2025 = Path("data/raw_external/fbref_player_seasons_manual/fbref_player_seasons_2024_2025.zip")
LOCKED_ROW_PREDICTIONS = Path("outputs/reports/feature_matrix_v2_tm_1x2_locked_row_predictions.csv")
LOCKED_V3_SELECTED_BETS = Path("outputs/reports/feature_matrix_v3_clubelo_locked_selected_bets.csv")

OUT_MATRIX = Path("data/processed/features/football_feature_matrix_v4_2_fbref_prior_season_partial.csv")
REPORT_DIR = Path("outputs/reports")

BUILD_REPORT_MD = REPORT_DIR / "feature_matrix_v4_2_fbref_build_report.md"
MAPPING_CSV = REPORT_DIR / "fbref_team_mapping_candidates.csv"
ALIASES_ACCEPTED_CSV = REPORT_DIR / "fbref_aliases_accepted_v1.csv"
ALIASES_MANUAL_CSV = REPORT_DIR / "fbref_aliases_manual_review_required_v1.csv"
DATE_SAFETY_CSV = REPORT_DIR / "feature_matrix_v4_2_fbref_date_safety_audit.csv"
COVERAGE_CSV = REPORT_DIR / "feature_matrix_v4_2_fbref_coverage_by_league_season.csv"
MISSINGNESS_CSV = REPORT_DIR / "feature_matrix_v4_2_fbref_missingness.csv"
DICT_CSV = REPORT_DIR / "feature_matrix_v4_2_fbref_feature_dictionary_delta.csv"
LEAKAGE_CSV = REPORT_DIR / "feature_matrix_v4_2_fbref_leakage_checks.csv"
SCOPE_MD = REPORT_DIR / "feature_matrix_v4_2_fbref_recommended_model_scope.md"

TOP5 = {"E0", "D1", "SP1", "I1", "F1"}
LEAGUE_TO_FBREF = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}
COMP_ALIASES = {
    "eng Premier League": "Premier League",
    "de Bundesliga": "Bundesliga",
    "es La Liga": "La Liga",
    "it Serie A": "Serie A",
    "fr Ligue 1": "Ligue 1",
}
LEGAL_WORDS = {
    "afc",
    "as",
    "athletic",
    "calcio",
    "cf",
    "club",
    "de",
    "fc",
    "football",
    "futbol",
    "futebol",
    "real",
    "sc",
    "sport",
    "sporting",
    "the",
    "u",
    "ud",
}

ACCEPTED_ALIASES = [
    ("E0", "Man City", "Manchester City", "abbreviation_vs_full_name"),
    ("E0", "Man United", "Manchester Utd", "football_data_abbreviation_vs_fbref_name"),
    ("E0", "Newcastle", "Newcastle Utd", "shortened_name_vs_fbref_name"),
    ("E0", "Nott'm Forest", "Nott'ham Forest", "apostrophe_abbreviation_variant"),
    ("E0", "Tottenham", "Tottenham", "exact_common_name"),
    ("E0", "West Brom", "West Brom", "exact_common_short_name"),
    ("E0", "Wolves", "Wolves", "exact_common_nickname"),
    ("E0", "QPR", "QPR", "exact_initialism"),
    ("D1", "Bayern Munich", "Bayern Munich", "exact_common_name"),
    ("D1", "Dortmund", "Dortmund", "exact_common_short_name"),
    ("D1", "Ein Frankfurt", "Eint Frankfurt", "football_data_abbreviation_vs_fbref_abbreviation"),
    ("D1", "FC Koln", "Köln", "english_transliteration_vs_fbref_local_name"),
    ("D1", "M'gladbach", "Gladbach", "football_data_abbreviation_vs_fbref_short_name"),
    ("D1", "RB Leipzig", "RB Leipzig", "exact_abbreviation"),
    ("D1", "Leverkusen", "Leverkusen", "exact_short_name"),
    ("D1", "Mainz", "Mainz 05", "short_name_vs_numbered_name"),
    ("D1", "Stuttgart", "Stuttgart", "exact_short_name"),
    ("D1", "Hertha", "Hertha BSC", "short_name_vs_fbref_name"),
    ("D1", "Bielefeld", "Arminia", "city_short_name_vs_fbref_club_short_name"),
    ("D1", "Hamburg", "Hamburger SV", "city_short_name_vs_fbref_name"),
    ("D1", "Hannover", "Hannover 96", "short_name_vs_numbered_name"),
    ("SP1", "Ath Madrid", "Atlético Madrid", "football_data_abbreviation_vs_fbref_name"),
    ("SP1", "Ath Bilbao", "Athletic Club", "football_data_abbreviation_vs_fbref_name"),
    ("SP1", "Celta", "Celta Vigo", "short_name_vs_full_name"),
    ("SP1", "Espanol", "Espanyol", "alternate_spelling"),
    ("SP1", "La Coruna", "La Coruña", "ascii_vs_accented_name"),
    ("SP1", "Sociedad", "Real Sociedad", "short_name_vs_full_name"),
    ("SP1", "Sp Gijon", "Sporting Gijón", "football_data_abbreviation_vs_fbref_name"),
    ("SP1", "Vallecano", "Rayo Vallecano", "short_name_vs_full_name"),
    ("SP1", "Huesca", "Huesca", "exact_short_name"),
    ("I1", "Inter", "Inter", "exact_common_short_name"),
    ("I1", "Milan", "Milan", "exact_common_short_name"),
    ("I1", "Spal", "SPAL", "case_variant"),
    ("F1", "Paris SG", "Paris S-G", "football_data_abbreviation_vs_fbref_name"),
    ("F1", "St Etienne", "Saint-Étienne", "abbreviation_vs_accented_name"),
    ("F1", "Clermont", "Clermont Foot", "short_name_vs_full_name"),
]

MANUAL_REVIEW = [
    ("I1", "Pisa", "", "not present in available FBref Serie A seasons"),
    ("SP1", "Oviedo", "", "not present in available FBref La Liga seasons"),
    ("D1", "Aachen", "", "not present in available FBref Bundesliga seasons"),
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: Path) -> tuple[int, int, str]:
    return path.stat().st_size, int(path.stat().st_mtime), file_sha256(path)


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [tok for tok in text.split() if tok not in LEGAL_WORDS]
    return " ".join(tokens)


def score_name(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    token_left = " ".join(sorted(left.split()))
    token_right = " ".join(sorted(right.split()))
    return max(SequenceMatcher(None, left, right).ratio(), SequenceMatcher(None, token_left, token_right).ratio())


def season_start_from_label(value: object) -> int | None:
    text = "" if pd.isna(value) else str(value)
    match = re.search(r"(20\d{2}|19\d{2})", text)
    return int(match.group(1)) if match else None


def read_csv_from_zip(path: Path, member: str) -> pd.DataFrame:
    with ZipFile(path) as zf:
        with zf.open(member) as handle:
            return pd.read_csv(handle, low_memory=False)


def standardize_player_seasons() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    schema_rows = []
    sources = [(FBREF_ZIP_2017_2024, None), (FBREF_ZIP_2024_2025, "players_data-2024_2025.csv")]
    for zpath, only_member in sources:
        with ZipFile(zpath) as zf:
            members = [m for m in zf.namelist() if m.endswith(".csv")]
            if only_member:
                members = [only_member]
            else:
                members = [m for m in members if m.startswith("cleaned_")]
            for member in members:
                raw = read_csv_from_zip(zpath, member)
                lower = {c.casefold(): c for c in raw.columns}
                player_col = lower.get("player")
                squad_col = lower.get("squad")
                comp_col = lower.get("comp")
                pos_col = lower.get("pos")
                season_col = lower.get("season")
                season_start = season_start_from_label(raw[season_col].dropna().iloc[0]) if season_col and raw[season_col].notna().any() else season_start_from_label(member)
                comp = raw[comp_col].map(lambda x: COMP_ALIASES.get(str(x), str(x))) if comp_col else pd.Series("", index=raw.index)
                out = pd.DataFrame(
                    {
                        "source_zip": str(zpath),
                        "source_member": member,
                        "fbref_season_start": season_start,
                        "player": raw[player_col] if player_col else np.nan,
                        "squad": raw[squad_col] if squad_col else np.nan,
                        "comp": comp,
                        "pos": raw[pos_col] if pos_col else np.nan,
                    }
                )
                colmap = {
                    "matches": ["Matches Played", "MP"],
                    "avg_mins": ["Avg Mins per Match", "Mn/MP"],
                    "minutes": ["Min"],
                    "goals": ["Goals", "Gls"],
                    "assists": ["Assists", "Ast"],
                    "xg": ["Expected Goals", "xG"],
                    "xa": ["xAG", "xA"],
                    "shots": ["Total Shots", "Sh"],
                    "key_passes": ["Key passes", "KP"],
                    "prog_passes": ["Progressive Passes", "PrgP"],
                    "prog_carries": ["Progressive Carries", "PrgC"],
                    "tackles": ["Tackles attempted", "Tkl"],
                    "interceptions": ["Interceptions", "Int"],
                    "sca90": ["Shot creating actions p 90", "SCA90"],
                    "gca90": ["Goal creating actions p 90", "GCA90"],
                    "sca": ["SCA"],
                    "gca": ["GCA"],
                    "save_pct": ["Saves %", "Save%"],
                }
                for new, candidates in colmap.items():
                    found = next((c for c in candidates if c in raw.columns), None)
                    out[new] = pd.to_numeric(raw[found], errors="coerce") if found else np.nan
                out["minutes"] = out["minutes"].fillna(out["matches"] * out["avg_mins"])
                frames.append(out)
                numeric_issues = {}
                for new, candidates in colmap.items():
                    found = next((c for c in candidates if c in raw.columns), None)
                    if found:
                        numeric_issues[found] = int(pd.to_numeric(raw[found], errors="coerce").isna().sum() - raw[found].isna().sum())
                schema_rows.append(
                    {
                        "archive": str(zpath),
                        "member": member,
                        "rows": int(len(raw)),
                        "columns": int(len(raw.columns)),
                        "season_represented": season_start,
                        "league_values": "|".join(sorted(pd.Series(comp).dropna().astype(str).unique())),
                        "squad_team_column_candidates": "|".join([c for c in ["squad", "Squad"] if c in raw.columns]),
                        "player_column_candidates": "|".join([c for c in ["player", "Player"] if c in raw.columns]),
                        "position_column_candidates": "|".join([c for c in ["pos", "Pos"] if c in raw.columns]),
                        "minutes_column_candidates": "|".join([c for c in ["Min", "Matches Played", "Avg Mins per Match", "MP", "Mn/MP"] if c in raw.columns]),
                        "xg_xa_shots_passing_progression_defense_keeper_columns_available": "|".join([c for c in raw.columns if c in {"Expected Goals", "xG", "xAG", "xA", "Total Shots", "Sh", "Key passes", "KP", "Progressive Passes", "PrgP", "Progressive Carries", "PrgC", "Tackles attempted", "Tkl", "Interceptions", "Int", "Saves %", "Save%"}]),
                        "missing_values": int(raw.isna().sum().sum()),
                        "numeric_parse_issues": {k: v for k, v in numeric_issues.items() if v > 0},
                        "duplicate_player_team_season_rows": int(out.duplicated(["player", "squad", "comp", "fbref_season_start"]).sum()),
                    }
                )
    players = pd.concat(frames, ignore_index=True)
    return players, pd.DataFrame(schema_rows)


def aggregate_profiles(players: pd.DataFrame) -> pd.DataFrame:
    rows = []
    use = players[players["comp"].isin(LEAGUE_TO_FBREF.values())].copy()
    for (season, comp, squad), g in use.groupby(["fbref_season_start", "comp", "squad"], dropna=False):
        minutes = pd.to_numeric(g["minutes"], errors="coerce").fillna(0.0)
        total_min = float(minutes.sum())
        denom90 = total_min / 90.0 if total_min > 0 else np.nan
        top_minutes = minutes.sort_values(ascending=False)
        top5_min = float(top_minutes.head(5).sum()) if len(top_minutes) else 0.0
        top11_min = float(top_minutes.head(11).sum()) if len(top_minutes) else 0.0
        xg = pd.to_numeric(g["xg"], errors="coerce").fillna(0.0)
        top5_xg = float(xg.loc[minutes.sort_values(ascending=False).head(5).index].sum()) if len(g) else 0.0
        total_xg = float(xg.sum())
        row = {
            "fbref_season_start": int(season),
            "comp": comp,
            "squad": squad,
            "fbref_prev_minutes_total": total_min,
            "fbref_prev_players_used": int(g["player"].nunique()),
            "fbref_prev_top5_minutes_share": top5_min / total_min if total_min > 0 else np.nan,
            "fbref_prev_top11_minutes_share": top11_min / total_min if total_min > 0 else np.nan,
            "fbref_prev_squad_xg_concentration_top5": top5_xg / total_xg if total_xg > 0 else np.nan,
            "fbref_prev_squad_minutes_concentration_top5": top5_min / total_min if total_min > 0 else np.nan,
            "fbref_prev_non_top5_xg_share": 1.0 - top5_xg / total_xg if total_xg > 0 else np.nan,
            "fbref_prev_attack_contribution_depth": float((xg > total_xg * 0.02).sum()) if total_xg > 0 else np.nan,
        }
        per90_raw = {
            "goals": "goals",
            "assists": "assists",
            "xg": "xg",
            "xa": "xa",
            "shots": "shots",
            "key_passes": "key_passes",
            "progressive_passes": "prog_passes",
            "progressive_carries": "prog_carries",
            "tackles": "tackles",
            "interceptions": "interceptions",
            "sca": "sca",
            "gca": "gca",
        }
        for out_name, col in per90_raw.items():
            vals = pd.to_numeric(g[col], errors="coerce")
            if col in {"sca", "gca"} and vals.isna().all():
                vals = pd.to_numeric(g[f"{col}90"], errors="coerce") * minutes / 90.0
            row[f"fbref_prev_{out_name}_per90"] = float(vals.fillna(0.0).sum() / denom90) if total_min > 0 and not vals.isna().all() else np.nan
        keepers = g[g["pos"].astype(str).str.contains("GK", na=False)].copy()
        if len(keepers):
            kpct = pd.to_numeric(keepers["save_pct"], errors="coerce")
            kmin = pd.to_numeric(keepers["minutes"], errors="coerce").fillna(0.0)
            row["fbref_prev_keeper_save_pct"] = float(np.average(kpct.dropna(), weights=kmin.loc[kpct.dropna().index])) if kpct.notna().any() and kmin.loc[kpct.dropna().index].sum() > 0 else float(kpct.mean())
        else:
            row["fbref_prev_keeper_save_pct"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_mapping(v3: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    squads = profiles[["comp", "squad"]].drop_duplicates().copy()
    squads["squad_norm"] = squads["squad"].map(normalize_name)
    by_comp = {k: g.reset_index(drop=True) for k, g in squads.groupby("comp")}
    teams = pd.concat(
        [
            v3[v3["league"].isin(TOP5)][["league", "home_team"]].rename(columns={"home_team": "football_team"}),
            v3[v3["league"].isin(TOP5)][["league", "away_team"]].rename(columns={"away_team": "football_team"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    teams["comp"] = teams["league"].map(LEAGUE_TO_FBREF)
    teams["team_norm"] = teams["football_team"].map(normalize_name)
    accepted_alias = pd.DataFrame(ACCEPTED_ALIASES, columns=["league", "football_team", "fbref_squad", "reason"])
    accepted_alias["comp"] = accepted_alias["league"].map(LEAGUE_TO_FBREF)
    valid_pairs = set(zip(squads["comp"], squads["squad"]))
    accepted_alias["alias_status"] = np.where(
        [pair in valid_pairs for pair in zip(accepted_alias["comp"], accepted_alias["fbref_squad"])],
        "accepted_high_confidence_aliases_v1",
        "manual_review_required",
    )
    accepted_alias["confidence"] = np.where(accepted_alias["alias_status"].eq("accepted_high_confidence_aliases_v1"), "high", "manual_review")
    manual = pd.DataFrame(MANUAL_REVIEW, columns=["league", "football_team", "fbref_squad", "reason"])
    manual["comp"] = manual["league"].map(LEAGUE_TO_FBREF)
    manual["alias_status"] = "manual_review_required"
    manual["confidence"] = "manual_review"
    accepted_rows = []
    candidate_rows = []
    for rec in teams.sort_values(["league", "football_team"]).to_dict("records"):
        candidates = by_comp.get(rec["comp"], pd.DataFrame(columns=["comp", "squad", "squad_norm"]))
        alias = accepted_alias[
            accepted_alias["league"].eq(rec["league"])
            & accepted_alias["football_team"].eq(rec["football_team"])
            & accepted_alias["alias_status"].eq("accepted_high_confidence_aliases_v1")
        ]
        exact = candidates[candidates["squad_norm"].eq(rec["team_norm"])]
        if len(alias) == 1:
            status_name = "accepted_high_confidence_aliases_v1"
            mapped = str(alias["fbref_squad"].iloc[0])
            top = candidates[candidates["squad"].eq(mapped)].assign(score=1.0).head(1)
        elif len(exact) == 1:
            status_name = "accepted_exact_normalized"
            mapped = str(exact["squad"].iloc[0])
            top = exact.assign(score=1.0).head(1)
        else:
            scored = candidates.copy()
            scored["score"] = scored["squad_norm"].map(lambda x: score_name(rec["team_norm"], x))
            top = scored.sort_values(["score", "squad"], ascending=[False, True]).head(5)
            top_score = float(top["score"].iloc[0]) if len(top) else 0.0
            second = float(top["score"].iloc[1]) if len(top) > 1 else 0.0
            if top_score >= 0.95 and top_score - second >= 0.03:
                status_name = "accepted_high_confidence_fuzzy"
                mapped = str(top["squad"].iloc[0])
            elif top_score >= 0.80:
                status_name = "manual_review_required"
                mapped = ""
            else:
                status_name = "unmatched"
                mapped = ""
        if mapped:
            accepted_rows.append(
                {
                    "league": rec["league"],
                    "football_team": rec["football_team"],
                    "comp": rec["comp"],
                    "fbref_squad": mapped,
                    "mapping_status": status_name,
                }
            )
        for rank, cand in enumerate(top.to_dict("records") if len(top) else [], start=1):
            candidate_rows.append(
                {
                    "league": rec["league"],
                    "football_team": rec["football_team"],
                    "comp": rec["comp"],
                    "team_norm": rec["team_norm"],
                    "mapping_status": status_name,
                    "accepted_fbref_squad": mapped,
                    "candidate_rank": rank,
                    "candidate_squad": cand["squad"],
                    "candidate_score": float(cand["score"]),
                }
            )
    mapping = pd.DataFrame(candidate_rows)
    accepted = pd.DataFrame(accepted_rows).drop_duplicates(["league", "football_team"])
    accepted_alias[accepted_alias["alias_status"].eq("accepted_high_confidence_aliases_v1")][
        ["league", "football_team", "fbref_squad", "alias_status", "confidence", "reason"]
    ].to_csv(ALIASES_ACCEPTED_CSV, index=False)
    pd.concat([accepted_alias[accepted_alias["alias_status"].ne("accepted_high_confidence_aliases_v1")], manual], ignore_index=True)[
        ["league", "football_team", "fbref_squad", "alias_status", "confidence", "reason"]
    ].to_csv(ALIASES_MANUAL_CSV, index=False)
    mapping.to_csv(MAPPING_CSV, index=False)
    return mapping, accepted, accepted_alias, manual


def attach_profiles(v3: pd.DataFrame, accepted: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prof_cols = [c for c in profiles.columns if c.startswith("fbref_prev_")]
    home_map = accepted.rename(columns={"football_team": "home_team", "fbref_squad": "_fbref_home_squad", "comp": "_fbref_home_comp"})[
        ["league", "home_team", "_fbref_home_squad", "_fbref_home_comp"]
    ]
    away_map = accepted.rename(columns={"football_team": "away_team", "fbref_squad": "_fbref_away_squad", "comp": "_fbref_away_comp"})[
        ["league", "away_team", "_fbref_away_squad", "_fbref_away_comp"]
    ]
    audit = v3[["match_id", "match_date", "league", "season_start_year", "season_end_year", "home_team", "away_team"]].copy()
    audit["fbref_allowed_prior_season_start"] = pd.to_numeric(audit["season_start_year"], errors="coerce") - 1
    audit = audit.merge(home_map, on=["league", "home_team"], how="left", validate="many_to_one")
    audit = audit.merge(away_map, on=["league", "away_team"], how="left", validate="many_to_one")
    home_prof = profiles.rename(columns={"comp": "_fbref_home_comp", "squad": "_fbref_home_squad", "fbref_season_start": "fbref_allowed_prior_season_start"})
    away_prof = profiles.rename(columns={"comp": "_fbref_away_comp", "squad": "_fbref_away_squad", "fbref_season_start": "fbref_allowed_prior_season_start"})
    home_cols = ["_fbref_home_comp", "_fbref_home_squad", "fbref_allowed_prior_season_start"] + prof_cols
    away_cols = ["_fbref_away_comp", "_fbref_away_squad", "fbref_allowed_prior_season_start"] + prof_cols
    audit = audit.merge(home_prof[home_cols], on=["_fbref_home_comp", "_fbref_home_squad", "fbref_allowed_prior_season_start"], how="left", validate="many_to_one")
    audit = audit.merge(away_prof[away_cols], on=["_fbref_away_comp", "_fbref_away_squad", "fbref_allowed_prior_season_start"], how="left", suffixes=("_home", "_away"), validate="many_to_one")
    features = pd.DataFrame(index=v3.index)
    for col in prof_cols:
        h = f"{col}_home"
        a = f"{col}_away"
        features[f"home_{col}"] = audit[h]
        features[f"away_{col}"] = audit[a]
        if col not in {"fbref_prev_profile_available_flag", "fbref_prev_missing_flag"}:
            features[f"home_minus_away_{col}"] = audit[h] - audit[a]
    features["fbref_home_profile_available_flag"] = audit["fbref_prev_minutes_total_home"].notna()
    features["fbref_away_profile_available_flag"] = audit["fbref_prev_minutes_total_away"].notna()
    features["fbref_both_profile_available_flag"] = features["fbref_home_profile_available_flag"] & features["fbref_away_profile_available_flag"]
    features["fbref_home_missing_flag"] = ~features["fbref_home_profile_available_flag"]
    features["fbref_away_missing_flag"] = ~features["fbref_away_profile_available_flag"]
    features["fbref_both_missing_flag"] = ~features["fbref_both_profile_available_flag"]
    features["fbref_home_profile_season_gap"] = audit["season_start_year"] - audit["fbref_allowed_prior_season_start"]
    features["fbref_away_profile_season_gap"] = audit["season_start_year"] - audit["fbref_allowed_prior_season_start"]
    audit["home_profile_available"] = features["fbref_home_profile_available_flag"]
    audit["away_profile_available"] = features["fbref_away_profile_available_flag"]
    audit["both_profile_available"] = features["fbref_both_profile_available_flag"]
    audit["home_mapped"] = audit["_fbref_home_squad"].notna()
    audit["away_mapped"] = audit["_fbref_away_squad"].notna()
    audit["same_season_forbidden_available_home"] = audit.apply(lambda r: ((profiles["fbref_season_start"].eq(r["season_start_year"])) & (profiles["comp"].eq(r["_fbref_home_comp"])) & (profiles["squad"].eq(r["_fbref_home_squad"]))).any(), axis=1)
    audit["same_season_forbidden_available_away"] = audit.apply(lambda r: ((profiles["fbref_season_start"].eq(r["season_start_year"])) & (profiles["comp"].eq(r["_fbref_away_comp"])) & (profiles["squad"].eq(r["_fbref_away_squad"]))).any(), axis=1)
    return features, audit


def segment_summary(frame: pd.DataFrame, segment: str, group_col: str | None = None) -> list[dict[str, object]]:
    rows = []
    groups = [(segment, frame)] if group_col is None else [(str(k), g) for k, g in frame.groupby(group_col, dropna=False)]
    for group, g in groups:
        rows.append(
            {
                "segment": segment,
                "group": group,
                "rows": int(len(g)),
                "home_available": int(g["fbref_home_profile_available_flag"].sum()) if len(g) else 0,
                "away_available": int(g["fbref_away_profile_available_flag"].sum()) if len(g) else 0,
                "both_available": int(g["fbref_both_profile_available_flag"].sum()) if len(g) else 0,
                "both_available_rate": float(g["fbref_both_profile_available_flag"].mean()) if len(g) else np.nan,
                "missing_prior_coverage_rows": int((~g["fbref_both_profile_available_flag"]).sum()) if len(g) else 0,
            }
        )
    return rows


def coverage_reports(v4: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rows.extend(segment_summary(v4, "all_rows"))
    rows.extend(segment_summary(v4[v4["league"].isin(TOP5)], "top5_only"))
    rows.extend(segment_summary(v4[v4["league"].isin(TOP5) & v4["season_start_year"].between(2020, 2025)], "top5_test_2020_2025"))
    rows.extend(segment_summary(v4, "by_league", "league"))
    rows.extend(segment_summary(v4, "by_season_start_year", "season_start_year"))
    if LOCKED_ROW_PREDICTIONS.exists():
        ids = set(pd.read_csv(LOCKED_ROW_PREDICTIONS, usecols=["match_id"])["match_id"])
        rows.extend(segment_summary(v4[v4["match_id"].isin(ids)], "locked_v3_prediction_row_universe"))
    if LOCKED_V3_SELECTED_BETS.exists():
        ids = set(pd.read_csv(LOCKED_V3_SELECTED_BETS, usecols=["match_id"])["match_id"])
        rows.extend(segment_summary(v4[v4["match_id"].isin(ids)], "locked_v3_selected_bets"))
    coverage = pd.DataFrame(rows)
    coverage.to_csv(COVERAGE_CSV, index=False)
    missing = pd.DataFrame(
        [{"column": c, "missing": int(features[c].isna().sum()), "missing_rate": float(features[c].isna().mean()), "non_null": int(features[c].notna().sum())} for c in features.columns]
    )
    missing.to_csv(MISSINGNESS_CSV, index=False)
    return coverage, missing


def leakage_checks(v3: pd.DataFrame, v4: pd.DataFrame, features: pd.DataFrame, audit: pd.DataFrame, before: dict[Path, tuple[int, int, str]], after: dict[Path, tuple[int, int, str]]) -> pd.DataFrame:
    original_bad = []
    for col in v3.columns:
        left, right = v3[col], v4[col]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            ok = np.allclose(pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce"), equal_nan=True, rtol=1e-12, atol=1e-12)
        else:
            ok = left.astype("string").fillna("<NA>").equals(right.astype("string").fillna("<NA>"))
        if not ok:
            original_bad.append(col)
    added = list(features.columns)
    bad_names = [c for c in added if pd.api.types.is_object_dtype(v4[c]) and re.search(r"player|team|squad|name|league|comp", c, re.I)]
    bad_result = [c for c in added if re.search(r"target|odds|result|fixture|match_result", c, re.I)]
    same_season_used = int(audit[audit["both_profile_available"] & audit["fbref_allowed_prior_season_start"].eq(audit["season_start_year"])].shape[0])
    wrong_prior = int(audit[audit["both_profile_available"] & ~audit["fbref_allowed_prior_season_start"].eq(audit["season_start_year"] - 1)].shape[0])
    bad_2024 = int(audit[audit["both_profile_available"] & audit["fbref_allowed_prior_season_start"].eq(2024) & ~audit["season_start_year"].eq(2025)].shape[0])
    rows = [
        ("no_same_season_fbref_aggregate_used", same_season_used == 0, same_season_used, ""),
        ("allowed_fbref_season_equals_fixture_season_start_year_minus_1", wrong_prior == 0, wrong_prior, ""),
        ("fbref_2024_25_used_only_for_season_start_year_2025", bad_2024 == 0, bad_2024, ""),
        ("no_player_team_name_string_columns_added_as_model_features", len(bad_names) == 0, len(bad_names), "|".join(bad_names)),
        ("no_current_fixture_result_odds_target_columns_from_fbref", len(bad_result) == 0, len(bad_result), "|".join(bad_result)),
        ("v3_row_count_preserved", len(v3) == len(v4), len(v4), f"v3={len(v3)} v4={len(v4)}"),
        ("v3_columns_unchanged", len(original_bad) == 0, len(original_bad), "|".join(original_bad[:20])),
        ("raw_zips_unchanged", before == after, 0, f"before={before} after={after}"),
    ]
    out = pd.DataFrame([{"check": n, "status": "pass" if ok else "fail", "count": int(count), "detail": detail} for n, ok, count, detail in rows])
    out.to_csv(LEAKAGE_CSV, index=False)
    return out


def feature_dictionary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in features.columns:
        allowed = not any(tok in col.lower() for tok in ["name", "team", "squad", "player", "league", "comp"])
        rows.append(
            {
                "column": col,
                "type": str(features[col].dtype),
                "definition": "Prior completed season FBref team profile feature, home/away value, difference, or missingness flag.",
                "allowed_as_model_feature": bool(allowed),
                "leakage_risk": "low_if_prior_completed_season_only" if allowed else "excluded_identity_string",
                "notes": "For fixture season_start_year Y, source season is Y-1 only; missing remains missing/flagged.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DICT_CSV, index=False)
    return out


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return view.to_markdown(index=False)


def write_reports(decision: str, schema: pd.DataFrame, mapping: pd.DataFrame, coverage: pd.DataFrame, checks: pd.DataFrame, added_cols: int, audit: pd.DataFrame) -> None:
    key_cov = coverage[coverage["segment"].isin(["all_rows", "top5_only", "top5_test_2020_2025", "locked_v3_prediction_row_universe", "locked_v3_selected_bets"])]
    mapping_counts = mapping.drop_duplicates(["league", "football_team"]).groupby(["league", "mapping_status"]).size().rename("teams").reset_index()
    BUILD_REPORT_MD.write_text(
        "\n".join(
            [
                "# Feature Matrix V4.2 FBref Prior-Season Build Audit",
                "",
                f"Decision: `{decision}`",
                "",
                "No predictive models, value searches, threshold optimization, Understat features, or locked v3 candidate changes were run. No confirmed edge is claimed.",
                "",
                "## Schema Summary",
                md_table(schema[["member", "rows", "columns", "season_represented", "league_values", "duplicate_player_team_season_rows"]], 20),
                "",
                "## Build Summary",
                f"- Output matrix: `{OUT_MATRIX}`",
                f"- Added FBref columns: {added_cols}",
                "- Join rule: fixture `season_start_year = Y` uses only FBref player-season aggregate `Y-1`.",
                "",
                "## Mapping Summary",
                md_table(mapping_counts, 50),
                "",
                "## Key Coverage",
                md_table(key_cov[["segment", "group", "rows", "both_available", "both_available_rate", "missing_prior_coverage_rows"]], 30),
                "",
                "## Date-Safety Counts",
                f"- Rows with allowed prior-season both-team profile: {int(audit['both_profile_available'].sum())}",
                f"- Rows missing prior season because profile/mapping unavailable: {int((~audit['both_profile_available']).sum())}",
                f"- Rows where 2024/25 profile is used only for 2025/26 fixtures: {int((audit['both_profile_available'] & audit['fbref_allowed_prior_season_start'].eq(2024) & audit['season_start_year'].eq(2025)).sum())}",
                f"- Rows where same-season aggregate existed but was forbidden: {int((audit['same_season_forbidden_available_home'] | audit['same_season_forbidden_available_away']).sum())}",
                "",
                "## Leakage Checks",
                md_table(checks, 50),
                "",
            ]
        ),
        encoding="utf-8",
    )
    SCOPE_MD.write_text(
        "\n".join(
            [
                "# V4.2 FBref Recommended Model Scope",
                "",
                f"Decision: `{decision}`",
                "",
                "Recommended safe scope: top-five leagues (`E0`, `D1`, `SP1`, `I1`, `F1`) for seasons where prior completed FBref team profiles are available.",
                "",
                "Use only prior completed season aggregate features, missing/profile flags, and home-away differences. Do not use player names, squad names, team names, same-season aggregates, Understat, or current-match data.",
                "",
                "No confirmed edge is claimed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    before = {p: fingerprint(p) for p in [FBREF_ZIP_2017_2024, FBREF_ZIP_2024_2025]}
    v3 = pd.read_csv(V3_MATRIX, low_memory=False)
    v3["match_date"] = pd.to_datetime(v3["match_date"], errors="coerce")
    players, schema = standardize_player_seasons()
    profiles = aggregate_profiles(players)
    profiles["fbref_prev_profile_available_flag"] = True
    profiles["fbref_prev_profile_season_gap"] = 1
    profiles["fbref_prev_missing_flag"] = False
    mapping, accepted, _, _ = build_mapping(v3, profiles)
    features, audit = attach_profiles(v3, accepted, profiles)
    v4 = pd.concat([v3.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    audit.to_csv(DATE_SAFETY_CSV, index=False)
    coverage, missing = coverage_reports(v4, features)
    feature_dictionary(features)
    after = {p: fingerprint(p) for p in [FBREF_ZIP_2017_2024, FBREF_ZIP_2024_2025]}
    checks = leakage_checks(v3.reset_index(drop=True), v4, features, audit, before, after)
    if checks["status"].ne("pass").any():
        decision = "v4_2_fbref_build_failed"
    else:
        top = coverage[coverage["segment"].eq("top5_only")].iloc[0]
        test = coverage[coverage["segment"].eq("top5_test_2020_2025")].iloc[0]
        if float(top["both_available_rate"]) >= 0.90 and float(test["both_available_rate"]) >= 0.90:
            decision = "v4_2_fbref_feature_build_ready_good"
        elif float(test["both_available_rate"]) >= 0.70:
            decision = "v4_2_fbref_feature_build_ready_partial"
        else:
            decision = "v4_2_fbref_mapping_ready_only"
    v4.to_csv(OUT_MATRIX, index=False)
    write_reports(decision, schema, mapping, coverage, checks, len(features.columns), audit)
    key = coverage[coverage["segment"].isin(["top5_only", "top5_test_2020_2025", "locked_v3_prediction_row_universe"])]
    print(
        {
            "decision": decision,
            "v3_rows": len(v3),
            "v4_rows": len(v4),
            "added_columns": len(features.columns),
            "coverage": {r.segment: round(float(r.both_available_rate), 6) for r in key.itertuples(index=False)},
            "failed_checks": int(checks["status"].ne("pass").sum()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
