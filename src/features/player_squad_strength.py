from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PLAYER_DATA_KEYWORDS = (
    "sofifa",
    "fifa",
    "ea_fc",
    "eafc",
    "player",
    "squad",
    "transfermarkt",
    "market_value",
    "market-value",
    "valuation",
    "lineup",
    "line-up",
    "roster",
    "transfer",
)
SKIP_PARTS = {".git", ".venv", "__pycache__", "outputs"}
SUPPORTED_SUFFIXES = {".csv", ".parquet", ".json", ".xlsx"}

DATE_COLUMNS = (
    "date",
    "snapshot_date",
    "as_of_date",
    "valuation_date",
    "market_value_date",
    "rating_date",
    "fifa_update_date",
    "sofifa_update_date",
)
CLUB_COLUMNS = ("club", "club_name", "team", "team_name", "squad", "current_club")
PLAYER_COLUMNS = ("player", "player_name", "short_name", "long_name", "name")
POSITION_COLUMNS = ("position", "positions", "player_positions", "primary_position")
AGE_COLUMNS = ("age", "player_age")
MARKET_VALUE_COLUMNS = ("market_value", "value", "market_value_eur", "eur_value", "transfermarkt_value")
OVERALL_COLUMNS = ("overall", "rating", "fifa_overall", "sofifa_overall")
POTENTIAL_COLUMNS = ("potential", "fifa_potential", "sofifa_potential")

FEATURE_COLUMNS = [
    "squad_market_value_total",
    "squad_market_value_mean",
    "squad_market_value_median",
    "squad_market_value_top11",
    "squad_market_value_top5",
    "squad_market_value_gk",
    "squad_market_value_def",
    "squad_market_value_mid",
    "squad_market_value_att",
    "squad_age_mean",
    "squad_age_top11",
    "fifa_overall_mean",
    "fifa_overall_top11",
    "fifa_overall_top5",
    "fifa_potential_mean",
    "fifa_overall_gk",
    "fifa_overall_def",
    "fifa_overall_mid",
    "fifa_overall_att",
    "rating_depth_gap",
]

TM_WINDOW_DAYS = (180, 365)
TM_WINDOW_BASE_FEATURES = (
    "tm_value_total",
    "tm_value_top11",
    "tm_value_top5",
    "tm_value_median",
    "tm_players_count",
)
TM_WINDOW_FEATURE_COLUMNS = [
    f"{feature}_{days}d" for days in TM_WINDOW_DAYS for feature in TM_WINDOW_BASE_FEATURES
]


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.casefold()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def discover_player_data_files(roots: Iterable[Path] = (Path("data"),)) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            lowered = str(path).casefold()
            if any(keyword in lowered for keyword in PLAYER_DATA_KEYWORDS):
                files.append(path)
    return sorted(files)


def _read_sample(path: Path, nrows: int | None = 5000) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix == ".xlsx":
        return pd.read_excel(path, nrows=nrows)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.DataFrame(payload).head(nrows)
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    return pd.DataFrame(value).head(nrows)
        return pd.DataFrame()
    raise ValueError(f"Unsupported file type: {path}")


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def inspect_player_file(path: Path) -> dict:
    try:
        sample = _read_sample(path)
        columns = list(sample.columns)
        date_column = first_present(columns, DATE_COLUMNS)
        club_column = first_present(columns, CLUB_COLUMNS)
        player_column = first_present(columns, PLAYER_COLUMNS)
        market_column = first_present(columns, MARKET_VALUE_COLUMNS)
        overall_column = first_present(columns, OVERALL_COLUMNS)
        potential_column = first_present(columns, POTENTIAL_COLUMNS)
        date_min = pd.NA
        date_max = pd.NA
        if date_column:
            dates = pd.to_datetime(sample[date_column], errors="coerce")
            if dates.notna().any():
                date_min = dates.min().date().isoformat()
                date_max = dates.max().date().isoformat()
        time_safe = bool(date_column and club_column and player_column and (market_column or overall_column))
        return {
            "path": str(path),
            "rows_sampled": len(sample),
            "columns": ";".join(map(str, columns)),
            "date_column": date_column or "",
            "club_column": club_column or "",
            "player_column": player_column or "",
            "market_value_column": market_column or "",
            "overall_column": overall_column or "",
            "potential_column": potential_column or "",
            "date_min": date_min,
            "date_max": date_max,
            "time_safe_candidate": time_safe,
            "time_safety_reason": (
                "has player, club, dated snapshot, and strength column"
                if time_safe
                else "missing dated player-club strength observations"
            ),
            "read_error": "",
        }
    except Exception as exc:
        return {
            "path": str(path),
            "rows_sampled": 0,
            "columns": "",
            "date_column": "",
            "club_column": "",
            "player_column": "",
            "market_value_column": "",
            "overall_column": "",
            "potential_column": "",
            "date_min": pd.NA,
            "date_max": pd.NA,
            "time_safe_candidate": False,
            "time_safety_reason": "file could not be read",
            "read_error": str(exc),
        }


def load_time_safe_observations(audit: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in audit[audit["time_safe_candidate"].fillna(False)].iterrows():
        path = Path(row["path"])
        frame = _read_sample(path, nrows=None)
        output = pd.DataFrame()
        output["source_file"] = str(path)
        output["snapshot_date"] = pd.to_datetime(frame[row["date_column"]], errors="coerce").dt.normalize()
        output["club_name"] = frame[row["club_column"]].astype(str)
        output["club_key"] = output["club_name"].map(normalize_name)
        output["player_name"] = frame[row["player_column"]].astype(str)
        output["player_key"] = output["player_name"].map(normalize_name)
        position_column = first_present(frame.columns, POSITION_COLUMNS)
        output["position_group"] = frame[position_column].map(position_group) if position_column else "UNK"
        age_column = first_present(frame.columns, AGE_COLUMNS)
        output["age"] = pd.to_numeric(frame[age_column], errors="coerce") if age_column else np.nan
        market_column = row.get("market_value_column")
        overall_column = row.get("overall_column")
        potential_column = row.get("potential_column")
        output["market_value"] = parse_market_values(frame[market_column]) if isinstance(market_column, str) and market_column else np.nan
        output["overall"] = pd.to_numeric(frame[overall_column], errors="coerce") if isinstance(overall_column, str) and overall_column else np.nan
        output["potential"] = pd.to_numeric(frame[potential_column], errors="coerce") if isinstance(potential_column, str) and potential_column else np.nan
        output = output.dropna(subset=["snapshot_date", "club_key", "player_key"])
        frames.append(output)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_transfermarkt_market_values(path: Path = Path("data/external/players/transfermarkt_market_values.csv")) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"valuation_date", "player_id", "club_name", "market_value_eur"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Transfermarkt market values missing columns: {missing}")
    output = frame.copy()
    output["valuation_date"] = pd.to_datetime(output["valuation_date"], errors="coerce").dt.normalize()
    output["market_value_eur"] = pd.to_numeric(output["market_value_eur"], errors="coerce")
    output["club_key"] = output["club_name"].map(normalize_name)
    output = output.dropna(subset=["valuation_date", "player_id", "club_key", "market_value_eur"])
    return output.sort_values(["club_key", "player_id", "valuation_date"]).reset_index(drop=True)


def load_player_squad_team_mapping(path: Path = Path("data/manual/player_squad_team_name_mapping.csv")) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"league", "match_team", "player_data_club_name", "confidence"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Player squad team mapping missing columns: {missing}")
    output = frame.copy()
    output["normalized_match_team"] = output.get("normalized_match_team", output["match_team"].map(normalize_name))
    output["normalized_match_team"] = output["normalized_match_team"].fillna("").map(normalize_name)
    output["player_data_club_name"] = output["player_data_club_name"].fillna("").astype(str)
    output["normalized_player_data_club"] = output.get(
        "normalized_player_data_club",
        output["player_data_club_name"].map(normalize_name),
    )
    output["normalized_player_data_club"] = output["normalized_player_data_club"].fillna("").map(normalize_name)
    for column in ["valid_from", "valid_to"]:
        if column in output.columns:
            output[column] = pd.to_datetime(output[column], errors="coerce").dt.normalize()
        else:
            output[column] = pd.NaT
    return output


def mapped_transfermarkt_club(
    mapping: pd.DataFrame,
    league: object,
    match_team: object,
    match_date: pd.Timestamp | None = None,
) -> str:
    if mapping.empty:
        return ""
    candidates = mapping[
        mapping["league"].astype(str).eq(str(league))
        & mapping["normalized_match_team"].eq(normalize_name(match_team))
        & mapping["player_data_club_name"].astype(str).ne("")
        & ~mapping["confidence"].astype(str).eq("unmatched")
    ].copy()
    if candidates.empty:
        return ""
    if match_date is not None and pd.notna(match_date):
        date = pd.Timestamp(match_date).normalize()
        candidates = candidates[
            (candidates["valid_from"].isna() | (candidates["valid_from"] <= date))
            & (candidates["valid_to"].isna() | (candidates["valid_to"] > date))
        ]
        if candidates.empty:
            return ""
    candidates["_has_window"] = candidates["valid_from"].notna() | candidates["valid_to"].notna()
    candidates = candidates.sort_values(["_has_window", "valid_from"], ascending=[False, False])
    return str(candidates.iloc[0]["player_data_club_name"])


def parse_market_values(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")

    def parse_one(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip().replace(",", "")
        text = text.replace("€", "").replace("£", "").replace("$", "")
        multiplier = 1.0
        lowered = text.casefold()
        if lowered.endswith("m"):
            multiplier = 1_000_000.0
            text = text[:-1]
        elif lowered.endswith("k"):
            multiplier = 1_000.0
            text = text[:-1]
        number = pd.to_numeric(text, errors="coerce")
        return float(number) * multiplier if pd.notna(number) else np.nan

    return values.map(parse_one)


def transfermarkt_window_features(
    market_values: pd.DataFrame,
    club_name: object,
    match_date: pd.Timestamp,
    windows: Iterable[int] = TM_WINDOW_DAYS,
) -> dict:
    row = {column: np.nan for column in [f"{feature}_{days}d" for days in windows for feature in TM_WINDOW_BASE_FEATURES]}
    if market_values.empty or not club_name or pd.isna(match_date):
        return row

    date = pd.Timestamp(match_date).normalize()
    club_key = normalize_name(club_name)
    club_values = market_values[
        market_values["club_key"].eq(club_key)
        & (market_values["valuation_date"] < date)
    ]
    if club_values.empty:
        return row

    for days in windows:
        start = date - pd.Timedelta(days=int(days))
        window = club_values[club_values["valuation_date"] >= start].copy()
        if window.empty:
            continue
        latest = (
            window.sort_values(["player_id", "valuation_date"])
            .drop_duplicates("player_id", keep="last")
        )
        values = pd.to_numeric(latest["market_value_eur"], errors="coerce").dropna().sort_values(ascending=False)
        if values.empty:
            continue
        row[f"tm_value_total_{days}d"] = float(values.sum())
        row[f"tm_value_top11_{days}d"] = float(values.head(11).sum())
        row[f"tm_value_top5_{days}d"] = float(values.head(5).sum())
        row[f"tm_value_median_{days}d"] = float(values.median())
        row[f"tm_players_count_{days}d"] = int(values.size)
    return row


def add_transfermarkt_window_features(
    frame: pd.DataFrame,
    market_values: pd.DataFrame,
    mapping: pd.DataFrame,
    windows: Iterable[int] = TM_WINDOW_DAYS,
    league_column: str = "league",
) -> pd.DataFrame:
    output = frame.copy()
    feature_columns = [f"{feature}_{days}d" for days in windows for feature in TM_WINDOW_BASE_FEATURES]
    if output.empty:
        return output

    if league_column not in output.columns and "Div" in output.columns:
        output[league_column] = output["Div"]

    home_rows = []
    away_rows = []
    home_clubs = []
    away_clubs = []
    for _, row in output.iterrows():
        match_date = pd.Timestamp(row["Date"]) if pd.notna(row.get("Date")) else pd.NaT
        league = row.get(league_column, row.get("Div", ""))
        home_club = mapped_transfermarkt_club(mapping, league, row.get("HomeTeam", ""), match_date)
        away_club = mapped_transfermarkt_club(mapping, league, row.get("AwayTeam", ""), match_date)
        home_clubs.append(home_club)
        away_clubs.append(away_club)
        home_rows.append(transfermarkt_window_features(market_values, home_club, match_date, windows))
        away_rows.append(transfermarkt_window_features(market_values, away_club, match_date, windows))

    output["home_tm_mapped_club_name"] = home_clubs
    output["away_tm_mapped_club_name"] = away_clubs
    home = pd.DataFrame(home_rows).add_prefix("home_")
    away = pd.DataFrame(away_rows).add_prefix("away_")
    output = pd.concat([output.reset_index(drop=True), home, away], axis=1)
    for column in feature_columns:
        home_column = f"home_{column}"
        away_column = f"away_{column}"
        output[f"home_minus_away_{column}"] = pd.to_numeric(output[home_column], errors="coerce") - pd.to_numeric(
            output[away_column], errors="coerce"
        )
    return output


def position_group(value: object) -> str:
    text = normalize_name(value)
    if not text:
        return "UNK"
    if "gk" in text or "keeper" in text:
        return "GK"
    if any(token in text.split() for token in ["cb", "lb", "rb", "lwb", "rwb", "defender", "def"]):
        return "DEF"
    if any(token in text.split() for token in ["cm", "cdm", "cam", "lm", "rm", "midfielder", "mid"]):
        return "MID"
    if any(token in text.split() for token in ["st", "cf", "lw", "rw", "forward", "attacker", "att"]):
        return "ATT"
    return "UNK"


def squad_aggregate(players: pd.DataFrame) -> dict:
    row = {column: np.nan for column in FEATURE_COLUMNS}
    if players.empty:
        return row
    market = pd.to_numeric(players["market_value"], errors="coerce").dropna().sort_values(ascending=False)
    overall = pd.to_numeric(players["overall"], errors="coerce").dropna().sort_values(ascending=False)
    potential = pd.to_numeric(players["potential"], errors="coerce").dropna()
    age = pd.to_numeric(players["age"], errors="coerce").dropna()
    if len(market):
        row["squad_market_value_total"] = float(market.sum())
        row["squad_market_value_mean"] = float(market.mean())
        row["squad_market_value_median"] = float(market.median())
        row["squad_market_value_top11"] = float(market.head(11).sum())
        row["squad_market_value_top5"] = float(market.head(5).sum())
    if len(age):
        row["squad_age_mean"] = float(age.mean())
        if len(market):
            top11_index = market.head(11).index
            row["squad_age_top11"] = float(pd.to_numeric(players.loc[top11_index, "age"], errors="coerce").mean())
    if len(overall):
        row["fifa_overall_mean"] = float(overall.mean())
        row["fifa_overall_top11"] = float(overall.head(11).mean())
        row["fifa_overall_top5"] = float(overall.head(5).mean())
        if len(overall) > 11:
            row["rating_depth_gap"] = float(overall.head(11).mean() - overall.iloc[11:].mean())
    if len(potential):
        row["fifa_potential_mean"] = float(potential.mean())
    for group, suffix in [("GK", "gk"), ("DEF", "def"), ("MID", "mid"), ("ATT", "att")]:
        subset = players[players["position_group"].eq(group)]
        row[f"squad_market_value_{suffix}"] = float(pd.to_numeric(subset["market_value"], errors="coerce").sum()) if len(subset) else np.nan
        row[f"fifa_overall_{suffix}"] = float(pd.to_numeric(subset["overall"], errors="coerce").mean()) if len(subset) else np.nan
    return row


def latest_squad_features(observations: pd.DataFrame, club_key: str, match_date: pd.Timestamp) -> dict:
    eligible = observations[(observations["club_key"].eq(club_key)) & (observations["snapshot_date"] < match_date)].copy()
    if eligible.empty:
        return {column: np.nan for column in FEATURE_COLUMNS}
    latest_date = eligible["snapshot_date"].max()
    return squad_aggregate(eligible[eligible["snapshot_date"].eq(latest_date)].copy())
