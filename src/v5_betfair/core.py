"""Streaming primitives for Betfair BASIC historical stream files.

Terminology is deliberately conservative: LTP is Last Traded Price, never an
available back/lay price, and normalized inverse LTP is only a probability proxy.
"""
from __future__ import annotations

import bz2
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping

CUTOFFS = {
    "t72h": timedelta(hours=72), "t48h": timedelta(hours=48),
    "t24h": timedelta(hours=24), "t12h": timedelta(hours=12),
    "t6h": timedelta(hours=6), "t3h": timedelta(hours=3),
    "t1h": timedelta(hours=1), "t30m": timedelta(minutes=30),
    "t15m": timedelta(minutes=15), "t5m": timedelta(minutes=5),
    "t1m": timedelta(minutes=1),
}
SIDES = ("home", "draw", "away")


def discover_raw_files(root: str | Path) -> list[Path]:
    """Recursively discover compressed streams with stable physical ordering."""
    return sorted(Path(root).resolve().rglob("*.bz2"), key=lambda p: p.as_posix())


def iter_json_lines(path: str | Path) -> Iterator[tuple[int, dict]]:
    """Yield valid JSON objects one at a time; never materialize a whole file."""
    with bz2.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON line {line_number} is not an object")
            yield line_number, value


def utc_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def normalize_team(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"\b(fc|afc|cf|the)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def raw_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_market_definition(market_id: str, definition: Mapping) -> dict:
    runners = []
    for runner in definition.get("runners") or []:
        runners.append({
            "id": int(runner["id"]), "name": runner.get("name"),
            "sort_priority": runner.get("sortPriority"),
            "status": runner.get("status"),
        })
    active = [r for r in runners if r["status"] in (None, "ACTIVE")]
    return {
        "market_id": str(market_id), "event_id": str(definition.get("eventId") or ""),
        "event_name": definition.get("eventName"), "market_type": definition.get("marketType"),
        "market_time_utc": iso(utc_datetime(definition.get("marketTime"))),
        "open_date_utc": iso(utc_datetime(definition.get("openDate"))),
        "timezone": definition.get("timezone"), "country_code": definition.get("countryCode"),
        "in_play": bool(definition.get("inPlay", False)), "status": definition.get("status"),
        "complete": bool(definition.get("complete", False)), "runners": runners,
        "runner_ids": [r["id"] for r in active],
        "runner_names": [r["name"] for r in active],
        "runner_sort_priorities": [r["sort_priority"] for r in active],
        "number_of_runners": len(active),
    }


def first_complete_definition(path: str | Path, max_messages: int = 100) -> dict:
    """Read only through the first complete definition and close immediately."""
    seen = 0
    for line_no, message in iter_json_lines(path):
        seen += 1
        for change in message.get("mc") or []:
            definition = change.get("marketDefinition")
            if definition and definition.get("complete") and definition.get("runners"):
                row = parse_market_definition(change.get("id", ""), definition)
                row.update(first_definition_line=line_no, first_definition_pt=message.get("pt"))
                return row
        if seen >= max_messages:
            break
    raise ValueError(f"no complete market definition in first {max_messages} messages")


def runner_orientation(definition: Mapping, home_name: str, away_name: str) -> dict[str, int]:
    wanted = {normalize_team(home_name): "home", normalize_team(away_name): "away"}
    result: dict[str, int] = {}
    for runner in definition.get("runners") or []:
        name = normalize_team(runner.get("name"))
        if name in {"draw", "the draw"} or "draw" == name:
            result["draw"] = int(runner["id"])
        elif name in wanted:
            result[wanted[name]] = int(runner["id"])
    if set(result) != set(SIDES):
        raise ValueError(f"unresolved runner orientation: {result}")
    return result


def apply_ltp_updates(state: dict[int, tuple[float, datetime]], changes: Iterable[Mapping], ts: datetime) -> int:
    count = 0
    for change in changes or []:
        if "ltp" not in change:
            continue
        price = float(change["ltp"])
        if price <= 1.0:
            continue
        state[int(change["id"])] = (price, ts)
        count += 1
    return count


def no_vig_ltp_proxy(prices: Mapping[str, float | None]) -> tuple[dict[str, float] | None, float | None]:
    if any(prices.get(side) is None or float(prices[side]) <= 1 for side in SIDES):
        return None, None
    raw = {side: 1.0 / float(prices[side]) for side in SIDES}
    total = sum(raw.values())
    return ({side: raw[side] / total for side in SIDES}, total)


def _snapshot(state: Mapping[int, tuple[float, datetime]], orientation: Mapping[str, int],
              cutoff: datetime, label: str) -> dict | None:
    if not all(orientation.get(side) in state for side in SIDES):
        return None
    values = {side: state[orientation[side]] for side in SIDES}
    if any(updated > cutoff for _, updated in values.values()):
        return None
    prices = {side: values[side][0] for side in SIDES}
    proxy, overround = no_vig_ltp_proxy(prices)
    row: dict = {"cutoff": label, "cutoff_utc": iso(cutoff), "proxy_overround": overround}
    for side in SIDES:
        price, updated = values[side]
        row.update({
            f"{side}_ltp": price, f"{side}_ltp_updated_utc": iso(updated),
            f"{side}_staleness_seconds": (cutoff - updated).total_seconds(),
            f"{side}_valid_price": price > 1,
            f"{side}_ltp_probability_proxy": proxy[side] if proxy else None,
        })
    return row


@dataclass
class ExtractedMarket:
    metadata: dict
    cutoffs: list[dict]
    trajectory: list[dict]
    warnings: list[str]


def extract_market(path: str | Path, mapping: Mapping, include_trajectory: bool = True) -> ExtractedMarket:
    """Statefully extract pre-start, pre-play as-of states from one stream."""
    start = utc_datetime(mapping["market_time_utc"])
    if start is None:
        raise ValueError("market start missing")
    deadlines = [(name, start - delta) for name, delta in CUTOFFS.items()]
    deadlines.sort(key=lambda x: x[1])
    pending = list(deadlines)
    state: dict[int, tuple[float, datetime]] = {}
    orientation = {side: int(mapping[f"{side}_runner_id"]) for side in SIDES}
    cutoffs: list[dict] = []
    trajectory: list[dict] = []
    warnings: list[str] = []
    first_ts = last_preplay = None
    message_count = update_count = inplay_count = 0
    in_play = False
    last_pt: datetime | None = None
    latest_complete: dict | None = None

    for line_no, message in iter_json_lines(path):
        message_count += 1
        ts = utc_datetime(message.get("pt"))
        if ts is None:
            warnings.append(f"line_{line_no}:missing_pt")
            continue
        if last_pt and ts < last_pt:
            warnings.append(f"line_{line_no}:non_monotonic_pt")
        last_pt = ts
        first_ts = first_ts or ts
        while pending and pending[0][1] < ts:
            label, deadline = pending.pop(0)
            snap = _snapshot(state, orientation, deadline, label)
            if snap:
                cutoffs.append(snap)
        for change in message.get("mc") or []:
            definition = change.get("marketDefinition")
            if definition is not None:
                in_play = bool(definition.get("inPlay", in_play))
            if in_play or ts >= start:
                if in_play:
                    inplay_count += 1
                continue
            update_count += apply_ltp_updates(state, change.get("rc") or [], ts)
        if in_play or ts >= start:
            continue
        snap = _snapshot(state, orientation, ts, "observed")
        if snap:
            last_preplay = ts
            latest_complete = snap
            if include_trajectory:
                trajectory.append(snap)

    while pending:
        label, deadline = pending.pop(0)
        snap = _snapshot(state, orientation, deadline, label)
        if snap:
            cutoffs.append(snap)
    if latest_complete:
        final = dict(latest_complete)
        final["cutoff"] = "last_preplay"
        final["cutoff_utc"] = iso(last_preplay)
        # Recompute staleness at the actual last complete observation time.
        for side in SIDES:
            updated = utc_datetime(final[f"{side}_ltp_updated_utc"])
            final[f"{side}_staleness_seconds"] = (last_preplay - updated).total_seconds()
        cutoffs.append(final)
    metadata = {
        "source_file": str(path), "market_id": str(mapping["market_id"]),
        "event_id": str(mapping.get("event_id", "")),
        "canonical_fixture_id": str(mapping["canonical_fixture_id"]),
        "league": "E0", "season": mapping.get("season"),
        "fixture_date": mapping.get("fixture_date"), "market_start_utc": iso(start),
        **{f"{side}_runner_id": orientation[side] for side in SIDES},
        "first_observed_timestamp": iso(first_ts), "last_preplay_timestamp": iso(last_preplay),
        "number_of_messages": message_count, "number_of_ltp_updates": update_count,
        "inplay_observation_count": inplay_count,
    }
    for row in cutoffs + trajectory:
        row.update(metadata)
    metadata["parse_warnings"] = ";".join(sorted(set(warnings)))
    return ExtractedMarket(metadata, cutoffs, trajectory, warnings)


def temporal_partitions(seasons: Iterable[int], test_season: int, purge_seasons: int = 0) -> dict[str, list[int]]:
    """Strict expanding temporal split; optional whole-season purge."""
    ordered = sorted(set(int(x) for x in seasons))
    train = [x for x in ordered if x < test_season - purge_seasons]
    if not train or test_season not in ordered:
        raise ValueError("insufficient temporal history")
    calibration = [train.pop()] if len(train) > 1 else []
    return {"train": train, "calibration": calibration, "test": [test_season]}
