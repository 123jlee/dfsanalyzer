from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import pandas as pd

from . import aggregate
from .util import (
    GameInfo,
    comparable_name,
    normalize_player_name,
    parse_game_info,
    short_hash,
    utc_timestamp,
)

LINEUP_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
LINEUP_SLOT_SET = set(LINEUP_SLOTS)
ENTRY_PATTERN = re.compile(r"^(?P<username>.+?)(?:\s*\((?P<used>\d+)(?:\s*/\s*(?P<max>\d+))?\))?$")


@dataclass
class IngestResult:
    output_dir: Path
    tables: Dict[str, pd.DataFrame]
    unmatched_players: List[str]
    ingest_time: str


def parse_entry_name(value: str) -> dict:
    if not isinstance(value, str):
        return {"username": "", "entries_used": 1, "entries_max": 1}
    value = value.strip()
    match = ENTRY_PATTERN.match(value)
    if not match:
        return {"username": value, "entries_used": 1, "entries_max": 1}
    username = match.group("username").strip()
    used = match.group("used")
    max_entries = match.group("max")
    used_int = int(used) if used else 1
    max_int = int(max_entries) if max_entries else max(used_int, 1)
    return {"username": username, "entries_used": used_int, "entries_max": max_int}


def parse_lineup(value: str) -> List[tuple[str, str]]:
    if not isinstance(value, str):
        return []
    tokens = value.replace("\n", " ").split()
    results: List[tuple[str, str]] = []
    current_slot: Optional[str] = None
    current_tokens: List[str] = []
    for token in tokens:
        slot = token.upper()
        if slot in LINEUP_SLOT_SET:
            if current_slot and current_tokens:
                name = normalize_player_name(" ".join(current_tokens))
                if name:
                    results.append((current_slot, name))
            current_slot = slot
            current_tokens = []
        else:
            current_tokens.append(token)
    if current_slot and current_tokens:
        name = normalize_player_name(" ".join(current_tokens))
        if name:
            results.append((current_slot, name))
    return results


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["", " ", "NA", "N/A"], keep_default_na=True)


def ingest_contest(
    standings_path: Path,
    salaries_path: Path,
    *,
    output_root: Path | None = None,
    sport: str = "nba",
    site: str = "draftkings",
    combo_config: aggregate.ComboConfig | None = None,
) -> IngestResult:
    combo_config = combo_config or aggregate.ComboConfig()
    standings = _load_csv(standings_path)
    standings = standings.loc[:, ~standings.columns.str.startswith("Unnamed")]  # drop placeholder columns
    entries_raw = standings.loc[standings["Lineup"].notna()].copy()
    entries_raw = entries_raw.drop_duplicates(subset=["EntryId"])
    field_raw = standings.loc[standings["Player"].notna()].copy()

    entries_raw["Rank"] = pd.to_numeric(entries_raw["Rank"], errors="coerce")
    entries_raw["Points"] = pd.to_numeric(entries_raw["Points"], errors="coerce")

    parsed_names = entries_raw["EntryName"].apply(parse_entry_name)
    entries_raw["username"] = parsed_names.apply(lambda d: d["username"])
    entries_raw["entries_used"] = parsed_names.apply(lambda d: d["entries_used"])
    entries_raw["entries_max"] = parsed_names.apply(lambda d: d["entries_max"])
    entries_raw["username_lc"] = entries_raw["username"].str.lower().fillna("")

    entries_raw["lineup_pairs"] = entries_raw["Lineup"].apply(parse_lineup)
    entries_raw["lineup_players"] = entries_raw["lineup_pairs"].apply(lambda pairs: [player for _, player in pairs])
    entries_raw["lineup_slots"] = entries_raw["lineup_pairs"].apply(lambda pairs: [slot for slot, _ in pairs])
    entries_raw["canonical_lineup_key"] = entries_raw["lineup_players"].apply(lambda players: "|".join(sorted(players)))
    entries_raw["canonical_hash"] = entries_raw["canonical_lineup_key"].apply(short_hash)
    entries_raw["dupe_count"] = entries_raw.groupby("canonical_lineup_key")["EntryId"].transform("count")
    entries_raw["user_total_lineups"] = entries_raw.groupby("username")["EntryId"].transform("nunique")

    entries = entries_raw.rename(
        columns={
            "EntryId": "entry_id",
            "Rank": "rank",
            "Points": "points",
            "TimeRemaining": "time_remaining",
            "Lineup": "lineup_raw",
        }
    )
    entries = entries.loc[:, [
        "entry_id",
        "rank",
        "points",
        "time_remaining",
        "EntryName",
        "username",
        "username_lc",
        "entries_used",
        "entries_max",
        "user_total_lineups",
        "lineup_raw",
        "lineup_pairs",
        "lineup_players",
        "lineup_slots",
        "canonical_lineup_key",
        "canonical_hash",
        "dupe_count",
    ]]
    entries = entries.rename(columns={"EntryName": "entry_name"})
    entries["entry_id"] = pd.to_numeric(entries["entry_id"], errors="coerce").astype("Int64")
    entries["rank"] = pd.to_numeric(entries["rank"], errors="coerce").astype("Int64")
    entries["points"] = pd.to_numeric(entries["points"], errors="coerce")

    entries = entries.sort_values("rank").reset_index(drop=True)
    entries = aggregate.enrich_with_percentiles(entries)

    salaries = _load_csv(salaries_path)
    salaries["Name"] = salaries["Name"].apply(normalize_player_name)
    salaries["name_key"] = salaries["Name"].apply(comparable_name)
    salaries["Salary"] = pd.to_numeric(salaries["Salary"], errors="coerce")
    salaries["Game Info"] = salaries["Game Info"].fillna("")

    game_infos: List[GameInfo] = salaries["Game Info"].apply(parse_game_info)
    salaries["away_team"] = [info.away_team for info in game_infos]
    salaries["home_team"] = [info.home_team for info in game_infos]
    salaries["game_id"] = [info.game_id for info in game_infos]

    salary_records = salaries.to_dict("records")
    salary_lookup: Dict[str, dict] = {}
    salary_lookup_secondary: Dict[str, dict] = {}
    for record in salary_records:
        name = record.get("Name")
        if isinstance(name, str):
            salary_lookup[name] = record
            salary_lookup_secondary[comparable_name(name)] = record

    exploded_rows: List[dict] = []
    salary_sum: List[Optional[float]] = []
    salary_avg: List[Optional[float]] = []
    salary_min: List[Optional[float]] = []
    salary_max: List[Optional[float]] = []
    salary_missing: List[int] = []
    unmatched_players: set[str] = set()

    for _, row in entries.iterrows():
        lineup_pairs: List[tuple[str, str]] = row.get("lineup_pairs", [])
        salaries_for_entry: List[float] = []
        missing_count = 0
        for slot, player in lineup_pairs:
            match = salary_lookup.get(player)
            if match is None:
                match = salary_lookup_secondary.get(comparable_name(player))
            if match is None:
                unmatched_players.add(player)
            salary_value = match.get("Salary") if match else None
            if match and isinstance(salary_value, (int, float)) and not math.isnan(salary_value):
                salaries_for_entry.append(float(salary_value))
            else:
                missing_count += 1
            exploded_rows.append(
                {
                    "entry_id": row["entry_id"],
                    "username": row["username"],
                    "rank": row["rank"],
                    "percentile": row["percentile"],
                    "points": row["points"],
                    "player": player,
                    "roster_slot": slot,
                    "salary": salary_value if match else None,
                    "dk_roster_position": match.get("Roster Position") if match else None,
                    "team": match.get("TeamAbbrev") if match else None,
                    "game_id": match.get("game_id") if match else None,
                    "away_team": match.get("away_team") if match else None,
                    "home_team": match.get("home_team") if match else None,
                }
            )
        salary_missing.append(missing_count)
        if salaries_for_entry:
            salary_sum.append(float(sum(salaries_for_entry)))
            salary_avg.append(float(mean(salaries_for_entry)))
            salary_min.append(float(min(salaries_for_entry)))
            salary_max.append(float(max(salaries_for_entry)))
        else:
            salary_sum.append(None)
            salary_avg.append(None)
            salary_min.append(None)
            salary_max.append(None)

    entries["salary_sum"] = salary_sum
    entries["salary_avg"] = salary_avg
    entries["salary_min"] = salary_min
    entries["salary_max"] = salary_max
    entries["salary_missing_count"] = salary_missing

    entries_exploded = pd.DataFrame(exploded_rows)

    field_records: List[dict] = []
    for _, row in field_raw.iterrows():
        player = normalize_player_name(row.get("Player"))
        if not player:
            continue
        match = salary_lookup.get(player)
        if match is None:
            match = salary_lookup_secondary.get(comparable_name(player))
        drafted_raw = row.get("%Drafted")
        if isinstance(drafted_raw, str):
            drafted_value = drafted_raw.replace("%", "").strip()
        else:
            drafted_value = drafted_raw
        field_pct = pd.to_numeric(drafted_value, errors="coerce")
        fpts = pd.to_numeric(row.get("FPTS"), errors="coerce")
        field_records.append(
            {
                "player": player,
                "roster_position": row.get("Roster Position"),
                "field_pct": float(field_pct) if not pd.isna(field_pct) else 0.0,
                "fpts": float(fpts) if not pd.isna(fpts) else None,
                "salary": match.get("Salary") if match else None,
                "team": match.get("TeamAbbrev") if match else None,
                "game_id": match.get("game_id") if match else None,
                "away_team": match.get("away_team") if match else None,
                "home_team": match.get("home_team") if match else None,
            }
        )
    field_players = pd.DataFrame(field_records)
    if not field_players.empty:
        field_players = (
            field_players.groupby("player", as_index=False)
            .agg(
                {
                    "roster_position": "first",
                    "field_pct": "mean",
                    "fpts": "mean",
                    "salary": "mean",
                    "team": "first",
                    "game_id": "first",
                    "away_team": "first",
                    "home_team": "first",
                }
            )
            .sort_values("player")
            .reset_index(drop=True)
        )

    user_exposure = aggregate.compute_user_exposure(entries, entries_exploded, field_players)
    combo_tables = aggregate.compute_name_combos(entries, combo_config)
    team_stacks = aggregate.compute_team_stacks(entries, entries_exploded, combo_config)
    game_stacks = aggregate.compute_game_stacks(entries, entries_exploded, combo_config)

    ingest_time = utc_timestamp()
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_root = output_root or Path("data")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    contest_meta = pd.DataFrame(
        [
            {
                "site": site,
                "sport": sport,
                "ingest_time": ingest_time,
                "n_entries": len(entries),
                "n_users": entries["username"].nunique(),
                "n_field_players": len(field_players),
                "storage_path": str(output_dir),
            }
        ]
    )

    unmatched_df = pd.DataFrame(sorted(unmatched_players), columns=["player"])

    tables: Dict[str, pd.DataFrame] = {
        "ContestMeta": contest_meta,
        "Entries": entries.drop(columns=["lineup_pairs"]),
        "EntriesExploded": entries_exploded,
        "FieldPlayers": field_players,
        "UserExposure": user_exposure,
        "TeamStacks": team_stacks,
        "GameStacks": game_stacks,
        "UnmatchedPlayers": unmatched_df,
    }
    for size, frame in combo_tables.items():
        tables[f"Combos{size}"] = frame

    for name, frame in tables.items():
        path = output_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)

    return IngestResult(
        output_dir=output_dir,
        tables=tables,
        unmatched_players=sorted(unmatched_players),
        ingest_time=ingest_time,
    )
