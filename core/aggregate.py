from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from statistics import median
from typing import Dict, List, Sequence

import pandas as pd

from .util import percentile_from_rank


@dataclass
class ComboConfig:
    min_size: int = 2
    max_size: int = 4
    team_stack_max: int = 4
    game_stack_max: int = 7
    top_n_cap: int = 5000


def compute_user_exposure(entries: pd.DataFrame, exploded: pd.DataFrame, field_players: pd.DataFrame) -> pd.DataFrame:
    user_lineups = entries.groupby("username")[["entry_id"]].nunique().rename(columns={"entry_id": "user_total_lineups"})
    exposure = (
        exploded.groupby(["username", "player"])  # type: ignore[arg-type]
        .agg(
            entry_count=("entry_id", "nunique"),
            best_rank=("rank", "min"),
            max_points=("points", "max"),
            best_percentile=("percentile", "min"),
        )
        .reset_index()
    )
    exposure = exposure.merge(user_lineups, on="username", how="left")
    exposure["user_exposure_pct"] = (
        exposure["entry_count"] / exposure["user_total_lineups"].replace({0: pd.NA}) * 100
    ).astype(float)
    field_pct = field_players.set_index("player")["field_pct"] if not field_players.empty else pd.Series(dtype=float)
    exposure["field_pct"] = exposure["player"].map(field_pct).fillna(0.0)
    exposure["delta_vs_field"] = exposure["user_exposure_pct"].fillna(0.0) - exposure["field_pct"]
    return exposure.sort_values(["username", "player"]).reset_index(drop=True)


def _combo_records(entries: pd.DataFrame, size: int) -> List[dict]:
    records: Dict[tuple[str, ...], Dict[str, List[float]]] = defaultdict(lambda: {
        "entry_ids": [],
        "ranks": [],
        "percentiles": [],
        "points": [],
    })
    for row in entries.itertuples(index=False):
        players: Sequence[str] = getattr(row, "lineup_players", [])
        if not players or len(players) < size:
            continue
        unique_players = sorted(dict.fromkeys(players))
        for combo in combinations(unique_players, size):
            bucket = records[combo]
            bucket["entry_ids"].append(getattr(row, "entry_id"))
            bucket["ranks"].append(getattr(row, "rank"))
            bucket["percentiles"].append(getattr(row, "percentile"))
            bucket["points"].append(getattr(row, "points"))
    return _combo_dict_to_records(records, size=size)


def _combo_dict_to_records(records: Dict[tuple[str, ...], Dict[str, List[float]]], *, size: int, extra: dict | None = None) -> List[dict]:
    output: List[dict] = []
    for combo, payload in records.items():
        ranks = payload["ranks"]
        percentiles = payload["percentiles"]
        points = payload["points"]
        entry_ids = payload["entry_ids"]
        if not entry_ids:
            continue
        extra_payload = extra or {}
        output.append(
            {
                "combo": " | ".join(combo),
                "players": list(combo),
                "size": size,
                "frequency": len(entry_ids),
                "best_rank": min(ranks),
                "best_percentile": min(percentiles),
                "median_rank": float(median(ranks)),
                "max_points": max(points),
                "entry_ids": entry_ids,
                "count_in_current_percentile": len(entry_ids),
                **extra_payload,
            }
        )
    return output


def compute_name_combos(entries: pd.DataFrame, config: ComboConfig) -> Dict[int, pd.DataFrame]:
    results: Dict[int, pd.DataFrame] = {}
    for size in range(config.min_size, config.max_size + 1):
        records = _combo_records(entries, size=size)
        frame = pd.DataFrame.from_records(records)
        if not frame.empty:
            frame = frame.sort_values(["size", "frequency", "best_rank"], ascending=[True, False, True])
        results[size] = frame.reset_index(drop=True)
    return results


def compute_team_stacks(entries: pd.DataFrame, exploded: pd.DataFrame, config: ComboConfig) -> pd.DataFrame:
    lookup = {entry_id: group for entry_id, group in exploded.groupby("entry_id")}
    records: Dict[tuple[str, tuple[str, ...]], Dict[str, List[float]]] = defaultdict(lambda: {
        "entry_ids": [],
        "ranks": [],
        "percentiles": [],
        "points": [],
    })
    for row in entries.itertuples(index=False):
        entry_id = getattr(row, "entry_id")
        group = lookup.get(entry_id)
        if group is None:
            continue
        by_team = group.dropna(subset=["team"]).groupby("team")
        for team, team_group in by_team:
            players = sorted(team_group["player"].unique())
            limit = min(len(players), config.team_stack_max)
            for size in range(config.min_size, limit + 1):
                for combo in combinations(players, size):
                    bucket = records[(team, combo)]
                    bucket["entry_ids"].append(entry_id)
                    bucket["ranks"].append(getattr(row, "rank"))
                    bucket["percentiles"].append(getattr(row, "percentile"))
                    bucket["points"].append(getattr(row, "points"))
    rows = []
    for (team, combo), payload in records.items():
        ranks = payload["ranks"]
        percentiles = payload["percentiles"]
        points = payload["points"]
        entry_ids = payload["entry_ids"]
        if not entry_ids:
            continue
        rows.append(
            {
                "team": team,
                "combo": " | ".join(combo),
                "players": list(combo),
                "size": len(combo),
                "frequency": len(entry_ids),
                "best_rank": min(ranks),
                "best_percentile": min(percentiles),
                "median_rank": float(median(ranks)),
                "max_points": max(points),
                "entry_ids": entry_ids,
                "count_in_current_percentile": len(entry_ids),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["team", "frequency", "best_rank"], ascending=[True, False, True]).reset_index(drop=True)


def compute_game_stacks(entries: pd.DataFrame, exploded: pd.DataFrame, config: ComboConfig) -> pd.DataFrame:
    lookup = {entry_id: group for entry_id, group in exploded.groupby("entry_id")}
    records: Dict[tuple[str, tuple[str, ...]], Dict[str, List[float]]] = defaultdict(lambda: {
        "entry_ids": [],
        "ranks": [],
        "percentiles": [],
        "points": [],
    })
    for row in entries.itertuples(index=False):
        entry_id = getattr(row, "entry_id")
        group = lookup.get(entry_id)
        if group is None:
            continue
        by_game = group.dropna(subset=["game_id"]).groupby("game_id")
        for game_id, game_group in by_game:
            players = sorted(game_group["player"].unique())
            limit = min(len(players), config.game_stack_max)
            for size in range(config.min_size, limit + 1):
                for combo in combinations(players, size):
                    bucket = records[(game_id, combo)]
                    bucket["entry_ids"].append(entry_id)
                    bucket["ranks"].append(getattr(row, "rank"))
                    bucket["percentiles"].append(getattr(row, "percentile"))
                    bucket["points"].append(getattr(row, "points"))
    rows = []
    for (game_id, combo), payload in records.items():
        ranks = payload["ranks"]
        percentiles = payload["percentiles"]
        points = payload["points"]
        entry_ids = payload["entry_ids"]
        if not entry_ids:
            continue
        rows.append(
            {
                "game_id": game_id,
                "combo": " | ".join(combo),
                "players": list(combo),
                "size": len(combo),
                "frequency": len(entry_ids),
                "best_rank": min(ranks),
                "best_percentile": min(percentiles),
                "median_rank": float(median(ranks)),
                "max_points": max(points),
                "entry_ids": entry_ids,
                "count_in_current_percentile": len(entry_ids),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["game_id", "frequency", "best_rank"], ascending=[True, False, True]).reset_index(drop=True)


def apply_percentile_filter(entries: pd.DataFrame, percentile: float | None, rank: int | None) -> pd.DataFrame:
    frame = entries
    if percentile is not None:
        frame = frame.loc[frame["percentile"] <= percentile]
    if rank is not None:
        frame = frame.loc[frame["rank"] <= rank]
    return frame


def enrich_with_percentiles(entries: pd.DataFrame) -> pd.DataFrame:
    total = len(entries)
    entries = entries.copy()
    entries["percentile"] = [percentile_from_rank(r, total) for r in entries["rank"]]
    return entries
