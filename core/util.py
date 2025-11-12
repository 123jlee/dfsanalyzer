from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

__all__ = [
    "GameInfo",
    "normalize_player_name",
    "comparable_name",
    "short_hash",
    "percentile_from_rank",
    "slugify",
]


@dataclass(frozen=True)
class GameInfo:
    away_team: Optional[str]
    home_team: Optional[str]
    game_id: Optional[str]


_GAME_RE = re.compile(r"(?P<away>[A-Z]{2,3})@(?P<home>[A-Z]{2,3})")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def normalize_player_name(name: Optional[str]) -> str:
    """Return a canonical player string with normalized whitespace."""
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", name)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"\s+", " ", normalized.strip())
    return normalized


def comparable_name(name: Optional[str]) -> str:
    """Return a lowercase punctuation-free variant for fuzzy matching."""
    normalized = normalize_player_name(name).lower()
    return _PUNCT_RE.sub("", normalized)


def short_hash(value: str, length: int = 12) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return digest[:length]


def percentile_from_rank(rank: int, total: int) -> float:
    if total <= 1:
        return 0.0
    return (max(rank - 1, 0) / (total - 1)) * 100.0


def slugify(value: str, sep: str = "-") -> str:
    normalized = comparable_name(value)
    normalized = normalized.replace(" ", sep)
    normalized = re.sub(rf"{sep}{{2,}}", sep, normalized)
    return normalized.strip(sep)


def parse_game_info(info: Optional[str]) -> GameInfo:
    if not info or isinstance(info, float):
        return GameInfo(None, None, None)
    info = info.strip().upper()
    match = _GAME_RE.search(info)
    if not match:
        return GameInfo(None, None, None)
    away = match.group("away")
    home = match.group("home")
    return GameInfo(away, home, f"{away}@{home}")


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def flatten(iterable: Iterable[Iterable[str]]) -> list[str]:
    return [item for group in iterable for item in group]
