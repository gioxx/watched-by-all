from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Any
import time

@dataclass
class Cache:
    """
    In-memory cache for computed results.
    """
    last_refresh_ts: float = 0.0
    users: Dict[str, str] = field(default_factory=dict)  # user_id -> friendly_name

    # If empty, all users are considered. Otherwise only these user_ids.
    selected_user_ids: Set[str] = field(default_factory=set)

    # Completion map: (user_id, rating_key) -> True
    completed: Set[Tuple[str, str]] = field(default_factory=set)

    # Movie keys seen in history
    movies: Set[str] = field(default_factory=set)

    # Show keys seen in history (grandparent/show rating key)
    shows: Set[str] = field(default_factory=set)
    # Season keys seen in history
    seasons: Set[str] = field(default_factory=set)

    # Episode list per show
    show_episodes: Dict[str, List[str]] = field(default_factory=dict)
    # Season runtime minutes aggregate
    season_runtime_minutes: Dict[str, int] = field(default_factory=dict)
    # Last viewed timestamp per season
    season_last_view: Dict[str, float] = field(default_factory=dict)
    # Track episodes already accounted for per season to avoid double counting
    season_episode_seen: Dict[str, Set[str]] = field(default_factory=dict)

    # Per-user progress
    # Movies: user_id -> rating_key -> {"percent": float 0-1, "completed": bool}
    user_movie_progress: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    # Shows: user_id -> show_rating_key -> set(episode_rating_keys completed)
    user_show_episodes: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)
    # Seasons: user_id -> season_id -> set(episode_ids completed)
    user_season_episodes: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)
    # Watched history per user (Jellyfin events)
    user_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Jellyfin users cache (id -> name)
    jellyfin_users: Dict[str, str] = field(default_factory=dict)
    # Jellyfin metadata cache (item_id -> minimal metadata)
    jellyfin_meta: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Precomputed results
    movies_by_all: List[str] = field(default_factory=list)   # rating keys
    shows_by_all: List[str] = field(default_factory=list)    # season ids

    def is_stale(self, refresh_minutes: int) -> bool:
        if self.last_refresh_ts <= 0:
            return True
        return (time.time() - self.last_refresh_ts) > (refresh_minutes * 60)
