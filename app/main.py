from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List

from httpx import HTTPStatusError

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.tautulli_client import TautulliClient
from app.plex_client import PlexClient
from app.store import Cache


TAUTULLI_URL = os.environ.get("TAUTULLI_URL", "").strip()
TAUTULLI_APIKEY = os.environ.get("TAUTULLI_APIKEY", "").strip()
PLEX_URL = os.environ.get("PLEX_URL", "").strip()
PLEX_TOKEN = os.environ.get("PLEX_TOKEN", "").strip()

WATCH_THRESHOLD = float(os.environ.get("WATCH_THRESHOLD", "0.95"))
REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", "30"))
HISTORY_MAX_ROWS = int(os.environ.get("HISTORY_MAX_ROWS", "200000"))

if not (TAUTULLI_URL and TAUTULLI_APIKEY and PLEX_URL and PLEX_TOKEN):
    # Fail fast with a clear error in logs.
    raise RuntimeError("Missing required env vars: TAUTULLI_URL, TAUTULLI_APIKEY, PLEX_URL, PLEX_TOKEN")

tautulli = TautulliClient(TAUTULLI_URL, TAUTULLI_APIKEY)
plex = PlexClient(PLEX_URL, PLEX_TOKEN)

cache = Cache()
app = FastAPI(title="Plex Watched-By-All Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")


def _is_completed_item(h: Dict[str, Any]) -> bool:
    """
    Determine whether a history entry counts as "completed".

    Priority:
      1) watched_status == 1 (best, aligns with Plex played state when present)
      2) percent_complete >= WATCH_THRESHOLD (fallback)
    """
    ws = h.get("watched_status")
    if ws is not None:
        try:
            return int(ws) == 1
        except Exception:
            pass

    pc = h.get("percent_complete")
    if pc is None:
        return False
    try:
        # percent_complete is usually 0..100
        pc_float = float(pc) / 100.0 if float(pc) > 1.0 else float(pc)
        return pc_float >= WATCH_THRESHOLD
    except Exception:
        return False


def _effective_user_ids() -> List[str]:
    """Return the user_ids that should be considered for computations.

    If `cache.selected_user_ids` is empty, all users are considered.
    Otherwise, only the selected IDs that still exist in the current user list.
    """
    if cache.selected_user_ids:
        return [uid for uid in cache.users.keys() if uid in cache.selected_user_ids]
    return list(cache.users.keys())


async def _plex_show_episode_keys(show_rating_key: str) -> List[str]:
    """Return all episode rating keys for a given show.

    Primary strategy: use Plex /allLeaves (fast + avoids season enumeration).
    Fallback: enumerate show -> seasons -> episodes via /children.

    IMPORTANT: Never crash the app if a ratingKey can't be resolved (404).
    """
    if show_rating_key in cache.show_episodes:
        return cache.show_episodes[show_rating_key]

    episode_keys: List[str] = []

    # 1) Try /allLeaves first.
    try:
        leaves_json = await plex.get_all_leaves(show_rating_key)
        leaves = leaves_json.get("MediaContainer", {}).get("Metadata", []) or []
        for e in leaves:
            ek = str(e.get("ratingKey", "")).strip()
            if ek:
                episode_keys.append(ek)
    except HTTPStatusError as e:
        if e.response is not None and e.response.status_code != 404:
            raise

    # 2) Fallback to /children traversal.
    if not episode_keys:
        try:
            seasons_json = await plex.get_children(show_rating_key)
            seasons = seasons_json.get("MediaContainer", {}).get("Metadata", []) or []

            for s in seasons:
                season_key = str(s.get("ratingKey", "")).strip()
                if not season_key:
                    continue
                eps_json = await plex.get_children(season_key)
                eps = eps_json.get("MediaContainer", {}).get("Metadata", []) or []
                for e in eps:
                    ek = str(e.get("ratingKey", "")).strip()
                    if ek:
                        episode_keys.append(ek)
        except HTTPStatusError as e:
            if e.response is not None and e.response.status_code != 404:
                raise

    cache.show_episodes[show_rating_key] = episode_keys
    return episode_keys


async def refresh_cache(force: bool = False) -> None:
    """
    Pull users + history from Tautulli, compute:
      - movies watched by all users
      - shows where all episodes watched by all users
    """
    if not force and not cache.is_stale(REFRESH_MINUTES):
        return

    # Reset
    cache.users.clear()
    cache.completed.clear()
    cache.movies.clear()
    cache.shows.clear()
    cache.movies_by_all.clear()
    cache.shows_by_all.clear()

    users_data = await tautulli.get_users()
    users = users_data.get("users", []) if isinstance(users_data, dict) else (users_data or [])
    for u in users:
        uid = str(u.get("user_id", "")).strip()
        name = (u.get("friendly_name") or u.get("username") or uid).strip()
        if uid:
            cache.users[uid] = name

    # Pull history with paging
    start = 0
    page_size = 1000
    pulled = 0

    while True:
        if pulled >= HISTORY_MAX_ROWS:
            break
        data = await tautulli.get_history(start=start, length=page_size)
        rows = data.get("data", []) if isinstance(data, dict) else (data or [])
        if not rows:
            break

        for h in rows:
            uid = str(h.get("user_id", "")).strip()
            media_type = (h.get("media_type") or "").strip().lower()

            if not uid or uid not in cache.users:
                continue

            if media_type == "movie":
                rk = str(h.get("rating_key", "")).strip()
                if rk:
                    cache.movies.add(rk)
                    if _is_completed_item(h):
                        cache.completed.add((uid, rk))

            elif media_type == "episode":
                ep_rk = str(h.get("rating_key", "")).strip()
                show_rk = str(h.get("grandparent_rating_key", "")).strip()
                if show_rk:
                    cache.shows.add(show_rk)
                if ep_rk and _is_completed_item(h):
                    cache.completed.add((uid, ep_rk))

        pulled += len(rows)
        start += page_size

        # If Tautulli returns total_records, we can stop early.
        total = None
        if isinstance(data, dict):
            total = data.get("recordsTotal") or data.get("total_records") or data.get("filteredCount")
        if total is not None:
            try:
                if start >= int(total):
                    break
            except Exception:
                pass

    user_ids = _effective_user_ids()

    # If no users are effectively selected, results should be empty but the app must not crash.
    if not user_ids:
        cache.movies_by_all = []
        cache.shows_by_all = []
        cache.last_refresh_ts = time.time()
        return

    # Movies watched by all
    movies_by_all: List[str] = []
    for mk in cache.movies:
        if all((uid, mk) in cache.completed for uid in user_ids):
            movies_by_all.append(mk)
    cache.movies_by_all = sorted(movies_by_all, key=lambda x: int(x) if x.isdigit() else x)

    # Shows watched by all (ALL episodes watched by ALL users)
    shows_by_all: List[str] = []
    for sk in cache.shows:
        eps = await _plex_show_episode_keys(sk)
        if not eps:
            continue
        ok = True
        for uid in user_ids:
            for epk in eps:
                if (uid, epk) not in cache.completed:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            shows_by_all.append(sk)
    cache.shows_by_all = sorted(shows_by_all, key=lambda x: int(x) if x.isdigit() else x)

    cache.last_refresh_ts = time.time()


async def _plex_title_thumb(rating_key: str) -> Dict[str, str] | None:
    """
    Fetch minimal metadata from Plex for UI.

    If Plex returns 404 (item missing/removed), return None so callers can skip it.
    """
    try:
        j = await plex.get_metadata(rating_key)
    except HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise

    md = (j.get("MediaContainer", {}).get("Metadata") or [{}])[0]
    title = md.get("title") or ""
    year = md.get("year")
    thumb = md.get("thumb") or ""
    typ = md.get("type") or ""

    thumb_url = ""
    if thumb:
        thumb_url = f"{PLEX_URL}{thumb}?X-Plex-Token={PLEX_TOKEN}"

    return {
        "ratingKey": str(rating_key),
        "title": str(title),
        "year": str(year) if year is not None else "",
        "type": str(typ),
        "thumb": thumb_url,
    }


@app.on_event("startup")
async def _startup() -> None:
    # Refresh on boot, then keep a background refresh loop.
    await refresh_cache(force=True)

    async def loop():
        while True:
            try:
                await refresh_cache(force=False)
            except Exception:
                # Keep the service running even if a refresh fails once.
                pass
            await asyncio.sleep(REFRESH_MINUTES * 60)

    asyncio.create_task(loop())


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/summary")
async def summary():
    await refresh_cache(force=False)
    return JSONResponse(
        {
            "users": len(_effective_user_ids()),
            "moviesByAll": len(cache.movies_by_all),
            "showsByAll": len(cache.shows_by_all),
            "lastRefresh": cache.last_refresh_ts,
        }
    )


@app.get("/api/users")
async def api_users():
    """Return all known users plus the current selection."""
    await refresh_cache(force=False)
    return JSONResponse(
        {
            "users": [{"user_id": uid, "name": name} for uid, name in cache.users.items()],
            "selected_user_ids": sorted(list(cache.selected_user_ids)),
        }
    )


@app.post("/api/selected-users")
async def api_set_selected_users(payload: Dict[str, Any] = Body(...)):
    """Set which users should be considered. Empty selection means 'all users'."""
    await refresh_cache(force=False)

    raw_ids = payload.get("selected_user_ids", [])
    if not isinstance(raw_ids, list):
        return JSONResponse({"error": "selected_user_ids must be a list"}, status_code=400)

    new_ids = {str(x).strip() for x in raw_ids if str(x).strip()}

    # Keep only IDs that exist in the current user list.
    known_ids = set(cache.users.keys())
    cache.selected_user_ids = {uid for uid in new_ids if uid in known_ids}

    # Force recompute with the new selection.
    await refresh_cache(force=True)

    return JSONResponse({"ok": True, "selected_user_ids": sorted(list(cache.selected_user_ids))})


@app.post("/api/refresh")
async def api_force_refresh():
    """Force a cache refresh (useful after library changes)."""
    await refresh_cache(force=True)
    return JSONResponse({"ok": True, "lastRefresh": cache.last_refresh_ts})


@app.get("/api/movies")
async def movies():
    await refresh_cache(force=False)
    items: List[Dict[str, str]] = []
    for rk in cache.movies_by_all:
        it = await _plex_title_thumb(rk)
        if it is not None:
            items.append(it)
    return JSONResponse({"items": items})


@app.get("/api/shows")
async def shows():
    await refresh_cache(force=False)
    items: List[Dict[str, str]] = []
    for rk in cache.shows_by_all:
        it = await _plex_title_thumb(rk)
        if it is not None:
            items.append(it)
    return JSONResponse({"items": items})