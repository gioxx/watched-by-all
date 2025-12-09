from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from httpx import HTTPStatusError

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.tautulli_client import TautulliClient
from app.plex_client import PlexClient
from app.jellyfin_client import JellyfinClient
from app.store import Cache


TAUTULLI_URL = os.environ.get("TAUTULLI_URL", "").strip()
TAUTULLI_APIKEY = os.environ.get("TAUTULLI_APIKEY", "").strip()
PLEX_URL = os.environ.get("PLEX_URL", "").strip()
PLEX_TOKEN = os.environ.get("PLEX_TOKEN", "").strip()
JELLYFIN_URL = os.environ.get("JELLYFIN_URL", "").strip()
JELLYFIN_APIKEY = os.environ.get("JELLYFIN_APIKEY", "").strip()
JELLYFIN_USER_MAP = os.environ.get("JELLYFIN_USER_MAP", "{}")

WATCH_THRESHOLD = float(os.environ.get("WATCH_THRESHOLD", "0.95"))
REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", "30"))
HISTORY_MAX_ROWS = int(os.environ.get("HISTORY_MAX_ROWS", "200000"))

if not (TAUTULLI_URL and TAUTULLI_APIKEY and PLEX_URL and PLEX_TOKEN):
    # Fail fast with a clear error in logs.
    raise RuntimeError("Missing required env vars: TAUTULLI_URL, TAUTULLI_APIKEY, PLEX_URL, PLEX_TOKEN")

tautulli = TautulliClient(TAUTULLI_URL, TAUTULLI_APIKEY)
plex = PlexClient(PLEX_URL, PLEX_TOKEN)
jellyfin: Optional[JellyfinClient] = None
if JELLYFIN_URL and JELLYFIN_APIKEY:
    jellyfin = JellyfinClient(JELLYFIN_URL, JELLYFIN_APIKEY)

cache = Cache()
app = FastAPI(title="Plex Watched-By-All Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")

logger = logging.getLogger("plex-watched-by-all")


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


def _completion_ratio(h: Dict[str, Any]) -> float:
    """Return completion ratio in 0..1 (best effort)."""
    pc = h.get("percent_complete")
    if pc is None:
        return 0.0
    try:
        pc_float = float(pc)
        return pc_float / 100.0 if pc_float > 1.0 else pc_float
    except Exception:
        return 0.0


def _effective_user_ids() -> List[str]:
    """Return the user_ids that should be considered for computations.

    If `cache.selected_user_ids` is empty, all users are considered.
    Otherwise, only the selected IDs that still exist in the current user list.
    """
    if cache.selected_user_ids:
        return [uid for uid in cache.users.keys() if uid in cache.selected_user_ids]
    return list(cache.users.keys())


def _provider_key(provider: str, ident: str) -> str:
    provider = provider.strip().lower()
    ident = ident.strip()
    if not provider or not ident:
        return ""
    return f"{provider}:{ident}"


def _provider_movie_key_from_guid(guid: str) -> str:
    guid = guid or ""
    if "themoviedb" in guid:
        m = re.search(r"themoviedb://(\d+)", guid)
        if m:
            return _provider_key("tmdb", m.group(1))
    if "imdb" in guid:
        m = re.search(r"imdb://(tt\d+)", guid)
        if m:
            return _provider_key("imdb", m.group(1))
    if "thetvdb" in guid:
        m = re.search(r"thetvdb://(\d+)", guid)
        if m:
            return _provider_key("tvdb", m.group(1))
    if "local" in guid:
        return ""
    return ""


def _provider_key_from_ids(ids: Dict[str, Any]) -> str:
    # Prefer TMDB, then IMDB, then TVDB
    if not ids:
        return ""
    tmdb = str(ids.get("Tmdb") or ids.get("tmdb") or "").strip()
    if tmdb:
        return _provider_key("tmdb", tmdb)
    imdb = str(ids.get("Imdb") or ids.get("imdb") or "").strip()
    if imdb:
        return _provider_key("imdb", imdb)
    tvdb = str(ids.get("Tvdb") or ids.get("tvdb") or "").strip()
    if tvdb:
        return _provider_key("tvdb", tvdb)
    return ""


def _ts_from_iso(val: str) -> float:
    if not val:
        return 0.0
    try:
        cleaned = val.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned).timestamp()
    except Exception:
        return 0.0


def _jellyfin_thumb(item_id: str, tag: str) -> str:
    if not (JELLYFIN_URL and JELLYFIN_APIKEY and item_id and tag):
        return ""
    return f"{JELLYFIN_URL}/Items/{item_id}/Images/Primary?tag={tag}&X-Emby-Token={JELLYFIN_APIKEY}"


def _record_history(user_id: str, event: Dict[str, Any]) -> None:
    cache.user_history.setdefault(user_id, []).append(event)


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
    cache.user_movie_progress.clear()
    cache.user_show_episodes.clear()
    cache.user_history.clear()

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

    provider_movie_to_plex: Dict[str, str] = {}

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

                    pk = _provider_movie_key_from_guid(str(h.get("guid", "")))
                    if pk:
                        provider_movie_to_plex.setdefault(pk, rk)

                    # Track per-user movie progress
                    cache.user_movie_progress.setdefault(uid, {})
                    entry = cache.user_movie_progress[uid].setdefault(rk, {"percent": 0.0, "completed": False})
                    entry["percent"] = max(entry.get("percent", 0.0), _completion_ratio(h))
                    entry["completed"] = bool(entry.get("completed")) or _is_completed_item(h)

                    event = {
                        "source": "plex",
                        "type": "movie",
                        "ratingKey": rk,
                        "providerKey": pk,
                        "percent": _completion_ratio(h),
                        "completed": _is_completed_item(h),
                        "date": float(h.get("date") or 0),
                        "title": h.get("title") or h.get("full_title") or "",
                    }
                    _record_history(uid, event)

            elif media_type == "episode":
                ep_rk = str(h.get("rating_key", "")).strip()
                show_rk = str(h.get("grandparent_rating_key", "")).strip()
                if show_rk:
                    cache.shows.add(show_rk)
                if ep_rk and _is_completed_item(h):
                    cache.completed.add((uid, ep_rk))

                if show_rk and ep_rk and _is_completed_item(h):
                    # Track per-user completed episodes for show progress
                    cache.user_show_episodes.setdefault(uid, {})
                    cache.user_show_episodes[uid].setdefault(show_rk, set()).add(ep_rk)

                if ep_rk:
                    event = {
                        "source": "plex",
                        "type": "episode",
                        "ratingKey": ep_rk,
                        "showRatingKey": show_rk,
                        "percent": _completion_ratio(h),
                        "completed": _is_completed_item(h),
                        "date": float(h.get("date") or 0),
                        "title": h.get("title") or "",
                        "grandparentTitle": h.get("grandparent_title") or "",
                    }
                    _record_history(uid, event)

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

    if jellyfin is not None:
        await _merge_jellyfin_history(provider_movie_to_plex)

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


async def _merge_jellyfin_history(provider_movie_to_plex: Dict[str, str]) -> None:
    """Augment completion data with Jellyfin history (movies only for now)."""
    if jellyfin is None:
        return

    try:
        jf_users = await jellyfin.get_users()
    except Exception:
        logger.warning("Jellyfin: failed to fetch users", exc_info=True)
        return

    try:
        user_map_raw = json.loads(JELLYFIN_USER_MAP or "{}")
    except Exception:
        user_map_raw = {}

    # Normalized map so you can map by ID or by name (case-insensitive):
    # {"jellyfinIdOrName": "plexUserIdOrName"}
    user_map = {str(k).lower(): str(v).strip() for k, v in user_map_raw.items()}
    name_to_plex = {v.lower(): k for k, v in cache.users.items()}

    def map_user(jf_user: Dict[str, Any]) -> Optional[str]:
        uid = str(jf_user.get("Id", "") or jf_user.get("id", "")).strip()
        name = (jf_user.get("Name") or jf_user.get("Username") or "").strip()
        name_lower = name.lower()

        # 1) Explicit mapping via env (can be by ID or name)
        mapped = user_map.get(uid.lower()) if uid else None
        if not mapped and name_lower:
            mapped = user_map.get(name_lower)
        if mapped:
            # The mapped value can be Plex user_id or friendly_name
            if mapped in cache.users:
                return mapped
            ml = mapped.lower()
            if ml in name_to_plex:
                return name_to_plex[ml]

        # 2) Fallback: match by name (case-insensitive)
        if name_lower and name_lower in name_to_plex:
            return name_to_plex[name_lower]

        return None

    mapped_users: List[str] = []
    skipped_users: List[str] = []

    for jf_u in jf_users or []:
        plex_uid = map_user(jf_u)
        jf_name = (jf_u.get("Name") or jf_u.get("Username") or jf_u.get("Id") or "").strip()
        if not plex_uid:
            skipped_users.append(jf_name)
            continue

        mapped_users.append(f"{jf_name}->{plex_uid}")

        try:
            items_resp = await jellyfin.get_user_items(str(jf_u.get("Id") or jf_u.get("id")))
        except Exception:
            logger.warning("Jellyfin: failed to fetch items for user %s", jf_name, exc_info=True)
            continue

        items = items_resp.get("Items", []) if isinstance(items_resp, dict) else (items_resp or [])
        for it in items:
            typ = (it.get("Type") or "").lower()
            if typ != "movie":
                continue

            pk = _provider_key_from_ids(it.get("ProviderIds", {}))
            plex_rk = provider_movie_to_plex.get(pk) if pk else None

            ud = it.get("UserData", {}) or {}
            played_pct = 0.0
            try:
                played_pct = float(ud.get("PlayedPercentage", 0.0)) / 100.0
            except Exception:
                played_pct = 0.0
            is_completed = bool(ud.get("Played")) or played_pct >= WATCH_THRESHOLD

            if plex_rk:
                cache.movies.add(plex_rk)
                if is_completed:
                    cache.completed.add((plex_uid, plex_rk))

                cache.user_movie_progress.setdefault(plex_uid, {})
                entry = cache.user_movie_progress[plex_uid].setdefault(plex_rk, {"percent": 0.0, "completed": False})
                entry["percent"] = max(entry.get("percent", 0.0), played_pct)
                entry["completed"] = bool(entry.get("completed")) or is_completed

            # Record history even if we cannot map to a Plex ratingKey
            event = {
                "source": "jellyfin",
                "type": "movie",
                "ratingKey": plex_rk,
                "providerKey": pk,
                "percent": played_pct,
                "completed": is_completed,
                "date": _ts_from_iso(ud.get("LastPlayedDate")) if isinstance(ud, dict) else 0.0,
                "title": it.get("Name") or "",
                "year": it.get("ProductionYear") or "",
                "jellyfinId": it.get("Id"),
                "thumb": _jellyfin_thumb(it.get("Id"), (it.get("PrimaryImageTag") or "")),
            }
            _record_history(plex_uid, event)

    if mapped_users:
        logger.info("Jellyfin: mapped users: %s", ", ".join(mapped_users))
    if skipped_users:
        logger.info("Jellyfin: skipped users without mapping/match: %s", ", ".join(skipped_users))


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


@app.get("/api/user/{user_id}/items")
async def user_items(user_id: str):
    """Return movies and show progress for a single user (also non-complete)."""
    await refresh_cache(force=False)

    if user_id not in cache.users:
        return JSONResponse({"error": "user not found"}, status_code=404)

    movies_resp: List[Dict[str, Any]] = []
    user_movies = cache.user_movie_progress.get(user_id, {})
    for rk, prog in user_movies.items():
        md = await _plex_title_thumb(rk)
        md.update(
            {
                "completed": bool(prog.get("completed")),
                "percent": round(float(prog.get("percent", 0.0)) * 100.0, 2),
            }
        )
        movies_resp.append(md)

    shows_resp: List[Dict[str, Any]] = []
    user_shows = cache.user_show_episodes.get(user_id, {})
    for show_rk, eps_seen in user_shows.items():
        total_eps = await _plex_show_episode_keys(show_rk)
        total = len(total_eps)
        watched = len(eps_seen)
        md = await _plex_title_thumb(show_rk)
        md.update(
            {
                "watchedEpisodes": watched,
                "totalEpisodes": total,
                "completed": total > 0 and watched >= total,
            }
        )
        shows_resp.append(md)

    return JSONResponse({"movies": movies_resp, "shows": shows_resp})


@app.get("/api/user/{user_id}/history")
async def user_history(user_id: str):
    """Return full watched history (Plex via Tautulli + Jellyfin) for a single user."""
    await refresh_cache(force=False)

    if user_id not in cache.users:
        return JSONResponse({"error": "user not found"}, status_code=404)

    events = cache.user_history.get(user_id, [])
    enriched: List[Dict[str, Any]] = []

    for ev in events:
        out = dict(ev)
        rk = ev.get("ratingKey")
        if rk:
            md = await _plex_title_thumb(str(rk))
            if md:
                out.update(md)
        enriched.append(out)

    def _sort_key(e: Dict[str, Any]):
        try:
            return float(e.get("date") or 0)
        except Exception:
            return 0.0

    enriched_sorted = sorted(enriched, key=_sort_key, reverse=True)
    return JSONResponse({"items": enriched_sorted})
