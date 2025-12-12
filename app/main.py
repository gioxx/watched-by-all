from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.jellyfin_client import JellyfinClient
from app.store import Cache


JELLYFIN_URL = os.environ.get("JELLYFIN_URL", "").strip()
JELLYFIN_APIKEY = os.environ.get("JELLYFIN_APIKEY", "").strip()

WATCH_THRESHOLD = float(os.environ.get("WATCH_THRESHOLD", "0.95"))
REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", "30"))

if not (JELLYFIN_URL and JELLYFIN_APIKEY):
    raise RuntimeError("Missing required env vars: JELLYFIN_URL, JELLYFIN_APIKEY")

jellyfin = JellyfinClient(JELLYFIN_URL, JELLYFIN_APIKEY)

cache = Cache()
app = FastAPI(title="Jellyfin Watched-By-All Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")

logger = logging.getLogger("jellyfin-watched-by-all")


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


async def _jellyfin_show_episode_keys(show_id: str) -> List[str]:
    if show_id in cache.show_episodes:
        return cache.show_episodes[show_id]
    if jellyfin is None:
        return []
    try:
        resp = await jellyfin.get_series_episodes(show_id)
        items = resp.get("Items", []) if isinstance(resp, dict) else (resp or [])
        eps = [str(e.get("Id")) for e in items if e.get("Id")]
        cache.show_episodes[show_id] = eps
        return eps
    except Exception:
        return []


async def _jellyfin_season_episode_keys(season_id: str) -> List[str]:
    if season_id in cache.show_episodes:
        return cache.show_episodes[season_id]
    try:
        resp = await jellyfin.get_season_episodes(season_id)
        items = resp.get("Items", []) if isinstance(resp, dict) else (resp or [])
        eps = [str(e.get("Id")) for e in items if e.get("Id")]
        cache.show_episodes[season_id] = eps
        return eps
    except Exception:
        return []


async def _show_episode_keys(show_rating_key: str) -> List[str]:
    if cache.show_episodes.get(show_rating_key):
        return cache.show_episodes[show_rating_key]
    return await _jellyfin_show_episode_keys(show_rating_key)


async def refresh_cache(force: bool = False) -> None:
    """Pull users + history from Jellyfin only and compute intersections."""
    if not force and not cache.is_stale(REFRESH_MINUTES):
        return

    cache.users.clear()
    cache.completed.clear()
    cache.movies.clear()
    cache.shows.clear()
    cache.seasons.clear()
    cache.movies_by_all.clear()
    cache.shows_by_all.clear()
    cache.user_movie_progress.clear()
    cache.user_show_episodes.clear()
    cache.user_season_episodes.clear()
    cache.user_history.clear()
    cache.jellyfin_users.clear()
    cache.jellyfin_meta.clear()
    cache.season_runtime_minutes.clear()
    cache.season_episode_seen.clear()

    try:
        jf_users = await jellyfin.get_users()
        for ju in jf_users or []:
            juid = str(ju.get("Id") or ju.get("id") or "").strip()
            jname = (ju.get("Name") or ju.get("Username") or juid).strip()
            if juid:
                cache.jellyfin_users[juid] = jname
                cache.users[juid] = jname
    except Exception:
        logger.warning("Jellyfin: failed to fetch users", exc_info=True)
        return

    for juid, _ in cache.users.items():
        try:
            items_resp = await jellyfin.get_user_items(juid)
        except Exception:
            logger.warning("Jellyfin: failed to fetch items for user %s", juid, exc_info=True)
            continue

        items = items_resp.get("Items", []) if isinstance(items_resp, dict) else (items_resp or [])
        for it in items:
            series_thumb_url = ""
            show_id = ""
            season_id = ""
            typ = (it.get("Type") or "").lower()
            if typ not in ("movie", "episode"):
                continue

            item_id = str(it.get("Id") or "").strip()
            if not item_id:
                continue

            primary_tag = it.get("PrimaryImageTag") or (it.get("ImageTags") or {}).get("Primary") or ""
            series_primary_tag = it.get("SeriesPrimaryImageTag") or ""
            thumb_url = ""
            if primary_tag:
                thumb_url = _jellyfin_thumb(item_id, primary_tag)
            elif series_primary_tag and it.get("SeriesId"):
                thumb_url = _jellyfin_thumb(str(it.get("SeriesId")), series_primary_tag)

            ud = it.get("UserData", {}) or {}
            played_pct = 0.0
            try:
                played_pct = float(ud.get("PlayedPercentage", 0.0)) / 100.0
            except Exception:
                played_pct = 0.0
            is_completed = bool(ud.get("Played")) or played_pct >= WATCH_THRESHOLD

            if typ == "movie":
                cache.movies.add(item_id)
                if is_completed:
                    cache.completed.add((juid, item_id))
                cache.user_movie_progress.setdefault(juid, {})
                entry = cache.user_movie_progress[juid].setdefault(item_id, {"percent": 0.0, "completed": False})
                entry["percent"] = max(entry.get("percent", 0.0), played_pct)
                entry["completed"] = bool(entry.get("completed")) or is_completed

                runtime_ticks = it.get("RunTimeTicks") or 0
                runtime_minutes = 0
                try:
                    runtime_minutes = int(int(runtime_ticks) / 10_000_000 / 60)
                except Exception:
                    runtime_minutes = 0

                official_rating = it.get("OfficialRating") or ""
                community_rating = it.get("CommunityRating") or None
                play_count = None
                try:
                    ud = it.get("UserData") or {}
                    pc_val = ud.get("PlayCount")
                    play_count = int(pc_val) if pc_val is not None else None
                except Exception:
                    play_count = None

                cache.jellyfin_meta[item_id] = {
                    "ratingKey": item_id,
                    "title": it.get("Name") or "",
                    "year": str(it.get("ProductionYear") or ""),
                    "type": "movie",
                    "thumb": thumb_url,
                    "runtimeMinutes": runtime_minutes,
                    "officialRating": official_rating,
                    "communityRating": community_rating,
                    "playCount": play_count,
                }

            elif typ == "episode":
                show_id = str(it.get("SeriesId") or "").strip()
                season_id = str(it.get("ParentId") or it.get("SeasonId") or "").strip()
                if series_primary_tag and show_id:
                    series_thumb_url = _jellyfin_thumb(show_id, series_primary_tag)
                if show_id:
                    cache.shows.add(show_id)
                if season_id:
                    cache.seasons.add(season_id)
                if is_completed:
                    cache.completed.add((juid, item_id))
                if show_id and is_completed:
                    cache.user_show_episodes.setdefault(juid, {})
                    cache.user_show_episodes[juid].setdefault(show_id, set()).add(item_id)
                if season_id and is_completed:
                    cache.user_season_episodes.setdefault(juid, {})
                    cache.user_season_episodes[juid].setdefault(season_id, set()).add(item_id)
                if show_id and show_id not in cache.jellyfin_meta:
                    cache.jellyfin_meta[show_id] = {
                        "ratingKey": show_id,
                        "title": it.get("SeriesName") or "",
                        "year": "",
                        "type": "show",
                        "thumb": series_thumb_url or thumb_url,
                    }
                if season_id and season_id not in cache.jellyfin_meta:
                    cache.jellyfin_meta[season_id] = {
                        "ratingKey": season_id,
                        "title": f"{it.get('SeriesName') or ''} {it.get('SeasonName') or ''}".strip(),
                        "year": "",
                        "type": "season",
                        "thumb": thumb_url,
                    }

                # Aggregate runtime per season once per episode
                runtime_ticks = it.get("RunTimeTicks") or 0
                runtime_minutes = 0
                try:
                    runtime_minutes = int(int(runtime_ticks) / 10_000_000 / 60)
                except Exception:
                    runtime_minutes = 0
                if season_id:
                    seen = cache.season_episode_seen.setdefault(season_id, set())
                    if item_id not in seen:
                        seen.add(item_id)
                        cache.season_runtime_minutes[season_id] = cache.season_runtime_minutes.get(season_id, 0) + runtime_minutes

            event = {
                "source": "jellyfin",
                "type": typ,
                "ratingKey": item_id,
                "providerKey": _provider_key_from_ids(it.get("ProviderIds", {})),
                "percent": played_pct,
                "completed": is_completed,
                "date": _ts_from_iso(ud.get("LastPlayedDate")) if isinstance(ud, dict) else 0.0,
                "title": it.get("Name") or "",
                "year": it.get("ProductionYear") or "",
                "seriesName": it.get("SeriesName") or "",
                "seasonName": it.get("SeasonName") or "",
                "episodeTitle": it.get("Name") or "",
                "episodeIndex": it.get("IndexNumber"),
                "seasonIndex": it.get("ParentIndexNumber"),
                "seriesId": show_id or "",
                "seasonId": season_id or "",
                "episodeId": item_id,
                "jellyfinId": item_id,
                "seriesThumb": series_thumb_url or thumb_url,
                "thumb": thumb_url,
            }
            _record_history(juid, event)

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

    # Seasons watched by all (ALL episodes in the season watched by ALL users)
    seasons_by_all: List[str] = []
    for season_id in cache.seasons:
        eps = await _jellyfin_season_episode_keys(season_id)
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
            seasons_by_all.append(season_id)
    cache.shows_by_all = sorted(seasons_by_all, key=lambda x: int(x) if x.isdigit() else x)

    cache.last_refresh_ts = time.time()


async def _item_title_thumb(rating_key: str) -> Dict[str, str] | None:
    jf_md = cache.jellyfin_meta.get(str(rating_key))
    if jf_md:
        return jf_md
    return None


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
        it = await _item_title_thumb(rk)
        if it is not None:
            items.append(it)
    return JSONResponse({"items": items})


@app.get("/api/shows")
async def shows():
    await refresh_cache(force=False)
    items: List[Dict[str, str]] = []
    for rk in cache.shows_by_all:
        it = await _item_title_thumb(rk)
        if it is not None:
            it["runtimeMinutes"] = cache.season_runtime_minutes.get(rk, 0)
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
        md = await _item_title_thumb(rk)
        if md is None:
            continue
        md.update(
            {
                "completed": bool(prog.get("completed")),
                "percent": round(float(prog.get("percent", 0.0)) * 100.0, 2),
            }
        )
        movies_resp.append(md)

    shows_resp: List[Dict[str, Any]] = []
    user_seasons = cache.user_season_episodes.get(user_id, {})
    for season_id, eps_seen in user_seasons.items():
        total_eps = await _jellyfin_season_episode_keys(season_id)
        total = len(total_eps)
        watched = len(eps_seen)
        md = await _item_title_thumb(season_id)
        if md is None:
            continue
        md_runtime = cache.season_runtime_minutes.get(season_id, 0)
        md.update(
            {
                "watchedEpisodes": watched,
                "totalEpisodes": total,
                "completed": total > 0 and watched >= total,
                "runtimeMinutes": md_runtime,
            }
        )
        shows_resp.append(md)

    return JSONResponse({"movies": movies_resp, "shows": shows_resp})


@app.get("/api/user/{user_id}/history")
async def user_history(user_id: str):
    """Return full watched history (Jellyfin)."""
    await refresh_cache(force=False)

    if user_id not in cache.users:
        return JSONResponse({"error": "user not found"}, status_code=404)

    events = cache.user_history.get(user_id, [])
    enriched: List[Dict[str, Any]] = []

    for ev in events:
        out = dict(ev)
        rk = ev.get("ratingKey")
        thumb = ev.get("thumb")
        series_id = ev.get("seriesId")

        if rk:
            md = await _item_title_thumb(str(rk))
            if md:
                out.update(md)
                thumb = out.get("thumb") or thumb

        if series_id:
            series_md = await _item_title_thumb(str(series_id))
            if series_md:
                if series_md.get("title") and not out.get("seriesName"):
                    out["seriesName"] = series_md.get("title")
                if series_md.get("thumb"):
                    out["seriesThumb"] = series_md.get("thumb")
                    if not thumb:
                        thumb = series_md.get("thumb")
                out["seriesRatingKey"] = series_id

        # If there is no poster at all, skip the entry.
        if not thumb:
            continue

        out["thumb"] = thumb
        if not out.get("title") and out.get("episodeTitle"):
            out["title"] = out.get("episodeTitle")
        enriched.append(out)

    def _sort_key(e: Dict[str, Any]):
        try:
            return float(e.get("date") or 0)
        except Exception:
            return 0.0

    enriched_sorted = sorted(enriched, key=_sort_key, reverse=True)
    return JSONResponse({"items": enriched_sorted})
