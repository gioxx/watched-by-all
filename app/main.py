from __future__ import annotations

import asyncio
import logging
import os
import time
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Set

import httpx
from fastapi import Body, FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.jellyfin_client import JellyfinClient
from app.store import Cache


JELLYFIN_URL = os.environ.get("JELLYFIN_URL", "").strip()
JELLYFIN_APIKEY = os.environ.get("JELLYFIN_APIKEY", "").strip()

WATCH_THRESHOLD = float(os.environ.get("WATCH_THRESHOLD", "0.95"))
REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", "30"))
PROXY_IMAGES = os.environ.get("PROXY_IMAGES", "true").strip().lower() in ("1", "true", "yes", "on")
IMAGE_CACHE_SECONDS = int(os.environ.get("IMAGE_CACHE_SECONDS", "86400"))
JELLYFIN_TIMEOUT = float(os.environ.get("JELLYFIN_TIMEOUT", "5.0"))
THUMB_CACHE_DIR = os.environ.get("THUMB_CACHE_DIR", "thumb_cache").strip() or "thumb_cache"
THUMB_CACHE_TTL_HOURS = float(os.environ.get("THUMB_CACHE_TTL_HOURS", "72"))
THUMB_FETCH_TIMEOUT = float(os.environ.get("THUMB_FETCH_TIMEOUT", "5.0"))
INTERNAL_HTTP_BASE = os.environ.get("INTERNAL_HTTP_BASE", "http://127.0.0.1:8088").strip()
JELLYFIN_TIMEOUT = float(os.environ.get("JELLYFIN_TIMEOUT", "5.0"))
THUMB_CACHE_DIR = os.environ.get("THUMB_CACHE_DIR", "thumb_cache").strip() or "thumb_cache"
THUMB_CACHE_TTL_HOURS = float(os.environ.get("THUMB_CACHE_TTL_HOURS", "72"))
THUMB_FETCH_TIMEOUT = float(os.environ.get("THUMB_FETCH_TIMEOUT", "5.0"))
JELLYFIN_TIMEOUT = float(os.environ.get("JELLYFIN_TIMEOUT", "5.0"))
try:
    THUMB_MAX_HEIGHT = int(os.environ.get("THUMB_MAX_HEIGHT", "500"))
except ValueError:
    THUMB_MAX_HEIGHT = 500
THUMB_MAX_HEIGHT = max(0, THUMB_MAX_HEIGHT)
try:
    THUMB_MAX_DOWNLOAD_BYTES = int(os.environ.get("THUMB_MAX_DOWNLOAD_BYTES", str(8 * 1024 * 1024)))
except ValueError:
    THUMB_MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
THUMB_MAX_DOWNLOAD_BYTES = max(256 * 1024, THUMB_MAX_DOWNLOAD_BYTES)
try:
    THUMB_REBUILD_BATCH = int(os.environ.get("THUMB_REBUILD_BATCH", "5"))
except ValueError:
    THUMB_REBUILD_BATCH = 5
try:
    THUMB_REBUILD_PAUSE_MS = float(os.environ.get("THUMB_REBUILD_PAUSE_MS", "40"))
except ValueError:
    THUMB_REBUILD_PAUSE_MS = 40.0
THUMB_REBUILD_BATCH = max(1, THUMB_REBUILD_BATCH)
THUMB_REBUILD_PAUSE_MS = max(0.0, THUMB_REBUILD_PAUSE_MS)

if not (JELLYFIN_URL and JELLYFIN_APIKEY):
    raise RuntimeError("Missing required env vars: JELLYFIN_URL, JELLYFIN_APIKEY")

os.makedirs(THUMB_CACHE_DIR, exist_ok=True)

jellyfin = JellyfinClient(JELLYFIN_URL, JELLYFIN_APIKEY, timeout=JELLYFIN_TIMEOUT)

cache = Cache()
thumb_cache_last_refresh = 0.0
thumb_cache_job_state: Dict[str, Any] = {
    "running": False,
    "phase": "idle",
    "startedAt": 0.0,
    "finishedAt": 0.0,
    "lastError": "",
    "lastAction": "",
    "processed": 0,
    "total": 0,
    "percent": 0.0,
}
thumb_cache_job_task: asyncio.Task | None = None
app = FastAPI(title="Jellyfin Watched-By-All Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/thumbs", StaticFiles(directory=THUMB_CACHE_DIR), name="thumbs")

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
    if not (item_id and tag):
        return ""
    size_params: List[str] = []
    if THUMB_MAX_HEIGHT > 0:
        size_params.append(f"maxHeight={THUMB_MAX_HEIGHT}")
    size_qs = "&".join(size_params)
    if PROXY_IMAGES:
        return f"/image/{item_id}?tag={tag}" + (f"&{size_qs}" if size_qs else "")
    if not (JELLYFIN_URL and JELLYFIN_APIKEY):
        return ""
    return f"{JELLYFIN_URL}/Items/{item_id}/Images/Primary?tag={tag}&X-Emby-Token={JELLYFIN_APIKEY}" + (f"&{size_qs}" if size_qs else "")


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


async def refresh_cache(force: bool = False, recache_thumbs: bool = False, low_priority: bool = False) -> None:
    global thumb_cache_last_refresh
    """Pull users + history from Jellyfin only and compute intersections."""
    if not force and not cache.is_stale(REFRESH_MINUTES):
        return

    cache.users.clear()
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
    cache.season_last_view.clear()

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

    processed_items = 0
    referenced_thumb_files: Set[str] = set()
    for juid, _ in cache.users.items():
        try:
            items_resp = await jellyfin.get_user_items(juid)
        except Exception:
            logger.warning("Jellyfin: failed to fetch items for user %s", juid, exc_info=True)
            continue

        items = items_resp.get("Items", []) if isinstance(items_resp, dict) else (items_resp or [])
        if recache_thumbs and low_priority:
            thumb_cache_job_state["total"] += sum(
                1
                for it in items
                if (it.get("Type") or "").lower() in ("movie", "episode") and str(it.get("Id") or "").strip()
            )
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
            thumb_url = await _cache_thumb(thumb_url, force=recache_thumbs)
            _track_cached_thumb(thumb_url, referenced_thumb_files)

            ud = it.get("UserData", {}) or {}
            played_pct = 0.0
            try:
                played_pct = float(ud.get("PlayedPercentage", 0.0)) / 100.0
            except Exception:
                played_pct = 0.0
            is_completed = bool(ud.get("Played")) or played_pct >= WATCH_THRESHOLD
            date_ts = _ts_from_iso(ud.get("LastPlayedDate")) if isinstance(ud, dict) else 0.0

            if typ == "movie":
                cache.movies.add(item_id)
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
                        "thumb": await _cache_thumb(series_thumb_url or thumb_url, force=recache_thumbs),
                    }
                    _track_cached_thumb(cache.jellyfin_meta[show_id].get("thumb", ""), referenced_thumb_files)
                if season_id and season_id not in cache.jellyfin_meta:
                    cache.jellyfin_meta[season_id] = {
                        "ratingKey": season_id,
                        "title": f"{it.get('SeriesName') or ''} {it.get('SeasonName') or ''}".strip(),
                        "year": "",
                        "type": "season",
                        "thumb": await _cache_thumb(thumb_url, force=recache_thumbs),
                    }
                    _track_cached_thumb(cache.jellyfin_meta[season_id].get("thumb", ""), referenced_thumb_files)

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
                if season_id and date_ts:
                    cache.season_last_view[season_id] = max(cache.season_last_view.get(season_id, 0.0), date_ts)

                thumb_url = await _cache_thumb(thumb_url, force=recache_thumbs)
                series_thumb_url = await _cache_thumb(series_thumb_url, force=recache_thumbs)
                _track_cached_thumb(thumb_url, referenced_thumb_files)
                _track_cached_thumb(series_thumb_url, referenced_thumb_files)

            event = {
                "source": "jellyfin",
                "type": typ,
                "ratingKey": item_id,
                "providerKey": _provider_key_from_ids(it.get("ProviderIds", {})),
                "percent": played_pct,
                "completed": is_completed,
                "date": date_ts,
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
            processed_items += 1
            if recache_thumbs and low_priority:
                thumb_cache_job_state["processed"] = processed_items
                total_items = int(thumb_cache_job_state.get("total") or 0)
                thumb_cache_job_state["percent"] = (processed_items * 100.0 / total_items) if total_items > 0 else 0.0
            if low_priority and recache_thumbs and (processed_items % THUMB_REBUILD_BATCH == 0):
                await asyncio.sleep(THUMB_REBUILD_PAUSE_MS / 1000.0)

    if not recache_thumbs and cache.users and processed_items > 0:
        _gc_unused_thumb_cache_files(referenced_thumb_files)

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
        if all(bool(cache.user_movie_progress.get(uid, {}).get(mk, {}).get("completed")) for uid in user_ids):
            movies_by_all.append(mk)
    cache.movies_by_all = sorted(movies_by_all, key=lambda x: int(x) if x.isdigit() else x)

    # Seasons watched by all (ALL episodes in the season watched by ALL users)
    seasons_by_all: List[str] = []
    for season_id in cache.seasons:
        eps = await _jellyfin_season_episode_keys(season_id)
        if not eps:
            continue
        eps_set = set(eps)
        ok = True
        for uid in user_ids:
            watched_eps = cache.user_season_episodes.get(uid, {}).get(season_id, set())
            if not eps_set.issubset(watched_eps):
                ok = False
                break
        if ok:
            seasons_by_all.append(season_id)
    cache.shows_by_all = sorted(seasons_by_all, key=lambda x: int(x) if x.isdigit() else x)

    cache.last_refresh_ts = time.time()
    thumb_cache_last_refresh = cache.last_refresh_ts


async def _item_title_thumb(rating_key: str) -> Dict[str, str] | None:
    jf_md = cache.jellyfin_meta.get(str(rating_key))
    if jf_md:
        return jf_md
    return None


def _thumb_cache_path(url: str) -> str:
    if not url:
        return ""
    fname = hashlib.blake2b(url.encode("utf-8"), digest_size=16).hexdigest()  # deterministic key
    return os.path.join(THUMB_CACHE_DIR, f"{fname}.jpg")


def _track_cached_thumb(local_url: str, referenced: Set[str]) -> None:
    if not local_url or not local_url.startswith("/thumbs/"):
        return
    filename = local_url.split("/thumbs/", 1)[1].split("?", 1)[0].strip()
    if filename:
        referenced.add(filename)


def _gc_unused_thumb_cache_files(referenced: Set[str]) -> int:
    if not os.path.isdir(THUMB_CACHE_DIR):
        return 0
    removed = 0
    try:
        for entry in os.scandir(THUMB_CACHE_DIR):
            if not entry.is_file():
                continue
            if entry.name in referenced:
                continue
            try:
                os.remove(entry.path)
                removed += 1
            except Exception:
                pass
    except Exception:
        return removed
    return removed


async def _cache_thumb(url: str, force: bool = False) -> str:
    """Cache a remote thumbnail locally and return the local URL."""
    if not url:
        return url
    request_url = url
    if url.startswith("/"):
        request_url = f"{INTERNAL_HTTP_BASE.rstrip('/')}{url}"
    fname = _thumb_cache_path(url)
    if not fname:
        return url
    ttl_seconds = max(THUMB_CACHE_TTL_HOURS, 0) * 3600
    now = time.time()
    try:
        if os.path.exists(fname) and not force:
            if ttl_seconds <= 0 or (now - os.path.getmtime(fname)) < ttl_seconds:
                return f"/thumbs/{os.path.basename(fname)}"
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=THUMB_FETCH_TIMEOUT) as client:
            r = await client.get(request_url, follow_redirects=True)
            r.raise_for_status()
            content_length = int(r.headers.get("content-length") or "0")
            if content_length > THUMB_MAX_DOWNLOAD_BYTES:
                return url
            body = bytearray()
            async for chunk in r.aiter_bytes():
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) > THUMB_MAX_DOWNLOAD_BYTES:
                    return url
            with open(fname, "wb") as dst:
                dst.write(body)
            os.utime(fname, (now, now))
            return f"/thumbs/{os.path.basename(fname)}"
    except Exception:
        return url


def _thumb_cache_status() -> Dict[str, Any]:
    files = 0
    size = 0
    if os.path.isdir(THUMB_CACHE_DIR):
        try:
            for entry in os.scandir(THUMB_CACHE_DIR):
                if entry.is_file():
                    files += 1
                    try:
                        size += entry.stat().st_size
                    except Exception:
                        pass
        except Exception:
            pass
    processed = int(thumb_cache_job_state.get("processed") or 0)
    total = int(thumb_cache_job_state.get("total") or 0)
    started_at = float(thumb_cache_job_state.get("startedAt") or 0.0)
    eta_seconds = 0
    if bool(thumb_cache_job_state.get("running")) and processed > 0 and total > processed and started_at > 0:
        elapsed = max(0.001, time.time() - started_at)
        rate = processed / elapsed
        if rate > 0:
            eta_seconds = max(0, int((total - processed) / rate))

    return {
        "files": files,
        "size": size,
        "lastRefresh": thumb_cache_last_refresh,
        "ttlHours": THUMB_CACHE_TTL_HOURS,
        "jobRunning": bool(thumb_cache_job_state.get("running")),
        "jobPhase": str(thumb_cache_job_state.get("phase") or "idle"),
        "jobStartedAt": started_at,
        "jobFinishedAt": float(thumb_cache_job_state.get("finishedAt") or 0.0),
        "jobLastError": str(thumb_cache_job_state.get("lastError") or ""),
        "jobLastAction": str(thumb_cache_job_state.get("lastAction") or ""),
        "jobProcessed": processed,
        "jobTotal": total,
        "jobPercent": float(thumb_cache_job_state.get("percent") or 0.0),
        "jobEtaSeconds": eta_seconds,
    }


def _clear_thumb_cache_files() -> int:
    removed = 0
    if not os.path.isdir(THUMB_CACHE_DIR):
        return 0
    try:
        for entry in os.scandir(THUMB_CACHE_DIR):
            if entry.is_file():
                try:
                    os.remove(entry.path)
                    removed += 1
                except Exception:
                    pass
    except Exception:
        pass
    return removed


async def _run_thumb_cache_job(clear_first: bool = False) -> None:
    thumb_cache_job_state["running"] = True
    thumb_cache_job_state["phase"] = "clearing" if clear_first else "refreshing"
    thumb_cache_job_state["startedAt"] = time.time()
    thumb_cache_job_state["finishedAt"] = 0.0
    thumb_cache_job_state["lastError"] = ""
    thumb_cache_job_state["lastAction"] = "clear_rebuild" if clear_first else "refresh"
    thumb_cache_job_state["processed"] = 0
    thumb_cache_job_state["total"] = 0
    thumb_cache_job_state["percent"] = 0.0
    try:
        if clear_first:
            _clear_thumb_cache_files()
            # Heavy job: rebuild cache from scratch.
            await refresh_cache(force=True, recache_thumbs=True, low_priority=True)
        else:
            # Lightweight refresh: update data and refresh thumbnails only when missing/expired.
            await refresh_cache(force=True, recache_thumbs=False, low_priority=False)
        total_items = int(thumb_cache_job_state.get("total") or 0)
        done_items = int(thumb_cache_job_state.get("processed") or 0)
        if total_items > 0:
            thumb_cache_job_state["percent"] = min(100.0, (done_items * 100.0 / total_items))
    except Exception as exc:
        thumb_cache_job_state["lastError"] = str(exc)
    finally:
        thumb_cache_job_state["running"] = False
        thumb_cache_job_state["phase"] = "idle"
        if not thumb_cache_job_state.get("lastError"):
            thumb_cache_job_state["percent"] = 100.0
        thumb_cache_job_state["finishedAt"] = time.time()


def _start_thumb_cache_job(clear_first: bool = False) -> bool:
    global thumb_cache_job_task
    if thumb_cache_job_task and not thumb_cache_job_task.done():
        return False
    thumb_cache_job_task = asyncio.create_task(_run_thumb_cache_job(clear_first=clear_first))
    return True


@app.on_event("startup")
async def _startup() -> None:
    # Quick reachability probe (non-blocking) to surface Jellyfin URL issues early.
    async def probe_jellyfin():
        test_url = f"{JELLYFIN_URL}/System/Info"
        try:
            async with httpx.AsyncClient(timeout=JELLYFIN_TIMEOUT) as client:
                r = await client.get(test_url, headers={"X-Emby-Token": JELLYFIN_APIKEY})
                r.raise_for_status()
                logger.info("Jellyfin probe OK: %s", test_url)
        except Exception as exc:
            logger.warning("Jellyfin probe failed (%s): %s", test_url, exc)

    # Refresh on boot without blocking startup (useful when Jellyfin is slow/offline).
    async def boot_refresh():
        try:
            await refresh_cache(force=True)
        except Exception:
            pass

    asyncio.create_task(probe_jellyfin())
    asyncio.create_task(boot_refresh())

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


@app.get("/image/{item_id}")
async def image_proxy(item_id: str, tag: str, maxHeight: int | None = None):
    """Proxy Jellyfin images to avoid mixed-content or private-host issues."""
    if not tag:
        return JSONResponse({"error": "tag is required"}, status_code=400)

    url = f"{JELLYFIN_URL}/Items/{item_id}/Images/Primary"
    headers = {"X-Emby-Token": JELLYFIN_APIKEY}
    params: Dict[str, Any] = {"tag": tag}
    if THUMB_MAX_HEIGHT > 0:
        effective_height = maxHeight if (maxHeight is not None and maxHeight > 0) else THUMB_MAX_HEIGHT
        params["maxHeight"] = min(effective_height, THUMB_MAX_HEIGHT)
    elif maxHeight is not None and maxHeight > 0:
        params["maxHeight"] = maxHeight

    try:
        async with httpx.AsyncClient(timeout=JELLYFIN_TIMEOUT) as client:
            r = await client.get(url, headers=headers, params=params)
            r.raise_for_status()
    except Exception as exc:
        status = getattr(exc.response, "status_code", 502) if hasattr(exc, "response") else 502
        level = logger.info if status == 404 else logger.warning
        level("Image proxy failed for %s (%s): %s", item_id, tag, exc)
        return JSONResponse({"error": "image fetch failed"}, status_code=status)

    media_type = r.headers.get("content-type", "image/jpeg")
    resp = Response(content=r.content, media_type=media_type)
    resp.headers["Cache-Control"] = f"public, max-age={IMAGE_CACHE_SECONDS}"
    return resp


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


@app.get("/api/thumb-cache/status")
async def api_thumb_cache_status():
    return JSONResponse({"ok": True, **_thumb_cache_status()})


@app.post("/api/thumb-cache/refresh")
async def api_thumb_cache_refresh():
    started = _start_thumb_cache_job(clear_first=False)
    return JSONResponse({"ok": True, "started": started, **_thumb_cache_status()})


@app.post("/api/thumb-cache/clear")
async def api_thumb_cache_clear():
    started = _start_thumb_cache_job(clear_first=True)
    return JSONResponse({"ok": True, "started": started, **_thumb_cache_status()})


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
            it["lastViewed"] = cache.season_last_view.get(rk, 0.0)
            items.append(it)
    items.sort(key=lambda x: x.get("lastViewed", 0) or 0, reverse=True)
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
