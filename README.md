# Watched (_former Jellyfin – Watched by Everyone_)

Web dashboard for Jellyfin to see what’s been watched and what is watched by **all** selected users. It runs as a FastAPI app that serves the static UI.

## Features
- **Movies**: shows titles marked as watched by every selected user, with runtime/ratings/playcount.
- **Seasons (complete)**: seasons where every episode was watched by everyone, plus total runtime.
- **User history**: full history for a chosen user (movies + series). Series cards open a per-series episode detail page grouped by season.
- **Episode detail page**: chronological view of watched episodes with posters and dates.
- **UI controls**: user selector, manual refresh, search, language toggle (EN/IT), light/dark toggle (persists), GitHub link.

## Requirements
- Jellyfin server with an API key.
- Python 3.11+ (or use Docker).
- Network reachability from this app to Jellyfin.

## Configuration
Environment variables (see `docker-compose.yml`):
- `JELLYFIN_URL` (required): base URL of your Jellyfin, e.g. `http://jellyfin:8096`.
- `JELLYFIN_APIKEY` (required): Jellyfin API key.
- `WATCH_THRESHOLD` (optional, default `0.95`): fraction (0-1) of an item that counts as watched if `Played` isn’t set.
- `REFRESH_MINUTES` (optional, default `30`): minimum minutes between automatic refreshes of cached data.
- `PROXY_IMAGES` (optional, default `true`): when `true`, posters are proxied through this app to avoid mixed-content or private-host issues (useful behind HTTPS/CDN tunnels).
- `IMAGE_CACHE_SECONDS` (optional, default `86400`): cache-control duration for proxied images.

## Run (Docker)
```sh
docker compose up -d
```
Then open http://localhost:8088.

### Prebuilt image (GitHub Container Registry or Docker Hub)
When published, you can pull instead of building locally:
```sh
# GitHub Container Registry
docker pull ghcr.io/gioxx/watched-by-all:latest

# or Docker Hub
docker pull gfsolone/watched-by-all:latest

docker run -d --name watched \
  -p 8088:8088 \
  -e JELLYFIN_URL=http://jellyfin:8096 \
  -e JELLYFIN_APIKEY=YOUR_JELLYFIN_API_KEY \
  -e WATCH_THRESHOLD=0.95 \
  -e REFRESH_MINUTES=30 \
  ghcr.io/gioxx/watched-by-all:latest
```

## Run (local Python)
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
JELLYFIN_URL=http://... JELLYFIN_APIKEY=... uvicorn app.main:app --reload --port 8088
```
Open http://localhost:8088.

## Using the web app
- **Tabs**: Movies, Seasons (complete), User history.
- **Users**: click “Users” to choose which Jellyfin users are considered for the “watched by all” logic (empty selection = everyone).
- **Refresh**: forces a recount against Jellyfin and updates the summary bar.
- **Search**: filters the current tab client-side.
- **History**: pick a user, browse their history; click a series card to open the episode detail page.
- **Language**: toggle EN/IT in the header (per-browser localStorage).
- **Theme**: light/dark toggle in the header (per-browser localStorage).

## Localization / customization
- Locale strings live in `static/locales/en.json` and `static/locales/it.json`. Add another by copying one, translating, and referencing it in `static/index.html` and `static/history_detail.html`.
- Favicon/logo: `static/watched_icon.webp`.
- Styles: `static/styles.css`.

## API surface (served by FastAPI)
- `GET /api/summary`: counts for users/movies/seasons + last refresh timestamp.
- `GET /api/movies`: movies watched by all selected users.
- `GET /api/shows`: seasons watched by all selected users.
- `GET /api/users`: detected Jellyfin users + currently selected IDs.
- `POST /api/selected-users`: `{ "selected_user_ids": ["..."] }` to persist selection.
- `POST /api/refresh`: force refresh the cache now.
- `GET /api/user/{user_id}/history`: full history for a user (movies + episodes).
- `GET /image/{item_id}?tag=...`: proxies a Jellyfin image (used by the UI when `PROXY_IMAGES=true`).
- Static assets at `/static/*` serve the UI.
