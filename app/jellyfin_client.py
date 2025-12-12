import httpx


class JellyfinClient:
    """Minimal Jellyfin API client (read-only)."""

    def __init__(self, base_url: str, apikey: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.apikey = apikey
        self.timeout = timeout

    async def _get(self, path: str, **params):
        url = f"{self.base_url}{path}"
        headers = {"X-Emby-Token": self.apikey, "Accept": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(url, headers=headers, params=params)
            r.raise_for_status()
            return r.json()

    async def get_users(self):
        # /Users returns all users
        return await self._get("/Users")

    async def get_user_items(self, user_id: str):
        # Fetch played movies/episodes for a user.
        return await self._get(
            f"/Users/{user_id}/Items",
            Filters="IsPlayed",
            IncludeItemTypes="Movie,Episode",
            Recursive=True,
            SortBy="DatePlayed",
            Fields="ProviderIds,SeriesId,ParentId,UserData,PrimaryImageTag,SeriesPrimaryImageTag,ImageTags,RunTimeTicks",
            EnableTotalRecordCount=False,
        )

    async def get_series_episodes(self, series_id: str):
        # All episodes for a series
        return await self._get(
            "/Items",
            ParentId=series_id,
            IncludeItemTypes="Episode",
            Recursive=True,
            Fields="Id",
            EnableTotalRecordCount=False,
        )

    async def get_season_episodes(self, season_id: str):
        return await self._get(
            "/Items",
            ParentId=season_id,
            IncludeItemTypes="Episode",
            Recursive=True,
            Fields="Id",
            EnableTotalRecordCount=False,
        )
