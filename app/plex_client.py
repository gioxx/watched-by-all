import httpx

class PlexClient:
    """
    Minimal Plex API client.

    Plex supports item detail via:
      /library/metadata/{ratingKey}?X-Plex-Token=...
    """
    def __init__(self, base_url: str, token: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    async def _get_json(self, path: str, **params):
        url = f"{self.base_url}{path}"
        q = {"X-Plex-Token": self.token}
        q.update({k: v for k, v in params.items() if v is not None})
        headers = {"Accept": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(url, params=q, headers=headers)
            r.raise_for_status()
            return r.json()

    async def get_metadata(self, rating_key: str):
        return await self._get_json(f"/library/metadata/{rating_key}")

    async def get_children(self, rating_key: str):
        # Used for seasons (children of show) and episodes (children of season)
        return await self._get_json(f"/library/metadata/{rating_key}/children")

    async def get_all_leaves(self, rating_key: str):
        # Returns all leaf items (e.g., all episodes for a show) when supported.
        return await self._get_json(f"/library/metadata/{rating_key}/allLeaves")