"""
Spotify Web API client — search only.

The AI pipeline uses Spotify exclusively to translate a free-text song
reference like "Blinding Lights The Weeknd" into a canonical track id, title,
and artist. Audio features come from ReccoBeats (see reccobeats_client.py),
which is keyed on Spotify track ids — that hop is what lets us avoid
Spotify's restricted /audio-features endpoint entirely.

Auth uses the Client Credentials flow — server-to-server, no user login. The
access token is cached in-process until it expires (Spotify returns 1 hour
windows) so we don't hammer /api/token on every request.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests

from loggings import get_logger

logger = get_logger(__name__)

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_API_BASE = "https://api.spotify.com/v1"
_DEFAULT_TIMEOUT = 10  # seconds


class SpotifyError(RuntimeError):
    """Base class for any Spotify-side failure."""


class SpotifyAuthError(SpotifyError):
    """Raised when /api/token rejects our client credentials."""


class SpotifyNotFoundError(SpotifyError):
    """Search returned no results, or the track id doesn't exist."""


class SpotifyClient:
    """
    Thin wrapper around Spotify Search.

    Reads SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET from the environment
    by default; pass explicit values for tests. Fails fast at construction
    time so the UI can show a useful error before the user types anything.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        self._client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID")
        self._client_secret = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
        if not self._client_id or not self._client_secret:
            raise SpotifyAuthError(
                "Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and "
                "SPOTIFY_CLIENT_SECRET in your environment."
            )
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

    # ------------------------------------------------------------------ auth

    def _get_token(self) -> str:
        """
        Return a valid access token, refreshing if the cached one is within
        30 seconds of expiry. The 30s buffer protects us from the case where
        a request gets issued just as the token flips to expired.
        """
        if self._token and time.time() < self._token_expires_at - 30:
            return self._token

        logger.info("Requesting new Spotify access token (client credentials).")
        resp = requests.post(
            _TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self._client_id, self._client_secret),
            timeout=_DEFAULT_TIMEOUT,
        )
        if resp.status_code != 200:
            raise SpotifyAuthError(
                f"Spotify auth failed ({resp.status_code}): {resp.text}"
            )
        body = resp.json()
        self._token = body["access_token"]
        self._token_expires_at = time.time() + int(body.get("expires_in", 3600))
        return self._token

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """GET a Spotify API path with auth + uniform error mapping."""
        url = f"{_API_BASE}{path}"
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._get_token()}"},
            params=params,
            timeout=_DEFAULT_TIMEOUT,
        )
        if resp.status_code == 401:
            # Token expired or revoked mid-flight: clear and let caller retry.
            self._token = None
            raise SpotifyAuthError("Spotify rejected our access token.")
        if resp.status_code == 404:
            raise SpotifyNotFoundError(f"Spotify 404 for {path}")
        if resp.status_code != 200:
            raise SpotifyError(
                f"Spotify GET {path} failed ({resp.status_code}): {resp.text}"
            )
        return resp.json()

    # ------------------------------------------------------------------ api

    def search_track(self, query: str) -> dict:
        """
        Resolve a free-text query to the top matching track. Returns the raw
        Spotify track object (id, name, artists, etc.). Raises
        SpotifyNotFoundError when there's no match — the AI pipeline treats
        that as "the user mentioned a song we couldn't find" and proceeds
        from the description alone.
        """
        logger.info("Spotify search: %r", query)
        body = self._get("/search", params={"q": query, "type": "track", "limit": 1})
        items = body.get("tracks", {}).get("items", [])
        if not items:
            raise SpotifyNotFoundError(f"No Spotify match for {query!r}")
        return items[0]
