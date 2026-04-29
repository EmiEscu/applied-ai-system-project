"""
ReccoBeats audio-features client — drop-in replacement for the deprecated
Spotify /audio-features endpoint.

ReccoBeats is a free, public, unauthenticated API that mirrors the original
Spotify feature schema (energy, valence, danceability, acousticness, tempo,
and several others). It accepts Spotify track IDs directly via the lookup
endpoint, which makes it a clean fallback after a Spotify Search call.

Two-step flow:
  1. /v1/track?ids=<spotify_id>     → returns the ReccoBeats internal UUID
  2. /v1/track/<rb_id>/audio-features → returns the feature vector

Why two hops instead of one:
  ReccoBeats keys their audio-features endpoint by their own UUID, not by the
  Spotify ID. The lookup call is cheap and we cache nothing across requests
  because in practice the AI flow only resolves one song per user prompt.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

from loggings import get_logger

logger = get_logger(__name__)


@dataclass
class TrackFeatures:
    """A flattened (track metadata + audio features) record.

    Field set mirrors the Spotify audio-features schema so a TrackFeatures
    can drop straight into a UserProfile via the AI pipeline. Fields that
    ReccoBeats sometimes omits (e.g. instrumentalness on rare entries)
    default to 0.0 / 0 so callers don't have to None-check every read.
    """
    track_id: str  # Spotify track id — what the AI pipeline searched on
    title: str
    artist: str
    energy: float
    valence: float
    danceability: float
    acousticness: float
    tempo: float
    instrumentalness: float = 0.0
    liveness: float = 0.0
    loudness: float = 0.0
    speechiness: float = 0.0
    mode: int = 0
    key: int = 0

_API_BASE = "https://api.reccobeats.com/v1"
_DEFAULT_TIMEOUT = 10  # seconds


class ReccoBeatsError(RuntimeError):
    """Base class for ReccoBeats failures (network, 5xx, malformed payload)."""


class ReccoBeatsNotFoundError(ReccoBeatsError):
    """The requested Spotify track is not in ReccoBeats' catalog."""


class ReccoBeatsClient:
    """
    Stateless client for the two ReccoBeats endpoints we use.

    No API key, no auth, no token refresh — the constructor takes nothing.
    A class is still useful here so tests can stub out the underlying GET
    and so the AI pipeline holds a single object instead of free functions.
    """

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------ api

    def lookup_by_spotify_id(self, spotify_id: str) -> str:
        """
        Resolve a Spotify track ID to ReccoBeats' internal UUID. Raises
        ReccoBeatsNotFoundError when the track isn't in their catalog so the
        AI pipeline can fall back to Gemini-only estimation.
        """
        logger.info("ReccoBeats lookup by Spotify id: %s", spotify_id)
        resp = requests.get(
            f"{_API_BASE}/track",
            params={"ids": spotify_id},
            timeout=self._timeout,
        )
        if resp.status_code != 200:
            raise ReccoBeatsError(
                f"ReccoBeats /track lookup failed ({resp.status_code}): {resp.text}"
            )
        items = resp.json().get("content", [])
        if not items:
            raise ReccoBeatsNotFoundError(
                f"ReccoBeats has no record for Spotify id {spotify_id!r}"
            )
        return items[0]["id"]

    def get_audio_features(self, reccobeats_id: str) -> dict:
        """Raw audio-features payload for a ReccoBeats track UUID."""
        logger.info("ReccoBeats audio-features: %s", reccobeats_id)
        resp = requests.get(
            f"{_API_BASE}/track/{reccobeats_id}/audio-features",
            timeout=self._timeout,
        )
        if resp.status_code == 404:
            raise ReccoBeatsNotFoundError(
                f"No audio features for ReccoBeats id {reccobeats_id!r}"
            )
        if resp.status_code != 200:
            raise ReccoBeatsError(
                f"ReccoBeats audio-features failed ({resp.status_code}): {resp.text}"
            )
        return resp.json()

    def find_features_by_spotify_id(
        self,
        spotify_id: str,
        title: str = "",
        artist: str = "",
    ) -> TrackFeatures:
        """
        Convenience: lookup → audio-features → flattened TrackFeatures.

        `title` and `artist` are pass-through metadata so the caller can use
        the same TrackFeatures shape it gets from SpotifyClient. ReccoBeats
        does return its own title/artist strings, but Spotify's are the
        canonical user-facing labels we already searched on.
        """
        rb_id = self.lookup_by_spotify_id(spotify_id)
        feats = self.get_audio_features(rb_id)

        # ReccoBeats doesn't always return every Spotify field — fall back to
        # 0/0.0 for missing entries rather than raising, so a partial payload
        # still produces a usable profile.
        def _f(name: str, default: float = 0.0) -> float:
            v = feats.get(name)
            return float(v) if v is not None else default

        def _i(name: str, default: int = 0) -> int:
            v = feats.get(name)
            return int(v) if v is not None else default

        return TrackFeatures(
            track_id=spotify_id,
            title=title,
            artist=artist,
            energy=_f("energy"),
            valence=_f("valence"),
            danceability=_f("danceability"),
            acousticness=_f("acousticness"),
            tempo=_f("tempo"),
            instrumentalness=_f("instrumentalness"),
            liveness=_f("liveness"),
            loudness=_f("loudness"),
            speechiness=_f("speechiness"),
            mode=_i("mode"),
            key=_i("key"),
        )
