"""
Data classes shared across the recommender pipeline.

`Song` is the catalog row shape; `UserProfile` is the taste profile that
gets scored against songs. Keeping these in their own module avoids a
circular import between scoring.py and recommender.py.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Song:
    """A single catalog song with its audio-feature vector.

    Field shape mirrors the Spotify / ReccoBeats audio-features schema so
    the same vector can come from either the local CSV or a live API call.
    `id` is a Spotify track id (alphanumeric string), not an integer.
    """
    id: str
    title: str
    artist: str
    genre: str
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float

    def dedup_key(self) -> tuple[str, str]:
        """A (title, artist) key for collapsing duplicates.

        Spotify assigns separate track IDs to the same recording across
        album editions (original album, deluxe, regional release), and the
        source CSV also has one row per (track, playlist) pair — so neither
        `id` alone nor (id, album) cleanly identifies a single song. Lower-
        cased (title, artist) does. Remixes ("Memories - Dillon Francis
        Remix") are intentionally NOT collapsed: their titles differ.
        """
        return (self.title.strip().lower(), self.artist.strip().lower())


@dataclass
class UserProfile:
    """
    A user's taste preferences.

    All fields are optional — leave a categorical field as the empty string
    or a numeric field as None to signal "no preference," and the scorer
    will skip it. This lets the same profile shape express both fully
    specified taste profiles (manual sliders) and partial ones (e.g., only
    a genre and a target energy).

    `mode` and `key` are integer-coded categorical fields (mode ∈ {0,1},
    key ∈ {0..11}) — None means "any."
    """
    genre: str = ""
    energy: Optional[float] = None
    tempo: Optional[float] = None
    valence: Optional[float] = None
    danceability: Optional[float] = None
    acousticness: Optional[float] = None
    instrumentalness: Optional[float] = None
    liveness: Optional[float] = None
    loudness: Optional[float] = None
    speechiness: Optional[float] = None
    mode: Optional[int] = None
    key: Optional[int] = None

    def is_empty(self) -> bool:
        """True when no field is set — used by the UI to decide whether to
        show recommendations or the "pick something first" prompt."""
        return (
            not self.genre
            and self.energy is None
            and self.tempo is None
            and self.valence is None
            and self.danceability is None
            and self.acousticness is None
            and self.instrumentalness is None
            and self.liveness is None
            and self.loudness is None
            and self.speechiness is None
            and self.mode is None
            and self.key is None
        )
