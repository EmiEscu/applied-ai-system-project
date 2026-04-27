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
    """A single catalog song with its audio-feature vector."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    A user's taste preferences.

    All fields are optional — leave a categorical field as the empty string
    or a numeric field as None to signal "no preference," and the scorer
    will skip it. This lets the same profile shape express both fully
    specified taste profiles (manual sliders) and partial ones (e.g., only
    a genre and a target energy).
    """
    genre: str = ""
    mood: str = ""
    energy: Optional[float] = None
    tempo: Optional[float] = None
    valence: Optional[float] = None
    danceability: Optional[float] = None
    acousticness: Optional[float] = None

    def is_empty(self) -> bool:
        """True when no field is set — used by the UI to decide whether to
        show recommendations or the "pick something first" prompt."""
        return (
            not self.genre
            and not self.mood
            and self.energy is None
            and self.tempo is None
            and self.valence is None
            and self.danceability is None
            and self.acousticness is None
        )
