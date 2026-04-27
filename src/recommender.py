"""
Top-level recommender pipeline: load a song catalog, derive a taste profile
from liked songs, and rank the catalog against a UserProfile.

The actual scoring math lives in `scoring.py`; the data classes live in
`models.py`. This file ties them together.
"""
import csv
from collections import Counter
from typing import List, Tuple

from loggings import get_logger
from models import Song, UserProfile
from scoring import mmr_rerank, score_song

logger = get_logger(__name__)


def load_songs(csv_path: str) -> List[Song]:
    """
    Load songs from a CSV file into Song objects.

    A missing file raises FileNotFoundError (the caller can't recover
    without one). Individual malformed rows are logged as warnings and
    skipped, so one bad row can't break the whole catalog.
    """
    logger.info("Loading songs from %s", csv_path)

    try:
        f = open(csv_path, newline="", encoding="utf-8")
    except FileNotFoundError:
        logger.error("Songs CSV not found at %s", csv_path)
        raise

    songs: List[Song] = []
    skipped = 0

    with f:
        reader = csv.DictReader(f)
        # line_no starts at 2 because line 1 is the header.
        for line_no, row in enumerate(reader, start=2):
            try:
                song = Song(
                    id=int(row["id"]),
                    title=row["title"],
                    artist=row["artist"],
                    genre=row["genre"],
                    mood=row["mood"],
                    energy=float(row["energy"]),
                    tempo_bpm=float(row["tempo_bpm"]),
                    valence=float(row["valence"]),
                    danceability=float(row["danceability"]),
                    acousticness=float(row["acousticness"]),
                )
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(
                    "Skipping malformed row at line %d: %s (row=%r)",
                    line_no, e, row,
                )
                skipped += 1
                continue
            songs.append(song)

    logger.info("Loaded %d songs (%d skipped)", len(songs), skipped)
    return songs


def profile_from_songs(songs: List[Song]) -> UserProfile:
    """
    Derive a UserProfile by averaging audio features over a list of "liked"
    songs and taking the modal genre / mood. Classic content-based-filtering
    setup — the taste profile is the centroid of the user's liked songs.

    Returns a default (empty) UserProfile for an empty input.
    """
    if not songs:
        return UserProfile()

    n = len(songs)
    return UserProfile(
        genre=Counter(s.genre for s in songs).most_common(1)[0][0],
        mood=Counter(s.mood for s in songs).most_common(1)[0][0],
        energy=sum(s.energy for s in songs) / n,
        valence=sum(s.valence for s in songs) / n,
        danceability=sum(s.danceability for s in songs) / n,
        acousticness=sum(s.acousticness for s in songs) / n,
        tempo=sum(s.tempo_bpm for s in songs) / n,
    )


class Recommender:
    """
    Content-based song recommender.

    Wraps a catalog of Songs and exposes ranking + explanation against any
    UserProfile. The class itself is a thin orchestrator — it delegates
    scoring to `scoring.score_song` and diversity to `scoring.mmr_rerank`.
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def rank(
        self,
        user: UserProfile,
        k: int = 5,
        diversity: float = 0.7,
    ) -> List[Tuple[Song, float, str]]:
        """
        Top-k matches as (song, score, explanation) triples.

        `diversity` is the MMR λ — see `scoring.mmr_rerank`.
        """
        scored = [(song, *score_song(song, user)) for song in self.songs]
        scored.sort(key=lambda triple: triple[1], reverse=True)
        return mmr_rerank(scored, k, diversity)

    def recommend(
        self,
        user: UserProfile,
        k: int = 5,
        diversity: float = 0.7,
    ) -> List[Song]:
        """Top-k Song objects in ranked order (drops scores/explanations)."""
        return [song for song, _, _ in self.rank(user, k, diversity)]

    def explain(self, user: UserProfile, song: Song) -> str:
        """Human-readable reason a given song matches (or doesn't) the profile."""
        _, explanation = score_song(song, user)
        return explanation
