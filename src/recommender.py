import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
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
    Represents a user's taste preferences.

    Field names mirror the dict-schema used by score_song / main.py so the two
    APIs stay in sync. All fields are optional — leave a field unset (empty
    string for categorical, None for numeric) to signal "no preference," and
    score_song will skip it.
    """
    genre: str = ""
    mood: str = ""
    energy: Optional[float] = None
    tempo: Optional[float] = None
    valence: Optional[float] = None
    danceability: Optional[float] = None
    acousticness: Optional[float] = None


_NUMERIC_PROFILE_KEYS = ("energy", "tempo", "valence", "danceability", "acousticness")


def _user_profile_to_prefs(user: "UserProfile") -> Dict:
    """Convert a UserProfile into the dict shape expected by score_song.

    Drops unset fields so score_song treats them as "no preference."
    """
    prefs: Dict = {}
    if user.genre:
        prefs["genre"] = user.genre
    if user.mood:
        prefs["mood"] = user.mood
    for key in _NUMERIC_PROFILE_KEYS:
        value = getattr(user, key)
        if value is not None:
            prefs[key] = value
    return prefs


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Initialize the recommender with a catalog of songs."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k song recommendations for the given user profile."""
        prefs = _user_profile_to_prefs(user)
        scored = [(song, score_song(asdict(song), prefs)[0]) for song in self.songs]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why a song was recommended."""
        prefs = _user_profile_to_prefs(user)
        _, explanation = score_song(asdict(song), prefs)
        return explanation

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    print(f"Loading songs from {csv_path}...")
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)
    return songs


# ---------------------------------------------------------------------------
# Scoring weights — tuned against the 20-song dataset in data/songs.csv.
#
# Design rationale:
#   - genre (2.0) and mood (1.5) are categorical exact-match bonuses.
#     They're the strongest signals when they fire, but sparse (14 genres,
#     15 moods across 20 songs), so numeric features must carry the load
#     when there's no categorical match.
#   - energy (1.5) has the widest useful spread (0.21–0.97) and is the
#     single best numeric separator between genres.
#   - acousticness (1.2) covers nearly the full 0–1 range and cleanly
#     separates electric (rock/metal/electronic) from acoustic (folk/jazz/lofi).
#   - valence (0.8) captures emotional tone but overlaps with mood labels,
#     so it gets a moderate weight.
#   - tempo (0.8) is normalized to 0–1 (÷200) before scoring; without
#     normalization its raw BPM values (58–180) would dominate.
#   - danceability (0.5) is the weakest separator — largely correlated
#     with energy — so it gets the lowest weight.
#
# Max possible score ≈ 8.3 (all features perfectly matched).
# ---------------------------------------------------------------------------
WEIGHTS = {
    "genre":        2.0,
    "mood":         1.5,
    "energy":       1.5,
    "acousticness": 1.2,
    "valence":      0.8,
    "tempo":        0.8,
    "danceability": 0.5,
}

# Categorical features: exact-string match → flat bonus.
# (pref_key, song_key)
_CATEGORICAL_FEATURES: List[Tuple[str, str]] = [
    ("genre", "genre"),
    ("mood",  "mood"),
]

# Numeric features: similarity = 1 - |song_val/norm - pref_val/norm|.
# (pref_key, song_key, normalize, reason_threshold, unit_label)
#   - normalize: divisor applied to both values before diffing (200 for tempo
#     so its raw BPM doesn't dominate; 1.0 for already-0-to-1 features).
#   - reason_threshold: similarity at/above which the feature is named in the
#     explanation string.
#   - unit_label: appended to the song value in the explanation (e.g. " BPM").
_NUMERIC_FEATURES: List[Tuple[str, str, float, float, str]] = [
    ("energy",       "energy",       1.0,   0.85, ""),
    ("acousticness", "acousticness", 1.0,   0.85, ""),
    ("valence",      "valence",      1.0,   0.85, ""),
    ("tempo",        "tempo_bpm",    200.0, 0.90, " BPM"),
    ("danceability", "danceability", 1.0,   0.85, ""),
]


def score_song(song: Dict, user_prefs: Dict) -> Tuple[float, str]:
    """
    Scores a single song against the user taste profile.

    Returns (score, explanation) where explanation names the top contributors.
    """
    score = 0.0
    reasons: List[str] = []

    # Categorical bonuses (binary, additive).
    for pref_key, song_key in _CATEGORICAL_FEATURES:
        if user_prefs.get(pref_key) and song.get(song_key) == user_prefs[pref_key]:
            weight = WEIGHTS[pref_key]
            score += weight
            reasons.append(f"{pref_key} match ({song[song_key]}) (+{weight})")

    # Numeric similarity contributions.
    for pref_key, song_key, norm, threshold, unit in _NUMERIC_FEATURES:
        if pref_key not in user_prefs:
            continue
        sim = 1.0 - abs(song[song_key] / norm - user_prefs[pref_key] / norm)
        contribution = WEIGHTS[pref_key] * sim
        score += contribution
        if sim >= threshold:
            reasons.append(
                f"{pref_key} close ({song[song_key]}{unit}) (+{contribution:.2f})"
            )

    explanation = (
        "Matches your taste: " + ", ".join(reasons)
        if reasons
        else "Partial numeric similarity to your profile"
    )
    return round(score, 3), explanation


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = [(song, *score_song(song, user_prefs)) for song in songs]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
