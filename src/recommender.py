import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

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
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

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
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why a song was recommended."""
        # TODO: Implement explanation logic
        return "Explanation placeholder"

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


def score_song(song: Dict, user_prefs: Dict) -> Tuple[float, str]:
    """
    Scores a single song against the user taste profile.

    Returns (score, explanation) where explanation names the top contributors.
    """
    score = 0.0
    reasons = []

    # --- Step 1: Categorical bonuses (binary, additive) ---
    if user_prefs.get("genre") and song.get("genre") == user_prefs["genre"]:
        score += WEIGHTS["genre"]
        reasons.append(f"genre match ({song['genre']}) (+{WEIGHTS['genre']})")

    if user_prefs.get("mood") and song.get("mood") == user_prefs["mood"]:
        score += WEIGHTS["mood"]
        reasons.append(f"mood match ({song['mood']}) (+{WEIGHTS['mood']})")

    # --- Step 2: Continuous similarity scoring ---
    # similarity = 1.0 - |song_value - target_value|, then multiply by weight.

    if "energy" in user_prefs:
        sim = 1.0 - abs(song["energy"] - user_prefs["energy"])
        contribution = WEIGHTS["energy"] * sim
        score += contribution
        if sim >= 0.85:
            reasons.append(f"energy close ({song['energy']}) (+{contribution:.2f})")

    if "acousticness" in user_prefs:
        sim = 1.0 - abs(song["acousticness"] - user_prefs["acousticness"])
        contribution = WEIGHTS["acousticness"] * sim
        score += contribution
        if sim >= 0.85:
            reasons.append(f"acousticness close ({song['acousticness']}) (+{contribution:.2f})")

    if "valence" in user_prefs:
        sim = 1.0 - abs(song["valence"] - user_prefs["valence"])
        contribution = WEIGHTS["valence"] * sim
        score += contribution
        if sim >= 0.85:
            reasons.append(f"valence close ({song['valence']}) (+{contribution:.2f})")

    if "tempo" in user_prefs:
        # Normalize both values to 0–1 using 200 BPM as the practical ceiling.
        sim = 1.0 - abs(song["tempo_bpm"] / 200.0 - user_prefs["tempo"] / 200.0)
        contribution = WEIGHTS["tempo"] * sim
        score += contribution
        if sim >= 0.90:
            reasons.append(f"tempo close ({song['tempo_bpm']} BPM) (+{contribution:.2f})")

    if "danceability" in user_prefs:
        sim = 1.0 - abs(song["danceability"] - user_prefs["danceability"])
        contribution = WEIGHTS["danceability"] * sim
        score += contribution
        if sim >= 0.85:
            reasons.append(f"danceability close ({song['danceability']}) (+{contribution:.2f})")

    # --- Step 4: Explanation generation ---
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
