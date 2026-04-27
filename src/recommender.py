import csv
from collections import Counter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from loggings import get_logger

logger = get_logger(__name__)

# Numeric columns required on every CSV row. Missing or unparseable values
# cause the row to be skipped (with a warning), not a hard crash.
_REQUIRED_NUMERIC_FIELDS = ("energy", "tempo_bpm", "valence", "danceability", "acousticness")

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

    def recommend(self, user: UserProfile, k: int = 5, diversity: float = 0.7) -> List[Song]:
        """Return the top-k song recommendations for the given user profile."""
        prefs = _user_profile_to_prefs(user)
        scored = [
            (song, score_song(asdict(song), prefs)[0], "")
            for song in self.songs
        ]
        scored.sort(key=lambda triple: triple[1], reverse=True)
        reranked = _mmr_rerank(scored, k, diversity)
        return [song for song, _, _ in reranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why a song was recommended."""
        prefs = _user_profile_to_prefs(user)
        _, explanation = score_song(asdict(song), prefs)
        return explanation

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.

    Logs progress via the project logger. A missing file raises
    FileNotFoundError (the caller can't recover without one). Individual
    malformed rows are logged as warnings and skipped, so one bad row
    can't break the whole catalog.
    """
    logger.info("Loading songs from %s", csv_path)

    try:
        f = open(csv_path, newline="", encoding="utf-8")
    except FileNotFoundError:
        logger.error("Songs CSV not found at %s", csv_path)
        raise

    songs: List[Dict] = []
    skipped = 0

    with f:
        reader = csv.DictReader(f)
        # line_no starts at 2 because line 1 is the header.
        for line_no, row in enumerate(reader, start=2):
            try:
                row["id"] = int(row["id"])
                for field in _REQUIRED_NUMERIC_FIELDS:
                    row[field] = float(row[field])
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(
                    "Skipping malformed row at line %d: %s (row=%r)",
                    line_no, e, row,
                )
                skipped += 1
                continue
            songs.append(row)

    logger.info("Loaded %d songs (%d skipped)", len(songs), skipped)
    return songs


# ---------------------------------------------------------------------------
# Scoring weights — tuned against the 20-song dataset in data/songs.csv.
#
# How they're used:
#   - genre and mood are categorical exact-match bonuses, added as flat amounts.
#   - The numeric weights act as importance multipliers inside a *weighted*
#     cosine similarity over the audio-feature vector. The cosine result
#     (in [0,1]) is then scaled by the sum of active numeric weights, so the
#     overall scoring scale matches the categorical bonuses and the
#     theoretical max stays at WEIGHTS["genre"] + WEIGHTS["mood"] +
#     sum(numeric_weights) ≈ 8.3.
#
# Per-feature rationale:
#   - genre (2.0) and mood (1.5) are the strongest categorical signals when
#     they fire, but sparse (14 genres, 15 moods across 20 songs).
#   - energy (1.5) has the widest useful spread (0.21–0.97) — the single
#     best numeric separator between genres.
#   - acousticness (1.2) covers nearly the full 0–1 range and cleanly
#     separates electric (rock/metal/electronic) from acoustic (folk/jazz/lofi).
#   - valence (0.8) captures emotional tone but overlaps with mood labels.
#   - tempo (0.8) is normalized by 200 BPM before scoring so its raw range
#     (58–180) doesn't dominate.
#   - danceability (0.5) is the weakest separator — largely correlated
#     with energy.
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

# Numeric features used in the weighted-cosine similarity vector.
# (pref_key, song_key, normalize, reason_threshold, unit_label)
#   - normalize: divisor applied to both values before computing similarity
#     (200 for tempo so its raw BPM range doesn't dominate the vector;
#     1.0 for already-0-to-1 features).
#   - reason_threshold: per-feature closeness (1 - |diff|/norm) at/above which
#     the feature is named in the human-readable explanation string. This
#     does NOT affect the score — it's purely for transparency.
#   - unit_label: appended to the song value in the explanation (e.g. " BPM").
_NUMERIC_FEATURES: List[Tuple[str, str, float, float, str]] = [
    ("energy",       "energy",       1.0,   0.85, ""),
    ("acousticness", "acousticness", 1.0,   0.85, ""),
    ("valence",      "valence",      1.0,   0.85, ""),
    ("tempo",        "tempo_bpm",    200.0, 0.90, " BPM"),
    ("danceability", "danceability", 1.0,   0.85, ""),
]


def _song_pair_cosine(a: Dict, b: Dict) -> float:
    """
    Weighted cosine similarity between two SONGS — used by MMR to detect
    near-duplicates in the candidate pool.

    Cosine is fine here (not for user-vs-song scoring, see score_song's note)
    because we're comparing two real catalog vectors against each other:
    songs that point the same direction in feature space genuinely have the
    same sonic *profile*, which is exactly what "near-duplicate" should mean.
    """
    dot = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for pref_key, song_key, norm, _, _ in _NUMERIC_FEATURES:
        ai = a[song_key] / norm
        bi = b[song_key] / norm
        w = WEIGHTS[pref_key]
        dot += w * ai * bi
        a_sq += w * ai * ai
        b_sq += w * bi * bi
    denom = (a_sq ** 0.5) * (b_sq ** 0.5)
    return dot / denom if denom > 0 else 0.0


def score_song(song: Dict, user_prefs: Dict) -> Tuple[float, str]:
    """
    Scores a single song against the user taste profile.

    Score = categorical bonuses (genre, mood — exact match)
          + per-feature weighted similarity, where similarity_i = 1 - |u_i - s_i|.

    Why not cosine for user-vs-song? Cosine measures *direction* in
    feature space and ignores magnitude — so for non-negative features in
    [0,1], a user vector of (1,1,1,1) looks ~95–100% similar to almost any
    positive song vector, which is exactly what we don't want for
    interpretable audio features. Per-feature absolute-difference respects
    magnitude (0.5 is genuinely "halfway" to 1.0), so it's the right tool
    here. Cosine is still used inside MMR for song-vs-song duplicate
    detection — see _song_pair_cosine.
    """
    score = 0.0
    reasons: List[str] = []

    # Categorical bonuses (binary, additive).
    for pref_key, song_key in _CATEGORICAL_FEATURES:
        if user_prefs.get(pref_key) and song.get(song_key) == user_prefs[pref_key]:
            weight = WEIGHTS[pref_key]
            score += weight
            reasons.append(f"{pref_key} match ({song[song_key]}) (+{weight})")

    # Per-feature weighted similarity contributions.
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


def _mmr_rerank(
    scored: List[Tuple[Dict, float, str]],
    k: int,
    diversity: float,
) -> List[Tuple[Dict, float, str]]:
    """
    Re-rank scored candidates with Maximal Marginal Relevance.

    At each step, picks the candidate that maximizes:
        diversity * normalized_relevance - (1 - diversity) * max_sim_to_picked

    diversity = 1.0 → pure relevance (no MMR effect; behaves like a top-k cut).
    diversity = 0.7 (default) → relevance-heavy; mild penalty for near-duplicates.
    diversity = 0.0 → ignore relevance, pick the most novel each step.

    `scored` is expected to be sorted by relevance descending.
    """
    lam = max(0.0, min(1.0, diversity))
    if lam >= 1.0 or k >= len(scored):
        return scored[:k]

    # Normalize relevance to [0,1] so it's on the same scale as cosine sim.
    max_rel = max((rel for _, rel, _ in scored), default=0.0) or 1.0

    remaining = list(scored)
    selected: List[Tuple[Dict, float, str]] = []
    while len(selected) < k and remaining:
        best_idx = 0
        best_mmr = -float("inf")
        for i, (song, rel, _) in enumerate(remaining):
            normalized_rel = rel / max_rel
            if selected:
                max_sim = max(_song_pair_cosine(song, s) for s, _, _ in selected)
            else:
                max_sim = 0.0
            mmr = lam * normalized_rel - (1.0 - lam) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))
    return selected


def profile_from_songs(songs: List[Dict]) -> Dict:
    """
    Derive a user_prefs dict by averaging audio features over a list of
    "liked" songs and taking the modal genre / mood. This is the classic
    content-based-filtering setup: the taste profile is a centroid of the
    songs the user has already favorited.

    Returns an empty dict for an empty input. The result is shaped to drop
    straight into score_song / recommend_songs.
    """
    if not songs:
        return {}

    n = len(songs)
    prefs: Dict = {
        "energy":       sum(s["energy"]       for s in songs) / n,
        "valence":      sum(s["valence"]      for s in songs) / n,
        "danceability": sum(s["danceability"] for s in songs) / n,
        "acousticness": sum(s["acousticness"] for s in songs) / n,
        "tempo":        sum(s["tempo_bpm"]    for s in songs) / n,
    }
    prefs["genre"] = Counter(s["genre"] for s in songs).most_common(1)[0][0]
    prefs["mood"]  = Counter(s["mood"]  for s in songs).most_common(1)[0][0]
    return prefs


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    diversity: float = 0.7,
) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.

    `diversity` is the MMR λ — see _mmr_rerank.
    """
    scored = [(song, *score_song(song, user_prefs)) for song in songs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return _mmr_rerank(scored, k, diversity)
