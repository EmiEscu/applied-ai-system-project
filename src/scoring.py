"""
Scoring + diversity logic for the recommender.

Three pieces live here:
  - `score_song`:       user-vs-song relevance (categorical bonuses + per-feature
                        weighted similarity).
  - `max_possible_score`: theoretical ceiling for a given UserProfile, used by
                        the UI to render a 0–100% "match" bar.
  - `mmr_rerank`:       Maximal Marginal Relevance re-rank that uses
                        `_song_pair_cosine` to penalize near-duplicate picks.
"""
from typing import List, Tuple

from models import Song, UserProfile


# ---------------------------------------------------------------------------
# Scoring weights — tuned for the Spotify audio-features schema.
#
# How they're used:
#   - Categorical features (genre, mode, key) are exact-match bonuses, added
#     as flat amounts when both the song and profile values agree.
#   - Numeric weights are per-feature multipliers on a similarity term in
#     [0,1]. Theoretical max ≈ sum of all weights when every field is set
#     and every feature matches perfectly.
#
# Per-feature rationale:
#   - genre (2.0) is the strongest categorical signal.
#   - energy (1.5) is the widest useful numeric separator across genres.
#   - acousticness (1.2) cleanly separates electric from acoustic music.
#   - valence (0.8) captures emotional positivity.
#   - tempo (0.8) is normalized by 140 BPM (60–200 slider range).
#   - instrumentalness (0.8) splits vocal pop/rap from instrumental tracks.
#   - loudness (0.6) — production-loudness signal, normalized by 60 dB.
#   - danceability (0.5) is largely correlated with energy.
#   - speechiness (0.4) flags rap / spoken word.
#   - liveness (0.4) flags live recordings.
#   - mode (0.4) major vs minor — categorical exact match.
#   - key (0.3) musical key — categorical exact match.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "genre":            2.0,
    "energy":           1.5,
    "acousticness":     1.2,
    "valence":          0.8,
    "tempo":            0.8,
    "instrumentalness": 0.8,
    "loudness":         0.6,
    "danceability":     0.5,
    "speechiness":      0.4,
    "liveness":         0.4,
    "mode":             0.4,
    "key":              0.3,
}

# Categorical features: exact-value match → flat bonus.
# (profile_attr, song_attr)
_CATEGORICAL_FEATURES: List[Tuple[str, str]] = [
    ("genre", "genre"),
    ("mode",  "mode"),
    ("key",   "key"),
]

# Numeric features used in per-feature weighted similarity.
# (profile_attr, song_attr, normalize, reason_threshold, unit_label)
#   - normalize: divisor applied to |u - s| before computing similarity, so
#     every feature's similarity lands in [0,1]. For 0–1 features this is 1.0.
#     - tempo: slider spans 60–200 BPM (a 140-BPM range), normalize by 140 so
#       a 70-BPM gap reads as sim=0.5.
#     - loudness: typical Spotify range is roughly -60..0 dB, normalize by 60
#       so a 30-dB gap reads as sim=0.5.
#   - reason_threshold: legacy field, kept for compatibility.
#   - unit_label: legacy field, per-feature formatting now lives in
#     `_format_numeric`.
_NUMERIC_FEATURES: List[Tuple[str, str, float, float, str]] = [
    ("energy",           "energy",           1.0,   0.85, ""),
    ("acousticness",     "acousticness",     1.0,   0.85, ""),
    ("valence",          "valence",          1.0,   0.85, ""),
    ("tempo",            "tempo",            140.0, 0.90, " BPM"),
    ("instrumentalness", "instrumentalness", 1.0,   0.85, ""),
    ("loudness",         "loudness",         60.0,  0.85, " dB"),
    ("danceability",     "danceability",     1.0,   0.85, ""),
    ("speechiness",      "speechiness",      1.0,   0.85, ""),
    ("liveness",         "liveness",         1.0,   0.85, ""),
]


_STRONG_SIM = 0.85    # numeric features at/above this are "strongly aligned"
_CLOSE_SIM = 0.60     # 0.60–0.85 is "close"; below 0.60 is "differs"


_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _format_numeric(u_key: str, s_val: float, u_val: float) -> str:
    """Human-readable 'song value vs your value' string for one numeric feature."""
    if u_key == "tempo":
        return f"tempo ({int(s_val)} BPM vs your {int(u_val)} BPM)"
    if u_key == "loudness":
        return f"loudness ({s_val:.1f} dB vs your {u_val:.1f} dB)"
    return f"{u_key} ({s_val:.2f} vs your {u_val:.2f})"


def _format_categorical(u_key: str, val) -> str:
    """Human-readable label for a categorical feature value."""
    if u_key == "mode":
        return "major" if val == 1 else "minor"
    if u_key == "key" and isinstance(val, int) and 0 <= val < 12:
        return _KEY_NAMES[val]
    return str(val)


def score_song(song: Song, user: UserProfile) -> Tuple[float, str]:
    """
    Score one song against a user's taste profile.

    Score = categorical bonuses (genre, mode, key — exact match)
          + per-feature weighted similarity, where similarity_i = 1 - |u_i - s_i|/norm.

    The explanation groups features into three tiers so the user can see
    *why* a song was picked and which preferences didn't quite line up:
      - "Strongly aligned": exact categorical match, or numeric sim ≥ 0.85
      - "Close":             numeric sim in [0.60, 0.85)
      - "Differs on":        categorical miss, or numeric sim < 0.60

    Why not cosine for user-vs-song? Cosine measures *direction* in feature
    space and ignores magnitude — a user vector of (1,1,1,1) looks ~95–100%
    similar to almost any positive song vector. Per-feature absolute
    difference respects magnitude (0.5 is genuinely "halfway" to 1.0). Cosine
    *is* used inside MMR for song-vs-song duplicate detection — see
    `_song_pair_cosine`.
    """
    score = 0.0
    strong: List[str] = []
    close: List[str] = []
    differs: List[str] = []

    for u_key, s_key in _CATEGORICAL_FEATURES:
        u_val = getattr(user, u_key)
        # Treat "" and None as "not set." 0 is a valid mode/key value.
        if u_val is None or u_val == "":
            continue
        s_val = getattr(song, s_key)
        if s_val == u_val:
            score += WEIGHTS[u_key]
            strong.append(f"{u_key} match ({_format_categorical(u_key, u_val)})")
        else:
            differs.append(
                f"{u_key} ({_format_categorical(u_key, s_val)}, "
                f"you wanted {_format_categorical(u_key, u_val)})"
            )

    for u_key, s_key, norm, _, _ in _NUMERIC_FEATURES:
        u_val = getattr(user, u_key)
        if u_val is None:
            continue
        s_val = getattr(song, s_key)
        sim = max(0.0, 1.0 - abs(s_val - u_val) / norm)
        score += WEIGHTS[u_key] * sim

        label = _format_numeric(u_key, s_val, u_val)
        if sim >= _STRONG_SIM:
            strong.append(label)
        elif sim >= _CLOSE_SIM:
            close.append(label)
        else:
            differs.append(label)

    parts: List[str] = []
    if strong:
        parts.append("Strongly aligned on " + ", ".join(strong))
    if close:
        parts.append("close on " + ", ".join(close))
    if differs:
        parts.append("differs on " + ", ".join(differs))

    explanation = "; ".join(parts) + "." if parts else "Partial similarity to your profile."
    return round(score, 3), explanation


def max_possible_score(user: UserProfile) -> float:
    """
    Theoretical max score given which UserProfile fields are set.

    Used by the UI to scale a raw score into a 0–100% match bar. Returns
    1.0 (not 0) for a fully unset profile so callers can divide safely.
    """
    total = 0.0
    if user.genre:
        total += WEIGHTS["genre"]
    if user.mode is not None:
        total += WEIGHTS["mode"]
    if user.key is not None:
        total += WEIGHTS["key"]
    for u_key, _, _, _, _ in _NUMERIC_FEATURES:
        if getattr(user, u_key) is not None:
            total += WEIGHTS[u_key]
    return total or 1.0


def _song_pair_cosine(a: Song, b: Song) -> float:
    """
    Weighted cosine similarity between two SONGS — used by MMR to detect
    near-duplicates in the candidate pool.

    Cosine is appropriate here (unlike user-vs-song scoring) because we're
    comparing two real catalog vectors: songs that point the same direction
    in feature space genuinely share the same sonic *profile*.
    """
    dot = a_sq = b_sq = 0.0
    for u_key, s_key, norm, _, _ in _NUMERIC_FEATURES:
        ai = getattr(a, s_key) / norm
        bi = getattr(b, s_key) / norm
        w = WEIGHTS[u_key]
        dot += w * ai * bi
        a_sq += w * ai * ai
        b_sq += w * bi * bi
    denom = (a_sq ** 0.5) * (b_sq ** 0.5)
    return dot / denom if denom > 0 else 0.0


def mmr_rerank(
    scored: List[Tuple[Song, float, str]],
    k: int,
    diversity: float,
) -> List[Tuple[Song, float, str]]:
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
    selected: List[Tuple[Song, float, str]] = []
    while len(selected) < k and remaining:
        best_idx = 0
        best_mmr = -float("inf")
        for i, (song, rel, _) in enumerate(remaining):
            normalized_rel = rel / max_rel
            max_sim = max(
                (_song_pair_cosine(song, s) for s, _, _ in selected),
                default=0.0,
            )
            mmr = lam * normalized_rel - (1.0 - lam) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))
    return selected
