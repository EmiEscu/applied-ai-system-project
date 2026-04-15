"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import os
from recommender import load_songs, recommend_songs


def print_rankings(label: str, user_prefs: dict, songs: list, top_k: int = 5) -> None:
    """Print the top-k ranked songs for a given user profile."""
    recommendations = recommend_songs(user_prefs, songs, k=top_k)
    print(f"\n{'='*60}")
    print(f"  PROFILE: {label}")
    print(f"  Prefs:   {user_prefs}")
    print(f"{'='*60}\n")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"  #{rank}  {song['title']} by {song['artist']}")
        print(f"      Genre: {song['genre']}  |  Mood: {song['mood']}")
        print(f"      Score: {score:.3f} / 8.3")
        print(f"      Why:   {explanation}")
        print()


def main() -> None:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(csv_path)

    # =================================================================
    # SECTION 1 — Three distinct user preference profiles
    # =================================================================

    high_energy_pop = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.90,
        "tempo": 130.0,
        "valence": 0.85,
        "danceability": 0.85,
        "acousticness": 0.10,
    }

    chill_lofi = {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.25,
        "tempo": 75.0,
        "valence": 0.55,
        "danceability": 0.55,
        "acousticness": 0.80,
    }

    deep_intense_rock = {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "tempo": 155.0,
        "valence": 0.30,
        "danceability": 0.60,
        "acousticness": 0.08,
    }

    print("\n" + "#"*60)
    print("#  STANDARD USER PROFILES")
    print("#"*60)

    print_rankings("High-Energy Pop", high_energy_pop, songs)
    print_rankings("Chill Lofi", chill_lofi, songs)
    print_rankings("Deep Intense Rock", deep_intense_rock, songs)

    # =================================================================
    # SECTION 2 — System Evaluation: Adversarial & Edge-Case Profiles
    #
    # These profiles are designed to stress-test the scoring logic by
    # creating internal contradictions or boundary conditions that might
    # produce unexpected rankings.
    # =================================================================

    print("\n" + "#"*60)
    print("#  SYSTEM EVALUATION — ADVERSARIAL / EDGE-CASE PROFILES")
    print("#"*60)

    # EDGE CASE 1: Contradictory energy vs. mood
    # High energy (0.95) but "chill" mood — these rarely coexist in real
    # songs. Tests whether categorical mood bonus can pull a low-energy
    # song above a high-energy song that lacks the mood match.
    contradictory_energy_mood = {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.95,
        "tempo": 140.0,
        "valence": 0.50,
        "danceability": 0.80,
        "acousticness": 0.70,
    }
    print_rankings(
        "EDGE 1: Contradictory (energy=0.95 + mood=chill + genre=lofi)",
        contradictory_energy_mood, songs,
    )

    # EDGE CASE 2: Contradictory valence vs. mood
    # Very high valence (bright/happy) but "angry" mood. In the dataset,
    # angry songs have low valence (~0.19). The mood bonus rewards anger
    # while the valence score penalizes it — do they cancel out?
    happy_but_angry = {
        "genre": "metal",
        "mood": "angry",
        "energy": 0.90,
        "tempo": 170.0,
        "valence": 0.95,       # maximally happy valence...
        "danceability": 0.50,
        "acousticness": 0.05,
    }
    print_rankings(
        "EDGE 2: Happy-Angry (mood=angry + valence=0.95)",
        happy_but_angry, songs,
    )

    # EDGE CASE 3: "All zeros" — the minimalist listener
    # Every numeric target at the floor. Only one song in the catalog
    # has values near zero across the board (Requiem for the Lost).
    # Tests whether the system degenerates or still differentiates.
    all_zeros = {
        "genre": "",           # no genre preference
        "mood": "",            # no mood preference
        "energy": 0.0,
        "tempo": 0.0,
        "valence": 0.0,
        "danceability": 0.0,
        "acousticness": 0.0,
    }
    print_rankings(
        "EDGE 3: All-Zeros (every numeric feature = 0.0, no genre/mood)",
        all_zeros, songs,
    )

    # EDGE CASE 4: "All maxed out" — the everything-lover
    # Every numeric target at 1.0 / 200 BPM. No real song matches all
    # features at once (e.g., acousticness 1.0 contradicts energy 1.0).
    # Tests whether the system spreads scores evenly or has a clear winner.
    all_maxed = {
        "genre": "",
        "mood": "",
        "energy": 1.0,
        "tempo": 200.0,
        "valence": 1.0,
        "danceability": 1.0,
        "acousticness": 1.0,
    }
    print_rankings(
        "EDGE 4: All-Maxed (every numeric feature at maximum)",
        all_maxed, songs,
    )

    # EDGE CASE 5: Genre/mood that doesn't exist in the catalog
    # The categorical bonuses will never fire, so the profile relies
    # entirely on numeric similarity. Tests the numeric-only fallback.
    nonexistent_genre = {
        "genre": "k-pop",
        "mood": "whimsical",
        "energy": 0.70,
        "tempo": 115.0,
        "valence": 0.75,
        "danceability": 0.80,
        "acousticness": 0.30,
    }
    print_rankings(
        "EDGE 5: Non-Existent Genre (k-pop/whimsical — no catalog match)",
        nonexistent_genre, songs,
    )


if __name__ == "__main__":
    main()
