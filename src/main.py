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


def main() -> None:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(csv_path)

    # Taste profile: target values for each audio feature.
    # Numeric features (0.0–1.0 unless noted) represent the "ideal" song attributes.
    # genre and mood are categorical — used for hard or soft matching.
    user_prefs = {
        "genre": "lofi",          # categorical: preferred genre label
        "mood": "chill",          # categorical: preferred mood label
        "energy": 0.25,           # 0.0 (passive) → 1.0 (intense)
        "tempo": 80.0,            # beats per minute
        "valence": 0.50,          # 0.0 (dark/sad) → 1.0 (bright/happy)
        "danceability": 0.55,     # 0.0 (not danceable) → 1.0 (very danceable)
        "acousticness": 0.70,     # 0.0 (electronic) → 1.0 (fully acoustic)
    }

    recommendations = recommend_songs(user_prefs, songs, k=len(songs))

    print("\nALL SONGS RANKED BY SCORE\n")

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"  #{rank}  {song['title']} by {song['artist']}")
        print(f"      Genre: {song['genre']}  |  Mood: {song['mood']}")
        print(f"      Score: {score:.3f} / 8.3")
        print(f"      Why:   {explanation}")
        print()


if __name__ == "__main__":
    main()
