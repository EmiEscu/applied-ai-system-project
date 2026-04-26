"""
Streamlit UI for the Music Recommender.

Run with:
    streamlit run src/app.py

The sidebar collects the user's taste profile (genre, mood, audio-feature
sliders). The main panel shows the top-k matches with a percentage match,
the genre/mood badges, and the same "why this matched" explanation that
the CLI version produces.
"""
import os
import sys
from pathlib import Path

# Make `from recommender import …` work regardless of where Streamlit is
# launched from (its cwd defaults to the user's terminal, not src/).
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from loggings import setup_logging
from recommender import WEIGHTS, load_songs, recommend_songs

setup_logging()

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


@st.cache_data
def get_songs():
    """Load the catalog once per Streamlit session."""
    return load_songs(DATA_PATH)


def max_possible_score(prefs: dict) -> float:
    """
    Theoretical maximum score given which preferences the user actually set.

    Used to convert raw scores into a 0–100% match. Categorical features
    only count when the user picked a non-empty value; numeric features
    always count (the UI always provides a slider value).
    """
    total = 0.0
    if prefs.get("genre"):
        total += WEIGHTS["genre"]
    if prefs.get("mood"):
        total += WEIGHTS["mood"]
    for key in ("energy", "tempo", "valence", "danceability", "acousticness"):
        if key in prefs:
            total += WEIGHTS[key]
    return total or 1.0  # avoid div-by-zero if literally nothing is set


def render_sidebar(genres: list, moods: list) -> tuple[dict, int]:
    """Render the input sidebar and return (user_prefs, k)."""
    st.sidebar.header("Your taste profile")

    genre = st.sidebar.selectbox(
        "Favorite genre",
        options=[""] + genres,
        format_func=lambda x: x if x else "Any",
    )
    mood = st.sidebar.selectbox(
        "Favorite mood",
        options=[""] + moods,
        format_func=lambda x: x if x else "Any",
    )

    st.sidebar.subheader("Audio features")
    energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5, 0.05)
    valence = st.sidebar.slider("Valence (positivity)", 0.0, 1.0, 0.5, 0.05)
    danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5, 0.05)
    acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.5, 0.05)
    tempo = st.sidebar.slider("Tempo (BPM)", 60, 200, 120, 1)

    st.sidebar.subheader("Output")
    k = st.sidebar.slider("Number of recommendations", 1, 10, 5)

    user_prefs = {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "acousticness": acousticness,
        "tempo": float(tempo),
    }
    return user_prefs, k


def render_results(results, max_score: float) -> None:
    """Render the top-k recommendation cards."""
    for rank, (song, score, explanation) in enumerate(results, start=1):
        pct = max(0, min(100, int(round(100 * score / max_score))))
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"**#{rank} — {song['title']}**  \n*by {song['artist']}*")
                st.caption(
                    f"🎼 {song['genre']}  ·  💭 {song['mood']}  ·  "
                    f"⚡ energy {song['energy']:.2f}  ·  🎵 {int(song['tempo_bpm'])} BPM"
                )
                st.write(explanation)
            with right:
                st.metric("Match", f"{pct}%")
                st.progress(pct / 100)


def main() -> None:
    st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")
    st.title("🎵 Music Recommender")
    st.caption("Content-based filtering — pick a vibe, get songs.")

    songs = get_songs()
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})

    user_prefs, k = render_sidebar(genres, moods)
    results = recommend_songs(user_prefs, songs, k=k)
    max_score = max_possible_score(user_prefs)

    st.subheader(f"Top {k} matches for your profile")
    render_results(results, max_score)


if __name__ == "__main__":
    main()
