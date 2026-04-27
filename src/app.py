"""
Streamlit UI for the Music Recommender.

Run with:
    streamlit run src/app.py

Two ways to build a taste profile:
  - Manual sliders: pick a genre/mood and dial the audio-feature sliders.
  - Liked songs:    pick a few songs you already like; the profile is the
                    average of their feature vectors (true content-based UX).

The main panel shows the top-k matches with percentage scores and the
"why this matched" explanation.
"""
import os
import sys
from pathlib import Path

# Make `from recommender import …` work regardless of where Streamlit is
# launched from (its cwd defaults to the user's terminal, not src/).
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from loggings import setup_logging
from recommender import WEIGHTS, load_songs, profile_from_songs, recommend_songs

setup_logging()

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


@st.cache_data
def get_songs():
    """Load the catalog once per Streamlit session."""
    return load_songs(DATA_PATH)


def max_possible_score(prefs: dict) -> float:
    """Theoretical maximum score given which preferences are set, for % scaling."""
    total = 0.0
    if prefs.get("genre"):
        total += WEIGHTS["genre"]
    if prefs.get("mood"):
        total += WEIGHTS["mood"]
    for key in ("energy", "tempo", "valence", "danceability", "acousticness"):
        if key in prefs:
            total += WEIGHTS[key]
    return total or 1.0


def _song_label(song: dict) -> str:
    return f"{song['title']} — {song['artist']} ({song['genre']})"


def render_output_controls() -> tuple[int, float]:
    """Sliders shared by both profile-building modes."""
    st.sidebar.subheader("Output")
    k = st.sidebar.slider("Number of recommendations", 1, 10, 5)
    diversity = st.sidebar.slider(
        "Diversity (1.0 = pure match · 0.0 = pure novelty)",
        0.0, 1.0, 0.7, 0.05,
        help=(
            "Maximal Marginal Relevance λ. Lower values penalize picks that are "
            "sonically too close to ones already chosen, breaking up filter bubbles."
        ),
    )
    return k, diversity


def render_manual_sidebar(genres: list, moods: list) -> dict:
    """Manual mode: dropdowns + sliders → user_prefs dict."""
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

    return {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "acousticness": acousticness,
        "tempo": float(tempo),
    }


def render_liked_sidebar(songs: list) -> tuple[dict, list[int]]:
    """Liked-songs mode: multiselect → averaged profile + selected ids."""
    st.sidebar.header("Songs you like")
    st.sidebar.caption(
        "Pick a handful of tracks you already love. The profile is built from "
        "the average of their audio features — Spotify-style content-based filtering."
    )

    songs_by_id = {s["id"]: s for s in songs}
    liked_ids = st.sidebar.multiselect(
        "Liked songs",
        options=list(songs_by_id.keys()),
        format_func=lambda sid: _song_label(songs_by_id[sid]),
    )
    liked = [songs_by_id[sid] for sid in liked_ids]
    prefs = profile_from_songs(liked)

    if liked:
        with st.sidebar.expander("Derived profile", expanded=False):
            st.write(
                {k: (round(v, 2) if isinstance(v, float) else v) for k, v in prefs.items()}
            )
    else:
        st.sidebar.info("Pick at least one song to generate recommendations.")

    return prefs, liked_ids


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

    mode = st.sidebar.radio(
        "Build profile from",
        options=["Manual sliders", "Liked songs"],
        horizontal=True,
    )
    st.sidebar.divider()

    liked_ids: list[int] = []
    if mode == "Manual sliders":
        prefs = render_manual_sidebar(genres, moods)
    else:
        prefs, liked_ids = render_liked_sidebar(songs)

    st.sidebar.divider()
    k, diversity = render_output_controls()

    if not prefs:
        st.info("👈 Pick at least one song in the sidebar to see recommendations.")
        return

    candidates = [s for s in songs if s["id"] not in liked_ids]
    results = recommend_songs(prefs, candidates, k=k, diversity=diversity)
    max_score = max_possible_score(prefs)

    st.subheader(f"Top {k} matches for your profile")
    render_results(results, max_score)


if __name__ == "__main__":
    main()
