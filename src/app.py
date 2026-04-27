"""
Streamlit UI for the Music Recommender.

Run with:
    streamlit run src/app.py

Two ways to build a taste profile:
  - Manual sliders: pick a genre/mood and dial the audio-feature sliders.
  - Liked songs:    pick a few songs you already like; the profile is the
                    average of their feature vectors (true content-based UX).
"""
import os
import sys
from pathlib import Path

# Make `from recommender import …` work regardless of where Streamlit is
# launched from (its cwd defaults to the user's terminal, not src/).
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from loggings import setup_logging
from models import Song, UserProfile
from recommender import Recommender, load_songs, profile_from_songs
from scoring import max_possible_score

setup_logging()

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


@st.cache_data
def get_songs() -> list[Song]:
    """Load the catalog once per Streamlit session."""
    return load_songs(DATA_PATH)


def _song_label(song: Song) -> str:
    return f"{song.title} — {song.artist} ({song.genre})"


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


def render_manual_sidebar(genres: list[str], moods: list[str]) -> UserProfile:
    """Manual mode: dropdowns + sliders → UserProfile."""
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

    return UserProfile(
        genre=genre,
        mood=mood,
        energy=energy,
        valence=valence,
        danceability=danceability,
        acousticness=acousticness,
        tempo=float(tempo),
    )


def render_liked_sidebar(songs: list[Song]) -> tuple[UserProfile, list[int]]:
    """Liked-songs mode: multiselect → averaged profile + selected ids."""
    st.sidebar.header("Songs you like")
    st.sidebar.caption(
        "Pick a handful of tracks you already love. The profile is built from "
        "the average of their audio features — Spotify-style content-based filtering."
    )

    songs_by_id = {s.id: s for s in songs}
    liked_ids = st.sidebar.multiselect(
        "Liked songs",
        options=list(songs_by_id.keys()),
        format_func=lambda sid: _song_label(songs_by_id[sid]),
    )
    liked = [songs_by_id[sid] for sid in liked_ids]
    profile = profile_from_songs(liked)

    if liked:
        with st.sidebar.expander("Derived profile", expanded=False):
            st.write({
                k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in profile.__dict__.items()
                if v not in (None, "")
            })
    else:
        st.sidebar.info("Pick at least one song to generate recommendations.")

    return profile, liked_ids


def render_results(
    results: list[tuple[Song, float, str]],
    max_score: float,
) -> None:
    """
    Render the top-k recommendation cards.

    Percentages are *absolute* — each song is scored against the theoretical
    perfect match for the user's profile (every feature exact, every
    categorical choice matching). A song that lines up closely with the
    slider values reads high; a poor numeric fit reads low, even if it
    happens to be the best of a weak field.
    """
    for rank, (song, score, explanation) in enumerate(results, start=1):
        pct = max(0, min(100, int(round(100 * score / max_score))))
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"**#{rank} — {song.title}**  \n*by {song.artist}*")
                st.caption(
                    f"🎼 {song.genre}  ·  💭 {song.mood}  ·  "
                    f"⚡ energy {song.energy:.2f}  ·  🎵 {int(song.tempo_bpm)} BPM"
                )
                st.write(explanation)
            with right:
                st.metric("Recommendation", f"{pct}%")
                st.progress(pct / 100)


def main() -> None:
    st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")
    st.title("🎵 Music Recommender")
    st.caption("Content-based filtering — pick a vibe, get songs.")

    songs = get_songs()
    genres = sorted({s.genre for s in songs})
    moods = sorted({s.mood for s in songs})

    mode = st.sidebar.radio(
        "Build profile from",
        options=["Manual sliders", "Liked songs"],
        horizontal=True,
    )
    st.sidebar.divider()

    liked_ids: list[int] = []
    if mode == "Manual sliders":
        profile = render_manual_sidebar(genres, moods)
    else:
        profile, liked_ids = render_liked_sidebar(songs)

    st.sidebar.divider()
    k, diversity = render_output_controls()

    if profile.is_empty():
        st.info("👈 Pick at least one song in the sidebar to see recommendations.")
        return

    candidates = [s for s in songs if s.id not in liked_ids]
    rec = Recommender(candidates)
    results = rec.rank(profile, k=k, diversity=diversity)

    st.subheader(f"Top {k} recommendations for your profile")
    render_results(results, max_possible_score(profile))


if __name__ == "__main__":
    main()
