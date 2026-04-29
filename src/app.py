"""
Streamlit UI for the Music Recommender.

Run with:
    streamlit run src/app.py

Three ways to build a taste profile:
  - Manual sliders: pick a genre and dial the audio-feature sliders.
  - Liked songs:    pick a few songs you already like; the profile is the
                    average of their feature vectors (true content-based UX).
  - AI prompt:      describe a vibe in natural language; Gemini parses it,
                    optionally pulls audio features for a referenced song
                    via ReccoBeats, and returns a slider profile + explanation.
                    Supports a refinement loop ("slower", "less aggressive", …).
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
from dotenv import load_dotenv

load_dotenv()
setup_logging()

# AI imports are deferred until AI mode is selected so the app still loads
# (and Manual / Liked modes still work) when GEMINI_API_KEY is missing.

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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


def render_manual_sidebar(genres: list[str]) -> UserProfile:
    """Manual mode: dropdowns + sliders → UserProfile."""
    st.sidebar.header("Your taste profile")

    genre = st.sidebar.selectbox(
        "Favorite genre",
        options=[""] + genres,
        format_func=lambda x: x if x else "Any",
    )

    st.sidebar.subheader("Audio features")
    energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5, 0.05)
    valence = st.sidebar.slider("Valence (positivity)", 0.0, 1.0, 0.5, 0.05)
    danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5, 0.05)
    acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.5, 0.05)
    tempo = st.sidebar.slider("Tempo (BPM)", 60, 200, 120, 1)

    with st.sidebar.expander("Advanced features", expanded=False):
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.05)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.05)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1, 0.05)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -8.0, 0.5)
        mode_label = st.selectbox(
            "Mode",
            options=["Any", "Major", "Minor"],
            help="Major = brighter, Minor = darker tonality.",
        )
        key_label = st.selectbox(
            "Key",
            options=["Any"] + _KEY_NAMES,
        )

    mode_map = {"Any": None, "Major": 1, "Minor": 0}
    mode = mode_map[mode_label]
    key = None if key_label == "Any" else _KEY_NAMES.index(key_label)

    return UserProfile(
        genre=genre,
        energy=energy,
        valence=valence,
        danceability=danceability,
        acousticness=acousticness,
        instrumentalness=instrumentalness,
        liveness=liveness,
        loudness=loudness,
        speechiness=speechiness,
        tempo=float(tempo),
        mode=mode,
        key=key,
    )


def render_liked_sidebar(songs: list[Song]) -> tuple[UserProfile, list[str]]:
    """Liked-songs mode: multiselect → averaged profile + selected ids.

    With a 30k-song catalog, building the multiselect over every track is a
    bad UX (and slow). We only show a sample — enough variety to demo the
    averaging behavior without a 5-second dropdown.
    """
    st.sidebar.header("Songs you like")
    st.sidebar.caption(
        "Pick a handful of tracks you already love. The profile is built from "
        "the average of their audio features — Spotify-style content-based filtering."
    )

    # The catalog is huge (~30k rows), so a multiselect over every track is
    # a usability cliff. Sample a manageable subset and let the user filter
    # by typing — Streamlit's multiselect already does substring matching.
    songs_by_id = {s.id: s for s in songs}
    SAMPLE_LIMIT = 500
    if len(songs_by_id) > SAMPLE_LIMIT:
        sample_ids = list(songs_by_id.keys())[:SAMPLE_LIMIT]
        st.sidebar.caption(
            f"Showing the first {SAMPLE_LIMIT:,} of {len(songs_by_id):,} tracks. "
            "Type to filter."
        )
    else:
        sample_ids = list(songs_by_id.keys())

    liked_ids = st.sidebar.multiselect(
        "Liked songs",
        options=sample_ids,
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


@st.cache_resource(show_spinner=False)
def get_ai_assistant():
    """
    Construct the AI stack once per session. Cached so we don't rebuild the
    Gemini client (and re-fetch a Spotify token) on every rerun.

    Returns (assistant, init_error_message). On failure we return (None, msg)
    rather than raising, so the caller can show a friendly UI error instead
    of crashing the page.
    """
    try:
        from ai_assistant import AIAssistant
        from spotify_client import SpotifyAuthError, SpotifyClient
    except Exception as e:
        return None, f"Could not import AI modules: {e}"

    try:
        spotify = SpotifyClient()
    except SpotifyAuthError as e:
        # Spotify is optional — fall through with assistant only.
        spotify = None
        spotify_warn = str(e)
    except Exception as e:
        spotify = None
        spotify_warn = f"Spotify client failed: {e}"
    else:
        spotify_warn = None

    try:
        assistant = AIAssistant(spotify=spotify)
    except Exception as e:
        return None, f"Could not start Gemini client: {e}"

    return assistant, spotify_warn


def render_ai_sidebar() -> tuple[UserProfile, str | None, str | None]:
    """
    AI prompt mode sidebar.

    Returns (profile, explanation_or_None, warning_or_None).
    All result state is held in st.session_state so the profile persists
    across reruns and refinement turns.
    """
    st.sidebar.header("Describe a vibe")
    st.sidebar.caption(
        "Examples: \"Something like Blinding Lights but more acoustic\"  ·  "
        "\"A happy upbeat dance song around 120 BPM\"  ·  "
        "\"Chill rainy-day jazz, low energy\""
    )

    assistant, init_warn = get_ai_assistant()
    if assistant is None:
        st.sidebar.error(init_warn or "AI assistant unavailable.")
        return UserProfile(), None, None
    if init_warn:
        st.sidebar.info(f"Spotify: {init_warn} — running Gemini-only.")

    user_text = st.sidebar.text_area(
        "Your request",
        key="ai_user_text",
        height=100,
        placeholder="e.g. Something like Blinding Lights but more acoustic",
    )
    submit = st.sidebar.button("✨ Generate profile", use_container_width=True)

    if submit and user_text.strip():
        with st.spinner("Asking Gemini…"):
            try:
                blend, base, warn, excluded_title = assistant.build_profile(user_text)
            except Exception as e:
                st.sidebar.error(f"AI request failed: {e}")
            else:
                st.session_state["ai_profile"] = blend.profile
                st.session_state["ai_explanation"] = blend.explanation
                st.session_state["ai_warning"] = warn
                st.session_state["ai_base"] = (
                    f"{base.title} — {base.artist}" if base else None
                )
                # The track the user explicitly referenced shouldn't show up
                # in their own recommendations — they already know about it.
                # Title-only match so the exclusion still fires when Spotify
                # isn't configured / lookup fails (covers every album edition
                # of the song too).
                st.session_state["ai_excluded_title"] = (
                    excluded_title.strip().lower() if excluded_title else None
                )

    profile: UserProfile = st.session_state.get("ai_profile") or UserProfile()
    explanation = st.session_state.get("ai_explanation")
    warning = st.session_state.get("ai_warning")
    base_label = st.session_state.get("ai_base")

    if not profile.is_empty():
        with st.sidebar.expander("Current AI profile", expanded=False):
            st.write({
                k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in profile.__dict__.items()
                if v not in (None, "")
            })
            if base_label:
                st.caption(f"Spotify base: {base_label}")

        st.sidebar.divider()
        st.sidebar.subheader("Refine")
        refine_text = st.sidebar.text_input(
            "Tweak the result",
            key="ai_refine_text",
            placeholder="e.g. slower tempo, less aggressive",
        )
        if st.sidebar.button("🔧 Apply tweak", use_container_width=True) and refine_text.strip():
            with st.spinner("Refining…"):
                try:
                    blend = assistant.refine_profile(profile, refine_text)
                except Exception as e:
                    st.sidebar.error(f"Refinement failed: {e}")
                else:
                    st.session_state["ai_profile"] = blend.profile
                    st.session_state["ai_explanation"] = blend.explanation
                    # Refinement doesn't re-query Spotify; keep prior warning/base.
                    profile = blend.profile
                    explanation = blend.explanation

        if st.sidebar.button("Clear AI profile", use_container_width=True):
            for key in (
                "ai_profile", "ai_explanation", "ai_warning",
                "ai_base", "ai_excluded_title",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    return profile, explanation, warning


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
                    f"🎼 {song.genre}  ·  ⚡ energy {song.energy:.2f}  ·  "
                    f"😊 valence {song.valence:.2f}  ·  🎵 {int(song.tempo)} BPM"
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

    mode = st.sidebar.radio(
        "Build profile from",
        options=["Manual sliders", "Liked songs", "AI prompt"],
        horizontal=True,
    )
    st.sidebar.divider()

    liked_ids: list[str] = []
    ai_explanation: str | None = None
    ai_warning: str | None = None
    if mode == "Manual sliders":
        profile = render_manual_sidebar(genres)
    elif mode == "Liked songs":
        profile, liked_ids = render_liked_sidebar(songs)
    else:
        profile, ai_explanation, ai_warning = render_ai_sidebar()

    st.sidebar.divider()
    k, diversity = render_output_controls()

    if profile.is_empty():
        if mode == "AI prompt":
            st.info("👈 Describe a vibe in the sidebar and click *Generate profile*.")
        else:
            st.info("👈 Pick at least one song in the sidebar to see recommendations.")
        return

    if ai_warning:
        st.warning(ai_warning)
    if ai_explanation:
        with st.container(border=True):
            st.markdown("**🤖 Gemini's reasoning**")
            st.write(ai_explanation)

    # Liked songs: exclude every album edition / playlist copy by sharing
    # the recommender's (title, artist) dedup key.
    songs_by_id = {s.id: s for s in songs}
    excluded_keys: set[tuple[str, str]] = {
        songs_by_id[sid].dedup_key() for sid in liked_ids if sid in songs_by_id
    }
    # AI mode: also exclude by title alone so the track the user named is
    # never recommended back — even when Spotify lookup failed and we never
    # resolved the artist. Scoped to AI mode so stale session state from a
    # prior AI run can't leak into manual / liked modes.
    excluded_title: str | None = None
    if mode == "AI prompt":
        excluded_title = st.session_state.get("ai_excluded_title")

    def _is_candidate(s: Song) -> bool:
        if s.dedup_key() in excluded_keys:
            return False
        if excluded_title and s.title.strip().lower() == excluded_title:
            return False
        return True

    candidates = [s for s in songs if _is_candidate(s)]
    rec = Recommender(candidates)
    results = rec.rank(profile, k=k, diversity=diversity)

    st.subheader(f"Top {k} recommendations for your profile")
    render_results(results, max_possible_score(profile))


if __name__ == "__main__":
    main()
