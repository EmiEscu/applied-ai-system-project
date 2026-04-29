"""
Gemini-powered natural-language layer for the recommender.

Pipeline (mirrors CLAUDE.md ①–⑦):
    user text
        → parse_request()              # Gemini extracts intent + (optional) song
        → SpotifyClient.search_track() # if a song was mentioned, get its id
        → ReccoBeatsClient.find_features_by_spotify_id()  # real audio features
        → blend_features()             # Gemini fuses base + modifiers
        → UserProfile + explanation
        → [optional] refine_profile()  # follow-up turn adjusts existing profile

Two design rules drive everything below:

  1. Gemini outputs strict JSON. We use response_mime_type="application/json"
     plus a response_schema so the model literally cannot return prose. That
     makes parsing predictable and removes the "stripping ```json fences"
     dance.

  2. We never trust Gemini's numbers blindly. Every value is clamped into the
     slider's valid range (0..1 for the [0,1] features, 60..200 BPM for tempo,
     -60..0 dB for loudness, integer ranges for mode/key) before being
     applied. This is the validation guardrail from CLAUDE.md
     "There should be some Validation".
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types as genai_types

from loggings import get_logger
from models import UserProfile
from reccobeats_client import (
    ReccoBeatsClient,
    ReccoBeatsError,
    ReccoBeatsNotFoundError,
    TrackFeatures,
)
from spotify_client import SpotifyClient, SpotifyNotFoundError

logger = get_logger(__name__)

# Default model: 2.5 Flash is cheap, fast, and follows JSON schemas well.
# Override with GEMINI_MODEL if you want to A/B another model.
_DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Slider ranges — must match the Streamlit sliders in app.py. Centralized
# here so the clamp logic and the schema agree.
_UNIT_FEATURES = (
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
)
_TEMPO_RANGE = (60.0, 200.0)
_LOUDNESS_RANGE = (-60.0, 0.0)
_KEY_RANGE = (0, 11)
_MODE_RANGE = (0, 1)


# ---------------------------------------------------------------------------
# Schemas (Gemini "controlled output").
#
# These are pydantic-style dicts that google-genai forwards to the model as
# a hard JSON-schema constraint, not just a hint. Keep them flat and explicit.
# ---------------------------------------------------------------------------

_PARSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        # The bare song "title artist" string the user mentioned, or "" if
        # they didn't reference a specific song. We let Gemini write the
        # search query directly so it can normalize "blinding lights weeknd"
        # → "Blinding Lights The Weeknd".
        "song_query": {"type": "string"},
        # Just the song title (no artist), used to exclude the referenced
        # track from its own recommendation list even when Spotify lookup
        # fails or isn't configured.
        "song_title": {"type": "string"},
        # Free-form modifiers the user wants applied on top of the base
        # song's features (e.g. "more acoustic", "slower", "less aggressive").
        # Empty string when the user only named a song with no adjectives.
        "modifiers": {"type": "string"},
        # Direct slider hints when the user didn't reference a song at all
        # (e.g. "happy upbeat dance song"). All optional.
        "energy": {"type": "number"},
        "valence": {"type": "number"},
        "danceability": {"type": "number"},
        "acousticness": {"type": "number"},
        "instrumentalness": {"type": "number"},
        "liveness": {"type": "number"},
        "speechiness": {"type": "number"},
        "loudness": {"type": "number"},
        "tempo": {"type": "number"},
        "mode": {"type": "integer"},
        "key": {"type": "integer"},
        "genre": {"type": "string"},
    },
    "required": ["song_query", "modifiers"],
}

_BLEND_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "energy": {"type": "number"},
        "valence": {"type": "number"},
        "danceability": {"type": "number"},
        "acousticness": {"type": "number"},
        "instrumentalness": {"type": "number"},
        "liveness": {"type": "number"},
        "speechiness": {"type": "number"},
        "loudness": {"type": "number"},
        "tempo": {"type": "number"},
        "mode": {"type": "integer"},
        "key": {"type": "integer"},
        "genre": {"type": "string"},
        # One short sentence explaining *why* these values were chosen,
        # surfaced in the UI per CLAUDE.md step ⑥.
        "explanation": {"type": "string"},
    },
    "required": [
        "energy", "valence", "danceability", "acousticness",
        "instrumentalness", "liveness", "speechiness", "loudness",
        "tempo", "explanation",
    ],
}


@dataclass
class ParsedRequest:
    """Structured form of the user's natural-language input."""
    song_query: str
    song_title: str  # bare title only, for exclusion-from-recs
    modifiers: str
    hints: dict  # any direct slider hints — keys are subset of UserProfile fields


@dataclass
class BlendResult:
    """Final UserProfile + the human-readable reason Gemini gave for it."""
    profile: UserProfile
    explanation: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _clamp_int(x, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def _clamp_profile_dict(d: dict) -> dict:
    """Coerce + clamp every numeric slider field that's present in d.
    This function is the inspector of the values gemini gave for the slider"""
    out = dict(d)
    for k in _UNIT_FEATURES:
        if k in out and out[k] is not None:
            out[k] = _clamp(out[k], 0.0, 1.0)
    if "tempo" in out and out["tempo"] is not None:
        out["tempo"] = _clamp(out["tempo"], *_TEMPO_RANGE)
    if "loudness" in out and out["loudness"] is not None:
        out["loudness"] = _clamp(out["loudness"], *_LOUDNESS_RANGE)
    if "mode" in out and out["mode"] is not None:
        out["mode"] = _clamp_int(out["mode"], *_MODE_RANGE)
    if "key" in out and out["key"] is not None:
        out["key"] = _clamp_int(out["key"], *_KEY_RANGE)
    return out


def _profile_from_dict(d: dict) -> UserProfile:
    """Build a UserProfile from Gemini's parsed dict, dropping unknown keys."""
    return UserProfile(
        genre=str(d.get("genre", "") or ""),
        energy=d.get("energy"),
        valence=d.get("valence"),
        danceability=d.get("danceability"),
        acousticness=d.get("acousticness"),
        instrumentalness=d.get("instrumentalness"),
        liveness=d.get("liveness"),
        loudness=d.get("loudness"),
        speechiness=d.get("speechiness"),
        tempo=d.get("tempo"),
        mode=d.get("mode"),
        key=d.get("key"),
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AIAssistant:
    """
    Encapsulates the Gemini calls + Spotify lookups for the natural-language
    mode. Stateless across requests — the caller (Streamlit session) holds
    the "current profile" and passes it back in for refinement turns.
    """

    def __init__(
        self,
        spotify: Optional[SpotifyClient] = None,
        reccobeats: Optional[ReccoBeatsClient] = None,
        gemini_api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Get one at "
                "https://aistudio.google.com/app/apikey and export it."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        # Spotify is optional: if construction fails (no creds, deprecated
        # endpoint, etc.), we still let the user run a Gemini-only flow.
        self._spotify = spotify
        # ReccoBeats is the fallback for Spotify's deprecated /audio-features
        # endpoint. Default to enabled — it needs no credentials.
        self._reccobeats = reccobeats if reccobeats is not None else ReccoBeatsClient()

    # ------------------------------------------------------------------ ①

    def parse_request(self, user_text: str) -> ParsedRequest:
        """
        Step ① in the pipeline: Gemini reads the user's free text and
        extracts (a) a song query if one was mentioned, (b) modifier text,
        and (c) any direct slider hints.
        """
        prompt = (
            "You are parsing a music recommendation request. Extract a JSON "
            "object describing the user's intent.\n\n"
            "Rules:\n"
            "- If the user named a specific song (with or without artist), "
            "put a clean search query like 'Title Artist' in 'song_query' "
            "AND put just the bare song title (no artist, no extra words) "
            "in 'song_title'. Otherwise leave both empty.\n"
            "- Put any qualitative modifiers ('more acoustic', 'slower', "
            "'less aggressive', 'happier', 'more instrumental', 'live "
            "recording') in 'modifiers'. Empty string if none.\n"
            "- If the user described a vibe with no song reference, you may "
            "fill direct slider hints. Ranges:\n"
            "    energy/valence/danceability/acousticness/instrumentalness/"
            "liveness/speechiness ∈ [0,1]\n"
            "    tempo ∈ [60,200] BPM, loudness ∈ [-60,0] dB\n"
            "    mode ∈ {0=minor, 1=major}, key ∈ [0,11] (0=C, 1=C#, ... 11=B)\n"
            "  Otherwise omit them.\n\n"
            f"User input: {user_text!r}"
        )
        raw = self._json_call(prompt, _PARSE_SCHEMA)
        # Pull hints out of the flat object so callers don't conflate them
        # with required fields.
        hint_keys = (
            *_UNIT_FEATURES,
            "tempo", "loudness", "mode", "key", "genre",
        )
        hints = {k: raw[k] for k in hint_keys if k in raw}
        if hints:
            hints = _clamp_profile_dict(hints)
        return ParsedRequest(
            song_query=raw.get("song_query", "") or "",
            song_title=raw.get("song_title", "") or "",
            modifiers=raw.get("modifiers", "") or "",
            hints=hints,
        )

    # ------------------------------------------------------------------ ②③④

    def blend_features(
        self,
        modifiers: str,
        base: Optional[TrackFeatures] = None,
        hints: Optional[dict] = None,
    ) -> BlendResult:
        """
        Step ④: produce the final slider values. If `base` is provided
        (Spotify lookup succeeded), Gemini starts from those numbers and
        applies `modifiers`. If not, Gemini works from `hints` (or pure
        modifier text) alone.
        """
        base_dict = (
            {
                "energy": base.energy,
                "valence": base.valence,
                "danceability": base.danceability,
                "acousticness": base.acousticness,
                "instrumentalness": base.instrumentalness,
                "liveness": base.liveness,
                "speechiness": base.speechiness,
                "loudness": base.loudness,
                "tempo": base.tempo,
                "mode": base.mode,
                "key": base.key,
            }
            if base else {}
        )
        prompt = (
            "You are tuning the sliders of a music recommender. Output JSON "
            "with the final slider values and a one-sentence explanation of "
            "the choices you made (mention which features you nudged and "
            "why).\n\n"
            "Hard ranges:\n"
            "  energy, valence, danceability, acousticness,\n"
            "  instrumentalness, liveness, speechiness ∈ [0, 1]\n"
            "  tempo ∈ [60, 200] BPM\n"
            "  loudness ∈ [-60, 0] dB\n"
            "  mode ∈ {0=minor, 1=major}\n"
            "  key ∈ [0, 11] (0=C, 1=C#, … 11=B)\n"
            "  genre is a short string or empty.\n\n"
            f"Base features (from a referenced song, may be empty): "
            f"{json.dumps(base_dict)}\n"
            f"Direct hints from user (override base if present): "
            f"{json.dumps(hints or {})}\n"
            f"Qualitative modifiers to apply: {modifiers!r}\n\n"
            "If there are no modifiers and no hints, return the base values "
            "unchanged with an explanation that says so."
        )
        raw = self._json_call(prompt, _BLEND_SCHEMA)
        explanation = raw.pop("explanation", "")
        clamped = _clamp_profile_dict(raw)
        logger.info("Gemini blend → %s", clamped)
        return BlendResult(
            profile=_profile_from_dict(clamped),
            explanation=explanation,
        )

    # ------------------------------------------------------------------ ⑦

    def refine_profile(
        self,
        prior: UserProfile,
        modifier_text: str,
    ) -> BlendResult:
        """
        Step ⑦: the user reacted to a previous result ("slower", "less
        aggressive"). Adjust the existing profile incrementally rather than
        starting over.
        """
        prior_dict = {
            k: v for k, v in prior.__dict__.items() if v not in (None, "")
        }
        prompt = (
            "You are refining an existing music recommender profile based on "
            "the user's follow-up feedback. Nudge the relevant sliders only "
            "— do not reset values the user didn't comment on. Output JSON.\n\n"
            "Hard ranges:\n"
            "  energy, valence, danceability, acousticness,\n"
            "  instrumentalness, liveness, speechiness ∈ [0, 1]\n"
            "  tempo ∈ [60, 200] BPM, loudness ∈ [-60, 0] dB\n"
            "  mode ∈ {0=minor, 1=major}, key ∈ [0, 11]\n\n"
            f"Current profile: {json.dumps(prior_dict)}\n"
            f"User feedback: {modifier_text!r}\n\n"
            "Explanation should describe *what changed* and why."
        )
        raw = self._json_call(prompt, _BLEND_SCHEMA)
        explanation = raw.pop("explanation", "")
        clamped = _clamp_profile_dict(raw)
        logger.info("Gemini refine → %s", clamped)
        return BlendResult(
            profile=_profile_from_dict(clamped),
            explanation=explanation,
        )

    # ------------------------------------------------------------------ orchestration

    def build_profile(
        self,
        user_text: str,
    ) -> tuple[BlendResult, Optional[TrackFeatures], Optional[str], str]:
        """
        End-to-end ① → ④ for a fresh request.

        Returns (blend_result, spotify_features_or_None, warning_or_None,
        excluded_title). `excluded_title` is the bare title of the song the
        user referenced (if any) — the UI uses it to filter that song out
        of its own recommendations. Prefer Spotify's canonical title when
        the lookup succeeded (handles the user's casing / punctuation
        quirks); fall back to what Gemini parsed so the exclusion still
        fires when Spotify isn't configured.

        The warning string is non-fatal — surface it in the UI when present
        (e.g. "couldn't reach Spotify, used Gemini-only estimate").
        """
        parsed = self.parse_request(user_text)
        warning: Optional[str] = None
        base: Optional[TrackFeatures] = None

        if parsed.song_query:
            base, warning = self._resolve_track_features(parsed.song_query)

        blend = self.blend_features(
            modifiers=parsed.modifiers,
            base=base,
            hints=parsed.hints,
        )
        excluded_title = (base.title if base else parsed.song_title) or ""
        return blend, base, warning, excluded_title

    def _resolve_track_features(
        self,
        query: str,
    ) -> tuple[Optional[TrackFeatures], Optional[str]]:
        """
        Resolve a free-text song reference to a feature vector.

        Spotify is used only for Search (which any developer app can hit) —
        the actual audio features come from ReccoBeats, a free unauthenticated
        service whose schema mirrors Spotify's deprecated /audio-features
        endpoint. ReccoBeats keys lookups on Spotify track IDs, so this
        two-call dance is unavoidable: there is no public name-search on
        ReccoBeats.

        Returns (features_or_None, warning_or_None). The warning is non-fatal
        — when it's set, the AI pipeline still produces a profile, it just
        won't have a concrete song's numbers to anchor on.
        """
        if self._spotify is None:
            return None, "Spotify is not configured — can't resolve song references."
        if self._reccobeats is None:
            return None, "ReccoBeats client is disabled — can't fetch audio features."

        # Step 1: Spotify Search → canonical track id, title, artist.
        try:
            track = self._spotify.search_track(query)
        except SpotifyNotFoundError:
            return None, f"Couldn't find {query!r} on Spotify — proceeding from your description alone."
        except Exception as e:
            logger.warning("Spotify search error: %s", e)
            return None, f"Spotify search failed: {e}. Falling back to Gemini-only estimate."

        spotify_id = track["id"]
        title = track["name"]
        artist = ", ".join(a["name"] for a in track.get("artists", []))

        # Step 2: ReccoBeats audio features keyed by Spotify id.
        try:
            base = self._reccobeats.find_features_by_spotify_id(
                spotify_id, title=title, artist=artist,
            )
            return base, None
        except ReccoBeatsNotFoundError:
            return None, f"{title} — {artist} found on Spotify but not in ReccoBeats. Falling back to Gemini-only estimate."
        except ReccoBeatsError as e:
            logger.warning("ReccoBeats error: %s", e)
            return None, f"ReccoBeats lookup failed: {e}. Falling back to Gemini-only estimate."

    # ------------------------------------------------------------------ internals

    def _json_call(self, prompt: str, schema: dict) -> dict:
        """
        Issue a Gemini call constrained to `schema` and return the parsed
        dict. Any non-JSON output here is a programming error (schema-
        constrained calls should always return parseable JSON), so we let
        json.JSONDecodeError propagate rather than papering over it.
        """
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        )
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        text = (resp.text or "").strip()
        return json.loads(text)
