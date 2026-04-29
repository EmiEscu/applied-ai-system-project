# 🎵 Music Recommender Simulation

## Project Summary

A content-based music recommender that takes a user's "taste profile" and ranks a catalog of ~32,000 Spotify tracks against it. The system started as a small CLI experiment with 20 hand-written songs and a 7-feature scoring rule, and grew into a Streamlit web app with three different ways to build a profile, a 12-feature weighted scorer, an MMR diversity re-ranker, and an AI assistant that turns free-text vibes into slider values.

Goals:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what the system gets right and wrong
- Reflect on how this mirrors real-world AI recommenders
- Layer a natural-language interface on top, mediated by a real LLM and a real audio-features API

---

## Demo

Walkthrough of the app and the three profile modes:

[![Music Recommender Demo](https://img.youtube.com/vi/ieWK0tLvIjw/maxresdefault.jpg)](https://youtu.be/ieWK0tLvIjw)

▶️ [Watch on YouTube](https://youtu.be/ieWK0tLvIjw)

---

## What's New (vs. the starter project)

| Area | Starter version | Current version |
|------|-----------------|-----------------|
| Interface | CLI (`python -m src.main`) | Streamlit web app (`streamlit run src/app.py`) |
| Catalog | 20 hand-authored songs | ~32,000 tracks from the Spotify audio-features dataset |
| Features used | 7 (genre, mood, energy, tempo, valence, danceability, acousticness) | 12 (adds instrumentalness, liveness, loudness, speechiness, mode, key) |
| Profile sources | Manual sliders only | Manual sliders **·** "songs I like" centroid **·** AI natural-language prompt |
| Ranking | Top-k by raw score | Top-k by score → MMR re-rank for diversity |
| Dedup | None | (title, artist) collapse so album editions don't dominate the list |
| AI layer | None | Gemini 2.5 Flash + Spotify Search + ReccoBeats audio features, with a refinement loop |
| Logging | `print()` | `loggings.py` with module-level loggers |
| Tests | Basic | `pytest` against the scorer and recommender |

---

## How The System Works

There are two common ways music recommendation systems work: **collaborative filtering** uses other users' behavior (if User A and User B both like Lady Gaga and Bad Bunny, and User A likes Bruno Mars, User B probably will too); **content-based filtering** uses the audio features of the songs themselves (tempo, mood, genre, danceability, etc.). Since this project only has a single user, the system is content-based — the features of a song decide whether it gets recommended.

### Data shape

- **`Song`** — one row from the Spotify audio-features schema. Fields: `id`, `title`, `artist`, `genre`, plus the 12 numeric/categorical audio features. `dedup_key()` returns lower-cased `(title, artist)` so the recommender can collapse album editions of the same recording.
- **`UserProfile`** — the same shape, but every field is optional. An empty string or `None` means "no preference," and the scorer skips that feature entirely. This lets a profile express both fully specified taste (manual sliders) and partial taste (e.g., "happy and around 120 BPM, anything else goes").

### Scoring

The scorer is in [src/scoring.py](src/scoring.py).

For each song, the score is the sum of:

- **Categorical exact-match bonuses** — flat amounts added when the song's value equals the profile's value:
  - `genre` × 2.0
  - `mode` × 0.4
  - `key` × 0.3
- **Numeric weighted similarity** — for each feature, `similarity = max(0, 1 − |song − profile| / norm)` (so similarity is in [0, 1]), then multiplied by the feature's weight:

| Feature | Weight | Norm | Why this weight |
|---|---|---|---|
| energy | 1.5 | 1.0 | Widest useful spread across genres |
| acousticness | 1.2 | 1.0 | Cleanly separates electric vs. acoustic |
| valence | 0.8 | 1.0 | Emotional positivity |
| tempo | 0.8 | 140 BPM | Slider range 60–200, so 140 normalizes a full-range gap to sim=0 |
| instrumentalness | 0.8 | 1.0 | Splits vocal pop/rap from instrumental |
| loudness | 0.6 | 60 dB | Production loudness, range −60..0 dB |
| danceability | 0.5 | 1.0 | Largely correlated with energy |
| speechiness | 0.4 | 1.0 | Flags rap / spoken word |
| liveness | 0.4 | 1.0 | Flags live recordings |

Every song in the catalog is scored — no song is dropped early. A song that misses on genre still receives full numeric similarity points, so sonically close picks always have a chance to surface.

**Why per-feature absolute difference instead of cosine?** Cosine measures direction in feature space and ignores magnitude — a profile of `(1,1,1,1)` looks ~95–100% similar to almost any positive song vector. Absolute difference respects magnitude (0.5 is genuinely "halfway" to 1.0). Cosine *is* used elsewhere — inside MMR, for song-vs-song duplicate detection.

### Diversity re-ranking (MMR)

Top-k by raw score has a failure mode: the top 5 are often near-duplicates of the top 1. The recommender re-ranks the candidates with **Maximal Marginal Relevance**:

```
mmr_score = λ · normalized_relevance − (1 − λ) · max_cosine_sim_to_already_picked
```

The Streamlit UI exposes λ as a "Diversity" slider:

- `λ = 1.0` → pure relevance (behaves like a plain top-k)
- `λ = 0.7` (default) → relevance-heavy with a mild duplicate penalty
- `λ = 0.0` → ignore relevance, pick the most novel each step

This is the main lever for fighting the "filter bubble" problem documented in the model card.

---

## Three Ways to Build a Profile

The Streamlit app ([src/app.py](src/app.py)) offers three modes, switchable from the sidebar.

### 1. Manual sliders

Pick a genre, then dial the audio-feature sliders. Advanced features (instrumentalness, liveness, speechiness, loudness, mode, key) live in a collapsible section so the simple case stays simple.

### 2. Liked songs (centroid)

Pick a handful of tracks you already like. The profile is the **centroid** of those songs' feature vectors — the modal genre/key/mode plus the average of every numeric feature. This is closer to how a real Spotify-style recommender bootstraps a new user. The catalog is sampled to 500 tracks for the multiselect to keep the UI fast (a 30k-row dropdown is unusable).

### 3. AI prompt (Gemini + Spotify + ReccoBeats)

Type something like *"Something like Blinding Lights but more acoustic"* or *"Chill rainy-day jazz, low energy"* and the AI assistant produces a slider profile, an explanation of its choices, and a refinement box for follow-up tweaks.

The pipeline is in [src/ai_assistant.py](src/ai_assistant.py):

```
user text
  → Gemini parse_request()              # extracts (song reference, modifiers, slider hints) as strict JSON
  → SpotifyClient.search_track()        # if a song was mentioned, get its track id
  → ReccoBeatsClient.find_features_by_spotify_id()  # real audio features for that track
  → Gemini blend_features()             # fuses base features + modifiers into a final slider profile
  → UserProfile + one-sentence explanation
  → [optional] refine_profile()         # follow-up turn ("slower", "less aggressive") nudges only the relevant sliders
```

Two design rules drive the AI layer:

1. **Strict JSON output.** Gemini calls use `response_mime_type="application/json"` plus a JSON schema, so the model literally cannot return prose. No regex, no fence-stripping.
2. **Never trust the model's numbers.** Every value gets clamped to the slider's valid range (`[0, 1]` for unit features, 60–200 BPM, −60–0 dB, integer ranges for mode/key) before being applied.

**Why ReccoBeats?** Spotify's `/audio-features` endpoint was deprecated for new apps mid-2024 — repeated 403s blocked the original plan. ReccoBeats is a free, unauthenticated mirror that accepts Spotify track IDs directly and returns the same schema. Spotify is still used (Search only) to canonicalize the user's free-text song reference into a track ID; ReccoBeats then provides the numbers.

The track the user explicitly named is filtered out of its own recommendation list — by `(title, artist)` when Spotify resolved it, by title alone otherwise (so the exclusion still fires when Spotify isn't configured).

### Catalog dedup

The Spotify dataset has **one row per (track, playlist) pair**, and Spotify itself assigns separate track IDs to the same recording across album editions. Without dedup, the top 5 for a Pop profile is often the same song five times. Dedup happens *after* scoring (so each genre tag gets its fair shot at the genre bonus) but *before* MMR (so diversity sees one row per song). Remixes are intentionally not collapsed — their titles differ.

---

## Some Prompts to Answer

- **What features does each `Song` use?**
  genre, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo. Twelve features in total — the full Spotify audio-features schema.
- **What information does the `UserProfile` store?**
  Target values (or `None`/`""` for "no preference") for every one of those 12 features, plus genre and the integer-coded mode and key.
- **How does the `Recommender` compute a score?**
  Categorical exact-match bonuses for genre, mode, and key, plus per-feature weighted similarity for the nine numeric features (see the table above). Score is rounded to three decimals.
- **How are recommendations chosen?**
  Score every song → dedup by `(title, artist)` keeping the highest-scoring copy → MMR re-rank with the user's chosen diversity λ → return the top *k*.

---

## Bias and Limitations

- **Genre dominates.** With weight 2.0, the genre bonus often outranks several numeric features combined. A user who wants "something that *sounds* like jazz but isn't tagged jazz" can be locked out unless they zero out their genre.
- **Filter bubble.** Pure content-based filtering keeps recommending sonically identical songs. MMR mitigates this — turning the diversity slider down is the explicit escape hatch.
- **Sparse profile, ambiguous targets.** A profile with `energy=0` is treated as "the user wants very low-energy songs," not "no preference." `None` is the only way to opt out of a feature.
- **AI mode dependencies.** Without `GEMINI_API_KEY` the AI mode is disabled. Without Spotify credentials the AI mode still works, but song references can't be resolved — the assistant falls back to a Gemini-only estimate.
- **Spotify dataset bias.** The catalog is whatever was in the Spotify audio-features sample — heavily skewed toward English-language Western pop/rock/EDM. Non-Western genres are underrepresented.

The full discussion is in the [Model Card](model_card.md).

---

## Getting Started

### Setup

1. (Optional) Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional, for AI mode) Create a `.env` file in the project root:

   ```env
   GEMINI_API_KEY=your_gemini_key_here
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
   ```

   - `GEMINI_API_KEY` — get one at https://aistudio.google.com/app/apikey
   - Spotify creds (Client Credentials flow) — get them at https://developer.spotify.com/dashboard. Used for Search only; audio features come from ReccoBeats and need no credentials.
   - Manual and Liked-songs modes work without any API keys.

4. Run the app:

   ```bash
   streamlit run src/app.py
   ```

### Running Tests

```bash
pytest
```

Tests live in [tests/test_recommender.py](tests/test_recommender.py).

---

## Project Layout

```
applied-ai-system-final/
├── data/
│   └── songs.csv               # ~32k Spotify audio-features rows
├── src/
│   ├── app.py                  # Streamlit UI (three profile modes)
│   ├── models.py               # Song + UserProfile dataclasses
│   ├── recommender.py          # Catalog loading, profile-from-songs, ranker
│   ├── scoring.py              # Weighted-similarity scorer + MMR re-rank
│   ├── ai_assistant.py         # Gemini parse → blend → refine pipeline
│   ├── spotify_client.py       # Spotify Search (Client Credentials)
│   ├── reccobeats_client.py    # ReccoBeats audio-features lookup
│   └── loggings.py             # Module-level logger setup
├── tests/                      # pytest tests
├── assets/img/                 # Phase-4 experiment screenshots
├── model_card.md               # Bias / intended-use / limitations
├── requirements.txt
└── README.md
```

---

## Phase-4 Experiments (original 20-song catalog)

These ran against the original hand-authored 20-song catalog before the project moved to the full Spotify dataset. They're kept here as a record of how the scoring rule was tuned.

**Experiment 1 — Genre weight from 2.0 to 0.5:**
Lowering genre's bonus let sonically similar songs from other genres rank higher. "Gym Hero" (pop) dropped below "Synthetic Sunrise" (electronic) for the Pop profile because numeric features like energy and danceability carried more influence. This showed how much the categorical bonus dominates rankings — and motivated adding the diversity slider in the current version.

**Experiment 2 — Adding tempo and valence:**
Before adding these, songs that shared genre and energy often tied. With tempo (normalized) and valence weighted at 0.8 each, the system could separate them. For a Deep Intense Rock profile, "Iron Cathedral" rose because its low valence matched the dark preference.

**Experiment 3 — Different user types:**
Standard profiles worked well — Pop fans got pop songs, lofi fans got lofi. Edge cases revealed issues. The contradictory profile (high energy + chill mood + lofi genre) still ranked lofi songs first because the genre bonus outweighed the energy mismatch. The all-zeros profile treated 0.0 as a real target instead of "no preference," ranking the lowest-energy song first — which is why the current `UserProfile` uses `None` to mean "no preference" and `0.0` only when the user genuinely wants zero.

### All Songs Ranked by Score

![All Songs Ranked by Score](assets/img/phase4_1.png)

### Standard User Profiles

**High-Energy Pop Profile**

![High-Energy Pop Profile Results](assets/img/phase4_2.png)

**Chill Lofi Profile**

![Chill Lofi Profile Results](assets/img/phase4_3.png)

**Deep Intense Rock Profile**

![Deep Intense Rock Profile Results](assets/img/phase4_4.png)

### Adversarial / Edge-Case Profiles

**Edge 1: Contradictory (energy=0.95 + mood=chill + genre=lofi)**

![Edge Case 1 - Contradictory Profile](assets/img/phase4_5.png)

**Edge 2: Happy-Angry (mood=angry + valence=0.95)**

![Edge Case 2 - Happy-Angry Profile](assets/img/phase4_6.png)

**Edge 3: All-Zeros (every numeric feature = 0.0, no genre/mood)**

![Edge Case 3 - All-Zeros Profile](assets/img/phase4_7.png)

**Edge 4: All-Maxed (every numeric feature at maximum)**

![Edge Case 4 - All-Maxed Profile](assets/img/phase4_8.png)

**Edge 5: Non-Existent Genre (k-pop/whimsical — no catalog match)**

![Edge Case 5 - Non-Existent Genre Profile](assets/img/phase4_9.png)

### Top-10 (original 20-song catalog)

![Top 10 Recommendations](assets/img/top10.png)

---

## Reflection

Read and complete [`model_card.md`](model_card.md):

[**Model Card**](model_card.md)

**What was your biggest learning moment during this project?**
My biggest learning moment was watching the project evolve past the simple weighted-score scorer. The first version felt like a real recommender. Then I added a 30k-song catalog and the top 5 became five copies of the same song; that's when I understood why MMR exists. Then I added the AI layer and learned how much guardrail code (JSON schemas, clamping, exclusion logic) you need to make an LLM behave like a reliable component instead of a chat toy.

**How did using AI tools help you, and when did you need to double-check them?**
AI tools accelerated almost every part of the build — generating the scoring weights to start from, scaffolding the Streamlit UI, drafting the Gemini prompts. But I had to double-check almost every numeric default they suggested (the original weights had genre at 3.0 and danceability tied with energy, which made no sense once I looked at the data spread). The Spotify `/audio-features` deprecation was a case where AI confidently told me to call an endpoint that no longer works for new apps — fixing that meant reading the actual Spotify changelog and finding ReccoBeats myself.

**What surprised you about how simple algorithms can still "feel" like recommendations?**
A weighted dot-product is genuinely all it takes. The first time the lofi profile returned three lofi songs in a row, before I had even added MMR or the AI layer, it already felt like Spotify's "Made For You." That's both impressive and slightly worrying — the explainability gap between what the algorithm is doing (a weighted average) and what the user thinks is happening ("it knows me") is enormous.

**What would you try next if you extended this project?**
I'd add a real implicit-feedback loop: thumbs up / thumbs down on each recommendation, and use those signals to nudge the profile incrementally — a poor man's online learning. I'd also try a hybrid approach where collaborative-filtering signals (cluster the catalog, see which clusters the user's liked songs land in) are blended into the content-based score, so the system can occasionally surface a "left field" pick that pure content-based filtering would never reach.

---

## Improvements / Struggles

- **Spotify Audio Features API was deprecated.** I started by writing the AI mode against Spotify's `/audio-features` endpoint and kept hitting 403s. After confirming the endpoint had been restricted to legacy apps, I switched to https://reccobeats.com/, which is free, unauthenticated, and accepts Spotify track IDs directly. This is why the AI pipeline goes Spotify (Search) → ReccoBeats (Features) instead of Spotify-only.
- **Catalog explosion broke the UI.** Loading 30k songs into a Streamlit multiselect took ~5 seconds per render. Sampling to 500 tracks for the "Liked songs" picker made it usable.
- **Duplicate songs.** The Spotify dataset has the same song under multiple album editions and multiple playlists, so the unfiltered top-5 was almost always five copies of one track. The `dedup_key()` + post-score dedup fixed it.
- **Empty vs. zero in `UserProfile`.** The first version treated `0.0` as "no preference," which broke for users who genuinely wanted very low-energy songs. The fix was making every field `Optional` and using `None` as the "skip this feature" sentinel.
