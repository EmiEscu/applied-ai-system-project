# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

There are two ways that music recommending systems work, we have the collaborative filtering and the content-based filtering. Collaborative filtering work by using other users as a way to recommend a song. For example, if two users have the similar task in music such as they both like Lady Gaga, Bad Bunny, and Rock, then if User A likes Bruno Mars, than there is a high chance that User B will also enjoy Bruno Mars. Then there is content-based filtering which uses the audio characteristics of a song to provide a personal recommendation, such as tempo, mood, genre, danceability, etc. Since this project doesnt more than one user the method to recommend songs will be content-based were the characteristics of a song will determine whether it will be recommended or not. 



Some prompts to answer:

- What features does each `Song` use in your system
  - genre, mood, energy, tempo_bpm, valence, danceability, and acousticness
- What information does your `UserProfile` store
  - The taste profile stores the user's target values for every feature: a preferred genre label, a preferred mood label, and numeric targets for energy, tempo, valence, danceability, and acousticness.
- How does your `Recommender` compute a score for each song
  - Every song in the catalog is scored — no song is dropped early. A song that misses on genre and mood still receives numeric similarity points, so sonically close songs always surface.
  - **Step 1 — Categorical bonuses (binary):** If the song's genre exactly matches the user's preferred genre, add +2.0. If the mood matches, add +1.5. These fire or they don't; there is no partial credit.
  - **Step 2 — Continuous similarity:** For each numeric feature, compute `similarity = 1.0 − |song_value − target_value|`. This gives 1.0 for a perfect match and 0.0 when the values are as far apart as possible.
  - **Step 3 — Weighted sum:** Multiply each similarity by its feature weight and add to the score. Weights reflect how strongly each feature separates genres in the catalog:
    ```
    energy        × 1.5   (widest spread: 0.21–0.97)
    acousticness  × 1.2   (covers full range; separates electric from acoustic)
    valence       × 0.8   (emotional tone; partially overlaps with mood label)
    tempo         × 0.8   (normalized ÷ 200 first, or raw BPM would dominate)
    danceability  × 0.5   (weakest separator; correlated with energy)
    ```
  - **Max possible score ≈ 8.3** (genre match 2.0 + mood match 1.5 + all five numeric features at perfect similarity).
- How do you choose which songs to recommend
  - All 20 scored tuples are sorted in descending order by score. The top `k` (default 5) are returned. The song closest to the user's full taste profile — across both categorical and numeric dimensions — ranks first.




Some biases of this system is that Genre could be over prioritized, meaning that other features might be shadowed and effect the user recommendation. This is also an issue especially if the user want something that sounds similar to a genre, but isnt really that genre. 

![Top 10 Recommendations](img/top10.png)



---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

