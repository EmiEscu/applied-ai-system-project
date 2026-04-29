You are working for a startup music platform that wants to understand how big-name apps like Spotify or TikTok predict what users will love next. Your mission is to simulate and explain how a basic music recommendation system works by designing a modular architecture in Python that transforms song data and "taste profiles" into personalized suggestions.

🎯 Goals
    Explain the transformation of data into predictions, distinguishing between input features, user preferences, and ranking algorithms.
    Design and implement a weighted-score recommender in Python that uses attributes like genre, mood, and energy to calculate song relevance.
    Identify and document algorithmic bias, using AI to brainstorm how simple content-based systems might create "filter bubbles" or over-favor certain genres.
    Communicate technical reasoning through a structured Model Card that details intended use, data limitations, and future improvements.

## Algorithm Context

This project implements and compares two recommendation algorithm types:

**Content-Based Filtering**
- Each song has a feature vector: genre, mood, energy, tempo, etc.
- A user's "taste profile" is built from the attributes of songs they've liked
- Recommendations are ranked by weighted similarity score between the taste profile and unheard songs
- This is the primary algorithm being implemented in this codebase

**Collaborative Filtering (for comparison/discussion)**
- Recommendations are derived from the behavior of similar users, not song attributes
- Not the focus of this implementation, but referenced to contrast with content-based approaches
- Key tradeoff: collaborative filtering discovers surprising picks but suffers from cold-start; content-based is explainable but creates filter bubbles

**Key concepts to keep in mind across all conversations:**
- "Taste profile" = a weighted average of feature vectors from a user's liked songs
- "Relevance score" = dot product or cosine similarity between taste profile and a candidate song
- Filter bubble = when content-based systems only ever recommend songs sonically identical to past listens
- The Model Card documents bias, limitations, and intended use of the final system

**Thing to make sure you include**
- Make sure that all the code you produce runs correctly and reproducibly. 
- Include logging or guardrails. The code should track what it does and handle errors safely
- Has Clear setup steps. Someone else should be able to run it without guessing what to install


**Integrating AI: How AI will work with this program**
- We will work with two datasets, the current songs.csv file for when the user wants to manually input his own attributes and then an input box that will use AI to create an attribute that will be feed into out algorithm 
- The way the second algorithm will work is: User Input in a box (Ex. "I want a happy song similar to {song name}") -> Song name gets sent to Spotify Audio Features API to get attributes -> Gemini handles the natural language, spotify handles the song data -> Gemini will return a short explanation (Ex. "I set energy to 0.85 and tempo to 128 BPM because you asked for something intense and fast.") 
- There needs to be a refinement loop. If the user doesnt like the results then the user should be allowed to add another input such as "make it less aggressive" or "slower tempo" that lets Gemini adjust the existing slider values incrementally rather than starting over. Turning the AI into a proper agentic conversational loop.
- There should be some Validation so that Gemini does not set values outside the expected range or misinterpret something like "chill but energetic". We need to Validate and clamp Gemini's output values to the slider ranges before applying them. We can also give Gemini a strict JSON output format like: { "energy": 0.75, "valence": 0.5, "tempo": 110, "danceability": 0.6, "acousticness": 0.2 }. This is to make parsing reliable and predictable.

- Architecture that should be followed:
    User types: "Something like Blinding Lights but more acoustic"
        ↓
① Gemini parses input
   → extracts slider intent (energy, mood, tempo, etc.)
   → extracts song name + artist if mentioned
        ↓
② [If song mentioned] Spotify Search API
   → search("Blinding Lights The Weeknd") → returns track_id
        ↓
③ ReccoBeats Audio Features API
   → fetch(track_id) → returns { energy, valence, tempo, etc. }
        ↓
④ Gemini blends features
   → takes ReccoBeats values as base
   → applies user modifiers ("more acoustic" → lower acousticness)
   → outputs final slider JSON + explanation
        ↓
⑤ Your existing ranking algorithm
   → scores songs against final slider values
        ↓
⑥ UI shows results + Gemini's explanation
        ↓
⑦ [Optional] User refines → "slower tempo"
   → loops back to step ④ with updated modifier