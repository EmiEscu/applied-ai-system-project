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
