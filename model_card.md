# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**Gen Recommendation V1**  

---

## 2. Intended Use  


- What kind of recommendations does it generate  

My recommender model is designed to give you the best recommendations possible of a list of songs base on the Genre you enjoy and the vibe you are going for. The model assumes the user really likes the genre they intially choose so it puts a lot more weight on it 'Genre' over any other feature. This would be more for classroom exploration rather than for a real user.

---

## 3. How the Model Works  

The model works simply by assigning a weight to all the features a song may have. It will then score that song by simply comparing how similar the weight is compared to that of the original target/user profile.

---

## 4. Data  

The catalog has 20 songs stored in `data/songs.csv`. No songs were added or removed.

There are 14 genres: pop, lofi, rock, ambient, jazz, synthwave, indie pop, classical, hip-hop, reggae, country, metal, blues, latin, folk, r&b, and electronic. There are 15 moods: happy, chill, intense, relaxed, moody, focused, melancholic, energetic, peaceful, nostalgic, angry, gritty, romantic, dreamy, uplifting, and euphoric. Most genres have only one song. Lofi has three and pop has two.

Each song has seven features: genre, mood, energy, tempo_bpm, valence, danceability, and acousticness. Energy ranges from 0.21 to 0.97. Tempo ranges from 58 to 180 BPM.

The dataset is missing several popular genres like k-pop, rap, afrobeats, and EDM. It has no songs in languages other than English. It also does not capture lyrics, artist popularity, or release year. A real listener's taste depends on all of these things, so this dataset only covers a narrow slice of musical preference.

---

## 5. Strengths  

Where does your system seem to work well  

The system works really well if the user is searching for a song with a similar genre to the one their profile has. It makes a good prediction on finding songs that are similar. 

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly is the weight imbalance. The problem here is how the system over prioritizes labels, meaning that for users whose taste is defined by the sound and the vibe of a song will be feed recommendation on labels that might not reflect their taste propperly. The root cause of this is simply the weight Genre carries.

---

## 7. Evaluation  

Edge 1: Contradictory (genre=lofi, mood=chill, energy=0.95)
A user who says they want lofi and chill but sets energy to 0.95 — an impossible combination in real music.

Surprise: The system happily recommended Midnight Coding (energy=0.51) at #1 with a score of 7.107. The genre+mood bonus of +3.5 completely buried the fact that this song's energy is 0.44 away from the target. Meanwhile Storm Runner (rock, energy=0.91 — nearly perfect energy match) was pushed to #5 at 3.886. The system cannot detect contradictions — it just adds bonuses and ignores that the recommendation doesn't actually match what the user numerically asked for.

Edge 2: Happy-Angry (mood=angry, valence=0.95)
Angry mood but maximally happy valence — another internal contradiction.

Surprise: Iron Cathedral (valence=0.19) won at 7.475 despite having a valence 0.76 away from the target. The genre+mood bonus of +3.5 swallowed the valence penalty entirely. The system doesn't notice that recommending a deeply dark song to someone who wants high valence is contradictory. The valence weight (0.8) is simply too weak to compete with categorical matches.

Edge 3: All-Zeros (every numeric feature = 0.0, no genre/mood)
The "silent listener" — no preferences at all.

Surprise: The system still differentiated songs. Requiem for the Lost (classical/melancholic) won at 2.815 because it has the lowest values across the board. But the scores are tightly clustered (2.2–2.8), meaning the system is barely distinguishing. Also, Iron Cathedral (metal/angry) appeared at #4 — it has high energy (0.97) but very low acousticness (0.04), so the acousticness proximity to zero helped it. A metal song recommended to someone with zero energy preference is a strange result.

Edge 4: All-Maxed (every feature at 1.0 / 200 BPM)
The "everything lover" — wants maximum everything.

Surprise: Corazon Caliente (latin/romantic) beat Synthetic Sunrise (electronic/euphoric, energy=0.95). No real song can have both acousticness=1.0 and energy=1.0 simultaneously, so scores compressed into a narrow 3.0–3.2 range. The system essentially gave random-feeling results because the profile is physically impossible.

Edge 5: Non-Existent Genre (k-pop, whimsical)
Genre and mood that don't exist in the catalog.

Surprise: This is actually where the system performed best. Without categorical bonuses distorting the results, ranking fell purely to numeric similarity. The top result — Golden Hour Glow (R&B/uplifting, 4.626) — is genuinely the closest-sounding song to the numeric profile. All top-5 songs are mid-energy, melodic, danceable tracks. This accidentally demonstrates that removing the genre/mood bonuses can produce more musically coherent recommendations.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

- Making sure that the weights are better distributed so genre doesnt shadows everything else.

- Adding some sort of randomization to it since sometimes when a user prefers how a song sounds like (vibe) rather then labels, its easier to find something at random that still has some similarities to their current preferences.

- Lastly I would want add two different options, one for the user to continue getting recommendations in their current genre and another one where the user can get random more vibe targetted music.

---

## 9. Personal Reflection  

This activity taught me a lot abour recommendation systems and their simple approach yet complex execution. You have to take into account so many features and be able to know which ones should carry more weight in the decision of recommending the next song. 

I think human judgment still matters even if a model seems smart because at the end of the day, songs are feelings. Humans have complex emotions which cant always be measured or weighted. 
