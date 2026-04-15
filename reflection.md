# Reflection: Profile Comparisons

## Standard Profiles

### High-Energy Pop vs. Chill Lofi

These users are opposites. Pop wants loud and fast. Lofi wants quiet and slow. Their top-5 lists share zero songs. The system separated them correctly.

### High-Energy Pop vs. Deep Intense Rock

Both want high energy and low acousticness. Gym Hero shows up for both. For Pop, it gets the genre bonus. For Rock, it gets the mood bonus. It is not a rock song but the label still pushes it up. Iron Cathedral sounds like rock but ranked #3 because its genre says "metal" not "rock." The system treats close genres as completely different.

### Chill Lofi vs. Deep Intense Rock

Most extreme pair. One wants silence, the other wants noise. No overlap at all. Spacewalk Thoughts (ambient/chill) scored well for Lofi but lost 2 points just for saying "ambient" instead of "lofi." A real listener would not care about that label.

## Edge-Case Profiles

### Edge 1 (Contradictory: lofi + energy 0.95) vs. Chill Lofi

Same genre and mood. Very different energy: 0.25 vs. 0.95. Their top-3 are nearly identical anyway. The genre+mood bonus of 3.5 buried the energy mismatch. Someone asking for "intense lofi" still gets sleepy study beats.

### Edge 2 (Happy-Angry) vs. Deep Intense Rock

Happy-Angry wants angry metal but cheerful sounding (valence 0.95). Rock wants dark sounding (valence 0.30). Both got Iron Cathedral near the top. Iron Cathedral has valence 0.19. The Happy-Angry user asked for 0.95. That is a huge gap. The system ignored it because genre+mood was worth more.

### Edge 3 (All-Zeros) vs. Edge 4 (All-Maxed)

All-Zeros set everything to 0. All-Maxed set everything to max. Scores were tightly packed for both. The system could barely tell songs apart. Neither profile has genre or mood, so the big 3.5-point bonus never fires. Iron Cathedral appeared at #4 for All-Zeros. A metal song for someone who wants zero energy. It snuck in because one feature (acousticness 0.04) happened to be near zero.

### Edge 5 (Non-Existent Genre: k-pop) vs. High-Energy Pop

Both want danceable, melodic, mid-high energy music. Pop scored 7.998 at the top. K-pop scored 4.626. The 3.4-point gap is the missing genre+mood bonus. The k-pop results were actually solid based on sound alone. Users with niche taste get worse scores not because the music is missing but because the label is.
