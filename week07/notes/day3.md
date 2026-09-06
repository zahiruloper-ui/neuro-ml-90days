# Week 7 — Day 3: Split

## Goal
Compare naive random split vs. a leakage-safe time-block split, to see how much 
apparent model performance was inflated by data leakage from overlapping windows.

## The leakage mechanism
- Window size = 128, step = 64 -> 50% overlap between consecutive windows
- Window 0 = samples [0,128), Window 1 = samples [64,192)
- Overlap = 64 samples = 50% of each window shared with its neighbor
- In a naive random split, overlapping windows can land on opposite sides of 
  train/test, so "test" data isn't truly unseen.

## Naive random split (Day 2) vs Time-block split (Day 3)

| Model | Split | Train acc | Test acc |
|---|---|---|---|
| Logistic Regression | Naive random | 0.6559 | 0.5957 |
| Logistic Regression | Time-block | 0.6703 | 0.4348 |
| Random Forest | Naive random | 1.0 | 0.8298 |
| Random Forest | Time-block | 1.0 | 0.587 |

## Time-block split method
- Train = first ~80% of windows (in time order)
- Test = last ~20% of windows
- Buffer of 1 window dropped at the boundary to fully remove overlap leakage

## Key interpretation
- Random forest's test accuracy dropped further (0.83 -> 0.587) than logistic 
  regression's (0.60 -> 0.43) once leakage was removed.
- Reason: random forest is a high-capacity, flexible model that can build very 
  specific split rules, letting it exploit near-duplicate overlapping windows 
  more effectively than a simple linear model can. More of its naive-split 
  "skill" was actually memorization, so it had more to lose.
- Logistic regression's time-block test accuracy (0.43) fell below the majority- 
  class baseline (~54%), suggesting the last time-block of the recording may be 
  non-stationary (signal statistics drift over time - electrode drift, fatigue, 
  attention lapses), not just "harder" in a generic sense.

## Critical limitation: time-block split != subject-wise split
- Subject-wise split: train/test on different PEOPLE entirely -> tests 
  generalization to a new brain. This is the gold standard for neuro-signal ML.
- Time-block split: train/test on different TIME segments of the SAME person 
  -> tests generalization to a later moment, not a new subject.
- This dataset has only 1 subject (no subject_id column), so a true subject-wise 
  split isn't possible here. Time-block split is a partial, weaker substitute 
  that only addresses temporal leakage, not cross-subject generalization.
- A multi-subject dataset (e.g., EEG Alcoholic/Control) would allow a real 
  subject-wise split later in the week if we switch datasets.
