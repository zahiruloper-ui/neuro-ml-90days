# Week 7 — Day 4: Analyze

## Goal
Perform error analysis on the time-block split results, checking WHERE and 
WHY the model fails, since this single-subject dataset has no subject IDs 
(time segments stand in for "subjects" here).

## Per-window error inspection
- Test block: 46 windows, 19 errors (matches Day 3 confusion matrix)
- True label distribution in test block: 42 eye-open, only 4 eye-closed 
  (heavily skewed vs. training block)
- ALL 19 errors were false positives (true=open, predicted=closed)
- ZERO false negatives - all 4 true eye-closed windows correctly identified

## Interpretation: boundary shift, not fixed bias
- The one-directional error pattern (only false positives) suggests the 
  model's decision boundary is not fundamentally broken - it correctly 
  identifies eye-closed when it truly occurs.
- Instead, eye-open windows in this later test block appear to have drifted 
  in mean/std feature space toward what eye-closed windows looked like during 
  training - consistent with non-stationarity (Day 3 finding).

## Rolling accuracy (window=5) - burst pattern, not steady decline
- High-accuracy patches: test idx 16-19 (0.80-1.00), idx 40-42 (0.80-1.00)
- Low-accuracy patches: test idx 22-24 (0.20-0.40), idx 44-45 (0.40)
- Errors oscillate in short bursts rather than forming one continuous bad 
  stretch - suggests intermittent artifacts (e.g., blinks, brief movements) 
  rather than one smooth, permanent drift event.

## Key caveat: feature choice vs. data quality are separate problems
- Bandpower (frequency-domain) features help capture the TRUE physiological 
  signal (alpha power) that distinguishes eye states in general.
- However, bandpower is still computed per window from the same raw samples, 
  so brief artifacts within a window can distort frequency content just as 
  much as time-domain stats - switching feature type does not fix noisy/
  contaminated raw data.
- A more direct fix for bursty errors: artifact detection/rejection per 
  window, or smoothing predictions using neighboring windows.

## Single-subject limitation carried from Day 3
- True subject-wise error analysis (comparing hard vs. easy SUBJECTS) isn't 
  possible here - only time-block-wise error analysis was possible, since 
  this dataset has one continuous recording from one person.