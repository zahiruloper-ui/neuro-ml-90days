# Week 7 — Day 5: Recall

## Goal
Active recall review of Week 7's core concepts: data leakage, autocorrelation, 
split strategies, and time-series pitfalls, consolidating Days 1-4.

## Key concepts reviewed

### Data leakage
- Information from outside the proper training set improperly influences 
  training/evaluation, making the model look better than it truly generalizes.

### Autocorrelation and random splits
- Time-series data has autocorrelation: nearby time points are statistically 
  similar. Random splits ignore time order, so train/test samples that are 
  close in time can end up on opposite sides of the split, inflating scores.

### Overlapping windows = concrete leakage source
- Step size (64) < window size (128) meant consecutive windows shared 50% of 
  raw samples. If split randomly, "test" windows could be half-identical to 
  training windows.

### Naive random split vs time-block split
- Naive: shuffles all windows randomly, ignoring time order - leakage risk.
- Time-block: train = earlier time chunk, test = later time chunk (with buffer 
  to remove boundary overlap) - respects time order, avoids window-overlap leakage.

### Subject-wise split (gold standard for multi-subject data)
- Assigns entire subjects to train OR test, never both.
- Tests generalization to a NEW PERSON's brain - the real-world requirement 
  for most clinical/practical EEG applications.
- Our single-subject dataset couldn't use this; time-block was the closest 
  available substitute (tests generalization across time, not across subjects).

### Why random forest degraded more than logistic regression under leakage removal
- Random forest = high-capacity/flexible -> can "memorize" near-duplicate 
  overlapping windows under naive split.
- Logistic regression = simple linear boundary -> can't memorize the same way.
- Removing leakage exposed more of RF's inflated performance as memorization.

### Error analysis findings (Day 4)
- All errors were false positives, zero false negatives -> boundary itself 
  wasn't broken; a specific class (eye-open) drifted to resemble the other 
  class's training patterns (feature drift, not general confusion).
- Bursty rolling-accuracy pattern (short dips, not steady decline) -> pointed 
  to intermittent artifacts (blinks, movement) rather than one smooth, 
  permanent drift.

### Feature choice vs. data quality
- Switching to frequency-domain features (bandpower) does NOT automatically 
  fix noise/artifacts within a window - both time- and frequency-domain 
  features are computed from the same raw (possibly corrupted) samples.
- Fixing noisy data requires separate steps: artifact detection/rejection, 
  or smoothing across neighboring windows.
