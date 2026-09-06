# Week 7 — Day 1: Segment

## Goal
Load an EEG time-series dataset, segment it into sliding windows, and extract 
simple features (mean, std) per window as a baseline for classification.

## Dataset
- EEG Eye State dataset (UCI/OpenML, via GitHub CSV)
- 14,980 time samples, 14 EEG channels (AF3, F7, F3, FC5, T7, P, O1, O2, P8, T8, FC6, F4, F8, AF4)
- Binary label `class`: 0 = eye-open, 1 = eye-closed
- Row-level label distribution: 8,257 (open) vs 6,723 (closed)

## Sliding windows
- Window size: 128 time samples
- Step size: 64 samples (50% overlap)
- Formula for number of windows: floor((n_samples - window_size) / step) + 1
- Result: 233 windows, shape (233, 128, 14)
- Window label = label at the last time point in the window
- Window-level label distribution: 126 (open) vs 107 (closed)

## Why sliding windows?
- Raw EEG is sampled far too fast to classify one time point at a time meaningfully.
- Windows group nearby time points into one example, matching the timescale at 
  which brain states (like eye-open/closed) actually change.
- Overlapping windows (via step < window_size) let you generate more training 
  examples from the same amount of raw data, at the cost of introducing correlated 
  (non-independent) samples — important to remember later when splitting data.

## Feature extraction (baseline)
- mean per channel per window: captures average signal level
- std per channel per window: captures variability/energy in that window
- Combined feature vector per window: 28 features (14 mean + 14 std)
- Mean and std are a reasonable *baseline* but likely insufficient alone, because 
  EEG differences between eye-open/closed are often frequency-based (e.g., alpha 
  band power increases with eyes closed — the "Berger effect"), not just amplitude-based.

## Key caveat
- Time-domain stats (mean, std) don't capture frequency content.
- Bandpower features (via FFT or filtering) are expected to be more discriminative 
  for this kind of EEG task, and are on the list for later in the week if feasible.
