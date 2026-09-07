# Week 7 — Day 6: Refactor

## Goal
Convert Days 1-4's copy-pasted scripts into clean, reusable functions: 
loading, windowing, feature extraction, splitting, training/evaluation.

## Functions built

### load_eeg_data(url)
- Returns X (n_samples, n_channels), y (n_samples,)
- Verified: (14980, 14) - matches Day 1

### create_sliding_windows(X, y, window_size, step)
- Returns X_windows (n_windows, window_size, n_channels), y_windows (n_windows,)
- Verified: (233, 128, 14) and [126, 107] label distribution - matches Day 1

### extract_features(X_windows)
- Returns X_features (n_windows, n_channels*2) - mean + std per channel
- Verified: (233, 28) - matches Day 1/2/3

### time_block_split(X_features, y_windows, test_fraction, buffer)
- Returns X_train, y_train, X_test, y_test, respecting time order + buffer
- Verified: train (185, 28), test (46, 28) - consistent with Day 3 logic

### train_and_evaluate(model, X_train, y_train, X_test, y_test, scale)
- Works with ANY sklearn-style classifier
- Returns dict: {train_acc, test_acc, confusion_matrix}
- Verified against both LogReg (scale=True) and RF (scale=False):
  - LogReg: train 0.6703, test 0.4348, cm=[[17,25],[1,3]]
  - RF: train 1.0, test 0.587, cm=[[23,19],[0,4]]
  - Both match Day 3 exactly (RF), LogReg confusion matrix is new detail: 
    also shows heavy false-positive bias, confirming Day 4's finding applies 
    to BOTH models, not just random forest - it's a property of the drifted 
    data, not one specific model.

## Key lesson: refactoring improves code, not data/model problems
- Easier to reuse: swap models, window sizes, split ratios, buffer values 
  with one argument instead of duplicating code blocks.
- Does NOT fix: leakage risk (still depends on correct buffer value), 
  non-stationarity in the EEG signal (same accuracy/errors as before), 
  single-subject limitation (still can't do true subject-wise split).
- Refactoring is a software engineering improvement, not a modeling fix.

## Regression testing habit
- After each refactor step, re-ran the code and compared against known-correct 
  numbers from earlier days (14980 raw samples, 233 windows, etc.) to confirm 
  behavior didn't change - a "regression test," standard practice before 
  trusting refactored code.
