# Week 7 — Day 2: Classify

## Goal
Train baseline classifiers (logistic regression, random forest) on window-level 
EEG features (mean, std) and evaluate performance with a naive train/test split.

## Setup
- Features: X_features (233, 28) — mean + std per channel per window
- Labels: y_windows (233,) — eye-open (0) / eye-closed (1)
- Split: naive random 80/20 (train_test_split, random_state=42)
- Train shape: (186, 28), Test shape: (47, 28)

## Logistic Regression results
- Train accuracy: 0.6559
- Test accuracy: 0.5957
- Confusion matrix: TN=20, FP=8, FN=11, TP=8
- Eye-open recall: ~71%, Eye-closed recall: ~42%
- Interpretation: model is biased toward predicting eye-open; mean/std (time-domain 
  only) likely miss the frequency-domain signature of eye-closed states (alpha band 
  power increase — the Berger effect), so linear separation is weak.

## Random Forest results
- Train accuracy: 1.0 (perfect — sign of overfitting)
- Test accuracy: 0.8298
- Confusion matrix: TN=24, FP=4, FN=4, TP=15
- Eye-open recall: ~86%, Eye-closed recall: ~79%
- Interpretation: random forest captures non-linear feature interactions logistic 
  regression cannot, giving a large accuracy jump and more balanced recall. 
  However, perfect train accuracy is a red flag for overfitting, and the naive 
  random split may leak information via overlapping windows (50% overlap between 
  consecutive windows means train/test windows can share raw time samples).

## Key caveat carried into Day 3
- This is still a NAIVE split — no subject-wise or time-block separation.
- High random forest accuracy might partly reflect leakage, not true generalization.
- Day 3 will directly test this by comparing naive split vs. a leakage-safe split.
