# Week 8 - Day 3: Compare (RF vs MLP with Blocked Time-Series CV)

**Date:** 2026-09-26
**Script:** `week8/day3/day3_blocked_compare.py`
**Dataset:** EEG Eye State (14 channels, 14,980 samples, single subject)
**Goal:** Fairly compare Random Forest vs the tuned MLP using blocked time-series cross-validation on the SAME EEG windowed features, with leakage-safe splits.

---

## 1. Data and feature pipeline (unchanged)

Same as Days 1–2:

- Windows: size 128, step 64.
- Per-window label: majority eye state (mean of samples > 0.5 -> eye-closed).
- Features: 14 channel means + 14 channel standard deviations = 28 features per window.
- Total windows: 233.
- Window labels: `[130 open, 103 closed]`.

Preprocessing:

- Scaling: `StandardScaler` fit on training windows only in each fold, then `transform` applied to train and test within that fold.
- RF uses raw features (scale-invariant); MLP uses scaled features.

---

## 2. Blocked time-series cross-validation

### Why blocked CV

Random k-fold CV would mix correlated windows (nearby in time) into both train and test, leaking information and inflating scores.
Blocked CV respects temporal dependence:

- Test sets are contiguous blocks of windows.
- Training sets contain windows strictly before and after each test block.
- A buffer (embargo) around the test block is removed from training to avoid overlap.

### Block construction

- Number of blocks: `N_BLOCKS = 5`.
- We call `np.array_split(np.arange(233), 5)` to create 5 contiguous window index blocks.
- Each block becomes a test set once; the remaining windows (minus the buffer) form the training set.

Blocked layout example (rough):

- Block 0: early windows, mix of open/closed.
- Block 1: slightly later segment, mixed.
- Block 2: stretch with many closed windows (hard block).
- Block 3: another mixed region.
- Block 4: late tail (the 44-open, 2-closed region seen on Day 1).

For each fold:

- `test_idx = block_indices[i]`.
- `train_idx = all other indices` where index < min(block) - buffer or > max(block) + buffer.

---

## 3. Models compared

Random Forest (RF):

- `RandomForestClassifier` with `n_estimators=200`, `max_depth=None`, `random_state=42`.
- Trained on raw features `X_train` per fold.

Tuned MLP:

- `MLPClassifier` with:
  - `hidden_layer_sizes=(64, 32)`.
  - `activation="tanh"`.
  - `solver="adam"`.
  - `max_iter=500`, `random_state=42`.
- Trained on scaled features `X_train_s` per fold.

Metrics per fold:

- Accuracy.
- Balanced accuracy (average of per-class recall).
- Confusion matrix on test windows.

---

## 4. Fold-by-fold behavior

### Fold 0 (early segment)

- RF: acc ≈ 0.574, bal_acc ≈ 0.570.
  - Confusion: `[[19  5],[15  8]]`.
- MLP: acc ≈ 0.553, bal_acc ≈ 0.553.
  - Confusion: `[[13 11],[10 13]]`.

Interpretation:

- Both models perform reasonably well.
- RF predicts eye-closed less often on open windows (5 false positives vs 11 for MLP).
- MLP catches more closed windows (13 vs 8), trading higher recall on the closed class for more false alarms on the open class.

### Fold 1

- RF: acc ≈ 0.511, bal_acc ≈ 0.530.
  - Confusion: `[[15  6],[17  9]]`.
- MLP: acc ≈ 0.383, bal_acc ≈ 0.387.
  - Confusion: `[[ 9 12],[17  9]]`.

Interpretation:

- Both models struggle with this segment.
- RF maintains slightly better balanced accuracy; MLP misclassifies more open windows as closed and does not compensate with better recall on closed windows.

### Fold 2 (hard block)

- RF: acc ≈ 0.213, bal_acc ≈ 0.471.
  - Confusion: `[[ 8  1],[36  2]]`.
- MLP: acc ≈ 0.191, bal_acc ≈ 0.458.
  - Confusion: `[[ 8  1],[37  1]]`.

Interpretation:

- This block is extremely difficult for both models: 36–37 closed windows misclassified.
- RF catches 2 closed windows; MLP catches 1.
- Balanced accuracy is low for both; this segment likely has patterns that neither model can express well with the current features.

### Fold 3

- RF: acc ≈ 0.391, bal_acc ≈ 0.462.
  - Confusion: `[[ 9 23],[ 5  9]]`.
- MLP: acc ≈ 0.217, bal_acc ≈ 0.196.
  - Confusion: `[[ 8 24],[12  2]]`.

Interpretation:

- RF clearly outperforms MLP here.
- MLP catches only a small fraction of closed windows (2 vs RF's 9) and misclassifies many open windows, leading to very low balanced accuracy.

### Fold 4 (tail segment)

- RF: acc ≈ 0.609, bal_acc ≈ 0.557.
  - Confusion: `[[27 17],[ 1  1]]`.
- MLP: acc ≈ 0.609, bal_acc ≈ 0.795.
  - Confusion: `[[26 18],[ 0  2]]`.

Interpretation:

- Accuracy is identical (0.609) for RF and MLP.
- RF misses 1 of 2 closed windows (recall for closed = 0.5).
- MLP catches **both** closed windows (recall for closed = 1.0), at the cost of one extra false positive on open windows.
- Balanced accuracy is much higher for MLP here (~0.795 vs ~0.557); this is the fold where the tuned MLP clearly wins for the minority class.

---

## 5. Average metrics and global comparison

Across all 5 folds:

- RF mean accuracy ≈ 0.460.
- RF mean balanced accuracy ≈ 0.518.
- MLP mean accuracy ≈ 0.391.
- MLP mean balanced accuracy ≈ 0.478.

Global verdict:

- On average, RF is the stronger model on this EEG windowed dataset:
  - Higher mean accuracy (~0.46 vs ~0.39).
  - Higher mean balanced accuracy (~0.52 vs ~0.48).
- The tuned MLP has **one segment (fold 4)** where it is clearly better for the closed class, but it underperforms RF on several other segments.

Key insight:

- RF remains a robust baseline on small tabular neuro data.
- The MLP can act as a "specialist" for certain regimes (e.g., late tail where closed windows are rare but important), but its overall generalization is more fragile.

---

## 6. What blocked CV changed compared to Day 1

Day 1 used a single tail split:

- Test block: last 46 windows (44 open, 2 closed).
- Verdict: RF > baseline MLP on both accuracy and balanced accuracy.

Blocked CV:

- Evaluates models on **multiple contiguous segments** with their own label mixes.
- Shows that some folds are easy for both models, some are very hard, and one fold is especially favorable to the tuned MLP.
- Provides **average metrics** that are less sensitive to the quirks of a single 2-closed-window tail.

Main lessons:

- RF's advantage is robust across folds, not just on one split.
- The tuned MLP is useful in specific segments (particularly when catching closed windows is critical), but it does not surpass RF overall.
- Honest time-aware evaluation (blocked CV with buffer) is essential to understand a model's behavior over time, beyond a single train/test split.
