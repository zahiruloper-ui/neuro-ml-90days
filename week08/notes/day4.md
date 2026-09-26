# Week 8 - Day 4: Analyze (RF vs MLP Error Analysis)

**Date:** 2026-09-26
**Script:** `week8/day4/day4_error_analysis.py`
**Dataset:** EEG Eye State (14 channels, 14,980 samples, single subject)
**Goal:** Analyze how Random Forest and the tuned MLP agree or disagree on individual EEG windows across blocked time-series folds.

---

## 1. Setup recap

Same data and features as Days 1–3:

- Windows: size 128, step 64.
- Per-window label: majority eye state (mean of samples > 0.5 -> eye-closed).
- Features: 14 means + 14 stds = 28 features per window.
- Total windows: 233.
- Window labels: `[130 open, 103 closed]`.

Blocked time-series CV (from Day 3):

- 5 contiguous blocks of windows (indices):
  - Block 0: windows 0–46, labels `[24 open, 23 closed]`.
  - Block 1: windows 47–93, labels `[21 open, 26 closed]`.
  - Block 2: windows 94–140, labels `[9 open, 38 closed]`.
  - Block 3: windows 141–186, labels `[32 open, 14 closed]`.
  - Block 4: windows 187–232, labels `[44 open, 2 closed]`.
- For each fold:
  - Test set = one block.
  - Train set = all other windows, removing a buffer of 1 window on each side of the test block.

Models:

- Random Forest (RF): `RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)`.
- Tuned MLP: `MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", solver="adam", max_iter=500, random_state=42)`.
- RF uses raw features; MLP uses scaled features (`StandardScaler` fit on train only per fold).

---

## 2. Agreement/disagreement definitions

For each test window in each fold, we classify into four cases:

- `both_correct`: RF and MLP predictions match the true label.
- `both_wrong`: RF and MLP predictions both differ from the true label.
- `RF_only_correct`: RF prediction equals the true label; MLP prediction is wrong.
- `MLP_only_correct`: MLP prediction equals the true label; RF prediction is wrong.

We count these per fold and globally across all folds.

---

## 3. Per-fold agreement/disagreement

### Fold 0 (Block 0: 24 open / 23 closed)

- RF: acc ≈ 0.574, bal_acc ≈ 0.570.
- MLP: acc ≈ 0.553, bal_acc ≈ 0.553.
- Counts:
  - `both_correct = 18`.
  - `both_wrong = 12`.
  - `RF_only_correct = 9`.
  - `MLP_only_correct = 8`.

Interpretation:

- Both models perform similarly on this balanced segment.
- RF rescues slightly more windows than MLP (9 vs 8), but they are close.

### Fold 1 (Block 1: 21 open / 26 closed)

- RF: acc ≈ 0.511, bal_acc ≈ 0.530.
- MLP: acc ≈ 0.383, bal_acc ≈ 0.387.
- Counts:
  - `both_correct = 14`.
  - `both_wrong = 19`.
  - `RF_only_correct = 10`.
  - `MLP_only_correct = 4`.

Interpretation:

- RF is clearly stronger here.
- RF-only-correct windows (10) far outnumber MLP-only-correct windows (4).
- MLP frequently disagrees with RF and is wrong more often in those disagreements.

### Fold 2 (Block 2: 9 open / 38 closed)

- RF: acc ≈ 0.213, bal_acc ≈ 0.471.
- MLP: acc ≈ 0.191, bal_acc ≈ 0.458.
- Counts:
  - `both_correct = 8`.
  - `both_wrong = 36`.
  - `RF_only_correct = 2`.
  - `MLP_only_correct = 1`.

Interpretation:

- This block is extremely hard for both models.
- Most closed windows are misclassified by both; RF rescues 2, MLP rescues 1.
- The error pattern is nearly identical; neither model handles this regime well with current features.

### Fold 3 (Block 3: 32 open / 14 closed)

- RF: acc ≈ 0.391, bal_acc ≈ 0.462.
- MLP: acc ≈ 0.217, bal_acc ≈ 0.196.
- Counts:
  - `both_correct = 4`.
  - `both_wrong = 22`.
  - `RF_only_correct = 14`.
  - `MLP_only_correct = 6`.

Interpretation:

- RF significantly outperforms MLP here.
- RF rescues over twice as many windows as MLP (14 vs 6).
- MLP struggles to capture the patterns in this mixed segment.

### Fold 4 (Block 4: tail, 44 open / 2 closed)

- RF: acc ≈ 0.609, bal_acc ≈ 0.557.
- MLP: acc ≈ 0.609, bal_acc ≈ 0.795.
- Counts:
  - `both_correct = 19`.
  - `both_wrong = 9`.
  - `RF_only_correct = 9`.
  - `MLP_only_correct = 9`.

Interpretation:

- RF and MLP have identical accuracy (0.609) and equal numbers of "only correct" windows.
- Balanced accuracy is much higher for MLP because:
  - MLP correctly classifies both closed windows.
  - RF misclassifies one closed window that MLP gets right.
- MLP also sometimes corrects RF's false positives on open windows in this tail block.

---

## 4. Global agreement/disagreement

Global counts across all folds:

- `both_correct = 63`.
- `both_wrong = 98`.
- `RF_only_correct = 44`.
- `MLP_only_correct = 28`.
- `total_test_windows = 233`.

Fractions:

- `frac_both_correct ≈ 0.27`.
- `frac_both_wrong ≈ 0.421`.
- `frac_RF_only_correct ≈ 0.189`.
- `frac_MLP_only_correct ≈ 0.12`.

Interpretation:

- About 27% of windows are correctly handled by both models.
- About 42% of windows are misclassified by both; these are genuinely hard under the current feature set.
- When RF and MLP disagree, RF is right more often:
  - RF-only-correct windows (44) vs MLP-only-correct windows (28).
  - RF rescues more of MLP's mistakes than MLP rescues RF's.

This supports seeing RF as the more reliable baseline overall.

---

## 5. Detailed tail-fold behavior (Fold 4)

Tail fold rows (simplified view):

- Closed windows (true = 1):
  - `win_idx = 187`: RF = 1, MLP = 1 -> both_correct.
  - `win_idx = 222`: RF = 0, MLP = 1 -> MLP_only_correct.
- Many open windows (true = 0) show patterns like:
  - RF = 0, MLP = 0 -> both_correct.
  - RF = 1, MLP = 1 -> both_wrong (both predicting closed on an open window).
  - RF = 0, MLP = 1 -> RF_only_correct (MLP false positive).
  - RF = 1, MLP = 0 -> MLP_only_correct (RF false positive that MLP corrects).

This tail segment highlights the tuned MLP's role:

- It **never misses a closed window** in this fold.
- It sometimes corrects RF's false positives on open windows.
- RF remains more conservative overall but can miss rare closed windows.

---

## 6. Model roles and complementary strengths

From Days 3–4 combined:

- Random Forest (RF):
  - Stronger average performance across folds (higher mean accuracy and balanced accuracy).
  - More often correct when the two models disagree (RF-only-correct > MLP-only-correct).
  - Robust baseline for this EEG windowed classification task.

- Tuned MLP:
  - Underperforms RF on several blocks, especially mixed segments.
  - Provides value in **specific regimes**, particularly the tail segment where catching rare closed windows is important.
  - Acts as a "closed-window specialist" that can complement RF in an ensemble or a parallel-check setup.
