# Week 8 — Intro Neural Nets (EEG Eye State)

**Date:** 2026-09-26  
**Module:** `week8/eeg_pipeline.py`  
**Scripts:** `week8/day1/day1_mlp_baseline.py`, `week8/day2/day2_mlp_tune_sweep.py`, `week8/day3/day3_blocked_compare.py`, `week8/day4/day4_error_analysis.py`, `week8/day6/day6_refactor.py`  
**Notes:** See `week8/notes/day1_notes.md` … `week8/notes/day7_notes.md`.

---

## Week goal

Extend the Week 7 EEG Eye State project by:

- building a small Multi-Layer Perceptron (MLP) on the SAME windowed features.
- using leakage-safe, time-aware splits (time-block with buffer, then blocked CV).
- comparing the MLP fairly against the best Week 7 Random Forest.

The aim is to understand what an MLP actually is, how it behaves on small tabular EEG data, and how evaluation design changes the RF vs MLP comparison.

---

## Dataset

- **Source:** EEG Eye State dataset.
- **Shape:** 14,980 samples, 14 EEG channels + 1 label.
- **Labels:**
  - `0` = eye open.
  - `1` = eye closed.

Week 8 reuses the same dataset as Week 7, treated as a single-subject time series.

---

## Pipeline overview

### 1. Windowing

- Sliding windows:
  - window size = 128 samples.
  - step = 64 samples (50% overlap).
- Per-window label:
  - majority eye state in the window (mean of label samples > 0.5 -> closed).

Result:

- 233 windows.
- Window label counts: 130 open, 103 closed.

### 2. Features

For each window:

- compute mean per channel.
- compute standard deviation per channel.
- concatenate into a 28-D feature vector:
  - 14 means + 14 standard deviations.

These keep the problem tabular and comparable between RF and MLP.

### 3. Splitting

Two protocols are used:

1. **Day 1 — Single time-block split:**

   - train on the first ~80% of windows in time order.
   - drop a 1-window buffer at the boundary.
   - test on the last ~20% of windows.

2. **Days 3–4 — Blocked time-series CV:**

   - split 233 windows into 5 contiguous blocks.
   - for each block:
     - test set = that block.
     - train set = all other windows, with a 1-window buffer around the test block.

Blocked CV provides multiple test segments (early, mixed, closed-heavy, tail) and a more stable average performance.

### 4. Scaling

- Random Forest:
  - trained on raw features (scale-invariant).

- MLP:
  - uses `StandardScaler` fit on train only per fold.
  - transforms train and test with the same scaler.

This preserves leakage safety and stabilizes gradient-based optimization.

---

## Models

### Random Forest (RF)

- `sklearn.ensemble.RandomForestClassifier`.
- Typical Week 8 config:
  - `n_estimators = 200`.
  - `max_depth = None`.
  - `random_state = 42`.

### Multi-Layer Perceptron (MLP)

- `sklearn.neural_network.MLPClassifier`.
- Final tuned config:
  - `hidden_layer_sizes = (64, 32)`.
  - `activation = 'tanh'`.
  - `solver = 'adam'`.
  - `max_iter = 500`.
  - `random_state = 42`.

The MLP runs on the same 28-D windowed features and uses the same splits as RF.

---

## Common module: `week8/eeg_pipeline.py`

Centralizes the pipeline logic:

- `DatasetConfig` — CSV path, window size, step.
- `SplitConfig` — number of blocks, buffer size.
- `ModelConfig` — RF and MLP hyperparameters.

Functions:

- `load_eeg_dataset(cfg)` — read CSV and return `(X_raw, y_raw)`.
- `create_sliding_windows(X, y, window_size, step)` — build `Xw`, `yw`.
- `extract_features(Xw)` — mean/std features per window.
- `make_block_indices(n_windows, n_blocks)` — contiguous blocks of window indices.
- `blocked_time_splits(n_windows, block_indices, buffer)` — leakage-safe (train, test) indices.
- `build_models(cfg)` — instantiate RF and MLP.
- `run_blocked_cv(X_features, y_windows, splits, cfg)` — train/evaluate RF and MLP per fold.
- `summarize_results(results)` — mean accuracy and balanced accuracy for RF and MLP.

Day-specific scripts import this module instead of reimplementing the pipeline.

---

## Scripts per day

### Day 1 — Baseline MLP

- `week8/day1/day1_mlp_baseline.py`.
- Rebuilds Week 7 pipeline on the single time-block split.
- Trains baseline MLP (32 hidden units, ReLU, 500 iters).
- Prints train/test accuracy and confusion matrix.

### Day 2 — Hyperparameter sweep

- `week8/day2/day2_mlp_tune_sweep.py`.
- Sweeps:
  - hidden sizes: `(16,)`, `(32,)`, `(64,)`, `(32, 32)`, `(64, 32)`.
  - activations: `relu`, `tanh`.
  - `max_iter`: `500`, `2000`.
- Reports train/test accuracy and balanced accuracy for each config.

### Day 3 — Blocked RF vs MLP comparison

- `week8/day3/day3_blocked_compare.py`.
- Implements blocked time-series CV with 5 folds.
- Trains RF and tuned MLP per fold.
- Prints per-fold metrics and average metrics.

### Day 4 — Error analysis

- `week8/day4/day4_error_analysis.py`.
- Uses blocked splits.
- Counts per window:
  - `both_correct`, `both_wrong`, `RF_only_correct`, `MLP_only_correct`.
- Prints per-fold and global agreement/disagreement.

### Day 6 — Refactor regression check

- `week8/day6/day6_refactor.py`.
- Uses `eeg_pipeline.py` to reproduce blocked CV metrics.
- Confirms refactored pipeline matches earlier results.

(Concept notes and flashcards live under `week8/notes/` for Days 1–7.)

---

## Results (blocked CV summary)

Using blocked time-series CV:

- **Random Forest**
  - mean accuracy ≈ 0.460.
  - mean balanced accuracy ≈ 0.518.

- **Tuned MLP** (hidden=(64,32), tanh)
  - mean accuracy ≈ 0.391.
  - mean balanced accuracy ≈ 0.478.

Fold-level patterns:

- RF is stronger on several folds (especially mixed segments).
- Both models struggle on the closed-heavy block.
- On the tail block (44 open, 2 closed), MLP matches RF's accuracy and achieves higher balanced accuracy by correctly classifying both closed windows.

Interpretation:

- RF remains a robust baseline on this small tabular EEG dataset.
- The tuned MLP adds value in specific regimes (e.g., tail segments with rare closed windows).
- RF and MLP are best seen as complementary:
  - RF as the primary classifier.
  - MLP as a specialist to improve recall on certain windows.

---

## How to run

From repo root, with `.venv` active:

- Install dependencies:

  ```bash
  .venv\Scripts\python.exe -m pip install scikit-learn pandas numpy
  ```

- Baseline MLP (Day 1):

  ```bash
  .venv\Scripts\python.exe week8\day1\day1_mlp_baseline.py
  ```

- MLP hyperparameter sweep (Day 2):

  ```bash
  .venv\Scripts\python.exe week8\day2\day2_mlp_tune_sweep.py
  ```

- Blocked RF vs MLP comparison (Day 3):

  ```bash
  .venv\Scripts\python.exe week8\day3\day3_blocked_compare.py
  ```

- Error analysis across models (Day 4):

  ```bash
  .venv\Scripts\python.exe week8\day4\day4_error_analysis.py
  ```

- Refactor regression check (Day 6):

  ```bash
  .venv\Scripts\python.exe week8\day6\day6_refactor.py
  ```

Use `week8/notes/` files for detailed explanations and flashcards per day.

---

## Dependencies

- Python 3.10+.
- `scikit-learn`.
- `pandas`.
- `numpy`.