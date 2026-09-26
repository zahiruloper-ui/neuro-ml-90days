# Week 8 - Day 6: Refactor (Common EEG Pipeline & Regression Check)

**Date:** 2026-09-26
**Scripts/Module:** `week8/eeg_pipeline.py`, `week8/day6_refactor.py`
**Goal:** Refactor the EEG windowed feature + model comparison logic into a reusable module, then regression-test it to confirm it reproduces Day 3–4 RF vs MLP results.

---

## 1. What we refactored

We moved from ad-hoc scripts to a **common pipeline module**:

Module: `eeg_pipeline.py`

- Centralizes:
  - Loading the EEG Eye State CSV.
  - Creating sliding windows.
  - Extracting mean/std features.
  - Building blocked time-series splits with a buffer.
  - Training and evaluating RF and MLP across folds.

Key components:

- `DatasetConfig`: CSV path, window size, step.
- `SplitConfig`: number of blocks, buffer size.
- `ModelConfig`: RF and MLP hyperparameters (n_estimators, hidden sizes, activation, max_iter, random states).
- Functions:
  - `load_eeg_dataset(cfg)`.
  - `create_sliding_windows(X, y, window_size, step)`.
  - `extract_features(Xw)`.
  - `make_block_indices(n_windows, n_blocks)`.
  - `blocked_time_splits(n_windows, block_indices, buffer)`.
  - `build_models(cfg)`.
  - `run_blocked_cv(X_features, y_windows, splits, cfg)`.
  - `summarize_results(results)`.

Script: `day6_refactor.py`

- Uses the module to:
  - Load data and create windows/features.
  - Build blocked splits.
  - Run RF and MLP across folds.
  - Print per-fold and mean metrics.

---

## 2. Regression check: pipeline vs earlier scripts

Output from `day6_refactor.py` using `DatasetConfig(csv_path="data/eeg-eye-state.csv")`:

- `X_raw shape: (14980, 14)`.
- Total windows: 233.
- Window label counts: `[130 open, 103 closed]`.

Blocked layout:

- Block 0: windows 0–46, labels `[24 open, 23 closed]`.
- Block 1: windows 47–93, labels `[21 open, 26 closed]`.
- Block 2: windows 94–140, labels `[9 open, 38 closed]`.
- Block 3: windows 141–186, labels `[32 open, 14 closed]`.
- Block 4: windows 187–232, labels `[44 open, 2 closed]`.

Blocked splits sizes:

- Fold 0: train=185, test=47.
- Fold 1: train=184, test=47.
- Fold 2: train=184, test=47.
- Fold 3: train=185, test=46.
- Fold 4: train=186, test=46.

These match the layout used in Days 3–4.

Per-fold metrics (RF vs tuned MLP):

- Fold 0: RF acc=0.574, bal_acc=0.570; MLP acc=0.553, bal_acc=0.553.
- Fold 1: RF acc=0.511, bal_acc=0.530; MLP acc=0.383, bal_acc=0.387.
- Fold 2: RF acc=0.213, bal_acc=0.471; MLP acc=0.191, bal_acc=0.458.
- Fold 3: RF acc=0.391, bal_acc=0.462; MLP acc=0.217, bal_acc=0.196.
- Fold 4: RF acc=0.609, bal_acc=0.557; MLP acc=0.609, bal_acc=0.795.

Mean metrics:

- RF mean accuracy ≈ 0.460.
- RF mean balanced accuracy ≈ 0.518.
- MLP mean accuracy ≈ 0.391.
- MLP mean balanced accuracy ≈ 0.478.

These match the Day 3–4 results, confirming the refactor did not change behavior.

---

## 3. Benefits of the refactor

### Cleaner organization

- All core pipeline logic lives in one module instead of being duplicated across scripts.
- Day-specific scripts (`day3_blocked_compare.py`, `day4_error_analysis.py`, `day6_refactor.py`) call the same functions, reducing copy-paste and potential inconsistencies.

### Easier experimentation

- Changing hyperparameters (e.g., MLP hidden sizes, RF depth) is centralized in `ModelConfig`.
- Adjusting block counts or buffer size is centralized in `SplitConfig`.
- Switching between local CSV and raw URL is a single line change in `DatasetConfig`.

### Safer evolution

- Regression check shows the refactor preserves previous results.
- Future additions (e.g., alternative models, different feature sets) can plug into the same pipeline without rewriting the entire workflow.

---

## 4. Practical refactor habits reinforced

Habits reflected in Day 6:

- **Modularization:** Separate reusable logic (module) from day-specific scripts.
- **Configuration via dataclasses:** Avoid hard-coding values; use config objects for paths and hyperparameters.
- **Regression testing:** Always re-run key scripts and compare metrics after refactors to catch silent behavior changes.
- **Separation of concerns:**
  - Data loading and preprocessing in one place.
  - Splitting logic in another.
  - Model training and evaluation in dedicated functions.

These habits scale beyond this EEG project to other ML pipelines.
