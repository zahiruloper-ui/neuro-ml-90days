# Week 8 - Day 2: Tune (MLP Hyperparameters)

**Date:** 2026-09-26
**Script:** `week8/day2/day2_mlp_tune_sweep.py`
**Dataset:** EEG Eye State (14 channels, 14,980 samples, single subject)
**Goal:** Explore how hidden layer size, number of layers, activation, and training iterations affect MLP behavior on the SAME leakage-safe split from Day 1.

---

## 1. Setup recap (same pipeline as Day 1)

- Windows: size 128, step 64.
- Per-window label: majority eye state (mean of samples > 0.5 -> eye-closed).
- Features: 14 means + 14 stds = 28 features per window.
- Leakage-safe split: time-block with buffer.
  - Train: first ~80% of windows.
  - Buffer: 1 window around the boundary dropped.
  - Test: last ~20% of windows.
- Scaling: `StandardScaler` fit on train only, transform train and test.

Shapes and label counts:

- `X_features`: `(233, 28)`.
- Train: `(185, 28)`, labels `[86 open, 99 closed]`.
- Test: `(46, 28)`, labels `[44 open, 2 closed]`.

**Important:** For Day 2 we kept this pipeline identical. Only the MLP hyperparameters changed.

---

## 2. Hyperparameter grid for Day 2

We swept these configurations:

- Hidden layer sizes:
  - `(16,)` - small single layer.
  - `(32,)` - baseline single layer from Day 1.
  - `(64,)` - wider single layer.
  - `(32, 32)` - two hidden layers, medium width.
  - `(64, 32)` - two hidden layers, wider first layer.
- Activations:
  - `"relu"` - piecewise linear, common default.
  - `"tanh"` - smooth, symmetric around zero.
- `max_iter` (training epochs upper bound):
  - `500` - baseline.
  - `2000` - allow more epochs to reach convergence.

All models used `solver="adam"` and `random_state=42`.

---

## 3. Underfitting vs. high-capacity regimes

### Clearly underfitting configs

Examples:

- `(16,), relu, 500`: train acc ~0.822, test acc ~0.283.
- `(16,), tanh, 500`: train acc ~0.805, test acc ~0.196.
- `(32,), tanh, 500`: train acc ~0.822, test acc ~0.174.

These models have **modest train accuracy** (not fully learned) and **low test accuracy**. The network is too small and/or trained for too few epochs to capture the training patterns.

### High-capacity / overfitting configs

Examples:

- `(32,), relu, 2000`: train acc ~1.000, test acc ~0.391.
- `(64,), tanh, 2000`: train acc ~1.000, test acc ~0.478.
- `(32, 32), relu/tanh, 500 or 2000`: train acc ~1.000, test acc ~0.348-0.370.
- `(64, 32), relu/tanh, 500 or 2000`: train acc ~1.000, test acc ~0.348-0.565.

These models **memorize the training set** (train acc ~1.000) but test accuracy remains in a narrow band (roughly 0.35 to 0.56). Optimization error is solved; generalization error is dominated by the data split.

---

## 4. Metrics: accuracy vs. balanced accuracy

Plain accuracy:

- For this test block (44 open, 2 closed), a constant "always eye-open" predictor would score ~0.957.
- Many tuned MLPs reach only ~0.17-0.56 test accuracy.

Balanced accuracy:

- Balanced accuracy = average of per-class recall.
- This metric is more honest on imbalanced data, because it weights each class equally.
- Day 1 baseline `(32,), relu, 500` had balanced accuracy ~0.386.

In the sweep, balanced accuracy ranged from ~0.386 (baseline) up to ~0.773.

---

## 5. Best tuned MLP (on this split)

### Strongest balanced-accuracy config

- Config: `hidden=(64, 32)`, `activation="tanh"`.
- At both `max_iter=500` and `2000`:
  - Train accuracy: ~1.000.
  - Test accuracy: ~0.565.
  - Balanced accuracy: ~0.773.
  - Confusion matrix (test): `[[24 20], [0 2]]`.

Interpretation:

- True open (class 0): 44 windows.
  - 24 correctly predicted open.
  - 20 incorrectly predicted closed.
  - Recall for class 0 ≈ 24 / 44 ≈ 0.545.
- True closed (class 1): 2 windows.
  - Both correctly predicted closed.
  - Recall for class 1 = 2 / 2 = 1.0.
- Balanced accuracy ≈ (0.545 + 1.0) / 2 ≈ 0.773.

This config **never misses a closed window**, at the cost of more false alarms on open windows. Precision for the closed class is low, but if the priority is "catch all closed windows," this is the best trade-off in the sweep.

### More conservative configs

If we prefer fewer false alarms and are willing to miss some closed windows:

- `(64,), tanh, 2000`: train 1.000, test ~0.478, balanced acc ~0.727.
- `(32, 32), relu`: train 1.000, test ~0.348-0.370, balanced acc ~0.659-0.670.

These configs still raise balanced accuracy compared to the Day 1 baseline, but they are less aggressive about predicting the closed class.

---

## 6. What tuning could and could not fix

### What tuning *did* change

- Solved underfitting for many configs: train accuracy climbed from ~0.8 to ~1.0.
- Changed the **shape of the decision boundary**, reflected in different confusion matrices and balanced accuracy.
- Gave a choice of trade-offs: catch more closed windows vs. reduce false positives, maximize balanced accuracy vs. raw accuracy.

### What tuning *did not* change

- The **train/test distribution shift**:
  - Train labels: `[86 open, 99 closed]` (near-balanced).
  - Test labels: `[44 open, 2 closed]` (strongly skewed open).
- The **extreme class imbalance** in the test block.
- The fact that the last part of the recording behaves differently from the earlier part.

No configuration in the sweep fully "solved" the split problem; none approached the 0.957 oracle or matched Random Forest's 0.609 test accuracy on this split.

> **Key takeaway:** Hyperparameter tuning can change **how** the MLP fails on this split, but it
> cannot eliminate the underlying distribution shift and imbalance. Tuning mainly gives you
> different trade-offs rather than a fundamental fix.

This is why the plan for later in Week 8 is to also adjust the **evaluation protocol** (e.g.
blocked time-series cross-validation), not just search over MLP hyperparameters.