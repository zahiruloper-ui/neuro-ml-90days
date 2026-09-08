# Week 8 - Day 1: Build (MLP Baseline)

**Date:** 2026-09-08
**Scripts:** `week8/day1/day1_mlp_baseline.py`, `week8/day1/day1_diagnose.py`
**Dataset:** EEG Eye State (14 channels, 14,980 samples, single subject)
**Goal:** Build a baseline MLP on Week 7's EEG windowed features using the identical leakage-safe time-block split, and get a first honest number to compare against Random Forest.

---

## 1. What an MLP is

A Multi-Layer Perceptron is a stack of layers. Each neuron computes a weighted sum of its inputs plus a bias, then applies a **nonlinear activation function**. For one hidden layer:

$$
\hat{y} = g\left(W_2 \, f(W_1 x + b_1) + b_2\right)
$$

- $x$ - 28-dim feature vector (14 channel means + 14 channel stds)
- $W_1, W_2$ - weight matrices learned during training
- $b_1, b_2$ - bias vectors
- $f$ - hidden activation, ReLU: $f(z) = \max(0, z)$
- $g$ - output activation, logistic (binary classification)

### Why nonlinearity is essential

If $f$ were the identity, then $W_2 W_1 x$ collapses into a single matrix - the network becomes plain logistic regression. Nonlinearity is the *only* reason depth adds expressive power.

### How it learns

    forward pass
      -> compute loss
      -> backpropagation (chain rule for dLoss/dW)
      -> gradient descent step (Adam)
      -> repeat per epoch

---

## 2. MLP vs. tree models, structurally

| Property | MLP | Random Forest |
| --- | --- | --- |
| Decision boundary | Smooth, uses all features jointly | Axis-aligned splits on single features |
| Scale sensitivity | High - needs standardization | None - scale-invariant |
| Out-of-range inputs | Extrapolates; can saturate to confident wrong answers | Falls into a terminal leaf; bounded output |
| Data hunger | High | Works on small $n$ |

### Why scaling is mandatory for the MLP

Features feed into weighted sums, so a feature with a large numeric range dominates the gradients. Fit `StandardScaler` on **train only**, then `transform` test - fitting on test would leak test statistics into preprocessing.

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   # learns mean/std from train
    X_test_s  = scaler.transform(X_test)        # applies the SAME numbers

---

## 3. Day 1 results (leakage-safe time-block split)

Pipeline reproduced Week 7 exactly: **233 windows x 28 features**, train `(185, 28)`, test `(46, 28)`.

| Model | Train acc | Test acc | Balanced acc | Macro F1 | Test confusion matrix |
| --- | --- | --- | --- | --- | --- |
| Baseline MLP (32, relu, 500 iter) | 0.881 | 0.283 | 0.386 | 0.24 | `[[12 32], [1 1]]` |
| MLP trained to convergence (2000 iter) | 1.000 | 0.391 | 0.443 | - | - |
| Random Forest (Week 7 best) | 1.000 | 0.609 | 0.557 | 0.42 | `[[27 17], [1 1]]` |

### Baseline honesty check on the test block

| Baseline | Test acc | Legitimate? |
| --- | --- | --- |
| `most_frequent` (learned from train, predicts eye-closed) | 0.043 | Yes |
| `stratified` (random at train proportions) | 0.391 | Yes |
| `uniform` (coin flip) | 0.435 | Yes |
| "Always eye-open" | 0.957 | **No - oracle, uses test labels** |

**Verdict:** RF (0.609) beats every legitimate baseline. The baseline MLP (0.283) beats only `most_frequent` and loses to a coin flip.

---

## 4. Core finding: distribution shift, not a bug

Train class counts `[86, 99]` (46% eye-open) vs. test class counts `[44, 2]` (96% eye-open). Label composition per consecutive time block shows why:

| Block | Windows | Fraction eye-closed |
| --- | --- | --- |
| 0 | 0-23 | 0.62 |
| 1 | 24-47 | 0.33 |
| 2 | 48-71 | 0.62 |
| 3 | 72-94 | 0.48 |
| 4 | 95-117 | 0.65 |
| 5 | 118-140 | **1.00** |
| 6 | 141-163 | **0.00** |
| 7 | 164-186 | 0.61 |
| 8 | 187-209 | **0.04** |
| 9 | 210-232 | **0.04** |

The subject's eye state comes in **long behavioral stretches**. Overall the dataset is 44% eye-closed (`[130, 103]`), but it is never balanced *at any moment in time*. Train spans blocks 0-7 (mixed); test is exactly blocks 8-9 (the eye-open tail).

### Feature-range evidence

Max standardized value is $|z| = 9.6$ in scaled train but $|z| = 28.3$ in scaled test - at least one test window sits 28 SDs outside the training range. The MLP extrapolates into that region and saturates; the RF's tree structure bounds its output, which is why it degrades more gracefully.

---

## 5. Optimization error vs. generalization error

| Error type | What it measures | Day 1 value | Fixable by more epochs? |
| --- | --- | --- | --- |
| Optimization | How well the model fits **train** | Loss 0.062, train acc 1.000 -> zero error | Yes (already solved) |
| Generalization | Does the pattern transfer to **test** | Test acc 0.391, plateaued | **No** |

Loss curve every 50 epochs (500-iter run):

    0.767 -> 0.645 -> 0.611 -> 0.578 -> 0.544 -> 0.512
          -> 0.480 -> 0.448 -> 0.417 -> 0.387 -> 0.360

Smooth monotonic descent, no plateau, `still decreasing = True`. So the `ConvergenceWarning` was real. But training to convergence (2000 iters) drove loss to 0.062 and train accuracy to a perfect 1.000, while test accuracy rose only 0.283 -> 0.391 and then stopped changing.

> **Key takeaway:** `max_iter`, learning rate, and epochs only ever attack *optimization* error.
> You cannot optimize your way out of a distribution shift.

---

## 6. Two distinct kinds of evaluation problem

| Aspect | Leakage (Week 7) | Distribution shift (Week 8 Day 1) |
| --- | --- | --- |
| Nature | Correctness **bug** | Honest **property of the data** |
| Mechanism | 50%-overlapping windows in both train and test | Test block is a different behavioral regime |
| Effect on score | **Inflates** (RF 0.587 -> 0.83) | **Deflates**, and makes metrics unstable |
| Fix | Time-block split + buffer | Better split design / CV; cannot be "fixed" away |

A time-block split **exposes** distribution shift rather than causing it. Fixing leakage does **not** guarantee a meaningful test set - you must separately verify the test block is representative and large enough per class.

With only 2 positive test windows, minority-class recall can take just three values (0, 0.5, 1.0), so one window flipping halves or doubles it.

### Test-block composition by split ratio

| train_frac | Test n | Test counts [open, closed] |
| --- | --- | --- |
| 0.5 | 116 | `[76, 40]` |
| 0.6 | 93 | `[76, 17]` |
| 0.7 | 69 | `[53, 16]` |
| **0.8 (used)** | **46** | `[44, 2]` |
| 0.9 | 23 | `[22, 1]` |

### Planned fix for Day 3

Blocked time-series cross-validation (multiple sequential folds, each with a buffer) so the MLP-vs-RF verdict rests on several test blocks instead of the single worst one. Keeps leakage safety and keeps the model as the only variable.

---

## 7. Caveats

- **Single-subject dataset** - a subject-wise split (gold standard for neuro data) is impossible here.
- **Raw accuracy is unusable** on a 96/4 test block; report balanced accuracy and macro F1.
- **Both models memorized all 185 training windows** (train acc 1.000), so train accuracy carries no signal.
- Time-domain features only (mean, std); no bandpower features, which are known to matter for eye-state classification.

---
