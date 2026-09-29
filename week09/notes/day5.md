# Week 9, Day 5 — Experiment

## Goal of the day
Run the same experiment 6 times with different WINDOW_SIZE values (128, 192, 256, 320,
384, 448) and observe how results accumulate in results_log.csv as a readable table.

## What the results show
- n_windows barely changed (233 → 228) across the full range, confirming step=64
  controls window count, not window_size.
- RF balanced accuracy: 0.518 → 0.588 → 0.507 → 0.525 → 0.572 → 0.565 (best at 192,
  worst at 256, no simple monotonic trend).
- MLP balanced accuracy: 0.478 → 0.406 → 0.480 → 0.478 → 0.460 → 0.421 (best at 256,
  worst at 192 and 448, consistently below RF at every window size).
- Window size genuinely affects model behavior, but there's no single "best" size
  across both models; RF consistently outperforms MLP on balanced accuracy.

## Why n_windows stayed roughly constant
The sliding window loop uses range(0, len(X) - window_size + 1, step), where step=64
is the stride between windows. Changing window_size from 128 to 448 only reduces the
upper bound by 320 samples total, which translates to dropping ~5 windows across the
entire dataset (233 → 228), negligible compared to the effect on feature aggregation.

## Why RF > MLP consistently
RF is an ensemble of decision trees, robust to feature scaling and less sensitive to
optimization hyperparameters. MLP (neural network) requires careful tuning of learning
rate, hidden layers, activation, and iterations; the ConvergenceWarnings indicate it's
hitting max_iter=500 without fully converging, suggesting underfitting or poor
optimization landscape at these settings.

## What this motivates next
Since different window sizes affect RF and MLP differently, and neither model has a
clear "best" size across all values, this motivates per-model hyperparameter tuning:
maybe RF's optimal window is 192 while MLP's is 256, and you'd want to tune them
separately rather than forcing one size for both.