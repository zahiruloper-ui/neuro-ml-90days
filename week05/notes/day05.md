# Week 5 Day 4 — Validate: Cross-Validation Basics

## Why cross-validation matters

With a single train/test split:
- You get **one accuracy number** (e.g., test = 1.0).
- That number depends heavily on how you split the data.
- If you split differently, accuracy might change.
- Can be misleading (e.g., all models get 1.0 by chance).

With cross-validation (e.g., 5-fold):
- Data is split into **5 equal parts** (folds).
- Train on 4 folds, test on 1 fold.
- Repeat 5 times, each fold used as test once.
- Average the 5 accuracy numbers.

Benefits:
- More **reliable** estimate of model performance.
- Less dependent on one random split.
- Better for comparing models fairly.
- Shows variability (standard deviation).

## Cross-validation vs single split on Iris

### Single train/test split (80/20):
- Logistic Regression: test = 1.0
- Decision Tree: train = 0.9583, test = 1.0
- Random Forest: train = 1.0, test = 1.0
- Gradient Boosting: train = 1.0, test = 1.0

All models got 100% test accuracy → no clear winner.

### Cross-validation (5-fold):
- Logistic Regression: **0.9733** (+/- 0.0249)
- Decision Tree: **0.9733** (+/- 0.0249)
- Random Forest: **0.9667** (+/- 0.0211)
- Gradient Boosting: **0.9600** (+/- 0.0327)

Now we see small differences:
- Logistic Regression and Decision Tree are slightly better.
- Random Forest and Boosting are slightly lower.
- CV reveals differences that a single split hid.

## Why cross-validation is more reliable than single split

- A single split gives one accuracy number that depends heavily on that one random split.
- Cross-validation gives 5 accuracy numbers from 5 different splits, and we average them.
- Because it uses multiple splits, it’s less dependent on one particular random split.
- This helps us compare models more fairly and get a more stable estimate of performance.

 
