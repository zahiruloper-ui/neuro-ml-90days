# Week 5 Day 3 — Boosting: Gradient Boosting

## What Gradient Boosting is

- Gradient Boosting = sequential ensemble of trees.
- Trees are trained **one after another**.
- Each new tree tries to **fix the errors** of the previous trees.
- It focuses on samples that were mispredicted or had large errors.
- Final prediction is a weighted sum of all trees.

## How Gradient Boosting builds models sequentially to reduce error

1. Start with a simple model (e.g., predict the average or most common class).
2. Look at the **errors**: which samples were wrong, and how far off?
3. Train a **new tree** to correct those errors, focusing more on wrong samples.
4. Add this tree to the model with a small adjustment controlled by `learning_rate`.
5. Repeat many times:
   - Each tree reduces the remaining error.
   - After many trees, the model is very accurate.

This is like studying for a test:
- First learn basics.
- Then look at wrong answers.
- Study those topics more.
- Each round fixes more mistakes.

## Bagging vs Boosting

| Aspect              | Bagging (Random Forest)                     | Boosting (Gradient Boosting)                |
|---------------------|---------------------------------------------|---------------------------------------------|
| Training order      | Trees trained **in parallel**               | Trees trained **sequentially**              |
| Data used           | Random subsets (bootstrap samples)          | All data, weighted by previous errors       |
| Focus               | Reduce **variance** via averaging           | Reduce **error/loss** via sequential fixes  |
| Overfitting risk    | Lower (averaging stabilizes)                | Higher if too many trees or high complexity |
| How predictions combine | Vote or average across all trees         | Weighted sum of all trees                   |
| Example             | Random Forest                               | Gradient Boosting                           |

Key points:
- Bagging trains trees in parallel, each sees random data → reduces variance.
- Boosting trains trees sequentially, each fixes previous errors → reduces loss.

## Our results on Iris

All three models achieve 100% test accuracy:

- Single Tree (max_depth=3):
  - Train: 0.9583, Test: 1.0
- Random Forest (n_estimators=100, max_depth=5):
  - Train: 1.0, Test: 1.0
- Gradient Boosting (n_estimators=100, max_depth=3, learning_rate=0.1):
  - Train: 1.0, Test: 1.0

Feature importances:
- Tree: petal_length dominates (0.93).
- Forest: more distributed (petal_length 0.45, petal_width 0.42).
- Boosting: petal_length dominant (0.67), petal_width second (0.32).


