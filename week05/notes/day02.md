# Week 5 Day 2 — Forest: Random Forest + Feature Importance

## What Random Forest is and how bagging works

- Random Forest = many decision trees trained together.
- **Bagging** (Bootstrap Aggregating):
  - Create many random subsets of the training data (sampling with replacement).
  - Train a separate tree on each subset.
  - For classification: all trees vote; the most common class is the prediction.
- Bagging reduces variance by averaging many unstable trees.

## Why Random Forest reduces variance compared to a single tree

- A single tree is unstable: small data changes → very different splits and predictions.
- Forest creates many trees on different data subsets.
- By averaging/voting, unstable decisions cancel out.
- Final prediction is more stable and robust to minor changes.

## Feature importance: useful but can be misleading

### Useful for:
- Understanding what the model cares about.
- Simplifying models (drop low-importance features).
- Explaining results to others.

### Can be misleading because:
- Importance does **not** mean causation.
- A feature might be correlated with the true cause but not cause the outcome.
- Correlated features can split importance oddly (both get low importance).
- Importance depends on dataset and model; patterns on Iris may not reflect real data.

## How Forest feature importance differs from single tree

- Single tree: one tree → importance can be dominated by one feature.
  - Example: `petal_length` ≈ 0.93, `petal_width` ≈ 0.07.
- Forest: average across 100 trees → importance is more distributed.
  - Example: `petal_length` ≈ 0.45, `petal_width` ≈ 0.42, sepal features also have some importance.
- Forest’s importance is more balanced because it averages many trees.

## Our results on Iris

- Single tree (max_depth=3):
  - Train: 0.9583, Test: 1.0
- Random Forest (n_estimators=100, max_depth=5):
  - Train: 1.0, Test: 1.0
- Both achieve 100% test accuracy on Iris (easy dataset).



