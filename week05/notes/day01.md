# Week 5 Day 1 — Splits: Decision Tree Intuition

## What a decision tree does when it splits

- A decision tree repeatedly asks simple questions like:
  - `petal_length (cm) <= 2.45?`
  - `petal_width (cm) <= 1.65?`
- At each split, it chooses the feature and threshold that make the groups
  **more pure** (mostly one class).
- The goal is to create leaves where most samples are the same class.

In our Iris experiment:
- First split: `petal_length (cm) <= 2.45`
- Second split: `petal_length (cm) <= 4.75`
- Third split: `petal_width (cm) <= 1.65` or `<= 1.75`
- Feature importance: `petal_length` ≈ 0.93, `petal_width` ≈ 0.07

## Why trees can overfit

- Overfitting = memorizing training data too closely, including noise.
- A very deep tree can:
  - Create many tiny branches.
  - Fit unusual points that don’t generalize.
- Signs of overfitting:
  - Train accuracy ≈ 1.0
  - Test accuracy < train accuracy (e.g., 0.85)
  - Large difference (train - test) > 0.1

On Iris, even a deep tree didn’t overfit badly because the dataset is small and easy.

## How `max_depth` controls complexity

- `max_depth=3`: tree can have at most 3 levels.
  - Simpler, more general rules.
  - Less likely to overfit.
- `max_depth=None` (or large): tree can grow very deep.
  - Can fit training data perfectly.
  - More likely to overfit on harder datasets.

In our experiment:
- Shallow tree: train = 0.9583, test = 1.0
- Deep tree: train = 1.0, test = 1.0

## Why feature importance is useful (but can be misleading)

- Feature importance shows how much each feature reduces impurity.
- In Iris: `petal_length` is the main driver (≈ 0.91).
- Useful for:
  - Understanding what the model cares about.
  - Simplifying models (drop low-importance features).
- Can be misleading because:
  - Importance depends on the dataset and model.
  - Correlated features can split importance oddly.
  - Doesn’t always match human intuition about “causality”.

---


