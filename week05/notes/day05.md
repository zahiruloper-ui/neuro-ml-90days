# Week 5 Day 5 — Recall: Bagging vs Boosting, Overfitting Signs

## Bagging vs Boosting (key differences)

### Bagging (Random Forest)
- Trees trained **in parallel**.
- Each tree sees a **random subset of data** (bootstrap sample).
- Predictions combined by **voting** (classification) or **averaging** (regression).
- Main goal: reduce **variance**.
- More stable, less likely to overfit.

### Boosting (Gradient Boosting)
- Trees trained **sequentially** (one after another).
- Each new tree tries to **fix the errors** of the previous trees.
- Focuses on samples that were mispredicted.
- Predictions combined as a **weighted sum**.
- Main goal: reduce **error/loss**.
- Can be very accurate but more sensitive to overfitting.

## Overfitting signs in trees and ensembles

### In a single decision tree:
- Train accuracy ≈ 1.0, test accuracy < 1.0.
- Large difference: train - test > 0.1.
- Very deep tree (e.g., max_depth=None) with many tiny branches.
- Tree memorizes noise instead of learning general patterns.

### In Random Forest:
- Less likely to overfit because of averaging.
- But if n_estimators is very large and trees are very deep, can still overfit slightly.

### In Gradient Boosting:
- More likely to overfit if:
  - Too many trees (n_estimators).
  - Trees are too deep (max_depth).
  - Learning rate is too high.
- Overfitting shows as:
  - Train accuracy ≈ 1.0, test accuracy lower.
  - Performance drops on new data.

## Week 5 summary: Trees & Ensembles

### Models you compared:
1. **Logistic Regression** (linear baseline).
2. **Decision Tree** (single tree, splits on features).
3. **Random Forest** (bagging, many trees, reduces variance).
4. **Gradient Boosting** (boosting, sequential trees, reduces loss).

### On Iris dataset:
- Single train/test split: all got 100% test accuracy.
- Cross-validation (5-fold):
  - Logistic Regression: 0.9733
  - Decision Tree: 0.9733
  - Random Forest: 0.9667
  - Gradient Boosting: 0.9600

### Key intuitions:
- Trees split to make groups more pure.
- Trees can overfit if too deep.
- Forest reduces variance via bagging (parallel trees).
- Boosting reduces loss via sequential error correction.
- Cross-validation gives more reliable comparison.





