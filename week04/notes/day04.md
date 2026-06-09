# Week 4 Day 3 — Handling Imbalance

## What I learned

Today I learned about **class imbalance** and how to handle it with:
- class weights
- oversampling

Class imbalance means one class appears much more often than the other in the dataset.

My Parkinson’s dataset is actually imbalanced in the opposite direction from what I expected:
- Parkinson’s (1) is the majority class
- Healthy (0) is the minority class

## Class weights

Class weights are used to tell the model that some classes should matter more during training.

In scikit-learn, `class_weight='balanced'` automatically sets weights using the formula:

\[
\text{weight}_i = \frac{n\_samples}{n\_classes \times \text{count of class } i}
\]

This makes the minority class have a higher weight, so mistakes on that class hurt the loss more.

I tested:
- plain logistic regression (no class weights)
- balanced logistic regression (`class_weight='balanced'`)

Results:
- plain model: precision=0.900, recall=0.931
- balanced model: precision=0.913, recall=0.724

Balanced model increased precision but reduced recall.

## Oversampling

Oversampling means duplicating some samples from the minority class so that the dataset becomes more balanced.

I did:
1. find indices of minority class in training set
2. randomly duplicate some of them
3. add them back to the training set

After oversampling, my model got:
- precision=0.913, recall=0.724

This was similar to the balanced model.

## Comparison

| Method     | Precision | Recall |
|------------|-----------|--------|
| Plain      | 0.900     | 0.931  |
| Balanced   | 0.913     | 0.724  |
| Oversampled| 0.913     | 0.724  |

For medical screening, I want **high recall**, so the **plain model** is better here.

## Important concepts

### Class imbalance
Class imbalance means one class has many more samples than the other.

### Class weights
Class weights change how much the model “care” about mistakes on each class.

### Oversampling
Oversampling increases the number of minority class samples by duplicating them.

### Why this matters
In medical detection, missing real patients is often worse than false alarms. So recall is often more important than precision.

## Files created
- `week04/day03_imbalance.py`
- `notes/week04_day03.md`

## Next step
Next I will learn about calibration and reliability curves.





