# Week 4 Day 2 — Threshold Tuning

## What I learned

Today I learned that logistic regression does not directly think in terms of class labels first.  
It first produces **probabilities**, and then those probabilities are turned into class predictions using a **decision threshold**.

The default threshold is usually **0.5**:
- if predicted probability >= 0.5, predict Parkinson’s
- if predicted probability < 0.5, predict healthy

Changing the threshold changes the behavior of the model.

## Why threshold matters

A model can be made more strict or more sensitive depending on the threshold.

- **Lower threshold** → more cases predicted as Parkinson’s
- **Higher threshold** → fewer cases predicted as Parkinson’s

This creates a tradeoff between **precision** and **recall**.

## Precision and recall tradeoff

### Lower threshold
When the threshold is low, the model labels more samples as positive.

This usually causes:
- **recall to increase**
- **precision to decrease**

Why:
- more real Parkinson’s cases are caught
- but more healthy people may also be flagged incorrectly

### Higher threshold
When the threshold is high, the model becomes more conservative.

This usually causes:
- **precision to increase**
- **recall to decrease**

Why:
- the model only predicts Parkinson’s when it is more confident
- but some real Parkinson’s cases may be missed

## My threshold experiment

I tested three thresholds:

| Threshold | Precision | Recall |
|----------|-----------|--------|
| 0.3 | 82.9% | 100.0% |
| 0.5 | 90.0% | 93.1% |
| 0.7 | 92.3% | 82.8% |

## Interpretation

At threshold **0.3**, recall became **100%**, which means all real Parkinson’s cases in the validation set were detected.

At threshold **0.7**, precision became highest, which means when the model predicted Parkinson’s, it was more often correct.

This clearly shows the tradeoff:
- lower threshold → higher recall
- higher threshold → higher precision

## My threshold choice

### Goal
I want to catch as many Parkinson’s cases as possible, even if some healthy people are flagged by mistake.

### Chosen threshold
**0.3**

### Reason
If the threshold is low, more cases will be flagged, so there is a higher chance of getting true positive cases even though there will also be more false positive cases.

For disease detection, this is often better because missing a real patient is worse than sending a healthy person for extra checking.

## Precision-recall curve

I also generated a precision-recall tradeoff plot.

This type of graph helps show how precision and recall change across many thresholds, instead of checking only one or two values manually.

It helps answer questions like:
- what threshold gives high recall?
- what threshold gives better precision?
- where does recall start dropping sharply?

## Important concepts

### `predict_proba()`
This function gives predicted probabilities for each class instead of final class labels.

For binary classification:
- column 0 = probability of healthy
- column 1 = probability of Parkinson’s

### Threshold
A threshold is the cutoff used to turn a probability into a class label.

### Why this matters in medicine
In screening settings, **high recall** is usually preferred because missing a true patient is risky.

That means a **lower threshold** is often better for first-pass disease detection.

## Files created
- `week04/day02_balance.py`
- `week04/precision_recall_curve.png`
- `notes/week04_day02.md`

## Next step
Next I will learn how to handle imbalanced data using methods like class weights or resampling.
