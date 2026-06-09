# Week 4 Day 1 — Baseline Logistic Regression + Confusion Matrix

## What I learned

Today I built my first baseline classification model using logistic regression on the Parkinson’s dataset.

This week is about **classification**, which is different from regression:
- Regression predicts continuous values like age or response amplitude.
- Classification predicts categories like healthy vs Parkinson’s.

I learned that before training a model, I need to inspect:
- class balance,
- missing values,
- feature columns,
- and possible non-useful columns like identifiers.

In this dataset:
- Healthy = 48
- Parkinson’s = 147
- Missing values = none

This means the dataset is **imbalanced**, so accuracy alone is not enough to judge the model.

## Important concepts

### Logistic regression
Logistic regression is a classification algorithm used for binary prediction.
It predicts the probability that a sample belongs to class 1, and then converts that probability into a class label.

### Why drop `name`
The `name` column is just an identifier.
It does not describe the patient’s voice measurements, so it should not be used as a feature.

### Train/validation split
I split the dataset into train and validation sets:
- Train size = 156
- Validation size = 39

I used `stratify=y` so that both sets keep a similar healthy/Parkinson’s ratio.

### Confusion matrix
The confusion matrix helps me see exactly what kinds of mistakes the model makes.

My confusion matrix values were:
- True Negatives (TN) = 7
- False Positives (FP) = 3
- False Negatives (FN) = 2
- True Positives (TP) = 27

Interpretation:
- TN: healthy people correctly predicted as healthy
- FP: healthy people incorrectly predicted as Parkinson’s
- FN: Parkinson’s cases incorrectly predicted as healthy
- TP: Parkinson’s cases correctly predicted as Parkinson’s

For medical screening, false negatives are usually worse than false positives because missing a real patient is more dangerous than a false alarm.

### Metrics
My model results:
- Accuracy = 87%
- Precision = 90%
- Recall = 93%

#### Accuracy
Accuracy tells me how many total predictions were correct.

#### Precision
Precision tells me:
“When the model predicts Parkinson’s, how often is it correct?”

#### Recall
Recall tells me:
“Out of all actual Parkinson’s cases, how many did the model catch?”

For medical problems, recall is often very important because missing real cases can be costly.

## What I noticed

The model made more false positives than false negatives.

That is usually a better error pattern for screening because it is safer to flag an extra healthy person than to miss someone who may actually have Parkinson’s.

## Files created
- `week04/day01_baseline.py`
- `week04/confusion_matrix.png`
- `notes/week04_day01.md`

## Next step
Next I will learn about precision/recall tradeoffs and threshold tuning.
