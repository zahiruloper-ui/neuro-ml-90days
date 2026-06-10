# Week 4 Day 5 — ROC and AUC

## What I learned

Today I learned about **ROC curves** and **AUC**.

ROC stands for **Receiver Operating Characteristic**.  
It shows how a classifier performs across all possible thresholds.

## ROC curve

The ROC curve plots:
- **x-axis** = False Positive Rate (FPR)
- **y-axis** = True Positive Rate (TPR)

As the threshold changes, the ROC curve shows the tradeoff between:
- catching more true positives
- creating more false positives

A good ROC curve moves toward the **top-left corner**, which means high TPR and low FPR.

## AUC

AUC means **Area Under the Curve**.

It gives one number that summarizes the ROC curve.

My AUC score was:
- **0.883**

This means my model is much better than random guessing.

## What FPR and TPR mean

### TPR
True Positive Rate = how many real positive cases were correctly detected.

This is also called:
- recall
- sensitivity

### FPR
False Positive Rate = how many negative cases were incorrectly predicted as positive.

## Interpretation of my result

An AUC of 0.883 means:
- the model separates the two classes well
- the model ranks positive cases above negative cases most of the time
- the model is clearly better than random

## Important concepts

### Random classifier
A random classifier has AUC around 0.5.

### Better classifier
A better classifier has AUC closer to 1.0.

### Threshold dependence
ROC and AUC evaluate performance across all thresholds, not just one fixed threshold.

## Files created
- `week04/day05_roc_auc.py`
- `week04/roc_curve.png`
- `notes/week04_day05.md`

## Next step
Next I will learn how to turn these evaluation ideas into a final model comparison and finish the week review.
