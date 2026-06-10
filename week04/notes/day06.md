# Week 4 Day 6 — Precision-Recall Curves

## What I learned

Today I learned about the **precision-recall (PR) curve**.

A PR curve shows how precision and recall change as the threshold changes.

This is useful because I care a lot about finding positive cases, especially in an imbalanced dataset.

## Precision-Recall curve

The PR curve helps me see the tradeoff between:
- **precision** = how many predicted positives are correct
- **recall** = how many real positives are found

When the threshold goes down:
- recall usually goes up
- precision usually goes down

When the threshold goes up:
- precision usually goes up
- recall usually goes down

## Average precision

I got an average precision score of:
- **0.953**

This means the model is very good at ranking positive cases above negative cases.

## Why PR is useful here

My dataset is imbalanced and my main goal is to catch Parkinson’s cases.

That means I care more about the positive class than the negative class.

PR is more useful than ROC for this kind of problem because it focuses on positive-class performance.

## Important concepts

### Precision
Precision tells me, out of all predicted positives, how many are actually positive.

### Recall
Recall tells me, out of all actual positives, how many were found.

### Average precision
Average precision summarizes the PR curve with one score.

### Why this matters
PR curves are especially useful when the positive class is important or the dataset is imbalanced.

## Files created
- `week04/day06_pr_curve.py`
- `week04/pr_curve.png`
- `notes/week04_day06.md`

## Next step
Next I will review everything from Week 4 and make sure I understand the main classification metrics.

