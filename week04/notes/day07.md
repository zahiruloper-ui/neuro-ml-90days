# Week 4 Day 7 — Review and Wrap-up

## What I learned

This week I learned how to evaluate binary classification models in a much deeper way than just using accuracy.

I learned how to use:
- confusion matrix
- accuracy
- precision
- recall
- threshold tuning
- class weights and oversampling
- calibration and reliability diagrams
- ROC curves and AUC
- precision-recall curves
- F1 score

## Main ideas from the week

### Confusion matrix
A confusion matrix shows the counts of:
- true positives
- false positives
- true negatives
- false negatives

It helps me understand what kind of mistakes the model is making.

### Precision vs recall
Precision tells me:
- out of all predicted positive cases, how many were actually positive

Recall tells me:
- out of all real positive cases, how many the model found

This is important because in medical detection, recall is often more important than precision.

### Threshold tuning
I learned that changing the threshold changes model behavior:
- lower threshold → higher recall, lower precision
- higher threshold → higher precision, lower recall

That means I can tune the model depending on the goal.

### Class imbalance
I learned that imbalanced datasets need special attention.

I tested:
- plain logistic regression
- class-weighted logistic regression
- oversampled logistic regression

I saw that different methods can change precision and recall differently.

### Calibration
I learned that predicted probabilities are not always trustworthy.

A calibration curve checks whether a predicted probability really matches the real-world frequency.

### ROC and AUC
I learned that ROC curves show the tradeoff between:
- true positive rate
- false positive rate

AUC summarizes the ROC curve into one number.

An AUC of 0.5 means random guessing, while higher values mean better class separation.

### Precision-recall curve
I learned that precision-recall curves are especially useful when:
- the dataset is imbalanced
- the positive class matters most

This is very relevant for disease detection problems.

### F1 score
I learned that F1 score is the harmonic mean of precision and recall.

It is useful when I want one metric that balances both.

## What felt hardest

The hardest part was understanding:
- when to use ROC vs PR curve
- when to prefer precision vs recall
- how calibration differs from normal accuracy-based evaluation

These ideas were harder because they depend on the real goal of the project, not just the model output.

## What I feel confident about now

I now feel more confident about:
- reading a confusion matrix
- interpreting precision and recall
- understanding threshold tradeoffs
- explaining AUC
- understanding why PR curves matter in imbalanced problems
- choosing metrics based on the task

## Week 4 takeaway

Week 4 taught me that classification evaluation is not just about getting a high score.

It is about understanding:
- what the model gets right
- what the model gets wrong
- what type of mistake matters most
- which metric matches the real-world goal
