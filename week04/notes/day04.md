# Week 4 Day 4 — Calibration and Reliability Curves

## What I learned

Today I learned about **calibration**, which checks whether a model’s predicted probabilities match reality.

A calibrated model should behave like this:
- if it predicts 0.8, then about 80% of those cases should really be positive
- if it predicts 0.2, then about 20% of those cases should really be positive

This matters because in classification, I do not only care about the class label.  
I also care about whether the probability is trustworthy.

## Predicted probabilities

I used `model.predict_proba(X_valid)[:, 1]` to get the predicted probability of class 1.

The `[:, 1]` part means:
- take all rows
- select column 1
- keep only the probability for the positive class

## Calibration curve

I used `calibration_curve(y_valid, probs, n_bins=10)` to create a reliability diagram.

This function compares:
- average predicted probability in each bin
- actual fraction of positives in that bin

If the curve is close to the diagonal line, the model is well calibrated.

## My probability summary

I found:
- min probability = 0.107
- max probability = 0.999
- mean probability = 0.714

That means the model is often confident and gives high predicted probabilities.

## Zigzag curve

My calibration plot looked zigzag.

This probably happened because:
- the validation set is small
- some bins have very few samples
- some probability ranges are crowded while others are sparse

Using fewer bins, like `n_bins=5`, should make the curve smoother and easier to interpret.

## Important concepts

### Calibration
Calibration means the predicted probabilities match the real-world outcome frequencies.

### Reliability diagram
A reliability diagram is a plot that shows whether a classifier’s probabilities are trustworthy.

### Underconfident
A model is underconfident when its probabilities are too low compared to reality.

### Overconfident
A model is overconfident when its probabilities are too high compared to reality.

## Files created
- `week04/day04_calibration.py`
- `week04/calibration_curve.png`
- `week04/calibration_curve_5bins.png`
- `notes/week04_day04.md`

## Next step
Next I will learn about the most important classification metrics in flashcard form and continue with the next Day 4 task if needed.


