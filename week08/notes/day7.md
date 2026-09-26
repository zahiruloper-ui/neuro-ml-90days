# Week 8 - Day 7: Review (MLP vs RF on EEG Windows)

**Date:** 2026-09-26
**Focus:** Final synthesis of Week 8 — understanding what an MLP is, how it behaved on EEG windowed features, and how it compares to Random Forest under leakage-safe, blocked time-series evaluation.

---

## 1. Week 8 in one sentence

We built a small MLP on the same EEG windowed features and leakage-safe splits as Week 7, tuned its hyperparameters, compared it to Random Forest using blocked time-series cross-validation, and found that RF remained the stronger overall model while the MLP acted as a specialist in certain segments.

---

## 2. Pipeline story: data -> windows -> features -> splits -> models -> metrics

End-to-end pipeline:

1. **Data loading:** EEG Eye State CSV (14 channels + label).
2. **Windowing:** sliding windows of 128 samples with a step of 64.
3. **Labeling:** per-window majority label (mean of samples > 0.5 -> eye-closed).
4. **Feature extraction:** mean and std per channel -> 28-dim feature vectors.
5. **Splitting:**
   - Day 1: single time-block split with buffer (train ~80%, test ~20%).
   - Days 3–4: blocked time-series CV (5 contiguous blocks with buffer).
6. **Models:** Random Forest and MLP (hidden=(64,32), activation=tanh).
7. **Metrics:** accuracy, balanced accuracy, confusion matrices, per-window agreement.

Key integrity points:

- Leakage-safe splits: no overlapping windows in train/test; buffer used around boundaries.
- Scaling: `StandardScaler` fit on train only for MLP; RF used raw features.
- Blocked CV: tests model behavior across multiple temporal segments, not just one tail block.

---

## 3. MLP behavior: underfitting, tuning, and limits

Day 1:

- Baseline MLP (32 hidden units, ReLU, 500 iters) on a single tail split:
  - Train acc ≈ 0.881.
  - Test acc ≈ 0.283.
  - Balanced acc ≈ 0.386.
  - Confusion matrix showed aggressive misclassification of open windows.

Day 2:

- Hyperparameter sweep over hidden sizes, layer counts, activations, and `max_iter`.
- Observations:
  - Small nets with few epochs underfit: train acc ≈ 0.8, low test acc.
  - Larger/deeper nets with more epochs reached train acc ≈ 1.0 but only modest test improvements.
  - Balanced accuracy varied widely; some configs (like `(64,32), tanh`) achieved ≈ 0.77 on the tail block.

Core lesson:

- Hyperparameter tuning fixed underfitting on train but did not eliminate the distribution shift and class imbalance in the test blocks.
- Optimization error and generalization error are distinct; tuning mainly adjusts the former.

---

## 4. RF vs MLP under blocked time-series CV

Blocked CV (Days 3–4):

- 5 folds, each a contiguous time block with a buffer.
- RF and tuned MLP trained and evaluated on each fold.

Per-fold patterns:

- Some folds where both models are decent (Fold 0).
- Some folds where both perform poorly (Fold 2).
- Several folds where RF clearly outperforms MLP (Folds 1 and 3).
- One fold (tail, Fold 4) where MLP matches RF’s accuracy and has much higher balanced accuracy by catching all closed windows.

Mean metrics across folds:

- RF mean acc ≈ 0.460, mean bal_acc ≈ 0.518.
- MLP mean acc ≈ 0.391, mean bal_acc ≈ 0.478.

Interpretation:

- RF is the stronger, more robust model overall.
- MLP is valuable as a specialist, particularly in segments where recall on rare closed windows matters more than precision.

---

## 5. Error analysis: agreement and complementary strengths

Global agreement/disagreement (Days 3–4):

- `both_correct` ≈ 27% of windows.
- `both_wrong` ≈ 42%.
- `RF_only_correct` ≈ 19%.
- `MLP_only_correct` ≈ 12%.

Takeaways:

- RF rescues more of MLP’s mistakes than MLP rescues RF’s, reinforcing RF as the primary baseline.
- The large `both_wrong` fraction indicates windows that are hard for both models under current features.
- Tail segment shows MLP’s complementary role: catching closed windows RF misses.

Conceptual synthesis:

- It’s more useful to view RF and MLP as **complementary tools** rather than competitors — RF for general robustness, MLP for specific regimes.

---

## 6. Evaluation lessons: leakage, distribution shift, and blocked CV

Evaluation lessons from Week 7–8:

- Naive random splits on overlapping windows can severely inflate scores due to leakage.
- Time-block splitting with a buffer reduces leakage but can still produce imbalanced or unrepresentative test blocks.
- Blocked time-series CV provides multiple test segments, giving a more stable and honest view of performance.

Key ideas:

- Always respect temporal structure in splitting for time-series/neuro datasets.
- Check both leakage and distribution representativeness.
- Use metrics that handle imbalance (balanced accuracy, per-class recall, confusion matrices).

---

## 7. Conceptual takeaways for future projects

From Week 8:

- MLP basics: layers, activations, scaling, backprop, over/underfitting.
- RF vs MLP tradeoffs on small tabular datasets:
  - RF is often a strong baseline.
  - MLP adds expressive power but needs careful tuning and honest evaluation.
- Evaluation design is as important as the model:
  - Leakage-safe splits.
  - Time-aware blocked CV.
  - Error analysis across models.

These patterns generalize to other ML projects involving temporal or structured data.

---

# Mini-quiz (Week 8 Day 7)

Answer these from memory; you can check notes afterward.

1. **Pipeline recall:** In one sentence, describe the full pipeline from raw EEG samples to RF/MLP evaluation.

2. **MLP structure:** Explain why nonlinearity in hidden layers is essential for an MLP to be more expressive than logistic regression.

3. **Scaling:** Why did we need `StandardScaler` for the MLP but not for Random Forest?

4. **Under/overfitting:** Give one example of an underfitting MLP config and one example of a high-capacity (overfitting) config from your sweeps.

5. **Blocked CV:** Why is blocked time-series cross-validation more honest than a single tail split for evaluating RF vs MLP on EEG windows?

6. **RF vs MLP verdict:** Summarize the RF vs MLP comparison across folds in two sentences.

7. **Complementary use:** Describe one way you could combine RF and MLP in a future EEG or tabular project to leverage their complementary strengths.

8. **Evaluation design:** What are two key evaluation pitfalls Week 7–8 helped you avoid or recognize?
