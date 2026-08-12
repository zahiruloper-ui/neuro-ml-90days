# Week 4 — Classification Evaluation (Parkinson’s Mini Project) — Summary

## Dataset

- **Dataset:** Parkinson’s disease classification dataset  
- **Target:** `status` (healthy vs Parkinson’s)  
- **Classes:**
  - Healthy = 48  
  - Parkinson’s = 147  
- **Problem:** Binary classification (medical screening)  
- **Note:** Imbalanced dataset → accuracy alone is not enough.

---

## Goal

Learn to evaluate binary classifiers beyond accuracy and choose metrics that match the real-world goal (disease detection).

---

## What I did

- Built a **logistic regression baseline**.
- Used a **stratified train/validation split** (train=156, valid=39).
- Evaluated the model with:
  - confusion matrix
  - precision, recall, accuracy
  - threshold tuning
  - class weights and oversampling
  - calibration / reliability diagram
  - ROC curve & AUC
  - precision-recall curve & average precision

---

## Key Results (Validation Set)

Baseline logistic regression (threshold 0.5):

- Accuracy = 87%
- Precision = 90%
- Recall = 93%
- AUC = 0.883
- Average precision = 0.953

Threshold experiments:

| Threshold | Precision | Recall |
|----------|-----------|--------|
| 0.3      | 82.9%     | 100.0% |
| 0.5      | 90.0%     | 93.1%  |
| 0.7      | 92.3%     | 82.8%  |

Class handling methods:

| Method      | Precision | Recall |
|-------------|-----------|--------|
| Plain       | 0.900     | 0.931  |
| Balanced    | 0.913     | 0.724  |
| Oversampled | 0.913     | 0.724  |

**Chosen setup:**  
Plain logistic regression with **threshold = 0.3** (prioritizes recall for disease detection).

---

## Main Takeaways

- Accuracy can be misleading on imbalanced data.
- For medical screening, **recall** is often more important than precision.
- Threshold tuning lets me control the precision–recall tradeoff.
- Class weights and oversampling do not always improve recall; they must be evaluated with the right metric.
- Predicted probabilities need calibration checks to be trustworthy.
- ROC/AUC and PR curves give a more complete picture than a single accuracy number.

---

## Files Created

- Code: `week04/day01_baseline.py` … `week04/day06_pr_curve.py`
- Plots: confusion matrix, PR curve, calibration curves, ROC curve, PR curve
- Notes: `notes/week04_day01.md` … `notes/week04_day07.md`

---

## Next

Use these evaluation skills in **Week 5** to compare linear/logistic vs Decision Tree vs Random Forest vs Gradient Boosting on the same dataset.
