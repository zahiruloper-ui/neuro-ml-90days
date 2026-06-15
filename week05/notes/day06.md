# Week 5 Day 6 — Compare: Pick Best Model

## Models compared on Iris

1. **Logistic Regression** (linear/logistic baseline)
   - CV accuracy: **0.9733** (+/- 0.0249)
   - Single split test: 1.0
   - Pros: Simple, fast, interpretable, stable.
   - Cons: Assumes linear relationship, may struggle on complex non-linear data.

2. **Decision Tree** (single tree)
   - CV accuracy: **0.9733** (+/- 0.0249)
   - Single split: train = 0.9583, test = 1.0
   - Pros: Very interpretable (can visualize splits), non-linear.
   - Cons: Can overfit if too deep, less stable than Forest.

3. **Random Forest** (bagging)
   - CV accuracy: **0.9667** (+/- 0.0211)
   - Single split: train = 1.0, test = 1.0
   - Pros: Reduces variance, more stable, less overfitting risk.
   - Cons: Less interpretable, more complex, slightly lower CV on Iris.

4. **Gradient Boosting** (boosting)
   - CV accuracy: **0.9600** (+/- 0.0327)
   - Single split: train = 1.0, test = 1.0
   - Pros: Can be very accurate on harder datasets, sequential error correction.
   - Cons: More prone to overfitting, less interpretable, highest CV variability.

## Best model choice

**Best model:** Logistic Regression.

**Reason:**
- It tied for the best cross-validation accuracy at 0.9733.
- It is simpler and more interpretable than the tree-based models.
- On the Iris dataset, that makes it a very strong and practical choice because the data is fairly well separated.
- The main trade-off is that it may not handle more complex non-linear datasets as well as trees or ensembles.

## Metrics + interpretability trade-off

- **Performance:**
  - Logistic Regression and Decision Tree: tied best CV (0.9733).
  - Forest and Boosting: slightly lower on Iris.
- **Interpretability:**
  - Logistic Regression: coefficients show direction and strength of each feature.
  - Decision Tree: visualization shows exact rules (e.g., petal_length ≤ 2.45).
  - Forest and Boosting: harder to interpret; feature importance is available but not as clear as tree rules or coefficients.
- **Complexity:**
  - Logistic Regression: simplest, fastest.
  - Tree: simple, but can overfit.
  - Forest and Boosting: more complex, slower, but more robust on harder data.

For Iris, **simplicity wins**: Logistic Regression is best overall.

---

## Flashcards (Day 6)

Continue numbering from Day 5 (next is Q31).

**Q31:** What models did you compare in Week 5?  
**A31:** Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting on the Iris dataset.

**Q32:** Which model had the best cross-validation accuracy on Iris?  
**A32:** Logistic Regression and Decision Tree tied at 0.9733 mean CV accuracy.

**Q33:** Why did you choose Logistic Regression as the best model?  
**A33:** It tied for best CV accuracy, is simpler and more interpretable than tree-based models, and works well on Iris because the data is fairly well separated.

**Q34:** What is the trade-off of choosing Logistic Regression over trees/ensembles?  
**A34:** Logistic Regression may not handle more complex non-linear datasets as well as trees or ensembles, which can capture non-linear patterns.

**Q35:** When might you prefer Random Forest over Logistic Regression?  
**A35:** When the dataset has complex non-linear relationships, is noisy, or you need more robust performance and less sensitivity to feature transformations.

**Q36:** When might you prefer Gradient Boosting over Random Forest?  
**A36:** When you want to maximize accuracy on harder datasets and can afford more tuning and risk of overfitting; boosting often achieves higher accuracy with careful tuning.
