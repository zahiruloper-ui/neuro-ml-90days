# Week 5 — Trees & Ensembles: Final Results

## Week goal
Compare:
- Linear/logistic baseline (Logistic Regression)
- Decision Tree
- Random Forest (bagging)
- Gradient Boosting (boosting)

on the **same dataset** (Iris).

## Week 5 results table

### Single train/test split (80/20, test_size=0.2, random_state=42)

| Model               | Train Accuracy | Test Accuracy |
|---------------------|----------------|---------------|
| Logistic Regression | 1.0000         | 1.0000        |
| Decision Tree       | 0.9583         | 1.0000        |
| Random Forest       | 1.0000         | 1.0000        |
| Gradient Boosting   | 1.0000         | 1.0000        |

All models got 100% test accuracy on Iris with a single split.

### Cross-validation (5-fold)

| Model               | Mean CV Accuracy | Std Accuracy |
|---------------------|------------------|--------------|
| Logistic Regression | 0.9733           | 0.0249       |
| Decision Tree       | 0.9733           | 0.0249       |
| Random Forest       | 0.9667           | 0.0211       |
| Gradient Boosting   | 0.9600           | 0.0327       |

Cross-validation shows small differences:
- Logistic Regression and Decision Tree: best (0.9733).
- Random Forest: slightly lower (0.9667).
- Gradient Boosting: lowest (0.9600), more variable (std = 0.0327).

### Feature importance comparison

| Model               | sepal length | sepal width | petal length | petal width |
|---------------------|--------------|-------------|--------------|-------------|
| Decision Tree       | 0.0000       | 0.0000      | 0.9346       | 0.0654      |
| Random Forest       | 0.1059       | 0.0280      | 0.4453       | 0.4208      |
| Gradient Boosting   | 0.0014       | 0.0147      | 0.6658       | 0.3182      |

- Tree: petal_length dominates.
- Forest: petal_length and petal_width both important.
- Boosting: petal_length dominant, petal_width second.

## Best model choice

**Best model:** Logistic Regression.

**Reason:**
- Tied for best cross-validation accuracy at 0.9733.
- Simple, fast, and interpretable.
- Works well on Iris because the data is fairly well separated.
- Trade-off: may not handle more complex non-linear datasets as well as trees or ensembles.

## Week 5 key learnings

### Conceptually
- A decision tree splits to make groups more pure (mostly one class).
- Trees can overfit if they grow too deep (many tiny branches).
- Random Forest reduces variance through **bagging** (parallel trees, random subsets).
- Gradient Boosting reduces error through **boosting** (sequential trees, each fixes previous errors).
- Bagging = parallel, reduces variance.
- Boosting = sequential, reduces loss.
- Feature importance is useful but can be misleading (not causation, correlated features).
- Cross-validation matters for more reliable model comparison.

### Practically
- Trained each model in scikit-learn:
  - `LogisticRegression`
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
  - `GradientBoostingClassifier`
- Compared models on the same dataset (Iris).
- Recorded results in a clean table (single split + CV).
- Explained why Logistic Regression wins on Iris.
- Thought about both performance and interpretability.

---

## Week 5 Quiz

Answer these in your own words (no code needed):

1. What does a decision tree do when it splits?
2. Why can decision trees overfit?
3. What is bagging in Random Forest?
4. How does boosting differ from bagging?
5. What are signs of overfitting in a tree?
6. Why is cross-validation better than a single train/test split?
7. Which model did you choose as best for Iris, and why?
8. What is the trade-off of choosing Logistic Regression over trees/ensembles?

You can write your answers in a separate file or just in your notes.

---

## Week 5 summary

Week 5 ended with:
- 4 models compared on Iris.
- Clear best model: Logistic Regression.
- Full understanding of:
  - Tree splits.
  - Bagging vs boosting.
  - Overfitting signs.
  - Cross-validation importance.
- Clean results table and quiz to review.


