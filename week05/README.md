# Week 5 Mini Project — Trees and Ensembles

## Project overview

This mini project compared four classification models on the **same Iris dataset**:

- Logistic Regression — linear baseline
- Decision Tree — single tree
- Random Forest — bagging ensemble
- Gradient Boosting — boosting ensemble

The goal was to compare predictive performance, understand model behavior, and consider the trade-off between accuracy and interpretability.

## Dataset

- **Dataset:** Iris
- **Samples:** 150
- **Features:** 4 numerical measurements
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- **Target:** 3 iris classes
- **Train/test split:** 80% training, 20% testing
- **Random state:** 42
- **Cross-validation:** 5-fold

## Single train/test results

| Model | Train accuracy | Test accuracy |
|---|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 |
| Decision Tree (`max_depth=3`) | 0.9583 | 1.0000 |
| Random Forest (`100` trees, `max_depth=5`) | 1.0000 | 1.0000 |
| Gradient Boosting (`100` trees, `learning_rate=0.1`) | 1.0000 | 1.0000 |

All four models achieved 100% test accuracy on this split. Because Iris is a small and relatively easy dataset, the single split did not provide a clear winner.

## Five-fold cross-validation results

| Model | Mean accuracy | Standard deviation |
|---|---:|---:|
| Logistic Regression | **0.9733** | 0.0249 |
| Decision Tree | **0.9733** | 0.0249 |
| Random Forest | 0.9667 | **0.0211** |
| Gradient Boosting | 0.9600 | 0.0327 |

Cross-validation showed small differences between the models. Logistic Regression and the Decision Tree had the highest mean accuracy. Random Forest had the lowest variability, while Gradient Boosting had the highest variability among the four models.

## Feature importance

| Model | Sepal length | Sepal width | Petal length | Petal width |
|---|---:|---:|---:|---:|
| Decision Tree | 0.0000 | 0.0000 | **0.9346** | 0.0654 |
| Random Forest | 0.1059 | 0.0280 | **0.4453** | 0.4208 |
| Gradient Boosting | 0.0014 | 0.0147 | **0.6658** | 0.3182 |

Petal length was the most important feature for all tree-based models. Feature importance is useful for understanding model behavior, but it does not prove that a feature causes the prediction. Correlated features and the choice of model can also affect importance values.

## Final model choice

**Selected model: Logistic Regression.**

Logistic Regression tied with the Decision Tree for the best mean cross-validation accuracy at 0.9733. It was selected because it is simple, fast, and easier to interpret than Random Forest and Gradient Boosting. The main trade-off is that Logistic Regression may not capture complex non-linear patterns as effectively as tree-based models on harder datasets.

## Main lessons

- Decision trees split data to create purer groups.
- Deep trees can overfit by memorizing training examples.
- Random Forest uses bagging and averaging to reduce variance.
- Gradient Boosting builds trees sequentially to reduce remaining error.
- Cross-validation gives a more reliable comparison than one train/test split.
- High accuracy on an easy dataset does not automatically mean a model will perform equally well on real-world data.
