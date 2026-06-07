# Week 3 Day 1: Baseline Regression

## What I Learned
- Linear regression predicts continuous numbers using y = mx + b
- Model learns by minimizing MSE (mean squared error)
- R² score = how much better than guessing average (0.5-0.7 = moderate)
- Train/test split prevents overfitting (80/20)
- Residuals should be random (no pattern) for linear model to work

## My Results
- MSE: 3309.16
- R²: 0.42 (explains 42% variance)
- Coefficient: ~2.5 (BMI → disease progression)
- Train/Test gap: small (no overfitting)

## Key Formulas
- Prediction: ŷ = mx + b
- MSE: (1/n) × Σ(y - ŷ)²
- RMSE: √MSE

## What's Next (Day 2)
- Feature scaling (StandardScaler)
- Train/valid/test split (not just train/test)
