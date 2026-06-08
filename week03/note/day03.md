# Week 3 Day 3: Ridge vs Lasso + MAE/RMSE

## What I Learned
- Ridge (L2): penalty = λ × Σ(coef²) → shrinks coefficients, keeps all features
- Lasso (L1): penalty = λ × |coef| → can set coefficients to 0 (feature selection)
- α=0 → no regularization; α=large → strong regularization
- MAE vs RMSE: RMSE squares errors (sensitive to outliers), MAE absolute (robust)
- RMSE > MAE when outliers present
- Residuals normal + random → linear model works well

## My Results
- Best Ridge: α=0 or 0.1 (no regularization needed)
- Best Lasso: α=0 (no feature selection happened)
- RMSE > MAE (RMSE penalizes outliers more)
- Residuals: normal distribution + random scatter (good!)

## Key Formulas
- Ridge: MSE + λ × Σ(coef²)
- Lasso: MSE + λ × |coef|
- RMSE: √MSE
- MAE: (1/n) × Σ|y - ŷ|

## What's Next (Day 4)
- Residual plots + failure cases
- Diagnose where model breaks
