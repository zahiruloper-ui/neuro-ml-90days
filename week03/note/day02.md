# Week 3 Day 2: Feature Scaling + Train/Valid/Test Split

## What I Learned
- StandardScaler: mean=0, std=1; formula: x_scaled = (x - mean) / std
- Fit scaler on TRAIN only → prevents data leakage
- Train/Valid/Test (60/20/20): valid set for model selection, test for final eval
- Scaled vs Unscaled: R² same, but coefficients smaller (different units)
- Linear Regression beat Ridge/Lasso (no overfitting, so regularization not needed)

## My Results
- Train/Valid/Test R²: small gaps (~0.08)
- Best model: LinearRegression (validation R² highest)
- Valid vs Test R²: close (good generalization)

## Key Formulas
- Scaling: x_scaled = (x - μ) / σ
- Train/Valid/Test split sizes: 60%, 20%, 20%

## What's Next (Day 3)
- Ridge vs Lasso regularization
- Compare MAE/RMSE
