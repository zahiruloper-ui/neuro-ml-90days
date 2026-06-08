# Mini-Project #1: Neuro Variable Prediction (Regression)

## Goal
Predict a continuous neuro variable using linear regression.

## Dataset
- **Name:** Diabetes dataset (scikit-learn)
- **Proxy for:** Neuro data (disease progression = response amplitude proxy)
- **Samples:** 442
- **Features:** 10 (age, sex, BMI, blood pressure, etc.)

## Model
- **Type:** Linear Regression + StandardScaler
- **Pipeline:** `final_model_pipeline.pkl`
- **Train/Valid/Test split:** 60/20/20

## Results

| Metric | Value |
|--------|-------|
| Train R² | 0.45 |
| Valid R² | 0.44 |
| Test R² | 0.45 |
| Test RMSE | 57.5 |
| Test MAE | 45.2 |

## Top Features
1. **s5** (highest coefficient)
2. **bmi** (second highest)

## How to Use
```python
from week03.day06_pipeline import predict_neuro_value

# Predict for new patient
features = [age, sex, bmi, ap, s1, s2, s3, s4, s5, s6]
prediction = predict_neuro_value(features)
print(f"Predicted value: {prediction}")
```

## Lessons Learned
- Linear regression works well for baseline (R² ~0.45)
- Ridge/Lasso didn't help (no overfitting)
- Residuals are random (linear model appropriate)
- Outliers exist but aren't extreme enough to need RANSAC

## Next Steps
- Try polynomial features (non-linear relationships)
- Add more neuro-specific features
- Try ensemble methods (RandomForest, GradientBoosting)