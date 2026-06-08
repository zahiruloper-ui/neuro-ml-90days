# Week 3 Day 4: Residual Plots + Failure Cases

## What I Learned
- Good residuals: random scatter, normal distribution, mean ~0
- Bad patterns: curve (non-linear), funnel (heteroscedasticity), clusters (missing feature)
- Worst 5 failure cases: error = 101–131 (outliers)
- RANSAC (robust regression) only helps when outliers are extreme/frequent
- OLS beat RANSAC because dataset is small (losing 48% data hurts more)

## My Results
- Mean residual: 13 (slightly under-predicts)
- Outliers present, no patterns
- Worst 5 errors: 101–131
- RANSAC inlier ratio: 52% (too aggressive)
- OLS R² > RANSAC R² (OLS better for this dataset)

## Key Formulas
- Residual: y - ŷ
- Inlier ratio: (number of inliers) / (total samples)

## What's Next (Day 5)
- Flashcards (loss, regularization, bias/variance)
- Polish project
