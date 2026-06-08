from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

# Split 60/20/20
X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=0.6, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Test different alpha values
alphas = [0, 0.1, 1.0, 10.0, 100.0]

print("=== Ridge Regression (L2) ===")
for alpha in alphas:
    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    test_r2 = r2_score(y_test, ridge.predict(X_test))
    test_rmse = np.sqrt(mean_squared_error(y_test, ridge.predict(X_test)))
    test_mae = mean_absolute_error(y_test, ridge.predict(X_test))
    print(f"α={alpha}: R²={test_r2:.3f}, RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, Coefs={ridge.coef_[:3]}")

print("\n=== Lasso Regression (L1) ===")
for alpha in alphas:
    lasso = linear_model.Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    test_r2 = r2_score(y_test, lasso.predict(X_test))
    test_rmse = np.sqrt(mean_squared_error(y_test, lasso.predict(X_test)))
    test_mae = mean_absolute_error(y_test, lasso.predict(X_test))
    # Count how many coefficients are ZERO (Lasso feature selection)
    zero_coefs = np.sum(lasso.coef_ == 0)
    print(f"α={alpha}: R²={test_r2:.3f}, RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, Zero coefs={zero_coefs}")




# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Regression Metrics ===")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.3f}")

print("\n=== What Each Metric Tells You ===")
print(f"RMSE = {rmse:.2f} → Average error is ~{rmse:.2f} units (sensitive to outliers)")
print(f"MAE  = {mae:.2f} → Average error is ~{mae:.2f} units (robust to outliers)")

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))

# Plot 1: Residuals histogram
plt.subplot(1, 2, 1)
plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
plt.axvline(x=0, color="red", linestyle="--")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Count")
plt.title("Residuals Distribution")

# Plot 2: Residuals vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted")

plt.tight_layout()
plt.savefig("week03/regression_metrics_plot.png", dpi=150)
plt.show()

print("\nResidual plot saved to: week03/regression_metrics_plot.png")