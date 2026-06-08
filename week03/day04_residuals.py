from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
# Load dataset
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

# Split 60/20/20
X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=0.6, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test - y_pred

print("=== Residual Analysis ===")
print(f"Mean residual: {residuals.mean():.4f} (should be ~0)")
print(f"Std residual: {residuals.std():.2f}")
print(f"Max error: {residuals.max():.2f}")
print(f"Min error: {residuals.min():.2f}")

# Find worst predictions (failure cases)
error_indices = np.argsort(np.abs(residuals))  # Sort by absolute error
worst_5_indices = error_indices[-5:]  # Top 5 worst

print("\n=== Top 5 Failure Cases ===")
for i in worst_5_indices[::-1]:  # Reverse (worst first)
    actual = y_test[i]
    predicted = y_pred[i]
    error = residuals[i]
    print(f"Actual={actual:.2f}, Predicted={predicted:.2f}, Error={error:.2f}")

# Plot multiple residual views
plt.figure(figsize=(15, 10))

# Plot 1: Residuals vs Predicted
plt.subplot(2, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.6, s=50)
plt.axhline(y=0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted")
plt.grid(alpha=0.3)

# Plot 2: Residuals histogram + normal curve
plt.subplot(2, 2, 2)
plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7, color="skyblue")
# Add normal curve
residual_std = residuals.std()
residual_mean = residuals.mean()
x = np.linspace(residual_mean - 3*residual_std, residual_mean + 3*residual_std, 100)
plt.plot(x, 50 * (1/(residual_std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-residual_mean)/residual_std)**2), 
         color="red", linewidth=2)
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Residuals Distribution + Normal Curve")

# Plot 3: Actual vs Predicted (with error lines)
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred, alpha=0.6, s=50, color="green")
# Plot worst 5
for i in worst_5_indices:
    plt.scatter(y_test[i], y_pred[i], color="red", s=100, marker="x", linewidths=3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         color="black", linestyle="--", linewidth=2, label="Perfect")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Red X = Worst 5)")
plt.legend()
plt.grid(alpha=0.3)

# Plot 4: Residuals over index (check for patterns)
plt.subplot(2, 2, 4)
plt.plot(range(len(residuals)), residuals, marker="o", alpha=0.6, linestyle="-", markersize=4)
plt.axhline(y=0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Sample Index")
plt.ylabel("Residual")
plt.title("Residuals Over Index (Check for Patterns)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("week03/residual_analysis_plot.png", dpi=150)
plt.show()

print("\nResidual analysis plot saved to: week03/residual_analysis_plot.png")






# Train OLS (ordinary least squares)
ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)

# Train RANSAC (robust to outliers)
ransac = RANSACRegressor(LinearRegression(), random_state=42)
ransac.fit(X_train, y_train)
y_pred_ransac = ransac.predict(X_test)

# Compare metrics
print("=== OLS vs RANSAC (Robust) ===")
print(f"OLS R²: {r2_score(y_test, y_pred_ols):.3f}")
print(f"RANSAC R²: {r2_score(y_test, y_pred_ransac):.3f}")
print(f"\nOLS RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ols)):.2f}")
print(f"RANSAC RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ransac)):.2f}")
print(f"\nOLS MAE: {mean_absolute_error(y_test, y_pred_ols):.2f}")
print(f"RANSAC MAE: {mean_absolute_error(y_test, y_pred_ransac):.2f}")

print(f"\nInlier ratio (RANSAC): {ransac.inlier_mask_.sum()}/{len(ransac.inlier_mask_)} = {ransac.inlier_mask_.mean():.2%}")

# Plot residuals for both
residuals_ols = y_test - y_pred_ols
residuals_ransac = y_test - y_pred_ransac

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(residuals_ols, bins=15, alpha=0.6, label="OLS", edgecolor="black")
plt.hist(residuals_ransac, bins=15, alpha=0.6, label="RANSAC", edgecolor="black")
plt.axvline(x=0, color="red", linestyle="--")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Residuals: OLS vs RANSAC")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_pred_ols, residuals_ols, alpha=0.6, label="OLS", color="blue")
plt.scatter(y_pred_ransac, residuals_ransac, alpha=0.6, label="RANSAC", color="green")
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted")
plt.legend()

plt.tight_layout()
plt.savefig("week03/robust_regression_plot.png", dpi=150)
plt.show()

print("\nPlot saved to: week03/robust_regression_plot.png")