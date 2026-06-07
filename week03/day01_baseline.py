import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load diabetes dataset (continuous target: disease progression)
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

# Use only 1 feature for visualization (BMI = feature index 2)
X = X[:, np.newaxis, 2]

# Split into train/test (80/20)
X_train = X[:-40]
X_test = X[-40:]
y_train = y[:-40]
y_test = y[-40:]

# Create + train baseline model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)

print("=== Baseline Linear Regression ===")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Plot
plt.scatter(X_test, y_test, color="black", label="Actual")
plt.plot(X_test, y_pred, color="blue", linewidth=3, label="Predicted")
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.legend()
plt.tight_layout()
plt.savefig("week03/regression_baseline_plot.png", dpi=150)
plt.show()







# Predict

y_train_pred = model.predict(X_train)

print("=== Model Details ===")
print(f"Coefficient (slope m): {model.coef_[0]:.4f}")
print(f"Intercept (bias b): {model.intercept_:.4f}")
print(f"\nEquation: y = {model.coef_[0]:.4f} * BMI + {model.intercept_:.4f}")

print("\n=== Performance on TRAIN vs TEST ===")
print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.2f}")
print(f"Train R²: {r2_score(y_train, y_train_pred):.2f}")
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Test R²: {r2_score(y_test, y_pred):.2f}")

print("\n=== What Does Coefficient Mean? ===")
print(f"For every 1 unit BMI increase, disease progression increases by {model.coef_[0]:.2f}")

# Plot residuals (errors)
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color="black", alpha=0.6)
plt.plot(X_test, y_pred, color="blue", linewidth=3)
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Actual vs Predicted")

# Plot 2: Residuals
plt.subplot(1, 2, 2)
plt.scatter(X_test, residuals, color="red", alpha=0.6)
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("BMI")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals (Errors)")

plt.tight_layout()
plt.savefig("week03/regression_residuals_plot.png", dpi=150)
plt.show()

print("\nResidual plot saved to: week03/regression_residuals_plot.png")