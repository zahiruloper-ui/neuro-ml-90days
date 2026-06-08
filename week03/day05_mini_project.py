
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import pickle

# Load dataset (diabetes as proxy for neuro data)
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

print("=== Mini-Project #1: Neuro Regression ===")
print(f"Samples: {len(X)}, Features: {len(feature_names)}")

# Split 60/20/20
X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=0.6, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

# Create pipeline (scale + model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", linear_model.LinearRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_train_pred = pipeline.predict(X_train)
y_valid_pred = pipeline.predict(X_valid)
y_test_pred = pipeline.predict(X_test)

print("\n=== Performance ===")
print(f"Train R²: {r2_score(y_train, y_train_pred):.3f}")
print(f"Valid R²: {r2_score(y_valid, y_valid_pred):.3f}")
print(f"Test R²: {r2_score(y_test, y_test_pred):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")

# Feature importance (coefficients)
model = pipeline.named_steps["model"]
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

print("\n=== Top Features ===")
print(coef_df.head(5).to_string(index=False))
print("\n=== Bottom Features ===")
print(coef_df.tail(5).to_string(index=False))

# Save model
with open("week03/neuro_regression_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

