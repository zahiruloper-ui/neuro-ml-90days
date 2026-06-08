import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load full diabetes dataset (all 10 features)
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

print("=== Before Scaling ===")
print(f"Feature 0 mean: {X[:, 0].mean():.2f}, std: {X[:, 0].std():.2f}")
print(f"Feature 1 mean: {X[:, 1].mean():.2f}, std: {X[:, 1].std():.2f}")

# Split: 60% train, 20% valid, 20% test
# Step 1: Split into train (60%) + rest (40%)
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, train_size=0.6, random_state=42
)

# Step 2: Split rest into valid (20%) + test (20%)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, random_state=42  # 50% of 40% = 20%
)

print(f"\nSplit sizes: Train={len(X_train)}, Valid={len(X_valid)}, Test={len(X_test)}")

# Scale features (fit on train, apply to all)
scaler = StandardScaler()
scaler.fit(X_train)  # Learn mean/std from TRAIN only!

X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

print("\n=== After Scaling (Train) ===")
print(f"Feature 0 mean: {X_train_scaled[:, 0].mean():.2f}, std: {X_train_scaled[:, 0].std():.2f}")
print(f"Feature 1 mean: {X_train_scaled[:, 1].mean():.2f}, std: {X_train_scaled[:, 1].std():.2f}")

# Train model on scaled data
model = linear_model.LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate on all sets
y_train_pred = model.predict(X_train_scaled)
y_valid_pred = model.predict(X_valid_scaled)
y_test_pred = model.predict(X_test_scaled)

print("\n=== Performance on Scaled Data ===")
print(f"Train R²: {r2_score(y_train, y_train_pred):.3f}")
print(f"Valid R²: {r2_score(y_valid, y_valid_pred):.3f}")
print(f"Test R²: {r2_score(y_test, y_test_pred):.3f}")

print(f"\nCoefficients: {model.coef_}")






# === UNSCALED ===
model_unscaled = linear_model.LinearRegression()
model_unscaled.fit(X_train, y_train)
r2_test_unscaled = r2_score(y_test, model_unscaled.predict(X_test))

# === SCALED ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = linear_model.LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
r2_test_scaled = r2_score(y_test, model_scaled.predict(X_test_scaled))

print("=== Scaled vs Unscaled Comparison ===")
print(f"Test R² (Unscaled): {r2_test_unscaled:.3f}")
print(f"Test R² (Scaled):   {r2_test_scaled:.3f}")
print(f"Difference: {abs(r2_test_scaled - r2_test_unscaled):.3f}")

print("\n=== Coefficients (Unscaled vs Scaled) ===")
print(f"Unscaled: {model_unscaled.coef_}")
print(f"Scaled:   {model_scaled.coef_}")




scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

models = {
    "LinearRegression": linear_model.LinearRegression(),
    "Ridge_1.0": linear_model.Ridge(alpha=1.0),
    "Ridge_10.0": linear_model.Ridge(alpha=10.0),
    "Lasso_0.1": linear_model.Lasso(alpha=0.1, max_iter=10000),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    valid_r2 = r2_score(y_valid, model.predict(X_valid))
    test_r2 = r2_score(y_test, model.predict(X_test))
    results.append((name, valid_r2, test_r2))

for name, valid_r2, test_r2 in results:
    print(f"{name}: valid R²={valid_r2:.3f}, test R²={test_r2:.3f}")

best = max(results, key=lambda x: x[1])
print(f"\nBest by validation: {best[0]} (valid R²={best[1]:.3f}, test R²={best[2]:.3f})")
