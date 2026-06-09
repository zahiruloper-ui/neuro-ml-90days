import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('week04/data/parkinsons.data')

# Prepare features and target
X = df.drop(['status', 'name'], axis=1)
X = X.fillna(X.mean())
y = df['status']

# Split train/valid
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train logistic regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Get predicted probabilities
probs = model.predict_proba(X_valid)[:, 1]

print("=== First 10 Predicted PD Probabilities ===")
for i, p in enumerate(probs[:10], start=1):
    print(f"Sample {i}: {p:.3f}")

print("\n=== Probability Summary ===")
print(f"Min probability: {probs.min():.3f}")
print(f"Max probability: {probs.max():.3f}")
print(f"Mean probability: {probs.mean():.3f}")


# --- TASK 2: Build calibration curve ---
print("\n=== Building Calibration Curve ===")

# Compute calibration curve with 10 bins
prob_true, prob_false = calibration_curve(y_valid, probs, n_bins=10)

print(f"Bins: {len(prob_true)}")
print(f"Average predicted prob per bin: {prob_true}")
print(f"Actual fraction of positives per bin: {prob_false}")

# Plot reliability diagram
plt.figure(figsize=(8, 6))

# Plot calibration curve
plt.plot(prob_true, prob_false, marker='o', label='Logistic Regression', linewidth=2)

# Plot diagonal (perfect calibration)
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

plt.xlabel('Average predicted probability', fontsize=12)
plt.ylabel('Fraction of positives', fontsize=12)
plt.title('Reliability Diagram (Calibration Curve)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('week04/calibration_curve.png', dpi=150)
print("\nPlot saved as: week04/calibration_curve.png")
plt.show()

print("\n=== Interpretation ===")
if prob_false.mean() > prob_true.mean():
    print("Curve is above diagonal → model is underconfident (probabilities too low)")
elif prob_false.mean() < prob_true.mean():
    print("Curve is below diagonal → model is overconfident (probabilities too high)")
else:
    print("Curve is close to diagonal → model is well calibrated")