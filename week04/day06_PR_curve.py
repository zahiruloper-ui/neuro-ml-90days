import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score

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

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Get positive class probabilities
probs = model.predict_proba(X_valid)[:, 1]

# Compute precision-recall curve and average precision
precision, recall, thresholds = precision_recall_curve(y_valid, probs)
ap_score = average_precision_score(y_valid, probs)

print("=== Precision-Recall ===")
print(f"Average precision: {ap_score:.3f}")
print("First 5 precision values:", precision[:5])
print("First 5 recall values:", recall[:5])
print("First 5 thresholds:", thresholds[:5])

# Plot PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})', linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('week04/pr_curve.png', dpi=150)
print("\nPlot saved as: week04/pr_curve.png")
plt.show()
