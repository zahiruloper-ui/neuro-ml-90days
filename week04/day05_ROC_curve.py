import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

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

# Get probabilities for positive class
probs = model.predict_proba(X_valid)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_valid, probs)
auc_score = roc_auc_score(y_valid, probs)

print("=== ROC / AUC ===")
print(f"AUC: {auc_score:.3f}")
print("First 5 FPR values:", fpr[:5])
print("First 5 TPR values:", tpr[:5])
print("First 5 thresholds:", thresholds[:5])

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier', linewidth=2)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('week04/roc_curve.png', dpi=150)
print("\nPlot saved as: week04/roc_curve.png")
plt.show()
