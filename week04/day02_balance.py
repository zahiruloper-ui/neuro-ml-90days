
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np

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

# --- TASK 1: Get probabilities instead of predictions ---
print("\n=== Getting Probabilities ===")

# Get probabilities (returns array of [prob_healthy, prob_pd])
probabilities = model.predict_proba(X_valid)

print(f"First 5 probability pairs:")
for i in range(5):
    prob_healthy = probabilities[i][0]
    prob_pd = probabilities[i][1]
    print(f"Sample {i}: Healthy={prob_healthy:.3f}, PD={prob_pd:.3f}")

# Compare default prediction vs probabilities
default_pred = model.predict(X_valid)
print(f"\nFirst 5 default predictions (threshold=0.5): {default_pred[:5]}")
print(f"First 5 PD probabilities: {probabilities[:5, 1]}")

# Explain the relationship
print("\n=== How threshold works ===")
print("Default (0.5): If PD probability >= 0.5 → predict 1, else → predict 0")
print("Example: PD prob = 0.73 → 0.73 >= 0.5 → predict 1 (Parkinson's)")
print("Example: PD prob = 0.32 → 0.32 < 0.5 → predict 0 (healthy)")


# Get probabilities
probabilities = model.predict_proba(X_valid)
prob_pd = probabilities[:, 1]  # Extract PD probability (column 1)

# --- TASK 2: Apply custom thresholds ---
print("\n=== Testing Different Thresholds ===")

# Define thresholds to test
thresholds = [0.3, 0.5, 0.7]

print(f"\nValidation set size: {len(y_valid)}")
print(f"Actual PD cases: {y_valid.sum()}")
print(f"Actual healthy cases: {len(y_valid) - y_valid.sum()}")

for threshold in thresholds:
    # Apply custom threshold (FIXED: add .astype(int))
    y_pred_custom = np.where(prob_pd >= threshold, 1, 0).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_valid, y_pred_custom)
    recall = recall_score(y_valid, y_pred_custom)
    
    # Count predictions
    predicted_pd = y_pred_custom.sum()
    predicted_healthy = len(y_pred_custom) - predicted_pd
    
    print(f"\n--- Threshold = {threshold} ---")
    print(f"Predicted PD: {predicted_pd}, Predicted healthy: {predicted_healthy}")
    print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"Recall:    {recall:.3f} ({recall*100:.1f}%)")

# Print the tradeoff summary
print("\n=== TRADEOFF SUMMARY ===")
print("Lower threshold (0.3) → Higher recall, Lower precision")
print("Higher threshold (0.7) → Lower recall, Higher precision")
print("\nFor medical screening: You usually want HIGH RECALL (don't miss patients)")
print("So you'd choose threshold = 0.3")



import matplotlib.pyplot as plt



# --- TASK 4: Plot precision/recall across all thresholds ---
print("\n=== Plotting Precision/Recall Curve ===")

# Create 100 thresholds from 0.0 to 0.99
thresholds = np.arange(0, 1, 0.01)
precisions = []
recalls = []

# Calculate precision/recall for each threshold
for threshold in thresholds:
    y_pred = np.where(prob_pd >= threshold, 1, 0).astype(int)
    precisions.append(precision_score(y_valid, y_pred))
    recalls.append(recall_score(y_valid, y_pred))

print(f"Tested {len(thresholds)} thresholds")
print(f"At threshold 0.3: Precision={precisions[30]:.3f}, Recall={recalls[30]:.3f}")
print(f"At threshold 0.5: Precision={precisions[50]:.3f}, Recall={recalls[50]:.3f}")
print(f"At threshold 0.7: Precision={precisions[70]:.3f}, Recall={recalls[70]:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision', color='purple', linewidth=2)
plt.plot(thresholds, recalls, label='Recall', color='orange', linewidth=2)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Precision/Recall Tradeoff Across Thresholds', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('week04/precision_recall_curve.png', dpi=150)
print("\nPlot saved as: week04/precision_recall_curve.png")
plt.show()

# Find threshold for 95% recall target
print("\n=== Finding Threshold for 95% Recall ===")
for i, threshold in enumerate(thresholds):
    if recalls[i] >= 0.95:
        print(f"To get ≥95% recall: use threshold = {threshold:.2f}")
        print(f"At this threshold: Precision={precisions[i]:.3f}, Recall={recalls[i]:.3f}")
        break