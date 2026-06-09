import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
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

print("=== Class Distribution ===")
print(y.value_counts())
print(f"Healthy (0): {(y == 0).sum()}")
print(f"Parkinson's (1): {(y == 1).sum()}")

# Model 1: no class weights
model_plain = LogisticRegression(random_state=42)
model_plain.fit(X_train, y_train)
pred_plain = model_plain.predict(X_valid)

print("\n=== Plain Logistic Regression ===")
cm_plain = confusion_matrix(y_valid, pred_plain)
print("Confusion matrix:")
print(cm_plain)
print(f"Accuracy:  {accuracy_score(y_valid, pred_plain):.3f}")
print(f"Precision: {precision_score(y_valid, pred_plain):.3f}")
print(f"Recall:    {recall_score(y_valid, pred_plain):.3f}")

# Model 2: balanced class weights
model_balanced = LogisticRegression(random_state=42, class_weight='balanced')
model_balanced.fit(X_train, y_train)
pred_balanced = model_balanced.predict(X_valid)

print("\n=== Balanced Logistic Regression ===")
cm_balanced = confusion_matrix(y_valid, pred_balanced)
print("Confusion matrix:")
print(cm_balanced)
print(f"Accuracy:  {accuracy_score(y_valid, pred_balanced):.3f}")
print(f"Precision: {precision_score(y_valid, pred_balanced):.3f}")
print(f"Recall:    {recall_score(y_valid, pred_balanced):.3f}")




# Reset index for easier slicing
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Split train/valid
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=== Train distribution (before oversampling) ===")
print(y_train.value_counts())

# Identify minority class (smaller count)
count_0 = (y_train == 0).sum()
count_1 = (y_train == 1).sum()

if count_0 < count_1:
    minority_class = 0
    majority_class = 1
else:
    minority_class = 1
    majority_class = 0

n_minority = min(count_0, count_1)
n_majority = max(count_0, count_1)

print(f"Minority class: {minority_class} (count={n_minority})")
print(f"Majority class: {majority_class} (count={n_majority})")

# Find indices of minority class in training set
minority_indices = y_train[y_train == minority_class].index

# Oversample minority class: duplicate enough to match majority class
oversample_size = n_majority - n_minority

if oversample_size <= 0:
    print("\nNo oversampling needed (classes are balanced or minority is majority).")
    X_train_oversampled = X_train
    y_train_oversampled = y_train
else:
    # Randomly choose minority indices to duplicate
    duplicated_indices = np.random.choice(minority_indices, size=oversample_size, replace=True)

    # Create oversampled training sets
    X_train_oversampled = pd.concat([X_train, X_train.loc[duplicated_indices]], ignore_index=True)
    y_train_oversampled = pd.concat([y_train, y_train.loc[duplicated_indices]], ignore_index=True)

print("\n=== Train distribution (after oversampling) ===")
print(y_train_oversampled.value_counts())

# Train model on oversampled data
model_oversampled = LogisticRegression(random_state=42)
model_oversampled.fit(X_train_oversampled, y_train_oversampled)
pred_oversampled = model_oversampled.predict(X_valid)

print("\n=== Oversampled Logistic Regression ===")
cm_oversampled = confusion_matrix(y_valid, pred_oversampled)
print("Confusion matrix:")
print(cm_oversampled)
print(f"Accuracy:  {accuracy_score(y_valid, pred_oversampled):.3f}")
print(f"Precision: {precision_score(y_valid, pred_oversampled):.3f}")
print(f"Recall:    {recall_score(y_valid, pred_oversampled):.3f}")

# Compare all three
print("\n=== Comparison ===")
print("Plain model:         Precision=0.900, Recall=0.931")
print("Balanced model:      Precision=0.913, Recall=0.724")
print("Oversampled model:  Precision={:.3f}, Recall={:.3f}".format(
    precision_score(y_valid, pred_oversampled),
    recall_score(y_valid, pred_oversampled)
))