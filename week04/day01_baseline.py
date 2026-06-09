import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt


# Load Parkinson's dataset
df = pd.read_csv('week04/data/parkinsons.data')

# Inspect: first 5 rows
print("=== First 5 Rows ===")
print(df.head())

# Inspect: shape, columns, dtypes
print("\n=== Dataset Info ===")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nDtypes:\n{df.dtypes}")

# Class distribution (status: 0=healthy, 1=Parkinson's)
print("\n=== Class Distribution ===")
print(df['status'].value_counts())
print(f"Healthy (0): {df['status'].value_counts().get(0, 0)}")
print(f"Parkinson's (1): {df['status'].value_counts().get(1, 0)}")

# Check missing values
print("\n=== Missing Values ===")
missing = df.isnull().sum()
print(missing[missing > 0])  # Only show columns with missing values
if missing.sum() == 0:
    print("No missing values found!")







# Prepare features (X) and target (y)
X = df.drop('status', axis=1)  # Remove 'status' column (it's the target)
X = X.drop('name', axis=1)     # Remove 'name' column (not useful for modeling)
y = df['status']                 # Target: 0=healthy, 1=Parkinson's

# Handle missing values (fill with mean if any exist)
X = X.fillna(X.mean())

# Split into train/valid (80/20) with STRATIFICATION
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== Train/Valid Split ===")
print(f"Train size: {len(X_train)}")
print(f"Valid size: {len(X_valid)}")
print(f"Train class distribution: {y_train.value_counts().tolist()}")
print(f"Valid class distribution: {y_valid.value_counts().tolist()}")

# Create and train baseline logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions on validation set
y_pred = model.predict(X_valid)

print("\n=== Baseline Logistic Regression ===")
print(f"Model trained successfully!")
print(f"First 5 predictions: {y_pred[:5]}")
print(f"Actual first 5 values: {y_valid[:5].tolist()}")






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
y_pred = model.predict(X_valid)

# --- TASK 3: Confusion Matrix ---
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_valid, y_pred)
print(cm)
print(f"\nBreakdown:")
print(f"True Negatives (healthy, predicted healthy): {cm[0, 0]}")
print(f"False Positives (healthy, predicted PD): {cm[0, 1]}")
print(f"False Negatives (PD, predicted healthy): {cm[1, 0]}")
print(f"True Positives (PD, predicted PD): {cm[1, 1]}")

# Plot confusion matrix
print("\n=== Plotting Confusion Matrix ===")
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Baseline Logistic Regression - Confusion Matrix")
plt.savefig("week04/confusion_matrix.png", dpi=150)
print("Plot saved as: week04/confusion_matrix.png")
plt.show()




# --- TASK 4: Classification Metrics ---
print("\n=== Classification Metrics ===")

# Using scikit-learn functions
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)

print(f"Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
print(f"Recall:    {recall:.3f} ({recall*100:.1f}%)")

# Manual calculation (verify the formulas)
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
print(f"\n=== Manual Verification ===")
print(f"Accuracy:  ({tn}+{tp})/{tn+tp+fn+fp} = {(tn+tp)/(tn+tp+fn+fp):.3f}")
print(f"Precision: {tp}/({tp}+{fp}) = {tp/(tp+fp):.3f}")
print(f"Recall:    {tp}/({tp}+{fn}) = {tp/(tp+fn):.3f}")

# Interpretation
print("\n=== Interpretation ===")
print(f"• Out of {len(y_valid)} validation samples, {int(accuracy*len(y_valid))} were correct")
print(f"• When predicting PD, you're {precision*100:.1f}% right (not many false alarms)")
print(f"• You catch {recall*100:.1f}% of actual PD cases (good — low false negatives)")
print(f"• For medical screening: This is a GOOD balance (high recall = don't miss patients)")