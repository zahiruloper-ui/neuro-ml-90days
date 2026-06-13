import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the Iris dataset
df = pd.read_csv("week05/data.csv")

# Features and target
X = df.drop(columns=["class"])
y = df["class"]

# Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a decision tree with limited depth to avoid overfitting
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print("Decision Tree Classifier (Iris dataset)")
print("Accuracy:", acc)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Iris-0", "Iris-1", "Iris-2"]))

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=["Iris-0", "Iris-1", "Iris-2"], filled=True)
plt.title("Decision Tree Structure (max_depth=3)")
plt.savefig("week05/tree_visualization.png")
print("\nTree visualization saved as: week05/tree_visualization.png")

# Show the first few splits
print("\nTree splits (feature importances):")
for i, imp in zip(X.columns, model.feature_importances_):
    print(f"{i}: {imp:.4f}")

# --- Task 2: Train vs test accuracy ---

# Accuracy on training set
y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

print("\n--- Overfitting check ---")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {acc:.4f}")
print(f"Difference:     {train_acc - acc:.4f}")

if train_acc - acc > 0.1:
    print("⚠️  Large difference → possible overfitting")
else:
    print("✓ Train and test are close → likely not overfitting badly")

# --- Task 4: Compare with a deeper tree ---

# Train a deeper tree (no depth limit)
deep_model = DecisionTreeClassifier(
    random_state=42
)  # no max_depth → very deep
deep_model.fit(X_train, y_train)

# Accuracies
deep_train_acc = accuracy_score(y_train, deep_model.predict(X_train))
deep_test_acc = accuracy_score(y_test, deep_model.predict(X_test))

print("\n--- Deep tree (max_depth=None) ---")
print(f"Train accuracy: {deep_train_acc:.4f}")
print(f"Test accuracy:  {deep_test_acc:.4f}")
print(f"Difference:     {deep_train_acc - deep_test_acc:.4f}")

if deep_train_acc - deep_test_acc > 0.1:
    print("⚠️  Large difference → possible overfitting")
else:
    print("✓ Train and test are close → likely not overfitting badly")

print("\nDeep tree feature importances:")
for i, imp in zip(X.columns, deep_model.feature_importances_):
    print(f"{i}: {imp:.4f}")

print("\n--- Compare: shallow vs deep ---")
print(f"Shallow train: {train_acc:.4f},  test: {acc:.4f}")
print(f"Deep   train: {deep_train_acc:.4f},  test: {deep_test_acc:.4f}")