import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Iris
df = pd.read_csv("week05/data.csv")
X = df.drop(columns=["class"])
y = df["class"]

# Same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Single tree
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)
tree_train_acc = accuracy_score(y_train, tree_model.predict(X_train))
tree_test_acc = accuracy_score(y_test, tree_model.predict(X_test))

# Random Forest
forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
forest_model.fit(X_train, y_train)
forest_train_acc = accuracy_score(y_train, forest_model.predict(X_train))
forest_test_acc = accuracy_score(y_test, forest_model.predict(X_test))

# Gradient Boosting
boost_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
boost_model.fit(X_train, y_train)

# Boosting predictions
boost_train_acc = accuracy_score(y_train, boost_model.predict(X_train))
boost_test_acc = accuracy_score(y_test, boost_model.predict(X_test))

# Print results
print("--- Single Tree (max_depth=3) ---")
print(f"Train accuracy: {tree_train_acc:.4f}")
print(f"Test accuracy:  {tree_test_acc:.4f}")

print("\n--- Random Forest (n_estimators=100, max_depth=5) ---")
print(f"Train accuracy: {forest_train_acc:.4f}")
print(f"Test accuracy:  {forest_test_acc:.4f}")

print("\n--- Gradient Boosting (n_estimators=100, max_depth=3, learning_rate=0.1) ---")
print(f"Train accuracy: {boost_train_acc:.4f}")
print(f"Test accuracy:  {boost_test_acc:.4f}")

print("\n--- Feature importances ---")
print("Tree:")
for i, imp in zip(X.columns, tree_model.feature_importances_):
    print(f"  {i}: {imp:.4f}")

print("Forest:")
for i, imp in zip(X.columns, forest_model.feature_importances_):
    print(f"  {i}: {imp:.4f}")

print("Boosting:")
for i, imp in zip(X.columns, boost_model.feature_importances_):
    print(f"  {i}: {imp:.4f}")

print("\n--- Boosting classification report (test) ---")
print(classification_report(y_test, boost_model.predict(X_test), target_names=["Iris-0", "Iris-1", "Iris-2"]))
