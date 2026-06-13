import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Load Iris
df = pd.read_csv("week05/data.csv")
X = df.drop(columns=["class"])
y = df["class"]

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
}

# Cross-validation (5-fold)
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    cv_results[name] = scores
    print(f"\n--- {name} ---")
    print(f"CV accuracies: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Std accuracy:  {scores.std():.4f}")

# Compare mean accuracies
print("\n--- Mean CV Accuracy Comparison ---")
for name, scores in cv_results.items():
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
