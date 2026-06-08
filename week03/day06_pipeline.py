import pickle
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Load dataset
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", linear_model.LinearRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Test R²
test_r2 = r2_score(y_test, pipeline.predict(X_test))
print(f"Test R²: {test_r2:.3f}")

# Save pipeline
with open("week03/final_model_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)



# === PREDICTION FUNCTION ===
def predict_neuro_value(new_features, model_path="week03/final_model_pipeline.pkl"):
    """
    Predict continuous neuro value from new feature data.
    
    Args:
        new_features: 1D array or list of features (e.g., [age, sex, bmi, ...])
        model_path: Path to saved pipeline
    
    Returns:
        predicted_value: The predicted neuro variable
    """
    # Load model
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    
    # Convert to array and reshape
    new_features = np.array(new_features).reshape(1, -1)
    
    # Predict
    prediction = pipeline.predict(new_features)[0]
    
    return prediction

# === TEST PREDICTION FUNCTION ===
print("\n=== Test Prediction ===")

# Example 1: Use mean values (average patient)
mean_features = X.mean(axis=0)
prediction = predict_neuro_value(mean_features)
print(f"Average patient → predicted value: {prediction:.2f}")

# Example 2: Use your own values (e.g., first test sample)
sample = X_test[0]
prediction = predict_neuro_value(sample)
actual = y_test[0]
print(f"\nSample: actual={actual:.2f}, predicted={prediction:.2f}, error={actual-prediction:.2f}")
