# ---- File 1: train_weather_classifier.py ---

import joblib
import json
import pandas as pd
import requests
import os
import sys
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Step 1: Fetch the Data
print("Fetching weather data for Jakarta (Indonesia)...")

# Jakarta's coordinates: Latitude -6.2088, Longitude 106.8456
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": -6.2088,
    "longitude": 106.8456,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "Asia/Jakarta",
}

# Fetching the data
response = requests.get(url, params=params)
response.raise_for_status()  # Check if the download was successful

# Convert raw JSON data into a clean Pandas DataFrame
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# Print a summary of the dataset
print("--- Dataset Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Statistical Summary ---")
print(df.describe())

# Step 2: Engineer Labels
# Final Label Rules: good_for_running:
# Label definition:
#
#   1 = Good for running
#   0 = Not good for running
#
# Final thresholds used to create the training labels:
#
#   - temperature_2m_max: 20°C to 30°C
#   - temperature_2m_min: >= 0°C
#   - precipitation_sum: < 3.0 mm
#   - wind_speed_10m_max: < 30 km/h
# 
# Jakarta is the selected weather location

good_temp_max = (df["temperature_2m_max"] >= 20) & (df["temperature_2m_max"] <= 33)
good_temp_min = df["temperature_2m_min"] >= 0
good_rain = df["precipitation_sum"] < 3.0
good_wind = df["wind_speed_10m_max"] < 30

# Combine conditions to create the binary target column
df["good_for_running"] = (good_temp_max & good_temp_min & good_rain & good_wind).astype(int)

# Print class distribution
print("--- Class Distribution ---")
print(
    "0 = Not good for running"
)

print(
    "1 = Good for running"
)

print("\nClass Distribution Percentage:")
print(
    df["good_for_running"]
    .value_counts(normalize=True)
)

good_fraction = (
    df["good_for_running"]
    .mean()
)

print(
    f"\nFraction of days labeled good for running: "
    f"{good_fraction:.2%}"
)

# Class Distribution Reflection:
# What fraction of days in your dataset are labeled "good for running"? Does that seem reasonable given the climate where you chose?
#
# The fraction of days labeled as good for running is shown above.
# This percentage reflects how often Jakarta's weather meets the
# selected running conditions.
#
# Since Jakarta has a tropical climate with frequent rainfall and
# consistently warm temperatures, the number of suitable running
# days may be limited by the temperature and precipitation rules.
#
# The resulting class distribution helps show whether the chosen
# labeling criteria are reasonable for this location.


# Step 3: Train and Tune
# 1. Define features (X) and target (y)
features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]
X = df[features]
y = df["good_for_running"]

# 2. Split data into train (80%) and test (20%) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create a Pipeline with StandardScaler and LogisticRegression
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42)),
    ]
)

# 4. Set up GridSearchCV over at least five values of C
param_grid = {"classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
)

# 5. Fit the Grid Search to the training data
grid_search.fit(X_train, y_train)

# Print the best C value and best CV AUC
print("--- Grid Search Results ---")
print(f"Best C value: {grid_search.best_params_['classifier__C']}")
print(f"Best Cross-Validation AUC: {grid_search.best_score_:.4f}\n")

# 6. Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_pred_proba)

print("--- Test Set Evaluation ---")
print(f"Test AUC: {test_auc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Plot and save the ROC curve
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(best_model, X_test, y_test, name="Logistic Regression")
plt.title("ROC Curve - Weather Running Classifier (Jakarta)")
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig("outputs/weather_roc.png", dpi=300, bbox_inches="tight")
plt.close()

print("ROC curve successfully saved to outputs/weather_roc.png")

# --- Step 4: Reflect on Evaluation ---
# The model achieved a test AUC of approximately 0.94, indicating
# that it distinguishes very well between good and bad running days.
# This performance is about what I expected because precipitation
# and temperature provide strong signals for the labels defined.
#
# The classification report shows that recall for good running days
# (0.82) is higher than precision (0.67), meaning the model finds
# most suitable running days but occasionally predicts that a day is
# good when it is actually not. In practice, this could recommend a
# run under less favorable conditions.
#
# If this model were used in a real application, I would consider
# increasing the decision threshold above 0.5 to reduce false
# positives, especially for runners who prefer to avoid poor weather.

# --- Step 5: Save the Model and Metadata ---

# 1. Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# 2. Save the best Pipeline using joblib
model_path = "models/weather_classifier.pkl"
joblib.dump(best_model, model_path)

# 3. Compile metadata dictionary
metadata = {
    "python_version": sys.version,
    "scikit-learn_version": sklearn.__version__,
    "features": features,
    "best_hyperparameters": grid_search.best_params_,
    "test_auc": float(test_auc),
    "location": {
        "city": "Jakarta, Indonesia",
        "latitude": -6.2088,
        "longitude": 106.8456,
    },
    "label_thresholds_description": (
    "Good for running is defined as: "
    "temperature_2m_max between 20°C and 33°C, "
    "temperature_2m_min >= 0°C, "
    "precipitation_sum < 3.0 mm, "
    "and wind_speed_10m_max < 30 km/h. "
    "The maximum temperature threshold was adjusted to reflect "
    "Jakarta's tropical climate, as permitted by the assignment."
    )
}

# 4. Save metadata to a JSON file
metadata_path = "models/weather_classifier_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(
    f"Success! Model saved to {model_path} and metadata saved to {metadata_path}."
)

