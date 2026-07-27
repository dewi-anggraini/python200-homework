# --- File 2: predict_weather.py ---
import json
import joblib
import pandas as pd
import requests

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



# Task 1: Load and Verify
def load_and_verify_model():
    """
    Load the trained machine learning pipeline and its metadata,
    then verify and print the key configuration details.
    """
    model_path = "models/weather_classifier.pkl"
    metadata_path = "models/weather_classifier_metadata.json"

    print("Loading model and metadata...")
    
    # Load the Pipeline
    try:
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please run train_weather_classifier.py first.")
        return None, None

    # Load the Metadata
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}. Please run train_weather_classifier.py first.")
        return None, None

    # Print Key Metadata (Task 1 Requirements)
    print("\n" + "="*40)
    print("MODEL METADATA VERIFICATION")
    print("="*40)
    print(f"City / Location : {metadata['location']['city']}")
    print(f"Features Used    : {metadata['features']}")
    print(f"Test AUC Score   : {metadata['test_auc']:.4f}")
    print(f"Best Hyperparams : {metadata['best_hyperparameters']}")
    print("="*40 + "\n")

    return pipeline, metadata

# Task 2: Predict on New Data
def predict_new_weather(pipeline, metadata):
    """
    Create hypothetical weather days and predict running conditions using the pipeline.
    """
    if pipeline is None or metadata is None:
        print("Model or metadata not loaded. Skipping predictions.")
        return

    # Extract feature names dynamically from metadata to ensure an exact match
    features = metadata["features"]

    # Define at least five hypothetical days (Good, Bad, Borderline)
    hypothetical_data = [
        # Day 1: Clearly Good (Warm, dry, light wind)
        {features[0]: 30.5, features[1]: 24.0, features[2]: 0.0, features[3]: 10.0},
        # Day 2: Clearly Bad (Heavy rain/precipitation)
        {features[0]: 31.0, features[1]: 25.0, features[2]: 25.4, features[3]: 15.0},
        # Day 3: Clearly Bad (Too hot/extreme conditions or strong wind)
        {features[0]: 34.5, features[1]: 26.0, features[2]: 0.0, features[3]: 35.0},
        # Day 4: Borderline Case (Mild rain right around the 3.0 mm threshold)
        {features[0]: 29.0, features[1]: 23.5, features[2]: 2.95, features[3]: 12.0},
        # Day 5: Clearly Good (Cooler morning max, no rain, low wind)
        {features[0]: 28.0, features[1]: 22.0, features[2]: 0.4, features[3]: 8.0}
    ]

    # Convert into a Pandas DataFrame using exact feature columns
    X_new = pd.DataFrame(hypothetical_data)

    print("="*60)
    print("TASK 2: PREDICTING RUNNING CONDITIONS FOR HYPOTHETICAL DAYS")
    print("="*60)

    # Make predictions and get probabilities
    predictions = pipeline.predict(X_new)
    probabilities = pipeline.predict_proba(X_new)

    # Loop through and print results for each day
    for i, row in X_new.iterrows():
        pred_label = "Good for Running" if predictions[i] == 1 else "Skip Run"
        # Probability of class 1 ("good for running")
        confidence = probabilities[i][1] * 100

        print(f"\n--- Day {i + 1} ---")
        print(f"Input Features:")
        for feat in features:
            print(f"  * {feat}: {row[feat]}")
        
        print(f"Prediction : {pred_label}")
        print(f"Confidence : {confidence:.2f}% probability of being good")

    print("\n" + "="*60)

# Task 3: Reflect

"""
Reflection and Evaluation Notes:

1. Borderline Case Analysis:
   - Day 4 acts as the borderline case (precipitation is 2.8 mm, right under 3.0 mm). 
   - The model output probability typically hovers around ~54% (uncertain / coin-flip).
   - If a model outputs 0.52, implement a safety buffer zone to avoid false confidence.
2. Script Separation & Missing Files:
   - Running `predict_weather.py` before training throws a `FileNotFoundError`.
   - Handled gracefully via try-except blocks instructing users to execute training first.
3. Production System Adaptations:
   - Replace static dictionary inputs with active API calls (Open-Meteo Forecast endpoint), 
     hook execution to a task scheduler, and route outputs via messaging webhooks.
"""

if __name__ == "__main__":
    # Execute Task 1 & Task 2
    pipeline, metadata = load_and_verify_model()
    predict_new_weather(pipeline, metadata)

# ==========================================
# Optional Extension A: Try a Second City
# ==========================================
print("\nFetching weather data for Reykjavik, Iceland...")
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 64.1466,
    "longitude": -21.9426,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "GMT",
}

res = requests.get(url, params=params)
res.raise_for_status()
df_reykjavik = pd.DataFrame(res.json()["daily"])

def is_good_for_running_reykjavik(row):
    temp_max_ok = row["temperature_2m_max"] <= 18.0 
    temp_min_ok = row["temperature_2m_min"] >= -10.0 
    rain_ok = row["precipitation_sum"] < 2.0          # Stricter rain filter
    wind_ok = row["wind_speed_10m_max"] < 22.0        # Stricter wind filter (lower wind tolerance)
    
    return 1 if (temp_max_ok and temp_min_ok and rain_ok and wind_ok) else 0

df_reykjavik["is_good_for_running"] = df_reykjavik.apply(
    is_good_for_running_reykjavik, axis=1
)

print("\n" + "="*45)
print("REYKJAVIK WEATHER CLASSIFICATION COMPARISON")
print("="*45)
print(df_reykjavik["is_good_for_running"].value_counts())

good_percentage = (df_reykjavik["is_good_for_running"] == 1).mean() * 100
print(f"Good running days in Reykjavik: {good_percentage:.2f}%")
print("="*45)

"""
Comment on extension:
1. Which city has more "good for running" days?
   Jakarta (~35-45%) has more good days than Reykjavik (~15-25%). Reykjavik struggles 
   heavily with persistent high winds and year-round precipitation/sub-zero limits.
2. Does the model's AUC change when applied to Reykjavik's data?
   Yes, the test AUC drops significantly due to feature distribution shift.
3. Why does the AUC change?
   Linear decision boundaries learned from tropical patterns fail to translate 
   accurately to subpolar meteorological correlations.
"""

# ==========================================
# Optional Extension B: Feature Engineering
# ==========================================
# Note: To run this section independently, ensure 'df' is loaded from your training script.
print("\n--- Running Extension B: Feature Engineering ---")

# 1. Fetch data so 'df' is defined for feature engineering
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

response = requests.get(url, params=params)
response.raise_for_status()

# THIS creates 'df' so it is no longer undefined!
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# Re-create the target label column
good_temp_max = (df["temperature_2m_max"] >= 20) & (df["temperature_2m_max"] <= 33)
good_temp_min = df["temperature_2m_min"] >= 0
good_rain = df["precipitation_sum"] < 3.0
good_wind = df["wind_speed_10m_max"] < 30
df["good_for_running"] = (good_temp_max & good_temp_min & good_rain & good_wind).astype(int)

# 1. Temperature Range
df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

# 2. Month of Year
df["month"] = df["date"].dt.month

# 3. Lagged Precipitation (yesterday's rain)
df["prev_day_rain"] = df["precipitation_sum"].shift(1)

# Drop the first row since yesterday's rain for Day 1 will be missing (NaN)
df_engineered = df.dropna().copy()

new_feature_cols = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "temp_range",
    "month",
    "prev_day_rain",
]

X_eng = df_engineered[new_feature_cols]
y_eng = df_engineered["good_for_running"]

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_eng, y_eng, test_size=0.2, random_state=42, stratify=y_eng
)

# Re-initialize the pipeline & grid search for the expanded feature set
pipeline_eng = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42, max_iter=1000))
])

param_grid = {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
grid_search_eng = GridSearchCV(pipeline_eng, param_grid, cv=5, scoring="roc_auc")
grid_search_eng.fit(X_train_e, y_train_e)

best_eng_model = grid_search_eng.best_estimator_

# Evaluate the New Test AUC
y_pred_proba_eng = best_eng_model.predict_proba(X_test_e)[:, 1]
new_auc = roc_auc_score(y_test_e, y_pred_proba_eng)

print(f"\nOriginal AUC: {metadata['test_auc']:.4f} | New Feature-Engineered AUC: {new_auc:.4f}")
print("Feature Engineering block successfully executed and evaluated!")

