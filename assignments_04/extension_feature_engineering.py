import json
import pandas as pd
import requests
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Optional Extension B: Feature Engineering

# Load existing metadata to compare the new AUC with the old test_auc
with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)

print(f"Original Test AUC: {metadata['test_auc']}")

print("\n--- Running Extension B: Feature Engineering ---")

# 1. Fetch data 
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

# creates dataframe [df]
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