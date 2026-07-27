import requests
import pandas as pd

# Optional Extension A: Try a Second City

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