# Video Link : https://drive.google.com/file/d/1DPkX5V3MmB-rBGrhDxWPMZAuK0iAMRGY/view?usp=sharing


import requests
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


# --- Step 1 : Extract ---

# Open-Meteo historical API
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
        "wind_speed_10m_max"
    ]
}

# Send request
response = requests.get(url, params=params)

# Check for errors
response.raise_for_status()

# Convert response to Python dictionary
data = response.json()

# Print a summary
print("Response received!")
print("Number of days:", len(data["daily"]["time"]))
print("Variables:", list(data["daily"].keys()))

# --- Step 02: Transform ---

daily = data["daily"]
records = [
    {
        "date":               daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum":  daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
    }
    for i in range(len(daily["time"]))
]
print(f"First record: {records[0]}\n")
print(f"Last record: {records[-1]}\n")
print(f"All records: {len(records)} records")

# I expect 365 records for 2023 because it is not a leap year.
# The actual record count may be lower if the API has missing days or other
# data discrepancies, so the returned count should be checked rather than assumed.

# --- # Step 03: Load ---

response = (
    supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
)

print(f"Upsert completed for {len(records)} records.")
# Idemptency is very important for ensuring data reliability, safe retries, and efficiency.


# --- Step 04: Verify ---

# Row count
count_response = supabase.table("weather_raw").select("date", count="exact").execute()
print(f"Rows in weather_raw: {count_response.count}\n")

# First & last record
earliest = (
    supabase.table("weather_raw")
    .select("*")
    .order("date", desc=False)
    .limit(1)
    .execute()
)

latest = (
    supabase.table("weather_raw")
    .select("*")
    .order("date", desc=True)
    .limit(1)
    .execute()
)

# Check for July 4, 2023
specific_date = (
    supabase.table("weather_raw")
    .select("*")
    .eq("date", "2023-07-04")
    .execute()
)

# Handle if July 4, 2023 is missing
if specific_date.data:
    print(f"Record for July 4, 2023: {specific_date.data}\n")
else:
    nearest = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", "2023-07-01")
        .lte("date", "2023-07-07")
        .execute()
    )
    print(f"July 4, 2023 was missing. Nearby records: {nearest.data}\n")

print(f"Earliest date: {earliest.data}\n")
print(f"Latest date: {latest.data}\n")

# The row count remained 365 after running the script a second time. 
# This shows that the upsert is idempotent because running the script again 
# does not create duplicate records.