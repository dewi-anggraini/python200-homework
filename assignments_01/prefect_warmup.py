# --- Pipeline Q2 ---
import pandas as pd
import numpy as np
from prefect import task, flow


# Copy from Pipeline Q1
# Step 1: Create Series
@task
def create_series(arr):
    return pd.Series(arr, name="values")

# Step 2: Clean Data
@task
def clean_data(series):
    return series.dropna()

# Step 3: Summarize Data
@task
def summarize_data(series):
    modes = series.mode()

    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": modes.iloc[0] if len(modes) > 0 else None
    }
#@task
#def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode().iloc[0]  # mode() returns a Series, and take first value
    }

# Step 4: Pipeline
@flow
def pipeline_flow():
    # arr = np.array([])
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    print(summary)
    return summary

# Run pipeline
if __name__ == "__main__":
    pipeline_flow()
