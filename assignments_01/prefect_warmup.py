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

@flow
def data_pipeline():
    # arr = np.array([])
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary

# Run pipeline
if __name__ == "__main__":
    data_pipeline()

# Reflection Questions
#
# 1. This pipeline is simple—just three small functions on a handful of
# numbers. Why might Prefect be more overhead than it is worth here?
#
# Answer:
# For a small pipeline like this, Prefect adds extra setup and decorators
# without providing much additional value. A regular Python script can perform
# the same work with less code and complexity.
#
# 2. Describe some realistic scenarios where a framework like Prefect could
# still be useful, even if the pipeline logic itself stays simple.
#
# Answer:
# Prefect is useful for scheduled workflows, recurring ETL jobs, cloud data
# pipelines, monitoring task execution, automatic retries after failures,
# logging, and coordinating tasks that depend on one another.