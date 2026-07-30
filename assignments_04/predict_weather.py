# --- File 2: predict_weather.py ---
import json
import joblib
import pandas as pd


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
    print(f"Features Used   : {metadata['features']}")
    print(f"Label Rules     : {metadata['label_thresholds_description']}")
    print(f"Test AUC Score  : {metadata['test_auc']:.4f}")
    print(f"Best Parameters : {metadata['best_hyperparameters']}")
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

# 1. Borderline Case
# The fourth hypothetical day was intentionally designed as a
# borderline example. Its precipitation (2.95 mm) is just below
# the 3.0 mm threshold used when creating the training labels,
# while the maximum temperature (33.0°C) and wind speed
# (29.0 km/h) are also very close to their respective limits.
# These values make the prediction more uncertain than the
# clearly good or clearly bad examples.
#
# 2. Handling Missing Files
# Before making predictions, the script attempts to load both
# the trained model and its metadata. If either file is missing,
# the program catches the FileNotFoundError and instructs the
# user to run train_weather_classifier.py first instead of
# crashing unexpectedly.
#
# 3. Future Improvements
# This script currently predicts using manually created
# hypothetical weather conditions. In a production application,
# these values could be replaced with live forecast data from
# the Open-Meteo Forecast API, allowing predictions to be made
# automatically for upcoming weather conditions.

if __name__ == "__main__":
    # Task 1 & Task 2
    pipeline, metadata = load_and_verify_model()
    predict_new_weather(pipeline, metadata)


