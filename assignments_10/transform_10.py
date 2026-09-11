# Video Link: https://drive.google.com/file/d/1blwMmdqTlgcL0gmHWkvaYbowJt4Gm1h0/view?usp=sharing 
# This link has been tested, is accessible, and works well.
import json
import re
import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
import joblib
from openai import OpenAI

# Step 1: Incremental Read
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)


supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
response = supabase.table("weather_raw").select("*").execute();
raw_rows = response.data

enriched_response = supabase.table("weather_enriched").select("date").execute();
already_done = {row["date"] for row in enriched_response.data}

to_classify = [row for row in raw_rows if row["date"] not in already_done]

print(f"Raw records: {len(raw_rows)}")
print(f"Already enriched: {len(already_done)}")
print(f"Will be processed: {len(to_classify)}") 

if not to_classify:
    print("Nothing to do — all records already enriched.")
    exit()


# Step 2: ML Transform

FEATURES = metadata["features"]
df = pd.DataFrame(to_classify)

X = df[FEATURES]

clf = joblib.load("models/weather_classifier.pkl")

predictions = clf.predict(X)
probabilities = clf.predict_proba(X)[:, 1]  
print(f"Good days predicted: {predictions.sum()} / {len(predictions)}")
print(f"Confidence range: {probabilities.min():.2f} – {probabilities.max():.2f}")

enrichment_records = []
for i, row in enumerate(to_classify):
    enrichment_records.append({
        "date":  row["date"],
        "good_for_running": bool(predictions[i]),
        "confidence": round(float(probabilities[i]), 4)
    })

print("Sample enrichment records:")
for r in enrichment_records[:3]:
    print(r)


# Check:
good_days = [r for r in enrichment_records if r["good_for_running"]]
skip_days = [r for r in enrichment_records if not r["good_for_running"]]

print(f"Good days: {len(good_days)} ({len(good_days)/len(enrichment_records):.0%})")
print(f"Skip days: {len(skip_days)}")

# High-confidence
enrichment_records.sort(key=lambda r: r["confidence"], reverse=True)
print("\nHigh-confidence good days for running:")
for r in enrichment_records[:3]:
    print(f" {r['date']}: {r['confidence']:.3f}")

# borderline predictions
enrichment_records.sort(key=lambda r: abs(r["confidence"] - 0.5))
print("\nMost borderline (closest to 0.5 confidence):")
for r in enrichment_records[:3]:
    print(f" {r['date']}: {r['confidence']:.3f}")




# Step 3: LLM Transform

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."

)

def make_user_message(row, good_for_running, confidence):
    prediction_text = "good for running" if good_for_running else "not ideal for running"
    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C, Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )
    
    

for i, record in enumerate(enrichment_records):
    try:
        raw_row = next(r for r in to_classify if r["date"] == record["date"])
    
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                "content": make_user_message(
                    raw_row,
                    record["good_for_running"],
                    record["confidence"],
                    )
                }
            ],
            max_tokens=100,
        )
        summary = response.choices[0].message.content.strip()

        # Check for exactly one sentence.
        sentence_endings = re.findall(
            r'[.!?](?=\s|$)',
            summary
        )
        if not summary or len(sentence_endings) != 1:
            summary = "Recommendation unavailable."

    except Exception as e:
        print(f"Error processing record {record['date']}: {e}")
        summary = "Recommendation unavailable."
        
    record["llm_summary"] = summary
        
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1} / {len(enrichment_records)} records")


# Step 4: Load

response = (
    supabase.table("weather_enriched")
    .upsert(enrichment_records, on_conflict="date")
    .execute()
)

print(f"Upserted {len(response.data)} rows into weather_enriched")



# Step 5: Verify

# Get total number of rows
response = supabase.table("weather_enriched").select("*").execute()
all_rows = response.data

print(f"Total rows: {len(all_rows)}")

# Count good days
good_days = sum(row["good_for_running"] for row in all_rows)
print(f"Good days: {good_days}")

# Print 5 sample rows
print("\nSample rows:")
for row in all_rows[:5]:
    print(
        f"\n{row['date']} | "
        f"good={row['good_for_running']} | "
        f"conf={row['confidence']:.2f}"
    )
    print(f"  {row['llm_summary']}")
    
# The summaries generally reflected the weather features and the model prediction.    
# A good example was the June 10 summary,
# which mentioned the warm temperature and minimal precipitation and also recognized that
# the model prediction was uncertain because the confidence was close to 0.50. Another good
# example was September 11, where the summary mentioned the warm temperature and light wind
# and also pointed out the uncertainty of the prediction. A weaker example was September 9
# because the model predicted that it was good for running, but the LLM returned
# "Recommendation unavailable." This may have been caused by an API error or because the
# LLM response did not pass the validation check.



# Step 6: Reflect
# My chosen city is Jakarta (my hometown) so, I used weather data from Jakarta, 
# which is different from the data the ML classifier was trained on, so its predictions may not be as accurate.
# The model may have learned weather patterns that do not apply as well to Jakarta.
# The LLM cannot override the classifier because it uses the classifier's prediction and weather features
# to create a recommendation. It is mainly adding a natural-language explanation to the ML prediction. 
# If I ran the pipeline on 50,000 records, the main concerns would be cost and processing time 
# because of the number of LLM calls. I would handle this by processing the data incrementally 
# and only sending new records to the LLM.