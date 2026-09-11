# --- ML vs. LLM in Pipelines ---

# Q1 
# The ML model is used for the binary prediction because it is trained for that classification task,
# while the LLM produces a recommendation or explanation in a readable natural language.
# The ML is appropriate for the decision because it is fast, inexpensive, deterministic, 
# and evaluated on labeled examples, while the LLM is appropriate for wording because it can
# explain structured facts naturally. If we used the LLM to make the binary good/skip prediction, 
# the result could be less consistent because LLM responses can vary. If we used the ML model
# to write the recommendation, it would not be suitable because a classifier
# produces predictions, not natural-language recommendations.

# Q2
# Converting a date to day-of-week: Deterministic code because standard date math is free and flawless.
# Classifying a job posting: An LLM, because the task requires interpreting varied natural-language responsibilities and context.
# Predicting customer churn:  A trained ML model, because a labeled structured dataset supports fast, repeatable predictive inference.
# Normalizing inconsistent city names: deterministic code because known city names can be mapped to one standard name.
# Summing a column of revenue figures: Deterministic code because math must be 100% precise.


# Q3
# Incremental processing means processing only the new data since the last run instead of processing all the data again. 
# It is important for this pipeline because it saves time and reduces the cost of using the LLM. 
# If the transform script processed all 365 records every time, 
# it would repeatedly process the same records, 
# which would increase the cost and could also create duplicate or inconsistent results.


# --- Prompt Design ---
# Q1
TWO_SENTENCE_PROMPT = (
    "You are writing a two-sentence running recommendation for a daily weather summary app. "
    "The first sentence should state the prediction, and the second sentence should explain the reasoning."
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly two sentences - direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."

)

# What I would change:
# From checking for one sentence to checking for exactly two complete sentences.
# I also make sure the first sentence contains the prediction and the second sentence gives the reasoning.


# Q2
import time
def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries + 1):
        try: 
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            raw_summary = response.choices[0].message.content
            return raw_summary
        except Exception as e:
            print(f"Error occurred: {e}")
            if attempt < max_retries:
                print(f"Retrying... ({max_retries - attempt} retries left)")
                time.sleep(2)
            else:
                print("Max retries reached. Returning None.")
                return None
            
# In production I would use this when calling an LLM API because
# temporary errors or network problems can happen. 
# Retrying can help the pipeline recover from temporary failures instead of stopping completely.