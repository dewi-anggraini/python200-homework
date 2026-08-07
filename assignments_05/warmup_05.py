# --- The Chat Completions API ---

# API Q1
from dotenv import load_dotenv
from openai import OpenAI
# import os
import json

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

response_model = response.model
tokens_used = response.usage.total_tokens

print("API Q1 Output:")
print(response.choices[0].message.content)
print(f"Model Used: {response_model}")
print(f"Total Tokens: {tokens_used}")

# API Q2
# temperatures = [0, 0.7, 1.5]
load_dotenv()
client = OpenAI()

prompt = "Suggest a creative name for a data engineering consultancy."

# Temperature (0.0)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0
)
print("API Q2 - Temperature 0.0:")
print("Temperature setting: 0.0")
print(f"Response: {response.choices[0].message.content}")

# Temperature (0.7)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7
)
print("API Q2 - Temperature 0.7:")
print("Temperature setting: 0.7")
print(f"Response: {response.choices[0].message.content}")

# Temperature (1.5)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.5
)
print("API Q2 - Temperature 1.5:")
print("Temperature setting: 1.5")
print(f"Response: {response.choices[0].message.content}")

# Comment: What do you notice about how the outputs differ? Which temperature would you use if you needed a consistent, reproducible output?
# At temperature 0.0, the answers were more predictable and repeated similar patterns, or consistent.
# At higher temperatures like 1.5, the responses became more creative but also less consistent.
# For a production application that requires reproducible results, I would choose a lower temperature such as 0.0.


# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
print("API Q3 Output:")
print(f"Model Used: {response.model}")
print(f"Total Tokens: {response.usage.total_tokens}")

for index, choice in enumerate(response.choices):
    print(f"Completion #{index + 1}:")
    print(choice.message.content)
    print("_" * 60)

# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)
print("API Q4 Output:")
print(response.choices[0].message.content)
print(f"Total Tokens: {response.usage.total_tokens}")

# Comment: What happened, and why might you want to use max_tokens in a real application?
# The response was shortened because max_tokens limits how many tokens the model can generate.
# In real applications this can be useful because it controls cost, prevents overly long responses,
# and keeps output within expected limits.

# --- System Messages and Personas ---
# System Q1

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}],
    
)
print("System Q1 - Patient Tutor:")
print(response.choices[0].message.content)

# Different role
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a humorous Python tutor. You explain concepts directly, yet in funny tone, and provide alternative explanations when useful."
        },
        {
            "role": "user",
            "content": "I don't understand what a list comprehension is."
        }
    ]
    
    
)
print("System Q1 - Humorous Tutor:")
print(response.choices[0].message.content)
# Add a comment noting what changed.
# The system message changed the assistant's personality from a patient,
# encouraging Python tutor into a humorous and informative tutor.
# It still answers the same question, but the tone and explanation style
# are different because the system message controls the assistant's behavior.

# System Q2
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}]
    
)
print("System Q2 Output:")
print(response.choices[0].message.content)

# Add a comment: Why does the model know Jordan's name, even though it's stateless?
# To the AI, it isn't "remembering" the past, it's just reading a transcript that I provided right now,
# as a full context.

# --- Prompt Engineering ---
# Prompt Q1
# Zero shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of the following text as positive, negative, or mixed.

Return your answer in exactly this format:

Review 1: <sentiment>
Review 2: <sentiment>
Review 3: <sentiment>

Review 1:
{reviews[0]}

Review 2:
{reviews[1]}

Review 3:
{reviews[2]}

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print("Prompt Q1 Output:")
print(response.choices[0].message.content)


# Prompt Q2 
# One shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify these reviews:

Review 1: {reviews[0]}
Review 2: {reviews[1]}
Review 3: {reviews[2]}

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print("Prompt Q2 Output:")
print(response.choices[0].message.content)
# Comment: Add a comment: Did adding one example change the format or consistency of the output compared to Q1?
#
# Adding one example improved the consistency of the output format
# because the model had a reference for how the answer should look.


# Prompt Q3 
# Few shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of the following text as positive, negative, or mixed.

Example:
Review: "The machine arrived early and works well."
Sentiment: positive

Example:
Review: "The vacuum cleaner isn't working, and there are no instructions."
Sentiment: negative

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now Classify these reviews:

Review 1: {reviews[0]}
Review 2: {reviews[1]}
Review 3: {reviews[2]}

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print("Prompt Q3 Output:")
print(response.choices[0].message.content)

# Comment: Add a comment comparing all three approaches (zero-shot, one-shot, few-shot):
# When would you choose each one?
#
# Zero-shot used when the task is simple with clear instructions
# One-shot is used when the output format matters
# Few shot is used when the task is complex, ambiguous, and when you need consistent pattern-following.

# Prompt Q4

prompt = f"""
Solve the following problem:

Show your step-by-step reasoning, then give the final answer on its own line labelled: 
Final answer: <value>

Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.

What is her final annual salary?

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print("Prompt Q4 Output:")
print(response.choices[0].message.content)
# Comment:
# Asking the model to reason step by step helps break a complex problem
# into smaller calculations, this often leads to more accurate answers
# on multi-step reasoning tasks. it helps reduce mistakes in arithmetic.

# Prompt Q5
review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the review below and return the result only with valid JSON. Do not include any extra text, markdown, or code fences.

The JSON must have these keys:
- sentiment
- confidence (a float from 0 to 1)
- reason (one sentence)

Review:
{review}

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

raw_response = response.choices[0].message.content
print("Prompt Q5 Output:")
print(raw_response)

try:
    result = json.loads(raw_response)

    print("\nParsed fields:")
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except json.JSONDecodeError:
    print("\nError: Response is not valid JSON.")
    print("Raw response for debugging:")
    print(raw_response)

# prompt Q6
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print("Prompt Q6 - First Output:")
print(response.choices[0].message.content)
# second prompt

passage = """The library was quiet in the afternoon, 
and sunlight fell across the tables where a few students were studying."""

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{passage}```

"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print("Prompt Q6 - Second Output:")
print(response.choices[0].message.content)

result = response.choices[0].message.content.strip()
print("Matches expected:", result == "No steps provided.")

# Comment:
# Delimiters help prevent the model from treating the user’s text as instructions. 
# They help separate data from the actual task the model should follow.


# --- Local Models with Ollama ---
# Ollama Q1

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain what a large language model is in two sentences."}
    ]
)
print("OpenAI Response:")
print(response.choices[0].message.content)


# Actual QWeb Ollama output (model: qwen3:0.6b): 
"""
$ ollama run qwen3:0.6b "Explain what a large language model is in two sentences."

A large language model is an AI system trained on vast amounts of text to understand and generate human-like
language. It processes and interprets this information to perform tasks like writing, answering questions, or
creating content, enabling it to understand context and adapt to different languages.

"""

# Comment: 
# What differences did you notice between the two responses?
# What is one advantage and one disadvantage of running a model locally?
#
# # Both OpenAI and the local Qwen3:0.6b model correctly explained what a large
# language model is, but the OpenAI response was more polished and detailed.
# The local model gave a shorter, simpler explanation while still answering
# the prompt correctly.
#
# One advantage of running a model locally is improved privacy because data
# stays on the user's computer and does not need to be sent to an external service.
#
# One disadvantage is that local models often require more computing resources
# and may produce lower-quality responses than larger cloud-hosted models.


# --- Optional / Extention Task ---
# Top-p experiment (add to warmup): 
# Add a question exploring top_p. Set temperature=1.0 and vary top_p between 0.1, 0.5, and 1.0 for the same prompt. 
# Print and compare the results. How does it differ from what you observed when varying temperature?

load_dotenv()
client = OpenAI()

prompt = "Suggest a creative name for a data engineering consultancy."

# top_p = 0.1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.0,
    top_p=0.1
)
print("Top-p setting: 0.1")
print(f"Response: {response.choices[0].message.content}")

# top_p = 0.5
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.0,
    top_p=0.5
)
print("Top-p setting: 0.5")
print(f"Response: {response.choices[0].message.content}")

# top_p = 1.0
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.0,
    top_p=1.0
)
print("Top-p setting: 1.0")
print(f"Response: {response.choices[0].message.content}")

# Comment: 
# top_p controls how many possible next words the model is allowed to consider before choosing one. 
# A lower top_p limits the model to the most likely words, more predictable responses. 
# A higher top_p allows a wider range of word choices, more varied and creative outputs. 
# Compared with changing temperature, 
# changing top_p mainly controls the candidate words rather than the level of randomness in selecting them.