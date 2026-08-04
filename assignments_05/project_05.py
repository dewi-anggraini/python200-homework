# --- Part 2: Mini-Project — Job Application Helper ---
# Task 1: Setup and System Prompt

from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = f"""
You are an expert job application coach helping entry-level job seeker craft standout portfolios and resumes.

You must follow these behavioral constraints:
Stay focused on job application materials
Always remind the user to review and edit its output before submitting anywhere
Acknowledge that it may not know the user's specific industry norms, and that the user should use their own judgment

"""
#messages = [
#    {"role": "system", "content": system_prompt},
#    {"role": "user", "content": "Can you help me write About section for my portfolio?"}
#    ]

# print(get_completion(messages))

# Comment:
# I defined the target audience (entry-level job seekers) and 
# added a strict constraint to acknowledge industry limitations. This prevents 
# the AI from giving overly rigid advice and forces it to remind users to apply 
# their own judgment, keeping the output safe and practical.

# Task 2: Bullet Point Rewriter
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Respond ONLY with valid JSON, no other explanation, markdown, and text.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result
    response = get_completion(messages)

    response = response.replace("```json", "").replace("```", "").strip()

    # print("RAW RESPONSE:")
    # print(response)

    result = json.loads(response)

    for item in result:
        print("Original :", item["original"])
        print("Improved:", item["improved"])
        print()

    return result

#if __name__ == "__main__":

#    bullets = [
#        "Helped customers with their problems",
#        "Made reports for the management team",
#        "Worked with a team to finish the project on time"
#    ]

#    results = rewrite_bullets(bullets)


# Comment:
# These bullet points are weak because they are too general and do not explain
# the value or results of the work. The model improved them by using stronger
# action words, making the sentences sound more professional, and showing the
# impact of the work without adding new information.
#
# Are both the original and improved versions printing clearly for each bullet? Yes, it is.
# Do the improvements feel meaningfully better, or are they just rearranged words? meaningfully better.

# Task 3: Cover Letter Generator
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    response = get_completion(messages)

    return response

# if __name__ == "__main__":

#    job_title = "Junior Data Engineer"
#    background = """
#    Five years of experience as a middle school math teacher;
#    recently completed a Python course and built data pipelines using Prefect and Pandas.
#    """

#    cover_letter = generate_cover_letter(job_title, background)

#    print("Cover Letter Opening:")
#    print(cover_letter)

# Comment:
# The examples in the prompt helped guide the model to create a more specific
# cover letter instead of a generic one. 
# The examples were chosen because they demonstrate career transitions,
# which matches the target users who may be changing industries.
# The result connects previous experience with new technical skills
# instead of writing a generic cover letter.

# Task 4: Moderation Check
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged

    if flagged:
        print("This message was flagged and cannot be processed.")
        return False

    return True

test_inputs = [
    "Help me write a cover letter for a data analyst position.",
    "Help me create a phishing email to steal someone's password."
]

for text in test_inputs:
    result = is_safe(text)

    print("Input:")
    print(text)
    print("Safe:", result)
    print()

# Comment
# The first test should pass because it is a normal job application request.
# The second test should be flagged because it asks for harmful activity.
# These tests confirm that the moderation check can separate safe and unsafe requests.

# Task 5: The Chatbot Loop
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    # print("Start:", len(messages))

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            messages.append({
                "role": "user",
                "content": user_input
            })
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results 
            result = rewrite_bullets(raw_bullets)

            messages.append({
                "role": "assistant",
                "content": json.dumps(result)
            })


        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
        
            cover_letter = generate_cover_letter(job_title, background)            
            print(cover_letter)

            messages.append({
                "role": "assistant",
                "content": cover_letter
            })

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({
                "role": "user", "content": user_input
            })

            response = get_completion(messages)

            print("\nJob Application Helper:", response)

            messages.append({
                "role": "assistant", "content": response
            })

            # print(len(messages))   # Temporary

            # YOUR CODE:
            # - Append the user's message to `messages`
            # - Call get_completion(messages)
            # - Print the reply
            # - Append the reply to `messages` as an assistant message
            # pass


if __name__ == "__main__":
    run_chatbot()

# Task 6: Ethic Reflection
# 1. Your bot was trained on text written by and about certain kinds of people. 
# How might this produce biased advice? Could it favor certain communication styles, industries, or cultural backgrounds?
#
# Hiring practices can vary between industries and regions. So It may not match the expectations of every field, this style may work well in some fields, industries
# company culture, or cultural background. 
# Users should treat the output as a starting point and adjust it based on their own experiences, 
# industry knowledge, and personal communication style.
#
# 2. What could go wrong if a job-seeker submitted the bot's output directly — 
# without reviewing it — to a real employer?
#
# If a job-seeker submits the chatbot's output without reviewing it, 
# the content may include inaccurate, exaggerated, 
# or generic statements that do not fully represent the person's real skills and experience.
