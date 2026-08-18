
import os
import string
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---
# Concepts Q1

# Scenario A: RAG
# RAG is best here because the assistant needs to search through many documents
# and use the most current information. Since the PDFs change regularly, retrieving
# the right source text is better than retraining the model each time.

# Scenario B: Fine-tuning
# Fine-tuning is best because the company has many examples of the exact style it
# wants. Training on those examples can teach the model to consistently write in
# that brand voice.

# Scenario C: Prompt engineering
# Prompt engineering is the best choice because there is only one short report.
# The report can be given to the model along with the question, so there is no
# need to train the model or build a large document database.


# Concepts Q2
# A confidently wrong answer can be more harmful because people may believe
# it is correct and act on it without checking. Saying "I am not sure" tells
# the user that the answer might be wrong, so they are more likely to check it.
#
# For example, if an AI gives a confident but wrong medical answer, a person
# might follow the advice and make their health worse. The confident tone can
# make the wrong answer seem trustworthy, even when the information is false.
#
# The way an AI sounds affects how much we trust it. If it sounds certain and
# professional, we may assume it knows the answer, even when it does not.


# Concepts Q3
#
# 1. Extract text from source documents — load the text from PDFs or other files.
# 2. Split text into chunks — break the documents into smaller pieces for retrieval.
# 3. Convert text chunks into embeddings — turn each chunk into vectors that capture meaning.
# 4. Receive the user's query — get the question from the user.
# 5. Embed the user's query — convert the question into a vector.
# 6. Retrieve the most relevant chunks — compare the query vector to the chunk vectors and select the best matches.
# 7. Inject retrieved chunks into the prompt — add the selected chunks to the LLM input.
# 8. Generate a response from the LLM — the model writes the final answer.
#
# Just for my undestanding: Easy way for me to remember it
# Documents → Chunks → Embeddings → Question → Search → Prompt → Answer


# --- Keyword RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

    
# Keyword Q1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# Run the retrieval function
selected_result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"\nSelected Document: {selected_result[0][0]}")

# Selected document: loyalty.txt, this was surprising because hours.txt is the document that actually answers the question.
# hours.txt matches "weekends," while hiring.txt and loyalty.txt each match the common word "your," 
# so all three documents score 1. Because the tuples are sorted in reverse order, loyalty.txt wins the
# alphabetical tie-break. This exposes two limitations: the stopword list is
# incomplete, and the tie-break is unrelated to meaning.



# Keyword Q2
query2 = "Do you have anything without caffeine?"

# Run the retrieval function with the second query
selected_result_2 = simple_keyword_retrieval(query2, documents, verbose=True)
print(f"\nSelected Document: {selected_result_2[0][0]}")

# No document was selected because none of the query words exactly matched words 
# in the documents after filtering. Although menu.txt is the most semantically relevant document, 
# keyword RAG cannot recognize that ‘without caffeine’ is related 
# to drink options like decaf or non-coffee beverages. 
# Semantic retrieval using embeddings would work better because it compares meaning 
# rather than exact keywords.


# Keyword Q3
# Prediction: loyalty.txt
# I think this because the query is about rewards, and the loyalty document is the most likely
# place to describe a rewards program.

query3 = "How do I sign up for rewards?"

# Run the retrieval function
selected_result_3 = simple_keyword_retrieval(query3, documents, verbose=True)
print(f"\nSelected Document: {selected_result_3[0][0]}")

# My prediction was loyalty.txt because "rewards" sounds most related to a loyalty program.
# However, the function returned None found because it only matches exact keywords, and the
# loyalty document uses different words like "loyalty," "points," and "redeem" instead of "rewards"
# or "sign up." This shows that keyword retrieval can miss conceptually relevant documents when
# the wording is different.



# --- Semantic RAG Concepts ---
# Semantic Q1
# 1. What is a vector embedding?
# A vector embedding is a way of turning text into numbers that represent
# its meaning. Texts with similar meanings usually have similar numbers.
#
# 2. Which chunk is more relevant?
# The chunk with a cosine similarity score of 0.85 is more relevant than
# the chunk with a score of 0.30. A higher score means the meaning of the
# chunk is more similar to the meaning of the user's question.
#
# 3. Why can semantic search find a chunk without the exact words?
# Semantic search looks at the meaning of the words, not just the exact words.
# For example, a search for "How do I fix my car?" could find a chunk about
# "repairing a vehicle" even though the exact words are different.


# Semantic Q2
# | Feature                 | Keyword RAG                    | Semantic RAG                     |
# |-------------------------|--------------------------------|----------------------------------|
# | What is compared?       | Exact word overlap             | Embedding similarity             |
# | What is retrieved?      | Full document                  | Most relevant text chun          |
# | Can it handle synonyms? | No                             | Yes                              |
# | Storage format          | Plain text dictionary          | Vector store / index of embeding |           |
# | Relevance score         | Number of overlapping keywords | Cosine similarity score          |




# --- LlamaIndex ---

# Load environment variables (makes sure OPENAI_API_KEY is available)
load_dotenv()

# LlamaIndex Q1
# 1. Read the Brightleaf Solar PDFs from the specified path
SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
DOCUMENTS_ROOT = REPOSITORY_ROOT.parent

docs_dir = (
    DOCUMENTS_ROOT
    / "python-200-v1"
    / "lessons"
    / "06_AI_augmentation"
    / "resources"
    / "brightleaf_pdfs"
)

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

documents = SimpleDirectoryReader(docs_dir).load_data()

# 2. Build the in-memory VectorStoreIndex (handles chunking, embedding, and storing)
index = VectorStoreIndex.from_documents(documents)


# 3. Create a query engine configured to retrieve the top 3 most relevant chunks
query_engine = index.as_query_engine(similarity_top_k=3)


# Define the questions to run
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]


# Execute queries and inspect results
for q in questions:
    print("=" * 70)
    print(f"QUESTION: {q}")
    print("=" * 70)

  # Get the response from the model
    response = query_engine.query(q)
    print(f"\n[MODEL ANSWER]\n{response}\n")

    print("[RETRIEVED SOURCE NODES (Top 3)]")
    # Retrieve the source nodes used to generate the answer
    for i, node in enumerate(response.source_nodes, 1):
        score = node.score if node.score is not None else "N/A"
        # Get the first 150 characters of the chunk text
        chunk_snippet = node.text[:150].replace("\n", " ")
        print(f"  - Node {i} | Similarity Score: {score}")
        print(f"    Snippet: {chunk_snippet}...\n")


# Comment:
#
# Query 1 - "What employee benefits does BrightLeaf offer?"
# The retrieved chunks were mostly relevant because the top chunk discussed
# BrightLeaf's employee benefits directly. The model sounded confident and
# specific because it listed several benefits from the documents. However, one
# of the lower-ranked chunks was about the company overview, so it was less
# relevant to the question.
#
# Query 2 - "What are BrightLeaf's security policies?"
# The retrieved chunks were also mostly relevant for this query because the top
# chunk discussed network and data security. The model sounded confident and
# detailed because it gave several security-related details. One of the
# retrieved chunks about employee benefits was unexpected and not directly
# relevant to the security question.


# LiamaIndex Q2
target_query = "What employee benefits does BrightLeaf offer?"

print("*" * 70)
print(f"TESTING WITH similarity_top_k = 1 for: '{target_query}'")
print("*" * 70)

# Configure query engine with top_k = 1
query_engine_top1 = index.as_query_engine(similarity_top_k=1)
response_top1 = query_engine_top1.query(target_query)

print(f"\n[MODEL ANSWER (top_k=1)]\n{response_top1}\n")
print("[RETRIEVED SOURCE NODES (Top 1)]")
for i, node in enumerate(response_top1.source_nodes, 1):
  score = node.score if node.score is not None else "N/A"
  print(
      f"  - Node {i} | Score: {score} | Text: {node.text[:100].replace(chr(10), ' ')}..."
  )


print("\n" + "*" * 70)
print(f"TESTING WITH similarity_top_k = 5 for: '{target_query}'")
print("*" * 70)

# Configure query engine with top_k = 5
query_engine_top5 = index.as_query_engine(similarity_top_k=5)
response_top5 = query_engine_top5.query(target_query)

print(f"\n[MODEL ANSWER (top_k=5)]\n{response_top5}\n")
print("[RETRIEVED SOURCE NODES (Top 5)]")
for i, node in enumerate(response_top5.source_nodes, 1):
  score = node.score if node.score is not None else "N/A"
  print(
      f"  - Node {i} | Score: {score} | Text: {node.text[:100].replace(chr(10), ' ')}..."
  )

# Comment:
#
# With similarity_top_k=1, the model retrieved only the most relevant chunk
# with a score of about 0.91. The answer was already detailed and focused
# on employee benefits.
#
# With similarity_top_k=5, the model retrieved more information and gave
# a slightly more detailed answer. However, some of the extra chunks were
# about security, a business partnership, and financial performance, which
# were not directly related to employee benefits.
#
# This shows that more retrieved context is not always better. Extra
# information can be unrelated and may confuse the model or add unnecessary
# details. In this case, the top 1 result was already enough to give a good
# answer.


# LlamaIndex Q3
# A query about something not likely covered or requiring synthesis across disparate parts
challenging_query = (
    "How much budget was allocated to the marketing team for the Q3 holiday"
    " party, and what are the rules on bringing guests?"
)

print("*" * 70)
print(f"CHALLENGING QUERY: '{challenging_query}'")
print("*" * 70)

# Using top_k = 3 to see what the system grabs
query_engine_challenge = index.as_query_engine(similarity_top_k=3)
challenge_response = query_engine_challenge.query(challenging_query)

print(f"\n[MODEL ANSWER]\n{challenge_response}\n")

print("[RETRIEVED SOURCE NODES]")
for i, node in enumerate(challenge_response.source_nodes, 1):
  score = node.score if node.score is not None else "N/A"
  chunk_snippet = node.text[:150].replace("\n", " ")
  print(f"  - Node {i} | Similarity Score: {score}")
  print(f"    Snippet: {chunk_snippet}...\n")

# Comment:
#
# I expected this query to be difficult because it asks about two specific
# pieces of information: the Q3 holiday party budget and the rules for guests.
#
# The model gave a confident and specific answer, saying the budget was $600
# and employees could bring one guest. However, the retrieved chunks do not
# appear to contain information about the holiday party, so the answer may
# be a hallucination.
#
# The retrieved chunks were also not very relevant. They were about employee
# benefits, security, and financial performance. To handle this better, I
# would add a way to detect when the retrieved chunks are not relevant enough
# and have the system say that it cannot find the answer instead of guessing.
# I could also improve the document chunking and retrieval settings so the
# system can find the correct information when it exists in the documents.


# LlamaIndex Q4
# -------------------------------------------------------------
# RAG EVALUATION USING LLM-AS-A-JUDGE
# -------------------------------------------------------------

# Instantiate the judge LLM using gpt-4o-mini
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)


# Set up the evaluators
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)


# -------------------------------------------------------------
# Evaluation Run 1: High-Quality Query (In-Domain)
# -------------------------------------------------------------
query_good = "What employee benefits does BrightLeaf offer?"
response_good = query_engine.query(query_good)

# Evaluate faithfulness (is the answer derived *only* from the retrieved context?)
faith_result_good = faithfulness_evaluator.evaluate_response(
    query=query_good,
    response=response_good
)

# Evaluate relevancy (does the response directly address the user's query?)
rel_result_good = relevancy_evaluator.evaluate_response(
    query=query_good, response=response_good
)

print("=" * 70)
print(f"QUERY 1 (Good): '{query_good}'")
print(f"Faithfulness Score: {faith_result_good.score}")
print(f"Relevancy Score: {rel_result_good.score}")



# -------------------------------------------------------------
# Evaluation Run 2: Low-Quality/Out-of-Domain Query
# -------------------------------------------------------------
query_bad = "What is the recipe for chocolate chip cookies?"
response_bad = query_engine.query(query_bad)

faith_result_bad = faithfulness_evaluator.evaluate_response(
    query=query_bad,
    response=response_bad
)

rel_result_bad = relevancy_evaluator.evaluate_response(
    query=query_bad, response=response_bad
)

print("=" * 70)
print(f"QUERY 2 (Challenging/Out-of-Domain): '{query_bad}'")
print(f"Faithfulness Score: {faith_result_bad.score}")
print(f"Relevancy Score: {rel_result_bad.score}")
print("=" * 70)

# Comment:
#
# A faithfulness score of 1.0 means the answer is fully supported by the
# information in the retrieved documents. A score of 0.0 means the answer
# is not supported by the retrieved information and may contain a hallucination.
#
# A relevancy score measures how well the answer answers the user's question.
# Faithfulness checks whether the answer is supported by the retrieved
# information, while relevancy checks whether the answer is related to and
# answers the question.
#
# The scores changed between the two queries. The employee benefits question
# received 1.0 for both faithfulness and relevancy because the BrightLeaf
# documents contained information about employee benefits. The cookie recipe
# question received 0.0 for both because the information was not in the
# BrightLeaf documents and was unrelated to the documents.
#
# LLM-as-a-judge means using another LLM to evaluate an AI-generated answer.
# It is useful for RAG because there can be many different correct ways to
# answer a question, so a simple accuracy score may not work well. An LLM
# judge can check whether the answer is supported by the sources and whether
# it actually answers the question.
