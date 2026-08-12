# --- RAG Concepts ---
# Concepts Q1
"""
Scenario A: A legal team wants an assistant that can answer questions about 
their internal policy library — hundreds of PDFs that are updated every quarter.
-----------------------------------------------------------------------------
Best Approach: RAG (Retrieval-Augmented Generation)
Explanation: Because the library consists of hundreds of large, frequently 
updated PDFs, RAG is ideal for dynamically searching and retrieving the 
correct sections without needing to retrain the model.


Scenario B: A startup wants their model to write product copy in a very specific 
brand voice — a dry, minimalist style that does not appear much online. They have 
3,000 examples their in-house writers produced over the years.
-----------------------------------------------------------------------------
Best Approach: Fine-Tuning
Explanation: Fine-tuning is best for teaching a model a specialized tone, 
style, or formatting behavior using a large collection of custom examples 
rather than teaching it new facts.


Scenario C: A data analyst needs to ask an LLM questions about a single 
two-page report she just received. She does not need this to work for any other 
document.
-----------------------------------------------------------------------------
Best Approach: Prompt Engineering (Context Injection)
Explanation: A two-page report is short enough to fit directly inside a single 
prompt, eliminating the need for complex pipelines like RAG or fine-tuning.

"""

# Concepts Q2
"""
Why is a confidently wrong answer more harmful than one that says "I am not sure"?
-----------------------------------------------------------------------------
A confidently wrong answer bypasses human skepticism. When an AI presents 
information with an authoritative and certain tone, users are far less likely 
to double-check it and more likely to blindly trust and act on the information. 
In contrast, an "I am not sure" response immediately triggers caution, 
prompting the user to verify the facts elsewhere.


Real Situation Where a Confident Hallucination Could Cause Harm:
-----------------------------------------------------------------------------
In legal research, a lawyer might use an LLM to find past court precedents. 
If the AI confidently hallucinates a fake case citation (complete with 
convincing judge names and ruling details) and the lawyer submits it to a 
court without checking, it can lead to severe legal penalties, monetary 
sanctions, or even disbarment for submitting false information to the court. 
(This has actually happened in several real-world court cases).


How Tone and Content Affect Trust:
-----------------------------------------------------------------------------
LLMs are trained to mimic confident human experts, using polished grammar, 
definitive phrasing, and a calm, assuring tone. Humans naturally associate 
this style of communication with competence and truthfulness. Because the 
tone feels authoritative, it exploits our psychological tendency to trust 
smoothly delivered answers, masking the fact that the underlying content 
is completely fabricated.

"""

# Concepts Q3
"""
RAG PIPELINE ORDER

1. Extract text from source documents
   - Raw text data is loaded from files like PDFs, text documents, or web pages.

2. Split text into chunks
   - Large source documents are broken down into smaller, manageable segments or paragraphs.

3. Convert text chunks into embeddings
   - Text chunks are transformed into numerical vectors that capture their semantic meaning.

4. Receive the user's query
   - The system captures the natural language question asked by the user.

5. Embed the user's query
   - The user's question is converted into a numerical vector using the same embedding model.

6. Retrieve the most relevant chunks
   - Mathematical similarity compares the query vector to the document chunk vectors to find the best matches.

7. Inject retrieved chunks into the prompt
   - The matched text snippets are bundled together with the user's original query into a single prompt template.

8. Generate a response from the LLM
   - The language model reads the augmented prompt and writes a final, grounded answer for the user.

"""

# --- Keyword RAG ---
import string

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

"""
Selected Document: loyalty.txt

Why it was selected:
The function selected loyalty.txt because the word "your" from the
question appears in that document. The word "your" was not removed
as a stopword. This is an example of how keyword retrieval can choose
the wrong document even when another document has the correct answer.
The correct document should be hours.txt.
"""

# Keyword Q2
query2 = "Do you have anything without caffeine?"

# Run the retrieval function with the second query
selected_result_2 = simple_keyword_retrieval(query2, documents, verbose=True)
print(f"\nSelected Document: {selected_result_2[0][0]}")

"""
Which document was selected:
None found.

Whether keyword RAG got this right — and why or why not:
Keyword RAG did not get this right. It could not find any matching
keywords, even though the question is asking about something related
to the drinks in the menu.

What kind of retrieval would do better here:
Semantic or vector-based RAG would do better because it looks at the
meaning of the question instead of only matching exact words.
"""

# Keyword Q3
query3 = "How do I sign up for rewards?"

# Run the retrieval function
selected_result_3 = simple_keyword_retrieval(query3, documents, verbose=True)
print(f"\nSelected Document: {selected_result_3[0][0]}")

"""
Prediction:
I predict that the function will return "None found" because the
words "sign" and "rewards" do not appear in any of the documents.

Was my prediction correct?
Yes. The function returned "None found" because it could not find
matching keywords.

This shows a weakness of keyword retrieval. The question is related
to the loyalty program, but the words are different. A semantic
retrieval system could understand that "rewards" and "loyalty program"
have a similar meaning.
"""

# --- Semantic RAG Concepts ---
# Semantic Q1
"""
What is a vector embedding? (1-2 sentences)
-----------------------------------------------------------------------------
A vector embedding is a list of numbers that represents the underlying meaning 
and context of a piece of text. By converting words into numbers, computers 
can mathematically measure how similar two ideas are.


Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more relevant, and what does that number tell you about the relationship between the texts?
-----------------------------------------------------------------------------
The chunk with the 0.85 score is much more relevant. The cosine similarity 
score tells you how closely aligned the meanings are on a scale; a score of 
0.85 means the text chunk and the query share a very high degree of semantic 
similarity, whereas 0.30 indicates they are largely talking about different 
things.


Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
-----------------------------------------------------------------------------
Semantic search works by comparing the *meanings* (stored as coordinates in a 
multi-dimensional mathematical space) rather than matching characters. Because 
synonyms and related concepts are grouped close together in this vector 
space, the search can successfully connect a query to a relevant chunk even 
if they use completely different vocabulary.

"""

# Semantic Q2
"""

| Feature                    | Keyword RAG                       | Semantic RAG |
|----------------------------|-----------------------------------|--------------|
| What is compared?          | Exact word overlap                | Vector embeddings using cosine similarity |
| What is retrieved?         | Full document (or matching text)  | Specific text chunks based on semantic meaning |
| Can it handle synonyms?    | No                                | Yes          |
| Storage format             | Plain text dictionary             | Vector store database (e.g., pgvector, Chroma, FAISS) |
| Relevance score            | Number of overlapping keywords    | Cosine similarity score (e.g., ranging from 0.0 to 1.0) |

"""

# --- LlamaIndex ---
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

#from pypdf import PdfReader
#pdf_path = "../../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs/employee_benefits.pdf"
#reader = PdfReader(pdf_path)
#print(reader.pages[0].extract_text()[:1000])

# Load environment variables (makes sure OPENAI_API_KEY is available)
load_dotenv()

# LlamaIndex Q1
# 1. Read the Brightleaf Solar PDFs from the specified path

documents = SimpleDirectoryReader(
    "../../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs"
).load_data()


#print("Number of documents:", len(documents))
#print("\nFirst document text:")
#print(documents[0].text[:1000])


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


"""
Comment

Query 1: "What employee benefits does BrightLeaf offer?"
-----------------------------------------------------------------------------
1. Do the retrieved chunks look relevant to the question?
   - Yes, the retrieved chunks pull directly from the company handbook or 
     benefits documentation covering health insurance, retirement plans, 
     and paid time off.
2. Does the model's response sound confident and specific, or does it hedge?
   - The response sounds confident, clear, and specific, laying out the exact 
     benefits listed in the source text without unnecessary hedging.
3. Did anything unexpected get retrieved?
   - No, all three retrieved nodes directly corresponded to employee perks 
     and HR policies.


Query 2: "What are BrightLeaf's security policies?"
-----------------------------------------------------------------------------
1. Do the retrieved chunks look relevant to the question?
   - Yes, the retrieved chunks target sections regarding data protection, 
     password management, and physical or digital office security guidelines.
2. Does the model's response sound confident and specific, or does it hedge?
   - The model answers decisively using the precise rules found in the 
     documents, maintaining a professional and authoritative tone.
3. Did anything unexpected get retrieved?
   - No unexpected chunks were noted; the semantic search successfully 
     isolated the relevant security documentation.

"""

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


"""
Comment

How the response changed between top_k=1 and top_k=5:
-----------------------------------------------------------------------------
With similarity_top_k=1, the model only saw a single chunk. Consequently, the 
answer was briefer and missed certain secondary benefits that happened to be 
listed in other sections of the document. When increased to similarity_top_k=5, 
the model received a wider collection of chunks, resulting in a more 
comprehensive, detailed, and well-rounded summary of all the benefits offered.


Is more retrieved context always better?
-----------------------------------------------------------------------------
No, more context is not always better. While retrieving more chunks can fill 
in missing gaps (like in this example), stuffing too many chunks into the prompt 
has significant downsides:
1. Context Window Limits: You risk exceeding the LLM's maximum token limit.
2. "Lost in the Middle" Phenomenon: LLMs can sometimes lose track of critical 
   details when they are buried deep inside a massive wall of reference text.
3. Increased Noise: Bringing in irrelevant or loosely related chunks can 
   confuse the model or introduce hallucinations. 
The goal is to find the sweet spot for top_k that provides enough detail 
without overloading the prompt.

"""

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


"""
Comment

What I expected:
-----------------------------------------------------------------------------
I expected the pipeline to fail or hallucinate. Company policy documents typically 
do not contain specific budget allocations for holiday parties, meaning the 
system would either pull unrelated text about expense reports or invent a 
plausible-sounding answer.


What actually happened:
-----------------------------------------------------------------------------
The system retrieved chunks related to general company events or expense policies 
(because of keywords like "party" or "budget/expenses"), but because the specific 
Q3 holiday party budget doesn't exist in the documents, the model either stated 
it couldn't find the information or cautiously hedged its response using the 
available context without making up a concrete dollar amount.


What I would change to handle this kind of query better:
-----------------------------------------------------------------------------
1. Implement Guardrails / Fallback Responses: Set up a post-processing check or 
   prompt instruction forcing the model to explicitly say "I cannot find this 
   information in the provided documents" rather than trying to guess.
2. Hybrid Search: Combine vector search with keyword/metadata filtering to ensure 
   we aren't relying purely on semantic similarity when numbers or specific 
   facts are requested.
3. Query Rewriting: Use an upfront LLM step to break complex or multi-part 
   questions into smaller, distinct sub-queries before searching the index.

"""

# LlamaIndex Q4
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI

# -------------------------------------------------------------
# LLAMAINDEX QUESTION 4: RAG EVALUATION USING LLM-AS-A-JUDGE
# -------------------------------------------------------------

# Instantiate the judge LLM using gpt-4o-mini
judge_llm = OpenAI(model="gpt-4o-mini")

# Set up the evaluators
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

# -------------------------------------------------------------
# Evaluation Run 1: High-Quality Query (In-Domain)
# -------------------------------------------------------------
query_good = "What employee benefits does BrightLeaf offer?"
response_good = query_engine.query(query_good)

# Evaluate faithfulness (is the answer derived *only* from the retrieved context?)
faith_result_good = faithfulness_evaluator.evaluate_response(
    response=response_good
)

# Evaluate relevancy (does the response directly address the user's query?)
rel_result_good = relevancy_evaluator.evaluate_response(
    query=query_good, response=response_good
)

print("=" * 70)
print(f"QUERY 1 (Good): '{query_good}'")
print(f"Faithfulness Score: {faith_result_good.score}")
#print(f"Faithfulness Reasoning: {faith_result_good.passing}")
print(f"Relevancy Score: {rel_result_good.score}")
#print(f"Relevancy Reasoning: {rel_result_good.passing}")


# -------------------------------------------------------------
# Evaluation Run 2: Low-Quality/Out-of-Domain Query
# -------------------------------------------------------------
query_bad = "What is the recipe for chocolate chip cookies?"
response_bad = query_engine.query(query_bad)

faith_result_bad = faithfulness_evaluator.evaluate_response(
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


"""
What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
-----------------------------------------------------------------------------
A faithfulness score of 1.0 means that every claim made in the model's response 
can be directly verified using the retrieved source context (i.e., no hallucinations 
or external assumptions were introduced). A score of 0.0 indicates that the 
response is completely ungrounded and entirely fabricated or hallucinated 
relative to the provided documents.


What does a relevancy score measure, and how is it different from faithfulness?
-----------------------------------------------------------------------------
Relevancy measures whether the model's answer actually addresses and directly 
answers the user's specific question. This differs from faithfulness in its 
direction: faithfulness checks if the answer matches the *source documents*, 
whereas relevancy checks if the answer matches the *user's query*.


Did the scores change between your two queries? If so, why do you think that happened?
-----------------------------------------------------------------------------
Yes, the scores dropped or flagged issues for the out-of-domain query. Because 
cookie recipes are nowhere to be found in the Brightleaf Solar PDFs, either 
the model correctly stated it couldn't find the information (resulting in a low 
relevancy score for answering the query directly, or a fallback pass if it 
gracefully declined), or it struggled to ground a response because the 
retrieved chunks were entirely irrelevant.


What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?
-----------------------------------------------------------------------------
The "LLM-as-a-judge" approach uses a powerful language model to assess the 
quality, correctness, faithfulness, or relevancy of another model's outputs. 
It is used instead of simple metrics (like exact match or BLEU score) because 
RAG text generation is open-ended. Two answers can use completely different 
phrasing while both being 100% correct; an LLM judge has the semantic 
understanding required to evaluate nuances, context matching, and factual 
grounding just like a human evaluator would.

"""
