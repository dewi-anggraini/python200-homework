# --- Groundwork Coffee Co. Q&A Assistant ---
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# --- Step 1: Setup ---

# Load environment variables
load_dotenv()
print("API key loaded.")

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
DOCUMENTS_ROOT = REPOSITORY_ROOT.parent

DOCS_DIR = (
    DOCUMENTS_ROOT
    / "python-200-v1"
    / "lessons"
    / "06_AI_augmentation"
    / "resources"
    / "groundwork_docs"
)

assert DOCS_DIR.exists(), f"Document directory not found: {DOCS_DIR}"
print(f"Document directory found: {DOCS_DIR}")


# Comment:
# I used Path(__file__) to build the path relative to this script instead of hard-coding
# a full absolute path. Since the lesson repo is stored separately from my homework repo,
# I go up to the Documents folder and then enter the python-200-v1 lesson folder.


# ---- Step 2: Load the Documents ----
documents = SimpleDirectoryReader(
    input_dir=str(DOCS_DIR)
).load_data()

print(f"\nLoaded {len(documents)} documents.")

print("Files loaded:")

for document in documents:
    print(f"- {document.metadata.get('file_name')}")



# ---- Step 3: Build the Index and Query Engine ----
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("\nIndex built successfully. Ready to answer questions.")



# ---- Step 4: Query the Assistant ----
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

# Run each question through the query engine 
for q in questions:
    print("=" * 70)
    print(f"QUESTION: {q}")
    print("=" * 70)

    response = query_engine.query(q)

    print(f"\nANSWER:\n{response}\n")

    # Get the top retrieved source node
    top_node = response.source_nodes[0]

    document_name = top_node.metadata.get("file_name", "Unknown")
    score = top_node.score if top_node.score is not None else "N/A"
    chunk_text = top_node.text[:200].replace("\n", " ")

    print("TOP RETRIEVED SOURCE:")
    print(f"Document: {document_name}")
    print(f"Similarity Score: {score}")
    print(f"Chunk: {chunk_text}...")
    print()


# Reflection:
# The assistant generally sounded confident and gave answers based on the
# Groundwork Coffee documents. The answers were mostly accurate when the
# information was clearly stated in the documents. Some answers may be
# surprising if the retrieved document contained information I did not
# expect. This shows that it is important to check the retrieved sources
# instead of trusting the answer by itself.



# --- Step 5: Find a Failure ----
failure_query = "What is the population of Tokyo?"

failure_response = query_engine.query(failure_query)

print("=" * 70)
print(f"FAILURE TEST QUESTION: {failure_query}")
print("=" * 70)

print(f"\nFULL RESPONSE:\n{failure_response}\n")

print("ALL THREE RETRIEVED SOURCE NODES:")

for i, node in enumerate(failure_response.source_nodes, 1):
    document_name = node.metadata.get("file_name", "Unknown")
    score = node.score if node.score is not None else "N/A"
    chunk_text = node.text[:200].replace("\n", " ")

    print(f"\nNode {i}:")
    print(f"Document: {document_name}")
    print(f"Similarity Score: {score}")
    print(f"Chunk: {chunk_text}...")


# Failure reflection:
# I asked about the population of Tokyo because this information is not
# related to the Groundwork Coffee documents. I expected the system to
# struggle because the answer should not be found in the documents.
#
# The retriever may still return three documents because it has to choose
# the closest matches, even if none of them are actually relevant. The
# model might then try to answer the question using information that is
# not in the retrieved documents.
#
# This shows that an AI can sometimes sound confident even when it does not
# have good information. We should not trust an answer only because it
# sounds confident. We should also check the retrieved sources.
#
# To improve the system, I would add a similarity score threshold so that
# the system can say "I don't have enough information in the documents"
# instead of trying to answer every question.



# --- Step 6: Reflection ---
# Comment:
#
# 1. The LlamaIndex implementation in my project took only two lines
# to build the query and index, while the manual semantic RAG implementation required
# many more lines for chunking, embedding, and indexing. This shows me
# that frameworks like LlamaIndex can save a lot of time and reduce the
# amount of code we need to write. The framework handles many of the
# complicated steps for us, which makes building a RAG system easier.
#
# 2. One useful business use case would be an employee help system.
# A company could give the system documents about employee benefits,
# vacation policies, health insurance, and workplace rules. Employees
# could ask questions about these policies and get answers based on
# the company's documents. This could save employees time and reduce
# the number of basic questions that HR needs to answer.
#
# 3. One failure mode that RAG cannot fully prevent is the AI giving
# an incorrect or made-up answer even when the correct document was
# retrieved. The model might misunderstand the retrieved information
# or combine it incorrectly. For example, if a company document says
# that employees receive 10 vacation days, the AI could still answer
# that employees receive 15 days. This shows that RAG can help reduce
# incorrect answers, but it cannot guarantee that the AI will always
# give the correct answer.




# --- Optional Extension C ---
# This section is extra and does not replace the required project steps.
# Extension C: Add A New Document

extension_query = "What new product is Groundwork launching?"

extension_response = query_engine.query(extension_query)

print("=" * 70)
print(f"EXTENSION C QUERY: {extension_query}")
print("=" * 70)

print(f"\nANSWER:\n{extension_response}\n")

top_node = extension_response.source_nodes[0]

document_name = top_node.metadata.get("file_name", "Unknown")
score = top_node.score if top_node.score is not None else "N/A"

print("TOP RETRIEVED SOURCE:")
print(f"Document: {document_name}")
print(f"Similarity Score: {score}")

# I added a new document called community_coffee_event.txt.
# It includes information about Groundwork's new Cold Brew Concentrate
# and the Community Coffee Tasting event.
#
# I tested it with the question:
# "What new product is Groundwork launching?"
# The assistant returned the correct answer and retrieved
# community_coffee_event.txt as the top source.
#
# This shows one advantage of RAG over fine-tuning: I can update the assistant
# by adding a new document and rebuilding the index, without retraining the model.
