# --- Groundwork Coffee Co. Q&A Assistant ---
# Step 1: Setup
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Load environment variables
load_dotenv()

print("Environment variables loaded.")

# Check that the document directory exists
docs_dir = Path("../../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

print(f"Document directory found: {docs_dir}")

# Step 2: Load the Documents
documents = SimpleDirectoryReader(
    input_dir=str(docs_dir)
).load_data()

print(f"\nLoaded {len(documents)} documents.")

print("Files loaded:")
for document in documents:
    print(f"- {document.metadata.get('file_name')}")

# Step 3: Build the Index and Query Engine
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("\nIndex built successfully. Ready to answer questions.")

# Step 4: Query the Assistant
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
    "What time is the Community Coffee Tasting?", # Extension C
    "What new product is Groundwork launching?" # Extension C
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

# Step 5: Find a Failure
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

# Step 6: Reflection
"""
1. The lesson built semantic RAG manually, which required many lines
of code for chunking, embedding, and indexing. In this project, the
LlamaIndex version required much less code. LlamaIndex handled much
of the complicated work for us. This shows that a framework can make
building a RAG system faster and easier, especially when we do not
need to build every part ourselves.

2. One useful business use case would be an employee help system for
a company. The company could give the system documents about employee
benefits, vacation policies, health insurance, and workplace rules.
Employees could then ask questions and get answers based on the
company's documents instead of searching through many files.

3. One failure mode that RAG cannot completely prevent is that the
AI model can still give a wrong or made-up answer. Even if the system
retrieves documents correctly, the model can misunderstand the
information or combine it incorrectly. This means that RAG can reduce
wrong answers, but it cannot guarantee that every answer will be
correct.
"""

# Extension C: Add A New Document

# I added a new document called community_coffee_event.txt.
# It has information about Groundwork's new Cold Brew Concentrate
# and the Community Coffee Tasting event.
#
# I tested it by asking:
# "What new product is Groundwork launching?"
# The assistant gave the correct answer, Cold Brew Concentrate,
# and retrieved community_coffee_event.txt as the top source.
#
# This shows an advantage of RAG because I can add new information
# by adding a new document and rebuilding the index. I do not have
# to retrain the AI model. This makes it easier to update the assistant.
