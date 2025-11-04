"""
Laptop Recommendation Chatbot with RAG
======================================
This application implements a complete RAG (Retrieval-Augmented Generation) system
for laptop recommendations using Azure OpenAI and ChromaDB.

Features:
- Separate embedding and LLM clients for Azure OpenAI
- ChromaDB vector database for semantic search
- Automated testing with mock queries (no manual input)
- RAG pattern: Retrieve relevant laptops, then generate recommendations

Author: Duke
Date: November 4, 2025
"""

# pip install openai chromadb python-dotenv
import chromadb
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

# ---- EMBEDDING CONFIG ----
AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", "")
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "")
AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ---- LLM CONFIG ----
AZURE_OPENAI_LLM_API_KEY = os.getenv("AZURE_OPENAI_LLM_API_KEY", "")
AZURE_OPENAI_LLM_ENDPOINT = os.getenv("AZURE_OPENAI_LLM_ENDPOINT", "")
AZURE_OPENAI_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL", "gpt-4o-mini")

print("="*80)
print("LAPTOP RECOMMENDATION CHATBOT WITH RAG")
print("="*80)
print(f"\n✅ Configuration loaded")
print(f"   Endpoint: {AZURE_OPENAI_EMBEDDING_ENDPOINT}")
print(f"   Embedding Model: {AZURE_OPENAI_EMBED_MODEL}")
print(f"   LLM Model: {AZURE_OPENAI_LLM_MODEL}\n")

# ---- CLIENTS ----
embedding_client = AzureOpenAI(
    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
    api_version="2024-07-01-preview"
)

llm_client = AzureOpenAI(
    api_key=AZURE_OPENAI_LLM_API_KEY,
    azure_endpoint=AZURE_OPENAI_LLM_ENDPOINT,
    api_version="2024-07-01-preview"
)

print("✅ Azure OpenAI clients initialized successfully\n")


# ---- GET EMBEDDING ----
def get_embedding(text):
    """
    Generate embedding for given text using Azure OpenAI embedding model.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    response = embedding_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBED_MODEL
    )
    return response.data[0].embedding


# ---- CALL LLM ----
def ask_llm(context, user_input):
    """
    Query the LLM for laptop recommendations based on context and user requirements.
    
    Args:
        context: Formatted context string with relevant laptop information
        user_input: User's requirements/query
        
    Returns:
        LLM's recommendation response
    """
    system_prompt = (
        "You are a helpful assistant specializing in laptop recommendations. "
        "Use the provided context to recommend the best laptop(s) for the user needs."
    )
    
    user_prompt = (
        f"User requirements: {user_input}\n\n"
        f"Context (top relevant laptops):\n{context}\n\n"
        "Based on the above, which laptop(s) would you recommend and why?"
    )
    
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},  # type: ignore
        {"role": "user", "content": user_prompt}  # type: ignore
    ]
    
    response = llm_client.chat.completions.create(
        model=AZURE_OPENAI_LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content


# ---- CHROMADB ----
print("🔄 Initializing ChromaDB...")
chroma_client = chromadb.Client()

# Delete existing collection if it exists
try:
    chroma_client.delete_collection(name="laptops")
except:
    pass

collection = chroma_client.create_collection(name="laptops")
print("✅ ChromaDB collection 'laptops' created\n")


# --------- Add Sample Laptops ---------
laptops = [
    {
        "id": "1",
        "name": "Gaming Beast Pro",
        "description": "A high-end gaming laptop with RTX 4080, 32GB RAM, and 1TB SSD. Perfect for hardcore gaming.",
        "tags": "gaming, high-performance, windows"
    },
    {
        "id": "2",
        "name": "Business Ultrabook X1",
        "description": "A lightweight business laptop with Intel i7, 16GB RAM, and long battery life. Great for productivity.",
        "tags": "business, ultrabook, lightweight"
    },
    {
        "id": "3",
        "name": "Student Basic",
        "description": "Affordable laptop with 8GB RAM, 256GB SSD, and a reliable battery. Ideal for students and general use.",
        "tags": "student, budget, general"
    },
    {
        "id": "4",
        "name": "Creative Pro Max",
        "description": "High-performance laptop with 4K display, 64GB RAM, and dedicated GPU. Perfect for video editing and 3D rendering.",
        "tags": "creative, professional, high-performance"
    },
    {
        "id": "5",
        "name": "Travel Companion",
        "description": "Ultra-portable laptop weighing only 2 lbs with 20-hour battery life. Ideal for frequent travelers.",
        "tags": "ultraportable, lightweight, long-battery"
    },
    {
        "id": "6",
        "name": "Budget Gaming Starter",
        "description": "Entry-level gaming laptop with GTX 1650, 16GB RAM, and 512GB SSD. Good for casual gaming on a budget.",
        "tags": "gaming, budget, starter"
    },
    {
        "id": "7",
        "name": "Developer Workstation",
        "description": "Powerful laptop with AMD Ryzen 9, 32GB RAM, and dual NVMe SSDs. Perfect for software development and virtualization.",
        "tags": "development, programming, high-performance"
    },
    {
        "id": "8",
        "name": "Office Essential",
        "description": "Reliable business laptop with Intel i5, 8GB RAM, and ergonomic keyboard. Great for office work and meetings.",
        "tags": "business, office, productivity"
    },
    {
        "id": "9",
        "name": "MacBook Air Rival",
        "description": "Sleek aluminum laptop with ARM processor, fanless design, and excellent battery life. Premium build quality.",
        "tags": "ultrabook, premium, fanless"
    },
    {
        "id": "10",
        "name": "2-in-1 Convertible Pro",
        "description": "Versatile touchscreen laptop that converts to tablet mode. Includes stylus for digital art and note-taking.",
        "tags": "convertible, touchscreen, creative"
    }
]


# ---- ADD LAPTOPS TO CHROMADB ----
print("🔄 Adding laptops to ChromaDB with embeddings...")
for laptop in laptops:
    embedding = get_embedding(laptop["description"])
    collection.add(
        embeddings=[embedding],
        documents=[laptop["description"]],
        ids=[laptop["id"]],
        metadatas=[{
            "name": laptop["name"],
            "tags": laptop["tags"]
        }],
    )
    print(f"   ✓ Added: {laptop['name']}")

print(f"\n✅ Successfully added {len(laptops)} laptops to the database\n")


# ---- BUILD CONTEXT FUNCTION ----
def build_context(results, n_context=3):
    """
    Build formatted context string from ChromaDB query results.
    
    Args:
        results: ChromaDB query results
        n_context: Number of results to include (default: 3)
        
    Returns:
        Formatted context string with laptop information
    """
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context_str = ""
    
    for doc, meta in zip(docs, metas):
        context_str += (
            f"Name: {meta['name']}\n"
            f"Description: {doc}\n"
            f"Tags: {meta['tags']}\n\n"
        )
    
    return context_str.strip()


# ---- AUTOMATED MOCK INPUTS (NO MANUAL INPUT) ----
user_queries = [
    "I want a lightweight laptop with long battery life for business trips.",
    "I need a laptop for gaming with the best graphics card available.",
    "Looking for a budget laptop suitable for student tasks and general browsing."
]


# ---- MAIN RAG LOOP ----
print("="*80)
print("RUNNING AUTOMATED TESTS WITH MOCK QUERIES")
print("="*80 + "\n")

for i, user_input in enumerate(user_queries, 1):
    print("="*80)
    print(f"TEST QUERY {i}/{len(user_queries)}")
    print("="*80)
    print(f"\n📝 User input: {user_input}\n")
    
    # Step 1: Retrieve relevant laptops via vector search
    print("🔍 Searching for relevant laptops...")
    query_embedding = get_embedding(user_input)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    # Step 2: Build context for LLM
    context = build_context(results)
    print("✅ Retrieved top 3 matching laptops\n")
    
    # Step 3: Get recommendation from LLM
    print("🤖 Generating LLM recommendation...\n")
    llm_output = ask_llm(context, user_input)
    
    print("💡 LLM Recommendation:\n")
    print(llm_output)
    print("\n" + "="*80 + "\n")


print("="*80)
print("✅ All tests completed successfully!")
print("="*80)
