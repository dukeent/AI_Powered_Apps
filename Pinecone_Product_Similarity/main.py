"""
Pinecone Vector Database for Product Similarity Search
======================================================
This application demonstrates using Pinecone vector database for e-commerce
product similarity search using Azure OpenAI embeddings.

Features:
- Initialize Pinecone serverless index
- Generate embeddings using Azure OpenAI text-embedding-3-small
- Upsert product vectors to Pinecone
- Perform similarity search queries
- Retrieve top-k most similar products

Author: Duke
Date: November 12, 2025
"""

# pip install pinecone-client openai python-dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

print("="*80)
print("PINECONE PRODUCT SIMILARITY SEARCH")
print("="*80)

# ---- AZURE OPENAI CONFIG ----
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ---- PINECONE CONFIG ----
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

print(f"\n✅ Configuration loaded")
print(f"   Azure Endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f"   Embedding Model: {AZURE_DEPLOYMENT_NAME}")
print(f"   Pinecone API Key: {'*' * 20}{PINECONE_API_KEY[-4:] if len(PINECONE_API_KEY) > 4 else '****'}\n")

# ---- INITIALIZE CLIENTS ----
print("🔄 Initializing Azure OpenAI client...")
client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)
print("✅ Azure OpenAI client initialized\n")

print("🔄 Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)
print("✅ Pinecone client initialized\n")

# ---- CREATE OR CONNECT TO INDEX ----
index_name = "product-similarity-index"
print(f"🔄 Checking for existing index '{index_name}'...")

existing_indexes = [index["name"] for index in pc.list_indexes()]
print(f"   Existing indexes: {existing_indexes if existing_indexes else 'None'}")

if index_name not in existing_indexes:
    print(f"🔄 Creating new index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimension matches text-embedding-3-small output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"✅ Index '{index_name}' created successfully")
else:
    print(f"✅ Index '{index_name}' already exists")

print(f"🔄 Connecting to index '{index_name}'...")
index = pc.Index(index_name)
print(f"✅ Connected to index\n")

# ---- SAMPLE PRODUCT DATASET ----
products = [
    {
        "id": "prod1",
        "title": "Red T-Shirt",
        "description": "Comfortable cotton t-shirt in bright red"
    },
    {
        "id": "prod2",
        "title": "Blue Jeans",
        "description": "Stylish denim jeans with relaxed fit"
    },
    {
        "id": "prod3",
        "title": "Black Leather Jacket",
        "description": "Genuine leather jacket with classic style"
    },
    {
        "id": "prod4",
        "title": "White Sneakers",
        "description": "Comfortable sneakers perfect for daily wear"
    },
    {
        "id": "prod5",
        "title": "Green Hoodie",
        "description": "Warm hoodie made of organic cotton"
    },
]

print("📦 Product Dataset:")
for p in products:
    print(f"   • {p['id']}: {p['title']} - {p['description']}")
print()

# ---- EMBEDDING FUNCTION ----
def get_embedding(text):
    """
    Generate embedding vector for given text using Azure OpenAI.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats representing the embedding vector (1536 dimensions)
    """
    response = client.embeddings.create(
        input=text,
        model=AZURE_DEPLOYMENT_NAME
    )
    return response.data[0].embedding

# ---- UPSERT PRODUCT VECTORS ----
print("🔄 Generating embeddings and upserting vectors to Pinecone...")
vectors = []
for p in products:
    # Combine title and description for richer embeddings
    text_to_embed = f"{p['title']}: {p['description']}"
    embedding = get_embedding(text_to_embed)
    vectors.append((p["id"], embedding, {"title": p["title"], "description": p["description"]}))
    print(f"   ✓ Embedded: {p['title']}")

# Upsert all vectors at once
index.upsert(vectors=vectors)
print(f"✅ Successfully upserted {len(vectors)} product vectors to Pinecone\n")

# ---- AUTOMATED QUERY TESTING (NO MANUAL INPUT) ----
test_queries = [
    "clothing item for summer",
    "footwear for casual wear",
    "warm clothing for winter"
]

print("="*80)
print("RUNNING AUTOMATED SIMILARITY SEARCH TESTS")
print("="*80 + "\n")

for i, query in enumerate(test_queries, 1):
    print("="*80)
    print(f"TEST QUERY {i}/{len(test_queries)}")
    print("="*80)
    print(f"\n📝 Search Query: \"{query}\"\n")
    
    # Step 1: Generate query embedding
    print("🔄 Generating query embedding...")
    query_embedding = get_embedding(query)
    print("✅ Query embedding generated\n")
    
    # Step 2: Query Pinecone for top 3 similar products
    print("🔍 Searching Pinecone for top 3 most similar products...")
    top_k = 3
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    print("✅ Search completed\n")
    
    # Step 3: Display all results (for debugging)
    print("📊 Raw Results:")
    print(f"   Matches found: {len(results.matches)}\n")  # type: ignore
    
    # Step 4: Display formatted results
    print(f"🎯 Top {top_k} Most Similar Products:\n")
    for rank, match in enumerate(results.matches, 1):  # type: ignore
        product_id = match.id  # type: ignore
        score = match.score  # type: ignore
        metadata = match.metadata  # type: ignore
        
        # Find product details from original dataset
        product = next((p for p in products if p["id"] == product_id), None)
        
        if product:
            print(f"   Rank {rank}:")
            print(f"   • Product ID: {product_id}")
            print(f"   • Title: {product['title']}")
            print(f"   • Description: {product['description']}")
            print(f"   • Similarity Score: {score:.4f}")
            print()
    
    print("="*80 + "\n")

print("="*80)
print("✅ All similarity search tests completed successfully!")
print("="*80)

# ---- SUMMARY STATISTICS ----
print("\n📈 Summary Statistics:")
index_stats = index.describe_index_stats()
print(f"   Total vectors in index: {index_stats.total_vector_count}")
print(f"   Index dimension: {index_stats.dimension}")
print(f"   Index fullness: {index_stats.index_fullness}")
print(f"\n✅ Program completed successfully!")
