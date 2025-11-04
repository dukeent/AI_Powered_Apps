"""
Semantic Search Engine for Clothing Products
============================================
This application demonstrates semantic search using Azure OpenAI's text-embedding-3-small model.
It creates embeddings for product descriptions and finds the most similar products to a user query.

Author: Duke
Date: November 4, 2025
"""

import os
from pathlib import Path
from openai import AzureOpenAI
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory as this script
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

# Sample clothing products dataset
products = [
    {
        "title": "Classic Blue Jeans",
        "short_description": "Comfortable blue denim jeans with a relaxed fit.",
        "price": 49.99,
        "category": "Jeans"
    },
    {
        "title": "Red Hoodie",
        "short_description": "Cozy red hoodie made from organic cotton.",
        "price": 39.99,
        "category": "Hoodies"
    },
    {
        "title": "Black Leather Jacket",
        "short_description": "Stylish black leather jacket with a slim fit design.",
        "price": 120.00,
        "category": "Jackets"
    },
    {
        "title": "White Cotton T-Shirt",
        "short_description": "Soft white cotton t-shirt with crew neck.",
        "price": 19.99,
        "category": "T-Shirts"
    },
    {
        "title": "Gray Sweatpants",
        "short_description": "Comfortable gray sweatpants for lounging and workouts.",
        "price": 34.99,
        "category": "Pants"
    },
    {
        "title": "Striped Polo Shirt",
        "short_description": "Classic striped polo shirt in navy and white.",
        "price": 44.99,
        "category": "Shirts"
    },
    {
        "title": "Denim Jacket",
        "short_description": "Vintage-style denim jacket with button closure.",
        "price": 79.99,
        "category": "Jackets"
    },
    {
        "title": "Running Shorts",
        "short_description": "Lightweight running shorts with moisture-wicking fabric.",
        "price": 29.99,
        "category": "Shorts"
    },
    {
        "title": "Wool Sweater",
        "short_description": "Warm wool sweater perfect for cold weather.",
        "price": 89.99,
        "category": "Sweaters"
    },
    {
        "title": "Cargo Pants",
        "short_description": "Utility cargo pants with multiple pockets.",
        "price": 54.99,
        "category": "Pants"
    },
    {
        "title": "Athletic Tank Top",
        "short_description": "Breathable athletic tank top for gym workouts.",
        "price": 24.99,
        "category": "Athletic Wear"
    },
    {
        "title": "Flannel Shirt",
        "short_description": "Soft flannel shirt in red and black plaid pattern.",
        "price": 39.99,
        "category": "Shirts"
    },
    {
        "title": "Slim Fit Chinos",
        "short_description": "Modern slim fit chinos in khaki color.",
        "price": 59.99,
        "category": "Pants"
    },
    {
        "title": "Zip-Up Hoodie",
        "short_description": "Full zip hoodie with front pockets and drawstring hood.",
        "price": 49.99,
        "category": "Hoodies"
    },
    {
        "title": "Graphic T-Shirt",
        "short_description": "Cotton t-shirt with vintage graphic print.",
        "price": 24.99,
        "category": "T-Shirts"
    }
]

# Auto-input queries for testing (no manual input() function)
test_queries = [
    "warm cotton sweatshirt",
    "comfortable pants for working out",
    "casual outfit for gym",
    "professional work attire"
]

# Step 1: Setup Azure OpenAI client
# Get API credentials and model name from environment variables
EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)

print("="*80)
print("SEMANTIC SEARCH ENGINE FOR CLOTHING PRODUCTS")
print("="*80)
print(f"\n✅ Azure OpenAI client initialized successfully")
print(f"   Endpoint: {AZURE_ENDPOINT}")
print(f"   Model: {EMBEDDING_MODEL}")
print(f"   API Version: 2024-07-01-preview\n")


# Step 2: Function to get embeddings from Azure OpenAI
def get_embedding(text):
    """
    Generate embedding for a given text using Azure OpenAI's embedding model.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding vector (1536 dimensions)
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    embedding = response.data[0].embedding
    return embedding


# Step 3: Function to compute cosine similarity
def similarity_score(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors.
    Returns a value between -1 (opposite) and 1 (identical).
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Similarity score (1 - cosine distance)
    """
    return 1 - cosine(vec1, vec2)  # cosine() returns distance; 1 - distance = similarity


def main():
    """Main function to run the semantic search engine."""
    
    # Step 4: Generate embeddings for all product descriptions
    print("🔄 Generating embeddings for all products...\n")
    for i, product in enumerate(products, 1):
        print(f"   Processing product {i}/{len(products)}: {product['title']}")
        product["embedding"] = get_embedding(product["short_description"])
    
    print(f"\n✅ Successfully generated {len(products)} embeddings")
    print(f"   Embedding dimension: {len(products[0]['embedding'])}\n")
    
    # Step 5: Process multiple test queries (auto-input, no manual input())
    print("="*80)
    print("RUNNING SEMANTIC SEARCH TESTS")
    print("="*80 + "\n")
    
    for query in test_queries:
        print(f"{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}\n")
        
        # Step 6: Get embedding for the user query
        query_embedding = get_embedding(query)
        
        # Step 7: Compute cosine similarity between query and each product
        scores = []
        for product in products:
            score = similarity_score(query_embedding, product["embedding"])
            scores.append((score, product))
        
        # Step 8: Sort products by similarity descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Step 9: Display top matches
        print(f"Top matching products for query: '{query}'\n")
        for i, (score, product) in enumerate(scores[:3], 1):  # top 3 results
            print(f"Rank #{i} - Similarity Score: {score:.4f}")
            print(f"  Title: {product['title']}")
            print(f"  Description: {product['short_description']}")
            print(f"  Price: ${product['price']:.2f}")
            print(f"  Category: {product['category']}")
            print()
    
    print("="*80)
    print("✅ All searches completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
