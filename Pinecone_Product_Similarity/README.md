# Pinecone Product Similarity Search

An e-commerce product similarity search system using Pinecone vector database and Azure OpenAI embeddings.

## 📋 Objective

- Initialize and interact with Pinecone vector database using the official Python client
- Set up a Pinecone serverless index and upsert sample product vectors
- Perform similarity search to retrieve the top three most similar products for given queries
- Demonstrate efficient vector-based product recommendations

## 🎯 Problem Statement

In e-commerce and recommendation systems, efficiently retrieving the most similar products based on descriptions or features is crucial for:
- Product recommendations ("Customers who viewed this also viewed...")
- Search result ranking
- Duplicate product detection
- Similar item suggestions

This application uses Pinecone's vector search to store product embeddings and find the nearest neighbors for query embeddings, enabling fast and accurate similarity searches at scale.

## ✨ Features

- **Pinecone Serverless**: AWS-based serverless vector database
- **Azure OpenAI Embeddings**: High-quality text-embedding-3-small (1536 dimensions)
- **5 Sample Products**: Diverse clothing items with descriptions
- **Automated Testing**: 3 pre-configured queries (no manual input)
- **Metadata Support**: Store and retrieve product information
- **Cosine Similarity**: Industry-standard similarity metric
- **Single Python File**: Complete implementation in `main.py`

## 🛠️ Installation

### 1. Install Dependencies

```bash
cd Assignment_10
pip install -r requirements.txt
```

Required packages:
- `pinecone-client>=3.0.0` - Pinecone vector database client
- `openai>=1.0.0` - Azure OpenAI client
- `python-dotenv>=1.0.0` - Environment variable management

### 2. Set Up Pinecone Account

1. Go to https://www.pinecone.io/
2. Sign up for a free account
3. Create a new API key from the dashboard
4. Note your API key for configuration

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 4. Run the Application

```bash
python main.py
```

The script will automatically:
1. Initialize Pinecone and Azure OpenAI clients
2. Create or connect to `product-similarity-index`
3. Generate embeddings for 5 products
4. Upsert vectors to Pinecone
5. Process 3 automated test queries
6. Display top 3 similar products for each query

## 🚀 Usage

### Automated Test Queries

The script includes 3 pre-configured queries:

1. **Summer Query**: "clothing item for summer"
2. **Footwear Query**: "footwear for casual wear"
3. **Winter Query**: "warm clothing for winter"

### Sample Output

```
================================================================================
PINECONE PRODUCT SIMILARITY SEARCH
================================================================================

✅ Configuration loaded
   Azure Endpoint: https://your-resource.openai.azure.com/
   Embedding Model: text-embedding-3-small
   Pinecone API Key: ********************abc1

🔄 Initializing Azure OpenAI client...
✅ Azure OpenAI client initialized

🔄 Initializing Pinecone client...
✅ Pinecone client initialized

🔄 Checking for existing index 'product-similarity-index'...
✅ Index 'product-similarity-index' already exists
🔄 Connecting to index 'product-similarity-index'...
✅ Connected to index

📦 Product Dataset:
   • prod1: Red T-Shirt - Comfortable cotton t-shirt in bright red
   • prod2: Blue Jeans - Stylish denim jeans with relaxed fit
   • prod3: Black Leather Jacket - Genuine leather jacket with classic style
   • prod4: White Sneakers - Comfortable sneakers perfect for daily wear
   • prod5: Green Hoodie - Warm hoodie made of organic cotton

🔄 Generating embeddings and upserting vectors to Pinecone...
   ✓ Embedded: Red T-Shirt
   ✓ Embedded: Blue Jeans
   ✓ Embedded: Black Leather Jacket
   ✓ Embedded: White Sneakers
   ✓ Embedded: Green Hoodie
✅ Successfully upserted 5 product vectors to Pinecone

================================================================================
TEST QUERY 1/3
================================================================================

📝 Search Query: "clothing item for summer"

🔄 Generating query embedding...
✅ Query embedding generated

🔍 Searching Pinecone for top 3 most similar products...
✅ Search completed

🎯 Top 3 Most Similar Products:

   Rank 1:
   • Product ID: prod1
   • Title: Red T-Shirt
   • Description: Comfortable cotton t-shirt in bright red
   • Similarity Score: 0.8234

   Rank 2:
   • Product ID: prod2
   • Title: Blue Jeans
   • Description: Stylish denim jeans with relaxed fit
   • Similarity Score: 0.7891

   Rank 3:
   • Product ID: prod5
   • Title: Green Hoodie
   • Description: Warm hoodie made of organic cotton
   • Similarity Score: 0.7456
```

## 💾 Product Database

| ID | Title | Description |
|----|-------|-------------|
| prod1 | Red T-Shirt | Comfortable cotton t-shirt in bright red |
| prod2 | Blue Jeans | Stylish denim jeans with relaxed fit |
| prod3 | Black Leather Jacket | Genuine leather jacket with classic style |
| prod4 | White Sneakers | Comfortable sneakers perfect for daily wear |
| prod5 | Green Hoodie | Warm hoodie made of organic cotton |

## 🔍 How It Works

### Architecture Overview

```
Product Data
    ↓
[1] Generate Embeddings (Azure OpenAI)
    ↓
[2] Upsert to Pinecone Index
    ↓
User Query
    ↓
[3] Generate Query Embedding
    ↓
[4] Similarity Search (Pinecone)
    ↓
[5] Return Top-K Results
```

### 1. Initialize Pinecone Client

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
```

### 2. Create Serverless Index

```python
index_name = "product-similarity-index"

if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # text-embedding-3-small dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
```

### 3. Generate Embeddings

```python
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding
```

### 4. Upsert Vectors with Metadata

```python
vectors = []
for product in products:
    text = f"{product['title']}: {product['description']}"
    embedding = get_embedding(text)
    vectors.append((
        product["id"],
        embedding,
        {"title": product["title"], "description": product["description"]}
    ))

index.upsert(vectors=vectors)
```

### 5. Query for Similarity

```python
query_embedding = get_embedding("clothing item for summer")

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

for match in results.matches:
    print(f"{match.metadata['title']}: {match.score:.4f}")
```

## 🧪 Technical Concepts

### Vector Embeddings
- Each product → 1536-dimensional vector (text-embedding-3-small)
- Similar products cluster in high-dimensional space
- Semantic similarity: "summer clothing" matches "t-shirt", "light fabrics"

### Pinecone Serverless
- **Serverless Architecture**: No infrastructure management
- **Auto-scaling**: Handles variable workloads automatically
- **Pay-per-use**: Cost-effective for development and testing
- **AWS us-east-1**: Fast, reliable region

### Cosine Similarity
- Metric: `cosine_similarity = (A · B) / (||A|| * ||B||)`
- Range: -1 to 1 (higher = more similar)
- Industry standard for text embeddings
- Invariant to vector magnitude

### Metadata Storage
- Store additional product information alongside vectors
- Filter queries by metadata (price, category, brand)
- Return rich results without separate database lookup

## 🚧 Challenges & Solutions

### Challenge 1: Index Already Exists
**Problem**: Re-running the script fails if index already exists.

**Solution**: Check existing indexes before creation:
```python
existing_indexes = [index["name"] for index in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(...)
```

### Challenge 2: Combining Title and Description
**Problem**: Embedding only title misses important context from description.

**Solution**: Concatenate both fields for richer embeddings:
```python
text_to_embed = f"{product['title']}: {product['description']}"
```

### Challenge 3: Metadata Retrieval
**Problem**: Need product details in search results, not just IDs.

**Solution**: Use `include_metadata=True` in queries:
```python
results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
```

### Challenge 4: Automated Testing
**Problem**: Assignment requires no manual input for grading.

**Solution**: Pre-defined query list with loop:
```python
test_queries = [
    "clothing item for summer",
    "footwear for casual wear",
    "warm clothing for winter"
]

for query in test_queries:
    # Process automatically
```

## 🎓 Learning Outcomes

After completing this assignment, you understand:

1. **Pinecone Vector Database**
   - Serverless index creation and configuration
   - Vector upsert operations
   - Similarity search queries
   - Metadata management

2. **Azure OpenAI Embeddings**
   - Generating semantic vectors
   - text-embedding-3-small model usage
   - Embedding best practices

3. **Similarity Search**
   - Cosine similarity metric
   - Top-k nearest neighbor retrieval
   - Score interpretation

4. **E-commerce Applications**
   - Product recommendation systems
   - Search ranking algorithms
   - Duplicate detection

## 📦 Dependencies

```
pinecone>=5.0.0          # Pinecone vector database client
openai>=1.0.0            # Azure OpenAI client library
python-dotenv>=1.0.0     # Environment variable management
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🔐 Security Notes

- `.env` file contains API keys - **never commit to git**
- Use `.env.example` as template
- `.gitignore` configured to exclude:
  - `.env` (credentials)
  - `__pycache__/` (Python cache)
  - `*.log` (log files)
  - `.DS_Store` (macOS files)
- Rotate API keys regularly
- Pinecone free tier has usage limits - monitor your consumption

## 🔄 Future Enhancements

- **Larger Dataset**: Scale to 1000+ products
- **Hybrid Search**: Combine vector search with keyword filtering
- **Multi-field Embeddings**: Separate embeddings for title, description, reviews
- **Persistent Client**: Use Pinecone persistent storage for production
- **Batch Operations**: Optimize upsert performance with batching
- **Monitoring**: Add logging and performance metrics
- **Web API**: Flask/FastAPI wrapper for HTTP requests
- **Real-time Updates**: Stream product updates to index

## 📚 Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Pinecone Python Client](https://github.com/pinecone-io/pinecone-python-client)
- [Azure OpenAI Embeddings](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/embeddings)
- [Vector Database Guide](https://www.pinecone.io/learn/vector-database/)
- [Cosine Similarity Explained](https://www.pinecone.io/learn/cosine-similarity/)

## 📄 License

This project is part of the AI Application Engineer course.

---

**Author**: Duke  
**Date**: November 12, 2025  
**Course**: AI Application Engineer
