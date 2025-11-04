# Laptop Recommendation Chatbot with RAG

An intelligent laptop recommendation system using RAG (Retrieval-Augmented Generation) pattern with Azure OpenAI and ChromaDB vector database.

## 📋 Objective

- Develop a chatbot that recommends laptops based on user requirements
- Implement RAG pattern: Retrieve relevant laptops → Generate AI recommendations
- Utilize separate Azure OpenAI clients for embeddings and chat completions
- Use ChromaDB for vector storage and semantic similarity search
- Provide automated testing with mock queries (no manual input required)

## 🎯 Problem Statement

Selecting the right laptop can be overwhelming with countless options available. Users need intelligent recommendations that understand their specific needs like gaming performance, business productivity, creative work capabilities, or budget constraints.

This application solves this by:
1. Converting laptop descriptions into semantic vector embeddings
2. Finding the most relevant laptops using similarity search
3. Generating personalized recommendations with AI-powered explanations

## ✨ Features

- **RAG Architecture**: Retrieval-Augmented Generation for accurate recommendations
- **Separate Azure Clients**: Independent embedding and LLM clients for optimal performance
- **Semantic Search**: ChromaDB vector database for intelligent laptop matching
- **10 Laptop Database**: Diverse selection across gaming, business, creative, and student categories
- **Automated Testing**: 3 mock queries for hands-free testing (no input() required)
- **Single Python File**: Complete implementation in `main.py`
- **Secure Configuration**: Environment variable management with separate embedding/LLM credentials

## 🛠️ Installation

### 1. Install Dependencies

```bash
cd Assignment_09
pip install -r requirements.txt
```

Required packages:
- `openai>=1.0.0` - Azure OpenAI client
- `chromadb>=0.4.0` - Vector database
- `python-dotenv>=1.0.0` - Environment variable management

### 2. Set Up Environment Variables

Copy `.env.example` to `.env` and add your Azure OpenAI credentials:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
# Azure OpenAI Embedding Configuration
AZURE_OPENAI_EMBEDDING_API_KEY=your_embedding_api_key_here
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-embedding-resource-name.openai.azure.com/
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Azure OpenAI LLM Configuration
AZURE_OPENAI_LLM_API_KEY=your_llm_api_key_here
AZURE_OPENAI_LLM_ENDPOINT=https://your-llm-resource-name.openai.azure.com/
AZURE_OPENAI_LLM_MODEL=gpt-4o-mini
```

**Note**: You can use the same API key and endpoint for both if using a single Azure OpenAI resource.

### 3. Run the Application

```bash
python main.py
```

The script will automatically:
1. Initialize Azure OpenAI clients (embedding + LLM)
2. Create ChromaDB collection
3. Embed and store 10 laptops
4. Process 3 automated test queries
5. Display recommendations for each query

## 🚀 Usage

### Running the Script

Simply execute:
```bash
python main.py
```

### Automated Test Queries

The script includes 3 pre-configured queries:

1. **Business Query**: "I want a lightweight laptop with long battery life for business trips."
2. **Gaming Query**: "I need a laptop for gaming with the best graphics card available."
3. **Student Query**: "Looking for a budget laptop suitable for student tasks and general browsing."

### Sample Output

```
================================================================================
LAPTOP RECOMMENDATION CHATBOT WITH RAG
================================================================================

✅ Configuration loaded
   Endpoint: https://your-resource.openai.azure.com/
   Embedding Model: text-embedding-3-small
   LLM Model: gpt-4o-mini

✅ Azure OpenAI clients initialized successfully

🔄 Initializing ChromaDB...
✅ ChromaDB collection 'laptops' created

🔄 Adding laptops to ChromaDB with embeddings...
   ✓ Added: Gaming Beast Pro
   ✓ Added: Business Ultrabook X1
   [...]

================================================================================
TEST QUERY 1/3
================================================================================

📝 User input: I want a lightweight laptop with long battery life for business trips.

🔍 Searching for relevant laptops...
✅ Retrieved top 3 matching laptops

🤖 Generating LLM recommendation...

💡 LLM Recommendation:

Based on your requirements, I recommend the **Business Ultrabook X1** as your top choice.
This laptop excels in both portability and battery life, making it perfect for business travel.
[...]
```

## 💾 Laptop Database

The application includes 10 diverse laptops:

| ID | Name | Category | Key Features |
|----|------|----------|--------------|
| 1 | Gaming Beast Pro | Gaming | RTX 4080, 32GB RAM, 1TB SSD |
| 2 | Business Ultrabook X1 | Business | Intel i7, lightweight, long battery |
| 3 | Student Basic | Budget | 8GB RAM, affordable, reliable |
| 4 | Creative Pro Max | Creative | 4K display, 64GB RAM, dedicated GPU |
| 5 | Travel Companion | Ultraportable | 2 lbs, 20-hour battery |
| 6 | Budget Gaming Starter | Gaming/Budget | GTX 1650, entry-level gaming |
| 7 | Developer Workstation | Development | AMD Ryzen 9, 32GB, dual NVMe |
| 8 | Office Essential | Business | Intel i5, ergonomic keyboard |
| 9 | MacBook Air Rival | Premium | ARM processor, fanless design |
| 10 | 2-in-1 Convertible Pro | Creative | Touchscreen, stylus, tablet mode |

Each laptop includes detailed description and tags for semantic matching.

## 🔍 How It Works (RAG Pattern)

### Architecture Overview

```
User Query
    ↓
[1] get_embedding(query)
    ↓
[2] ChromaDB similarity search
    ↓
[3] build_context(top_3_laptops)
    ↓
[4] ask_llm(context, query)
    ↓
AI Recommendation
```

### 1. Separate Azure OpenAI Clients

```python
# Embedding client for vector generation
embedding_client = AzureOpenAI(
    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
    api_version="2024-07-01-preview"
)

# LLM client for chat completions
llm_client = AzureOpenAI(
    api_key=AZURE_OPENAI_LLM_API_KEY,
    azure_endpoint=AZURE_OPENAI_LLM_ENDPOINT,
    api_version="2024-07-01-preview"
)
```

### 2. Embedding Function

```python
def get_embedding(text):
    response = embedding_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBED_MODEL
    )
    return response.data[0].embedding
```

### 3. ChromaDB Storage

```python
# Create collection
collection = chroma_client.create_collection(name="laptops")

# Add laptops with embeddings
for laptop in laptops:
    embedding = get_embedding(laptop["description"])
    collection.add(
        embeddings=[embedding],
        documents=[laptop["description"]],
        ids=[laptop["id"]],
        metadatas=[{"name": laptop["name"], "tags": laptop["tags"]}]
    )
```

### 4. Semantic Retrieval

```python
# Query for similar laptops
query_embedding = get_embedding(user_input)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3  # Top 3 matches
)
```

### 5. Context Building

```python
def build_context(results):
    context_str = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_str += f"Name: {meta['name']}\n"
        context_str += f"Description: {doc}\n"
        context_str += f"Tags: {meta['tags']}\n\n"
    return context_str
```

### 6. LLM Recommendation

```python
def ask_llm(context, user_input):
    messages = [
        {"role": "system", "content": "You are a laptop recommendation specialist..."},
        {"role": "user", "content": f"User: {user_input}\n\nContext:\n{context}"}
    ]
    
    response = llm_client.chat.completions.create(
        model=AZURE_OPENAI_LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content
```

## 🧪 Technical Concepts

### RAG (Retrieval-Augmented Generation)
- **Retrieve**: Find relevant laptops using semantic similarity
- **Augment**: Build context with retrieved laptop information
- **Generate**: Create personalized recommendations using LLM with context

### Vector Embeddings
- Each laptop description → 1536-dimensional vector (text-embedding-3-small)
- Similar concepts have similar vectors in high-dimensional space
- Enables semantic matching: "gaming" matches "RTX graphics", "high-performance"

### Separate Clients Pattern
- **Embedding Client**: Dedicated to vector generation (efficient, focused)
- **LLM Client**: Dedicated to chat completions (optimal configuration)
- Allows different API keys, endpoints, or rate limits per service

### ChromaDB
- In-memory vector database for fast similarity search
- Cosine similarity for finding nearest neighbors
- Automatic distance calculation between query and stored embeddings
- Metadata storage for filtering and enrichment

## 🚧 Challenges & Solutions

### Challenge 1: Type Annotations with OpenAI SDK
**Problem**: Pylance strict type checking errors for chat completion messages.

**Solution**: Import proper types and add type hints:
```python
from openai.types.chat import ChatCompletionMessageParam

messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": system_prompt},  # type: ignore
    {"role": "user", "content": user_prompt}  # type: ignore
]
```

### Challenge 2: Separate Environment Variables
**Problem**: Need distinct credentials for embedding and LLM services.

**Solution**: Use explicit naming convention:
- `AZURE_OPENAI_EMBEDDING_API_KEY` / `AZURE_OPENAI_EMBEDDING_ENDPOINT`
- `AZURE_OPENAI_LLM_API_KEY` / `AZURE_OPENAI_LLM_ENDPOINT`

### Challenge 3: ChromaDB Collection Persistence
**Problem**: Collection may already exist from previous runs.

**Solution**: Delete and recreate collection on each run:
```python
try:
    chroma_client.delete_collection(name="laptops")
except:
    pass
collection = chroma_client.create_collection(name="laptops")
```

### Challenge 4: Automated Testing (No input())
**Problem**: Assignment requires no manual input for grading.

**Solution**: Use pre-defined query list with loop:
```python
user_queries = [
    "I want a lightweight laptop with long battery life...",
    "I need a laptop for gaming with the best graphics...",
    "Looking for a budget laptop suitable for student..."
]

for user_input in user_queries:
    # Process each query automatically
```

## 🎓 Learning Outcomes

After completing this assignment, you understand:

1. **RAG Pattern Implementation**
   - Retrieval: Semantic search with vector databases
   - Augmentation: Context building from retrieved results
   - Generation: LLM-powered recommendations with context

2. **Azure OpenAI Integration**
   - Separate clients for embeddings vs chat completions
   - Embedding generation with text-embedding-3-small
   - Chat completions with GPT-4o-mini
   - Environment variable configuration

3. **Vector Databases**
   - ChromaDB collection management
   - Embedding storage and retrieval
   - Similarity search with cosine distance
   - Metadata enrichment

4. **Production-Ready Code**
   - Single Python file deployment
   - Automated testing without manual input
   - Proper error handling and type hints
   - Secure credential management

## 📦 Dependencies

```
openai>=1.0.0          # Azure OpenAI client library
chromadb>=0.4.0        # Vector database for similarity search
python-dotenv>=1.0.0   # Environment variable management
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🔐 Security Notes

- `.env` file contains API keys - **never commit to git**
- Use `.env.example` as template for others
- `.gitignore` configured to exclude:
  - `.env` (credentials)
  - `__pycache__/` (Python cache)
  - `chroma_db/` (database files)
  - `.DS_Store` (macOS files)
- Rotate API keys regularly
- Use separate keys for different environments (dev/prod)

## 🔄 Future Enhancements

- **Advanced Filtering**: Price range, brand, and category filters in ChromaDB queries
- **Multi-Modal**: Add image embeddings for visual laptop comparison
- **Larger Database**: Expand to 50+ laptops with real-world data
- **Persistent Storage**: Use ChromaDB persistent client for data retention
- **API Endpoint**: Flask/FastAPI wrapper for web integration
- **Streaming Responses**: Use OpenAI streaming for real-time LLM output
- **A/B Testing**: Compare different embedding models and LLM prompts

## 📚 Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Chat Completions Guide](https://platform.openai.com/docs/guides/chat)
- [Vector Embeddings Explained](https://platform.openai.com/docs/guides/embeddings)

## 📄 License

This project is part of the AI Application Engineer course.

---

**Author**: Duke  
**Date**: November 4, 2025  
**Course**: AI Application Engineer
