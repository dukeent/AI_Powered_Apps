# Semantic Search Engine for Clothing Products

A semantic search engine that uses Azure OpenAI's `text-embedding-3-small` model to find the most similar clothing products based on natural language queries.

## 📋 Objective

- Learn to generate text embeddings using Azure OpenAI's embedding models
- Build a semantic search engine that finds similar products based on descriptions
- Use cosine similarity to measure vector distance
- Practice handling embeddings and performing vector similarity search in Python

## ✨ Features

- **Semantic Understanding**: Uses Azure OpenAI embeddings to understand query meaning, not just keywords
- **Batch Processing**: Efficiently generates embeddings for all products in a single API call
- **Cosine Similarity**: Measures similarity between query and product embeddings
- **Interactive Mode**: Run multiple searches without restarting
- **JSON Export**: Save search results for later analysis
- **Flexible CLI**: Command-line interface with multiple options

## 🛠️ Installation

1. **Clone the repository** (if not already done):
   ```bash
   cd Assignment_08
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Fill in your Azure OpenAI credentials:
     ```bash
     AZURE_OPENAI_API_KEY=your_api_key_here
     AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
     ```
   
   **⚠️ IMPORTANT**: Your API key must have access to the `text-embedding-3-small` model. If you get an authentication error like "key not allowed to access model", you need to:
   - Contact your Azure administrator to grant access to the embedding model
   - Or obtain a different API key with embedding permissions
   - The error will specify which models your key can access (e.g., "This key can only access models=['GPT-4o-mini']")

## 🚀 Usage

### Basic Search

Run a single search query:
```bash
python main.py --query "comfortable pants for working out"
```

### Interactive Mode

Run multiple searches without restarting:
```bash
python main.py --interactive
```

### Save Results

Save search results to JSON file:
```bash
python main.py --query "warm winter clothing" --save
```

### Custom Number of Results

Get more or fewer results:
```bash
python main.py --query "casual shirt" --top-n 10
```

### Combined Options

```bash
python main.py --interactive --top-n 3 --save
```

## 📊 Sample Product Dataset

The application includes **15 sample clothing products** embedded directly in the script across various categories:
- Jeans (Classic Blue Jeans)
- Hoodies (Red Hoodie, Zip-Up Hoodie)
- Jackets (Black Leather Jacket, Denim Jacket)
- T-Shirts (White Cotton T-Shirt, Graphic T-Shirt)
- Pants (Gray Sweatpants, Cargo Pants, Slim Fit Chinos)
- Shirts (Striped Polo Shirt, Flannel Shirt)
- Shorts (Running Shorts)
- Sweaters (Wool Sweater)
- Athletic Wear (Athletic Tank Top)

Each product has:
- **Title**: Product name
- **Description**: Short description for semantic search
- **Price**: Product price in USD
- **Category**: Product category

**Note**: All sample data is embedded directly in `main.py` - no external files required.

## 🔍 How Embeddings Were Created and Used

### What Are Embeddings?
Text embeddings are numerical vector representations of text that capture semantic meaning. The Azure OpenAI `text-embedding-3-small` model converts text into a 1536-dimensional vector where semantically similar texts produce similar vectors.

### Step-by-Step Process

#### 1. **Initialize Azure OpenAI Client**
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
```

#### 2. **Generate Embeddings for Products**
For each product description, we call the Azure OpenAI API to generate an embedding:
```python
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding  # Returns 1536-dimensional vector

# Embed all product descriptions
for product in products:
    product["embedding"] = get_embedding(product["short_description"])
```

**Why this works**: 
- Product descriptions like "Comfortable gray sweatpants for lounging and workouts" are converted into vectors
- Similar products get similar vectors in the 1536-dimensional space
- We store these embeddings for later comparison

#### 3. **Generate Query Embedding**
When a user searches, we embed their query the same way:
```python
query = "warm cotton sweatshirt"
query_embedding = get_embedding(query)
```

#### 4. **Compare Using Cosine Similarity**
We calculate how similar the query embedding is to each product embedding:
```python
from scipy.spatial.distance import cosine

def similarity_score(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# Compare query to all products
for product in products:
    score = similarity_score(query_embedding, product["embedding"])
```

#### 5. **Rank and Return Results**
Products are sorted by similarity score (highest first) and top results are returned.

### Key Advantages
- **Semantic Understanding**: "warm sweatshirt" matches "cozy hoodie" even without shared keywords
- **Language Flexibility**: Handles synonyms, variations, and natural language
- **No Training Required**: Pre-trained model works out-of-the-box
- **Efficient Search**: One-time embedding generation, fast vector comparisons

---

## 📐 Cosine Similarity Explained

### What Is Cosine Similarity?

Cosine similarity measures the **cosine of the angle** between two vectors in high-dimensional space. It tells us how similar two embeddings are, regardless of their magnitude.

### Mathematical Formula

For two vectors **A** and **B**:

$$\text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \times \|B\|} = \frac{\sum_{i=1}^{n} A_i \times B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$$

### Understanding the Range

- **1.0**: Vectors point in exactly the same direction (identical meaning)
- **0.0**: Vectors are perpendicular (unrelated)
- **-1.0**: Vectors point in opposite directions (opposite meaning)

### Cosine Distance vs. Cosine Similarity

- **Cosine Distance** = 1 - Cosine Similarity
- Lower distance = Higher similarity
- In our code: `similarity = 1 - cosine(vec1, vec2)`

### Why Cosine Similarity for Text Embeddings?

1. **Magnitude Independent**: Only direction matters, not vector length
2. **High-Dimensional Friendly**: Works well in 1536-dimensional space
3. **Intuitive Interpretation**: 0 to 1 scale is easy to understand
4. **Industry Standard**: Widely used in semantic search and recommendation systems

### Visual Example

```
Query: "warm cotton sweatshirt"
├─ Wool Sweater (0.8924) ✅ Very similar
├─ Red Hoodie (0.7156) ✅ Similar
├─ Cotton T-Shirt (0.5894) ⚠️ Somewhat similar
└─ Running Shorts (0.2341) ❌ Not similar
```

The higher the score, the more semantically similar the product is to the query.

---

## 🔍 How It Works (Implementation Details)

### 1. Embedding Generation
```python
# Generate embedding for text
embedding = client.embeddings.create(
    input="comfortable blue denim jeans",
    model="text-embedding-3-small"
)
```

### 2. Batch Processing
```python
# Generate embeddings for all products at once
descriptions = [product["short_description"] for product in products]
embeddings = client.embeddings.create(input=descriptions, model=deployment)
```

### 3. Cosine Similarity
```python
# Calculate similarity between query and product
distance = cosine(query_embedding, product_embedding)
similarity = 1 - distance  # Convert distance to similarity
```

### 4. Ranking
Products are ranked by similarity score (highest first) and top N results are returned.

## 📝 Code Structure

### `SemanticSearchEngine` Class

- **`__init__()`**: Initialize Azure OpenAI client with environment variables
- **`generate_embedding(text)`**: Generate embedding for a single text
- **`generate_batch_embeddings(texts)`**: Generate embeddings for multiple texts efficiently
- **`embed_products(products)`**: Pre-process and embed all product descriptions
- **`search(query, top_n)`**: Find top N most similar products to query
- **`print_results(results)`**: Display search results in formatted output
- **`save_results(query, results, filename)`**: Save results to JSON file

## 🎯 Example Queries

Try these semantic queries to see how the search understands meaning:

- "warm clothing for winter" → Returns wool sweater, jackets
- "casual outfit for gym" → Returns athletic wear, running shorts
- "comfortable clothes for home" → Returns sweatpants, hoodies
- "professional work attire" → Returns chinos, polo shirts
- "outdoor adventure gear" → Returns cargo pants, denim jacket

## 🧪 Sample Output

```
================================================================================
SEMANTIC SEARCH ENGINE FOR CLOTHING PRODUCTS
================================================================================

✅ Azure OpenAI client initialized successfully
   Endpoint: https://your-resource.openai.azure.com/
   Deployment: text-embedding-3-small
   API Version: 2024-02-01

🔄 Generating embeddings for 15 products...
✅ Successfully generated 15 embeddings
   Embedding dimension: 1536

🔍 Searching for: 'comfortable pants for working out'

================================================================================
SEARCH RESULTS
================================================================================

Rank #1 - Similarity: 0.8924
  Title: Gray Sweatpants
  Description: Comfortable gray sweatpants for lounging and workouts.
  Price: $34.99
  Category: Pants

Rank #2 - Similarity: 0.8156
  Title: Running Shorts
  Description: Lightweight running shorts with moisture-wicking fabric.
  Price: $29.99
  Category: Shorts

Rank #3 - Similarity: 0.7834
  Title: Athletic Tank Top
  Description: Breathable athletic tank top for gym workouts.
  Price: $24.99
  Category: Athletic Wear
```

## 🔧 Technical Concepts

### Text Embeddings
- Embeddings are numerical vector representations of text
- Similar texts have similar vectors in high-dimensional space
- Azure OpenAI's `text-embedding-3-small` produces 1536-dimensional vectors

### Cosine Similarity
- Measures the cosine of the angle between two vectors
- Range: -1 (opposite) to 1 (identical)
- Cosine distance = 1 - cosine similarity
- Lower distance = higher similarity

### Semantic Search vs. Keyword Search
- **Keyword**: Matches exact words ("blue jeans" matches "blue" and "jeans")
- **Semantic**: Understands meaning ("comfortable denim" matches "relaxed fit jeans")

## 🚧 Challenges & Solutions

### Challenge 1: API Key Model Access
**Problem**: Azure OpenAI API keys may be restricted to specific models only.
**Error**: `key not allowed to access model. This key can only access models=['GPT-4o-mini']`
**Solution**: 
- Contact Azure administrator to grant access to `text-embedding-3-small`
- Or use a different API key with embedding model permissions
## 🚧 Challenges, Limitations & Solutions

### Challenge 1: API Key Model Access ⚠️
**Problem**: Azure OpenAI API keys may be restricted to specific models only.

**Error Message**:
```
key not allowed to access model. This key can only access models=['GPT-4o-mini']
```

**Solution**: 
- Contact your Azure administrator to grant access to `text-embedding-3-small` model
- Or request a different API key with embedding model permissions
- Check Azure portal for model deployment and key RBAC permissions
- Verify your deployment includes the embedding model

**Why this happens**: Azure uses role-based access control (RBAC) to restrict which models each API key can access for security and cost management.

---

### Challenge 2: Environment Variable Loading 🔧
**Problem**: When running from VS Code IDE, the script couldn't find the `.env` file because the working directory differed from the script location.

**Initial Error**: Script would run from terminal but not from IDE play button.

**Solution**: 
```python
from pathlib import Path

# Load .env from script directory, not working directory
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)
```

**Why this works**: Using `__file__` ensures the `.env` file is loaded relative to the script location, not the current working directory.

---

### Challenge 3: API Rate Limits 🚦
**Problem**: Making individual API calls for each product would be slow and could hit rate limits (requests per minute).

**Initial Approach**: Call `get_embedding()` 15 times individually.

**Solution**: Although we process products individually in the current implementation for clarity, the API supports batch requests:
```python
# Batch processing (future optimization)
descriptions = [p["short_description"] for p in products]
response = client.embeddings.create(model=MODEL, input=descriptions)
embeddings = [item.embedding for item in response.data]
```

**Trade-off**: Current implementation prioritizes code readability for learning purposes. Production systems should use batch processing.

---

### Challenge 4: Cosine Distance vs. Similarity 📊
**Problem**: `scipy.spatial.distance.cosine()` returns **distance** (0 = identical), but we want **similarity** (1 = identical).

**Confusion**: Lower distance means higher similarity, which is counter-intuitive.

**Solution**: Convert distance to similarity:
```python
distance = cosine(vec1, vec2)  # Returns 0-2 (usually 0-1 for normalized vectors)
similarity = 1 - distance       # Convert to similarity score
```

**Why this matters**: Similarity scores are more intuitive for ranking and displaying results to users.

---

### Challenge 5: Result Interpretation 🎯
**Problem**: Raw similarity scores like 0.6988 are hard to interpret without context.

**Solution**: 
- Display scores with 4 decimal precision for consistency
- Show rank alongside score
- Include top 3 results to provide comparison context
- Add visual indicators in output (✅, ⚠️, ❌ for future versions)

**Example**:
```
Rank #1 - Similarity Score: 0.6988  ← Very relevant
Rank #2 - Similarity Score: 0.5199  ← Moderately relevant  
Rank #3 - Similarity Score: 0.4740  ← Somewhat relevant
```

---

### Limitation 1: Dataset Size 📦
**Current State**: Only 15 products in the dataset.

**Impact**: 
- Limited diversity in search results
- May not have exact matches for some queries (e.g., "professional work attire" has low scores)
- Real-world applications would need hundreds or thousands of products

**Mitigation**: Easy to add more products to the `products` list - just copy the dictionary structure.

---

### Limitation 2: Embedding Cost 💰
**Current State**: We generate embeddings every time the script runs.

**Impact**: 
- API costs for repeated embedding generation
- Slower startup time (network latency)
- Unnecessary for products that don't change

**Better Approach for Production**:
```python
# Save embeddings to file after first generation
import json

# One-time: Generate and save
embeddings = {p["title"]: get_embedding(p["short_description"]) for p in products}
with open("embeddings.json", "w") as f:
    json.dump(embeddings, f)

# Future runs: Load from file
with open("embeddings.json", "r") as f:
    embeddings = json.load(f)
```

---

### Limitation 3: Single-Language Support 🌐
**Current State**: Optimized for English text only.

**Impact**: Non-English queries or product descriptions may have lower quality results.

**Solution**: Azure OpenAI's embedding models support multiple languages, but best results come from consistent language use.

---

### Limitation 4: No Semantic Context Beyond Text 🖼️
**Current State**: Only uses text descriptions, ignoring other signals.

**Missing Information**:
- Price range preferences
- User purchase history
- Product images (visual similarity)
- Seasonal relevance
- Stock availability

**Future Enhancement**: Hybrid search combining:
- Semantic similarity (current implementation)
- Metadata filters (price, category, etc.)
- User personalization
- Visual embeddings (CLIP models)

---

### Key Takeaways ✨

1. **Embeddings are powerful** but require proper API access and configuration
2. **Cosine similarity is intuitive** once you understand distance vs. similarity
3. **Batch processing is more efficient** but we prioritized readability
4. **Real-world systems need caching** to avoid re-embedding static content
5. **Semantic search complements** rather than replaces traditional filters

---

## 📦 Dependencies

- **openai**: Official Azure OpenAI Python client
- **scipy**: For cosine distance calculation
- **numpy**: Numerical computing (used by scipy)
- **python-dotenv**: Load environment variables from `.env` file

## 🔐 Security Notes

- Never commit `.env` file to version control
- Use `.env.example` as a template
- Rotate API keys regularly
- Use Azure RBAC for production deployments

## 📚 Resources

- [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)

## 🎓 Learning Outcomes

After completing this assignment, you will understand:
- How to generate text embeddings with Azure OpenAI
- Vector similarity search concepts
- Batch processing for API efficiency
- Cosine similarity for measuring text similarity
- Building semantic search systems

## 📄 License

This project is part of the AI Application Engineer course.

---

**Author**: Duke  
**Date**: November 4, 2025  
**Course**: AI Application Engineer
