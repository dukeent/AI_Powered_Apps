# Walmart RAG Chatbot - Retail Support Assistant

## Assignment 14 - Retrieval-Augmented Generation for Retail Operations

A production-ready RAG (Retrieval-Augmented Generation) chatbot that helps Walmart employees and customers get instant, accurate answers about product policies, warranties, returns, and store services.

---

## 📋 Table of Contents

1. [Objective](#objective)
2. [Problem Statement](#problem-statement)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Architecture](#architecture)
7. [Policy Database](#policy-database)
8. [Key Features](#key-features)
9. [RAG Pipeline Explained](#rag-pipeline-explained)
10. [Test Queries](#test-queries)
11. [Technical Implementation](#technical-implementation)
12. [Learning Outcomes](#learning-outcomes)
13. [Business Impact](#business-impact)
14. [Future Enhancements](#future-enhancements)

---

## 🎯 Objective

Develop a Retrieval-Augmented Generation (RAG) chatbot that:
- Helps retail employees and customers quickly get accurate answers
- References in-memory Walmart policy documents
- Generates clear, human-friendly responses using Azure OpenAI
- Improves service efficiency and customer satisfaction

---

## 🔍 Problem Statement

**Challenges:**
- Retail staff and customers have questions about product specifications, warranties, in-store services, and return/exchange policies
- Staff spend time searching manuals or waiting for approvals
- Customers face longer wait times without instant access to information
- Inconsistent policy communication across different channels

**Solution:**
By deploying a RAG chatbot, retail operations can deliver fast, reliable, and context-aware answers that improve service efficiency and customer satisfaction.

---

## 🛠️ Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Programming language | 3.9+ |
| **LangChain** | RAG framework and chains | 0.3+ |
| **FAISS** | Vector database for semantic search | 1.7.4+ |
| **Azure OpenAI** | LLM for answer generation | GPT-4o-mini |
| **Azure OpenAI Embeddings** | Text embeddings | text-embedding-3-small |
| **python-dotenv** | Environment variable management | 1.0+ |

---

## 📥 Installation

### 1. Clone or Navigate to Project Directory

```bash
cd Assignment_14
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```properties
# Azure OpenAI Embedding Configuration
AZURE_OPENAI_EMBEDDING_API_KEY=your_embedding_api_key
AZURE_OPENAI_EMBEDDING_ENDPOINT=your_embedding_endpoint
AZURE_OPENAI_EMBED_MODEL=text-embedding-3-small

# Azure OpenAI LLM Configuration
AZURE_OPENAI_LLM_API_KEY=your_llm_api_key
AZURE_OPENAI_LLM_ENDPOINT=your_llm_endpoint
AZURE_OPENAI_LLM_MODEL=gpt-4o-mini
```

### 4. Run the Application

```bash
# Set environment variable for FAISS on macOS (if needed)
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the chatbot
python main.py
```

---

## 🚀 Usage

### Automated Demo Mode (Default)

The application runs 6 pre-configured test queries demonstrating various use cases:

```bash
python main.py
```

### Interactive Mode (Optional)

Uncomment the last line in `main.py` to enable interactive mode:

```python
# Uncomment to enable interactive mode:
interactive_mode()
```

Then run:
```bash
python main.py
```

Example interaction:
```
You: Can I return electronics without a receipt?
🤖 Walmart Assistant: If you don't have a receipt, most returns are eligible 
for store credit with valid photo identification. For electronics specifically, 
they can be returned within 30 days with a receipt and original packaging...
```

---

## 🏗️ Architecture

### RAG Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query Input                          │
│           "Can I return my laptop?"                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              FAISS Vector Store Retrieval                    │
│  • Converts query to embedding (text-embedding-3-small)      │
│  • Searches 15 Walmart policy documents                      │
│  • Returns top 3 most relevant documents                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              Context Formatting                              │
│  • Formats retrieved documents into readable context         │
│  • Combines with user query                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              Prompt Template                                 │
│  • System: "You are a Walmart assistant..."                  │
│  • Context: Retrieved policy documents                       │
│  • Question: User query                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              Azure OpenAI LLM (GPT-4o-mini)                  │
│  • Generates human-friendly answer                           │
│  • Temperature: 0.3 (factual responses)                      │
│  • Cites relevant policy details                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              Output Parser                                   │
│  • Extracts clean text response                              │
│  • Returns to user                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 Policy Database

The chatbot references 15 in-memory Walmart policy documents covering:

### Categories:
- **Returns & Exchanges** (10 documents)
  - Electronics: 30-day return window
  - Groceries: 90-day return (non-perishable)
  - Vision Center: 60-day return
  - Cell phones: Must be unlocked and data erased
  - Bicycles: 90 days if unused outdoors
  - Online orders: In-store or mail returns
  - No receipt: Store credit with ID

- **Warranties** (1 document)
  - 1-year warranty on electronics/appliances

- **Membership Benefits** (1 document)
  - Walmart Plus: Free shipping, no minimum

- **Pricing** (1 document)
  - Price matching with Walmart.com and competitors

- **Services** (1 document)
  - Gift card policies

- **Special Cases** (1 document)
  - Prescription medications: No returns
  - Seasonal items: Modified windows
  - Fraud prevention policies

---

## ✨ Key Features

### 1. **Semantic Search with FAISS**
- Vector-based similarity search
- Retrieves top 3 most relevant documents per query
- Fast, efficient retrieval from in-memory database

### 2. **RAG Architecture**
- Combines retrieval with generation
- Grounded responses based on actual policies
- Reduces hallucination and improves accuracy

### 3. **Azure OpenAI Integration**
- GPT-4o-mini for answer generation
- text-embedding-3-small for semantic embeddings
- Low temperature (0.3) for factual consistency

### 4. **Context-Aware Responses**
- Answers based on retrieved policy documents
- Cites relevant policy details
- Handles both customer and employee queries

### 5. **Professional Tone**
- Friendly and helpful language
- Clear, concise answers
- Acknowledges limitations when needed

### 6. **Automated Testing**
- 6 pre-configured test scenarios
- Demonstrates various use cases
- Shows retrieved documents and generated answers

---

## 🔄 RAG Pipeline Explained

### What is RAG?

**Retrieval-Augmented Generation** is a technique that combines:
1. **Retrieval**: Finding relevant documents from a knowledge base
2. **Generation**: Using an LLM to create answers based on retrieved context

### Why RAG for Retail?

| Traditional Chatbot | RAG Chatbot |
|---------------------|-------------|
| May hallucinate facts | Grounded in real policies |
| Generic responses | Context-specific answers |
| Hard to update | Easy to update knowledge base |
| Limited to training data | Can reference current policies |

### RAG Components in This Project:

1. **Vector Store (FAISS)**
   - Stores document embeddings
   - Enables semantic search (meaning-based, not keyword-based)
   - Fast retrieval from 15 policy documents

2. **Embeddings (Azure OpenAI)**
   - Converts text to numerical vectors
   - Model: `text-embedding-3-small`
   - Captures semantic meaning of queries and documents

3. **Retriever**
   - Searches vector store for similar documents
   - Returns top 3 most relevant matches
   - Provides context for answer generation

4. **LLM (GPT-4o-mini)**
   - Generates human-friendly answers
   - Uses retrieved documents as context
   - Temperature 0.3 for factual responses

5. **Chain (LangChain)**
   - Connects all components
   - Manages data flow: Query → Retrieve → Format → Generate → Parse

---

## 🧪 Test Queries

The application includes 6 automated test queries:

### 1. Customer - Electronics Return
**Query:** "Can I return my laptop if I don't like it? I bought it 2 weeks ago."

**Expected:** Information about 30-day electronics return policy

### 2. Employee - Groceries Without Receipt
**Query:** "A customer wants to return groceries without a receipt. What are our options?"

**Expected:** Store credit with photo ID policy

### 3. Customer - Walmart Plus Benefits
**Query:** "What are the benefits of Walmart Plus membership?"

**Expected:** Free shipping with no minimum order

### 4. Employee - Warranty Question
**Query:** "Customer is asking about warranty coverage for a washing machine."

**Expected:** 1-year warranty on electronics/appliances

### 5. Customer - Prescription Returns
**Query:** "Can I return prescription medications if I don't need them anymore?"

**Expected:** Not eligible for return or exchange

### 6. Employee - Price Matching
**Query:** "Does Walmart do price matching? Customer has a competitor's ad."

**Expected:** Price matching policy for identical items

---

## 💻 Technical Implementation

### Code Structure

```
Assignment_14/
├── main.py                 # Main application code
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variable template
├── .env                   # Your credentials (not committed)
├── .gitignore            # Git ignore file
└── README.md             # This documentation
```

### Key Code Sections

#### 1. Document Loading
```python
walmart_documents = [
    Document(
        page_content="Policy text here...",
        metadata={"doc_id": 1, "category": "returns"}
    ),
    # ... 14 more documents
]
```

#### 2. Vector Store Creation
```python
embedding_model = AzureOpenAIEmbeddings(...)
vectorstore = FAISS.from_documents(walmart_documents, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

#### 3. RAG Chain
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

#### 4. Query Processing
```python
answer = rag_chain.invoke(user_question)
```

---

## 📖 Learning Outcomes

By completing this assignment, you will understand:

### 1. **RAG Architecture**
- How retrieval improves LLM accuracy
- Combining search with generation
- Benefits over pure LLM approaches

### 2. **Vector Databases**
- FAISS for semantic search
- Embedding-based retrieval
- Similarity search concepts

### 3. **LangChain Framework**
- Building RAG pipelines
- Chaining components together
- Prompt engineering for RAG

### 4. **Azure OpenAI Services**
- Using embeddings API
- Chat completion API
- Environment configuration

### 5. **Production Considerations**
- Document structure and metadata
- Retrieval strategies (top-k)
- Temperature tuning for factual responses
- Error handling and validation

---

## 💼 Business Impact

### Quantifiable Benefits:

1. **Reduced Response Time**
   - Instant answers vs. manual lookup
   - 24/7 availability
   - No wait times

2. **Improved Accuracy**
   - Consistent policy information
   - Reduced human error
   - Always up-to-date

3. **Cost Savings**
   - Reduced customer service load
   - Staff can focus on complex issues
   - Scalable without adding headcount

4. **Better Customer Experience**
   - Fast, accurate self-service
   - Professional, friendly responses
   - Available across all channels

### Use Cases:

- **Customer Self-Service**: Website, mobile app, in-store kiosks
- **Employee Training**: New staff learning policies
- **Call Center Support**: Quick reference for agents
- **In-Store Assistance**: Tablet-based employee tool

---

## 🚀 Future Enhancements

### 1. **Expanded Knowledge Base**
- Product catalogs and specifications
- Store locations and hours
- Shipping and delivery policies
- International policies

### 2. **Advanced Retrieval**
- Hybrid search (semantic + keyword)
- Re-ranking for better relevance
- Metadata filtering (category, product type)
- Query expansion and reformulation

### 3. **Multi-Modal Support**
- Image understanding (product photos)
- Voice interface integration
- Video tutorials and demonstrations

### 4. **Personalization**
- User history and preferences
- Location-based responses
- Membership tier awareness

### 5. **Analytics & Monitoring**
- Query logging and analysis
- User satisfaction tracking
- Policy gap identification
- A/B testing for improvements

### 6. **Integration**
- Inventory management systems
- Point-of-sale systems
- Customer relationship management (CRM)
- Order tracking systems

### 7. **Additional Features**
- Multi-language support
- Sentiment analysis
- Escalation to human agents
- Proactive suggestions

---

## 📊 Performance Considerations

### Retrieval Quality:
- **Top-K Setting**: k=3 balances relevance and context length
- **Embedding Model**: text-embedding-3-small provides good accuracy with low latency
- **Document Chunking**: Current documents are pre-chunked optimally

### Generation Quality:
- **Temperature**: 0.3 for factual, consistent responses
- **Model**: GPT-4o-mini balances quality and cost
- **Prompt Engineering**: Clear instructions for professional tone

### Scalability:
- **FAISS**: Handles millions of documents efficiently
- **In-Memory**: Fast for current dataset size (15 docs)
- **Can Move to**: Persistent storage (Pinecone, Weaviate, Azure Cognitive Search)

---

## 🔒 Security & Privacy

### Best Practices Implemented:

1. **Environment Variables**: API keys in `.env` (not committed)
2. **Type Safety**: SecretStr for sensitive credentials
3. **No User Data**: Mock dataset only, no PII
4. **Audit Trail**: Queries can be logged (if enabled)

### Recommendations for Production:

- Implement authentication and authorization
- Encrypt data in transit and at rest
- Regular security audits
- Compliance with retail data regulations (PCI-DSS, etc.)
- Rate limiting and abuse prevention

---

## 📝 Submission Checklist

- [x] `main.py` - Complete RAG implementation
- [x] `requirements.txt` - All dependencies listed
- [x] `.env.example` - Environment variable template
- [x] `.gitignore` - Excludes sensitive files
- [x] `README.md` - Comprehensive documentation
- [x] Automated test queries (6 scenarios)
- [x] Working FAISS vector store
- [x] Azure OpenAI integration
- [x] Clean, well-commented code
- [x] Professional output formatting

---

## 🤝 Support

For questions or issues:
- Review the comprehensive code comments
- Check Azure OpenAI API documentation
- Verify environment variable configuration
- Ensure all dependencies are installed

---

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

---

**Author:** Duke  
**Date:** November 18, 2025  
**Assignment:** 14 - RAG Retail Chatbot  
**Course:** AI-Powered Applications
