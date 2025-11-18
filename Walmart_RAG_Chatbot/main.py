"""
Walmart RAG Chatbot - Retrieval-Augmented Generation for Retail Support
==========================================================================
This chatbot helps retail employees and customers get instant answers about:
- Product details and specifications
- Store policies and warranties
- Return and exchange guidelines
- Walmart Plus benefits and services

Uses RAG (Retrieval-Augmented Generation) with:
- LangGraph for state management and workflow orchestration
- FAISS for semantic document retrieval
- Azure OpenAI for generating human-friendly answers
- In-memory Walmart policy documents

Author: Duke
Date: November 18, 2025
"""

import os
from typing import TypedDict, Optional, List
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from pathlib import Path
from pydantic import SecretStr

# Load environment variables
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

print("=" * 80)
print("WALMART RAG CHATBOT - RETAIL SUPPORT ASSISTANT")
print("=" * 80)

# ================================================================================
# WALMART POLICY AND PRODUCT DOCUMENTS (Mock Dataset)
# ================================================================================
walmart_documents = [
    Document(
        page_content="Walmart customers may return electronics within 30 days with a receipt and original packaging.",
        metadata={"doc_id": 1, "category": "returns", "product_type": "electronics"}
    ),
    Document(
        page_content="Grocery items at Walmart can be returned within 90 days with proof of purchase, except perishable products.",
        metadata={"doc_id": 2, "category": "returns", "product_type": "grocery"}
    ),
    Document(
        page_content="Walmart offers a 1-year warranty on most electronics and appliances. See product details for exceptions.",
        metadata={"doc_id": 3, "category": "warranty", "product_type": "electronics"}
    ),
    Document(
        page_content="Walmart Plus members get free shipping with no minimum order amount.",
        metadata={"doc_id": 4, "category": "membership", "product_type": "service"}
    ),
    Document(
        page_content="Prescription medications purchased at Walmart are not eligible for return or exchange.",
        metadata={"doc_id": 5, "category": "returns", "product_type": "pharmacy"}
    ),
    Document(
        page_content="Open-box items are eligible for return at Walmart within the standard return period, but must include all original accessories.",
        metadata={"doc_id": 6, "category": "returns", "product_type": "general"}
    ),
    Document(
        page_content="If a Walmart customer does not have a receipt, most returns are eligible for store credit with valid photo identification.",
        metadata={"doc_id": 7, "category": "returns", "product_type": "general"}
    ),
    Document(
        page_content="Walmart allows price matching for identical items found on Walmart.com and local competitor ads.",
        metadata={"doc_id": 8, "category": "pricing", "product_type": "general"}
    ),
    Document(
        page_content="Walmart Vision Center purchases may be returned or exchanged within 60 days with a receipt.",
        metadata={"doc_id": 9, "category": "returns", "product_type": "vision"}
    ),
    Document(
        page_content="Returns on cell phones at Walmart require the device to be unlocked and all personal data erased.",
        metadata={"doc_id": 10, "category": "returns", "product_type": "electronics"}
    ),
    Document(
        page_content="Walmart gift cards cannot be redeemed for cash except where required by law.",
        metadata={"doc_id": 11, "category": "gift_cards", "product_type": "service"}
    ),
    Document(
        page_content="Seasonal merchandise at Walmart (e.g., holiday decorations) may have modified return windows, see in-store signage.",
        metadata={"doc_id": 12, "category": "returns", "product_type": "seasonal"}
    ),
    Document(
        page_content="Bicycles purchased at Walmart can be returned within 90 days if not used outdoors and with all accessories present.",
        metadata={"doc_id": 13, "category": "returns", "product_type": "bicycles"}
    ),
    Document(
        page_content="For online Walmart orders, customers can return items in store or by mail using the prepaid label.",
        metadata={"doc_id": 14, "category": "returns", "product_type": "online"}
    ),
    Document(
        page_content="Walmart reserves the right to deny returns suspected of fraud or abuse.",
        metadata={"doc_id": 15, "category": "returns", "product_type": "policy"}
    ),
]

print(f"\n📦 Walmart Policy Database: {len(walmart_documents)} documents loaded")
print(f"   Categories: Returns, Warranties, Membership, Pricing, Services")

# ================================================================================
# ENVIRONMENT CONFIGURATION
# ================================================================================
AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", "")
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "https://aiportalapi.stu-platform.live/jpe")
AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-small")

AZURE_OPENAI_LLM_API_KEY = os.getenv("AZURE_OPENAI_LLM_API_KEY", "")
AZURE_OPENAI_LLM_ENDPOINT = os.getenv("AZURE_OPENAI_LLM_ENDPOINT", "https://aiportalapi.stu-platform.live/jpe")
AZURE_OPENAI_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL", "gpt-4o-mini")

print(f"\n✅ Configuration loaded")
print(f"   LLM Endpoint: {AZURE_OPENAI_LLM_ENDPOINT}")
print(f"   LLM Model: {AZURE_OPENAI_LLM_MODEL}")
print(f"   Embedding Model: {AZURE_OPENAI_EMBED_MODEL}")

# ================================================================================
# SETUP FAISS VECTOR STORE FOR RAG
# ================================================================================
print(f"\n🔄 Initializing FAISS vector store for semantic search...")

embedding_model = AzureOpenAIEmbeddings(
    model=AZURE_OPENAI_EMBED_MODEL,
    api_key=SecretStr(AZURE_OPENAI_EMBEDDING_API_KEY),
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
    api_version="2024-02-15-preview",
)

# Create FAISS vector store from Walmart documents with InMemoryDocstore
vectorstore = FAISS.from_documents(
    walmart_documents,
    embedding_model,
    docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(walmart_documents)}),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant docs

print(f"✅ FAISS vector store initialized with {len(walmart_documents)} documents")
print(f"   Retrieval strategy: Top 3 most relevant documents per query")

# ================================================================================
# SETUP AZURE OPENAI LLM
# ================================================================================
print(f"\n🔄 Initializing Azure OpenAI LLM...")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_LLM_ENDPOINT,
    api_key=SecretStr(AZURE_OPENAI_LLM_API_KEY),
    azure_deployment=AZURE_OPENAI_LLM_MODEL,
    api_version="2024-02-15-preview",
    temperature=0.3,  # Lower temperature for more factual, consistent answers
)

print(f"✅ Azure OpenAI LLM initialized")
print(f"   Temperature: 0.3 (factual, consistent responses)")

# ================================================================================
# LANGGRAPH STATE DEFINITION
# ================================================================================
class RAGState(TypedDict):
    """State for the RAG workflow."""
    question: str
    context: Optional[str]
    answer: Optional[str]
    retrieved_docs: Optional[List[Document]]

# ================================================================================
# RAG PROMPT TEMPLATE
# ================================================================================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful Walmart customer service assistant. 
Your role is to provide accurate, friendly, and professional answers to questions about Walmart's policies, products, and services.

Use the provided policy documents to answer the customer's question. Always cite relevant policy details in your answer.

Instructions:
- Provide a clear, direct answer based on the retrieved documents
- Be friendly and professional in your tone
- If the documents don't contain enough information, acknowledge what you know and suggest the customer contact Walmart customer service
- Keep your answer concise but complete"""
    ),
    (
        "human",
        "{context}\n\nUser question: {question}"
    )
])

# ================================================================================
# LANGGRAPH NODES - RETRIEVE AND GENERATE
# ================================================================================
def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant documents from the vector store."""
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([f"Policy {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    return {**state, "context": context, "retrieved_docs": docs}

def generate_node(state: RAGState) -> RAGState:
    """Generate answer using the LLM with retrieved context."""
    formatted_prompt = prompt.format(
        context=state["context"],
        question=state["question"]
    )
    response = llm.invoke(formatted_prompt)
    answer_text = str(response.content) if response.content else ""
    return {**state, "answer": answer_text}

# ================================================================================
# BUILD LANGGRAPH WORKFLOW
# ================================================================================
print(f"\n🔄 Building LangGraph RAG workflow...")

graph_builder = StateGraph(RAGState)

# Add nodes
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)

# Define edges
graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.set_finish_point("generate")

# Compile the graph
rag_graph = graph_builder.compile()

print(f"✅ LangGraph workflow compiled successfully")
print(f"   Pipeline: retrieve → generate → END")

# ================================================================================
# TEST QUERIES - AUTOMATED DEMONSTRATION
# ================================================================================
test_queries = [
    {
        "user_type": "Customer",
        "question": "Can I return my laptop if I don't like it? I bought it 2 weeks ago."
    },
    {
        "user_type": "Employee",
        "question": "A customer wants to return groceries without a receipt. What are our options?"
    },
    {
        "user_type": "Customer",
        "question": "What are the benefits of Walmart Plus membership?"
    },
    {
        "user_type": "Employee",
        "question": "Customer is asking about warranty coverage for a washing machine."
    },
    {
        "user_type": "Customer",
        "question": "Can I return prescription medications if I don't need them anymore?"
    },
    {
        "user_type": "Employee",
        "question": "Does Walmart do price matching? Customer has a competitor's ad."
    },
]

print("\n" + "=" * 80)
print("RUNNING AUTOMATED CHATBOT DEMONSTRATIONS")
print("=" * 80)

for idx, query in enumerate(test_queries, 1):
    print(f"\n{'=' * 80}")
    print(f"QUERY {idx}/{len(test_queries)}")
    print(f"{'=' * 80}")
    print(f"\n👤 {query['user_type']}: {query['question']}")
    
    # Invoke the LangGraph RAG workflow
    result = rag_graph.invoke({"question": query['question']})  # type: ignore
    
    # Display retrieved documents
    if result.get('retrieved_docs'):
        retrieved_docs = result['retrieved_docs']
        print(f"\n📚 Retrieved Documents ({len(retrieved_docs)}):")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"   {i}. [Doc {doc.metadata.get('doc_id', 'N/A')}] {doc.page_content[:80]}...")
    
    # Display the generated answer
    print(f"\n🤖 Walmart Assistant:")
    print(f"{'-' * 80}")
    print(result['answer'])
    print(f"{'-' * 80}")

# ================================================================================
# INTERACTIVE MODE (Optional - Can be uncommented for manual testing)
# ================================================================================
def interactive_mode():
    """Run the chatbot in interactive mode for manual testing."""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE - Ask questions about Walmart policies")
    print("=" * 80)
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['exit', 'quit', '']:
            print("\n👋 Thank you for using Walmart Assistant!")
            break
        
        # Invoke LangGraph RAG workflow
        result = rag_graph.invoke({"question": question})  # type: ignore
        
        # Display retrieved documents count
        if result.get('retrieved_docs'):
            print(f"\n📚 Found {len(result['retrieved_docs'])} relevant policies")
        
        # Display answer
        print(f"\n🤖 Walmart Assistant:\n{result['answer']}\n")

# Uncomment to enable interactive mode:
# interactive_mode()

# ================================================================================
# SUMMARY
# ================================================================================
print("\n" + "=" * 80)
print("✅ ALL DEMONSTRATIONS COMPLETED")
print("=" * 80)

print(f"\n📈 Summary:")
print(f"   Total policy documents: {len(walmart_documents)}")
print(f"   Test queries processed: {len(test_queries)}")
print(f"   RAG pipeline: FAISS Retrieval → Azure OpenAI Generation")

print(f"\n💡 Key Features Demonstrated:")
print(f"   ✅ Semantic search with FAISS vector store")
print(f"   ✅ Retrieval-Augmented Generation (RAG) architecture")
print(f"   ✅ Azure OpenAI GPT-4o-mini for answer generation")
print(f"   ✅ Context-aware responses based on retrieved policies")
print(f"   ✅ Support for both customer and employee queries")
print(f"   ✅ Clear citation of relevant policy information")

print(f"\n🏪 Use Cases:")
print(f"   • Customer self-service for policy questions")
print(f"   • Employee support for quick policy lookups")
print(f"   • Consistent, accurate information delivery")
print(f"   • Reduced wait times and improved satisfaction")

print(f"\n💼 Business Impact:")
print(f"   • Faster response times for customer inquiries")
print(f"   • Reduced burden on human customer service staff")
print(f"   • Consistent policy communication across all channels")
print(f"   • Improved customer and employee experience")

# ================================================================================
# REFLECTION: RAG vs. TRADITIONAL APPROACHES
# ================================================================================
print(f"\n📝 REFLECTION: RAG vs. Traditional Support Methods")
print(f"{'=' * 80}")

print(f"\n🔍 RAG Advantages Over Static FAQs:")
print(f"   1. Semantic Understanding")
print(f"      • RAG understands query intent, not just keywords")
print(f"      • Example: 'Can I return my laptop?' matches 'electronics return policy'")
print(f"      • Static FAQs require exact keyword matches")

print(f"\n   2. Context-Aware Answers")
print(f"      • Combines multiple relevant policies automatically")
print(f"      • Generates natural, conversational responses")
print(f"      • Static FAQs provide rigid, pre-written answers")

print(f"\n   3. Easy Updates & Scalability")
print(f"      • Add new policies by simply updating document database")
print(f"      • No need to manually create Q&A pairs")
print(f"      • Static FAQs require manual updates for each question variation")

print(f"\n   4. Handles Complex Queries")
print(f"      • Answers multi-faceted questions (e.g., 'return without receipt')")
print(f"      • Synthesizes information from multiple sources")
print(f"      • Static FAQs struggle with complex or combined queries")

print(f"\n   5. Reduces Maintenance Burden")
print(f"      • One source of truth (policy documents)")
print(f"      • Automatic retrieval eliminates manual curation")
print(f"      • Keyword search requires constant synonym updates")

print(f"\n🚀 Future Enhancement Opportunities:")
print(f"   1. Multi-Turn Conversations")
print(f"      • Add conversation history to RAGState")
print(f"      • Enable follow-up questions and clarifications")
print(f"      • Example: 'What about if I lost the receipt?' after initial query")

print(f"\n   2. Product Recommendations")
print(f"      • Integrate product catalog with policies")
print(f"      • Suggest alternatives when items can't be returned")
print(f"      • Example: 'Looking for laptops with extended warranty?'")

print(f"\n   3. Hybrid Retrieval")
print(f"      • Combine semantic search with keyword filtering")
print(f"      • Add metadata-based filtering (category, product type)")
print(f"      • Improve precision for specific product queries")

print(f"\n   4. Real-Time Data Integration")
print(f"      • Connect to inventory management systems")
print(f"      • Provide stock availability information")
print(f"      • Check customer order history for personalized support")

print(f"\n   5. Feedback Loop & Learning")
print(f"      • Track query satisfaction ratings")
print(f"      • Identify knowledge gaps in policy database")
print(f"      • Continuously improve retrieval quality")

print(f"\n   6. Multi-Language Support")
print(f"      • Translate queries and responses")
print(f"      • Serve diverse customer base")
print(f"      • Maintain single policy database")

print(f"\n{'=' * 80}")
print(f"End of Walmart RAG Chatbot Demonstration")
print(f"{'=' * 80}\n")
