"""
AI-Powered Patient Information Chatbot with LangGraph
======================================================
This chatbot interactively collects patient information (name, age, symptoms)
and provides preliminary health advice using:
- LangGraph for conversation flow management
- AzureChatOpenAI as the language model
- FAISS vector store for medical knowledge retrieval
- Tavily for real-time web search (optional)

Author: Duke
Date: November 18, 2025
"""

import os
from typing import Annotated
from langchain.docstore.document import Document
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
from pathlib import Path
from pydantic import SecretStr

# Load environment variables
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

print("="*80)
print("AI-POWERED PATIENT INFORMATION CHATBOT")
print("="*80)

# ---- MOCK MEDICAL KNOWLEDGE BASE ----
mock_chunks = [
    Document(
        page_content="Patients with a sore throat should drink warm fluids and avoid cold beverages."
    ),
    Document(
        page_content="Mild fevers under 38.5°C can often be managed with rest and hydration."
    ),
    Document(
        page_content="If a patient reports dizziness, advise checking their blood pressure and hydration level."
    ),
    Document(
        page_content="Persistent coughs lasting more than 2 weeks should be evaluated for infections or allergies."
    ),
    Document(
        page_content="Patients experiencing fatigue should consider iron deficiency or poor sleep as potential causes."
    ),
    Document(
        page_content="Headaches accompanied by fever may indicate flu or infection. Rest and over-the-counter pain relievers can help."
    ),
    Document(
        page_content="For minor cuts and scrapes, clean the wound with water and apply an antibiotic ointment."
    ),
    Document(
        page_content="Stomach pain with nausea may be due to food poisoning or gastritis. Stay hydrated and avoid solid foods temporarily."
    ),
    Document(
        page_content="Chest pain should always be evaluated immediately, especially if accompanied by shortness of breath."
    ),
    Document(
        page_content="Insomnia can be improved by maintaining a regular sleep schedule and avoiding caffeine before bed."
    ),
]

print(f"\n📚 Medical Knowledge Base: {len(mock_chunks)} documents loaded")

# ---- ENVIRONMENT CONFIGURATION ----
AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", "")
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "https://aiportalapi.stu-platform.live/jpe")
AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-small")

AZURE_OPENAI_LLM_API_KEY = os.getenv("AZURE_OPENAI_LLM_API_KEY", "")
AZURE_OPENAI_LLM_ENDPOINT = os.getenv("AZURE_OPENAI_LLM_ENDPOINT", "https://aiportalapi.stu-platform.live/jpe")
AZURE_OPENAI_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL", "gpt-4o-mini")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

print(f"\n✅ Configuration loaded")
print(f"   LLM Endpoint: {AZURE_OPENAI_LLM_ENDPOINT}")
print(f"   LLM Model: {AZURE_OPENAI_LLM_MODEL}")
print(f"   Embedding Model: {AZURE_OPENAI_EMBED_MODEL}")

# ---- SETUP FAISS RETRIEVER FROM MOCK CHUNKS ----
print(f"\n🔄 Initializing FAISS vector store...")

embedding_model = AzureOpenAIEmbeddings(
    model=AZURE_OPENAI_EMBED_MODEL,
    api_key=SecretStr(AZURE_OPENAI_EMBEDDING_API_KEY),
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
    api_version="2024-02-15-preview",
)

db = FAISS.from_documents(mock_chunks, embedding_model)
retriever = db.as_retriever()

print(f"✅ FAISS vector store initialized with {len(mock_chunks)} documents")

# ---- TOOL 1: RETRIEVE MEDICAL ADVICE ----
@tool
def retrieve_advice(user_input: str) -> str:
    """Searches internal medical documents for relevant patient advice based on symptoms or conditions."""
    docs = retriever.get_relevant_documents(user_input)
    if docs:
        return "\n".join(doc.page_content for doc in docs)
    return "No relevant medical advice found in knowledge base."

# ---- TOOL 2: TAVILY SEARCH (Optional) ----
tavily_api_key = os.getenv("TAVILY_API_KEY")
if tavily_api_key and tavily_api_key != "your_tavily_api_key_here":
    tavily_tool = TavilySearchResults(max_results=2)
    tools_list = [retrieve_advice, tavily_tool]
    tavily_enabled = True
else:
    tools_list = [retrieve_advice]
    tavily_enabled = False

# ---- LLM SETUP WITH TOOLS ----
print(f"\n🔄 Initializing Azure OpenAI LLM...")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_LLM_ENDPOINT,
    api_key=SecretStr(AZURE_OPENAI_LLM_API_KEY),
    azure_deployment=AZURE_OPENAI_LLM_MODEL,
    api_version="2024-02-15-preview",
    temperature=0.7,
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools_list)

print(f"✅ Azure OpenAI LLM initialized with tools")
if tavily_enabled:
    print(f"   Tools: retrieve_advice, tavily_search_results")
else:
    print(f"   Tools: retrieve_advice (Tavily disabled - API key not configured)")

# ---- MODEL NODE: CALL LLM WITH TOOLS ----
def call_model(state: MessagesState):
    """Invoke the LLM with current conversation messages."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ---- CONDITIONAL ROUTING: DECIDE NEXT STEP ----
def should_continue(state: MessagesState):
    """Determine if tools should be called or conversation should end."""
    last_message = state["messages"][-1]
    # If LLM requests tool calls, route to tools node
    if last_message.tool_calls:  # type: ignore
        return "tools"
    # Otherwise, end the conversation
    return END

# ---- TOOL NODE: EXECUTE TOOL CALLS ----
tool_node = ToolNode(tools_list)

# ---- BUILD THE LANGGRAPH WORKFLOW ----
print(f"\n🔄 Building LangGraph workflow...")

graph_builder = StateGraph(MessagesState)

# Add nodes
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "call_model")
graph_builder.add_conditional_edges(
    "call_model",
    should_continue,
    ["tools", END]
)
graph_builder.add_edge("tools", "call_model")

# Compile the graph
graph = graph_builder.compile()

print(f"✅ LangGraph workflow compiled successfully")

# ---- AUTOMATED TEST CASES (NO MANUAL INPUT) ----
test_cases = [
    {
        "patient_name": "John Doe",
        "age": 28,
        "symptoms": "I feel tired and have a sore throat. What should I do?"
    },
    {
        "patient_name": "Sarah Chen",
        "age": 35,
        "symptoms": "I have a persistent cough for 3 weeks and mild fever. Should I be concerned?"
    },
    {
        "patient_name": "Michael Rodriguez",
        "age": 42,
        "symptoms": "I'm experiencing dizziness and headaches. What could be causing this?"
    },
]

print("\n" + "="*80)
print("RUNNING AUTOMATED PATIENT CONSULTATIONS")
print("="*80)

for idx, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"CONSULTATION {idx}/{len(test_cases)}")
    print(f"{'='*80}")
    print(f"\n👤 Patient Information:")
    print(f"   Name: {test_case['patient_name']}")
    print(f"   Age: {test_case['age']}")
    print(f"   Symptoms: {test_case['symptoms']}\n")
    
    # Create system message with patient context
    system_prompt = f"""You are a helpful medical assistant chatbot. 
    
Your role is to:
1. Analyze patient symptoms
2. Provide preliminary health advice
3. Use the retrieve_advice tool to search your medical knowledge base
4. Optionally use tavily_search_results to get additional up-to-date information
5. Provide clear, actionable recommendations
6. Always remind patients to consult healthcare professionals for serious concerns

Patient Information:
- Name: {test_case['patient_name']}
- Age: {test_case['age']}

Be empathetic, clear, and professional. Focus on preliminary advice based on the symptoms."""

    # Invoke the graph
    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=test_case['symptoms']),
            ]
        }
    )
    
    # Extract and display the final response
    final_response = result["messages"][-1].content  # type: ignore
    
    print(f"🤖 Medical Assistant Response:")
    print(f"{'-'*80}")
    print(final_response)
    print(f"{'-'*80}\n")
    
    # Display conversation flow details
    print(f"📊 Conversation Flow:")
    print(f"   Total messages exchanged: {len(result['messages'])}")
    
    # Check if tools were used
    tool_calls_count = sum(1 for msg in result["messages"] if hasattr(msg, "tool_calls") and msg.tool_calls)  # type: ignore
    print(f"   Tool calls made: {tool_calls_count}")
    
    if tool_calls_count > 0:
        print(f"   Tools used:")
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:  # type: ignore
                for tool_call in msg.tool_calls:  # type: ignore
                    print(f"      - {tool_call['name']}")  # type: ignore

print("\n" + "="*80)
print("✅ ALL PATIENT CONSULTATIONS COMPLETED")
print("="*80)

print(f"\n📈 Summary:")
print(f"   Total consultations: {len(test_cases)}")
print(f"   Medical knowledge documents: {len(mock_chunks)}")
print(f"   Tools available: retrieve_advice, tavily_search_results")

print(f"\n💡 Key Features Demonstrated:")
print(f"   ✅ LangGraph conversation flow management")
print(f"   ✅ Azure OpenAI LLM integration")
print(f"   ✅ FAISS vector store for medical knowledge retrieval")
print(f"   ✅ Tool-based architecture (retrieve_advice, Tavily search)")
print(f"   ✅ Automated patient consultations (no manual input)")
print(f"   ✅ Context-aware preliminary health advice")

print(f"\n⚕️ Disclaimer: This is a prototype for educational purposes.")
print(f"   Always consult qualified healthcare professionals for medical advice.\n")
