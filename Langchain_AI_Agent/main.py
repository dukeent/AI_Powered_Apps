"""
Langchain AI Agent with Weather and Web Search
==============================================
This application demonstrates a Langchain-based AI agent that can handle
real-time weather queries and web searches using OpenWeather and Tavily APIs.

Features:
- OpenWeather API integration for weather queries
- Tavily Search API for real-time web search
- Langchain tool routing and agent decision-making
- ReAct-style conversational interface
- Automated testing with mock questions

Author: Duke
Date: November 12, 2025
"""

# pip install langchain-openai langchain-community langchain-tavily langgraph pyowm python-dotenv
import os
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from pathlib import Path
from pydantic import SecretStr

# Load environment variables from .env file
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

print("="*80)
print("LANGCHAIN AI AGENT - WEATHER & WEB SEARCH")
print("="*80)

# ---- AZURE OPENAI CONFIG ----
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_LLM_MODEL", "gpt-4o-mini")

# ---- API KEYS ----
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Set environment variables for libraries
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["AZURE_DEPLOYMENT_NAME"] = AZURE_DEPLOYMENT_NAME
os.environ["OPENWEATHERMAP_API_KEY"] = OPENWEATHERMAP_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

print(f"\n✅ Configuration loaded")
print(f"   Azure Endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f"   Deployment: {AZURE_DEPLOYMENT_NAME}")
print(f"   OpenWeather API Key: {'*' * 20}{OPENWEATHERMAP_API_KEY[-4:] if len(OPENWEATHERMAP_API_KEY) > 4 else '****'}")
print(f"   Tavily API Key: {'*' * 20}{TAVILY_API_KEY[-4:] if len(TAVILY_API_KEY) > 4 else '****'}\n")

# ---- DEFINE WEATHER TOOL ----
print("🔄 Initializing OpenWeather API wrapper...")
weather = OpenWeatherMapAPIWrapper()
print("✅ OpenWeather API wrapper initialized\n")

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.
    
    Args:
        city (str): The name of the city to get the weather for.
        
    Returns:
        str: A string describing the current weather in the specified city.
    """
    print(f"   🌤️  get_weather tool calling: Getting weather for {city}")
    return weather.run(city)

# ---- INITIALIZE TAVILY SEARCH TOOL ----
print("🔄 Initializing Tavily Search API...")
tavily_search_tool = TavilySearch(
    max_results=1,
    topic="general",
)
print("✅ Tavily Search API initialized\n")

# ---- INITIALIZE AZURE OPENAI LLM ----
print("🔄 Initializing Azure OpenAI LLM...")
llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-07-01-preview",
    api_key=SecretStr(AZURE_OPENAI_API_KEY) if AZURE_OPENAI_API_KEY else None,  # type: ignore
)
print("✅ Azure OpenAI LLM initialized\n")

# ---- SETUP LANGCHAIN AGENT ----
print("🔄 Setting up Langchain ReAct agent with tools...")
tools = [get_weather, tavily_search_tool]
agent = create_react_agent(
    model=llm,
    tools=tools,
)
print(f"✅ Agent created with {len(tools)} tools: {[tool.name for tool in tools]}\n")

# ---- AUTOMATED MOCK QUESTIONS (NO MANUAL INPUT) ----
mock_questions = [
    "What's the weather in Hanoi?",
    "Tell me about the latest news in AI.",
    "Who won the last World Cup?",
    "What's the weather in Tokyo?",
]

print("="*80)
print("STARTING CONVERSATIONAL AGENT WITH AUTOMATED QUERIES")
print("="*80)
print("ℹ️  The agent will automatically process mock questions without manual input.\n")

messages = []

for i, user_input in enumerate(mock_questions, 1):
    print("="*80)
    print(f"QUERY {i}/{len(mock_questions)}")
    print("="*80)
    print(f"\n👤 User: {user_input}\n")
    
    # Add user message to conversation history
    messages.append({"role": "user", "content": user_input})
    
    # Invoke agent
    print("🤖 AI Agent is thinking and selecting tools...\n")
    response = agent.invoke({"messages": messages})
    
    # Extract AI response
    ai_response = response["messages"][-1].content
    messages.append({"role": "assistant", "content": ai_response})
    
    print(f"🤖 AI: {ai_response}\n")
    print("="*80 + "\n")

print("="*80)
print("✅ All automated queries completed successfully!")
print("="*80)

# ---- SUMMARY ----
print(f"\n📊 Conversation Summary:")
print(f"   Total queries processed: {len(mock_questions)}")
print(f"   Total messages exchanged: {len(messages)}")
print(f"   Tools available: {[tool.name for tool in tools]}")
print(f"\n✅ Program completed successfully!")
