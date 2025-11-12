# Langchain AI Agent - Weather & Web Search

An intelligent AI agent built with Langchain that can handle real-time weather queries and web searches using OpenWeather and Tavily APIs.

## 📋 Objective

- Build a Langchain-based AI agent that handles real-time weather and web search queries
- Integrate OpenWeather API and Tavily Search API as custom tools within the agent
- Implement a ReAct-style conversational interface with tool routing capabilities
- Demonstrate automated agent decision-making and tool selection

## 🎯 Problem Statement

Build an AI assistant agent that can intelligently answer questions like:
- "What's the weather in Paris today?"
- "Search for the latest news on AI in healthcare."
- "Who won the last World Cup?"

The agent must:
- **Route queries** to OpenWeather API for weather-related questions
- **Route queries** to Tavily Search API for real-time web search
- **Use Langchain's tool routing** to automatically select the appropriate tool for each query
- **Maintain conversation history** for context-aware responses

## ✨ Features

- **Langchain ReAct Agent**: Reasoning and Acting agent architecture
- **OpenWeather Integration**: Real-time weather data for any city
- **Tavily Search Integration**: Real-time web search and news
- **Automated Tool Selection**: Agent intelligently chooses the right tool
- **Conversation History**: Context-aware multi-turn conversations
- **Automated Testing**: 4 pre-configured mock questions (no manual input)
- **Single Python File**: Complete implementation in `main.py`

## 🛠️ Installation

### 1. Install Dependencies

```bash
cd Assignment_11
pip install -r requirements.txt
```

Required packages:
- `langchain-openai>=0.1.0` - Azure OpenAI integration
- `langchain-community>=0.2.0` - Community tools (OpenWeather)
- `langchain-tavily>=0.1.0` - Tavily search integration
- `langgraph>=0.1.0` - Agent graph framework
- `pyowm>=3.3.0` - OpenWeatherMap Python wrapper
- `python-dotenv>=1.0.0` - Environment variable management

### 2. Get API Keys

#### Azure OpenAI
- Already configured from previous assignments

#### OpenWeather API
1. Go to https://openweathermap.org/api
2. Sign up for a free account
3. Navigate to API Keys section
4. Generate a new API key

#### Tavily Search API
1. Go to https://app.tavily.com/
2. Sign up for an account
3. Create a new API key from the dashboard

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
AZURE_OPENAI_LLM_MODEL=gpt-4o-mini

# OpenWeather API Configuration
OPENWEATHERMAP_API_KEY=your_openweather_api_key_here

# Tavily Search API Configuration
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Run the Application

```bash
python main.py
```

The script will automatically:
1. Initialize Langchain agent with tools
2. Connect to Azure OpenAI, OpenWeather, and Tavily APIs
3. Process 4 automated test queries
4. Display agent reasoning and tool selection
5. Show formatted responses

## 🚀 Usage

### Automated Test Queries

The script includes 4 pre-configured queries:

1. **Weather Query 1**: "What's the weather in Hanoi?"
2. **Web Search Query**: "Tell me about the latest news in AI."
3. **General Knowledge Query**: "Who won the last World Cup?"
4. **Weather Query 2**: "What's the weather in Tokyo?"

### Sample Output

```
================================================================================
LANGCHAIN AI AGENT - WEATHER & WEB SEARCH
================================================================================

✅ Configuration loaded
   Azure Endpoint: https://your-resource.openai.azure.com/
   Deployment: gpt-4o-mini
   OpenWeather API Key: ********************abc1
   Tavily API Key: ********************xyz9

🔄 Initializing OpenWeather API wrapper...
✅ OpenWeather API wrapper initialized

🔄 Initializing Tavily Search API...
✅ Tavily Search API initialized

🔄 Initializing Azure OpenAI LLM...
✅ Azure OpenAI LLM initialized

🔄 Setting up Langchain ReAct agent with tools...
✅ Agent created with 2 tools: ['get_weather', 'tavily_search']

================================================================================
QUERY 1/4
================================================================================

👤 User: What's the weather in Hanoi?

🤖 AI Agent is thinking and selecting tools...

   🌤️  get_weather tool calling: Getting weather for Hanoi

🤖 AI: The current weather in Hanoi is partly cloudy with a temperature of 
approximately 25°C (77°F). The humidity is around 70%, and there's a light 
breeze from the southeast.

================================================================================

================================================================================
QUERY 2/4
================================================================================

👤 User: Tell me about the latest news in AI.

🤖 AI Agent is thinking and selecting tools...

🤖 AI: According to recent news, there have been significant developments in 
AI, including advances in large language models, new applications in healthcare 
diagnostics, and growing discussions around AI ethics and regulation...

```

## 🔍 How It Works

### Architecture Overview

```
User Query
    ↓
[1] Langchain ReAct Agent
    ↓
[2] Tool Selection (LLM decides)
    ↓
[3a] get_weather(city)     [3b] tavily_search(query)
    ↓                            ↓
[4] OpenWeather API         [4] Tavily Search API
    ↓                            ↓
[5] Format Response
    ↓
Return to User
```

### 1. Define Weather Tool

```python
from langchain.tools import tool
from langchain_community.utilities import OpenWeatherMapAPIWrapper

weather = OpenWeatherMapAPIWrapper()

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.
    
    Args:
        city (str): The name of the city to get the weather for.
        
    Returns:
        str: A string describing the current weather in the specified city.
    """
    return weather.run(city)
```

### 2. Initialize Tavily Search

```python
from langchain_tavily import TavilySearch

tavily_search_tool = TavilySearch(
    max_results=1,
    topic="general",
)
```

### 3. Setup Azure OpenAI LLM

```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-07-01-preview",
    api_key=AZURE_OPENAI_API_KEY,
)
```

### 4. Create ReAct Agent

```python
from langgraph.prebuilt import create_react_agent

tools = [get_weather, tavily_search_tool]
agent = create_react_agent(
    model=llm,
    tools=tools,
)
```

### 5. Invoke Agent with Conversation History

```python
messages = []

for user_input in mock_questions:
    messages.append({"role": "user", "content": user_input})
    response = agent.invoke({"messages": messages})
    messages.append({"role": "assistant", "content": response["messages"][-1].content})
    print("AI:", response["messages"][-1].content)
```

## 🧪 Technical Concepts

### ReAct Agent Pattern
- **Re**asoning + **Act**ing architecture
- Agent reasons about which tool to use
- Executes tool and observes results
- Continues reasoning until answer is found

### Langchain Tools
- **@tool decorator**: Defines custom tools with docstrings
- **Tool description**: LLM uses docstring to understand when to use tool
- **Tool arguments**: Typed arguments help LLM format calls correctly

### Tool Routing
- LLM analyzes user query semantics
- Matches query intent to tool capabilities
- Automatically selects appropriate tool(s)
- Can chain multiple tools if needed

### Conversation History
- Maintains message list with roles (user/assistant)
- Provides context for multi-turn conversations
- Enables follow-up questions and references

## 🚧 Challenges & Solutions

### Challenge 1: Tool Selection Accuracy
**Problem**: Agent might select wrong tool for ambiguous queries.

**Solution**: Clear, detailed tool docstrings help LLM understand use cases:
```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.
    
    Use this tool when the user asks about:
    - Current weather conditions
    - Temperature
    - Weather forecasts for a specific location
    """
```

### Challenge 2: API Key Management
**Problem**: Multiple API keys need to be configured correctly.

**Solution**: Centralized environment variable loading with validation:
```python
load_dotenv(dotenv_path=env_path)
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
os.environ["OPENWEATHERMAP_API_KEY"] = OPENWEATHERMAP_API_KEY
```

### Challenge 3: Response Formatting
**Problem**: Raw API responses may not be user-friendly.

**Solution**: LLM automatically formats responses in natural language based on tool outputs.

### Challenge 4: Automated Testing
**Problem**: Assignment requires no manual input for grading.

**Solution**: Pre-defined mock questions with loop:
```python
mock_questions = [
    "What's the weather in Hanoi?",
    "Tell me about the latest news in AI.",
    "Who won the last World Cup?",
    "What's the weather in Tokyo?",
]

for user_input in mock_questions:
    # Process automatically
```

## 🎓 Learning Outcomes

After completing this assignment, you understand:

1. **Langchain Agent Architecture**
   - ReAct pattern implementation
   - Tool definition and registration
   - Agent invocation and message handling

2. **Multi-Tool Integration**
   - OpenWeather API for weather data
   - Tavily Search API for web search
   - Tool routing and selection logic

3. **Conversational AI**
   - Message history management
   - Context-aware responses
   - Multi-turn conversations

4. **API Integration Best Practices**
   - Environment variable management
   - Error handling
   - Response formatting

## 📦 Dependencies

```
langchain-openai>=0.1.0      # Azure OpenAI integration
langchain-community>=0.2.0   # Community tools and utilities
langchain-tavily>=0.1.0      # Tavily search integration
langgraph>=0.1.0             # Agent graph framework
pyowm>=3.3.0                 # OpenWeatherMap wrapper
python-dotenv>=1.0.0         # Environment variable management
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
- OpenWeather free tier: 60 calls/minute, 1,000,000 calls/month
- Tavily free tier: Check current limits on their website

## 🔄 Future Enhancements

- **Additional Tools**: Add calculator, database query, file operations
- **Tool Chaining**: Combine multiple tools for complex queries
- **Error Handling**: Graceful degradation when tools fail
- **Caching**: Cache weather and search results to reduce API calls
- **Streaming Responses**: Real-time streaming for better UX
- **Custom Prompts**: Optimize system prompts for better tool selection
- **Web Interface**: Flask/Gradio UI for interactive chat
- **Multi-language Support**: Support queries in multiple languages

## 📚 Resources

- [Langchain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenWeather API Docs](https://openweathermap.org/api)
- [Tavily Search API](https://docs.tavily.com/)
- [ReAct Pattern Paper](https://arxiv.org/abs/2210.03629)
- [Langchain Tools Guide](https://python.langchain.com/docs/modules/agents/tools/)

## 📄 License

This project is part of the AI Application Engineer course.

---

**Author**: Duke  
**Date**: November 12, 2025  
**Course**: AI Application Engineer
