# Assignment 13: AI-Powered Patient Information Chatbot with LangGraph

## 📋 Objective

Build an **AI-driven conversational chatbot** that interactively collects patient information (name, age, symptoms) and provides preliminary health advice using:
- **LangGraph** for conversation flow management
- **AzureChatOpenAI** as the language model
- **FAISS vector store** for medical knowledge retrieval
- **Tavily** for real-time web search (optional)

This demonstrates advanced conversational AI with stateful dialogue management and tool integration.

---

## 🎯 Problem Statement

Patients often need quick, preliminary health assessments before seeing healthcare professionals. This chatbot:
- Engages users naturally to gather essential patient details
- Analyzes collected information to offer relevant preliminary health advice
- Enhances advice with real-time web information via Tavily (optional)
- Provides interactive, guided dialogue flow controlled by LangGraph

---

## 🔧 Technologies Used

- **Python 3.9+**
- **LangChain** - Framework for LLM applications
- **LangGraph** - State machine for conversation flow
- **Azure OpenAI** - GPT-4o-mini for natural language understanding
- **FAISS** - Vector database for medical knowledge retrieval
- **Tavily** - Real-time web search API (optional)
- **Azure OpenAI Embeddings** - text-embedding-3-small for semantic search

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- Azure OpenAI API access (embedding + LLM endpoints)
- (Optional) Tavily API key for real-time web search

### Step 1: Navigate to Assignment Directory

```bash
cd Assignment_13
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langchain>=0.1.0` - LLM application framework
- `langchain-openai>=0.1.0` - Azure OpenAI integration
- `langchain-community>=0.1.0` - Community tools (Tavily, FAISS)
- `langgraph>=0.1.0` - Conversation flow graph builder
- `langchain-tavily>=0.1.0` - Tavily search integration
- `faiss-cpu>=1.7.4` - Vector similarity search
- `python-dotenv>=1.0.0` - Environment variable management

### Step 3: Configure Environment Variables

Create a `.env` file in the `Assignment_13` directory:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# Azure OpenAI Embedding Configuration
AZURE_OPENAI_EMBEDDING_API_KEY=your_embedding_api_key
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://aiportalapi.stu-platform.live/jpe
AZURE_OPENAI_EMBED_MODEL=text-embedding-3-small

# Azure OpenAI LLM Configuration
AZURE_OPENAI_LLM_API_KEY=your_llm_api_key
AZURE_OPENAI_LLM_ENDPOINT=https://aiportalapi.stu-platform.live/jpe
AZURE_OPENAI_LLM_MODEL=gpt-4o-mini

# Tavily API Key (Optional)
TAVILY_API_KEY=your_tavily_api_key
```

---

## 🚀 Usage

### Running the Application

```bash
python main.py
```

The application will:
1. Load medical knowledge base (10 documents)
2. Initialize FAISS vector store with embeddings
3. Set up Azure OpenAI LLM with tools
4. Build LangGraph conversation workflow
5. Run 3 automated patient consultations
6. Display preliminary health advice for each case

### Expected Output

```
================================================================================
AI-POWERED PATIENT INFORMATION CHATBOT
================================================================================

📚 Medical Knowledge Base: 10 documents loaded

✅ Configuration loaded
   LLM Endpoint: https://aiportalapi.stu-platform.live/jpe
   LLM Model: gpt-4o-mini
   Embedding Model: text-embedding-3-small

🔄 Initializing FAISS vector store...
✅ FAISS vector store initialized with 10 documents

🔄 Initializing Azure OpenAI LLM...
✅ Azure OpenAI LLM initialized with tools

🔄 Building LangGraph workflow...
✅ LangGraph workflow compiled successfully

================================================================================
RUNNING AUTOMATED PATIENT CONSULTATIONS
================================================================================

================================================================================
CONSULTATION 1/3
================================================================================

👤 Patient Information:
   Name: John Doe
   Age: 28
   Symptoms: I feel tired and have a sore throat. What should I do?

🤖 Medical Assistant Response:
--------------------------------------------------------------------------------
Based on your symptoms of fatigue and sore throat, here are some preliminary recommendations:

1. **Sore Throat Care**: Drink warm fluids such as tea with honey or warm water. Avoid cold beverages as they may irritate your throat further.

2. **Rest and Hydration**: Make sure you're getting adequate rest and staying well-hydrated. Fatigue can be related to various factors including poor sleep or nutritional deficiencies.

3. **Monitor Your Symptoms**: Keep an eye on your symptoms. If they worsen or you develop additional symptoms like high fever, difficulty swallowing, or persistent fatigue, please consult a healthcare professional.

Please remember, these are preliminary suggestions. If your symptoms persist or worsen, it's important to seek advice from a qualified healthcare provider.
--------------------------------------------------------------------------------

📊 Conversation Flow:
   Total messages exchanged: 5
   Tool calls made: 1
   Tools used:
      - retrieve_advice

================================================================================
✅ ALL PATIENT CONSULTATIONS COMPLETED
================================================================================

📈 Summary:
   Total consultations: 3
   Medical knowledge documents: 10
   Tools available: retrieve_advice, tavily_search_results

💡 Key Features Demonstrated:
   ✅ LangGraph conversation flow management
   ✅ Azure OpenAI LLM integration
   ✅ FAISS vector store for medical knowledge retrieval
   ✅ Tool-based architecture (retrieve_advice, Tavily search)
   ✅ Automated patient consultations (no manual input)
   ✅ Context-aware preliminary health advice

⚕️ Disclaimer: This is a prototype for educational purposes.
   Always consult qualified healthcare professionals for medical advice.
```

---

## 🏗️ Architecture

### LangGraph Workflow

```
START
  ↓
call_model (LLM with tools)
  ↓
should_continue? (conditional routing)
  ├─→ tools (if tool_calls exist)
  │    ↓
  │  Execute tools (retrieve_advice, tavily_search)
  │    ↓
  │  call_model (with tool results)
  │    ↓
  │  should_continue?
  │
  └─→ END (if no tool_calls)
```

### Components

1. **Medical Knowledge Base**: 10 documents with common health advice
2. **FAISS Vector Store**: Semantic search over medical documents
3. **Tools**:
   - `retrieve_advice`: Search internal medical knowledge base
   - `tavily_search_results`: Fetch real-time web information
4. **LLM**: Azure GPT-4o-mini with tool calling capabilities
5. **LangGraph**: Manages conversation state and tool execution flow

---

## 📚 Medical Knowledge Base

The system includes 10 medical advisory documents covering:
- Sore throat care
- Fever management
- Dizziness assessment
- Persistent cough evaluation
- Fatigue causes
- Headache management
- Wound care
- Stomach pain
- Chest pain warnings
- Insomnia improvement

---

## 🧠 Key Concepts Covered

### 1. **Conversational AI**
Building a chatbot that interacts naturally with patients using context-aware responses.

### 2. **LangGraph State Management**
```python
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("call_model", should_continue, ["tools", END])
```

**Benefits**:
- Clear conversation flow visualization
- Deterministic state transitions
- Easy to debug and extend
- Tool integration with automatic routing

### 3. **Tool Integration**
```python
@tool
def retrieve_advice(user_input: str) -> str:
    """Searches internal medical documents for relevant patient advice."""
    docs = retriever.get_relevant_documents(user_input)
    return "\n".join(doc.page_content for doc in docs)

llm_with_tools = llm.bind_tools([retrieve_advice, tavily_tool])
```

**How it works**:
1. LLM receives user message
2. LLM decides which tool(s) to use
3. Tools execute and return results
4. LLM synthesizes final response

### 4. **FAISS Vector Similarity Search**
```python
embedding_model = AzureOpenAIEmbeddings(...)
db = FAISS.from_documents(mock_chunks, embedding_model)
retriever = db.as_retriever()
```

**Purpose**: Semantic search over medical knowledge base to find relevant advice based on patient symptoms.

### 5. **Conditional Routing**
```python
def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END
```

**Logic**: Decides whether to execute tools or end conversation based on LLM's tool call decisions.

---

## 🔍 Automated Test Cases

The application includes 3 pre-configured patient scenarios:

### Test Case 1: Fatigue + Sore Throat
- **Patient**: John Doe, 28
- **Symptoms**: Tired, sore throat
- **Expected**: Advice on warm fluids, rest, hydration, iron deficiency check

### Test Case 2: Persistent Cough + Fever
- **Patient**: Sarah Chen, 35
- **Symptoms**: Cough for 3 weeks, mild fever
- **Expected**: Evaluation advice, infection/allergy assessment

### Test Case 3: Dizziness + Headaches
- **Patient**: Michael Rodriguez, 42
- **Symptoms**: Dizziness, headaches
- **Expected**: Blood pressure check, hydration advice, medical evaluation

---

## 🛠️ Technical Implementation Details

### LangGraph vs Traditional Chatbots

| Aspect | Traditional | LangGraph |
|--------|-------------|-----------|
| State Management | Manual tracking | Built-in MessagesState |
| Tool Routing | If/else logic | Conditional edges |
| Conversation Flow | Linear | Graph-based (flexible) |
| Debugging | Print statements | Visual graph inspection |
| Scalability | Complex for multi-step | Clean node additions |

### Tool Execution Flow

1. **User Input** → System receives patient symptoms
2. **LLM Analysis** → GPT-4o-mini analyzes and decides to use `retrieve_advice` tool
3. **Tool Execution** → FAISS searches medical knowledge base
4. **Result Processing** → Tool returns relevant documents
5. **Response Generation** → LLM synthesizes advice based on retrieved knowledge
6. **Output** → Patient receives preliminary health recommendations

### Message State Structure

```python
MessagesState = {
    "messages": [
        SystemMessage(content="You are a medical assistant..."),
        HumanMessage(content="I feel tired..."),
        AIMessage(content="...", tool_calls=[...]),
        ToolMessage(content="...", tool_call_id="..."),
        AIMessage(content="Based on your symptoms..."),
    ]
}
```

---

## 🎓 Learning Outcomes

### 1. **LangGraph Mastery**
- Building state machines for conversations
- Adding nodes and edges programmatically
- Conditional routing based on LLM decisions
- Tool node integration

### 2. **Azure OpenAI Integration**
- Separate endpoints for embeddings and LLM
- Tool binding for function calling
- Temperature and parameter tuning

### 3. **Vector Database Usage**
- FAISS for semantic similarity search
- Document embedding and retrieval
- Building knowledge bases from text documents

### 4. **Healthcare AI Applications**
- Preliminary symptom assessment
- Medical knowledge retrieval
- Patient interaction patterns
- Disclaimer and limitation awareness

### 5. **Prompt Engineering**
- Context-aware system prompts
- Patient information integration
- Tool usage instructions
- Safety and disclaimers

---

## 📊 Example Interaction Breakdown

**Patient Input**: "I feel tired and have a sore throat"

**LangGraph Processing**:
1. START → call_model
2. LLM analyzes: "Need medical advice → use retrieve_advice tool"
3. should_continue → "tools" (tool_calls detected)
4. tools → Execute retrieve_advice("tired sore throat")
5. FAISS retrieves:
   - "Sore throat → warm fluids"
   - "Fatigue → iron deficiency or sleep"
6. call_model → LLM synthesizes response with tool results
7. should_continue → END (no more tool_calls)

**Final Output**: Structured advice with actionable recommendations + medical consultation reminder

---

## ⚠️ Limitations & Disclaimers

1. **Not a Medical Diagnosis Tool**: This is for educational purposes only
2. **Preliminary Advice Only**: Always consult healthcare professionals
3. **Limited Knowledge Base**: 10 documents (expandable in production)
4. **No Emergency Detection**: Cannot handle life-threatening situations
5. **Prototype Quality**: Not for actual medical use

---

## 🚀 Future Enhancements

1. **Expanded Knowledge Base**: 1000+ medical documents
2. **Symptom Severity Classification**: Urgent vs. non-urgent triage
3. **Multi-Turn Conversations**: Ask follow-up questions
4. **Patient History Tracking**: Store previous consultations
5. **Multilingual Support**: Multiple language options
6. **Voice Input/Output**: Speech-to-text integration
7. **Appointment Booking**: Schedule with healthcare providers
8. **Emergency Detection**: Identify critical symptoms

---

## 🔗 Related Technologies

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Tavily Search API](https://tavily.com/)

---

## 📝 Submission Checklist

- [x] Single Python file (`main.py`) with complete implementation
- [x] Automated test cases (3 patient scenarios)
- [x] No manual input (`input()` function)
- [x] LangGraph conversation flow implemented
- [x] Azure OpenAI LLM integration
- [x] FAISS vector store for knowledge retrieval
- [x] Tool-based architecture (retrieve_advice, Tavily)
- [x] Sample inputs and outputs
- [x] Comprehensive README with architecture explanation
- [x] Environment configuration files
- [x] Requirements.txt with dependencies

---

## 👤 Author

**Duke**  
Assignment 13 - AI-Powered Applications Course  
Date: November 18, 2025

---

## 📄 License

Educational project - for learning purposes only.

---

**⚕️ Medical Disclaimer**: This chatbot is a prototype for educational purposes demonstrating AI conversation flow management. It does NOT provide actual medical advice. Always consult qualified healthcare professionals for medical concerns.
