# Assignment 04: Azure OpenAI Function Calling with Batch Processing

## Overview
This project demonstrates the use of Azure OpenAI API with the modern `tools` API (replacing deprecated `functions`), batch processing, rate limit handling, and robust exception management. The implementation focuses on generating travel itineraries for multiple destinations efficiently.

## Features
- ✅ **Tool Calling**: Structured output using Azure OpenAI's tools API (modern replacement for function calling)
- ✅ **Batch Processing**: Efficient handling of multiple API requests
- ✅ **Rate Limit Handling**: Automatic retry with exponential backoff using tenacity
- ✅ **Exception Handling**: Graceful error management without crashing
- ✅ **Detailed Output**: Day-by-day itineraries with activities, budgets, and travel tips
- ✅ **Statistics Tracking**: Success/failure rates and retry attempts

## Requirements
```
openai>=1.0.0
tenacity>=8.0.0
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project directory with your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_DEPLOYMENT_NAME=gpt-4o-mini
```

**Security Note**: The `.env` file is already in `.gitignore` to prevent accidental commits of sensitive data.

### 3. Run the Script
```bash
python main.py
```

## How It Works

### Architecture
The solution uses a functional programming approach with the following components:

1. **Tools Schema Definition**: Defines the structure for travel itinerary outputs using the modern `tools` API
2. **Retry Decorator**: Handles rate limits and transient API errors
3. **Batch Processor**: Manages multiple requests sequentially with delays
4. **Result Parser**: Extracts and formats tool call responses
5. **Statistics Tracker**: Monitors success/failure rates

### Design Choices

#### 1. Batch Processing Strategy
- **Sequential Processing**: Processes requests one at a time to avoid overwhelming the API
- **Configurable Delays**: 1-second delay between requests (adjustable)
- **Rationale**: Prevents aggressive rate limit hits while maintaining simplicity

#### 2. Retry Logic with Tenacity
```python
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    reraise=True
)
```

**Design Decisions**:
- **Exponential Backoff**: Random wait time between 1-10 seconds to avoid thundering herd
- **Max 5 Attempts**: Balance between persistence and resource usage
- **Specific Exceptions**: Only retry on RateLimitError and APIError (not all exceptions)
- **Reraise**: Ensures final failures are caught by outer exception handler

#### 3. Exception Handling Hierarchy
```
try-except at multiple levels:
├── API Call Level: Catches RateLimitError, APIError (with retry)
├── Parsing Level: Handles JSON decode errors
└── Batch Level: Catches all exceptions to prevent script crash
```

#### 4. Tools API (Modern Approach)
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_itinerary",
            "description": "Generate a detailed travel itinerary...",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "days": {"type": "integer"},
                    "daily_activities": {"type": "array", ...},
                    "budget_estimate": {"type": "string"},
                    "best_season": {"type": "string"}
                },
                "required": ["destination", "days", "daily_activities", "budget_estimate", "best_season"]
            }
        }
    }
]
```

**Design Decisions**:
- Uses modern `tools` API instead of deprecated `functions`
- All major fields are required to ensure complete itineraries
- Structured daily_activities array with morning/afternoon/evening breakdown
- Ensures consistent, detailed responses with budget and seasonal information

## Sample Output

### Console Output
```
============================================================
BATCH PROCESSING RESULTS
============================================================

Destination: Paris
Days: 3
User Prompt: Plan a comprehensive travel itinerary with cultural experiences, local cuisine, and must-see attractions.
Response: {
  "destination": "Paris",
  "days": 3,
  "daily_activities": [
    {
      "day": 1,
      "morning": "Visit Eiffel Tower and Trocadéro Gardens",
      "afternoon": "Explore Louvre Museum",
      "evening": "Seine River cruise and dinner"
    },
    {
      "day": 2,
      "morning": "Notre-Dame Cathedral and Latin Quarter",
      "afternoon": "Musée d'Orsay",
      "evening": "Montmartre and Sacré-Cœur"
    },
    {
      "day": 3,
      "morning": "Palace of Versailles",
      "afternoon": "Champs-Élysées shopping",
      "evening": "Arc de Triomphe and farewell dinner"
    }
  ],
  "budget_estimate": "$1500-2500 USD per person",
  "best_season": "Spring (April-June) or Fall (September-November)"
}
------------------------------------------------------------
```

### Statistics Output
```
============================================================
📈 PROCESSING STATISTICS
============================================================
Total Requests: 3
Successful: 3
Failed: 0
Retry Attempts: 0
Success Rate: 100.00%
============================================================
```

## Challenges and Solutions

### Challenge 1: Rate Limit Management
**Problem**: Azure OpenAI has rate limits that can cause request failures.

**Solution**: 
- Implemented tenacity retry decorator with exponential backoff
- Added configurable delays between batch requests
- Tracked retry attempts for monitoring

### Challenge 2: Tool Calling Response Parsing
**Problem**: Modern OpenAI API uses `tool_calls` instead of deprecated `function_call`.

**Solution**:
- Updated to use `tools` array with proper structure
- Changed parsing to access `message.tool_calls[0]` instead of `message.function_call`
- Maintained backward compatibility with fallback to message content
- Required all important fields to ensure complete responses

### Challenge 3: Maintaining Script Robustness
**Problem**: Any unhandled exception could crash the entire batch process.

**Solution**:
- Multi-level exception handling (API, parsing, batch levels)
- Individual request failures don't stop batch processing
- Detailed error logging for debugging while continuing execution

### Challenge 4: Incomplete AI Responses
**Problem**: AI was only returning minimal required fields (destination and days) without detailed itineraries.

**Solution**:
- Enhanced user prompts to explicitly request detailed content
- Made all important fields required in the tools schema
- Added required properties to nested objects (daily activities)
- Included specific instructions about activities, restaurants, and attractions in prompts

## File Structure
```
Assignment_04/
├── main.py                      # Main implementation (functional approach)
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── .env                         # Environment variables (not in git)
├── .gitignore                   # Prevents committing sensitive files
└── itinerary_results_*.json     # Generated output files
```

## Testing
The script includes 3 sample inputs as required:
1. **Paris** (3 days) - European destination
2. **Tokyo** (5 days) - Asian destination
3. **New York** (4 days) - North American destination

These diverse inputs test the system's ability to handle different:
- Trip durations
- Cultural contexts
- Geographic regions

## Extensibility

### Adding More Destinations
Simply extend the `batch_inputs` list:
```python
batch_inputs.append({
    "prompt": "Your custom prompt",
    "destination": "Rome",
    "days": 7
})
```

### Changing the Use Case
Modify the `tools` schema to support different domains:
- Recipe summarization
- Customer support email drafting
- Product descriptions
- Code documentation

Example for recipe generation:
```python
tools = [{
    "type": "function",
    "function": {
        "name": "generate_recipe",
        "parameters": {
            "type": "object",
            "properties": {
                "dish_name": {"type": "string"},
                "ingredients": {"type": "array"},
                "steps": {"type": "array"}
            },
            "required": ["dish_name", "ingredients", "steps"]
        }
    }
}]
```

### Async Processing (Optional Enhancement)
For higher throughput, the code can be refactored to use `asyncio` and `aiohttp`:
```python
async def batch_process_async(inputs):
    tasks = [call_openai_function_async(inp) for inp in inputs]
    return await asyncio.gather(*tasks)
```

## Best Practices Demonstrated
1. ✅ Modular, functional programming design
2. ✅ Modern tools API (not deprecated functions API)
3. ✅ Comprehensive error handling
4. ✅ Environment variables for security
5. ✅ Configurable parameters
6. ✅ Type hints and documentation
7. ✅ DRY (Don't Repeat Yourself) principle
8. ✅ Separation of concerns
9. ✅ Output saved to current working directory
10. ✅ Clear, formatted console output

## License
Educational project for Elevate AI Workshop

## Author
Created as part of Assignment 04 - Azure OpenAI Function Calling Exercise
