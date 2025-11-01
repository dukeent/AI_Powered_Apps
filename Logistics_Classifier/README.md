# Assignment 06: Logistics Delay Classification with NLP and Azure OpenAI

## 1. Objective
Build an automated two-stage classification pipeline that:
- Analyzes unstructured delivery and maintenance logs
- Automatically determines root causes of delivery delays
- Maps free-text log entries into 8 predefined operational categories
- Combines keyword-based heuristics with Azure OpenAI refinement
- Reduces manual workload and accelerates operations reporting for logistics teams

## 2. Problem Statement
- Logistics operations generate thousands of free-text incident reports from drivers and maintenance teams
- Examples:
  - "unexpected rainstorm at warehouse"
  - "vehicle failed mid-route"
- Manual analysis is inefficient and error-prone
- Need: Automated classification pipeline using NLP + Azure ChatOpenAI refinement layer

## 3. Inputs/Shared Artifacts

### Azure OpenAI Resource
- API Key
- Endpoint URL
- Deployment Name
- (Set as environment variables)

### Mock Dataset
| log_id | log_entry |
|--------|-----------|
| 1 | "Driver reported heavy traffic on highway due to construction" |
| 2 | "Package not accepted, customer unavailable at given time" |
| 3 | "Vehicle engine failed during route, replacement dispatched" |
| 4 | "Unexpected rainstorm delayed loading at warehouse" |
| 5 | "Sorting label missing, required manual barcode scan" |
| 6 | "Driver took a wrong turn and had to reroute" |
| 7 | "No issue reported, arrived on time" |
| 8 | "Address was incorrect, customer unreachable" |
| 9 | "System glitch during check-in at loading dock" |
| 10 | "Road accident caused a long halt near delivery point" |

## 4. Expected Outcome

### Classification Categories
```
[
  "Traffic",
  "Customer Issue",
  "Vehicle Issue",
  "Weather",
  "Sorting/Labeling Error",
  "Human Error",
  "Technical System Failure",
  "Other"
]
```

### Example Input → Output

| Log Entry | Predicted Category |
|-----------|-------------------|
| "Driver reported heavy traffic..." | Traffic |
| "Customer not available..." | Customer Issue |
| "Engine failed during route..." | Vehicle Issue |
| "Unexpected rainstorm..." | Weather |
| "Sorting label missing..." | Sorting/Labeling Error |
| "Wrong turn, rerouted..." | Human Error |
| "System glitch during check-in..." | Technical System Failure |
| "Address was incorrect..." | Customer Issue |
| "Road accident caused a halt..." | Traffic |

## 5. Concepts Covered
- Azure OpenAI API usage
- Prompt Engineering
- NLP Task - Text Classification
- Environment Variable Management
- Error Handling and Validation

## 6. Implementation Steps

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure Azure OpenAI credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your actual Azure OpenAI credentials
   ```

   Required environment variables in `.env`:
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_API_KEY`
   - `AZURE_DEPLOYMENT_NAME`

### Architecture
**Two-Stage Classification Pipeline:**
1. **Stage 1: Heuristic Pre-classifier**
   - Fast keyword-based pattern matching
   - Initial categorization based on common terms
   - Low latency, no API calls

2. **Stage 2: Azure OpenAI Refinement**
   - Validates and corrects initial classification
   - Context-aware semantic understanding
   - Handles ambiguous cases

### Execution
```bash
python logistics_classifier.py
```

### Output
- Console: Real-time classification progress and results table
- JSON file: `classification_results_YYYYMMDD_HHMMSS.json`

## 7. Project Structure
```
Assignment_06/
├── logistics_classifier.py      # Main implementation
├── requirements.txt             # Dependencies
├── .env.example                 # Configuration template
├── .gitignore                   # Git exclusions
├── README.md                    # This file
├── CLASSIFICATION_RESULTS.md    # Results table
└── APPROACH_COMPARISON.md       # Classification vs RAG comparison
```

## 8. Submission Checklist
- [x] Single .py file with complete implementation
- [x] Uses dummy input list (auto input, no manual input())
- [x] Two-stage pipeline: keyword-based + AI refinement
- [x] Displays predicted category for all 10 sample log entries
- [x] Classification results table (CLASSIFICATION_RESULTS.md)
- [x] Approach comparison summary (APPROACH_COMPARISON.md)
- [x] requirements.txt with python-dotenv
- [x] .env.example for configuration template
- [x] .gitignore to exclude sensitive files
- [x] Clean, optimized code without redundant logic

## 9. Key Differences: Classification vs. Retrieval (RAG)

> **See [APPROACH_COMPARISON.md](APPROACH_COMPARISON.md) for detailed comparison**

### This Classification Approach
- **Goal**: Assign predefined labels to text
- **Output**: Single category from fixed set
- **Method**: Keyword matching + LLM validation
- **Best for**: Structured categorization, routing, analytics

### Retrieval-Based (RAG) Approach
- **Goal**: Find relevant information from knowledge base
- **Output**: Retrieved documents + generated answer
- **Method**: Vector search + context-augmented generation
- **Best for**: Q&A, document search, knowledge lookup

### When to Use Classification
✓ Fixed, well-defined categories  
✓ Routing/triage decisions  
✓ Structured reporting  
✓ High-volume repetitive tasks  
✓ Labels known in advance  

### When to Use RAG
✓ Open-ended questions  
✓ Large knowledge bases  
✓ Dynamic information  
✓ Source attribution needed  
✓ Complex reasoning  

---

## Usage Example

```python
from logistics_classifier import LogisticsClassifier

# Initialize
classifier = LogisticsClassifier()

# Classify single entry
result = classifier.classify_log(
    log_id=1,
    text="Driver reported heavy traffic on highway"
)

print(f"Category: {result['final_category']}")
# Output: Category: Traffic
```

## Sample Output

```
======================================================================
CLASSIFICATION RESULTS
======================================================================

ID   Log Entry                                          Category
---- -------------------------------------------------- -------------------------
1    Driver reported heavy traffic on highway due...   Traffic
2    Package not accepted, customer unavailable a...   Customer Issue
3    Vehicle engine failed during route, replacem...   Vehicle Issue
4    Unexpected rainstorm delayed loading at ware...   Weather
5    Sorting label missing, required manual barco...   Sorting/Labeling Error
6    Driver took a wrong turn and had to reroute      Human Error
7    No issue reported, arrived on time                Other
8    Address was incorrect, customer unreachable      Customer Issue
9    System glitch during check-in at loading dock    Technical System Failure
10   Road accident caused a long halt near delive...   Traffic
```

---

## 10. Documentation

- **[CLASSIFICATION_RESULTS.md](CLASSIFICATION_RESULTS.md)** - Complete results table with predicted categories
- **[APPROACH_COMPARISON.md](APPROACH_COMPARISON.md)** - Classification vs RAG approach analysis

---

**Author**: Duc Nguyen  
**Date**: October 29, 2025
