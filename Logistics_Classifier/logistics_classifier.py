"""
Assignment 06: Logistics Delay Classification with NLP and Azure OpenAI

Author: Duc Nguyen
Date: October 29, 2025
"""

import os
from openai import AzureOpenAI
from typing import Dict, List
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()


class LogisticsClassifier:
    """Two-stage classifier: keyword-based + Azure OpenAI refinement."""
    
    CATEGORIES = [
        "Traffic", "Customer Issue", "Vehicle Issue", "Weather",
        "Sorting/Labeling Error", "Human Error", "Technical System Failure", "Other"
    ]
    
    def __init__(self):
        """Initialize Azure OpenAI client."""
        self.client = AzureOpenAI(
            api_version="2024-07-01-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
        )
        self.deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        if not self.deployment_name:
            raise ValueError("AZURE_DEPLOYMENT_NAME environment variable is not set")
        
        self.keywords = {
            "traffic": "Traffic", "road accident": "Traffic", "construction": "Traffic",
            "customer": "Customer Issue", "unavailable": "Customer Issue", "unreachable": "Customer Issue",
            "engine": "Vehicle Issue", "vehicle": "Vehicle Issue",
            "rain": "Weather", "storm": "Weather",
            "label": "Sorting/Labeling Error", "barcode": "Sorting/Labeling Error",
            "wrong turn": "Human Error", "reroute": "Human Error", "incorrect": "Human Error",
            "system": "Technical System Failure", "glitch": "Technical System Failure"
        }
    
    def initial_classify(self, text: str) -> str:
        """Stage 1: Keyword-based classification."""
        text_lower = text.lower()
        for keyword, category in self.keywords.items():
            if keyword in text_lower:
                return category
        return "Other"
    
    def refine_classification(self, text: str, initial_label: str) -> str:
        """Stage 2: Azure OpenAI refinement."""
        prompt = f"""You are a logistics assistant. A log entry has been auto-categorized as "{initial_label}". 
Confirm or correct it by choosing one category: {', '.join(self.CATEGORIES)}

Log Entry: \"\"\"{text}\"\"\"

Return only the category name, no explanation."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,  # type: ignore
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content
            return content.strip() if content else initial_label
        except Exception as e:
            print(f"⚠️  Error: {e}")
            return initial_label
    
    def classify_log(self, log_id: int, text: str) -> Dict:
        """Classify a single log entry."""
        initial = self.initial_classify(text)
        final = self.refine_classification(text, initial)
        return {
            "log_id": log_id,
            "log_entry": text,
            "initial_category": initial,
            "final_category": final,
            "timestamp": datetime.now().isoformat()
        }
    
    def classify_batch(self, log_entries: List[Dict]) -> List[Dict]:
        """Classify a batch of log entries."""
        print(f"\n{'='*70}\nLOGISTICS DELAY CLASSIFICATION PIPELINE\n{'='*70}\n")
        print(f"Processing {len(log_entries)} log entries...\n")
        
        results = []
        for entry in log_entries:
            print(f"[{entry['log_id']}/10] Classifying...")
            result = self.classify_log(entry['log_id'], entry['log_entry'])
            results.append(result)
            print(f"  Initial: {result['initial_category']} → Final: {result['final_category']}\n")
        
        return results
    
    def generate_report(self, results: List[Dict]):
        """Generate classification report."""
        print(f"\n{'='*70}\nCLASSIFICATION RESULTS\n{'='*70}\n")
        print(f"{'ID':<4} {'Log Entry':<50} {'Category':<25}")
        print(f"{'-'*4} {'-'*50} {'-'*25}")
        
        for r in results:
            log_entry = r['log_entry'][:47] + "..." if len(r['log_entry']) > 50 else r['log_entry']
            print(f"{r['log_id']:<4} {log_entry:<50} {r['final_category']:<25}")
        
        # Category distribution
        category_counts = {}
        for r in results:
            category_counts[r['final_category']] = category_counts.get(r['final_category'], 0) + 1
        
        print(f"\n{'='*70}\nCATEGORY DISTRIBUTION\n{'-'*70}")
        for category in sorted(category_counts.keys()):
            count = category_counts[category]
            print(f"  {category:<30} {count:>3} ({count/len(results)*100:.1f}%)")
        
        # Statistics
        refinements = sum(1 for r in results if r['initial_category'] != r['final_category'])
        print(f"\n{'='*70}\nSTATISTICS\n{'-'*70}")
        print(f"  Total entries: {len(results)}")
        print(f"  Unique categories: {len(category_counts)}")
        print(f"  Refined by AI: {refinements} ({refinements/len(results)*100:.1f}%)")
        print(f"{'='*70}\n")


def main():
    """Main execution function."""
    
    # Step 1: Input Data - Mock Dataset
    log_entries = [
        {"log_id": 1, "log_entry": "Driver reported heavy traffic on highway due to construction"},
        {"log_id": 2, "log_entry": "Package not accepted, customer unavailable at given time"},
        {"log_id": 3, "log_entry": "Vehicle engine failed during route, replacement dispatched"},
        {"log_id": 4, "log_entry": "Unexpected rainstorm delayed loading at warehouse"},
        {"log_id": 5, "log_entry": "Sorting label missing, required manual barcode scan"},
        {"log_id": 6, "log_entry": "Driver took a wrong turn and had to reroute"},
        {"log_id": 7, "log_entry": "No issue reported, arrived on time"},
        {"log_id": 8, "log_entry": "Address was incorrect, customer unreachable"},
        {"log_id": 9, "log_entry": "System glitch during check-in at loading dock"},
        {"log_id": 10, "log_entry": "Road accident caused a long halt near delivery point"}
    ]
    
    # Step 2: Initialize Classifier
    classifier = LogisticsClassifier()
    
    # Step 3: Classify All Entries
    results = classifier.classify_batch(log_entries)
    
    # Step 4: Generate Report
    classifier.generate_report(results)
    
    # Step 5: Save Results to JSON
    output_filename = f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_filename}\n")
    
    # Step 6: Summary Analysis
    print(f"{'='*70}")
    print("APPROACH SUMMARY")
    print(f"{'='*70}\n")
    
    summary = """
This classification approach uses a two-stage pipeline:

1. HEURISTIC PRE-CLASSIFIER (Stage 1):
   - Fast keyword-based pattern matching
   - Low computational cost
   - Good for obvious cases with clear keywords
   - Serves as initial filter and baseline

2. AZURE OPENAI REFINEMENT (Stage 2):
   - Context-aware semantic understanding
   - Handles ambiguous or complex entries
   - Validates and corrects initial classifications
   - Leverages GPT-4o-mini's language comprehension

HOW THIS DIFFERS FROM RETRIEVAL-BASED PIPELINES:

Classification Approach (This Assignment):
• Goal: Assign predefined labels to text
• Output: Single category from fixed set
• Method: Pattern matching + LLM validation
• Use case: Structured categorization, tagging, routing

Retrieval-Based Approach (RAG):
• Goal: Find relevant information from knowledge base
• Output: Retrieved documents + generated answer
• Method: Vector search + context-augmented generation
• Use case: Question answering, document search, knowledge lookup

WHEN TO USE CLASSIFICATION:
✓ Fixed, well-defined categories
✓ Routing/triage decisions
✓ Structured reporting and analytics
✓ High-volume repetitive categorization
✓ When labels are known in advance

WHEN TO USE RETRIEVAL (RAG):
✓ Open-ended questions
✓ Large knowledge bases
✓ Dynamic/changing information
✓ Need for source attribution
✓ Complex multi-document reasoning

BEST FIT FOR THIS LOGISTICS SCENARIO:
This classification approach is ideal because:
• Delay categories are predefined and stable
• Need fast, consistent categorization at scale
• Clear operational categories map to business processes
• Results feed into structured dashboards and reports
• Two-stage design balances speed and accuracy
    """
    
    print(summary)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
