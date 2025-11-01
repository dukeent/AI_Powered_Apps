"""
Azure OpenAI Function Calling with Batch Processing and Rate Limit Handling

This module demonstrates:
- Function calling with Azure OpenAI API
- Batch processing of multiple inputs
- Rate limit handling using tenacity decorators
- Robust exception handling
- Travel itinerary generation use case
"""

from openai import AzureOpenAI, RateLimitError, APIError
import time
import os
import json
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type
)
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify required environment variables are set
required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_DEPLOYMENT_NAME"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}\n"
        f"Please create a .env file with these variables or set them in your environment."
    )

# Initialize Azure OpenAI client
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

assert endpoint is not None, "AZURE_OPENAI_ENDPOINT is required"
assert api_key is not None, "AZURE_OPENAI_API_KEY is required"
assert deployment_name is not None, "AZURE_DEPLOYMENT_NAME is required"

client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=endpoint,
    api_key=api_key,
)

# Define tools schema for structured output
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_itinerary",
            "description": "Generate a detailed travel itinerary for a given destination and duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "Travel destination city or country"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to plan for"
                    },
                    "daily_activities": {
                        "type": "array",
                        "description": "List of daily activities",
                        "items": {
                            "type": "object",
                            "properties": {
                                "day": {"type": "integer"},
                                "morning": {"type": "string"},
                                "afternoon": {"type": "string"},
                                "evening": {"type": "string"}
                            },
                            "required": ["day", "morning", "afternoon", "evening"]
                        }
                    },
                    "budget_estimate": {
                        "type": "string",
                        "description": "Estimated budget range"
                    },
                    "best_season": {
                        "type": "string",
                        "description": "Best season to visit"
                    }
                },
                "required": ["destination", "days", "daily_activities", "budget_estimate", "best_season"],
            },
        }
    }
]

# Statistics tracking
successful_requests = 0
failed_requests = 0
retry_attempts = 0


@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    reraise=True
)
def call_openai_function(prompt, destination, days):
    """
    Call Azure OpenAI API with function calling and retry logic.
    
    Args:
        prompt (str): The user prompt for itinerary generation
        destination (str): Travel destination
        days (int): Number of days for the trip
        
    Returns:
        dict: API response with itinerary data
        
    Raises:
        RateLimitError: When rate limit is exceeded
        APIError: When API encounters an error
    """
    global successful_requests, retry_attempts
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,  # type: ignore
            messages=[
                {
                    "role": "system",
                    "content": "You are a travel planning expert. Generate detailed, realistic itineraries."
                },
                {
                    "role": "user",
                    "content": f"{prompt} Create a detailed itinerary for {destination} for {days} days with specific activities, restaurants, and attractions."
                }
            ],
            tools=tools,  # type: ignore
            tool_choice={"type": "function", "function": {"name": "generate_itinerary"}}
        )
        
        successful_requests += 1
        return response
        
    except RateLimitError as e:
        retry_attempts += 1
        print(f"⚠️  Rate limit hit for {destination}. Retrying with exponential backoff...")
        raise
    except APIError as e:
        retry_attempts += 1
        print(f"⚠️  API error for {destination}. Retrying...")
        raise
    except Exception as e:
        print(f"❌ Unexpected error in API call for {destination}: {type(e).__name__}: {e}")
        raise


def parse_function_response(response):
    """
    Parse the function calling response from Azure OpenAI.
    
    Args:
        response: API response object
        
    Returns:
        dict: Parsed function arguments or None if parsing fails
    """
    try:
        # Extract the tool calls from the response
        message = response.choices[0].message
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            return {
                'function_name': tool_call.function.name,
                'arguments': function_args
            }
        else:
            # Fallback to content if tool_calls is not available
            return {
                'function_name': 'generate_itinerary',
                'arguments': {'content': message.content}
            }
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Error parsing response: {e}")
        return None


def batch_process(inputs, delay_between_requests=1):
    """
    Process multiple itinerary requests in batch with rate limiting.
    
    Args:
        inputs (list): List of input dictionaries with prompt, destination, and days
        delay_between_requests (int): Seconds to wait between requests
        
    Returns:
        list: Results for each input (None for failed requests)
    """
    global failed_requests
    
    results = []
    total_inputs = len(inputs)
    
    for idx, input_data in enumerate(inputs, 1):
        try:
            prompt = input_data["prompt"]
            destination = input_data["destination"]
            days = input_data["days"]
            
            # Call OpenAI API with retry logic
            response = call_openai_function(prompt, destination, days)
            
            # Parse the structured response
            parsed_result = parse_function_response(response)
            
            if parsed_result:
                results.append({
                    'input': input_data,
                    'success': True,
                    'result': parsed_result,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                results.append({
                    'input': input_data,
                    'success': False,
                    'error': 'Failed to parse response',
                    'timestamp': datetime.now().isoformat()
                })
                failed_requests += 1
            
            # Rate limiting: sleep between requests
            if idx < total_inputs:
                time.sleep(delay_between_requests)
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"❌ Error processing {input_data.get('destination', 'unknown')}: {error_msg}\n")
            results.append({
                'input': input_data,
                'success': False,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            failed_requests += 1
            
    return results


def print_results(results):
    """
    Print formatted results of batch processing.
    
    Args:
        results (list): List of result dictionaries
    """
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING RESULTS")
    print(f"{'='*60}\n")
    
    for idx, result in enumerate(results, 1):
        destination = result['input']['destination']
        days = result['input']['days']
        prompt = result['input']['prompt']
        
        if result['success']:
            parsed_data = result['result']
            
            # Print AI response content
            if 'content' in parsed_data['arguments']:
                response_content = parsed_data['arguments']['content']
            else:
                response_content = json.dumps(parsed_data['arguments'], indent=2)
            
            print(f'Destination: {destination}')
            print(f'Days: {days}')
            print(f'User Prompt: {prompt}')
            print(f'Response: {response_content}')
            print(f"{'-'*60}\n")
        else:
            print(f'Destination: {destination}')
            print(f'Days: {days}')
            print(f'User Prompt: {prompt}')
            print(f'Response: Error - {result.get("error", "Unknown error")}')
            print(f"{'-'*60}\n")


def print_statistics():
    """Print processing statistics."""
    global successful_requests, failed_requests, retry_attempts
    
    total = successful_requests + failed_requests
    
    print(f"\n{'='*60}")
    print(f"📈 PROCESSING STATISTICS")
    print(f"{'='*60}")
    print(f"Total Requests: {total}")
    print(f"Successful: {successful_requests}")
    print(f"Failed: {failed_requests}")
    print(f"Retry Attempts: {retry_attempts}")
    print(f"Success Rate: {(successful_requests/total*100) if total > 0 else 0:.2f}%")
    print(f"{'='*60}\n")


def main():
    """Main execution function."""
    
    # Define batch inputs (3 sample inputs as required)
    batch_inputs = [
        {
            "prompt": "Plan a comprehensive travel itinerary with cultural experiences, local cuisine, and must-see attractions.",
            "destination": "Paris",
            "days": 3
        },
        {
            "prompt": "Create a detailed travel itinerary including traditional activities, modern attractions, and food recommendations.",
            "destination": "Tokyo",
            "days": 5
        },
        {
            "prompt": "Generate a travel plan covering iconic landmarks, entertainment, and dining experiences.",
            "destination": "New York",
            "days": 4
        },
    ]
    
    # Process the batch
    results = batch_process(batch_inputs, delay_between_requests=1)
    
    # Display results
    print_results(results)
    
    # Display statistics
    print_statistics()
    
    # Save results to file in current working directory
    current_dir = os.getcwd()
    output_file = os.path.join(current_dir, f"itinerary_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
