import os
import json
from typing import Optional, Dict, Any
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# System prompt for instruction generation
SYSTEM_PROMPT = """You are an AI assistant generating detailed work instructions for automotive manufacturing tasks.
Your response MUST be a valid JSON object in the exact format shown below, with no additional text before or after the JSON.

Required JSON format:
{
    "safety_precautions": [
        "List each safety precaution as a separate string",
        "Include all relevant PPE requirements",
        "List any special safety considerations"
    ],
    "required_tools": [
        "List each required tool as a separate string",
        "Include any special equipment needed",
        "Specify tool specifications where relevant"
    ],
    "steps": [
        "List each step as a separate string",
        "Steps should be clear and specific",
        "Include measurements and specifications"
    ],
    "acceptance_checks": [
        "List each quality check as a separate string",
        "Include specific measurements or criteria",
        "Define pass/fail conditions"
    ]
}

Ensure your response:
1. Is ONLY the JSON object, with no additional text
2. Uses proper JSON syntax with quoted strings
3. Has all four required arrays, even if empty
4. Contains detailed, relevant information for the task"""

# Step 1: Mock Input Data
task_descriptions = [
    "Install the battery module in the rear compartment, connect to the high-voltage harness, and verify torque on fasteners.",
    "Calibrate the ADAS (Advanced Driver Assistance Systems) radar sensors on the front bumper using factory alignment targets.",
    "Apply anti-corrosion sealant to all exposed welds on the door panels before painting.",
    "Perform leak test on coolant system after radiator installation. Record pressure readings and verify against specifications.",
    "Program the infotainment ECU with the latest software package and validate connectivity with dashboard display."
]

# Step 2: OpenAI Azure Client Setup
client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=str(os.getenv("AZURE_OPENAI_ENDPOINT")),
    api_key=str(os.getenv("AZURE_OPENAI_API_KEY")),
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

def generate_instructions(prompt: str) -> Optional[str]:
    """Generate instructions using Azure OpenAI.
    
    Args:
        prompt (str): The prompt to send to the API
        
    Returns:
        str: The generated instructions
    """
    try:
        from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
        
        # Create properly typed messages
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=SYSTEM_PROMPT
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=prompt
            ),
            ChatCompletionSystemMessageParam(
                role="system",
                content="Remember to respond ONLY with a valid JSON object, no other text."
            )
        ]

        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=0,  # Lower temperature for more consistent JSON formatting
            max_tokens=2000,  # Increased token limit for detailed instructions
            response_format={"type": "json_object"}  # Request JSON response
        )
        
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                # Try to parse JSON to validate format
                json.loads(content)  # Will raise JSONDecodeError if invalid
                return content.strip()
        return None
    except json.JSONDecodeError as e:
        print(f"API returned invalid JSON: {str(e)}")
        return None
    except Exception as e:
        print(f"Error generating instructions: {str(e)}")
        return None

def generate_work_instructions(task_description: str) -> Optional[Dict[str, Any]]:
    """Generate detailed work instructions for a given automotive manufacturing task.

    Args:
        task_description (str): Description of the task to generate instructions for.

    Returns:
        dict: Generated work instructions in a structured format.
    """
    prompt = f"""Generate detailed work instructions for this automotive manufacturing task:

Task: {task_description}

Remember:
1. Respond ONLY with a JSON object
2. Include all required sections
3. Be specific and detailed
4. Use proper JSON syntax"""

    # Get response from OpenAI
    response_text = generate_instructions(prompt)
    
    if not response_text:
        return None

    try:
        # Parse JSON response
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {str(e)}")
        return None

def main():
    print("Generating work instructions for automotive manufacturing tasks...")
    print("=" * 80)
    
    # Generate instructions for each task
    all_instructions = []
    for i, task in enumerate(task_descriptions, 1):
        print(f"\nTASK {i:03d}")
        print("-" * 80)
        print(f"Description: {task}")
        print("-" * 80)
        
        instructions = generate_work_instructions(task)
        if instructions:
            instructions['task_id'] = f"TASK-{i:03d}"
            instructions['description'] = task
            all_instructions.append(instructions)
            
            # Print formatted instructions
            print("\nSAFETY PRECAUTIONS:")
            for item in instructions['safety_precautions']:
                print(f"• {item}")
            
            print("\nREQUIRED TOOLS:")
            for item in instructions['required_tools']:
                print(f"• {item}")
            
            print("\nSTEPS:")
            for i, step in enumerate(instructions['steps'], 1):
                print(f"{i}. {step}")
            
            print("\nACCEPTANCE CHECKS:")
            for item in instructions['acceptance_checks']:
                print(f"• {item}")
            
            print("\n" + "=" * 80)
        else:
            print(f"Failed to generate instructions for Task {i}")
            print("\n" + "=" * 80)

    # Save instructions to JSON file
    if all_instructions:
        output_file = 'work_instructions.json'
        print(f"\nSaving instructions to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(all_instructions, f, indent=2)
        print(f"Instructions saved to {output_file}")
    else:
        print("No instructions were generated successfully.")

if __name__ == "__main__":
    main()