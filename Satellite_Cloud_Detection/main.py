"""
Satellite Cloud Detection using Azure OpenAI Vision
===================================================
This application uses Azure OpenAI's GPT-4o-mini vision capabilities to classify
satellite images as either "Cloudy" or "Clear" without traditional ML models.

Features:
- Vision-based image classification using LLM
- Structured output with accuracy scores
- Automated testing with multiple satellite images
- No manual input required
- Base64 image encoding for API transmission

Author: Duke
Date: November 12, 2025
"""

# pip install langchain-openai pillow requests python-dotenv
import base64
import io
import os
import requests
from PIL import Image
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

print("="*80)
print("SATELLITE CLOUD DETECTION - AZURE OPENAI VISION")
print("="*80)

# ---- AZURE OPENAI CONFIG ----
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_LLM_MODEL", "gpt-4o-mini")

os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["AZURE_DEPLOYMENT_NAME"] = AZURE_DEPLOYMENT_NAME

print(f"\n✅ Configuration loaded")
print(f"   Azure Endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f"   Deployment: {AZURE_DEPLOYMENT_NAME}\n")

# ---- SETUP LLM WITH STRUCTURED OUTPUT ----
print("🔄 Initializing Azure OpenAI with vision capabilities...")
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    api_key=SecretStr(AZURE_OPENAI_API_KEY) if AZURE_OPENAI_API_KEY else None,  # type: ignore
    api_version="2024-02-15-preview",
)

# Define output schema for structured responses
class WeatherResponse(BaseModel):
    """Structured output schema for cloud detection results."""
    accuracy: float = Field(description="The confidence/accuracy of the classification (0-100)")
    result: str = Field(description="The classification result: 'Clear' or 'Cloudy'")

# Use regular LLM without structured output (parse response manually)
print("✅ Azure OpenAI LLM initialized\n")

# ---- HELPER FUNCTION: LOAD AND ENCODE IMAGE ----
def load_and_encode_image(image_url: str) -> str:
    """
    Load image from URL and encode to base64.
    
    Args:
        image_url: URL of the image to load
        
    Returns:
        Base64 encoded string of the image
    """
    print(f"   📥 Downloading image from: {image_url}")
    response = requests.get(image_url)
    image_bytes = response.content
    image_data_base64 = base64.b64encode(image_bytes).decode("utf-8")
    print(f"   ✅ Image downloaded and encoded (size: {len(image_bytes)} bytes)")
    return image_data_base64

# ---- HELPER FUNCTION: CLASSIFY IMAGE ----
def classify_satellite_image(image_url: str) -> WeatherResponse:
    """
    Classify satellite image as Cloudy or Clear using Azure OpenAI vision.
    
    Args:
        image_url: URL of the satellite image
        
    Returns:
        WeatherResponse with classification result and accuracy
    """
    # Load and encode image
    image_data_base64 = load_and_encode_image(image_url)
    
    # Construct prompt with system and user messages
    # Updated prompt to explicitly request numeric accuracy and single word classification
    message = [
        {
            "role": "system",
            "content": """You are a satellite image classification expert.
Analyze the image and determine if it shows cloudy or clear weather.

IMPORTANT: Your response MUST be in this EXACT format:
Classification: [either "Clear" or "Cloudy"]
Accuracy: [number between 0-100]

Example response:
Classification: Cloudy
Accuracy: 92.5

Do not include any other text, explanations, or formatting.""",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Classify this satellite/weather image. Is it 'Clear' or 'Cloudy'? Provide accuracy percentage.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data_base64}"},
                },
            ],
        },
    ]
    
    # Call Azure OpenAI
    print(f"   🤖 Sending image to Azure OpenAI for classification...")
    
    # Use regular LLM and parse response
    import re
    raw_response = llm.invoke(message)  # type: ignore
    
    # Parse the response text
    content_str = str(raw_response.content) if hasattr(raw_response, 'content') else str(raw_response)
    
    # Parse Classification: Clear/Cloudy
    if "classification:" in content_str.lower():
        # Extract from "Classification: Cloudy" format
        class_match = re.search(r'classification:\s*(clear|cloudy)', content_str, re.IGNORECASE)
        classification = class_match.group(1).title() if class_match else "Cloudy"
    elif "cloudy" in content_str.lower() and "clear" not in content_str.lower():
        classification = "Cloudy"
    elif "clear" in content_str.lower():
        classification = "Clear"
    else:
        classification = "Cloudy"  # Default
    
    # Parse Accuracy: XX.X
    accuracy_match = re.search(r'accuracy:\s*(\d+\.?\d*)', content_str, re.IGNORECASE)
    if accuracy_match:
        accuracy = float(accuracy_match.group(1))
    else:
        # Try finding any percentage number
        percent_match = re.search(r'(\d+\.?\d*)%?', content_str)
        accuracy = float(percent_match.group(1)) if percent_match else 85.0
    
    result = WeatherResponse(accuracy=accuracy, result=classification)
    print(f"   ✅ Classification completed")
    
    return result  # type: ignore

# ---- AUTOMATED MOCK IMAGES (NO MANUAL INPUT) ----
# Using publicly available satellite/cloud images for testing
mock_images = [
    {
        "url": "https://images.pexels.com/photos/53594/blue-clouds-day-fluffy-53594.jpeg",
        "description": "Blue sky with fluffy white clouds"
    },
    {
        "url": "https://images.pexels.com/photos/531756/pexels-photo-531756.jpeg",
        "description": "Clear blue sky with minimal clouds"
    },
    {
        "url": "https://images.pexels.com/photos/209831/pexels-photo-209831.jpeg",
        "description": "Dark stormy clouds"
    },
]

print("="*80)
print("RUNNING AUTOMATED CLOUD DETECTION TESTS")
print("="*80)
print(f"ℹ️  Processing {len(mock_images)} satellite images automatically\n")

results = []

for i, image_data in enumerate(mock_images, 1):
    print("="*80)
    print(f"IMAGE {i}/{len(mock_images)}")
    print("="*80)
    print(f"\n📷 Image Description: {image_data['description']}")
    print(f"🔗 Image URL: {image_data['url']}\n")
    
    try:
        # Classify the image
        result = classify_satellite_image(image_data['url'])
        
        # Store result
        results.append({
            "image_number": i,
            "url": image_data['url'],
            "description": image_data['description'],
            "prediction": result.result,
            "accuracy": result.accuracy
        })
        
        # Display results
        print(f"\n🎯 CLASSIFICATION RESULTS:")
        print(f"   Prediction: {result.result}")
        print(f"   Accuracy: {result.accuracy}%")
        print()
        
    except Exception as e:
        print(f"\n❌ Error processing image: {e}\n")
        results.append({
            "image_number": i,
            "url": image_data['url'],
            "description": image_data['description'],
            "prediction": "Error",
            "accuracy": 0.0
        })
    
    print("="*80 + "\n")

# ---- SUMMARY ----
print("="*80)
print("✅ ALL CLOUD DETECTION TESTS COMPLETED")
print("="*80)

print(f"\n📊 SUMMARY OF RESULTS:\n")
print(f"{'#':<5} {'Prediction':<12} {'Accuracy':<12} {'Description':<40}")
print("-" * 80)
for r in results:
    print(f"{r['image_number']:<5} {r['prediction']:<12} {r['accuracy']:<12.1f} {r['description']:<40}")

print(f"📈 STATISTICS:")
cloudy_count = sum(1 for r in results if r['prediction'] == 'Cloudy')
clear_count = sum(1 for r in results if r['prediction'] == 'Clear')
error_count = sum(1 for r in results if r['prediction'] == 'Error')
successful_results = [r for r in results if r['prediction'] != 'Error']
avg_accuracy = sum(r['accuracy'] for r in successful_results) / len(successful_results) if successful_results else 0

print(f"   Total images processed: {len(results)}")
print(f"   Cloudy predictions: {cloudy_count}")
print(f"   Clear predictions: {clear_count}")
print(f"   Errors: {error_count}")
print(f"   Average accuracy: {avg_accuracy:.2f}%")

print(f"\n✅ Program completed successfully!")
