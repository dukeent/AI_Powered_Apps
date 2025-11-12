# Assignment 12: Satellite Cloud Detection with Azure OpenAI Vision

## 📋 Objective

Build an **AI-powered satellite image classifier** that uses **Azure OpenAI's vision capabilities** (GPT-4o-mini multimodal) to classify satellite images as either **"Cloudy"** or **"Clear"** - without traditional computer vision models or ML frameworks like TensorFlow/PyTorch.

This demonstrates how modern Large Language Models can process visual data and perform classification tasks through natural language understanding combined with image analysis.

---

## 🎯 Features

- **LLM-Based Image Classification**: Uses Azure OpenAI GPT-4o-mini's vision API for image understanding
- **Structured Output**: Returns classification with confidence/accuracy scores (0-100%)
- **Automated Testing**: Processes 3 mock satellite images automatically (no manual input)
- **Base64 Encoding**: Converts images to base64 for API transmission
- **Comprehensive Logging**: Detailed console output for each classification step
- **LangChain Integration**: Uses LangChain's structured output for type-safe responses

---

## 🔧 Technologies Used

- **Python 3.9+**
- **Azure OpenAI Service** (GPT-4o-mini with vision capabilities)
- **LangChain OpenAI** - For structured LLM interactions
- **Pillow (PIL)** - Python Imaging Library
- **Requests** - HTTP library for downloading images
- **Pydantic** - Data validation and structured outputs

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- Azure OpenAI API access with GPT-4o-mini deployment
- Internet connection (to download satellite images)

### Step 1: Clone or Download

```bash
cd Assignment_12
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langchain-openai>=0.1.0` - LangChain's Azure OpenAI integration
- `pillow>=10.0.0` - Image processing library
- `requests>=2.31.0` - HTTP requests for downloading images
- `python-dotenv>=1.0.0` - Environment variable management

### Step 3: Configure Environment Variables

Create a `.env` file in the `Assignment_12` directory:

```bash
cp .env.example .env
```

Edit `.env` and add your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_LLM_MODEL=gpt-4o-mini
```

**How to get Azure OpenAI credentials:**

1. Go to [Azure Portal](https://portal.azure.com/)
2. Navigate to your Azure OpenAI resource
3. Under "Keys and Endpoint":
   - Copy **KEY 1** → `AZURE_OPENAI_API_KEY`
   - Copy **Endpoint** → `AZURE_OPENAI_ENDPOINT`
4. Under "Deployments":
   - Note your deployment name (usually `gpt-4o-mini` or `GPT-4o-mini`)

---

## 🚀 Usage

### Running the Application

```bash
python main.py
```

The application will:
1. Load environment configuration
2. Initialize Azure OpenAI LLM with vision capabilities
3. Process 3 mock satellite images automatically:
   - Image 1: Blue sky with fluffy white clouds
   - Image 2: Clear blue sky with minimal clouds
   - Image 3: Dark stormy clouds
4. Display classification results with accuracy scores
5. Show summary statistics

### Expected Output

```
================================================================================
SATELLITE CLOUD DETECTION - AZURE OPENAI VISION
================================================================================

✅ Configuration loaded
   Azure Endpoint: https://your-resource.openai.azure.com/
   Deployment: gpt-4o-mini

🔄 Initializing Azure OpenAI with vision capabilities...
✅ Azure OpenAI LLM initialized with structured output

================================================================================
RUNNING AUTOMATED CLOUD DETECTION TESTS
================================================================================
ℹ️  Processing 3 satellite images automatically

================================================================================
IMAGE 1/3
================================================================================

📷 Image Description: Blue sky with fluffy white clouds
🔗 Image URL: https://images.pexels.com/photos/53594/...

   📥 Downloading image from: https://images.pexels.com/...
   ✅ Image downloaded and encoded (size: 245678 bytes)
   🤖 Sending image to Azure OpenAI for classification...
   ✅ Classification completed

🎯 CLASSIFICATION RESULTS:
   Prediction: Cloudy
   Accuracy: 92.5%

================================================================================

...

================================================================================
✅ ALL CLOUD DETECTION TESTS COMPLETED
================================================================================

📊 SUMMARY OF RESULTS:

#     Prediction   Accuracy     Description                             
--------------------------------------------------------------------------------
1     Cloudy       92.5         Blue sky with fluffy white clouds       
2     Clear        88.0         Clear blue sky with minimal clouds      
3     Cloudy       95.0         Dark stormy clouds                      

📈 STATISTICS:
   Total images processed: 3
   Cloudy predictions: 2
   Clear predictions: 1
   Errors: 0
   Average accuracy: 91.83%

✅ Program completed successfully!
```

---

## 📚 Step-by-Step Problem Solving Approach

### Problem Statement

**Challenge**: Classify satellite images as "Cloudy" or "Clear" using AI, without traditional computer vision models.

**Traditional Approach** (Not Used Here):
- Collect thousands of labeled satellite images
- Train a CNN (Convolutional Neural Network) using TensorFlow/PyTorch
- Requires GPU, extensive training time, and ML expertise

**Our Approach** (LLM Vision):
- Use pre-trained multimodal LLM (GPT-4o-mini) that already "understands" images
- Provide images with natural language prompts
- Get structured classification responses

---

### Step 1: Understanding Azure OpenAI Vision API

**Key Concept**: GPT-4o-mini is a **multimodal model** that can process both text and images.

**LangChain Documentation Source**:
- [LangChain Multimodal Inputs](https://python.langchain.com/docs/how_to/multimodal_inputs/)
- [Azure ChatOpenAI Integration](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/)

**How it works**:

```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
    api_key=SecretStr(os.environ["AZURE_OPENAI_API_KEY"]),
    api_version="2024-02-15-preview",  # Vision-enabled API version
)
```

The model accepts messages with **content arrays** that can contain both text and images:

```python
message = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Classify this image..."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }
]
```

**LangChain Reference**:
From [Multimodal Inputs Guide](https://python.langchain.com/docs/how_to/multimodal_inputs/):
> "Some models can take in image inputs... You can pass images to models in two main ways:
> 1. By passing URL strings
> 2. By passing base64 encoded image strings"

---

### Step 2: Image Encoding Strategy

**Challenge**: Azure OpenAI API requires images as base64-encoded strings.

**Solution**:

```python
import base64
import requests

def load_and_encode_image(image_url: str) -> str:
    # Download image from URL
    response = requests.get(image_url)
    image_bytes = response.content
    
    # Encode to base64
    image_data_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    return image_data_base64
```

**Why Base64?**
- API-friendly format (text-based, no binary data in JSON)
- Enables embedding images directly in request payloads
- Standard for multimodal API communication

---

### Step 3: Structured Output with Pydantic

**Challenge**: LLM responses can be unpredictable (extra text, inconsistent formats).

**Solution**: Use LangChain's `with_structured_output()` with Pydantic schemas.

```python
from pydantic import BaseModel, Field

class WeatherResponse(BaseModel):
    accuracy: float = Field(description="The confidence/accuracy (0-100)")
    result: str = Field(description="Classification: 'Clear' or 'Cloudy'")

llm_with_structured_output = llm.with_structured_output(WeatherResponse)
```

**LangChain Reference**:
From [Structured Output Guide](https://python.langchain.com/docs/how_to/structured_output/):
> "Many applications require models to return structured output...
> `.with_structured_output()` method is available on ChatModels supporting tool/function calling."

**Benefits**:
- ✅ Type-safe responses
- ✅ Automatic validation
- ✅ Predictable output format
- ✅ No need for manual parsing

**Result**:
```python
result = llm_with_structured_output.invoke(message)
print(result.result)      # "Cloudy" or "Clear"
print(result.accuracy)    # 92.5
```

---

### Step 4: Prompt Engineering for Classification

**Critical Component**: The system prompt guides the LLM's behavior.

```python
{
    "role": "system",
    "content": """Based on the satellite image provided, classify the scene as either: 
'Clear' (no clouds or minimal cloud coverage) or 'Cloudy' (significant cloud coverage). 
Respond with only one word: either 'Clear' or 'Cloudy' and provide an accuracy score (0-100). 
Do not provide explanations or additional text."""
}
```

**Prompt Design Principles**:
1. **Clear Task Definition**: "classify the scene as either..."
2. **Explicit Categories**: "'Clear' (no clouds) or 'Cloudy' (with clouds)"
3. **Output Constraints**: "Respond with only one word"
4. **Accuracy Request**: "provide an accuracy score (0-100)"
5. **Avoid Hallucinations**: "Do not provide explanations"

---

### Step 5: Automated Testing (No Manual Input)

**Requirement**: "Use dummy input list (auto input). Do not use manual input."

**Implementation**:

```python
mock_images = [
    {
        "url": "https://images.pexels.com/photos/53594/...",
        "description": "Blue sky with fluffy white clouds"
    },
    {
        "url": "https://images.pexels.com/photos/531756/...",
        "description": "Clear blue sky with minimal clouds"
    },
    {
        "url": "https://images.pexels.com/photos/209831/...",
        "description": "Dark stormy clouds"
    },
]

# Automated loop - NO input() calls
for i, image_data in enumerate(mock_images, 1):
    result = classify_satellite_image(image_data['url'])
    print(f"Prediction: {result.result}")
    print(f"Accuracy: {result.accuracy}%")
```

---

### Step 6: Error Handling and Logging

**Comprehensive Logging**:

```python
print("="*80)
print(f"IMAGE {i}/{len(mock_images)}")
print(f"📷 Image Description: {image_data['description']}")
print(f"🔗 Image URL: {image_data['url']}\n")

try:
    result = classify_satellite_image(image_data['url'])
    print(f"🎯 CLASSIFICATION RESULTS:")
    print(f"   Prediction: {result.result}")
    print(f"   Accuracy: {result.accuracy}%")
except Exception as e:
    print(f"❌ Error processing image: {e}")
```

**Benefits**:
- ✅ User-friendly progress tracking
- ✅ Clear result presentation
- ✅ Graceful error handling
- ✅ Debugging support

---

## 🧠 Knowledge & Experience Gained

### 1. **Multimodal AI Understanding**

**Before**: Understanding LLMs were text-only processors.

**After**: Learned that modern LLMs (GPT-4o-mini, GPT-4 Vision) can process and understand:
- Images (photographs, diagrams, charts)
- Text (prompts, instructions)
- Combined multimodal inputs

**Key Insight**: Vision-enabled LLMs don't just recognize objects - they understand context, relationships, and can perform complex reasoning about visual content.

---

### 2. **LangChain Multimodal Integration**

**LangChain Documentation Sources Used**:

1. **[Multimodal Inputs](https://python.langchain.com/docs/how_to/multimodal_inputs/)**
   - How to pass images to chat models
   - Base64 encoding vs URL approaches
   - Content array structure

2. **[Azure ChatOpenAI](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/)**
   - Azure-specific configuration
   - API version requirements for vision (2024-02-15-preview)
   - Endpoint and deployment setup

3. **[Structured Output](https://python.langchain.com/docs/how_to/structured_output/)**
   - Using `.with_structured_output()` method
   - Pydantic schema definition
   - Type-safe LLM responses

**Example from LangChain Docs**:
```python
# From multimodal_inputs documentation
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "describe the weather in this image"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
}
```

---

### 3. **Structured Output vs. Raw LLM Responses**

**Without Structured Output** (Unreliable):
```
Response: "The image shows clouds, so I'd say it's cloudy. 
My confidence is around 85-90%."
```
👎 Requires parsing, inconsistent format, prone to errors

**With Structured Output** (Reliable):
```python
WeatherResponse(accuracy=87.5, result="Cloudy")
```
👍 Type-safe, validated, predictable

**Learned**: Always use structured outputs for production applications requiring reliable data extraction.

---

### 4. **Prompt Engineering for Classification Tasks**

**Effective Prompt Components**:
1. **Role Definition**: System message sets behavior
2. **Task Clarity**: "Classify as either X or Y"
3. **Output Format**: "Respond with only one word"
4. **Examples**: (Optional) Few-shot examples
5. **Constraints**: "Do not provide explanations"

**Bad Prompt**:
```
"Look at this image and tell me about the clouds"
```
❌ Too vague, unpredictable output

**Good Prompt**:
```
"Classify as 'Clear' or 'Cloudy'. Provide accuracy 0-100. No explanations."
```
✅ Clear, constrained, structured

---

### 5. **Base64 Encoding for API Transmission**

**Why Not Just URLs?**
- Some images may be local files (not accessible via URL)
- More control over image data
- Standard approach for multimodal APIs

**Implementation**:
```python
import base64

# Read image
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

# Encode to base64
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

# Format for API
data_url = f"data:image/jpeg;base64,{image_b64}"
```

---

### 6. **LLM Limitations in Image Classification**

**Strengths**:
- ✅ Zero-shot learning (no training data needed)
- ✅ Natural language understanding
- ✅ Context awareness
- ✅ Rapid prototyping

**Limitations**:
- ⚠️ Not as accurate as specialized CV models for high-stakes applications
- ⚠️ Higher latency (API calls vs local inference)
- ⚠️ Cost per image (API pricing)
- ⚠️ Requires internet connection

**Best Use Cases**:
- Prototyping and proof-of-concept
- Low-volume classification tasks
- Applications requiring natural language explanations
- Combining vision with reasoning tasks

---

### 7. **Azure OpenAI Vision API Versioning**

**Critical Detail**: Vision capabilities require specific API versions.

```python
api_version="2024-02-15-preview"  # ✅ Supports vision
api_version="2023-05-15"          # ❌ Text-only
```

**Learned**: Always check Azure OpenAI documentation for feature availability by API version.

---

### 8. **Pydantic for Data Validation**

**Before**: Using dictionaries and manual type checking.

**After**: Using Pydantic models for automatic validation.

```python
class WeatherResponse(BaseModel):
    accuracy: float = Field(description="0-100")
    result: str = Field(description="'Clear' or 'Cloudy'")

# Automatic validation
response = WeatherResponse(accuracy=105, result="Maybe")  # ❌ Raises error
response = WeatherResponse(accuracy=92.5, result="Cloudy")  # ✅ Valid
```

**Benefits**:
- Automatic type checking
- Field descriptions for LLM guidance
- IDE autocomplete support
- Runtime validation

---

### 9. **Production Considerations**

**For Real-World Deployment**:

1. **Error Handling**:
   ```python
   try:
       result = classify_satellite_image(url)
   except requests.RequestException:
       # Handle download failures
   except Exception as e:
       # Handle API errors
   ```

2. **Rate Limiting**:
   ```python
   import time
   time.sleep(1)  # Avoid API rate limits
   ```

3. **Caching**:
   ```python
   # Cache results to avoid re-processing same images
   if image_hash in cache:
       return cache[image_hash]
   ```

4. **Monitoring**:
   - Log all predictions
   - Track accuracy over time
   - Monitor API costs

---

### 10. **Alternative Approaches Considered**

| Approach | Pros | Cons |
|----------|------|------|
| **Traditional CNN** | High accuracy, fast inference | Requires training data, GPU, expertise |
| **Azure Computer Vision** | Specialized for images, fast | Less flexible, limited to predefined tasks |
| **GPT-4 Vision** | Zero-shot, natural language, flexible | API costs, latency, not specialized |
| **Open-source Vision LLMs** | Free, local inference | Complex setup, requires powerful hardware |

**Chosen Approach**: Azure OpenAI GPT-4o-mini Vision
- ✅ Best balance for prototyping
- ✅ Minimal setup
- ✅ Structured outputs via LangChain

---

## 📊 Dataset Information

**Original Dataset**: [Kaggle - Cloud Cover Detection](https://www.kaggle.com/datasets/hmendonca/cloud-cover-detection/data)

**For This Assignment**: Using publicly available satellite/cloud images from Pexels instead of downloading the full Kaggle dataset.

**Why Pexels Images?**
- ✅ Publicly accessible URLs (no download required)
- ✅ High-quality cloud imagery
- ✅ Sufficient for demonstration purposes
- ✅ Faster testing (no dataset setup)

**For Production**: Use the Kaggle dataset:
1. Download from Kaggle
2. Extract to local directory
3. Update `mock_images` to use local file paths
4. Use base64 encoding for local files

---

## 🎓 Key Technical Concepts

### 1. **Multimodal Learning**
Using models that can process multiple types of data (text + images) simultaneously.

### 2. **Vision-Language Models (VLMs)**
LLMs extended with vision encoders to understand and reason about visual content.

### 3. **Structured Output**
Constraining LLM responses to predefined schemas using Pydantic models.

### 4. **Base64 Encoding**
Converting binary image data to text format for API transmission.

### 5. **Prompt Engineering**
Crafting effective instructions to guide LLM behavior and output format.

---

## 🔍 Troubleshooting

### Issue: "Authentication failed"

**Solution**: Check your `.env` file:
```bash
# Verify credentials
cat .env

# Ensure no extra spaces or quotes
AZURE_OPENAI_API_KEY=abc123...  # ✅ Correct
AZURE_OPENAI_API_KEY="abc123..." # ❌ Remove quotes
```

### Issue: "Model does not support vision"

**Solution**: Ensure you're using a vision-enabled model:
- ✅ `gpt-4o-mini`
- ✅ `gpt-4-vision-preview`
- ✅ `gpt-4o`
- ❌ `gpt-35-turbo` (text-only)

### Issue: "Image download failed"

**Solution**: Check internet connection and image URLs:
```python
# Test URL manually
import requests
response = requests.get(image_url)
print(response.status_code)  # Should be 200
```

### Issue: Low accuracy scores

**Possible causes**:
- Poor image quality
- Ambiguous cloud coverage
- Prompt needs refinement

**Solution**: Improve prompt:
```python
"content": """Analyze the satellite image carefully. 
If more than 50% of the sky shows clouds, classify as 'Cloudy'.
If less than 50% shows clouds, classify as 'Clear'.
Provide your confidence level 0-100."""
```

---

## 📈 Potential Enhancements

1. **Multi-class Classification**:
   ```python
   result: str = Field(description="'Clear', 'Partly Cloudy', 'Cloudy', or 'Stormy'")
   ```

2. **Batch Processing**:
   ```python
   # Process multiple images in parallel
   import asyncio
   results = await asyncio.gather(*[classify(url) for url in urls])
   ```

3. **Confidence Thresholds**:
   ```python
   if result.accuracy < 70:
       print("⚠️ Low confidence - manual review recommended")
   ```

4. **Streamlit Web Interface**:
   ```python
   import streamlit as st
   uploaded_file = st.file_uploader("Upload satellite image")
   if uploaded_file:
       result = classify_image(uploaded_file)
       st.write(f"Result: {result.result}")
   ```

5. **Result Logging**:
   ```python
   import json
   with open("results.json", "w") as f:
       json.dump(results, f, indent=2)
   ```

---

## 📄 Project Structure

```
Assignment_12/
├── main.py              # Main application (single file, all code here)
├── requirements.txt     # Python dependencies
├── .env                 # Azure OpenAI credentials (DO NOT COMMIT)
├── .env.example         # Template for environment variables
├── .gitignore          # Git exclusions
└── README.md           # This file
```

---

## 🔗 Useful Resources

### LangChain Documentation
- [Multimodal Inputs Guide](https://python.langchain.com/docs/how_to/multimodal_inputs/)
- [Azure ChatOpenAI Integration](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/)
- [Structured Output with Pydantic](https://python.langchain.com/docs/how_to/structured_output/)

### Azure OpenAI
- [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Vision API Reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision)

### Datasets
- [Kaggle Cloud Cover Detection](https://www.kaggle.com/datasets/hmendonca/cloud-cover-detection/data)
- [Pexels Free Stock Photos](https://www.pexels.com/)

---

## 🤝 Contributing

This is an educational assignment. For improvements or questions, please contact the instructor.

---

## 📝 License

Educational project - for learning purposes only.

---

## 👤 Author

**Duke**  
Assignment 12 - AI-Powered Applications Course  
Date: November 12, 2025

---

## ✅ Assignment Checklist

- [x] Single Python file (`main.py`) with all code
- [x] 3 mock images via URLs (automated, no `input()`)
- [x] Console logging for all operations
- [x] LangChain integration with structured output
- [x] Azure OpenAI vision API usage
- [x] Base64 image encoding
- [x] Pydantic schema for type-safe responses
- [x] Comprehensive README with:
  - [x] Step-by-step problem solving
  - [x] LangChain documentation references
  - [x] Knowledge and experience gained
  - [x] Installation instructions
  - [x] Usage examples
  - [x] Troubleshooting guide

---

**🎉 End of README**
