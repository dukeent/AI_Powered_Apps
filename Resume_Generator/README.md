# Assignment 05: AI-Powered Resume Generator using LLaMA 3

## Overview
This project demonstrates an AI-powered resume generation system using LLaMA 3 (LLaMA 2 7B Chat model) running locally. The implementation ensures complete data privacy, offline capability, and professional resume output.

## Features
- ✅ **Local LLaMA 3 Deployment**: Run AI models locally without cloud dependencies
- ✅ **Object-Oriented Architecture**: Clean, maintainable code structure
- ✅ **Professional Resume Generation**: ATS-friendly, well-structured resumes
- ✅ **Exception Handling**: Robust error management throughout the pipeline
- ✅ **Prompt Engineering**: Optimized prompts for quality outputs
- ✅ **Batch Processing**: Generate multiple resumes efficiently
- ✅ **Privacy-First**: All data stays on your local machine

## Requirements

### System Requirements
- Python 3.9+
- 8GB+ RAM (16GB recommended)
- ~4GB disk space for model
- CPU or GPU (CUDA support optional)

### Dependencies
```
llama-cpp-python>=0.2.0
huggingface-hub>=0.20.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Installation

### 1. Install llama-cpp-python
```bash
pip install llama-cpp-python
```

For GPU acceleration (optional):
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

### 2. Download LLaMA Model
Download the quantized LLaMA 2 7B Chat model and place it in the Assignment_05 folder:
```bash
cd Assignment_05
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

Or download manually from [HuggingFace](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) and place in the Assignment_05 directory.

## Usage

### Run the Resume Generator
```bash
python resume_generator.py
```

The script will:
1. Load the LLaMA model from the local file
2. Generate resumes for 3 sample users
3. Save individual resumes to `generated_resumes/` folder
4. Save complete results to JSON file

## Project Structure

```
Assignment_05/
├── resume_generator.py          # Main Python script
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── llama-2-7b-chat.Q4_K_M.gguf # LLaMA model (download separately)
└── generated_resumes/           # Output folder
    ├── resume_1_Alex_Johnson.txt
    ├── resume_2_Sarah_Chen.txt
    ├── resume_3_Michael_Rodriguez.txt
    └── all_resumes_*.json       # Complete results
```

## How It Works

### Architecture

The solution uses an object-oriented approach with two main classes:

#### 1. LlamaModel Class
Wraps the LLaMA C++ Python bindings with:
- Model initialization with proper error handling
- Text generation with configurable parameters
- Exception handling for all model operations

#### 2. ResumeGenerator Class
Specialized resume generation with:
- Prompt engineering for professional resumes
- User data formatting and structuring
- Batch processing capabilities
- Output display and export functions

### Sample User Data Structure

```python
{
    'name': 'Alex Johnson',
    'email': 'alex.johnson@email.com',
    'phone': '+1 (555) 123-4567',
    'job_title': 'Senior Software Engineer',
    'years_experience': 7,
    'skills': ['Python', 'JavaScript', 'React', ...],
    'experience': [
        {
            'title': 'Senior Software Engineer',
            'company': 'TechCorp Inc.',
            'duration': '2020-Present',
            'description': '...'
        }
    ],
    'education': [
        {
            'degree': 'Bachelor of Science',
            'field': 'Computer Science',
            'institution': 'State University',
            'year': '2017'
        }
    ],
    'summary': 'Professional summary...'
}
```

## Sample Outputs

The notebook includes 3 pre-configured sample users:

1. **Alex Johnson** - Senior Software Engineer
   - 7 years experience
   - Skills: Python, JavaScript, React, AWS, Docker
   
2. **Sarah Chen** - Data Scientist
   - 5 years experience
   - Skills: Python, Machine Learning, TensorFlow, SQL
   
3. **Michael Rodriguez** - Senior Project Manager
   - 10 years experience
   - Skills: Agile, Scrum, Stakeholder Management, PMP

Each generates a complete, professional resume with:
- Contact Information
- Professional Summary
- Skills Section
- Work Experience (detailed)
- Education

## Design Decisions

### 1. Local Model Choice
**Decision**: Use LLaMA 2 7B Chat (Q4_K_M quantized version)

**Rationale**:
- Good balance between quality and resource requirements
- Quantized version (4-bit) reduces memory footprint
- Chat-tuned for instruction following
- Can run on CPU without GPU

### 2. Prompt Engineering
**Decision**: Use structured instruction format with [INST] tags

**Rationale**:
- LLaMA 2 Chat is trained with this format
- Clear separation of instructions and data
- Consistent, predictable outputs
- Easy to maintain and modify

### 3. Object-Oriented Architecture
**Decision**: Separate LlamaModel and ResumeGenerator classes

**Rationale**:
- **Separation of Concerns**: Model operations vs. business logic
- **Reusability**: LlamaModel can be used for other tasks
- **Testability**: Each class can be tested independently
- **Maintainability**: Easy to extend or modify

### 4. Exception Handling Strategy
**Decision**: Multi-level exception handling

**Rationale**:
```
Model Level:
├── FileNotFoundError: Missing model file
├── RuntimeError: Model loading/generation failures
└── Generic Exception: Unexpected errors

Application Level:
├── Try-except in generate_resume()
└── Graceful degradation (continue with other users)
```

## Challenges and Solutions

### Challenge 1: Model Size and Performance
**Problem**: Full LLaMA models are too large for typical machines.

**Solution**:
- Use quantized Q4_K_M version (reduces size by ~75%)
- Adjustable context window (n_ctx parameter)
- CPU-friendly configuration with GPU acceleration option
- Efficient token generation with stop sequences

### Challenge 2: Prompt Quality
**Problem**: Generic prompts produce inconsistent resume formats.

**Solution**:
- Structured prompt template with clear sections
- Include all user data in organized format
- Specify exact output format requirements
- Use instruction-tuned model (Chat variant)

### Challenge 3: Data Privacy
**Problem**: Cloud-based APIs expose sensitive user information.

**Solution**:
- 100% local execution
- No external API calls
- Data never leaves the machine
- Offline capability after model download

### Challenge 4: Error Resilience
**Problem**: Model failures could crash entire batch processing.

**Solution**:
- Try-except blocks at multiple levels
- Graceful error handling for each user
- Continue processing on individual failures
- Detailed error messages and logging

## Performance Optimization

### Tips for Faster Generation:

1. **Use GPU if available**:
   ```python
   llama_model = LlamaModel(
       model_path=MODEL_PATH,
       n_gpu_layers=35  # Offload layers to GPU
   )
   ```

2. **Reduce context window**:
   ```python
   n_ctx=1024  # Instead of 2048
   ```

3. **Adjust generation parameters**:
   ```python
   max_tokens=500  # Shorter responses
   temperature=0.5  # More deterministic
   ```

## Extending the System

### Add More Resume Sections
```python
def create_resume_prompt(self, user_data: Dict) -> str:
    # Add certifications, projects, languages, etc.
    certifications = user_data.get('certifications', [])
    # Include in prompt...
```

### Different Resume Styles
```python
def generate_creative_resume(self, user_data: Dict):
    # Modify prompt for creative industries
    prompt = "Create a creative, modern resume..."
```

### Export to Different Formats
```python
def export_to_pdf(self, result: Dict):
    # Use ReportLab or similar to create PDF
    pass

def export_to_docx(self, result: Dict):
    # Use python-docx to create Word document
    pass
```

## Best Practices Demonstrated

1. ✅ **Type Hints**: All functions use proper type annotations
2. ✅ **Documentation**: Comprehensive docstrings for all classes/methods
3. ✅ **Error Handling**: Robust exception management
4. ✅ **Code Organization**: Clean separation of concerns
5. ✅ **Privacy First**: No external dependencies for core functionality
6. ✅ **Resource Management**: Efficient model usage
7. ✅ **Scalability**: Batch processing support
8. ✅ **Maintainability**: Clear, readable code structure

## Troubleshooting

### Model Loading Issues
```bash
# If model fails to load, try:
pip install --upgrade llama-cpp-python

# For Mac M1/M2:
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

### Memory Issues
```python
# Reduce context size
n_ctx=512  # Minimum viable

# Use smaller model (if needed)
# Download llama-2-7b-chat.Q2_K.gguf instead
```

### Slow Generation
- First run is always slower (model loading)
- Subsequent generations are faster
- Use GPU acceleration if available
- Reduce max_tokens parameter

## License
Educational project for Elevate AI Workshop

## Author
Created as part of Assignment 05 - AI-Powered Resume Generator Exercise

## Submission Checklist

✅ Complete Python script with all code  
✅ 3 sample user inputs provided  
✅ Professional resumes generated for all samples  
✅ Object-oriented design (LlamaModel + ResumeGenerator classes)  
✅ Exception handling throughout  
✅ No manual input() calls (automated with sample data)  
✅ Individual resume files saved to generated_resumes/ folder  
✅ Complete results saved to JSON file  
✅ Comprehensive README with instructions  
✅ Local LLaMA deployment with local model file  
✅ Prompt engineering best practices applied  
✅ requirements.txt included
