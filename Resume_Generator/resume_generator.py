"""
AI-Powered Resume Generator using LLaMA 3 (Local Deployment)

This script demonstrates:
- LLaMA C++ Python bindings for local AI execution
- Object-oriented design with exception handling
- Professional resume generation from user data
- Batch processing with multiple samples

Author: Duc Nguyen
Date: October 28, 2025
"""

from llama_cpp import Llama
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from huggingface_hub import hf_hub_download

class LlamaModel:
    """
    A wrapper class for LLaMA model with exception handling and text generation capabilities.
    """
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0):
        """
        Initialize the LLaMA model.
        
        Args:
            model_path (str): Path to the LLaMA model file (.gguf)
            n_ctx (int): Context window size (default: 2048)
            n_gpu_layers (int): Number of layers to offload to GPU (default: 0 for CPU)
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to load
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            self.model_path = model_path

            print(f"✅ LLaMA model loaded successfully from {model_path}")
            print(f"   Context size: {n_ctx} tokens")
        except Exception as e:
            raise RuntimeError(f"Failed to load LLaMA model: {e}")
    
    def generate_text(self, prompt: str, max_tokens: int = 500, 
                     temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text using the loaded LLaMA model.
        
        Args:
            prompt (str): Input prompt for text generation
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (0.0-1.0)
            top_p (float): Nucleus sampling parameter
        
        Returns:
            str: Generated text
        
        Raises:
            RuntimeError: If text generation fails
        """
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["\n\n\n", "</s>"],
                stream=False
            )
            return output['choices'][0]['text'].strip()  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to generate text: {e}")
    
    def __repr__(self):
        return f"LlamaModel(model_path='{self.model_path}')"


class ResumeGenerator:
    """
    AI-powered resume generator using LLaMA model.
    """
    
    def __init__(self, llama_model: LlamaModel):
        """
        Initialize the resume generator.
        
        Args:
            llama_model (LlamaModel): Initialized LLaMA model instance
        """
        self.model = llama_model
        self.generation_count = 0
    
    def create_resume_prompt(self, user_data: Dict) -> str:
        """
        Create a structured prompt for resume generation.
        
        Args:
            user_data (Dict): Dictionary containing user information
        
        Returns:
            str: Formatted prompt for LLaMA model
        """
        prompt = f"""
            [INST] 
            You are a professional resume writer. Create a well-structured, professional resume based on the following information:
                Name: {user_data.get('name', 'N/A')}
                Email: {user_data.get('email', 'N/A')}
                Phone: {user_data.get('phone', 'N/A')}
                Job Title: {user_data.get('job_title', 'N/A')}
                Years of Experience: {user_data.get('years_experience', 'N/A')}

                Skills: {', '.join(user_data.get('skills', []))}

                Work Experience:
                {self._format_experience(user_data.get('experience', []))}

                Education:
                {self._format_education(user_data.get('education', []))}

                Professional Summary: {user_data.get('summary', 'N/A')}

                Create a professional resume with the following sections:
                1. Contact Information
                2. Professional Summary
                3. Skills
                4. Work Experience
                5. Education

                Format it professionally and make it ATS-friendly. 
            [/INST]
            """
        return prompt
    
    def _format_experience(self, experiences: List[Dict]) -> str:
        """Format work experience entries."""
        if not experiences:
            return "No work experience provided."
        
        formatted = []
        for exp in experiences:
            formatted.append(
                f"- {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')} "
                f"({exp.get('duration', 'N/A')}): {exp.get('description', 'N/A')}"
            )
        return "\n".join(formatted)
    
    def _format_education(self, education: List[Dict]) -> str:
        """Format education entries."""
        if not education:
            return "No education provided."
        
        formatted = []
        for edu in education:
            formatted.append(
                f"- {edu.get('degree', 'N/A')} in {edu.get('field', 'N/A')} "
                f"from {edu.get('institution', 'N/A')} ({edu.get('year', 'N/A')})"
            )
        return "\n".join(formatted)
    
    def generate_resume(self, user_data: Dict, max_tokens: int = 800) -> Dict:
        """
        Generate a professional resume.
        
        Args:
            user_data (Dict): User information dictionary
            max_tokens (int): Maximum tokens for generation
        
        Returns:
            Dict: Resume generation result with metadata
        """
        try:
            print(f"\n{'='*60}")
            print(f"Generating resume for: {user_data.get('name', 'Unknown')}")
            print(f"{'='*60}")
            
            # Create prompt
            prompt = self.create_resume_prompt(user_data)
            
            # Generate resume
            resume_text = self.model.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            self.generation_count += 1
            
            result = {
                'success': True,
                'user_data': user_data,
                'resume': resume_text,
                'timestamp': datetime.now().isoformat(),
                'generation_id': self.generation_count
            }
            
            print(f"\n✅ Resume generated successfully!\n")
            return result
            
        except Exception as e:
            print(f"\n❌ Error generating resume: {e}\n")
            return {
                'success': False,
                'user_data': user_data,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def display_resume(self, result: Dict):
        """
        Display generated resume in a formatted way.
        
        Args:
            result (Dict): Resume generation result
        """
        if result['success']:
            print(f"\n{'='*60}")
            print(f"RESUME #{result.get('generation_id', 'N/A')}")
            print(f"Generated at: {result.get('timestamp', 'N/A')}")
            print(f"{'='*60}\n")
            print(result['resume'])
            print(f"\n{'='*60}\n")
        else:
            print(f"\n❌ Failed to generate resume")
            print(f"Error: {result.get('error', 'Unknown error')}\n")


def main():
    """Main execution function."""
    
    # Sample user data (3 samples as required)
    sample_users = [
        {
            'name': 'Alex Johnson',
            'email': 'alex.johnson@email.com',
            'phone': '+1 (555) 123-4567',
            'job_title': 'Senior Software Engineer',
            'years_experience': 7,
            'skills': ['Python', 'JavaScript', 'React', 'Node.js', 'Docker', 'AWS', 'MongoDB', 'Git'],
            'experience': [
                {
                    'title': 'Senior Software Engineer',
                    'company': 'TechCorp Inc.',
                    'duration': '2020-Present',
                    'description': 'Led development of microservices architecture, improved system performance by 40%'
                },
                {
                    'title': 'Software Engineer',
                    'company': 'StartupXYZ',
                    'duration': '2017-2020',
                    'description': 'Developed full-stack web applications, mentored junior developers'
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
            'summary': 'Passionate software engineer with 7 years of experience building scalable web applications and leading development teams.'
        },
        {
            'name': 'Sarah Chen',
            'email': 'sarah.chen@email.com',
            'phone': '+1 (555) 987-6543',
            'job_title': 'Data Scientist',
            'years_experience': 5,
            'skills': ['Python', 'R', 'Machine Learning', 'TensorFlow', 'SQL', 'Tableau', 'Statistics', 'Deep Learning'],
            'experience': [
                {
                    'title': 'Data Scientist',
                    'company': 'DataDrive Analytics',
                    'duration': '2021-Present',
                    'description': 'Built predictive models for customer churn, increased retention by 25%'
                },
                {
                    'title': 'Junior Data Analyst',
                    'company': 'Marketing Insights Co.',
                    'duration': '2019-2021',
                    'description': 'Analyzed customer data and created dashboards for business intelligence'
                }
            ],
            'education': [
                {
                    'degree': 'Master of Science',
                    'field': 'Data Science',
                    'institution': 'Tech University',
                    'year': '2019'
                }
            ],
            'summary': 'Data scientist specializing in machine learning and predictive analytics with proven track record of driving business value.'
        },
        {
            'name': 'Michael Rodriguez',
            'email': 'michael.r@email.com',
            'phone': '+1 (555) 456-7890',
            'job_title': 'Senior Project Manager',
            'years_experience': 10,
            'skills': ['Agile', 'Scrum', 'JIRA', 'Risk Management', 'Stakeholder Management', 'Budget Planning', 'Team Leadership'],
            'experience': [
                {
                    'title': 'Senior Project Manager',
                    'company': 'Global Solutions Ltd.',
                    'duration': '2018-Present',
                    'description': 'Managed cross-functional teams of 20+ members, delivered $5M+ projects on time and under budget'
                },
                {
                    'title': 'Project Manager',
                    'company': 'Enterprise Systems Inc.',
                    'duration': '2014-2018',
                    'description': 'Led digital transformation initiatives, improved project delivery efficiency by 30%'
                }
            ],
            'education': [
                {
                    'degree': 'MBA',
                    'field': 'Project Management',
                    'institution': 'Business School',
                    'year': '2014'
                },
                {
                    'degree': 'PMP Certification',
                    'field': 'Project Management Professional',
                    'institution': 'PMI',
                    'year': '2015'
                }
            ],
            'summary': 'Certified PMP with 10 years of experience leading complex projects and driving organizational change.'
        }
    ]
    
    # Model configuration - use local model file
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "llama-2-7b-chat.Q4_K_M.gguf")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Error: Model file not found at {MODEL_PATH}")
        print("Please ensure llama-2-7b-chat.Q4_K_M.gguf is in the Assignment_05 folder.")
        return
    
    try:
        # Initialize LLaMA model
        print("\nLoading LLaMA model...")
        print("=" * 60)
        llama_model = LlamaModel(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_gpu_layers=0  # Use CPU (set higher for GPU)
        )
        print("=" * 60)
        
        # Initialize resume generator
        resume_generator = ResumeGenerator(llama_model)
        print("\n✅ Resume Generator initialized successfully!")
        
        # Store all results
        all_results = []
        
        print("\n" + "="*60)
        print("STARTING BATCH RESUME GENERATION")
        print("="*60)
        
        # Generate resumes for all sample users
        for i, user_data in enumerate(sample_users, 1):
            print(f"\n[{i}/3] Processing: {user_data['name']}...")
            
            try:
                result = resume_generator.generate_resume(user_data, max_tokens=800)
                all_results.append(result)
                
            except Exception as e:
                print(f"❌ Failed to generate resume for {user_data['name']}: {e}")
                all_results.append({
                    'success': False,
                    'user_data': user_data,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        print("\n" + "="*60)
        print("BATCH GENERATION COMPLETE")
        print("="*60)
        
        # Display all resumes
        for i, result in enumerate(all_results, 1):
            print(f"\n\n{'#'*60}")
            print(f"RESUME OUTPUT #{i}")
            print(f"{'#'*60}\n")
            resume_generator.display_resume(result)
        
        # Create output directory for resumes
        output_dir = "generated_resumes"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each resume to individual text file
        for i, result in enumerate(all_results, 1):
            if result['success']:
                name = result['user_data']['name'].replace(' ', '_')
                resume_filename = os.path.join(output_dir, f"resume_{i}_{name}.txt")
                with open(resume_filename, 'w', encoding='utf-8') as f:
                    f.write(result['resume'])
                print(f"💾 Resume #{i} saved to: {resume_filename}")
        
        # Save complete results to JSON file
        json_filename = os.path.join(output_dir, f"all_resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Complete results saved to: {json_filename}")
        print(f"\n📊 Generation Statistics:")
        print(f"   Total resumes generated: {resume_generator.generation_count}")
        print(f"   Successful: {sum(1 for r in all_results if r['success'])}")
        print(f"   Failed: {sum(1 for r in all_results if not r['success'])}")
        print(f"   Output directory: {output_dir}/")

        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure the model file exists at the specified path.")
        print("Download it using:")
        print("wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf")
    except RuntimeError as e:
        print(f"\n❌ Error during model operation: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
