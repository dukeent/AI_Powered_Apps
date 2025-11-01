"""
Meeting Transcript Summarizer using Azure OpenAI

This script automatically processes meeting transcripts and generates summaries using Azure OpenAI.
It reads from the transcripts directory and generates summaries using Azure OpenAI GPT models.

Author: DucNT104
Date: October 19, 2025
"""

import os
import sys
import glob
from datetime import datetime
from typing import Dict, Optional, Any
from dotenv import load_dotenv
from openai import AzureOpenAI

class MeetingSummarizer:
    """A class that handles meeting transcript summarization using Azure OpenAI."""
    
    def __init__(self) -> None:
        """Initialize the MeetingSummarizer with Azure OpenAI configuration."""
        self.client: Optional[AzureOpenAI] = None
        self.model_name: Optional[str] = None
        if not self.setup_openai():
            sys.exit(1)

    def setup_openai(self) -> bool:
        """Set up the Azure OpenAI client with credentials from environment variables."""
        try:
            if not load_dotenv():
                raise ValueError("Failed to load .env file")
            
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            
            if not all([api_key, azure_endpoint, model_name]):
                raise ValueError("Missing required Azure OpenAI credentials in .env file")
            
            # Cast the environment variables as they are guaranteed to be strings at this point
            self.client = AzureOpenAI(
                api_key=str(api_key),
                api_version="2024-02-15-preview",
                azure_endpoint=str(azure_endpoint)
            )
            self.model_name = str(model_name)
            return True
            
        except Exception as e:
            print(f"Error setting up Azure OpenAI client: {str(e)}")
            return False

    def read_transcript(self, file_path: str) -> Optional[str]:
        """Read and return the content of a transcript file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading transcript {file_path}: {str(e)}")
            return None

    def generate_summary(self, transcript: str) -> Optional[str]:
        """Generate a summary of the transcript using Azure OpenAI."""
        if not self.client or not self.model_name:
            print("Azure OpenAI client not properly initialized")
            return None

        try:
            system_message = """You are a professional meeting summarizer. Your task is to analyze meeting 
            transcripts and provide clear, concise summaries that capture the key points, decisions, and action 
            items. Focus on extracting the most important information while maintaining accuracy."""
            
            user_message = f"""Please provide a summary of the following meeting transcript, including:
            1. Main topics discussed
            2. Key decisions made
            3. Action items and assignments
            4. Any important deadlines mentioned

            Transcript:
            {transcript}"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            if (response.choices and 
                response.choices[0].message and 
                response.choices[0].message.content):
                return response.choices[0].message.content.strip()
            return None
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return None

    def process_transcripts(self) -> Dict[str, str]:
        """Process all transcript files in the transcripts directory."""
        results = {}
        transcript_files = glob.glob("transcripts/*.txt")
        
        if not transcript_files:
            print("No transcript files found in the 'transcripts' directory.")
            return results
        
        for file_path in transcript_files:
            print(f"Processing {file_path}...")
            transcript = self.read_transcript(file_path)
            
            if transcript:
                summary = self.generate_summary(transcript)
                if summary:
                    results[file_path] = summary
                    print(f"\nSummary for: {file_path}")
                    print("=" * 80)
                    print(summary)
                    print("=" * 80 + "\n")
                else:
                    print(f"Failed to generate summary for {file_path}")
            else:
                print(f"Failed to read transcript from {file_path}")
        
        return results

    def save_results(self, results: Dict[str, str]) -> bool:
        """Save the generated summaries to a results directory."""
        try:
            os.makedirs("results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/summaries_{timestamp}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for file_path, summary in results.items():
                    f.write(f"Summary for: {file_path}\n")
                    f.write("=" * 80 + "\n")
                    f.write(summary + "\n\n")
            
            print(f"Results saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False


def main() -> None:
    """Run the meeting summarizer application."""
    try:
        summarizer = MeetingSummarizer()
        results = summarizer.process_transcripts()
        
        if results:
            if summarizer.save_results(results):
                print("Processing completed successfully!")
            else:
                print("Failed to save results.")
        else:
            print("No summaries were generated.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()