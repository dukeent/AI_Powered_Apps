"""
Text-to-Speech (TTS) Application using Hugging Face Models
============================================================

This application demonstrates:
1. Cloning a pre-trained TTS model from Hugging Face Hub
2. Performing inference to convert text to speech
3. Saving and playing the generated audio

Model: facebook/mms-tts-vie (Vietnamese TTS)
Author: Duke Nguyen
Date: November 3, 2025
"""

import os
import torch
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
from datetime import datetime
import argparse


class TextToSpeechGenerator:
    """
    Text-to-Speech Generator using Hugging Face pre-trained models
    
    This class handles:
    - Model loading and initialization
    - Text tokenization
    - Audio waveform generation
    - Audio file saving
    """
    
    def __init__(self, model_name="facebook/mms-tts-vie"):
        """
        Initialize the TTS Generator
        
        Args:
            model_name (str): Hugging Face model identifier
        
        Challenges faced:
        - Model download can be slow on first run (solved by caching)
        - Different models have different tokenizer formats
        - Audio sample rates vary between models
        """
        print(f"🔄 Loading TTS model: {model_name}")
        print("   This may take a moment on first run...")
        
        try:
            # Step 1: Load the pre-trained TTS model from Hugging Face
            # The model is downloaded and cached locally
            self.model = VitsModel.from_pretrained(model_name)
            
            # Step 2: Load the corresponding tokenizer
            # Tokenizer converts text into tokens that the model can understand
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Store model configuration
            self.model_name = model_name
            self.sampling_rate = self.model.config.sampling_rate
            
            print(f"✅ Model loaded successfully!")
            print(f"   Sampling rate: {self.sampling_rate} Hz")
            print(f"   Model type: {type(self.model).__name__}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def generate_speech(self, text, output_path="output.wav"):
        """
        Generate speech from input text
        
        Args:
            text (str): Input text to convert to speech
            output_path (str): Path to save the generated audio file
        
        Returns:
            tuple: (waveform, sampling_rate, output_path)
        
        Steps:
        1. Tokenize the input text
        2. Generate audio waveform using the model
        3. Save the audio to a file
        """
        print(f"\n🎤 Generating speech from text...")
        print(f"   Input text: '{text}'")
        
        try:
            # Step 3: Tokenize the input text
            # This converts the text into a format the model can process
            inputs = self.tokenizer(text, return_tensors="pt")
            print(f"   Tokenized input shape: {inputs['input_ids'].shape}")
            
            # Step 4: Perform inference to generate the waveform
            # torch.no_grad() is used to disable gradient computation (faster inference)
            print("   Generating audio waveform...")
            with torch.no_grad():
                output = self.model(**inputs).waveform
            
            # Step 5: Extract the waveform data
            # The output is a tensor, we need to convert it to numpy array
            waveform = output.squeeze().cpu().numpy()
            
            print(f"   Generated waveform shape: {waveform.shape}")
            print(f"   Audio duration: {len(waveform) / self.sampling_rate:.2f} seconds")
            
            # Step 6: Save the audio to a file
            self._save_audio(waveform, output_path)
            
            return waveform, self.sampling_rate, output_path
            
        except Exception as e:
            print(f"❌ Error generating speech: {str(e)}")
            raise
    
    def _save_audio(self, waveform, output_path):
        """
        Save the generated audio waveform to a file
        
        Args:
            waveform: Audio waveform data
            output_path: Path to save the file
        
        Challenges faced:
        - Need to ensure output directory exists
        - Handle different audio formats (WAV is most compatible)
        - Proper sample rate is crucial for playback quality
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            
            # Save as WAV file using soundfile
            # WAV format is uncompressed and widely supported
            sf.write(output_path, waveform, self.sampling_rate)
            
            file_size = os.path.getsize(output_path)
            print(f"✅ Audio saved successfully!")
            print(f"   Output file: {output_path}")
            print(f"   File size: {file_size / 1024:.2f} KB")
            
        except Exception as e:
            print(f"❌ Error saving audio: {str(e)}")
            raise
    
    def generate_multiple(self, texts, output_dir="outputs"):
        """
        Generate speech for multiple text inputs
        
        Args:
            texts (list): List of text strings
            output_dir (str): Directory to save output files
        
        Returns:
            list: List of output file paths
        """
        print(f"\n🎵 Generating speech for {len(texts)} texts...")
        
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        for i, text in enumerate(texts, 1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"tts_output_{i}_{timestamp}.wav")
            
            print(f"\n--- Processing {i}/{len(texts)} ---")
            waveform, rate, path = self.generate_speech(text, output_path)
            output_files.append(path)
        
        print(f"\n✅ All files generated successfully!")
        print(f"   Output directory: {output_dir}")
        
        return output_files


def main():
    """
    Main function to demonstrate TTS functionality
    
    This demonstrates:
    1. Loading a pre-trained TTS model
    2. Converting text to speech
    3. Saving the audio output
    """
    parser = argparse.ArgumentParser(description="Text-to-Speech Generator using Hugging Face")
    parser.add_argument("--text", type=str, 
                       default="Xin chào anh em đến với bài tập của khoá AI Application Engineer",
                       help="Text to convert to speech")
    parser.add_argument("--model", type=str, 
                       default="facebook/mms-tts-vie",
                       help="Hugging Face model name")
    parser.add_argument("--output", type=str, 
                       default="output.wav",
                       help="Output audio file path")
    parser.add_argument("--batch", action="store_true",
                       help="Generate multiple samples")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎙️  Text-to-Speech Generator")
    print("=" * 60)
    
    try:
        # Initialize the TTS generator
        tts = TextToSpeechGenerator(model_name=args.model)
        
        if args.batch:
            # Generate multiple samples
            sample_texts = [
                "Xin chào anh em đến với bài tập của khoá AI Application Engineer",
                "Công nghệ trí tuệ nhân tạo đang phát triển rất nhanh",
                "Chúc các bạn học tập tốt và thành công"
            ]
            output_files = tts.generate_multiple(sample_texts)
            
            print("\n📊 Summary:")
            for i, file in enumerate(output_files, 1):
                print(f"   {i}. {file}")
        else:
            # Generate single audio
            waveform, rate, output_path = tts.generate_speech(args.text, args.output)
            
            print("\n📊 Summary:")
            print(f"   Model: {args.model}")
            print(f"   Input text: '{args.text}'")
            print(f"   Output file: {output_path}")
            print(f"   Sampling rate: {rate} Hz")
            print(f"   Duration: {len(waveform) / rate:.2f} seconds")
        
        print("\n" + "=" * 60)
        print("✅ TTS Generation Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    """
    Entry point for the TTS application
    
    Usage:
        # Basic usage with default text
        python main.py
        
        # Custom text
        python main.py --text "Your custom text here"
        
        # Different model
        python main.py --model "facebook/mms-tts-eng"
        
        # Custom output path
        python main.py --output "my_audio.wav"
        
        # Generate multiple samples
        python main.py --batch
    
    Challenges and Solutions:
    1. Model Loading:
       - Challenge: Large model files take time to download
       - Solution: Models are cached after first download
    
    2. Memory Usage:
       - Challenge: Loading models requires significant RAM
       - Solution: Use torch.no_grad() to reduce memory footprint
    
    3. Audio Quality:
       - Challenge: Ensuring proper sample rate for playback
       - Solution: Use model's native sampling_rate configuration
    
    4. Cross-platform Compatibility:
       - Challenge: Different audio players support different formats
       - Solution: WAV format is universally supported
    """
    exit(main())
