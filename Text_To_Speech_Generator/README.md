# Text-to-Speech (TTS) Generator

A Python application that uses Hugging Face's pre-trained models to convert text into natural-sounding speech.

## 🎯 Objective

- Clone a pre-trained Text-to-Speech (TTS) model from Hugging Face
- Perform inference to convert input text into spoken audio
- Save and play the generated audio files

## 🚀 Features

- **Easy Model Loading**: Automatically downloads and caches TTS models from Hugging Face
- **Multiple Language Support**: Works with various TTS models (Vietnamese, English, etc.)
- **Audio Export**: Saves generated speech as WAV files
- **Batch Processing**: Generate speech for multiple texts at once
- **Command-line Interface**: Flexible CLI for different use cases

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- soundfile
- IPython (for notebook playback)

## 🔧 Installation

1. **Clone the repository** (or navigate to Assignment_07 folder):
   ```bash
   cd AI_Powered_Apps/Assignment_07
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Basic Usage

Generate speech with default Vietnamese text:
```bash
python main.py
```

This will:
- Load the `facebook/mms-tts-vie` model
- Convert the default Vietnamese text to speech
- Save the output as `output.wav`

### Custom Text

Generate speech from your own text:
```bash
python main.py --text "Xin chào thế giới"
```

### Different Model

Use a different TTS model:
```bash
python main.py --model "facebook/mms-tts-eng" --text "Hello world"
```

### Custom Output Path

Save to a specific file:
```bash
python main.py --output "my_speech.wav"
```

### Batch Processing

Generate multiple audio files:
```bash
python main.py --batch
```

## 📝 Code Structure

```python
# Step 1: Import required libraries
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf

# Step 2: Load the pre-trained model
model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

# Step 3: Prepare input text
text = "Xin chào anh em"

# Step 4: Tokenize the text
inputs = tokenizer(text, return_tensors="pt")

# Step 5: Generate audio waveform
with torch.no_grad():
    output = model(**inputs).waveform

# Step 6: Save the audio
waveform = output.squeeze().cpu().numpy()
sf.write('output.wav', waveform, model.config.sampling_rate)
```

## 🎯 Concepts Covered

1. **Model Cloning**: Loading pre-trained models from Hugging Face Hub
2. **Tokenization**: Converting text into model-compatible format
3. **Inference**: Generating audio waveforms using the TTS model
4. **Audio Processing**: Saving and handling audio data in Python
5. **Error Handling**: Managing model loading and inference errors

## 🔍 Available Models

The application works with various Hugging Face TTS models:

- `facebook/mms-tts-vie` - Vietnamese TTS
- `facebook/mms-tts-eng` - English TTS
- `facebook/mms-tts-fra` - French TTS
- `facebook/mms-tts-spa` - Spanish TTS
- And many more on [Hugging Face](https://huggingface.co/models?pipeline_tag=text-to-speech)

## ⚠️ Challenges & Solutions

### 1. Model Download Time
**Challenge**: Large models take time to download on first run  
**Solution**: Models are automatically cached locally after first download

### 2. Memory Usage
**Challenge**: Loading models requires significant RAM  
**Solution**: Use `torch.no_grad()` during inference to reduce memory footprint

### 3. Audio Quality
**Challenge**: Ensuring proper sample rate for playback  
**Solution**: Use the model's native `sampling_rate` configuration

### 4. Cross-platform Compatibility
**Challenge**: Different audio players support different formats  
**Solution**: WAV format is universally supported

## 📂 Output Files

Generated audio files are saved as WAV format:
- **Single generation**: `output.wav` (or custom name)
- **Batch generation**: `outputs/tts_output_1_TIMESTAMP.wav`, etc.

## 🧪 Testing

1. **Generate sample audio**:
   ```bash
   python main.py
   ```

2. **Play the generated audio**:
   - On macOS: `afplay output.wav`
   - On Linux: `aplay output.wav`
   - On Windows: Double-click `output.wav`

3. **Verify output**:
   - Check that `output.wav` exists
   - File size should be > 0 KB
   - Audio duration matches text length

## 📊 Example Output

```
============================================================
🎙️  Text-to-Speech Generator
============================================================
🔄 Loading TTS model: facebook/mms-tts-vie
   This may take a moment on first run...
✅ Model loaded successfully!
   Sampling rate: 16000 Hz
   Model type: VitsModel

🎤 Generating speech from text...
   Input text: 'Xin chào anh em đến với bài tập của khoá AI Application Engineer'
   Tokenized input shape: torch.Size([1, 44])
   Generating audio waveform...
   Generated waveform shape: (70400,)
   Audio duration: 4.40 seconds
✅ Audio saved successfully!
   Output file: output.wav
   File size: 137.58 KB

📊 Summary:
   Model: facebook/mms-tts-vie
   Input text: 'Xin chào anh em đến với bài tập của khoá AI Application Engineer'
   Output file: output.wav
   Sampling rate: 16000 Hz
   Duration: 4.40 seconds

============================================================
✅ TTS Generation Complete!
============================================================
```

## 📚 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [VITS Model Paper](https://arxiv.org/abs/2106.06103)
- [Audio Processing with soundfile](https://pysoundfile.readthedocs.io/)

## 👤 Author

Duke Nguyen - AI Application Engineer Course

## 📅 Date

November 3, 2025

---

**Note**: First run will download the model (~300MB), subsequent runs will use cached model.
