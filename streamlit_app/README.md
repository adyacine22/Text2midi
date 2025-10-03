# Text2MIDI Streamlit App

A user-friendly web interface for generating MIDI files from text descriptions using the Text2MIDI model.

## Features

✨ **Implemented Features (P0 & P1):**

### P0 - Core Features (Must Have)
- ✅ **Text Input & MIDI Generation** - Simple textarea with validation
- ✅ **Generation Parameters** - Control max_length, temperature, top-k, top-p
- ✅ **MIDI Playback & Download** - Audio preview + download MIDI/WAV files
- ✅ **Model Information** - Display model statistics and system info

### P1 - Enhanced Features (Should Have)
- ✅ **Generation History** - Track all generations in the current session
- ✅ **Prompt Library** - 40+ pre-built prompts organized by category:
  - Piano, Electronic, Classical, Jazz, World Music
  - Rock/Pop, Mood-based, Christmas

## Quick Start

### Prerequisites

1. **System Requirements:**
   ```bash
   # macOS
   brew install fluid-synth
   
   # Ubuntu/Debian
   sudo apt-get install fluidsynth
   
   # Windows
   # Download from: https://github.com/FluidSynth/fluidsynth/releases
   ```

2. **Model Files:**
   Ensure you have the following files in the `artifacts/` directory:
   - `pytorch_model.bin` (~440MB)
   - `vocab_remi.pkl` (~2MB)

3. **Soundfont:**
   Ensure `soundfont/FluidR3_GM.sf2` exists in the project root

### Installation

```bash
# From the project root directory
cd streamlit_app

# Install dependencies
pip install -r requirements_streamlit.txt
```

### Running the App

```bash
# From the streamlit_app directory
streamlit run app.py

# Or with custom port
streamlit run app.py --server.port 8501
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Basic Generation

1. Enter a text description in the input area
2. Click "🎼 Generate MIDI"
3. Wait for generation (10-30 seconds)
4. Preview audio and download files

**Example prompt:**
```
A melodic piano piece in C major with a relaxing atmosphere, 
slow tempo and gentle dynamics
```

### 2. Quick Examples

Click one of the quick example buttons:
- 🎹 Piano
- 🎸 Rock
- 🎻 Classical
- 🎧 Electronic

### 3. Advanced Parameters

Adjust in the sidebar:
- **Max Length** (100-2048): Number of tokens to generate
- **Temperature** (0.1-2.0): Creativity level
- **Top-k** (0-100): Sampling diversity
- **Top-p** (0.0-1.0): Nucleus sampling

### 4. Prompt Library

Browse 40+ categorized prompts:
- Click "💡 Browse Prompt Library"
- Select a category tab
- Click "Use" on any prompt

### 5. Generation History

- View past generations below the main interface
- Download MIDI or WAV files from history
- Clear all history with "🗑️ Clear All"

## File Structure

```
streamlit_app/
├── app.py                      # Main Streamlit application
├── components/
│   ├── __init__.py
│   ├── generator.py            # MIDI generation logic
│   ├── history.py              # History management
│   └── prompt_library.py       # Pre-built prompts (40+ examples)
├── utils/
│   ├── __init__.py
│   ├── model_loader.py         # Model initialization & caching
│   ├── midi_processor.py       # MIDI to WAV conversion
│   └── validators.py           # Input validation
├── generated/                  # Output directory
│   ├── midi/                   # Generated MIDI files
│   └── audio/                  # Generated WAV files
├── requirements_streamlit.txt  # Python dependencies
└── README.md                   # This file
```

## Tips for Best Results

### Prompt Writing

**Good prompts include:**
- ✅ Musical style/genre (classical, jazz, electronic)
- ✅ Instruments (piano, guitar, strings, drums)
- ✅ Key and time signature (C major, 4/4 time)
- ✅ Tempo (slow, moderate, fast, or BPM)
- ✅ Mood/atmosphere (happy, sad, relaxing, energetic)

**Example of a good prompt:**
```
A classical string quartet in G major with elegant melodies, 
moderate tempo of 90 BPM, featuring violin, viola, cello and 
contrabass. The piece has a romantic and emotional character.
```

**Example of a poor prompt:**
```
music
```

### Parameter Settings

| Use Case | Max Length | Temperature | Top-k | Top-p |
|----------|-----------|-------------|-------|-------|
| Short melody | 200-500 | 1.0 | 0 | 0.0 |
| Standard song | 800-1200 | 1.0 | 0 | 0.0 |
| Long composition | 1500-2048 | 1.0 | 0 | 0.0 |
| More creative | any | 1.2-1.5 | 50 | 0.9 |
| More predictable | any | 0.7-0.9 | 0 | 0.0 |

## Troubleshooting

### Model Loading Issues

**Error: "Model weights not found"**
```bash
# Ensure model files exist
ls -lh ../artifacts/
# Should show: pytorch_model.bin, vocab_remi.pkl
```

### Audio Conversion Issues

**Error: "FluidSynth not found"**
```bash
# macOS
brew install fluid-synth

# Linux
sudo apt-get install fluidsynth

# Verify installation
which fluidsynth
```

**Error: "Soundfont not found"**
```bash
# Check soundfont exists
ls -lh ../soundfont/FluidR3_GM.sf2

# If missing, download from project repository
```

### Memory Issues

**Error: "Out of memory"**
- Reduce `Max Length` parameter (try 500-800)
- Close other applications
- Use CPU instead of GPU if on limited GPU memory

### Slow Generation

**Taking too long (>60 seconds)?**
- Reduce `Max Length` (shorter sequences = faster)
- Check if using CPU instead of GPU (normal on CPU)
- Ensure no other heavy processes running

## Performance

**Expected generation times:**
- CPU: 20-40 seconds for 1000 tokens
- MPS (Apple Silicon): 10-20 seconds
- CUDA GPU: 5-15 seconds

**Memory usage:**
- Model: ~2GB RAM
- Generation: ~1-2GB additional
- Total: ~3-4GB RAM recommended

## Keyboard Shortcuts

- `Ctrl + Enter` - Generate MIDI (when in text area)
- `Esc` - Close expanded sections

## Future Enhancements (Roadmap)

See `STREAMLIT_APP_SPECIFICATION.md` for planned P2 features:
- Batch generation (multiple variations)
- MIDI editor (basic editing)
- Export to MusicXML/PDF
- Cloud deployment
- User accounts

## Credits

- **Text2MIDI Model**: [AMAAI Lab](https://github.com/AMAAI-Lab/text2midi)
- **Paper**: Accepted at AAAI 2025
- **Authors**: Keshav Bhandari, Abhinaba Roy, Kyra Wang, Geeta Puri, Simon Colton, Dorien Herremans

## Links

- 📄 [Paper](https://arxiv.org/abs/2412.16526)
- 🤗 [HuggingFace Model](https://huggingface.co/amaai-lab/text2midi)
- 🎮 [Demo](https://huggingface.co/spaces/amaai-lab/text2midi)
- 💻 [GitHub](https://github.com/AMAAI-Lab/text2midi)

## License

This project follows the license of the main Text2MIDI repository.

---

**Happy composing! 🎵**
