#!/bin/bash

# Text2MIDI Streamlit App - Quick Start Script

echo "🎵 Text2MIDI Streamlit App - Quick Start"
echo "========================================"
echo ""

# Check if we're in the streamlit_app directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Please run this script from the streamlit_app directory"
    echo "   cd streamlit_app && ./start_app.sh"
    exit 1
fi

# Check for required model files
echo ""
echo "Checking for required files..."

if [ ! -f "../artifacts/pytorch_model.bin" ]; then
    echo "❌ Error: Model file not found at ../artifacts/pytorch_model.bin"
    exit 1
fi
echo "✅ Model file found"

if [ ! -f "../artifacts/vocab_remi.pkl" ]; then
    echo "❌ Error: Tokenizer not found at ../artifacts/vocab_remi.pkl"
    exit 1
fi
echo "✅ Tokenizer found"

if [ ! -f "../soundfont/FluidR3_GM.sf2" ]; then
    echo "⚠️  Warning: Soundfont not found at ../soundfont/FluidR3_GM.sf2"
    echo "   Audio preview will not be available"
fi

# Check for FluidSynth
if ! command -v fluidsynth &> /dev/null; then
    echo "⚠️  Warning: FluidSynth not found"
    echo "   Install with: brew install fluid-synth (macOS) or apt-get install fluidsynth (Linux)"
    echo "   Audio conversion will not work without it"
fi

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: Streamlit not installed"
    echo "   Install with: pip install -r requirements_streamlit.txt"
    exit 1
fi
echo "✅ Streamlit installed"

# Create generated directories if they don't exist
mkdir -p generated/midi generated/audio

echo ""
echo "✨ All checks passed!"
echo ""
echo "Starting Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

# Start the app
streamlit run app.py
