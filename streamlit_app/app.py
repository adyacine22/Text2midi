"""
Text2MIDI Streamlit App
Generate MIDI files from text descriptions using the Text2MIDI model
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Streamlit app directories
STREAMLIT_APP_DIR = Path(__file__).parent
GENERATED_MIDI_DIR = STREAMLIT_APP_DIR / "generated" / "midi"
GENERATED_AUDIO_DIR = STREAMLIT_APP_DIR / "generated" / "audio"

# Create directories if they don't exist
GENERATED_MIDI_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Import utilities and components
from utils.model_loader import load_text2midi_model
from utils.validators import (
    validate_prompt,
    sanitize_prompt,
    validate_generation_params,
)
from utils.midi_processor import midi_to_wav
from components.generator import (
    generate_midi_from_prompt,
    save_midi_file,
    get_generation_summary,
)
from components.history import render_history_section, add_to_history
from components.prompt_library import render_quick_examples, render_prompt_library
from instruments_mapping import INSTRUMENT_CLASSES


# Page configuration
st.set_page_config(
    page_title="Text2MIDI",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Minimal CSS styling
st.markdown(
    """
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Minimal header */
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    /* Subtle subtitle */
    .subtitle {
        font-size: 0.95rem;
        color: #6B7280;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Clean buttons */
    .stButton>button {
        border-radius: 6px;
        border: 1px solid #E5E7EB;
        background: white;
        color: #374151;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        border-color: #D1D5DB;
        background: #F9FAFB;
    }
    
    .stButton>button[kind="primary"] {
        background: #111827;
        color: white;
        border: none;
    }
    
    .stButton>button[kind="primary"]:hover {
        background: #1F2937;
    }
    
    /* Minimal dividers */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #F3F4F6;
    }
    
    /* Clean text area */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        font-size: 0.95rem;
    }
    
    /* Minimal metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    /* Clean expanders */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #374151;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state variables"""
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "selected_prompt" not in st.session_state:
        st.session_state.selected_prompt = ""
    if "last_generation" not in st.session_state:
        st.session_state.last_generation = None


def load_model():
    """Load the Text2MIDI model with progress indicator"""
    if not st.session_state.model_loaded:
        with st.spinner("🎼 Loading Text2MIDI model... This may take a few seconds."):
            try:
                model, remi_tokenizer, t5_tokenizer, device, model_info = (
                    load_text2midi_model()
                )
                st.session_state.model = model
                st.session_state.remi_tokenizer = remi_tokenizer
                st.session_state.t5_tokenizer = t5_tokenizer
                st.session_state.device = device
                st.session_state.model_info = model_info
                st.session_state.model_loaded = True
                return True
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                st.info(
                    "Please ensure the model files are in the `artifacts/` directory."
                )
                return False
    return True


def main():
    """Main application"""
    initialize_session_state()

    # Load model
    if not load_model():
        st.stop()

    # Minimal header
    st.markdown('<h1 class="main-header">Text2MIDI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Generate MIDI from text descriptions</p>',
        unsafe_allow_html=True,
    )

    # Quick examples (minimal)
    render_quick_examples()

    # Text input
    prompt = st.text_area(
        "Musical description",
        value=st.session_state.selected_prompt,
        height=120,
        placeholder="A melodic piano piece in C major with a relaxing atmosphere...",
        label_visibility="collapsed",
    )

    # Minimal character counter
    char_count = len(prompt) if prompt else 0
    if char_count > 500:
        st.caption(f"⚠️ {char_count}/500 characters")
    else:
        st.caption(f"{char_count}/500")

    # Instrument class selection (shown directly under the prompt)
    instrument_options = sorted(INSTRUMENT_CLASSES.keys())
    selected_instruments = st.multiselect(
        "Allowed instrument classes (optional)",
        options=instrument_options,
        default=[],
        help=(
            "Select high-level instrument classes to allow "
            "(e.g., Piano, Strings, Drums). Leave empty to "
            "allow all instruments."
        ),
    )
    instrument_programs = []
    for cls in selected_instruments:
        instrument_programs.extend(INSTRUMENT_CLASSES.get(cls, []))

    # Settings expander (minimal)
    with st.expander("⚙️ Settings"):
        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider(
                "Length", 100, 2048, 1000, 100, help="Number of tokens to generate"
            )
            temperature = st.slider(
                "Temperature",
                0.1,
                2.0,
                1.0,
                0.1,
                help="Higher = more creative, Lower = more predictable",
            )
        # No additional settings in col2 for now

    # Generate button
    generate_button = st.button("Generate", type="primary", use_container_width=True)

    # Prompt library (minimal)
    with st.expander("Browse examples"):
        render_prompt_library()

    # Handle generation
    if generate_button:
        # Validate prompt
        is_valid, error_message = validate_prompt(prompt)
        if not is_valid:
            st.error(error_message)
        else:
            # Validate parameters
            params_valid, params_error = validate_generation_params(
                max_length, temperature
            )
            if not params_valid:
                st.error(params_error)
            else:
                # Sanitize prompt
                clean_prompt = sanitize_prompt(prompt)

                # Generate MIDI
                try:
                    with st.spinner("Generating..."):
                        progress_bar = st.progress(0)
                        progress_bar.progress(20)

                        # Generate MIDI tokens
                        midi_object, gen_time, num_tokens = generate_midi_from_prompt(
                            clean_prompt,
                            st.session_state.model,
                            st.session_state.remi_tokenizer,
                            st.session_state.t5_tokenizer,
                            st.session_state.device,
                            max_length=max_length,
                            temperature=temperature,
                            allowed_programs=instrument_programs,
                        )

                        progress_bar.progress(60)

                        # Save MIDI file to streamlit_app/generated/midi/
                        midi_path = save_midi_file(
                            midi_object, GENERATED_MIDI_DIR, filename_prefix="generated"
                        )

                        progress_bar.progress(80)

                        # Convert to audio in streamlit_app/generated/audio/
                        audio_filename = Path(midi_path).stem + ".wav"
                        audio_path = GENERATED_AUDIO_DIR / audio_filename
                        audio_success = midi_to_wav(str(midi_path), str(audio_path))

                        progress_bar.progress(100)
                        progress_bar.empty()

                    # Create summary
                    summary = get_generation_summary(
                        clean_prompt, gen_time, num_tokens, str(midi_path)
                    )

                    # Add audio path if available
                    if audio_success:
                        summary["audio_path"] = str(audio_path)

                    # Add to history
                    add_to_history(summary)

                    # Store in session state
                    st.session_state.last_generation = {
                        "midi_path": str(midi_path),
                        "audio_path": str(audio_path) if audio_success else None,
                        "summary": summary,
                    }

                    st.success("✓ Generated")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.last_generation = None

    # Display last generation (minimal)
    if st.session_state.last_generation:
        st.divider()
        gen = st.session_state.last_generation
        summary = gen["summary"]

        # Compact metrics
        cols = st.columns(3)
        cols[0].metric("Tokens", summary["num_tokens"])
        cols[1].metric("Time", summary["generation_time"])
        cols[2].metric("Size", summary["file_size"])

        # Audio player (minimal)
        if gen["audio_path"] and os.path.exists(gen["audio_path"]):
            with open(gen["audio_path"], "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/wav")

        # Download buttons (minimal)
        col1, col2 = st.columns(2)
        with col1:
            with open(gen["midi_path"], "rb") as f:
                st.download_button(
                    "Download MIDI",
                    f,
                    Path(gen["midi_path"]).name,
                    "audio/midi",
                    use_container_width=True,
                )
        with col2:
            if gen["audio_path"] and os.path.exists(gen["audio_path"]):
                with open(gen["audio_path"], "rb") as f:
                    st.download_button(
                        "Download WAV",
                        f,
                        Path(gen["audio_path"]).name,
                        "audio/wav",
                        use_container_width=True,
                    )

    # History (minimal)
    with st.expander("History"):
        render_history_section()

    # Minimal footer
    st.divider()
    st.caption(
        "Text2MIDI • [Paper](https://arxiv.org/abs/2412.16526) • [GitHub](https://github.com/AMAAI-Lab/text2midi)"
    )


if __name__ == "__main__":
    main()
