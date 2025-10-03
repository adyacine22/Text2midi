"""Generation history component"""

import streamlit as st
from pathlib import Path
import os


def initialize_history():
    """Initialize history in session state"""
    if "generation_history" not in st.session_state:
        st.session_state.generation_history = []


def add_to_history(summary):
    """Add a generation to history"""
    initialize_history()
    st.session_state.generation_history.insert(0, summary)  # Add to beginning
    # Keep only last 20 generations
    st.session_state.generation_history = st.session_state.generation_history[:20]


def clear_history():
    """Clear all history"""
    st.session_state.generation_history = []


def render_history_section():
    """Render the generation history section with minimal design"""
    initialize_history()

    history = st.session_state.generation_history

    if not history:
        st.caption("No generations yet")
        return

    # Header with count and clear button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"{len(history)} generation{'s' if len(history) != 1 else ''}")
    with col2:
        if st.button("Clear", key="clear_history", use_container_width=True):
            clear_history()
            st.rerun()

    st.divider()

    # Display each generation (minimal)
    for idx, item in enumerate(history):
        # Prompt (truncated)
        prompt = item.get("prompt", "Unknown prompt")
        if len(prompt) > 60:
            prompt = prompt[:60] + "..."

        st.markdown(f"**{prompt}**")

        # Metadata in single line
        metadata = f"{item.get('num_tokens', 0)} tokens • {item.get('generation_time', '0s')} • {item.get('file_size', '0 KB')}"
        st.caption(metadata)

        # Download buttons (compact)
        col1, col2 = st.columns(2)

        midi_path = item.get("midi_path", "")
        if midi_path and os.path.exists(midi_path):
            with col1:
                with open(midi_path, "rb") as f:
                    st.download_button(
                        "MIDI",
                        f,
                        Path(midi_path).name,
                        "audio/midi",
                        key=f"midi_{idx}",
                        use_container_width=True,
                    )

        audio_path = item.get("audio_path", "")
        if audio_path and os.path.exists(audio_path):
            with col2:
                with open(audio_path, "rb") as f:
                    st.download_button(
                        "WAV",
                        f,
                        Path(audio_path).name,
                        "audio/wav",
                        key=f"wav_{idx}",
                        use_container_width=True,
                    )

        # Separator between items (minimal)
        if idx < len(history) - 1:
            st.markdown(
                "<div style='margin: 1rem 0; border-top: 1px solid #F3F4F6;'></div>",
                unsafe_allow_html=True,
            )
