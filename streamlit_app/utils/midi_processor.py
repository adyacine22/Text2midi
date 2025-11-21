"""MIDI processing and audio conversion utilities"""

import os
import subprocess
from pathlib import Path
import streamlit as st


def midi_to_wav(midi_path, wav_path, soundfont_path=None):
    """
    Convert MIDI file to WAV using FluidSynth

    Args:
        midi_path: Path to input MIDI file
        wav_path: Path to output WAV file
        soundfont_path: Path to soundfont file (optional)

    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Resolve soundfont path
        if soundfont_path is not None:
            soundfont_path = Path(soundfont_path)
        else:
            root_dir = Path(__file__).parent.parent.parent
            soundfont_dir = root_dir / "soundfont"

            # Pick the first .sf2 file found in the directory (if any)
            if soundfont_dir.exists():
                sf2_files = list(soundfont_dir.glob("*.sf2"))
                if sf2_files:
                    soundfont_path = sf2_files[0]

        # If we still don't have a valid soundfont path, abort with a warning
        if soundfont_path is None or not soundfont_path.exists():
            root_dir = Path(__file__).parent.parent.parent
            soundfont_dir = root_dir / "soundfont"
            st.warning(
                f"No .sf2 soundfont found. "
                f"Place a soundfont file in {soundfont_dir} to enable audio preview."
            )
            return False

        soundfont_path = str(soundfont_path)

        # FluidSynth command
        cmd = [
            "fluidsynth",
            "-ni",  # Non-interactive
            str(soundfont_path),
            str(midi_path),
            "-F",  # Output to file
            str(wav_path),
            "-r",  # Sample rate
            "44100",
        ]

        # Run FluidSynth
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

        if result.returncode == 0 and os.path.exists(wav_path):
            return True
        else:
            st.warning(
                "Audio conversion failed. MIDI file is still available for download."
            )
            return False

    except FileNotFoundError:
        st.warning(
            "FluidSynth not found. Please install it with: brew install fluid-synth (macOS) or apt-get install fluidsynth (Linux)"
        )
        return False
    except subprocess.TimeoutExpired:
        st.warning("Audio conversion timed out.")
        return False
    except Exception as e:
        st.warning(f"Audio conversion error: {str(e)}")
        return False


def get_file_size(file_path):
    """Get file size in human-readable format"""
    try:
        size_bytes = os.path.getsize(file_path)

        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0

        return f"{size_bytes:.1f} TB"
    except:
        return "Unknown"


def validate_midi_file(midi_path):
    """
    Validate that the MIDI file was created successfully

    Args:
        midi_path: Path to MIDI file

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        return os.path.exists(midi_path) and os.path.getsize(midi_path) > 0
    except:
        return False
