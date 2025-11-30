import os
import subprocess
import requests
import zipfile
from io import BytesIO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_soundfont(soundfont_dir):
    """Finds the first .sf2 file in the given directory."""
    if not os.path.exists(soundfont_dir):
        return None
    for file in os.listdir(soundfont_dir):
        if file.lower().endswith(".sf2"):
            return os.path.join(soundfont_dir, file)
    return None


def download_and_extract_soundfont(target_dir=None):
    if target_dir is None:
        target_dir = os.path.join(BASE_DIR, "soundfont")
    
    soundfont_filepath = get_soundfont(target_dir)
    if soundfont_filepath:
        print(f"✓ SoundFont already exists: {soundfont_filepath}")
        return soundfont_filepath

    print(f"📥 Downloading SoundFont to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)

    response = requests.get(soundfont_zip_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    content = b""
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content += chunk
                pbar.update(len(chunk))

    print("📦 Extracting SoundFont...")
    with zipfile.ZipFile(BytesIO(content)) as z:
        sf2_file = next(name for name in z.namelist() if name.lower().endswith(".sf2"))
        # Extract to target_dir
        z.extract(sf2_file, target_dir)
        soundfont_filepath = os.path.join(target_dir, sf2_file)

    print(f"✓ SoundFont ready: {soundfont_filepath}")
    return soundfont_filepath


def save_wav(midi_filepath, wav_filepath, soundfont_path=None):
    if os.path.isfile(wav_filepath):
        return wav_filepath

    if soundfont_path is None:
        # Fallback to default location search
        default_dir = os.path.join(BASE_DIR, "soundfont")
        soundfont_path = get_soundfont(default_dir)
        if soundfont_path is None:
            print("❌ No SoundFont found. Please provide soundfont_path or download it.")
            return None

    try:
        env = os.environ.copy()
        env["SDL_AUDIODRIVER"] = "dummy"
        env["SDL_VIDEODRIVER"] = "dummy"
        
        subprocess.run(
            [
                "fluidsynth",
                "-a",
                "null",
                "-o",
                "synth.polyphony=1024",
                "-r",
                "48000",
                soundfont_path,
                "-g",
                "1.0",
                "--quiet",
                "--no-shell",
                midi_filepath,
                "-T",
                "wav",
                "-F",
                wav_filepath,
            ],
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {os.path.basename(midi_filepath)}: {e}")
        return None

    return wav_filepath


def process_midi_file(args):
    midi_filepath, soundfont_path = args
    wav_filepath = os.path.join(
        wav_root, os.path.basename(midi_filepath).replace(".mid", ".wav")
    )
    os.makedirs(wav_root, exist_ok=True)
    save_wav(midi_filepath, wav_filepath, soundfont_path)


def main():
    print("🎵 MIDI to WAV Converter Starting...")
    download_and_extract_soundfont()

    if not os.path.exists(midi_root):
        print(f"❌ MIDI directory not found: {midi_root}")
        return

    midi_files = [
        os.path.join(midi_root, f)
        for f in os.listdir(midi_root)
        if f.lower().endswith(".mid")
    ]

    if not midi_files:
        print(f"❌ No MIDI files found in: {midi_root}")
        return

    print(f"🎼 Found {len(midi_files)} MIDI files")
    print(f"📁 Output directory: {wav_root}")

    os.makedirs(wav_root, exist_ok=True)

    with Pool(cpu_count() // 2) as pool:
        results = list(
            tqdm(
                pool.imap(process_midi_file, midi_files),
                total=len(midi_files),
                desc="🎧 Converting",
            )
        )

    print(f"✅ Conversion complete! WAV files saved to: {wav_root}")


if __name__ == "__main__":
    main()
