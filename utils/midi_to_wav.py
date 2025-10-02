import os
import subprocess
import requests
import zipfile
from io import BytesIO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
soundfont_filepath = os.path.join(BASE_DIR, "soundfont", "FluidR3_GM.sf2")
soundfont_zip_url = "https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip"
midi_root = os.path.join(BASE_DIR, "generated_midi")
wav_root = os.path.join(BASE_DIR, "wav")


def download_and_extract_soundfont():
    if os.path.isfile(soundfont_filepath):
        print(f"✓ SoundFont already exists: {soundfont_filepath}")
        return

    print("📥 Downloading SoundFont...")
    os.makedirs(os.path.dirname(soundfont_filepath), exist_ok=True)

    response = requests.get(soundfont_zip_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    downloaded = 0
    content = b""

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content += chunk
                downloaded += len(chunk)
                pbar.update(len(chunk))

    print("📦 Extracting SoundFont...")
    with zipfile.ZipFile(BytesIO(content)) as z:
        sf2_file = next(name for name in z.namelist() if name.lower().endswith(".sf2"))
        with z.open(sf2_file) as src, open(soundfont_filepath, "wb") as dst:
            dst.write(src.read())

    print(f"✓ SoundFont ready: {soundfont_filepath}")


def save_wav(midi_filepath, wav_filepath):
    if os.path.isfile(wav_filepath):
        return wav_filepath

    try:
        subprocess.run(
            [
                "fluidsynth",
                "-r",
                "48000",
                soundfont_filepath,
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
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {os.path.basename(midi_filepath)}: {e}")
        return None

    return wav_filepath


def process_midi_file(midi_filepath):
    wav_filepath = os.path.join(
        wav_root, os.path.basename(midi_filepath).replace(".mid", ".wav")
    )
    os.makedirs(wav_root, exist_ok=True)
    save_wav(midi_filepath, wav_filepath)


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
