import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

from symusic import Score

# Make project root importable so we can reuse mappings
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.instruments_mapping import INSTRUMENT_CLASSES, INSTRUMENTS


# ---------- Music analysis helpers ----------

MAJOR_KEYS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab"]
MINOR_KEYS = ["A", "E", "B", "F#", "C#", "G#", "D#", "A#", "D", "G", "C", "F"]
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def key_from_notes(pitch_classes: Iterable[int]) -> str:
    pcs_list = [pc % 12 for pc in pitch_classes]
    if not pcs_list:
        return "unknown"
    hist = [0] * 12
    for pc in pcs_list:
        hist[pc] += 1

    best_score, best_key, best_mode = -1e9, "C", "major"
    for tonic in range(12):
        rotated = hist[tonic:] + hist[:tonic]
        maj_score = sum(x * y for x, y in zip(rotated, MAJOR_PROFILE))
        min_score = sum(x * y for x, y in zip(rotated, MINOR_PROFILE))
        if maj_score > best_score:
            best_score, best_key, best_mode = maj_score, MAJOR_KEYS[tonic], "major"
        if min_score > best_score:
            best_score, best_key, best_mode = min_score, MINOR_KEYS[tonic], "minor"
    return f"{best_key} {best_mode}"


def tempo_word(bpm: float) -> str:
    if bpm < 76:
        return "slow"
    if bpm < 110:
        return "moderate"
    if bpm < 150:
        return "fast"
    return "very fast"


def program_to_name(program: int, is_drum: bool) -> str:
    """Convert MIDI program number to instrument name."""
    if is_drum:
        return INSTRUMENTS.get(-1, "Drums")
    return INSTRUMENTS.get(program, f"Program {program}")


def program_to_classes(program: int, is_drum: bool) -> List[str]:
    """Get instrument class for a program (for metadata)."""
    if is_drum:
        return ["Drums / Percussion"]
    classes = []
    for name, plist in INSTRUMENT_CLASSES.items():
        if program in plist:
            classes.append(name)
    return classes


def get_track_complexity_adjectives(n_tracks: int) -> List[str]:
    """Get adjectives describing piece complexity based on track count.
    
    Returns a list of possible adjectives with proper articles.
    More variety than the old version.
    """
    if n_tracks <= 1:
        # Solo or very minimal
        return [
            "a solo", "a minimal", "a simple", "a sparse",
            "an intimate", "a focused", "a stripped-down"
        ]
    elif n_tracks <= 3:
        # Small ensemble
        return [
            "a simple", "a minimal", "a clean", "a concise",
            "an understated", "a straightforward", "a modest",
            "a chamber-style"
        ]
    elif n_tracks <= 6:
        # Medium complexity
        return [
            "a moderately complex", "a layered", "a textured",
            "a balanced", "a well-structured", "a cohesive",
            "a nuanced", "an ensemble"
        ]
    elif n_tracks <= 10:
        # Rich orchestration
        return [
            "a rich", "a detailed", "an intricate",
            "a sophisticated", "a full", "an elaborate",
            "a lush", "a multi-layered"
        ]
    else:
        # Very complex
        return [
            "a highly complex", "an elaborate", "a densely arranged",
            "a grand", "an orchestral",
            "a richly textured"
        ]


def get_ensemble_type(instrument_programs: List[str], n_tracks: int) -> str:
    """Determine ensemble type based on instruments and count.
    
    Returns descriptive ensemble type like 'duo', 'string quartet', 'jazz ensemble', etc.
    Uses priority-based detection with nuanced rules for accurate classification.
    """
    # Analyze instrument families
    has_strings = any(inst in ["Violin", "Viola", "Cello", "Contrabass"] for inst in instrument_programs)
    has_winds = any("Flute" in inst or "Clarinet" in inst or "Oboe" in inst or "Bassoon" in inst or "Piccolo" in inst for inst in instrument_programs)
    has_brass = any("Trumpet" in inst or "Trombone" in inst or "Horn" in inst or "Tuba" in inst for inst in instrument_programs)
    has_piano = any("Piano" in inst for inst in instrument_programs)
    has_guitar = any("Guitar" in inst and "Bass" not in inst for inst in instrument_programs)
    has_bass_guitar = any("Bass" in inst or "Contrabass" in inst for inst in instrument_programs)
    has_drums = "Drums" in instrument_programs
    has_sax = any("Sax" in inst for inst in instrument_programs)
    has_percussion = any("Timpani" in inst or "Marimba" in inst or "Xylophone" in inst or "Vibraphone" in inst for inst in instrument_programs)
    
    # Count instruments by family
    num_strings = sum(1 for inst in instrument_programs if inst in ["Violin", "Viola", "Cello", "Contrabass", "Harp"])
    num_winds = sum(1 for inst in instrument_programs if any(w in inst for w in ["Flute", "Clarinet", "Oboe", "Bassoon", "Piccolo"]))
    num_brass = sum(1 for inst in instrument_programs if any(b in inst for b in ["Trumpet", "Trombone", "Horn", "Tuba"]))
    num_sax = sum(1 for inst in instrument_programs if "Sax" in inst)
    
    # Count classical orchestra instrument families
    classical_count = sum([has_strings, has_winds, has_brass, has_percussion])
    
    # PRIORITY 1: Large Orchestral/Band Ensembles (7+ tracks with classical instruments)
    if classical_count >= 2 and n_tracks > 6:
        # Big Band (jazz-style): saxes + brass + rhythm, typically 10-20 pieces
        if has_sax and num_sax >= 3 and has_brass and (has_piano or has_drums):
            return "big band"
        
        # Concert Band (classical wind band): winds + brass, no/minimal strings
        if has_winds and has_brass and not has_strings:
            return "concert band"
        
        # Symphony Orchestra: strings + winds/brass
        if has_strings and (has_winds or has_brass):
            return "orchestral"
        
        # Wind Orchestra: primarily winds with some brass
        if has_winds and num_winds > num_brass:
            return "wind orchestra"
        
        # Default large classical ensemble
        return "orchestral"
    
    # PRIORITY 2: Jazz Ensembles (specific combo requirements)
    if has_sax and classical_count <= 1:
        # Jazz combo: sax + rhythm section
        if (has_piano or has_guitar) and (has_bass_guitar or has_drums):
            return "jazz ensemble"
        # Small sax group without full rhythm
        elif n_tracks <= 6:
            return "jazz ensemble"
    
    # Jazz without sax: piano trio, guitar trio, etc.
    if has_piano and has_bass_guitar and has_drums and not has_strings and not has_winds and not has_brass:
        return "jazz ensemble"
    
    # PRIORITY 3: Rock/Pop Ensembles
    if has_guitar and has_drums:
        # Classic rock: guitar + bass + drums
        if has_bass_guitar:
            return "rock ensemble"
        # Rock without bass (power duo/trio)
        elif n_tracks <= 4:
            return "rock ensemble"
    
    # PRIORITY 4: Pure Classical Ensembles (single family)
    # String ensemble: ALL instruments are strings
    all_strings = all(inst in ["Violin", "Viola", "Cello", "Contrabass", "Harp"] for inst in instrument_programs)
    if all_strings and has_strings:
        if n_tracks <= 10:
            return "string ensemble"
        else:
            return "string orchestra"
    
    # Wind ensemble: primarily winds, no strings/brass
    if has_winds and not has_strings and not has_brass and n_tracks <= 10:
        return "wind ensemble"
    
    # Brass ensemble: primarily brass, no strings/winds
    if has_brass and not has_strings and not has_winds and n_tracks <= 10:
        return "brass ensemble"
    
    # PRIORITY 5: Chamber Music (mixed classical, small/medium group)
    if classical_count >= 2 and n_tracks <= 8:
        return "chamber ensemble"
    
    # PRIORITY 6: Orchestral Fallback (any multi-family classical)
    if classical_count >= 2:
        if has_winds and has_brass and not has_strings:
            return "concert band"
        return "orchestral"
    
    # PRIORITY 7: Piano-based Small Groups
    if has_piano and n_tracks <= 3 and not has_drums and not has_guitar:
        return "piano ensemble"
    
    # PRIORITY 8: Numerical Names (for non-genre-specific small groups)
    if n_tracks == 2:
        return "duo"
    elif n_tracks == 3:
        return "trio"
    elif n_tracks == 4:
        return "quartet"
    elif n_tracks == 5:
        return "quintet"
    
    # PRIORITY 9: Generic Size-Based Fallback
    if n_tracks <= 3:
        return "small ensemble"
    elif n_tracks <= 8:
        return "ensemble"
    else:
        return "large ensemble"


# ========== INLINE TESTING ==========
if __name__ == "__main__" and __name__ != "__main__":  # Only runs when explicitly called
    print("Testing Ensemble Type Detection...")
    print("=" * 80)
    
    test_cases = [
        # (instruments, n_tracks, expected, description)
        # Orchestral/Band
        (['Piano', 'Alto Sax', 'Baritone Sax', 'Bassoon', 'Clarinet', 'Flute', 'French Horn', 'Marimba', 'Oboe', 'Piccolo', 'Tenor Sax', 'Timpani', 'Trombone', 'Trumpet', 'Tuba'], 15, 'big band', 'Large concert band with saxes'),
        (['Violin', 'Viola', 'Cello', 'Flute', 'Clarinet', 'Oboe', 'Trumpet', 'Trombone', 'Timpani'], 9, 'orchestral', 'Symphony orchestra'),
        (['Flute', 'Clarinet', 'Oboe', 'Bassoon', 'Trumpet', 'Trombone', 'Tuba'], 7, 'concert band', 'Concert band (no strings)'),
        
        # Jazz
        (['Alto Sax', 'Piano', 'Bass', 'Drums'], 4, 'jazz ensemble', 'Jazz quartet'),
        (['Tenor Sax', 'Alto Sax', 'Baritone Sax', 'Trumpet', 'Trombone', 'Piano', 'Bass', 'Drums'], 8, 'big band', 'Jazz big band'),
        (['Piano', 'Bass', 'Drums'], 3, 'jazz ensemble', 'Piano trio (jazz)'),
        
        # Rock/Pop
        (['Guitar', 'Bass', 'Drums'], 3, 'rock ensemble', 'Rock power trio'),
        (['Guitar', 'Drums'], 2, 'rock ensemble', 'Rock duo'),
        
        # Classical Chamber
        (['Violin', 'Viola', 'Cello', 'Contrabass'], 4, 'string ensemble', 'String quartet'),
        (['Piano', 'Violin'], 2, 'duo', 'Piano-violin duo'),
        (['Flute', 'Clarinet', 'Oboe', 'Bassoon', 'Horn'], 5, 'wind ensemble', 'Wind quintet'),
        (['Violin', 'Viola', 'Cello', 'Piano'], 4, 'chamber ensemble', 'Piano quartet'),
        
        # Edge cases
        (['Piano'], 1, 'small ensemble', 'Solo piano'),
        (['Violin', 'Viola', 'Cello', 'Contrabass', 'Violin', 'Viola', 'Cello', 'Contrabass', 'Violin', 'Viola', 'Cello'], 11, 'string orchestra', 'Large string ensemble'),
        (['Trumpet', 'Trombone', 'Tuba'], 3, 'brass ensemble', 'Brass trio'),
    ]
    
    passed = 0
    failed = 0
    
    for instruments, n_tracks, expected, description in test_cases:
        result = get_ensemble_type(instruments, n_tracks)
        status = "✅" if result == expected else "❌"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} {description}")
        print(f"   Instruments: {', '.join(instruments[:3])}{'...' if len(instruments) > 3 else ''} ({n_tracks} tracks)")
        print(f"   Expected: {expected}, Got: {result}")
        print()
    
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print(f"Success rate: {passed/len(test_cases)*100:.1f}%")



def get_folder_complexity(path: Path) -> str:
    """Get complexity descriptor based on folder structure."""
    path_str = str(path).lower()
    
    if "classical" in path_str:
        return "classical"
    elif "contemporary" in path_str:
        return "contemporary"
    elif "baroque" in path_str:
        return "baroque"
    elif "romantic" in path_str:
        return "romantic"
    elif "jazz" in path_str:
        return "jazz"
    elif "pop" in path_str or "rock" in path_str:
        return "modern"
    else:
        return None  # No folder-based complexity


def analyze_midi(path: Path) -> Dict:
    score = Score(str(path))
    pcs: List[int] = []
    classes_set: Set[str] = set()
    programs_set: Set[str] = set()  # NEW: Track specific instrument programs
    
    for tr in score.tracks:
        if tr.is_drum:
            classes_set.update(program_to_classes(-1, True))
            programs_set.add("Drums")
        else:
            classes_set.update(program_to_classes(tr.program, False))
            programs_set.add(program_to_name(tr.program, False))  # NEW: Add program name
        
        for note in tr.notes:
            pcs.append(note.pitch)

    key = key_from_notes(pcs)
    tempo_val = score.tempos[0].qpm if score.tempos else 120
    ts = score.time_signatures[0] if score.time_signatures else None
    time_sig = f"{ts.numerator}/{ts.denominator}" if ts else "4/4"
    
    # Get ensemble type based on instruments
    ensemble_type = get_ensemble_type(list(programs_set), len(score.tracks))
    
    # Get folder-based complexity
    folder_complexity = get_folder_complexity(path)

    return {
        "key": key,
        "tempo": tempo_val,
        "time_signature": time_sig,
        "instrument_classes": sorted(classes_set),  # Keep for metadata
        "instrument_programs": sorted(programs_set),  # Specific instruments for caption
        "adjectives_tracks": get_track_complexity_adjectives(len(score.tracks)),
        "ensemble_type": ensemble_type,  # NEW: ensemble type
        "folder_complexity": folder_complexity,  # NEW: folder-based complexity
    }


# ---------- Caption generation helpers ----------


# Opening phrases - expanded to 18 options
OPENING_TEMPLATES = [
    "A {adjective} {piece_type}",
    "A {piece_type} that is {adjective}",
    "A {piece_type}",
    "{piece_type}"
    "generate a {adjective} {piece_type}",
    "generate a {piece_type} that is {adjective}",
    "generate a {piece_type}",
    "create a {adjective} {piece_type}",
    "create a {piece_type} that is {adjective}",
    "create a {piece_type}",
    "compose a {adjective} {piece_type}",
    "compose a {piece_type} that is {adjective}",
    "compose a {piece_type}",
]

# Piece types - expanded to 22 options
PIECE_TYPES = [
    "piece", "composition", "arrangement", "work", "performance",
    "musical piece", "musical work", "track", "score", "orchestration",
    "creation", "number", "selection", "movement", "opus",
    "musical arrangement", "musical composition", "musical creation",
    "musical selection", "musical performance", "musical number", "recording"
]

# Instrument description - expanded to 25 options
INSTRUMENT_INTRO_PHRASES = [
    "featuring {instruments}",
    "with {instruments}",
    "showcasing {instruments}",
    "highlighting {instruments}",
    "for {instruments}",
    "performed by {instruments}",
    "played by {instruments}",
    "built around {instruments}",
    "centered on {instruments}",
    "driven by {instruments}",
    "led by {instruments}",
    "starring {instruments}",
    "presenting {instruments}",
    "employing {instruments}",
    "utilizing {instruments}",
    "incorporating {instruments}",
    "combining {instruments}",
    "layering {instruments}",
    "with {instruments} at its core",
    "arranged for {instruments}",
    "composed for {instruments}",
    "written for {instruments}",
    "scored for {instruments}",
    "featuring the sounds of {instruments}",
    "bringing together {instruments}",
]

# Key descriptions - expanded to 14 options
KEY_TIME_TEMPLATES = [
    "in {key}",
    "in the key of {key}",
    "{key}",
    "set in {key}",
    "written in {key}",
    "composed in {key}",
    "arranged in {key}",
    "using {key}",
    "using {key} tonality",
    "framed in {key}",
    "based in {key}",
    "centered in {key}",
    "rooted in {key}",
    "in the tonality of {key}",
]

# Time signature - expanded to 12 options
TIME_SIG_TEMPLATES = [
    "in {time_signature}",
    "in {time_signature} time",
    "with a {time_signature} meter",
    "with {time_signature}",
    "using {time_signature}",
    "with a {time_signature} time signature",
    "set to {time_signature}",
    "in a {time_signature} meter",
    "{time_signature} time",
    "{time_signature} meter",
    "with {time_signature} timing",
    "featuring {time_signature}",
]

# Tempo descriptions - expanded to 18 options
TEMPO_TEMPLATES = [
    "at {bpm} BPM",
    "{bpm} BPM",
    "at a {tempo_word} {bpm} BPM",
    "moving at {bpm} BPM",
    "with a tempo of {bpm} BPM",
    "with a {tempo_word} tempo of {bpm} BPM",
    "paced at {bpm} BPM",
    "at a {tempo_word} pace ({bpm} BPM)",
    "flowing at {bpm} BPM",
    "progressing at {bpm} BPM",
    "driven at {bpm} BPM",
    "maintaining {bpm} BPM",
    "set to {bpm} BPM",
    "clocking in at {bpm} BPM",
    "running at {bpm} BPM",
    "with {bpm} beats per minute",
    "at approximately {bpm} BPM",
    "tempo: {bpm} BPM",
]

# Tempo descriptors - expanded to 10 options
TEMPO_DESCRIPTORS = [
    "giving it a {tempo_word} feel",
    "creating a {tempo_word} atmosphere",
    "with a {tempo_word} character",
    "resulting in a {tempo_word} vibe",
    "producing a {tempo_word} energy",
    "establishing a {tempo_word} mood",
    "conveying a {tempo_word} sense",
    "evoking a {tempo_word} quality",
    "maintaining a {tempo_word} pace",
    "delivering a {tempo_word} experience",
]

# Complexity descriptors - expanded options
OPTIONAL_DESCRIPTORS = [
    "The texture is {complexity}.",
    "It has a {complexity} arrangement.",
    "The piece features a {complexity} structure.",
    "The orchestration is {complexity}.",
    "The arrangement is {complexity}.",
    "It presents a {complexity} texture.",
    "The composition is {complexity}.",
    "The structure is {complexity}.",
]

COMPLEXITY_WORDS = {
    "simple": ["minimalist", "sparse", "clean", "straightforward", "uncluttered"],
    "moderate": ["balanced", "cohesive", "integrated", "well-structured", "organized"],
    "rich": ["intricate", "elaborate", "detailed", "sophisticated", "nuanced", "complex"],
    "complex": ["dense", "multifaceted", "layered", "intricate", "elaborate", "sophisticated"],
}


def pick_phrase(choices: List[str], **kwargs) -> str:
    return random.choice(choices).format(**kwargs)


def get_complexity_descriptor(adjective: str) -> str:
    """Get a complexity descriptor based on track adjective."""
    if "simple" in adjective or "minimal" in adjective:
        return random.choice(COMPLEXITY_WORDS["simple"])
    elif "complex" in adjective or "elaborate" in adjective:
        return random.choice(COMPLEXITY_WORDS["complex"])
    elif "rich" in adjective or "detailed" in adjective:
        return random.choice(COMPLEXITY_WORDS["rich"])
    else:
        return random.choice(COMPLEXITY_WORDS["moderate"])


def format_instrument_list(instruments: List[str]) -> str:
    """Format instrument list with proper grammar."""
    if len(instruments) == 0:
        return "various instruments"
    elif len(instruments) == 1:
        return instruments[0]
    elif len(instruments) == 2:
        return f"{instruments[0]} and {instruments[1]}"
    else:
        return f"{', '.join(instruments[:-1])}, and {instruments[-1]}"


def caption_from_meta(meta: Dict) -> str:
    """Generate highly varied caption with probabilistic element inclusion and 8 different structures."""
    
    # ===== STEP 1: Gather all possible elements =====
    instrument_list = meta["instrument_programs"]
    if not instrument_list:
        instrument_list = ["various orchestral sections"]
    instruments_str = format_instrument_list(instrument_list)
    
    ensemble_type = meta.get("ensemble_type", "")
    folder_complexity = meta.get("folder_complexity")
    key = meta["key"]
    time_sig = meta["time_signature"]
    bpm = int(meta["tempo"])
    tempo_desc = tempo_word(meta["tempo"])
    
    # Build adjective from ensemble/folder/track info
    if folder_complexity and ensemble_type:
        adjective = f"{folder_complexity} {ensemble_type}"
        article = "a"
    elif ensemble_type:
        adjective = ensemble_type
        article = "a" if ensemble_type[0] not in "aeiou" else "an"
    else:
        adjective = random.choice(meta["adjectives_tracks"])
        article = "an" if adjective[0] in "aeiou" else "a"
    
    # ===== STEP 2: Probabilistic element inclusion =====
    include_tempo = random.random() < 0.8  # 80%
    include_time_sig = random.random() < 0.8  # 80%
    include_ensemble = random.random() < 0.6 and (ensemble_type or folder_complexity)  # 60%
    include_complexity = random.random() < 0.5  # 50%
    include_tempo_desc = random.random() < 0.3  # 30%
    
    # ===== STEP 3: Select phrasing for each element =====
    piece_type = random.choice(PIECE_TYPES)
    instrument_phrase = pick_phrase(INSTRUMENT_INTRO_PHRASES, instruments=instruments_str)
    key_phrase = pick_phrase(KEY_TIME_TEMPLATES, key=key)
    
    tempo_phrase = None
    if include_tempo:
        if include_tempo_desc:
            tempo_phrase = pick_phrase(TEMPO_TEMPLATES, bpm=bpm, tempo_word=tempo_desc)
        else:
            # Use simpler tempo templates without tempo_word
            simple_tempo_templates = [t for t in TEMPO_TEMPLATES if "{tempo_word}" not in t]
            tempo_phrase = pick_phrase(simple_tempo_templates, bpm=bpm, tempo_word=tempo_desc)
    
    time_sig_phrase = None
    if include_time_sig:
        time_sig_phrase = pick_phrase(TIME_SIG_TEMPLATES, time_signature=time_sig)
    
    complexity_desc = None
    if include_complexity:
        complexity_word = get_complexity_descriptor(adjective)
        complexity_desc = pick_phrase(OPTIONAL_DESCRIPTORS, complexity=complexity_word)
    
    tempo_descriptor_phrase = None
    if include_tempo_desc and include_tempo:
        tempo_descriptor_phrase = pick_phrase(TEMPO_DESCRIPTORS, tempo_word=tempo_desc)
    
    # ===== STEP 4: Choose sentence structure (1-8) =====
    structure = random.randint(1, 8)
    
    if structure == 1:
        # STRUCTURE 1: Full descriptive (3 sentences)
        # "This is a classical string quartet piece featuring Violin, Viola, Cello, and Contrabass.
        #  It is in C major with a 4/4 meter.
        #  The tempo is at 120 BPM."
        opening = pick_phrase(OPENING_TEMPLATES, article=article, adjective=adjective, piece_type=piece_type)
        sent1 = f"{opening} {instrument_phrase}."
        
        if time_sig_phrase:
            sent2 = f"It is {key_phrase} {time_sig_phrase}."
        else:
            sent2 = f"It is {key_phrase}."
        
        if tempo_phrase and tempo_descriptor_phrase:
            sent3 = f"The tempo is {tempo_phrase}, {tempo_descriptor_phrase}."
        elif tempo_phrase:
            sent3 = f"The tempo is {tempo_phrase}."
        else:
            sent3 = ""
        
        caption = f"{sent1} {sent2}" + (f" {sent3}" if sent3 else "")
    
    elif structure == 2:
        # STRUCTURE 2: Compact descriptive (2 sentences)
        # "A contemporary jazz ensemble composition.
        #  It features Alto Sax, Piano, and Drums in D minor at 140 BPM in 4/4."
        opening = pick_phrase(OPENING_TEMPLATES, article=article, adjective=adjective, piece_type=piece_type)
        sent1 = f"{opening}."
        
        parts = [f"It features {instruments_str}", key_phrase]
        if tempo_phrase:
            parts.append(tempo_phrase)
        if time_sig_phrase:
            parts.append(time_sig_phrase)
        
        sent2 = " ".join(parts) + "."
        caption = f"{sent1} {sent2}"
    
    elif structure == 3:
        # STRUCTURE 3: Flowing single sentence
        # "A classical duo piece featuring Piano and Violin in C major, 4/4 time, moderate tempo at 120 BPM."
        opening = pick_phrase(OPENING_TEMPLATES, article=article, adjective=adjective, piece_type=piece_type)
        
        parts = [opening, instrument_phrase, key_phrase]
        if time_sig_phrase:
            parts.append(time_sig_phrase)
        if tempo_phrase:
            if include_tempo_desc:
                parts.append(f"{tempo_desc} tempo at {bpm} BPM")
            else:
                parts.append(tempo_phrase)
        
        caption = ", ".join(parts) + "."
    
    elif structure == 4:
        # STRUCTURE 4: Comma-separated list (concise)
        # "Piano and Violin, C major, 120 BPM, 4/4"
        parts = [instruments_str, key_phrase]
        if tempo_phrase:
            parts.append(tempo_phrase)
        if time_sig_phrase:
            parts.append(time_sig_phrase)
        
        caption = ", ".join(parts)
    
    elif structure == 5:
        # STRUCTURE 5: Key-first structure
        # "C major string quartet at 120 BPM in 4/4 with Violin, Viola, Cello, and Contrabass"
        parts = [key_phrase]
        if include_ensemble:
            parts.append(adjective)
        else:
            parts.append(piece_type)
        
        if tempo_phrase:
            parts.append(tempo_phrase)
        if time_sig_phrase:
            parts.append(time_sig_phrase)
        parts.append(f"with {instruments_str}")
        
        caption = " ".join(parts)
    
    elif structure == 6:
        # STRUCTURE 6: Tempo-first structure
        # "120 BPM contemporary trio in D minor featuring Piano, Bass, and Drums"
        parts = []
        if tempo_phrase:
            parts.append(tempo_phrase)
        if include_ensemble:
            parts.append(adjective)
        else:
            parts.append(piece_type)
        parts.append(key_phrase)
        parts.append(instrument_phrase)
        
        caption = " ".join(parts)
    
    elif structure == 7:
        # STRUCTURE 7: Instrument-first minimal
        # "Piano and Violin duo, C major, moderate tempo (120 BPM)"
        parts = [instruments_str]
        if include_ensemble and ensemble_type:
            parts[0] = f"{instruments_str} {ensemble_type}"
        
        parts.append(key_phrase)
        if tempo_phrase and include_tempo_desc:
            parts.append(f"{tempo_desc} tempo ({bpm} BPM)")
        elif tempo_phrase:
            parts.append(tempo_phrase)
        
        caption = ", ".join(parts)
    
    else:  # structure == 8
        # STRUCTURE 8: Mixed format (2 sentences, random combination)
        # "A string quartet in C major. Features Violin, Viola, Cello, and Contrabass at 120 BPM."
        if include_ensemble:
            sent1 = f"A {adjective} {key_phrase}."
        else:
            sent1 = f"A {piece_type} {key_phrase}."
        
        parts = [f"Features {instruments_str}"]
        if tempo_phrase:
            parts.append(tempo_phrase)
        if time_sig_phrase:
            parts.append(time_sig_phrase)
        
        sent2 = " ".join(parts) + "."
        caption = f"{sent1} {sent2}"
    
    # ===== STEP 5: Add optional complexity descriptor =====
    if complexity_desc:
        caption = f"{caption} {complexity_desc}"
    
    # ===== STEP 6: Ensure proper capitalization =====
    if caption and caption[0].islower():
        caption = caption[0].upper() + caption[1:]
    
    return caption


# ---------- Main pipeline ----------


def build_records(
    midi_root: Path,
    seed: int,
    max_variants: int = 3,
) -> List[Dict]:
    """Build caption records with 2-3 variants per MIDI file for maximum variety."""
    midi_paths = sorted(midi_root.rglob("*.mid"))
    print(f"Found {len(midi_paths)} MIDI files under {midi_root}")
    random.seed(seed)
    records: List[Dict] = []

    for p in midi_paths:
        # Store location relative to the SymphonyNet root
        rel = str(p.relative_to(midi_root))
        print(f"Processing {rel}...")
        meta = analyze_midi(p)
        
        # Generate 2-max_variants different captions for each MIDI file
        num_variants = random.randint(2, max_variants)
        for variant_idx in range(num_variants):
            # Each variant gets a different caption due to randomization
            caption = caption_from_meta(meta)
            record = {
                "location": rel,
                "caption": caption,
                "caption_variant": variant_idx,  # Track which variant this is
                "instrument_classes": meta["instrument_classes"],
                "key": meta["key"],
                "split": "train",  # placeholder, filled after shuffle
                "stage": "pretrain",
                "dataset": "symphonynet",
            }
            records.append(record)

    # Shuffle all records (including all variants) together
    random.shuffle(records)
    n = len(records)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)
    print(f"Total records: {n} ({len(midi_paths)} MIDI files × ~2.5 variants) -> train: {n_train}, val: {n_val}, test: {n - n_train - n_val}")
    
    for i, rec in enumerate(records):
        if i < n_train:
            rec["split"] = "train"
        elif i < n_train + n_val:
            rec["split"] = "validation"
        else:
            rec["split"] = "test"
    return records


def write_jsonl(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(records)} records to {output_path}")
    with output_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SymphonyNet captions with modular templates.")
    parser.add_argument("--root", default="data/symphonynet", help="Path to SymphonyNet MIDI root.")
    parser.add_argument("--output", default="captions/symphonynet_captions.json", help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling/splitting.")
    parser.add_argument("--max-variants", type=int, default=3, help="Maximum caption variants per MIDI file (default: 3).")
    args = parser.parse_args()

    midi_root = PROJECT_ROOT / args.root
    records = build_records(midi_root, seed=args.seed, max_variants=args.max_variants)
    write_jsonl(records, PROJECT_ROOT / args.output)
    print("Done.")


if __name__ == "__main__":
    main()
