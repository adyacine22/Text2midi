"""
Custom REMI tokenizer (Remi-Z) that allows for selective inclusion of
time signature and tempo tokens. This is useful for aligning with vocabularies
from pre-trained models.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Sequence

from miditok import TokenizerConfig
from symusic import Note, Score, TimeSignature, Track, Tempo


class TokenizedSequence:
    """Wrapper class to make tokenizer output compatible with miditok API."""
    def __init__(self, ids: List[int]):
        self.ids = ids
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self.ids[idx]


class RemiZTokenizer:
    """Minimal REMI-z tokenizer implemented with Miditok building blocks.

    - Track-first ordering within each bar (tracks sorted by avg pitch descending).
    - Within a track, notes are ordered by onset then pitch.
    - Tokens per note: instrument (once per track per bar) + onset + pitch + duration.
    - Optional bar-level time signature / tempo tokens are supported but off by default.
    - Quantization: 48th-note grid (ticks_per_quarter / 12).
    """

    def __init__(
        self,
        config: TokenizerConfig,
        include_time_signature: bool = False,
        include_tempo: bool = False,
    ):
        self.config = config
        self.include_time_signature = include_time_signature
        self.include_tempo = include_tempo

        self.vocab, self.token_to_id = self._build_vocab()
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    # Vocabulary -----------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _build_vocab(self) -> tuple[list[str], dict[str, int]]:
        vocab: list[str] = []
        config_specials = getattr(self.config, "special_tokens", None)
        if config_specials:
            specials: list[str] = []
            for tok in config_specials:
                if tok.endswith("_None") or "-" in tok:
                    specials.append(tok)
                else:
                    specials.append(f"{tok}_None")
        else:
            specials = ["PAD_None", "BOS_None", "EOS_None", "MASK_None"]
        vocab.extend(specials)

        # Instrument (0-127 are programs, 128 is drum set as in Table 6)
        if self.config.use_programs:
            max_prog = max(p for p in self.config.programs if p >= 0)
            vocab.extend([f"i-{i}" for i in range(max_prog + 1)])
            # Ensure drum track token exists exactly once
            vocab.append("i-128")

        # Onset / position within bar
        num_positions = max(self.config.beat_res.values()) * 4  # Assuming 4/4 time
        vocab.extend([f"o-{i}" for i in range(num_positions)])

        # Pitch (0-127 melodic, 128-255 drums)
        # This range is fixed to cover all possible MIDI pitches + the drum offset.
        vocab.extend([f"p-{i}" for i in range(256)])

        # Duration
        num_durations = max(self.config.beat_res.values()) * 4
        vocab.extend([f"d-{i}" for i in range(num_durations)])

        # End of bar
        vocab.append("b-1")

        if self.include_time_signature:
            # Define a reasonable number of time signature tokens
            # Common time signatures: 2/4, 3/4, 4/4, 5/4, 6/8, 7/8, 9/8, 12/8, etc.
            # We'll use 32 slots to cover common variations
            num_time_sigs = 32
            vocab.extend([f"s-{i}" for i in range(num_time_sigs)])
        if self.include_tempo:
            # Use the num_tempos from config (typically 32 or 49)
            vocab.extend([f"t-{i}" for i in range(self.config.num_tempos)])

        token_to_id = {tok: i for i, tok in enumerate(vocab)}
        return vocab, token_to_id

    def __len__(self) -> int:
        return len(self.vocab)

    def __getitem__(self, token: str) -> int:
        return self.token_to_id[token]

    # Encoding -------------------------------------------------------------------
    def __call__(self, score_or_path: str | Path | Score) -> TokenizedSequence:
        if isinstance(score_or_path, (str, Path)):
            score = Score(score_or_path)
        else:
            score = score_or_path
        ids = self.encode_score(score)
        return TokenizedSequence(ids)

    def encode_score(self, score: Score) -> List[str]:
        """Encode a symusic Score to a token list following REMI-z ordering."""
        self._ensure_time_sig(score)
        bar_ticks = self._bar_ticks(score)
        step_ticks = score.ticks_per_quarter / 12.0  # 48th-note grid

        # Collect notes per bar per track
        max_bar = 0
        per_track_bar: list[dict[int, list[Note]]] = []
        for track in score.tracks:
            bar_map: dict[int, list[Note]] = {}
            for n in track.notes:
                bar_idx = int(n.start // bar_ticks)
                max_bar = max(max_bar, bar_idx)
                bar_map.setdefault(bar_idx, []).append(n)
            per_track_bar.append(bar_map)

        tokens: list[str] = []
        for bar_idx in range(max_bar + 1):
            # Optional bar-level tokens
            if self.include_time_signature:
                ts_token = self._time_sig_token(score, bar_ticks)
                tokens.append(ts_token)
            if self.include_tempo:
                tempo_token = self._tempo_token(score)
                tokens.append(tempo_token)

            # Tracks present in this bar
            tracks_in_bar = []
            for track_idx, bar_map in enumerate(per_track_bar):
                if bar_idx in bar_map and len(bar_map[bar_idx]) > 0:
                    notes = bar_map[bar_idx]
                    avg_pitch = (
                        float(np.mean([n.pitch for n in notes]))
                        if not score.tracks[track_idx].is_drum
                        else -1.0
                    )
                    tracks_in_bar.append((track_idx, avg_pitch, notes))

            # Sort by average pitch descending to get zig-zag ordering
            tracks_in_bar.sort(key=lambda x: x[1], reverse=True)

            for track_idx, _, notes in tracks_in_bar:
                track = score.tracks[track_idx]
                instrument = track.program if not track.is_drum else 128
                tokens.append(f"i-{instrument}")

                # Order notes by onset then pitch
                notes_sorted = sorted(notes, key=lambda n: (n.start, n.pitch))
                for n in notes_sorted:
                    onset_pos = self._quantize_tick(n.start % bar_ticks, step_ticks)
                    num_positions = max(self.config.beat_res.values()) * 4
                    dur = self._quantize_tick(n.duration, step_ticks)
                    num_durations = max(self.config.beat_res.values()) * 4
                    pitch_val = n.pitch + 128 if track.is_drum else n.pitch
                    tokens.extend(
                        [
                            f"o-{min(onset_pos, num_positions - 1)}",
                            f"p-{min(pitch_val, 255)}",
                            f"d-{min(dur, num_durations - 1)}",
                        ]
                    )

            tokens.append("b-1")

        ids = [self.token_to_id[tok] for tok in tokens if tok in self.token_to_id]
        return ids

    def decode(self, ids: Sequence[int]) -> List[str]:
        """Lightweight decode to token strings (note: does not reconstruct MIDI)."""
        return [self.id_to_token.get(i, "<unk>") for i in ids]

    def tokens_to_score(self, tokens: List[str], default_program: int = 0) -> Score:
        """
        Convert a sequence of REMI-z tokens back into a symusic.Score object.
        Handles flexible token order (o-p-d or p-d-o) and missing instrument tokens.
        """
        score = Score(480)  # Standard ticks per quarter
        tracks: Dict[int, Track] = {}

        current_bar = 0
        # Initialize with default program if no instrument token is found
        current_instrument_program = default_program

        bar_ticks = score.ticks_per_quarter * 4  # Assuming 4/4 time signature
        step_ticks = score.ticks_per_quarter / 12.0  # 48th-note grid

        # State tracking
        current_tick = 0
        current_bar_ticks = bar_ticks # Default 4/4
        
        # State buffer for note attributes
        current_note = {"pitch": None, "duration": None, "onset": None}

        def flush_note():
            nonlocal current_note
            if (
                current_note["pitch"] is not None
                and current_note["duration"] is not None
                and current_note["onset"] is not None
            ):
                onset_val = current_note["onset"]
                pitch_val = current_note["pitch"]
                dur_val = current_note["duration"]

                # Calculate start tick based on current bar start + onset
                # Note: onset_val is quantized step index.
                start_tick = current_tick + (onset_val * step_ticks)
                duration_ticks = dur_val * step_ticks

                is_drum_note = current_instrument_program == 128
                pitch = pitch_val - 128 if is_drum_note and pitch_val >= 128 else pitch_val

                # Ensure pitch is within valid MIDI range (0-127)
                final_pitch = int(pitch)
                final_pitch = max(0, min(127, final_pitch))

                note = Note(
                    time=int(start_tick),
                    duration=int(duration_ticks),
                    pitch=final_pitch,
                    velocity=int(80),
                )

                # Lazily create track if it does not exist yet
                if current_instrument_program not in tracks:
                    is_drum = current_instrument_program == 128
                    prog = 0 if is_drum else current_instrument_program
                    tracks[current_instrument_program] = Track(
                        program=prog,
                        is_drum=is_drum,
                        name=f"track_{current_instrument_program}",
                    )

                tracks[current_instrument_program].notes.append(note)

                # Reset buffer
                current_note = {"pitch": None, "duration": None, "onset": None}

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.startswith("i-"):
                flush_note()  # Flush any pending note
                try:
                    current_instrument_program = int(token.split("-")[1])
                    if current_instrument_program not in tracks:
                        is_drum = current_instrument_program == 128
                        prog = 0 if is_drum else current_instrument_program
                        tracks[current_instrument_program] = Track(
                            program=prog,
                            is_drum=is_drum,
                            name=f"track_{current_instrument_program}",
                        )
                except (ValueError, IndexError):
                    pass
                i += 1

            elif token.startswith("o-"):
                try:
                    val = int(token.split("-")[1])
                    current_note["onset"] = val
                    flush_note()  # Check if complete
                except (ValueError, IndexError):
                    pass
                i += 1

            elif token.startswith("p-"):
                try:
                    val = int(token.split("-")[1])
                    current_note["pitch"] = val
                    flush_note()  # Check if complete
                except (ValueError, IndexError):
                    pass
                i += 1

            elif token.startswith("d-"):
                try:
                    val = int(token.split("-")[1])
                    current_note["duration"] = val
                    flush_note()  # Check if complete
                except (ValueError, IndexError):
                    pass
                i += 1

            elif token.startswith("b-"):
                flush_note()  # Flush any pending note
                current_bar += 1
                current_tick += current_bar_ticks # Advance time by current bar length
                # Do NOT reset instrument program here, persist it across bars
                # unless explicit i-x token changes it.
                i += 1

            elif token.startswith("s-"):
                # Time Signature
                try:
                    ts_id = int(token.split("-")[1])
                    # Reverse mapping from _time_sig_token
                    # This is a bit tricky as the mapping is many-to-one in the "uncommon" case.
                    # But for common ones we can map back.
                    # For now, let's just support the common ones we defined.
                    
                    # Reconstruct the map
                    common_time_sigs_inv = {
                        0: (4, 4), 1: (3, 4), 2: (2, 4), 3: (6, 8), 4: (5, 4),
                        5: (7, 8), 6: (9, 8), 7: (12, 8), 8: (2, 2), 9: (3, 8), 10: (6, 4)
                    }
                    
                    if ts_id in common_time_sigs_inv:
                        num, den = common_time_sigs_inv[ts_id]
                        # Calculate time in ticks
                        # We need to know the current bar start time.
                        # But wait, bar_ticks is currently fixed to 4/4!
                        # If we change time signature, bar_ticks changes for FUTURE bars.
                        # But the current bar's start time depends on PAST bars' lengths.
                        
                        # This simple decoder assumes fixed bar length (line 205: bar_ticks = score.ticks_per_quarter * 4).
                        # To support variable time signatures, we need to track absolute time properly.
                        
                        # Let's update the TimeSignature in the score
                        # We'll place it at the start of the current bar.
                        # Note: This simple implementation might not handle complex mixed meter perfectly
                        # if we don't track the accumulated ticks from previous bars correctly.
                        
                        # For now, let's just add the event.
                        # Correct logic requires tracking `current_tick` instead of `current_bar * fixed_bar_ticks`.
                        
                        # Update current_bar_ticks for FUTURE bars (starting from the *next* b-1 token)
                        # Wait, if s-X is at the start of the CURRENT bar, it affects the CURRENT bar's length.
                        # Yes, s-X appears at the start of the bar.
                        # So we should update current_bar_ticks immediately.
                        
                        current_bar_ticks = score.ticks_per_quarter * (num * 4 / den)
                        
                        # Add TimeSignature event to score
                        score.time_signatures.append(TimeSignature(int(current_tick), num, den))
                    
                except (ValueError, IndexError):
                    pass
                i += 1

            elif token.startswith("t-"):
                # Tempo
                try:
                    tempo_id = int(token.split("-")[1])
                    # Reverse mapping from _tempo_token
                    tempo_bins = np.linspace(
                        self.config.tempo_range[0],
                        self.config.tempo_range[1],
                        num=self.config.num_tempos, # Use config value
                        endpoint=True,
                    )
                    if 0 <= tempo_id < len(tempo_bins):
                        qpm = float(tempo_bins[tempo_id])
                        # Add Tempo event to score
                        from symusic import Tempo # Ensure imported if not top-level, but it is.
                        score.tempos.append(Tempo(int(current_tick), qpm))
                        
                except (ValueError, IndexError):
                    pass
                i += 1

            else:
                # Ignore special tokens
                i += 1

        for track in tracks.values():
            score.tracks.append(track)

        return score

    # Helpers --------------------------------------------------------------------
    @staticmethod
    def _quantize_tick(ticks: float, step: float) -> int:
        if step <= 0:
            return 0
        return int(round(ticks / step))

    @staticmethod
    def _ensure_time_sig(score: Score) -> None:
        if len(score.time_signatures) == 0:
            # Default to 4/4 at tick 0 if missing
            score.time_signatures.append(TimeSignature(0, 4, 4))

    @staticmethod
    def _bar_ticks(score: Score) -> float:
        ts = score.time_signatures[0]
        # bar length = quarter_note_ticks * (numerator * 4 / denominator)
        return score.ticks_per_quarter * (ts.numerator * 4 / ts.denominator)

    def _time_sig_token(self, score: Score, bar_ticks: float) -> str:
        ts = score.time_signatures[0]
        
        # Map common time signatures to specific IDs (0-31)
        common_time_sigs = {
            (4, 4): 0,   # Most common
            (3, 4): 1,   # Waltz
            (2, 4): 2,   # March
            (6, 8): 3,   # Compound duple
            (5, 4): 4,   # Take Five
            (7, 8): 5,   # Irregular
            (9, 8): 6,   # Compound triple
            (12, 8): 7,  # Compound quadruple
            (2, 2): 8,   # Cut time
            (3, 8): 9,   # Simple triple
            (6, 4): 10,  # Compound duple (alternative)
        }
        
        key = (ts.numerator, ts.denominator)
        if key in common_time_sigs:
            ts_id = common_time_sigs[key]
        else:
            # Hash uncommon time signatures to remaining slots (11-31)
            ts_id = 11 + ((ts.numerator * 16 + ts.denominator) % 21)
        
        return f"s-{ts_id}"

    def _tempo_token(self, score: Score) -> str:
        if score.tempos:
            qpm = score.tempos[0].qpm
        else:
            qpm = 120.0
        tempo_bins = np.linspace(
            self.config.tempo_range[0],
            self.config.tempo_range[1],
            num=49,
            endpoint=True,
        )
        tempo_id = int(np.argmin(np.abs(tempo_bins - qpm)))
        return f"t-{tempo_id}"


def run_inline_test() -> None:
    """Smoke test for REMI-z ordering and quantization."""
    # Config values mirror build_vocab_remi defaults
    from miditok import TokenizerConfig

    beat_res = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
    cfg = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res=beat_res,
        num_velocities=32,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_chords=False,
        use_rests=False,
        use_tempos=False,
        use_time_signatures=False,
        use_programs=True,
        num_tempos=32,
        tempo_range=(40, 250),
    )
    tokenizer = RemiZTokenizer(cfg)

    score = Score(480)
    score.time_signatures.append(TimeSignature(0, 4, 4))

    lead = Track(name="lead", program=0, is_drum=False)
    lead.notes.append(Note(time=0, duration=480, pitch=72, velocity=90))
    lead.notes.append(Note(time=960, duration=240, pitch=74, velocity=88))

    drum = Track(name="drums", program=0, is_drum=True)
    drum.notes.append(Note(time=0, duration=240, pitch=35, velocity=100))

    score.tracks.append(lead)
    score.tracks.append(drum)

    seq = tokenizer.encode_score(score)
    expected_tokens = [
        "i-0",
        "o-0",
        "p-72",
        "d-12",
        "o-24",
        "p-74",
        "d-6",
        "i-128",
        "o-0",
        "p-163",
        "d-6",
        "b-1",
    ]
    print("Expected tokens:", expected_tokens)
    print("Actual tokens  :", seq)
    assert seq == expected_tokens, f"Unexpected tokens: {seq}"
    assert len(seq) == len(expected_tokens)
    print("Test passed. Token count:", len(seq))


__all__ = ["RemiZTokenizer", "run_inline_test"]
