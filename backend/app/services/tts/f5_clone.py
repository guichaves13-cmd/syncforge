"""F5-TTS voice cloning provider.

Requires `f5-tts` CLI installed (pip install f5-tts) and a 10-30s WAV sample
of the target voice. Returns audio + (approximate) sentence boundaries via
faster-whisper on the resulting audio.

For SyncForge, the sentence boundaries are needed by the SyncEngine to know
which clause maps to which time range. Edge-TTS gives them natively; F5-TTS
doesn't, so we re-transcribe with Whisper-turbo.
"""
from __future__ import annotations
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .edge import TTSResult


@dataclass
class F5Config:
    sample_wav: str = ""           # Path to 10-30s voice sample
    sample_text: str = ""          # What the sample says (improves cloning)
    model: str = "F5-TTS_v1"       # F5-TTS model variant
    speed: float = 1.0
    whisper_model: str = "turbo"   # for transcription
    timeout_s: int = 600


def synth_clone(script: str, voice_sample_wav: str, output_audio_path: str,
                cfg: F5Config | None = None) -> TTSResult:
    """Clone the voice in voice_sample_wav and synthesize `script`."""
    cfg = cfg or F5Config()
    cfg.sample_wav = cfg.sample_wav or voice_sample_wav
    if not Path(cfg.sample_wav).exists():
        raise FileNotFoundError(f"voice sample not found: {cfg.sample_wav}")

    if not shutil.which("f5-tts_infer-cli") and not shutil.which("f5-tts"):
        raise RuntimeError("f5-tts CLI not installed (pip install f5-tts)")

    # F5-TTS expects the script in a file
    with open(Path(output_audio_path).with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(script)

    cli = shutil.which("f5-tts_infer-cli") or shutil.which("f5-tts")
    cmd = [
        cli,
        "--model", cfg.model,
        "--ref_audio", cfg.sample_wav,
        "--ref_text", cfg.sample_text or "",
        "--gen_text", script,
        "--output_dir", str(Path(output_audio_path).parent),
        "--output_file", Path(output_audio_path).name,
        "--speed", str(cfg.speed),
    ]
    subprocess.run(cmd, capture_output=True, timeout=cfg.timeout_s)

    if not Path(output_audio_path).exists():
        raise RuntimeError("F5-TTS produced no output")

    # Transcribe to recover sentence boundaries
    sentences = _transcribe_sentences(output_audio_path, model=cfg.whisper_model)
    return TTSResult(audio_path=output_audio_path, sentences=sentences)


def _transcribe_sentences(audio_path: str, model: str = "turbo") -> list[dict]:
    """Use faster-whisper to recover (start, end, text) per sentence."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []
    wm = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _ = wm.transcribe(audio_path, vad_filter=True,
                                 word_timestamps=False, language=None)
    out: list[dict] = []
    for seg in segments:
        out.append({"start": float(seg.start), "end": float(seg.end),
                     "text": (seg.text or "").strip()})
    return out
