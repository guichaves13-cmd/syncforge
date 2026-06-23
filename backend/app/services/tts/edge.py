"""Edge-TTS — synthesize audio + capture word-level timing via SentenceBoundary.

Returns: (audio_path, sentence_boundaries) where each boundary is
{start, end, text} in seconds. The sync pipeline will turn these into clauses.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TTSResult:
    audio_path: str
    sentences: list[dict]   # [{start, end, text}]


async def synth(script: str, voice: str, output_audio_path: str,
                rate: str = "-5%") -> TTSResult:
    import edge_tts
    communicate = edge_tts.Communicate(script, voice, rate=rate)
    sentences: list[dict] = []
    with open(output_audio_path, "wb") as f:
        async for chunk in communicate.stream():
            t = chunk.get("type")
            if t == "audio":
                f.write(chunk["data"])
            elif t == "SentenceBoundary":
                offs = chunk.get("offset", 0) / 1e7   # 100ns → s
                dur = chunk.get("duration", 0) / 1e7
                txt = chunk.get("text", "") or ""
                sentences.append({
                    "start": float(offs),
                    "end": float(offs + dur),
                    "text": txt.strip(),
                })
    return TTSResult(audio_path=output_audio_path, sentences=sentences)


def synth_sync(script: str, voice: str, output_audio_path: str,
               rate: str = "-5%") -> TTSResult:
    return asyncio.run(synth(script, voice, output_audio_path, rate))
