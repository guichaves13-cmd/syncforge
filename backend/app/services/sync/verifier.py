"""Vision verifier — Gemini 2.5 Pro Vision watches sampled frames of a candidate
and returns relevance + anachronism + off-topic flags.

LLM-as-judge: aprovação requer score ≥70 E !anachronism E !off_topic.
"""
from __future__ import annotations
import base64
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .ranker import Candidate


_VERIFY_PROMPT = """You are a senior video director auditing whether a candidate video clip semantically MATCHES a narration passage.

NARRATION PASSAGE (what the viewer is hearing):
\"\"\"{clause}\"\"\"

CONTEXT:
- Topic: {topic}
- Era: {era}
- Mood: {mood}

You will see {n_frames} frames uniformly sampled from the candidate clip.

Judge:
1) RELEVANCE — does the clip visually support the narration? (0-100)
2) ANACHRONISM — does the clip show wrong era/setting? (modern phones in 1920s narration, etc.)
3) OFF_TOPIC — does the clip show something unrelated (random product, irrelevant scene)?
4) Quality concerns (text overlays, watermarks, low resolution, weird crops).

Respond ONLY in JSON, no prose:
{{
  "relevance_score": 0-100 integer,
  "description": "what the frames actually show (1 sentence)",
  "anachronism": true/false,
  "off_topic": true/false,
  "quality_issues": ["watermark", "text overlay", ...] or [],
  "approved": true/false,
  "rationale": "1-sentence why approved or rejected"
}}

approved = (relevance_score >= 70) AND (anachronism == false) AND (off_topic == false)
"""


@dataclass
class VerifyConfig:
    provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"
    frames_per_clip: int = 8
    min_score: int = 70
    timeout_s: int = 45


class VisionVerifier:
    def __init__(self, config: VerifyConfig):
        self.cfg = config

    def verify(self, candidate: Candidate, clause: str, topic: str = "",
               era: str = "modern", mood: str = "neutral") -> dict:
        """Returns {approved, relevance_score, description, ..., raw}.
        Mutates candidate.vision and candidate.vision_reason."""
        if not candidate.local_path or not Path(candidate.local_path).exists():
            return self._reject(candidate, "no local path")
        frames = self._sample_frames(candidate.local_path, self.cfg.frames_per_clip)
        if not frames:
            return self._reject(candidate, "no frames sampled")
        prompt = _VERIFY_PROMPT.format(
            clause=clause, topic=topic or "general",
            era=era or "modern", mood=mood or "neutral",
            n_frames=len(frames),
        )
        try:
            data = self._call_gemini(prompt, frames)
        except Exception as e:
            return self._reject(candidate, f"vision error: {e!s}")
        candidate.vision = float(data.get("relevance_score", 0) or 0)
        candidate.vision_reason = (data.get("rationale") or data.get("description") or "")[:200]
        return data

    # ─────────────────────────────────────────────────────────────────
    # Gemini call
    # ─────────────────────────────────────────────────────────────────

    def _call_gemini(self, prompt: str, frames: list[bytes]) -> dict:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.cfg.gemini_api_key)
        parts = [types.Part.from_text(prompt)] + [
            types.Part.from_bytes(data=f, mime_type="image/jpeg") for f in frames
        ]
        resp = client.models.generate_content(
            model=self.cfg.gemini_model,
            contents=types.Content(role="user", parts=parts),
            config=types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=400,
                response_mime_type="application/json",
            ),
        )
        text = resp.text or ""
        return _parse_json_lenient(text)

    # ─────────────────────────────────────────────────────────────────
    # Frame sampling (same as embedder but inline for separation)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _sample_frames(video_path: str, n: int) -> list[bytes]:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", video_path],
                capture_output=True, text=True, timeout=10,
            )
            duration = float((r.stdout or "0").strip() or 0)
        except Exception:
            return []
        if duration <= 0:
            return []
        timestamps = [duration * (i + 1) / (n + 1) for i in range(n)]
        out: list[bytes] = []
        with tempfile.TemporaryDirectory() as td:
            for i, ts in enumerate(timestamps):
                p = Path(td) / f"f{i}.jpg"
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
                     "-vframes", "1", "-q:v", "3", "-vf", "scale=512:-2",
                     str(p)],
                    capture_output=True, timeout=20,
                )
                if p.exists() and p.stat().st_size > 1000:
                    out.append(p.read_bytes())
        return out

    @staticmethod
    def _reject(c: Candidate, reason: str) -> dict:
        c.vision = 0
        c.vision_reason = reason[:200]
        return {"approved": False, "relevance_score": 0, "rationale": reason,
                "anachronism": False, "off_topic": False, "quality_issues": []}


def _parse_json_lenient(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise
