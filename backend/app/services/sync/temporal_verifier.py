"""Temporal Vision verifier — Gemini 2.5 Pro Video understanding mode.

Legacy verifier (verifier.py) samples 8 frames and sends them as static
images. Result: it can confirm a paddle is PRESENT but not whether it's
being USED (mid-rally vs. sitting on a shelf).

This module uploads the actual video clip (≤20s, ≤20MB) and asks Gemini
to judge motion, action match, and temporal coherence. Use this when:
  • The narration describes a specific action ('serving', 'cooking')
  • You're filtering YouTube clips (more likely to be miscategorised)
  • You're paying for higher quality

Falls back to frame-based verifier on:
  • Clip exceeds 20s / 20MB and ffmpeg trim fails
  • Gemini File API rejects the upload
  • Network / quota errors

Cost: one Gemini call per clip, ~$0.004 per 8-second clip.
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .ranker import Candidate


_MAX_INLINE_BYTES = 20_000_000     # 20 MB hard limit
_MAX_DURATION_S = 20.0             # 20 s hard limit


_TEMPORAL_PROMPT = """You are a senior video director scrutinising a candidate b-roll clip for a narrated video.

NARRATION (what the viewer is hearing):
\"\"\"{clause}\"\"\"

CONTEXT:
- Topic: {topic}
- Era: {era}
- Mood: {mood}
- Required action: {action}

You will receive an actual video clip. Watch the MOTION, not just stills.

Judge:
1) ACTION_MATCH — does the visible motion match the narration? (rally vs. static shot, cooking vs. ingredients on shelf, etc.)
2) RELEVANCE — does the clip semantically support the narration? (0-100)
3) ANACHRONISM — wrong era/setting? (modern phones in 1920s narration, etc.)
4) OFF_TOPIC — completely unrelated to narration?
5) QUALITY — text overlays, watermarks, low motion (frozen), shaky cam, weird crops?

Respond ONLY in JSON, no prose:
{{
  "relevance_score": 0-100 integer,
  "action_match_score": 0-100 integer,
  "description": "<what the clip actually shows in motion, 1 sentence>",
  "detected_action": "<the dominant verb-phrase, e.g. 'serving ball', 'caramelising onions'>",
  "anachronism": true/false,
  "off_topic": true/false,
  "quality_issues": ["<watermark>", "<frozen frame>", ...] or [],
  "approved": true/false,
  "rationale": "<1 sentence why>"
}}

approved = (relevance_score >= 70) AND (action_match_score >= 60) AND (anachronism == false) AND (off_topic == false)
"""


@dataclass
class TemporalVerifyConfig:
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"
    max_clip_duration: float = _MAX_DURATION_S
    max_clip_bytes: int = _MAX_INLINE_BYTES
    min_relevance: int = 70
    min_action_match: int = 60
    upload_poll_s: int = 1
    upload_max_s: int = 30
    timeout_s: int = 60
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


class TemporalVisionVerifier:
    def __init__(self, config: TemporalVerifyConfig):
        self.cfg = config

    def verify(self, candidate: Candidate, clause: str,
               topic: str = "", era: str = "modern", mood: str = "neutral",
               required_action: str = "") -> dict:
        """Returns {approved, relevance_score, action_match_score, ..., raw}.
        Mutates candidate.vision and candidate.vision_reason."""
        if not candidate.local_path or not Path(candidate.local_path).exists():
            return self._reject(candidate, "no local path")

        try:
            mp4 = self._prepare_clip(candidate.local_path)
        except Exception as e:
            return self._reject(candidate, f"prep failed: {e!s}")

        prompt = _TEMPORAL_PROMPT.format(
            clause=clause, topic=topic or "general",
            era=era or "modern", mood=mood or "neutral",
            action=required_action or "any plausible action",
        )

        try:
            data = self._call_gemini_video(prompt, mp4)
        except Exception as e:
            return self._reject(candidate, f"vision error: {e!s}")
        finally:
            if mp4 != candidate.local_path:
                try: Path(mp4).unlink(missing_ok=True)
                except: pass

        relevance = int(data.get("relevance_score", 0) or 0)
        action = int(data.get("action_match_score", 0) or 0)
        candidate.vision = float(relevance)   # primary score remains relevance
        rationale = data.get("rationale") or data.get("description") or ""
        candidate.vision_reason = f"[T] rel={relevance} act={action} | {rationale}"[:200]

        # Stricter approve gate: BOTH scores must clear thresholds
        approved = (
            relevance >= self.cfg.min_relevance
            and action >= self.cfg.min_action_match
            and not data.get("anachronism")
            and not data.get("off_topic")
        )
        data["approved"] = approved
        return data

    # ── ffmpeg prep ─────────────────────────────────────────────────────

    def _prepare_clip(self, src: str) -> str:
        """Trim+downscale the clip if it exceeds limits. Returns a path
        to a Gemini-safe MP4 (≤20s, ≤20MB)."""
        duration = self._probe_duration(src)
        size = Path(src).stat().st_size
        if duration <= self.cfg.max_clip_duration and size <= self.cfg.max_clip_bytes:
            return src

        # Take a midframe-centered window of max_clip_duration
        clip_seconds = min(duration, self.cfg.max_clip_duration)
        start = max(0.0, (duration - clip_seconds) / 2.0)
        # mkstemp leaks an open fd on Windows → close it explicitly so
        # ffmpeg can overwrite the file AND callers can delete it.
        fd, tmp_path = tempfile.mkstemp(prefix="syncforge_temporal_", suffix=".mp4")
        os.close(fd)
        tmp = Path(tmp_path)
        # Conservative re-encode: 480p, 24fps, CRF 26 → comfortably <20MB for ≤20s
        subprocess.run(
            [self.cfg.ffmpeg, "-y", "-ss", f"{start:.2f}", "-i", src,
             "-t", f"{clip_seconds:.2f}",
             "-vf", "scale=854:480:force_original_aspect_ratio=decrease",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
             "-an", "-movflags", "+faststart", str(tmp)],
            capture_output=True, timeout=120,
        )
        if not tmp.exists() or tmp.stat().st_size < 5_000:
            raise RuntimeError("ffmpeg trim produced empty output")
        if tmp.stat().st_size > self.cfg.max_clip_bytes:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"trimmed clip still too big ({tmp.stat().st_size} bytes)")
        return str(tmp)

    def _probe_duration(self, path: str) -> float:
        try:
            r = subprocess.run(
                [self.cfg.ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", path],
                capture_output=True, text=True, timeout=10,
            )
            return float((r.stdout or "0").strip() or 0)
        except Exception:
            return 0.0

    # ── Gemini call ─────────────────────────────────────────────────────

    def _call_gemini_video(self, prompt: str, mp4_path: str) -> dict:
        """Upload video → ask model → parse JSON response."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.cfg.gemini_api_key)

        # 1) Upload file
        uploaded = client.files.upload(file=mp4_path)
        # Wait for ACTIVE state (Gemini transcodes server-side)
        waited = 0
        while waited < self.cfg.upload_max_s:
            state = getattr(uploaded, "state", "")
            state_name = getattr(state, "name", str(state))
            if state_name == "ACTIVE":
                break
            if state_name == "FAILED":
                raise RuntimeError("Gemini upload state=FAILED")
            time.sleep(self.cfg.upload_poll_s)
            waited += self.cfg.upload_poll_s
            uploaded = client.files.get(name=uploaded.name)

        # 2) Generate content with the uploaded video + text prompt
        resp = client.models.generate_content(
            model=self.cfg.gemini_model,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=500,
                response_mime_type="application/json",
            ),
        )

        # 3) Best-effort delete the uploaded file (don't crash if it fails)
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

        return _parse_json_lenient(resp.text or "")

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _reject(c: Candidate, reason: str) -> dict:
        c.vision = 0
        c.vision_reason = reason[:200]
        return {"approved": False, "relevance_score": 0,
                "action_match_score": 0, "rationale": reason,
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
