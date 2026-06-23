"""Runway Gen-4 generative fallback — alternative to Veo 3.

API: https://docs.dev.runwayml.com/api/
Cost: ~$0.05/sec at 720p (so ~$0.30 for 6s clip).
"""
from __future__ import annotations
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

from ..sync.pipeline import Intent, NarrationClause
from ..sync.ranker import Candidate


class RunwayGen4Generator:
    name = "runway_gen4"
    BASE = "https://api.dev.runwayml.com/v1"

    def __init__(self, api_key: str = "", duration: int = 5,
                 ratio: str = "1280:720", poll_interval: int = 5,
                 poll_max: int = 180):
        self.api_key = api_key or os.getenv("RUNWAY_API_KEY", "")
        self.duration = duration
        self.ratio = ratio
        self.poll_interval = poll_interval
        self.poll_max = poll_max

    def generate(self, intent: Intent, clause: NarrationClause,
                 output_dir: Path) -> Optional[Candidate]:
        if not self.api_key:
            return None
        prompt = self._build_prompt(intent, clause)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Runway-Version": "2024-11-06",
            "Content-Type": "application/json",
        }
        try:
            r = requests.post(
                f"{self.BASE}/text_to_video",
                headers=headers,
                json={
                    "model": "gen4_turbo",
                    "promptText": prompt,
                    "ratio": self.ratio,
                    "duration": self.duration,
                },
                timeout=20,
            )
            r.raise_for_status()
            task_id = r.json().get("id")
            if not task_id:
                return None
            # Poll
            waited = 0
            while waited < self.poll_max:
                time.sleep(self.poll_interval)
                waited += self.poll_interval
                s = requests.get(f"{self.BASE}/tasks/{task_id}",
                                 headers=headers, timeout=10)
                data = s.json()
                status = data.get("status")
                if status == "SUCCEEDED":
                    out_url = (data.get("output") or [None])[0]
                    if not out_url:
                        return None
                    dest = output_dir / f"runway_{uuid.uuid4().hex[:8]}.mp4"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with requests.get(out_url, stream=True, timeout=60) as g:
                        g.raise_for_status()
                        with open(dest, "wb") as f:
                            for chunk in g.iter_content(chunk_size=1 << 16):
                                f.write(chunk)
                    return Candidate(
                        source=self.name,
                        source_id=dest.stem,
                        url=f"generated://{dest.name}",
                        title=f"[Runway Gen-4] {prompt[:80]}",
                        description=prompt,
                        local_path=str(dest),
                        duration=float(self.duration),
                        vision=85,
                        vision_reason="generated from intent",
                    )
                if status == "FAILED":
                    return None
            return None
        except Exception:
            return None

    @staticmethod
    def _build_prompt(intent: Intent, clause: NarrationClause) -> str:
        bits = []
        if intent.key_visuals:
            bits.append(", ".join(intent.key_visuals[:2]))
        elif intent.main_entity:
            bits.append(intent.main_entity)
        if intent.action:
            bits.append(intent.action)
        if intent.location:
            bits.append(intent.location)
        mood = intent.mood or "cinematic"
        return f"{', '.join(filter(None, bits))}, {mood}, realistic".strip(", ")
