"""Multimodal embedder — text/image/video → shared vector space.

Primary: Gemini Embedding 2 (google-genai, 1024-dim)
Fallback: SigLIP-2 local (transformers, 1024-dim)

Frames de vídeo são sampleados (5 frames uniformemente espaçados) e
embeddados como imagens; vetor final = média.

Cache em disco (sha256(content) → vector) para evitar re-embedding.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass
class EmbedConfig:
    provider: str = "gemini"          # "gemini" | "siglip2"
    gemini_api_key: str = ""
    gemini_model: str = "models/gemini-embedding-002"
    siglip_model: str = "google/siglip2-large-patch16-512"
    cache_dir: Path = Path("storage/cache/embeddings")
    frames_per_video: int = 5
    use_cache: bool = True


class MultimodalEmbedder:
    def __init__(self, config: EmbedConfig):
        self.cfg = config
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._siglip_lazy = None  # loaded on first use

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        key = self._cache_key(f"text::{self.cfg.provider}::{text}")
        if self.cfg.use_cache and (v := self._load_cache(key)) is not None:
            return v
        if self.cfg.provider == "gemini":
            v = self._gemini_text(text)
        else:
            v = self._siglip_text(text)
        self._save_cache(key, v)
        return v

    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        key = self._cache_key(f"img::{self.cfg.provider}::{hashlib.sha256(image_bytes).hexdigest()}")
        if self.cfg.use_cache and (v := self._load_cache(key)) is not None:
            return v
        if self.cfg.provider == "gemini":
            v = self._gemini_image(image_bytes)
        else:
            v = self._siglip_image(image_bytes)
        self._save_cache(key, v)
        return v

    def embed_video(self, video_path: str) -> np.ndarray:
        """Sample N frames uniformemente, embeda cada um, retorna a média."""
        key = self._cache_key(f"vid::{self.cfg.provider}::{video_path}::{Path(video_path).stat().st_size}")
        if self.cfg.use_cache and (v := self._load_cache(key)) is not None:
            return v
        frames = self._sample_frames(video_path, self.cfg.frames_per_video)
        if not frames:
            raise RuntimeError(f"No frames sampled from {video_path}")
        vectors = [self.embed_image(f) for f in frames]
        avg = np.mean(np.stack(vectors), axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-9)  # re-normalize
        self._save_cache(key, avg)
        return avg

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity (assume L2-normalized inputs)."""
        return float(np.dot(a, b))

    # ─────────────────────────────────────────────────────────────────────
    # Gemini Embedding 2 backend
    # ─────────────────────────────────────────────────────────────────────

    def _gemini_text(self, text: str) -> np.ndarray:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.cfg.gemini_api_key)
        r = client.models.embed_content(
            model=self.cfg.gemini_model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=1024),
        )
        v = np.array(r.embeddings[0].values, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def _gemini_image(self, image_bytes: bytes) -> np.ndarray:
        # Gemini Embedding 2 accepts inline image via Part.from_bytes
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.cfg.gemini_api_key)
        r = client.models.embed_content(
            model=self.cfg.gemini_model,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
            config=types.EmbedContentConfig(output_dimensionality=1024),
        )
        v = np.array(r.embeddings[0].values, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    # ─────────────────────────────────────────────────────────────────────
    # SigLIP-2 local backend
    # ─────────────────────────────────────────────────────────────────────

    def _siglip(self):
        if self._siglip_lazy is None:
            from transformers import AutoProcessor, AutoModel
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            proc = AutoProcessor.from_pretrained(self.cfg.siglip_model)
            model = AutoModel.from_pretrained(self.cfg.siglip_model).to(device).eval()
            self._siglip_lazy = (proc, model, device, torch)
        return self._siglip_lazy

    def _siglip_text(self, text: str) -> np.ndarray:
        proc, model, device, torch = self._siglip()
        with torch.no_grad():
            inputs = proc(text=[text], return_tensors="pt", padding="max_length",
                          truncation=True).to(device)
            v = model.get_text_features(**inputs).cpu().numpy()[0]
        return v / (np.linalg.norm(v) + 1e-9)

    def _siglip_image(self, image_bytes: bytes) -> np.ndarray:
        proc, model, device, torch = self._siglip()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with torch.no_grad():
            inputs = proc(images=img, return_tensors="pt").to(device)
            v = model.get_image_features(**inputs).cpu().numpy()[0]
        return v / (np.linalg.norm(v) + 1e-9)

    # ─────────────────────────────────────────────────────────────────────
    # Frame sampling (ffmpeg)
    # ─────────────────────────────────────────────────────────────────────

    def _sample_frames(self, video_path: str, n: int) -> list[bytes]:
        """Sample n frames uniformly across the video; return jpeg bytes."""
        duration = self._probe_duration(video_path)
        if duration <= 0:
            return []
        timestamps = [duration * (i + 1) / (n + 1) for i in range(n)]
        frames: list[bytes] = []
        with tempfile.TemporaryDirectory() as td:
            for i, ts in enumerate(timestamps):
                out = Path(td) / f"f{i}.jpg"
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
                     "-vframes", "1", "-q:v", "3", "-vf", "scale=512:-2",
                     str(out)],
                    capture_output=True, timeout=20,
                )
                if out.exists() and out.stat().st_size > 1000:
                    frames.append(out.read_bytes())
        return frames

    @staticmethod
    def _probe_duration(path: str) -> float:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", path],
                capture_output=True, text=True, timeout=10,
            )
            return float((r.stdout or "0").strip() or 0)
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # Cache
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _load_cache(self, key: str) -> np.ndarray | None:
        p = self.cfg.cache_dir / f"{key}.npy"
        if p.exists():
            try:
                return np.load(p)
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, vec: np.ndarray) -> None:
        p = self.cfg.cache_dir / f"{key}.npy"
        try:
            np.save(p, vec.astype(np.float32))
        except Exception:
            pass
