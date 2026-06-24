"""End-to-end runner — call from FastAPI background job.

Steps:
  1. Script  (assumed pre-written; for now we just use the title as a one-line script)
  2. TTS     (Edge-TTS → audio + sentence boundaries)
  3. Sync    (SyncEngine over the sentence boundaries)
  4. Render  (ffmpeg compose + karaoke burn)
  5. Cleanup (delete intermediates)

This is intentionally minimal so the wiring is visible end-to-end.
"""
from __future__ import annotations
import gc
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable


def _force_delete(p: Path, retries: int = 6) -> bool:
    """Delete a file or dir, retrying on Windows-style transient lock errors.
    For directories, walks bottom-up so a single locked file doesn't block the rest.
    Returns True if path no longer exists at the end.
    """
    if not p.exists():
        return True
    for attempt in range(retries):
        try:
            if p.is_dir():
                # Walk bottom-up: delete files first, then dirs
                for entry in sorted(p.rglob("*"), key=lambda x: -len(x.parts)):
                    try:
                        if entry.is_file() or entry.is_symlink():
                            entry.unlink(missing_ok=True)
                        elif entry.is_dir():
                            entry.rmdir()
                    except (PermissionError, OSError):
                        pass
                p.rmdir()
            else:
                p.unlink(missing_ok=True)
            if not p.exists():
                return True
        except (PermissionError, OSError):
            pass
        # Exponential backoff + GC to release Python-side handles
        gc.collect()
        time.sleep(0.5 * (attempt + 1))
    return not p.exists()


def _cleanup_intermediates(final_path: str, paths: list[Path]) -> None:
    """Delete intermediates, never touching the FINAL. Windows-safe with retries.
    Brief pause + double GC first so post-ffmpeg handles get fully released."""
    final_resolved = Path(final_path).resolve()
    gc.collect()
    time.sleep(1.0)  # let Windows release ffmpeg-child handles + Defender scan
    gc.collect()
    for p in paths:
        if not p.exists():
            continue
        try:
            if p.resolve() == final_resolved:
                continue
        except Exception:
            continue
        _force_delete(p)

from .generate.factory import build_generative_fn
from .llm.chain import LLMKeys
from .llm.intent import extract_intent
from .llm.translate import translate_to_english
from .render.composer import compose, ComposeConfig
from .stock.factory import build_sources, download_by_source
from .subtitles.karaoke import build_ass, burn
from .sync import (
    EmbedConfig, MultimodalEmbedder,
    MultiSignalRanker,
    MultiSourceRetriever, RetrieverConfig,
    SyncEngine, SyncEngineConfig,
    VerifyConfig, VisionVerifier,
    NarrationClause,
)
from .sync.dedup import DedupStore, phash_video
from .tts.edge import synth_sync


def run_pipeline(
    script_text: str,
    voice: str,
    title: str,
    theme: str,
    output_dir: Path,
    *,
    enable_vision: bool = True,
    enable_embeddings: bool = True,
    enable_generative_fallback: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    progress = progress or (lambda _e: None)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:10]

    # ── 1. TTS + sentence boundaries (no separate script step yet) ─────
    progress({"event": "step", "step": "tts", "status": "start"})
    audio_path = str(output_dir / f"{job_id}_audio.mp3")
    tts = synth_sync(script_text, voice, audio_path)
    progress({"event": "step", "step": "tts", "status": "done",
              "sentences": len(tts.sentences)})

    if not tts.sentences:
        return {"ok": False, "error": "no sentence boundaries from TTS"}

    clauses = [NarrationClause(start=s["start"], end=s["end"], text=s["text"])
               for s in tts.sentences if s.get("text")]

    # ── 2. Build SyncEngine ────────────────────────────────────────────
    progress({"event": "step", "step": "build_engine", "status": "start"})
    searches, downloads = build_sources()
    keys = LLMKeys.from_env()

    retriever = MultiSourceRetriever(
        config=RetrieverConfig(max_per_source=8, max_total_pool=60, workers=6),
        sources=searches,
    )

    embedder = None
    if enable_embeddings:
        embedder = MultimodalEmbedder(EmbedConfig(
            provider="gemini" if keys.gemini else "siglip2",
            gemini_api_key=keys.gemini,
        ))

    verifier = None
    if enable_vision and keys.gemini:
        verifier = VisionVerifier(VerifyConfig(gemini_api_key=keys.gemini))

    ranker = MultiSignalRanker(embedder=embedder)
    dedup = DedupStore()

    def _intent_fn(clause: NarrationClause, theme: str):
        intent = extract_intent(clause, theme, keys)
        # Auto-translate any non-English queries → improves stock-lib hit rate
        intent.queries = [translate_to_english(q, keys) for q in intent.queries]
        return intent

    def _download_fn(c, out_dir):
        path = download_by_source(c, out_dir, downloads)
        if not path:
            return None
        # Perceptual-hash anti-duplicate
        try:
            hashes = phash_video(path, samples=3)
            if hashes and dedup.is_duplicate(hashes):
                Path(path).unlink(missing_ok=True)
                return None
            if hashes:
                dedup.add(hashes)
        except Exception:
            pass  # never let pHash failure block the pipeline
        return path

    gen_fn = build_generative_fn() if enable_generative_fallback else None

    engine = SyncEngine(
        config=SyncEngineConfig(
            theme=theme,
            enable_embeddings=embedder is not None,
            enable_vision=verifier is not None,
            enable_generative_fallback=gen_fn is not None,
        ),
        retriever=retriever,
        ranker=ranker,
        verifier=verifier,
        intent_fn=_intent_fn,
        download_fn=_download_fn,
        generative_fn=gen_fn,
        progress=progress,
        download_dir=output_dir / "_clips",
    )

    # ── 3. Plan beats ──────────────────────────────────────────────────
    progress({"event": "step", "step": "sync", "status": "start",
              "clauses": len(clauses)})
    beats = engine.plan_for_clauses(clauses)
    solved = sum(1 for b in beats if b.is_solved)
    progress({"event": "step", "step": "sync", "status": "done",
              "solved": solved, "total": len(beats)})

    # ── 4. Compose ─────────────────────────────────────────────────────
    progress({"event": "step", "step": "compose", "status": "start"})
    composed = str(output_dir / f"{job_id}_composed.mp4")
    compose(beats, audio_path, composed, ComposeConfig(),
            work_dir=output_dir / "_compose")
    progress({"event": "step", "step": "compose", "status": "done"})

    # ── 5. Karaoke subs ────────────────────────────────────────────────
    progress({"event": "step", "step": "karaoke", "status": "start"})
    ass_path = output_dir / f"{job_id}.ass"
    build_ass(tts.sentences, ass_path)
    final = str(output_dir / f"{job_id}_FINAL.mp4")
    try:
        burn(composed, ass_path, final)
    except Exception as e:
        progress({"event": "step", "step": "karaoke", "status": "failed",
                  "error": str(e)})
        final = composed
    progress({"event": "step", "step": "karaoke", "status": "done"})

    # ── 6. Cleanup intermediates (preserve FINAL) — Windows-safe with retry ─
    # Threshold 5KB confirms the FINAL is a real file (not just an empty/corrupt write).
    if Path(final).exists() and Path(final).stat().st_size > 5_000:
        _cleanup_intermediates(
            final, [
                Path(audio_path),
                Path(composed),
                ass_path,
                output_dir / "_clips",
                output_dir / "_compose",
            ],
        )

    progress({"event": "done", "final": final, "solved": solved,
              "total": len(beats)})
    return {
        "ok": True, "final": final, "solved": solved,
        "total": len(beats),
        "video_count": sum(1 for b in beats
                            if b.is_solved and b.chosen and
                            not b.chosen.source.endswith("_photo")),
        "photo_count": sum(1 for b in beats
                            if b.is_solved and b.chosen and
                            b.chosen.source.endswith("_photo")),
    }


# ═════════════════════════════════════════════════════════════════════════
# Viderush-mode pipeline (Phase 6 wire-up)
# ═════════════════════════════════════════════════════════════════════════

from .sync.beat_snap import beat_snap, BeatSnapConfig
from .sync.concept import (
    ConceptWindow,
    group_into_clusters,
    segment_into_windows,
)
from .sync.pipeline import Intent, SyncedBeat
from .sync.sticky import build_sticky_pools
from .sync.temporal_verifier import (
    TemporalVerifyConfig,
    TemporalVisionVerifier,
)
from .stock.factory import build_brand_searches
from .subtitles.styles import auto_style, build_ass as build_ass_styled


def run_pipeline_viderush(
    script_text: str,
    voice: str,
    title: str,
    theme: str,
    output_dir: Path,
    *,
    mood: str = "",
    enable_temporal_vision: bool = True,
    enable_brand_aware: bool = True,
    enable_beat_snap: bool = True,
    enable_color_grading: bool = True,
    subtitle_style: str | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """End-to-end pipeline with ALL Phase 6 modules wired:
      • Concept-window segmentation (2-4s cuts)
      • Beat-snap to TTS silences
      • Sticky b-roll allocation per cluster
      • NER + brand-aware retrieval
      • Temporal Vision verification (optional)
      • Color grading per-clip
      • Auto-picked subtitle style

    Returns the same shape as run_pipeline() so callers can swap freely.
    """
    progress = progress or (lambda _e: None)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:10]

    # ── 1. TTS ─────────────────────────────────────────────────────────
    progress({"event": "step", "step": "tts", "status": "start"})
    audio_path = str(output_dir / f"{job_id}_audio.mp3")
    tts = synth_sync(script_text, voice, audio_path)
    progress({"event": "step", "step": "tts", "status": "done",
              "sentences": len(tts.sentences)})
    if not tts.sentences:
        return {"ok": False, "error": "no sentence boundaries from TTS"}

    # ── 2. Sentence clauses → Concept windows ──────────────────────────
    progress({"event": "step", "step": "concept_windows", "status": "start"})
    clauses = [NarrationClause(start=s["start"], end=s["end"], text=s["text"])
               for s in tts.sentences if s.get("text")]
    windows = segment_into_windows(clauses)
    if enable_beat_snap:
        windows = beat_snap(windows, audio_path, BeatSnapConfig())
    group_into_clusters(windows)
    progress({"event": "step", "step": "concept_windows", "status": "done",
              "windows": len(windows),
              "clusters": len({w.cluster_id for w in windows})})

    # ── 3. Build sticky allocator ──────────────────────────────────────
    progress({"event": "step", "step": "sticky_pools", "status": "start"})
    searches, downloads = build_sources()
    brand_searches = build_brand_searches() if enable_brand_aware else {}

    def _base_search(query, n=4):
        # Probe pexels first (fastest) then pixabay as backup
        out = searches.get("pexels", lambda q, n: [])(query, n)
        if len(out) < n:
            out.extend(searches.get("pixabay", lambda q, n: [])(query, n - len(out)))
        return out

    def _brand_search(brand, n=2):
        out = []
        for name, fn in brand_searches.items():
            try: out.extend(fn(brand, n))
            except Exception: pass
            if len(out) >= n:
                break
        return out

    allocator = build_sticky_pools(
        windows,
        base_search=_base_search,
        brand_search=_brand_search if enable_brand_aware else None,
    )
    progress({"event": "step", "step": "sticky_pools", "status": "done",
              "pools": len(allocator.pools)})

    # ── 4. Per-window allocation → download → optional Vision ──────────
    progress({"event": "step", "step": "sync", "status": "start",
              "windows": len(windows)})
    keys = LLMKeys.from_env()
    dedup = DedupStore()
    temporal_verifier = None
    if enable_temporal_vision and keys.gemini:
        temporal_verifier = TemporalVisionVerifier(
            TemporalVerifyConfig(gemini_api_key=keys.gemini)
        )

    clips_dir = output_dir / "_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    beats: list[SyncedBeat] = []
    for i, w in enumerate(windows):
        cand = allocator.next_for(w)
        if cand is None:
            beats.append(SyncedBeat(clause=w, intent=Intent(),
                                     chosen=None, error="no allocation"))
            continue
        # Download
        path = download_by_source(cand, clips_dir, downloads)
        if not path:
            beats.append(SyncedBeat(clause=w, intent=Intent(),
                                     chosen=None, error="download failed"))
            continue
        # pHash dedup
        try:
            hashes = phash_video(path, samples=3)
            if hashes and dedup.is_duplicate(hashes):
                Path(path).unlink(missing_ok=True)
                beats.append(SyncedBeat(clause=w, intent=Intent(),
                                         chosen=None, error="duplicate"))
                continue
            if hashes: dedup.add(hashes)
        except Exception:
            pass
        cand.local_path = path
        # Temporal Vision (optional, costly)
        if temporal_verifier is not None:
            verdict = temporal_verifier.verify(
                cand, w.text, topic=theme,
                era="modern", mood=w.mood or mood or "neutral",
            )
            if not verdict.get("approved"):
                beats.append(SyncedBeat(clause=w, intent=Intent(),
                                         chosen=None,
                                         rejected=[cand],
                                         error="vision rejected"))
                continue
        beats.append(SyncedBeat(clause=w, intent=Intent(), chosen=cand))
        if (i + 1) % 10 == 0:
            progress({"event": "step", "step": "sync", "progress": (i + 1) / len(windows)})

    solved = sum(1 for b in beats if b.is_solved)
    progress({"event": "step", "step": "sync", "status": "done",
              "solved": solved, "total": len(beats)})

    # ── 5. Compose with color grading ──────────────────────────────────
    progress({"event": "step", "step": "compose", "status": "start"})
    cfg = ComposeConfig()
    composed = str(output_dir / f"{job_id}_composed.mp4")
    compose(beats, audio_path, composed, cfg, work_dir=output_dir / "_compose")
    # Apply color grading as a separate pass (keeps composer tidy)
    if enable_color_grading:
        try:
            from .render.grading import auto_preset, grade_clip
            preset = auto_preset(theme=theme, mood=mood)
            graded = str(output_dir / f"{job_id}_graded.mp4")
            grade_clip(composed, graded, preset=preset)
            composed = graded
            progress({"event": "step", "step": "color_grading",
                      "preset": preset, "status": "done"})
        except Exception as e:
            progress({"event": "step", "step": "color_grading",
                      "status": "failed", "error": str(e)[:100]})

    progress({"event": "step", "step": "compose", "status": "done"})

    # ── 6. Karaoke subs with auto-style ────────────────────────────────
    progress({"event": "step", "step": "subtitles", "status": "start"})
    style = subtitle_style or auto_style(theme=theme, mood=mood)
    ass_path = output_dir / f"{job_id}.ass"
    build_ass_styled(tts.sentences, ass_path, style=style)
    final = str(output_dir / f"{job_id}_FINAL.mp4")
    try:
        burn(composed, ass_path, final)
    except Exception as e:
        progress({"event": "step", "step": "subtitles", "status": "failed",
                  "error": str(e)})
        final = composed
    progress({"event": "step", "step": "subtitles", "status": "done",
              "style": style})

    # ── 7. Cleanup intermediates ───────────────────────────────────────
    if Path(final).exists() and Path(final).stat().st_size > 5_000:
        _cleanup_intermediates(final, [
            Path(audio_path),
            Path(composed) if composed != final else Path("nope"),
            ass_path, clips_dir, output_dir / "_compose",
        ])

    progress({"event": "done", "final": final, "solved": solved,
              "total": len(beats),
              "style": style,
              "viderush_mode": True})
    return {
        "ok": True, "final": final, "solved": solved,
        "total": len(beats),
        "windows": len(windows),
        "clusters": len({w.cluster_id for w in windows}),
        "style": style,
        "video_count": sum(1 for b in beats
                            if b.is_solved and b.chosen and
                            not b.chosen.source.endswith("_photo")),
        "photo_count": sum(1 for b in beats
                            if b.is_solved and b.chosen and
                            b.chosen.source.endswith("_photo")),
    }
