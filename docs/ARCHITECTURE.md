# SyncForge Architecture

## High-level

```
┌────────────────────────────────────────────────────────────────┐
│                       Next.js 14 Frontend                       │
│  /create  /dashboard  /jobs/[id] (live)  /settings              │
└────────────────┬────────────────────────────┬──────────────────┘
                 │ REST                       │ WebSocket
┌────────────────▼────────────────────────────▼──────────────────┐
│                    FastAPI Backend                              │
│  POST /api/jobs   GET /api/jobs/{id}   WS /ws/jobs/{id}         │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                       SyncEngine                                │
│  ┌──────────┐  ┌─────────┐  ┌────────┐  ┌──────────┐            │
│  │Intent    │→ │Retriever│→ │Ranker  │→ │Verifier  │ → SyncedBeat
│  │Extractor │  │(7 srcs) │  │(BM25+  │  │(Vision)  │            │
│  └──────────┘  └─────────┘  │ embed +│  └──────────┘            │
│                             │  RRF)  │                          │
│                             └────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│  services/                                                       │
│   sync/      embedder · ranker · retriever · verifier · pipeline│
│   stock/     pexels · pixabay · youtube · coverr · wiki · mixkit│
│   llm/       gemini · groq · cerebras · openrouter · deepseek   │
│   tts/       edge · openai · elevenlabs · f5_clone              │
│   subtitles/ karaoke (TTS WordBoundary)                         │
│   render/    ffmpeg composer · bgm mixer · avatar overlay       │
│   generate/  veo3 · runway_gen4 (generative fallback)           │
└─────────────────────────────────────────────────────────────────┘
```

## Principles

1. **Provider-pluggable** — every service has a `base.py` + factory; switching
   from Gemini Vision to Claude Vision is one config change.
2. **Cache-first** — embeddings, intents, vision verdicts all hit disk before
   network. Sha256 keys, never invalidate (clips are immutable).
3. **Bulletproof** — every external call has a fallback chain ending in either
   a generative fallback or a graceful empty result. No segment ever crashes the run.
4. **Real-time observable** — every stage publishes structured events to a WS
   queue. The frontend gets a live timeline, not a polling guess.
5. **Auto-cleanup** — composed.mp4, audio.mp3, clip dirs all deleted as soon as
   FINAL.mp4 exists. Disk usage stays flat at one video.

## Data flow per clause (the hot loop)

```
clause ─► IntentExtractor (Gemini, fallback chain) ─► Intent{8 queries}
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
   Retriever ──parallel──► Pool[60-100 candidates]
                  │
                  ▼ BM25 rank → top-10 download
                  ▼
              Embedder (Gemini Emb 2 / SigLIP-2)
                  ▼ cosine narration ↔ frames
                  ▼
              RRF(BM25, embed, vision) — Verifier on top-K
                  ▼
              First with approved=true → reserved & returned
                  │
                  ▼ (if all reject)
              Generative fallback (Veo 3 / Runway Gen-4)
                  ▼
              SyncedBeat{clause, chosen, ...}
```

## Storage layout

```
storage/
  videos/        FINAL.mp4 files (kept)
  audio/         TTS narrations (kept)
  clips/         downloaded source clips (auto-purged after compose)
  music/         user-provided BGM tracks
  cache/
    embeddings/  sha256.npy  (1024-dim float32, ~4KB each)
    intents/     sha256.json
    vision/      sha256.json
```

## Scaling notes

- Single-process today; in-memory `JOBS`.
- Swap to Redis + RQ workers for multi-tenant.
- Embedding cache can grow large — add LRU eviction past N GB.
- Gemini calls are the bottleneck; budget ~100 calls per 15-min video.
