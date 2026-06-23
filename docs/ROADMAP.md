# SyncForge Roadmap

## Phase 0 — Foundation ✅
- [x] Folder structure
- [x] README, ARCHITECTURE, SYNC_PIPELINE docs
- [x] SyncEngine core (embedder, ranker, retriever, verifier, pipeline)
- [x] FastAPI + WebSocket scaffold
- [x] requirements.txt

## Phase 1 — Wire the engine (1-2 weeks)
- [ ] `services/llm/gemini_intent.py` — Gemini 2.5 Pro intent extraction with 4-tier fallback
- [ ] `services/stock/{pexels,pixabay,youtube,coverr,mixkit,wikimedia}.py` adapters
- [ ] `services/tts/{edge,elevenlabs,openai}.py`
- [ ] `services/subtitles/karaoke.py` (TTS WordBoundary)
- [ ] `services/render/composer.py` (ffmpeg)
- [ ] Wire `_run_job` in main.py to call SyncEngine end-to-end
- [ ] First end-to-end render: 1 video, no avatar, all features

## Phase 2 — Frontend (1 week)
- [ ] Next.js 14 scaffold (App Router, TypeScript, Tailwind, dark mode)
- [ ] `/create` page with mode toggle (title/script/avatar-narrated)
- [ ] `/jobs/[id]` live timeline (WebSocket subscription, stage-by-stage)
- [ ] `/dashboard` job history
- [ ] `/settings` API key management (encrypted at rest)

## Phase 3 — Quality bumps (ongoing)
- [ ] SigLIP-2 local fallback (free, no API)
- [ ] LatentSync HD avatar integration
- [ ] Mouth-region Real-ESRGAN 4K
- [ ] F5-TTS voice cloning
- [ ] Generative fallback (Veo 3 / Runway Gen-4)
- [ ] pHash global anti-duplicate
- [ ] Multi-language auto-translate queries

## Phase 4 — Productization (2 weeks)
- [ ] Stripe + license server
- [ ] Multi-tenant workers (Redis + RQ)
- [ ] User auth (Clerk or NextAuth)
- [ ] Usage metering + quotas
- [ ] CDN for FINAL.mp4 serving
- [ ] Audit log, analytics
- [ ] Public landing page

## Phase 5 — Distribution
- [ ] Standalone .exe launcher (PyInstaller)
- [ ] Auto-update flow
- [ ] Docs site (Astro + MDX)
- [ ] Open-source the SyncEngine core under AGPL
