# SyncForge Changelog

## Phase 4 — Productization (2026-06-23)

**155 tests, 155 PASS** (vs Phase 3's 118). +37 tests, 0 regressions.

### Added (7 modules)
- **`auth/users.py`** — file-backed UserStore with PBKDF2-SHA256 (200k iters)
  salted hashing + thread-safe persistence. JWT-like HS256 tokens.
- **`core/quota.py`** — per-user monthly quota (videos + bytes), YYYY-MM
  partition. Locked file writes.
- **`core/audit.py`** — append-only JSONL audit log with `tail(actor=, event=)`.
- **`core/license.py`** — offline license keys `SF-{plan}-{uid}-{exp}-{sig}`
  signed HMAC-SHA256. Lifetime + dated licenses supported.
- **`billing/stripe_handler.py`** — Stripe Checkout creation + webhook
  signature verification (5-min replay window) + event→action translator.

### New API surface (8 endpoints)
- `POST /api/auth/register` · `POST /api/auth/login` · `GET /api/auth/me`
- `POST /api/billing/checkout` · `POST /api/billing/webhook`
- `GET /api/license/issue` · `POST /api/license/verify`
- `GET /api/audit/tail` (user-scoped unless ADMIN_USER_ID matches)

### Quota gated on /api/jobs
- Anonymous in dev (SYNCFORGE_REQUIRE_AUTH=1 to force login).
- Free=5 videos/mo, Pro=100, Team=1000. 429 on exhaustion.
- Records to audit log on creation + on quota_exceeded.

### Bug fixed
- License key signature contained `-` (urlsafe-b64), broke `split("-")`.
  Fixed with `split("-", 4)` so the sig stays intact as the last part.

## Phase 3 — Quality bumps (2026-06-23)

**81 tests, 81 PASS** (vs Phase 1's 60). +23 new tests, 0 regressions.

### Added (7 modules)
- **`sync/dedup.py`** — perceptual hash (pHash) anti-duplicate across all sources.
  Samples 3 frames per video, compares Hamming distance vs threshold (default 6).
  Catches Pexels re-uploads, YouTube mirror channels, photo crops.
- **`llm/translate.py`** — auto-translates non-English queries → English via LLM chain.
  Heuristic-skip already-English queries (saves API calls). Per-query memory cache.
- **`generate/veo.py`** — Google Veo 3 generative fallback (6-8s clips, `gemini-2.5-pro-video`).
- **`generate/runway.py`** — Runway Gen-4 alternative (gen4_turbo, $0.05/s).
- **`generate/factory.py`** — env-driven provider picker (Veo if Gemini key, else Runway).
- **`tts/f5_clone.py`** — F5-TTS voice cloning (CLI wrapper) + whisper re-transcribe for sentence boundaries.
- **`avatar/latentsync.py`** — LatentSync HD lip-sync wrapper (256px + Real-ESRGAN mouth SR) + `overlay_corner()` ffmpeg helper.

### Wired into runner
- Intent queries now pass through `translate_to_english` before retrieval.
- Downloaded clips pass through `phash_video` + `DedupStore`; duplicates are deleted in place.
- `enable_generative_fallback=True` now actually fires Veo 3 / Runway via factory.

### Bug fixed
- `is_already_english("a casa é grande")` returned True (regex stripped accents).
  Fixed: any non-ASCII char → return False; empty word list → return False (translate).

## Phase 2 — Next.js 14 Frontend (2026-06-23)

Full UI shipped. App Router + TypeScript + Tailwind + dark mode + WebSocket live timeline.

### Pages
- `/` — landing
- `/dashboard` — job history with auto-refresh every 2s
- `/create` — 3-mode picker (TTS / avatar overlay / avatar full) + voices + features
- `/jobs/[id]` — **live WebSocket timeline** with 5-stage pipeline view + clause-by-clause log
- `/settings` — provider key manager (browser localStorage; backend reads `.env`)

### Architecture
- `lib/api.ts` — REST helpers + `subscribeJob()` WebSocket subscriber
- `components/Sidebar.tsx` — nav with active-route highlighting
- `components/JobTimeline.tsx` — WS-driven stage tracker; aggregates `clause_done`
  events into a tail-following log; auto-shows FINAL.mp4 path on completion.

### Stack
- Next.js 14.2.15 (App Router)
- TypeScript 5.6 strict
- Tailwind 3.4 with custom dark palette (`bg`, `ink`, `accent`)
- Lucide-react icons only (no other deps)
- 8 source files + 4 config files

### How to run
```
cd C:\Users\Guilherme\Desktop\SyncForge
start-backend.bat   # FastAPI on :8000
start-frontend.bat  # Next.js on :3000 (npm install on first run)
```

## Phase 1.1 — Tested (2026-06-23)

**60 tests, 58 PASS, 2 skipped** (smoke tests requiring Pexels/Pixabay keys).

### Bug fixes from testing
- **Cleanup skip bug**: `_cleanup_intermediates` was guarded by `final_size > 100KB`,
  which silently SKIPPED cleanup for short videos and left clip files behind.
  Lowered to 5KB (still proves valid write) so cleanup always runs on success.
- **Windows file lock retry**: cleanup now does GC + 1s sleep + GC before deleting,
  then walks dirs bottom-up with per-file retry (6 attempts, exponential backoff).
  Handles post-ffmpeg handle release race.

### Test coverage
- 21 import smoke tests (every module loads cleanly)
- 9 ranker unit tests (BM25, RRF, multi-signal fusion)
- 7 intent + karaoke ASS unit tests
- 5 retriever orchestration tests (dedup, pool limits, exception handling)
- 3 pipeline mock tests (3-clause end-to-end with stubs)
- 5 FastAPI tests (REST + WebSocket replay)
- 3 real-API smoke tests (Wikimedia PASS, Pexels/Pixabay skipped sem key)
- 6 integration tests with real ffmpeg + Edge-TTS + karaoke burn + runner E2E

## Phase 1 — Wired End-to-End (2026-06-23)

Tudo cabeado entre FastAPI → SyncEngine → ffmpeg final. Faltam só os
adapters de Veo/Runway (Phase 3) e o frontend Next.js (Phase 2).

### Adicionado
- **6 adapters de stock**: `pexels`, `pixabay`, `youtube` (yt-dlp), `coverr`,
  `mixkit`, `wikimedia` (+ `pexels_photo` fallback). Cada um implementa
  `search()` + `download()` com timeouts e error handling.
- **`stock/factory.py`** — monta as maps `{name: search_fn}` e `{name: download_fn}`
  a partir do env, opcionalmente desabilitando fontes sem chave.
- **LLM chain de 5 níveis** (`llm/chain.py`): Groq → Gemini 2.5 Flash →
  Cerebras → OpenRouter → DeepSeek.
- **`llm/intent.py`** — extractor JSON-mode com fallback determinístico se
  todos os providers falharem.
- **`tts/edge.py`** — Edge-TTS com captura de SentenceBoundary (timing exato).
- **`render/composer.py`** — ffmpeg pipeline: normaliza cada clip pro tamanho
  alvo, aplica Ken Burns em fotos, concat + mux áudio.
- **`subtitles/karaoke.py`** — ASS builder + ffmpeg burn (timeout 2400s).
- **`services/runner.py`** — pipeline end-to-end chamável por background job.
- **`main.py`** atualizado: WebSocket publica todos os eventos do `progress`
  callback do `SyncEngine` em tempo real.

### Próximo (Phase 2)
- Next.js 14 frontend com `/create`, `/dashboard`, `/jobs/[id]` (live timeline
  via WS), `/settings` (API keys).

## Phase 0 — Foundation (2026-06-23)
- Estrutura de pastas + docs (README, SYNC_PIPELINE, ARCHITECTURE, ROADMAP).
- SyncEngine core (`embedder`, `ranker`, `retriever`, `verifier`, `pipeline`).
- FastAPI + WebSocket scaffold com job store em memória.
