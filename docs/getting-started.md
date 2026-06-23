# Getting Started

## Install

### Windows — prebuilt
1. Download `SyncForge-vX.Y.Z-win64.zip` from [Releases](https://github.com/syncforge/syncforge/releases).
2. Unzip anywhere.
3. Run `install.bat`. Creates a Desktop shortcut to launch the backend.

### From source
```bash
git clone https://github.com/syncforge/syncforge
cd SyncForge
# Backend
python -m venv backend/venv
backend\venv\Scripts\activate         # Linux/Mac: source backend/venv/bin/activate
pip install -r backend/requirements.txt
# Frontend
cd frontend && npm install && cd ..
```

## Configure

Copy `.env.example` to `.env` and fill in:

| Key | Required? | What for |
|---|---|---|
| `GEMINI_API_KEY` | recommended | Intent extraction + Vision verifier + embeddings |
| `PEXELS_API_KEY` | recommended | HD stock videos (free tier 200 req/hr) |
| `PIXABAY_API_KEY` | recommended | HD stock videos (free) |
| `GROQ_API_KEY` | optional | Faster LLM tier (Llama 3.3 70B) |
| `ELEVENLABS_API_KEY` | optional | Premium voice cloning |
| `STRIPE_API_KEY` | optional | Billing |
| `SYNCFORGE_JWT_SECRET` | **required for prod** | Token signing |
| `LICENSE_SECRET` | **required for prod** | License signing |

## Run

```bash
start-backend.bat   # http://localhost:8000
start-frontend.bat  # http://localhost:3000
```

Open <http://localhost:3000/create>, paste a title, click Start.

## First video

Try title:
> "5 Table Tennis Paddle Brands ROBBING You Blind"

Set theme:
> "ping pong paddle blade rubber tournament"

Length: 10 minutes. Voice: Andrew (en-US).

The pipeline runs:
1. **TTS** — Edge-TTS narrates the script, captures word-level timing.
2. **Sync** — for each sentence: extract intent → retrieve candidates → embedding rank → Gemini Vision verify.
3. **Compose** — ffmpeg concatenates the b-roll synced to the audio.
4. **Karaoke** — burn ASS subs with TTS WordBoundary timing.
5. **Cleanup** — intermediates deleted automatically.

Watch the live timeline on `/jobs/{id}` (WebSocket stream).
