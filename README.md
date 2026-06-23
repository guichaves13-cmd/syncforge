# SyncForge

> Geração de vídeos com sincronização semântica perfeita entre narração e clips reais
> (YouTube + Pexels + Pixabay + Wikimedia + Coverr + Mixkit).
>
> Foco único: **cada frame do b-roll bate com o que está sendo dito.**

## Por que SyncForge

Ferramentas como MoneyPrinterTurbo e VideoForge-Clone fazem "match aproximado":
pegam palavras-chave do roteiro e procuram no Pexels. Resultado: você ouve
"raquete japonesa Butterfly" e vê uma raquete genérica de tênis.

SyncForge usa o pipeline de retrieval-augmented vision (RAV) de 2026:

```
Narração ─┬─> Whisper-v3-turbo (word-level timing)
          ├─> Sentence segmentation + per-clause intent extraction
          ├─> Embedding multimodal (Gemini Embedding 2 ou SigLIP-2)
          └─> Geração de queries em 8 ângulos por LLM

Pool ─────┬─> Pexels + Pixabay + YouTube (yt-dlp) + Coverr + Mixkit + Wikimedia
          └─> 60-100 candidatos por intent (paralelo)

Ranking ──┬─> SigLIP-2/Gemini Emb 2 cosine (narração ↔ frames sampleados)
          ├─> BM25 título/descrição
          └─> Reciprocal Rank Fusion

Verify ───┬─> Gemini 2.5 Pro Vision assiste o clip top-10 (8 frames)
          └─> Score 0-100, aprova ≥70

Fallback ─┬─> Veo 3 ou Runway Gen-4 GERA clip se nada bater
          └─> Garante 0% segmentos vazios

Compose ──> ffmpeg + karaoke subs via TTS WordBoundary + auto-cleanup
```

## Stack

| Camada     | Tech                                       |
|------------|--------------------------------------------|
| Backend    | FastAPI + WebSocket (real-time progress)   |
| Frontend   | Next.js 14 + TypeScript + dark mode        |
| LLM        | Gemini 2.5 + Groq + Cerebras + OpenRouter + DeepSeek |
| Embedding  | Gemini Embedding 2 (primary) / SigLIP-2 (local fallback) |
| Vision     | Gemini 2.5 Pro Vision                      |
| TTS        | Edge-TTS + ElevenLabs + OpenAI TTS + F5-TTS clone |
| Stock      | Pexels + Pixabay + YouTube + Coverr + Mixkit + Wikimedia |
| Generative | Veo 3 / Runway Gen-4 (fallback)            |
| Avatar     | LatentSync HD + Wav2Lip                    |
| Render     | ffmpeg + Real-ESRGAN 4K mouth region       |

## Roadmap

- [x] Estrutura + docs
- [ ] SyncEngine core (text→video semantic match com embeddings)
- [ ] Multi-source retrieval paralelo
- [ ] Gemini Vision verification pipeline
- [ ] FastAPI + WebSocket scaffold
- [ ] Next.js frontend (dashboard + create + settings + jobs)
- [ ] Generative fallback (Veo 3 / Runway Gen-4)
- [ ] Avatar pipeline integrado
- [ ] Karaoke subs via TTS WordBoundary
- [ ] Tests + CI

Ver [`docs/SYNC_PIPELINE.md`](docs/SYNC_PIPELINE.md) para detalhes técnicos da sincronização.
