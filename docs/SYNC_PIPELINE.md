# Sync Pipeline — Detalhes Técnicos

O diferencial do SyncForge é o **pipeline de sincronização semântica**:
cada frame de b-roll bate com o que está sendo dito, em qualquer nicho.

Baseado em retrieval-augmented vision (RAV) com modelos SOTA 2026.

---

## Estágio 1 — Análise da Narração

**Input:** Áudio TTS ou narração humana.

1. **Whisper-v3-turbo** (faster-whisper) → word-level timestamps
   - Por que turbo: 8x mais rápido que large-v3 mantendo 90% accuracy
   - Output: `[{word, start, end, prob}]`
2. **Sentence-boundary detection** (spaCy `en_core_web_sm`)
3. **Agrupamento em clauses** (3-12s, mín 8 palavras)
4. **Output normalizado:**
   ```python
   [{"start": 0.0, "end": 9.2, "text": "Butterfly's flagship blade...",
     "words": [...]}, ...]
   ```

---

## Estágio 2 — Intent Extraction (per-clause)

Para cada clause, LLM extrai estrutura semântica:

```json
{
  "main_entity": "Butterfly Viscaria blade",
  "action": "showing in close-up",
  "location": "indoor",
  "era": "modern (2010s+)",
  "objects": ["table tennis paddle", "wood blade", "rubber sheets"],
  "mood": "scrutinizing / analytical",
  "key_visuals": ["close-up paddle", "wood grain", "lab measurement"],
  "queries": [
    "Butterfly Viscaria table tennis blade close up",
    "professional ping pong paddle wood grain detail",
    "table tennis equipment review",
    "...8 ângulos no total..."
  ]
}
```

LLM chain: **Gemini 2.5 Pro → Groq Llama 3.3 → Cerebras → OpenRouter → DeepSeek**

---

## Estágio 3 — Multi-Source Retrieval (paralelo)

Para cada query, busca em paralelo:

| Fonte         | API/Method                   | Tipo        |
|---------------|------------------------------|-------------|
| Pexels        | `/v1/videos/search`          | Vídeo HD    |
| Pixabay       | `/api/videos/`               | Vídeo HD    |
| YouTube       | `yt-dlp` + duration filter   | Real footage|
| Coverr        | `/api/v2/coverr/videos/search`| Cinemagraphs|
| Mixkit        | scraping API                 | Free videos |
| Wikimedia     | Commons API                  | CC0 video   |
| Pexels Photos | `/v1/search`                 | Foto + Ken Burns |

**Pool típico: 60-100 candidatos por intent.**

Anti-repeat global via `dedup_set` (URL + pHash perceptual).

---

## Estágio 4 — Embedding-Based Ranking ★ DIFERENCIAL

Aqui é onde o SyncForge vence as ferramentas atuais.

### Provider primário: Gemini Embedding 2

API: `models/gemini-embedding-002`

Embeda em vetor compartilhado:
- **Texto narração** (1024-dim)
- **3-5 frames sampleados de cada vídeo candidato** (1024-dim por frame, média)
- **Metadados** (title + description + tags) (1024-dim)

### Provider local fallback: SigLIP-2

`google/siglip2-large-patch16-512` via transformers

- Roda em CPU/GPU local
- Sem custo de API
- ~80% performance da Gemini Embedding 2

### Score por candidato

```python
score = 0.5 * cos(narração, mean(frames)) +
        0.3 * cos(narração, metadados) +
        0.2 * BM25(narração, title + description)
```

**Top-10 por score avança pro Estágio 5.**

---

## Estágio 5 — Vision Verification (LLM-as-judge)

Para os top-10 de cada intent:

**Gemini 2.5 Pro Vision** recebe:
- O clause de narração (texto)
- 8 frames sampleados do clip (uniformemente espaçados)

Prompt:
```
Você é um diretor de vídeo. A narração diz: "{clause_text}"
O candidato visual mostra: [8 frames]

Responda em JSON:
{
  "relevance_score": 0-100,
  "description": "...",
  "anachronism": true/false,
  "off_topic": true/false,
  "approved": true/false  (true se relevance ≥70 E !anachronism E !off_topic)
}
```

Aprovados entram no compose. Rejeitados voltam pro pool e tentam o próximo.

---

## Estágio 6 — Reciprocal Rank Fusion (RRF)

Combina os 3 sinais:

```python
RRF_score = 1/(k + rank_BM25) + 1/(k + rank_embedding) + 1/(k + rank_vision)
```

Com `k=60` (padrão Cormack et al.).

Winner = highest RRF.

---

## Estágio 7 — Generative Fallback

Se TODOS os 100 candidatos foram rejeitados (raro mas possível em nichos
super-específicos como "lab Test Spin Coefficient table tennis 0.62"):

**Gera o clip via Veo 3 ou Runway Gen-4:**
- Prompt: `key_visuals + main_entity + mood`
- Duração: 6-8s
- Custo: ~$0.30 por clip (Veo 3)

**Garantia: 0% segmentos vazios.**

---

## Estágio 8 — Composition

- ffmpeg concat com xfade (transições suaves)
- Karaoke subs via TTS WordBoundary (timing perfeito, zero Whisper drift)
- BGM auto-mood (royalty-free do Pixabay Music)
- Avatar overlay top-corner (opcional, com LatentSync HD)
- Auto-cleanup: deleta intermediários após FINAL.mp4

---

## Métricas de qualidade

| Métrica                    | Target | Como medir              |
|----------------------------|--------|-------------------------|
| Semantic relevance         | ≥85%   | Gemini Vision audit     |
| Coverage (segmentos OK)    | 100%   | `solved/total`          |
| Repetition rate            | 0%     | pHash duplicate count   |
| Real-video % (vs fotos)    | ≥80%   | source breakdown        |
| Time-to-first-frame        | <2min  | wall clock              |
| Cost per 15min video       | <$1    | API token aggregation   |

---

## Por que isso é melhor que MoneyPrinterTurbo/VideoForge

| Aspecto | MPT/VForge | SyncForge |
|---------|------------|-----------|
| Match | keyword substring | semantic embedding |
| Verification | nenhum | LLM-as-judge multi-frame |
| Fonts | 2 (Pexels+Pixabay) | 7 (+YT+Coverr+Mixkit+Wiki+CC0) |
| Fallback | repete clip random | gera com Veo 3 |
| Anachronism check | nenhum | dedicated guard |
| Sub timing | Whisper drift | TTS WordBoundary exato |
| Repetition control | sem | pHash global + cooldown infinito |
