# 🎙️ Voice-Enabled Multilingual RAG Pipeline

A voice-first Retrieval-Augmented Generation (RAG) system supporting **English, Hindi, and Marathi** queries, built for the HH Goa 2026 Shortlisting Task 2.

**Dataset:** [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)  
**Live Demo:** `http://localhost:8000`

---

## Pipeline Architecture

```
Voice Input (Browser Mic)
    │
    ▼
[WebM/Opus Audio Blob ~20KB]
    │
    ▼
Sarvam AI STT (saaras:v4)          ← Speech-to-Text
    │  transcript + latency_ms
    ▼
Language Detection                  ← Devanagari heuristic (Hindi vs Marathi)
    │  lang_filter ∈ {en, hi, mr, hi_mr}
    ▼
Safety Guardrail (keyword blocklist) ← Pre-LLM check
    │
    ▼
Query Embedding (E5-Base FP16/ONNX) ← intfloat/multilingual-e5-base
    │  768-dim vector
    ▼
LanceDB IVF-PQ ANN Search           ← nprobes=80, BTree scalar index on `language`
    │  top-8 passages + distances
    ▼
Input Relevance Check               ← cosine similarity threshold
    │
    ▼
Groq LLM Generation (openai/gpt-oss-20b) ← streaming, temperature=0.0, tenacity retries
    │  token stream
    ▼
Output Grounding Guardrail          ← embedding similarity + LLM fallback for borderline
    │  GROUNDED / REFUSAL / NOT GROUNDED
    ▼
SSE Stream to Frontend Dashboard
```

---

## ⚡ Latency — Honest Disclosure

The specification sets a target of **under 200ms** for the full pipeline. Here is a transparent breakdown of what our system achieves based on our latest isolated benchmark runs (reflecting clean, uncontaminated GPU execution and real-world API/network variance):

| Sub-pipeline / Step | P50 (Median) | P100 (Max) | Within 200ms? |
| :--- | :---: | :---: | :---: |
| Query Embedding only | ~33 ms | ~55 ms | ✅ |
| Vector DB Search only | ~94 ms | ~258 ms | ✅ / ❌ |
| **Combined Retrieval (Embed + Search)** | **~125 ms** | **~311 ms** | ✅ / ❌ |
| **Sarvam AI STT RTT (Speech-to-Text)** | **~1148 ms** | **~5890 ms** | ❌ (external API + network) |
| LLM TTFT (Groq network RTT) | ~624–901 ms | ~1280–2053 ms | ❌ (external API) |
| Full Text RAG (Retrieval + LLM + Guardrails) | ~903–1095 ms | ~1798–2444 ms | ❌ (by design) |
| **Voice End-to-End (STT + RAG)** | **~2050 ms** | **~8877 ms** | ❌ (by design) |

> ### 📊 Understanding Run-to-Run & Network Variance
> 1. **STT Network & API Load Variance:** The STT round-trip time ranges from **~1148 ms (P50)** up to a maximum of **~5890 ms (P100 / Max)**. This wide spread is driven entirely by network quality, audio upload payload size, and load/queuing on Sarvam AI's hosted API servers.
> 2. **Groq Model & API Queuing:** The LLM's Time-to-First-Token (TTFT) has a spread (P50 of ~624ms on English, up to a P100 of ~2.05 seconds under heavy API load or rate-limit pacing delays). Since we run on hosted infrastructure, latency is highly sensitive to remote concurrency and network health.
> 3. **GPU State Warmup vs Contention:** In clean isolated runs, our local embedding step runs in **~33ms P50**. During benchmarks, we bypassed the WDDM driver low-power transition latency (which occurs when the GPU is idle during the 12s sleep/network waits) to capture the actual execution speed.
>
> **What can realistically achieve <200ms:**  
> When running locally, the retrieval sub-pipeline (Embedding + LanceDB vector search) comfortably completes in **~125ms P50**, which is well within the 200ms budget. Any step involving remote API calls (Sarvam STT, Groq LLM) will inherently exceed 200ms due to internet propagation delay.

See [`data/latency_benchmark_report.md`](data/latency_benchmark_report.md) for full P50/P70/P90/P95/P100 numbers across all languages and pipeline steps.

---

## Technical Requirements Coverage

### 1. Speech-to-Text — Sarvam AI ✅
- Provider: **Sarvam AI `saaras:v4`** via REST API
- File: [`stt_provider.py`](stt_provider.py)
- Browser records in **WebM/Opus** (~20KB for 4s audio), a 12× compression vs raw WAV
- 8-second strict timeout with graceful silent-audio and error fallbacks
- Latency logged per-request to `data/stt_latency_log.csv`

### 2. Chunking Strategy — Multiple Approaches ✅
- File: [`chunking.py`](chunking.py)
- **Three strategies implemented and benchmarked:**

| Strategy | Description |
| :--- | :--- |
| `naive_fixed_size` | Character-based sliding window (size=200, overlap=50). Baseline. |
| `sentence_based` | Regex multilingual sentence splitter (`.?!।`) grouping up to 300 chars. Handles Devanagari punctuation. Long unpunctuated blocks are recursively split with 50-char overlap. |
| `metadata_aware` | Sentence grouping + augmented prefix: `Language | Category | Keywords | Passage`. Keywords extracted via multilingual stopword-filtered frequency ranking. Prefix enriches the embedding space without hardcoding query terms. |

- **Production index** uses `metadata_aware` exclusively (best recall in evaluation)
- Multilingual stopword lists for English, Hindi, Marathi
- Chunk size bounded to prevent outlier run-on blocks

### 3. Latency Analytics — P50 / P70 / P100 ✅
- File: [`run_benchmarks.py`](run_benchmarks.py)
- 20 queries across 3 languages (10 EN, 5 HI, 5 MR)
- Measures per step: embedding, vector search, retrieval, TTFT, full generation, grounding, total
- Reports: **Mean, P50, P70, P90, P95, P100 (max)**
- STT latency tracked separately from `data/stt_latency_log.csv`
- CUDA cold-start warmup query run before benchmarking to prevent skew

### 4. Model Harness ✅
- File: [`llm_orchestrator.py`](llm_orchestrator.py) + [`rag_orchestrator.py`](rag_orchestrator.py)
- Abstract `LLMProvider` base class — swappable backends
- **Tenacity retry logic:** 3 attempts, exponential backoff (2s–10s) on rate limits / transient failures
- **Structured SSE streaming:** each pipeline stage emits a typed event (`stt_complete`, `retrieval_complete`, `generation_start`, `token`, `grounding_complete`)
- Zero temperature for deterministic grounding
- 3-second strict Groq client timeout for fast failure recovery

### 5. Guardrails ✅
- File: [`guardrails.py`](guardrails.py)

| Guardrail | Trigger | Action |
| :--- | :--- | :--- |
| **Safety blocklist** | Pre-LLM | Blocks weapons, hacking, self-harm keywords (EN/HI/MR) |
| **Input relevance** | Post-retrieval | Cosine similarity check on retrieved docs |
| **Output grounding (auto-pass)** | Post-generation | Cosine sim > 0.85 → GROUNDED |
| **Output grounding (auto-block)** | Post-generation | Cosine sim < 0.80 → NOT GROUNDED → fallback answer |
| **LLM grounding eval** | Borderline (0.80–0.85) | Fast Groq call for GROUNDED / NOT GROUNDED verdict |
| **Refusal detection** | Post-generation | Regex on refusal phrases → bypass semantic check |

- Multilingual fallback answers in Hindi and Marathi when blocked
- System returns `"I don't know based on the provided context."` rather than hallucinating

---

## Vector Database — LanceDB

- **Index:** IVF-PQ ANN (`num_partitions=256`, `num_sub_vectors=192`, `metric=cosine`)
- **Scalar Index:** BTree index on `language` column — prevents linear scan during language-filtered queries (dropped search from 108ms → 32ms P50)
- **nprobes=80** for high-recall ANN search
- **768-dimensional** embeddings from `intfloat/multilingual-e5-base`
- FP16 on CUDA GPU / ONNX quantised model as CPU fallback
- Full dataset: 5,000 rows × 3 languages = ~300K indexed chunks

---

## Setup & Running

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Variables
Copy `.env.example` to `.env` and fill in:
```
SARVAM_API_KEY=your_sarvam_key
GROQ_API_KEY=your_groq_key
```

### Build the Index (one-time)
```bash
python index_and_retrieve.py
```

### Start the Server
```bash
python app.py
# → http://localhost:8000
```

### Run Latency Benchmarks
```bash
python run_benchmarks.py
# → data/latency_benchmark_report.md
```

---

## ☁️ Cloud Deployment (Render / Railway)

Because the LanceDB vector database is ~1.1GB (too large for Git) and generating 300,000 embeddings on a free cloud CPU would exceed build timeouts, the pipeline uses a robust **pre-built download fallback**:

### Step 1: Upload the Vector DB
1. A compressed `lancedb.zip` file has been created in your project root.
2. Push your code to GitHub.
3. On GitHub, create a new **Release** (e.g., `v1.0.0`) and upload `lancedb.zip` as a release asset.
4. Copy the direct download link to `lancedb.zip` from your GitHub Release page (e.g., `https://github.com/your-username/your-repo/releases/download/v1.0.0/lancedb.zip`).

### Step 2: Deploy to Render
We have included a `render.yaml` blueprint file in the repository root for one-click deployment:
1. Go to the **Render Dashboard** and click **New** → **Blueprint**.
2. Connect your GitHub repository.
3. Render will detect `render.yaml` and prompt you for the environment variables:
   - `SARVAM_API_KEY`: Your Sarvam AI STT subscription key.
   - `GROQ_API_KEY`: Your Groq API key.
   - `LANCE_DB_DOWNLOAD_URL`: The direct GitHub release asset download link copied in Step 1.
4. Click **Apply**. Render will deploy the service and run `ensure_database()` on startup to download and extract the vectors in seconds.
5. If the CPU environment does not have access to the local ONNX weights directory, the pipeline automatically falls back to downloading the base model weights from HuggingFace to run CPU inference.

---

## Model Migration Note (August 16, 2026)

Groq deprecated `llama-3.1-8b-instant` on August 16, 2026. The pipeline was migrated to `openai/gpt-oss-20b` with zero code changes to the harness — only the `model_name` parameter in [`llm_orchestrator.py`](llm_orchestrator.py) was updated. This demonstrates the value of the abstract `LLMProvider` interface.
