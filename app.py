import os
import sys
import json
import asyncio
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure environment variables are loaded overriding any shell cache
load_dotenv(override=True)

# Import RAG and STT modules
from rag_orchestrator import RAGPipeline
from stt_provider import SarvamSTT

# Initialize FastAPI App
app = FastAPI(title="Voice-Enabled Multilingual RAG Backend")

# Mount Static Files (serves styles and client JS)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root dashboard router
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# Global variables for lazy loading
pipeline = None
stt_engine = None

@app.on_event("startup")
def startup_event():
    global pipeline, stt_engine
    print("Starting up server components...", flush=True)
    pipeline = RAGPipeline()
    stt_engine = SarvamSTT()
    print("Server components loaded successfully.", flush=True)

# Helper function to format Server-Sent Events
def format_sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# Helper function to log STT API latency to file
def log_stt_latency(filename: str, status: str, latency_ms: float, transcript: str):
    log_file = "data/stt_latency_log.csv"
    os.makedirs("data", exist_ok=True)
    write_header = not os.path.exists(log_file)
    with open(log_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("timestamp,filename,status,latency_ms,transcript\n")
        safe_transcript = (transcript or "").replace('"', '""')
        f.write(f"{time.time()},{filename},{status},{latency_ms:.2f},\"{safe_transcript}\"\n")

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "model": "intfloat/multilingual-e5-base", "gpu_active": torch.cuda.is_available() if 'torch' in sys.modules else False}

@app.post("/api/query-text")
def query_text_endpoint(payload: dict):
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
        
    async def sse_generator():
        try:
            # We run the synchronous query_stream_events generator in an executor or direct loop
            # Since it yields tokens, we run it directly. To make it non-blocking, we yield periodically.
            events = pipeline.query_stream_events(query)
            for event in events:
                yield format_sse(event)
                await asyncio.sleep(0.01) # Yield execution to event loop
        except Exception as e:
            print(f"Error in text query endpoint: {e}", file=sys.stderr)
            yield format_sse({"event": "error", "message": str(e)})

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.post("/api/query-voice")
async def query_voice_endpoint(file: UploadFile = File(...), language: str = Form("en")):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")
        
    # Read the uploaded file bytes
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded audio: {e}")

    async def voice_sse_generator():
        try:
            # 1. Execute Speech-to-Text Transcription via Sarvam AI
            print("Invoking Sarvam AI Speech-to-Text...", flush=True)
            transcript, stt_latency = stt_engine.transcribe(audio_bytes, filename=file.filename)
            
            # 2. Handle STT Failure Modes explicitly
            if transcript is None:
                # Case A: API call failed / timeout -> return fallback answer
                log_stt_latency(file.filename, "ERROR", stt_latency, "")
                print("STT Failure: Sarvam API call failed or timed out. Routing to fallback.", file=sys.stderr)
                yield format_sse({
                    "event": "stt_error",
                    "latency_ms": stt_latency,
                    "fallback_answer": "I don't know based on the provided context."
                })
                return
                
            if transcript == "":
                # Case B: Completed successfully but empty (silent/noisy audio) -> return fallback answer
                log_stt_latency(file.filename, "SILENT", stt_latency, "")
                print("STT Warning: Empty transcription returned (possible silence). Routing to fallback.", file=sys.stderr)
                yield format_sse({
                    "event": "silent_audio",
                    "latency_ms": stt_latency,
                    "fallback_answer": "I don't know based on the provided context."
                })
                return
                
            # Case C: Successful transcription -> yield transcript and continue to RAG pipeline
            log_stt_latency(file.filename, "SUCCESS", stt_latency, transcript)
            print(f"STT Success: Transcribed = '{transcript}' in {stt_latency:.2f} ms", flush=True)
            yield format_sse({
                "event": "stt_complete",
                "text": transcript,
                "latency_ms": stt_latency
            })
            await asyncio.sleep(0.01)

            # 3. Stream RAG Pipeline Events using the transcribed text
            events = pipeline.query_stream_events(transcript)
            for event in events:
                yield format_sse(event)
                await asyncio.sleep(0.01)

        except Exception as e:
            print(f"Error in voice query endpoint: {e}", file=sys.stderr)
            yield format_sse({"event": "error", "message": str(e)})

    return StreamingResponse(voice_sse_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Start FastAPI server on port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
