import os
import sys
import time
import requests

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

class SarvamSTT:
    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            print("Warning: SARVAM_API_KEY environment variable not found. Please set it in your .env file.", file=sys.stderr)

    def transcribe(self, audio_bytes: bytes, filename: str = "query.wav", language_code: str = None) -> tuple[str, float]:
        """
        Transcribes the provided audio bytes using Sarvam AI's REST API.
        
        Returns:
            (transcript, latency_ms)
            - transcript: The transcribed text, or empty string/None if failed.
            - latency_ms: The actual API round-trip time in milliseconds.
        """
        if not self.api_key:
            print("Error: Sarvam STT cannot run because SARVAM_API_KEY is not set.", file=sys.stderr)
            return None, 0.0

        headers = {
            "api-subscription-key": self.api_key
        }

        # Prepare files and data payload
        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        data = {
            "model": "saaras:v4"
        }
        if language_code:
            data["language_code"] = language_code

        t0 = time.perf_counter()
        try:
            # We set a strict timeout of 8 seconds to prevent hanging the RAG pipeline
            response = requests.post(SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=8.0)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            
            # Check for HTTP errors
            if response.status_code != 200:
                print(f"Sarvam API Error: Received HTTP status code {response.status_code}. Response: {response.text}", file=sys.stderr)
                return None, latency_ms
                
            resp_json = response.json()
            transcript = resp_json.get("transcript", "").strip()
            
            # If API completed but returned empty transcription (e.g. silence)
            if not transcript:
                print("Sarvam API Warning: STT completed successfully but returned empty transcript (possible silence/noise).", file=sys.stderr)
                return "", latency_ms
                
            return transcript, latency_ms

        except requests.exceptions.Timeout:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print("Sarvam API Error: Request timed out (8.0s limit reached).", file=sys.stderr)
            return None, latency_ms
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Sarvam API Exception: An error occurred during request: {e}", file=sys.stderr)
            return None, latency_ms
