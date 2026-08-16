import os
import sys
import json
import requests

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

API_URL = "http://127.0.0.1:8000/api/query-voice"

def main():
    if len(sys.argv) < 2:
        print("Usage: .venv\\Scripts\\python test_voice_api.py <path_to_audio_file.wav>")
        print("Example: .venv\\Scripts\\python test_voice_api.py data\\sample.wav")
        return
        
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        return

    print(f"Reading audio file: {audio_path}")
    filename = os.path.basename(audio_path)
    
    try:
        with open(audio_path, "rb") as f:
            files = {
                "file": (filename, f, "audio/wav")
            }
            print(f"Sending POST request to {API_URL}...", flush=True)
            response = requests.post(API_URL, files=files, stream=True)
            
            if response.status_code != 200:
                print(f"API Error: Received status code {response.status_code}")
                print(response.text)
                return
                
            print("\n" + "="*50)
            print("STREAMING RESPONSE FROM LOCAL BACKEND")
            print("="*50 + "\n")
            
            # Read Server-Sent Events stream
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        data_json = decoded[6:]
                        try:
                            data = json.loads(data_json)
                            event = data.get("event")
                            
                            if event == "stt_complete":
                                print(f"🎙️ [STT Complete] Transcript: '{data.get('text')}' (API Latency: {data.get('latency_ms'):.2f} ms)\n")
                            elif event == "retrieval_complete":
                                print(f"🔍 [Retrieval Complete] Language: {data.get('language').upper()} | Relevance Score: {data.get('relevance_score'):.4f} (Latency: {data.get('retrieval_ms'):.2f} ms)\n")
                            elif event == "generation_start":
                                print(f"⚡ [Generation Start] Groq TTFT: {data.get('ttft_ms'):.2f} ms")
                                print("💬 Response: ", end="", flush=True)
                            elif event == "token":
                                # If we stream tokens (we currently buffer, but this is future-proof)
                                print(data.get("text"), end="", flush=True)
                            elif event == "borderline_evaluation":
                                print(f"\n⚠️ [Borderline Grounding] Score: {data.get('score'):.4f}. Running LLM check...\n")
                            elif event == "grounding_complete":
                                print(f"\n\n🛡️ [Grounding Complete] Status: {data.get('status')} | Grounding Score: {data.get('score')} | Path Latency: {data.get('latency_s'):.2f} s")
                                print(f"\nFinal Answer:\n{data.get('answer')}")
                            elif event == "stt_error":
                                print(f"❌ [STT Error] Latency: {data.get('latency_ms'):.2f} ms. Routing to fallback.")
                                print(f"Fallback Answer: {data.get('fallback_answer')}")
                            elif event == "silent_audio":
                                print(f"🔇 [Silent Audio] Latency: {data.get('latency_ms'):.2f} ms. Routing to fallback.")
                                print(f"Fallback Answer: {data.get('fallback_answer')}")
                            elif event == "safety_block":
                                print(f"🛑 [Safety Blocked] Pre-Groq check triggered.")
                                print(f"Fallback Answer: {data.get('fallback_answer')}")
                            elif event == "error":
                                print(f"\n❌ [Server Error] {data.get('message')}")
                        except Exception as parse_err:
                            pass
                            
            print("\n\n" + "="*50)
            print("STREAM COMPLETE")
            print("="*50)

    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()
