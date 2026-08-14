import os
import sys
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    model_id = "intfloat/multilingual-e5-small"
    save_dir = "./model_onnx"
    
    print(f"Loading and exporting {model_id} to ONNX format...", flush=True)
    try:
        # Load and automatically export to ONNX
        model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Save locally
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Successfully saved ONNX model and tokenizer to {save_dir}!", flush=True)
    except Exception as e:
        print(f"Error during export: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
