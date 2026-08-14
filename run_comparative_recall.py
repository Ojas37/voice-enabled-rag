import os
import sys
import time
import random
import multiprocessing
import numpy as np
import pandas as pd
import lancedb
import pyarrow as pa
import torch
from tqdm import tqdm
from onnxruntime import SessionOptions
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoModel, AutoTokenizer

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Import the Chunker class
from chunking import Chunker

# ----------------- Configuration -----------------
LIMIT_ROWS = 1000        # CUDA enabled: 1k rows (~64k chunks) will index in seconds on GPU!
NUM_EVAL_SAMPLES = 100   # Number of validation queries to run per language
DB_DIR_PREFIX = "data/lancedb_compare"

MODELS_CONFIG = {
    "small": {
        "model_id": "intfloat/multilingual-e5-small",
        "onnx_dir": "./model_onnx_small",
        "db_path": f"{DB_DIR_PREFIX}_small",
        "dimension": 384
    },
    "base": {
        "model_id": "intfloat/multilingual-e5-base",
        "onnx_dir": "./model_onnx_base",
        "db_path": f"{DB_DIR_PREFIX}_base",
        "dimension": 768
    }
}

# Device detection (use GPU if CUDA is available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device configuration: {DEVICE.type.upper()} active.", flush=True)

# ----------------- Load Parquet rows -----------------
def load_passages_and_queries(limit_rows=1000):
    print(f"Loading first {limit_rows} rows from parquet...", flush=True)
    df_hi = pd.read_parquet("data/hinval_real_mini.parquet").head(limit_rows)
    df_mr = pd.read_parquet("data/marval_real_mini.parquet").head(limit_rows)
    
    passages_list = []
    seen_english = set()
    eval_queries = {"en": [], "hi": [], "mr": []}
    
    for idx, row in df_hi.iterrows():
        q_id = row['query_id']
        eng_q = row['Eng_Query']
        hi_q = row['query']
        q_type = row.get('query_type', 'general')
        
        is_selected = list(row['passages']['is_selected'])
        gt_indices = [i for i, val in enumerate(is_selected) if val == 1]
        
        if gt_indices:
            eval_queries["en"].append({"query_id": q_id, "query_text": eng_q, "gt_index": gt_indices[0]})
            eval_queries["hi"].append({"query_id": q_id, "query_text": hi_q, "gt_index": gt_indices[0]})
            
        eng_pass = row['passages']['English_passages']
        hi_pass = row['passages']['Translated_passages']
        
        for i in range(len(eng_pass)):
            if (q_id, i) not in seen_english:
                passages_list.append(Chunker(q_id, i, "en", eng_pass[i], eng_q, q_type))
                seen_english.add((q_id, i))
            if i < len(hi_pass):
                passages_list.append(Chunker(q_id, i, "hi", hi_pass[i], hi_q, q_type))
                
    for idx, row in df_mr.iterrows():
        q_id = row['query_id']
        mar_q = row['query']
        q_type = row.get('query_type', 'general')
        
        is_selected = list(row['passages']['is_selected'])
        gt_indices = [i for i, val in enumerate(is_selected) if val == 1]
        
        if gt_indices:
            eval_queries["mr"].append({"query_id": q_id, "query_text": mar_q, "gt_index": gt_indices[0]})
            
        mar_pass = row['passages']['Translated_passages']
        for i in range(len(mar_pass)):
            passages_list.append(Chunker(q_id, i, "mr", mar_pass[i], mar_q, q_type))
            
    return passages_list, eval_queries

# ----------------- Export ONNX Helper (Fallback for CPU) -----------------
def ensure_onnx_model(model_name, save_dir):
    if DEVICE.type == "cpu":
        if not os.path.exists(save_dir) or not os.listdir(save_dir):
            print(f"\nONNX model not found in {save_dir}. Exporting {model_name}...", flush=True)
            os.makedirs(save_dir, exist_ok=True)
            model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"ONNX model saved successfully to {save_dir}!", flush=True)

# ----------------- Text Embeddings -----------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_texts(texts, tokenizer, model, batch_size=64, is_query=False):
    prefix = "query: " if is_query else "passage: "
    prefixed_texts = [prefix + t for t in texts]
    
    all_embeddings = []
    # Use larger batch size on GPU for parallel acceleration
    effective_batch_size = 128 if DEVICE.type == "cuda" else batch_size
    
    for i in range(0, len(prefixed_texts), effective_batch_size):
        batch = prefixed_texts[i:i+effective_batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
        # Move tensors to target device (GPU/CPU)
        encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        normalized = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())
        
    return np.vstack(all_embeddings)

# ----------------- Indexing and Evaluating -----------------
def run_model_pipeline(model_key, passages, eval_queries):
    config = MODELS_CONFIG[model_key]
    
    # 1. Load Model & Tokenizer based on hardware provider
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if DEVICE.type == "cuda":
        print(f"\nLoading native PyTorch model {config['model_id']} on GPU (CUDA)...", flush=True)
        model = AutoModel.from_pretrained(config["model_id"]).to(DEVICE)
    else:
        # CPU Fallback: Load ONNX Model
        ensure_onnx_model(config["model_id"], config["onnx_dir"])
        print(f"\nConfiguring ONNX session options for CPU execution...", flush=True)
        options = SessionOptions()
        cores = multiprocessing.cpu_count()
        options.intra_op_num_threads = cores
        options.inter_op_num_threads = 1
        model = ORTModelForFeatureExtraction.from_pretrained(config["onnx_dir"], session_options=options)
        
    # 2. Chunk base passages
    all_chunks = []
    for p in passages:
        chunks = p.metadata_aware(max_char_len=300)
        all_chunks.extend(chunks)
    print(f"Total chunks generated: {len(all_chunks)}", flush=True)
    
    # 3. Embed and Index in LanceDB
    print(f"Embedding chunks using {model_key} model on {DEVICE.type.upper()}...", flush=True)
    t_start_indexing = time.time()
    chunk_texts = [c["text"] for c in all_chunks]
    
    # Batch sizes: 128 for GPU, 32 for CPU cache optimization
    run_batch = 128 if DEVICE.type == "cuda" else 32
    
    embeddings = []
    for i in tqdm(range(0, len(chunk_texts), run_batch), desc=f"Embedding Chunks ({model_key})"):
        batch = chunk_texts[i:i+run_batch]
        batch_emb = embed_texts(batch, tokenizer, model, batch_size=run_batch, is_query=False)
        embeddings.append(batch_emb)
    embeddings = np.vstack(embeddings)
    
    print("Writing to LanceDB...", flush=True)
    db = lancedb.connect(config["db_path"])
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), config["dimension"])),
        pa.field("chunk_id", pa.string()),
        pa.field("query_id", pa.int64()),
        pa.field("passage_index", pa.int32()),
        pa.field("language", pa.string()),
        pa.field("text", pa.string()),
        pa.field("raw_body", pa.string())
    ])
    
    tbl = db.create_table("multilingual_passages", schema=schema, mode="overwrite")
    
    records = []
    for idx, c in enumerate(all_chunks):
        records.append({
            "vector": embeddings[idx].tolist(),
            "chunk_id": c["chunk_id"],
            "query_id": int(c["query_id"]),
            "passage_index": int(c["passage_index"]),
            "language": c["language"],
            "text": c["text"],
            "raw_body": c["raw_body"]
        })
    tbl.add(records)
    
    # Create IVF-PQ ANN Index to guarantee sub-10ms search time
    print("Building IVF-PQ ANN Index...", flush=True)
    t_index_start = time.perf_counter()
    tbl.create_index(
        num_partitions=64, 
        num_sub_vectors=96 if model_key == "small" else 192, 
        metric="cosine",
        replace=True
    )
    t_index_build = time.perf_counter() - t_index_start
    indexing_duration = time.time() - t_start_indexing
    
    print(f"Indexing complete in {indexing_duration:.2f}s (Index build: {t_index_build:.2f}s)", flush=True)
    
    # 4. Evaluate Recall@5, 8, 10
    languages = ["en", "hi", "mr"]
    recall_results = {}
    
    for lang in languages:
        lang_queries = eval_queries[lang]
        random.seed(42)
        sampled = random.sample(lang_queries, min(len(lang_queries), NUM_EVAL_SAMPLES))
        
        hits_at_5 = 0
        hits_at_8 = 0
        hits_at_10 = 0
        total_latencies = []
        
        for q in sampled:
            q_text = q["query_text"]
            q_id = q["query_id"]
            gt_idx = q["gt_index"]
            
            t0 = time.perf_counter()
            # Embed query on GPU/CPU
            q_vector = embed_texts([q_text], tokenizer, model, batch_size=1, is_query=True)[0]
            # Search LanceDB using the IVF-PQ Index
            results = tbl.search(q_vector.tolist()).limit(10).to_pandas()
            lat = (time.perf_counter() - t0) * 1000.0
            total_latencies.append(lat)
            
            # Check Recall at K
            # Recall@5
            res_at_5 = results.head(5)
            if len(res_at_5[(res_at_5["query_id"] == q_id) & (res_at_5["passage_index"] == gt_idx)]) > 0:
                hits_at_5 += 1
            # Recall@8
            res_at_8 = results.head(8)
            if len(res_at_8[(res_at_8["query_id"] == q_id) & (res_at_8["passage_index"] == gt_idx)]) > 0:
                hits_at_8 += 1
            # Recall@10
            if len(results[(results["query_id"] == q_id) & (results["passage_index"] == gt_idx)]) > 0:
                hits_at_10 += 1
                
        recall_results[lang] = {
            "recall_5": (hits_at_5 / len(sampled)) * 100.0,
            "recall_8": (hits_at_8 / len(sampled)) * 100.0,
            "recall_10": (hits_at_10 / len(sampled)) * 100.0,
            "avg_latency": np.mean(total_latencies)
        }
        
    return recall_results

# ----------------- Execution -----------------
def main():
    passages, eval_queries = load_passages_and_queries(LIMIT_ROWS)
    
    # Run small model pipeline
    print("\n" + "="*50)
    print("RUNNING MULTILINGUAL-E5-SMALL PIPELINE")
    print("="*50)
    results_small = run_model_pipeline("small", passages, eval_queries)
    
    # Run base model pipeline
    print("\n" + "="*50)
    print("RUNNING MULTILINGUAL-E5-BASE PIPELINE")
    print("="*50)
    results_base = run_model_pipeline("base", passages, eval_queries)
    
    # Print comparison report
    print("\n" + "="*70)
    print("COMPARATIVE RECALL & LATENCY REPORT")
    print("="*70)
    
    headers = f"{'Language':<10} | {'Model':<10} | {'Recall@5':<10} | {'Recall@8':<10} | {'Recall@10':<10} | {'Avg Latency (ms)':<16}"
    print(headers)
    print("-" * len(headers))
    
    for lang in ["en", "hi", "mr"]:
        r_small = results_small[lang]
        r_base = results_base[lang]
        
        print(f"{lang.upper():<10} | {'E5-Small':<10} | {r_small['recall_5']:.2f}% | {r_small['recall_8']:.2f}% | {r_small['recall_10']:.2f}% | {r_small['avg_latency']:.2f} ms")
        print(f"{lang.upper():<10} | {'E5-Base':<10} | {r_base['recall_5']:.2f}% | {r_base['recall_8']:.2f}% | {r_base['recall_10']:.2f}% | {r_base['avg_latency']:.2f} ms")
        print("-" * len(headers))

if __name__ == "__main__":
    main()
