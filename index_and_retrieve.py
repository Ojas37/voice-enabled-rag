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

# Import the chunking logic from chunking.py
from chunking import Chunker

# ----------------- Configuration -----------------
# Approved: Index the full 5,000 rows per language!
LIMIT_ROWS = 5000
NUM_EVAL_SAMPLES = 100   # Number of validation queries to run per language
MODEL_ID = "intfloat/multilingual-e5-base"  # Approved winner of Phase 3 comparative recall test
DB_DIR = "data/lancedb"
TABLE_NAME = "multilingual_passages"

# Device detection (use CUDA GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device configuration: {DEVICE.type.upper()} active.", flush=True)

# ----------------- Load & Sub-sample base passages -----------------
def load_passages_and_queries(limit_rows=5000):
    print(f"Loading first {limit_rows} rows from parquet files...", flush=True)
    df_hi = pd.read_parquet("data/hinval_real_mini.parquet").head(limit_rows)
    df_mr = pd.read_parquet("data/marval_real_mini.parquet").head(limit_rows)
    
    passages_list = []
    seen_english = set()
    
    # Store queries for the evaluation stage
    eval_queries = {
        "en": [],
        "hi": [],
        "mr": []
    }
    
    # Process Hindi file (contains English and Hindi)
    for idx, row in df_hi.iterrows():
        q_id = row['query_id']
        eng_q = row['Eng_Query']
        hi_q = row['query']
        q_type = row.get('query_type', 'general')
        
        # Ground-truth passage selection
        is_selected = list(row['passages']['is_selected'])
        gt_indices = [i for i, val in enumerate(is_selected) if val == 1]
        
        # Add queries for evaluation
        if gt_indices:
            eval_queries["en"].append({"query_id": q_id, "query_text": eng_q, "gt_index": gt_indices[0]})
            eval_queries["hi"].append({"query_id": q_id, "query_text": hi_q, "gt_index": gt_indices[0]})
            
        eng_pass = row['passages']['English_passages']
        hi_pass = row['passages']['Translated_passages']
        
        for i in range(len(eng_pass)):
            # 1. Add English passage (if not seen)
            if (q_id, i) not in seen_english:
                passages_list.append(Chunker(q_id, i, "en", eng_pass[i], eng_q, q_type))
                seen_english.add((q_id, i))
                
            # 2. Add Hindi passage
            if i < len(hi_pass):
                passages_list.append(Chunker(q_id, i, "hi", hi_pass[i], hi_q, q_type))
                
    # Process Marathi file (contains Marathi)
    for idx, row in df_mr.iterrows():
        q_id = row['query_id']
        mar_q = row['query']
        q_type = row.get('query_type', 'general')
        
        is_selected = list(row['passages']['is_selected'])
        gt_indices = [i for i, val in enumerate(is_selected) if val == 1]
        
        # Add queries for evaluation
        if gt_indices:
            eval_queries["mr"].append({"query_id": q_id, "query_text": mar_q, "gt_index": gt_indices[0]})
            
        mar_pass = row['passages']['Translated_passages']
        for i in range(len(mar_pass)):
            passages_list.append(Chunker(q_id, i, "mr", mar_pass[i], mar_q, q_type))
            
    print(f"Extracted {len(passages_list)} total base passages across 3 languages.", flush=True)
    return passages_list, eval_queries

# ----------------- Text Embeddings -----------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_texts(texts, tokenizer, model, batch_size=32, is_query=False):
    prefix = "query: " if is_query else "passage: "
    prefixed_texts = [prefix + t for t in texts]
    
    all_embeddings = []
    for i in range(0, len(prefixed_texts), batch_size):
        batch = prefixed_texts[i:i+batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
        # Move tensors to GPU if active
        encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # L2 normalize embeddings
        normalized = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())
        
    return np.vstack(all_embeddings)

# ----------------- Main Pipeline -----------------
def main():
    # 1. Load Model & Tokenizer based on device
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if DEVICE.type == "cuda":
        print(f"\nLoading native PyTorch model {MODEL_ID} on GPU (CUDA)...", flush=True)
        model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    else:
        # Fallback to local CPU ONNX
        print("\nFallback CPU: Loading local ONNX model...", flush=True)
        options = SessionOptions()
        cores = multiprocessing.cpu_count()
        options.intra_op_num_threads = cores
        options.inter_op_num_threads = 1
        model = ORTModelForFeatureExtraction.from_pretrained("./model_onnx_base", session_options=options)
        
    print(f"Model loaded in {time.time() - t0:.2f} seconds.", flush=True)
    
    # 2. Extract and Chunk Base Passages
    passages, eval_queries = load_passages_and_queries(LIMIT_ROWS)
    
    print("\nGenerating metadata-aware chunks from base passages...", flush=True)
    all_chunks = []
    for p in tqdm(passages, desc="Chunking Passages"):
        # Use our safe Metadata-Aware chunker
        chunks = p.metadata_aware(max_char_len=300)
        all_chunks.extend(chunks)
    print(f"Total metadata-aware chunks generated: {len(all_chunks)}", flush=True)
    
    # 3. Setup LanceDB Table (768 dimensions for E5-Base)
    print("\nSetting up LanceDB database...", flush=True)
    os.makedirs(DB_DIR, exist_ok=True)
    db = lancedb.connect(DB_DIR)
    
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 768)), # E5-Base dimension is 768
        pa.field("chunk_id", pa.string()),
        pa.field("query_id", pa.int64()),
        pa.field("passage_index", pa.int32()),
        pa.field("language", pa.string()),
        pa.field("text", pa.string()),
        pa.field("raw_body", pa.string())
    ])
    
    tbl = db.create_table(TABLE_NAME, schema=schema, mode="overwrite")
    
    # 4. Generate Embeddings & Index in batches
    # GPU: Safe batch size of 32 to avoid driver paging and VRAM swapping
    batch_size = 32
    print(f"\nEmbedding and indexing {len(all_chunks)} chunks in batches of {batch_size} on {DEVICE.type.upper()}...", flush=True)
    
    t_start_indexing = time.time()
    
    chunk_texts = [c["text"] for c in all_chunks]
    
    embeddings = []
    for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Embedding Chunks"):
        batch = chunk_texts[i:i+batch_size]
        batch_emb = embed_texts(batch, tokenizer, model, batch_size=batch_size, is_query=False)
        embeddings.append(batch_emb)
        
    embeddings = np.vstack(embeddings)
    
    # Build list of dictionaries for pyarrow
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
    
    # 5. Build IVF-PQ Approximate Nearest Neighbors index to optimize search speed
    print("\nBuilding IVF-PQ ANN index for sub-10ms query execution...", flush=True)
    t_index_start = time.perf_counter()
    # Simple IvyPq index setup (standard LanceDB configuration)
    tbl.create_index(
        num_partitions=256,        # Larger partitions for 300,000 vectors
        num_sub_vectors=192,       # Quantization subvectors
        metric="cosine",
        replace=True
    )
    t_index_build = time.perf_counter() - t_index_start
    indexing_duration = time.time() - t_start_indexing
    
    print(f"Indexing complete! Added {len(records)} records in {indexing_duration:.2f} seconds (Index build: {t_index_build:.2f}s).", flush=True)
    print(f"Average throughput: {len(records)/indexing_duration:.2f} chunks/sec.", flush=True)
    
    # 6. Evaluate Retrieval Recall & Latency per language (Recall@8 for win strategy)
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION BENCHMARK (Recall@8)")
    print("="*60)
    
    for lang in ["en", "hi", "mr"]:
        lang_queries = eval_queries[lang]
        if len(lang_queries) < NUM_EVAL_SAMPLES:
            sampled_queries = lang_queries
        else:
            random.seed(42)
            sampled_queries = random.sample(lang_queries, NUM_EVAL_SAMPLES)
            
        print(f"\nEvaluating {len(sampled_queries)} queries for language: {lang.upper()}...", flush=True)
        
        hits_at_8 = 0
        total_latencies = []
        
        for q in tqdm(sampled_queries, desc=f"Querying {lang.upper()}"):
            q_text = q["query_text"]
            q_id = q["query_id"]
            gt_idx = q["gt_index"]
            
            t0 = time.perf_counter()
            q_vector = embed_texts([q_text], tokenizer, model, batch_size=1, is_query=True)[0]
            # Retrieve 8 results to align with approved strategy
            results = tbl.search(q_vector.tolist()).limit(8).to_pandas()
            lat = (time.perf_counter() - t0) * 1000.0
            
            total_latencies.append(lat)
            
            matches = results[(results["query_id"] == q_id) & (results["passage_index"] == gt_idx)]
            if len(matches) > 0:
                hits_at_8 += 1
                
        recall = (hits_at_8 / len(sampled_queries)) * 100.0
        avg_total_lat = np.mean(total_latencies)
        p95_total_lat = np.percentile(total_latencies, 95)
        
        print(f"\nResults for {lang.upper()}:")
        print(f"  Recall@8:                {recall:.2f}%")
        print(f"  Avg Total Retrieval:     {avg_total_lat:.2f} ms")
        print(f"  95th Percentile Retrieval: {p95_total_lat:.2f} ms")

if __name__ == "__main__":
    main()
