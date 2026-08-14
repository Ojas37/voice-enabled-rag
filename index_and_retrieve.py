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
from transformers import AutoTokenizer

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Import the chunking logic from chunking.py
from chunking import Chunker

# ----------------- Configuration -----------------
LIMIT_ROWS = 300         # Under 1k rows per language for rapid, responsive benchmarking
NUM_EVAL_SAMPLES = 100   # Number of validation queries to run per language
MODEL_DIR = "./model_onnx"
DB_DIR = "data/lancedb"
TABLE_NAME = "multilingual_passages"

# ----------------- Load & Sub-sample base passages -----------------
def load_passages_and_queries(limit_rows=300):
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
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # L2 normalize embeddings
        normalized = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())
        
    return np.vstack(all_embeddings)

# ----------------- Main Pipeline -----------------
def main():
    # 1. Setup ONNX Runtime Session Options for Multithreading Optimization on CPU
    print("\nConfiguring CPU multithreading options for ONNX Runtime...", flush=True)
    options = SessionOptions()
    cores = multiprocessing.cpu_count()
    print(f"Detected {cores} CPU cores. Setting intra_op_num_threads to {cores} for parallel inference...", flush=True)
    options.intra_op_num_threads = cores
    options.inter_op_num_threads = 1
    
    # 2. Load ONNX Model and Tokenizer
    print("Loading local ONNX model...", flush=True)
    t0 = time.time()
    model = ORTModelForFeatureExtraction.from_pretrained(MODEL_DIR, session_options=options)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    print(f"ONNX Model loaded in {time.time() - t0:.2f} seconds.", flush=True)
    
    # 3. Extract and Chunk Base Passages
    passages, eval_queries = load_passages_and_queries(LIMIT_ROWS)
    
    print("\nGenerating metadata-aware chunks from base passages...", flush=True)
    all_chunks = []
    for p in passages:
        # Use our safe Metadata-Aware chunker
        chunks = p.metadata_aware(max_char_len=300)
        all_chunks.extend(chunks)
    print(f"Total metadata-aware chunks generated: {len(all_chunks)}", flush=True)
    
    # 4. Setup LanceDB Table
    print("\nSetting up LanceDB database...", flush=True)
    os.makedirs(DB_DIR, exist_ok=True)
    db = lancedb.connect(DB_DIR)
    
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 384)), # Dimension for multilingual-e5-small
        pa.field("chunk_id", pa.string()),
        pa.field("query_id", pa.int64()),
        pa.field("passage_index", pa.int32()),
        pa.field("language", pa.string()),
        pa.field("text", pa.string()),
        pa.field("raw_body", pa.string())
    ])
    
    tbl = db.create_table(TABLE_NAME, schema=schema, mode="overwrite")
    
    # 5. Generate Embeddings & Index in batches
    batch_size = 32  # Reduced batch size for CPU cache efficiency and speed
    print(f"\nEmbedding and indexing {len(all_chunks)} chunks in batches of {batch_size} on CPU...", flush=True)
    
    t_start_indexing = time.time()
    
    chunk_texts = [c["text"] for c in all_chunks]
    
    # We embed chunks in batches to show progress
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
    indexing_duration = time.time() - t_start_indexing
    print(f"Indexing complete! Added {len(records)} records in {indexing_duration:.2f} seconds ({len(records)/indexing_duration:.2f} chunks/sec).", flush=True)
    
    # 6. Evaluate Retrieval Recall & Latency per language
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION BENCHMARK")
    print("="*60)
    
    for lang in ["en", "hi", "mr"]:
        lang_queries = eval_queries[lang]
        if len(lang_queries) < NUM_EVAL_SAMPLES:
            sampled_queries = lang_queries
        else:
            # Sample queries deterministically for reproducibility
            random.seed(42)
            sampled_queries = random.sample(lang_queries, NUM_EVAL_SAMPLES)
            
        print(f"\nEvaluating {len(sampled_queries)} queries for language: {lang.upper()}...", flush=True)
        
        hits = 0
        emb_latencies = []
        search_latencies = []
        total_latencies = []
        
        for q in tqdm(sampled_queries, desc=f"Querying {lang.upper()}"):
            q_text = q["query_text"]
            q_id = q["query_id"]
            gt_idx = q["gt_index"]
            
            # Step A: Embed the query
            t_emb_start = time.perf_counter()
            q_vector = embed_texts([q_text], tokenizer, model, batch_size=1, is_query=True)[0]
            t_emb_end = time.perf_counter()
            
            # Step B: Perform LanceDB search
            t_search_start = time.perf_counter()
            results = tbl.search(q_vector.tolist()).limit(5).to_pandas()
            t_search_end = time.perf_counter()
            
            # Record latencies in milliseconds
            emb_lat = (t_emb_end - t_emb_start) * 1000.0
            search_lat = (t_search_end - t_search_start) * 1000.0
            
            emb_latencies.append(emb_lat)
            search_latencies.append(search_lat)
            total_latencies.append(emb_lat + search_lat)
            
            # Step C: Evaluate Recall@5 (whether the ground-truth passage was retrieved)
            matches = results[(results["query_id"] == q_id) & (results["passage_index"] == gt_idx)]
            if len(matches) > 0:
                hits += 1
                
        recall = (hits / len(sampled_queries)) * 100.0
        avg_emb_lat = np.mean(emb_latencies)
        avg_search_lat = np.mean(search_latencies)
        avg_total_lat = np.mean(total_latencies)
        p95_total_lat = np.percentile(total_latencies, 95)
        
        print(f"\nResults for {lang.upper()}:")
        print(f"  Recall@5:                {recall:.2f}%")
        print(f"  Avg Embedding Latency:   {avg_emb_lat:.2f} ms")
        print(f"  Avg Vector DB Search:     {avg_search_lat:.2f} ms")
        print(f"  Avg Total Retrieval:     {avg_total_lat:.2f} ms")
        print(f"  95th Percentile Retrieval: {p95_total_lat:.2f} ms")

if __name__ == "__main__":
    main()
