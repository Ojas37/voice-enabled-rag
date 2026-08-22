import os
import sys
import time
import urllib.request
import pandas as pd
import numpy as np
import lancedb
import pyarrow as pa
import torch
from transformers import AutoModel, AutoTokenizer
from chunking import Chunker

# Cloud config
MODEL_ID = "intfloat/multilingual-e5-small"  # 384 dimensions, ~130MB on CPU
DB_DIR = "data/lancedb_cloud"
TABLE_NAME = "multilingual_passages"
LIMIT_ROWS = 15  # Smaller subset for sub-20s startup indexing on CPU

DATA_FILES = {
    "hinval_real_mini.parquet": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet",
    "marval_real_mini.parquet": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/marval.parquet"
}

def download_data():
    os.makedirs("data", exist_ok=True)
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    
    for filename, url in DATA_FILES.items():
        path = os.path.join("data", filename)
        if not os.path.exists(path):
            print(f"Downloading validation dataset: {filename}...", flush=True)
            urllib.request.urlretrieve(url, path)
            print(f"Finished downloading {filename}.", flush=True)

def load_passages(limit_rows):
    df_hi = pd.read_parquet("data/hinval_real_mini.parquet").head(limit_rows)
    df_mr = pd.read_parquet("data/marval_real_mini.parquet").head(limit_rows)
    
    passages_list = []
    seen_english = set()
    
    # Process Hindi file (contains English and Hindi)
    for idx, row in df_hi.iterrows():
        q_id = row['query_id']
        eng_q = row['Eng_Query']
        hi_q = row['query']
        q_type = row.get('query_type', 'general')
        
        eng_pass = row['passages']['English_passages']
        hi_pass = row['passages']['Translated_passages']
        
        for i in range(len(eng_pass)):
            if (q_id, i) not in seen_english:
                passages_list.append(Chunker(q_id, i, "en", eng_pass[i], eng_q, q_type))
                seen_english.add((q_id, i))
                
            if i < len(hi_pass):
                passages_list.append(Chunker(q_id, i, "hi", hi_pass[i], hi_q, q_type))
                
    # Process Marathi file (contains Marathi)
    for idx, row in df_mr.iterrows():
        q_id = row['query_id']
        mar_q = row['query']
        q_type = row.get('query_type', 'general')
        
        mar_pass = row['passages']['Translated_passages']
        for i in range(len(mar_pass)):
            passages_list.append(Chunker(q_id, i, "mr", mar_pass[i], mar_q, q_type))
            
    return passages_list

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_texts(texts, tokenizer, model, batch_size=32):
    prefixed_texts = ["passage: " + t for t in texts]
    all_embeddings = []
    
    for i in range(0, len(prefixed_texts), batch_size):
        batch = prefixed_texts[i:i+batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        normalized = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())
        
    return np.vstack(all_embeddings)

def bootstrap():
    db_path = os.path.join(DB_DIR, f"{TABLE_NAME}.lance")
    if os.path.exists(db_path):
        print("Cloud database already exists. Bootstrapping skipped.", flush=True)
        return
        
    print("Starting database bootstrap for cloud deployment...", flush=True)
    t0 = time.time()
    
    download_data()
    passages = load_passages(LIMIT_ROWS)
    
    print("Generating chunks...", flush=True)
    all_chunks = []
    for p in passages:
        all_chunks.extend(p.metadata_aware(max_char_len=300))
        
    print(f"Generated {len(all_chunks)} chunks. Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    
    print("Embedding chunks...", flush=True)
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(chunk_texts, tokenizer, model, batch_size=32)
    
    print("Writing to LanceDB...", flush=True)
    db = lancedb.connect(DB_DIR)
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 384)), # 384 dimensions
        pa.field("chunk_id", pa.string()),
        pa.field("query_id", pa.int64()),
        pa.field("passage_index", pa.int32()),
        pa.field("language", pa.string()),
        pa.field("text", pa.string()),
        pa.field("raw_body", pa.string())
    ])
    
    tbl = db.create_table(TABLE_NAME, schema=schema, mode="overwrite")
    
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
    
    # Build BTree scalar index on language column
    print("Building scalar BTree index on language column...", flush=True)
    tbl.create_scalar_index("language")
    
    print(f"Bootstrap completed successfully in {time.time() - t0:.2f} seconds.", flush=True)

if __name__ == "__main__":
    bootstrap()
