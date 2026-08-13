import os
import sys
import re
import time
import pandas as pd
import numpy as np

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Helper: Regex-based multilingual sentence splitter (handles danda '।' and English punctuation)
def split_sentences(text):
    if not text:
        return []
    # Split on punctuation followed by whitespace or end of string
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# ----------------- Chunker class -----------------
class Chunker:
    def __init__(self, query_id, passage_index, language, text, query_text=None):
        self.query_id = query_id
        self.passage_index = passage_index
        self.language = language
        self.text = text
        self.query_text = query_text

    # 1. Naive Fixed-Size Chunking (Character-based with overlap)
    def naive_fixed_size(self, chunk_size=200, overlap=50):
        chunks = []
        text_len = len(self.text)
        if text_len <= chunk_size:
            return [{
                "chunk_id": f"{self.query_id}_{self.passage_index}_{self.language}_naive_0",
                "query_id": self.query_id,
                "passage_index": self.passage_index,
                "language": self.language,
                "text": self.text,
                "strategy": "naive"
            }]
            
        start = 0
        seq = 0
        while start < text_len:
            end = start + chunk_size
            chunk_text = self.text[start:end]
            chunks.append({
                "chunk_id": f"{self.query_id}_{self.passage_index}_{self.language}_naive_{seq}",
                "query_id": self.query_id,
                "passage_index": self.passage_index,
                "language": self.language,
                "text": chunk_text,
                "strategy": "naive"
            })
            start += (chunk_size - overlap)
            seq += 1
        return chunks

    # 2. Sentence/Semantic-Based Splitting (Grouping sentences up to max character limit)
    def sentence_based(self, max_char_len=300):
        sentences = split_sentences(self.text)
        if not sentences:
            return []
            
        chunks = []
        current_chunk = []
        current_len = 0
        seq = 0
        
        for s in sentences:
            s_len = len(s)
            # If adding this sentence exceeds limit and current_chunk is not empty, flush
            if current_len + s_len > max_char_len and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "chunk_id": f"{self.query_id}_{self.passage_index}_{self.language}_sentence_{seq}",
                    "query_id": self.query_id,
                    "passage_index": self.passage_index,
                    "language": self.language,
                    "text": chunk_text,
                    "strategy": "sentence"
                })
                current_chunk = []
                current_len = 0
                seq += 1
            
            current_chunk.append(s)
            current_len += s_len + 1  # +1 for space join
            
        # Flush final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": f"{self.query_id}_{self.passage_index}_{self.language}_sentence_{seq}",
                "query_id": self.query_id,
                "passage_index": self.passage_index,
                "language": self.language,
                "text": chunk_text,
                "strategy": "sentence"
            })
        return chunks

    # 3. Metadata-Aware Chunking (Splitting logically and augmenting text with context prefix)
    def metadata_aware(self, max_char_len=250):
        # We split by sentences, but we prepend a contextual prefix (e.g. language, query_id)
        # to the chunk text itself to enrich its vector representation!
        sentences = split_sentences(self.text)
        if not sentences:
            return []
            
        chunks = []
        current_chunk = []
        current_len = 0
        seq = 0
        
        # Base context to prepend to chunk text
        lang_name = "English" if self.language == "en" else ("Hindi" if self.language == "hi" else "Marathi")
        context_prefix = f"Language: {lang_name} | Query context: {self.query_text} | Passage text: " if self.query_text else f"Language: {lang_name} | Passage text: "
        
        for s in sentences:
            s_len = len(s)
            # We measure chunk size including the prefix!
            if len(context_prefix) + current_len + s_len > max_char_len and current_chunk:
                chunk_body = " ".join(current_chunk)
                full_text = f"{context_prefix}{chunk_body}"
                chunks.append({
                    "chunk_id": f"{self.query_id}_{self.passage_index}_{self.language}_meta_{seq}",
                    "query_id": self.query_id,
                    "passage_index": self.passage_index,
                    "language": self.language,
                    "text": full_text,
                    "strategy": "metadata_aware",
                    "raw_body": chunk_body  # preserve clean raw text
                })
                current_chunk = []
                current_len = 0
                seq += 1
                
            current_chunk.append(s)
            current_len += s_len + 1
            
        if current_chunk:
            chunk_body = " ".join(current_chunk)
            full_text = f"{context_prefix}{chunk_body}"
            chunks.append({
                "chunk_id": f"{self.query_id}_{self.passage_index}_{self.language}_meta_{seq}",
                "query_id": self.query_id,
                "passage_index": self.passage_index,
                "language": self.language,
                "text": full_text,
                "strategy": "metadata_aware",
                "raw_body": chunk_body
            })
        return chunks

# ----------------- Load & Process -----------------
def load_base_passages():
    print("Loading datasets...", flush=True)
    df_hi = pd.read_parquet("data/hinval_real_mini.parquet")
    df_mr = pd.read_parquet("data/marval_real_mini.parquet")
    
    print(f"Loaded {df_hi.shape[0]} Hindi rows and {df_mr.shape[0]} Marathi rows.", flush=True)
    
    passages_list = []
    
    # Track unique English passages using (query_id, passage_index) to avoid duplicates
    seen_english = set()
    
    # Process Hindi file (contains English and Hindi)
    for _, row in df_hi.iterrows():
        q_id = row['query_id']
        eng_q = row['Eng_Query']
        hi_q = row['query']
        
        eng_pass = row['passages']['English_passages']
        hi_pass = row['passages']['Translated_passages']
        
        for i in range(len(eng_pass)):
            # 1. Add English passage (if not seen)
            if (q_id, i) not in seen_english:
                passages_list.append(Chunker(q_id, i, "en", eng_pass[i], eng_q))
                seen_english.add((q_id, i))
                
            # 2. Add Hindi passage
            if i < len(hi_pass):
                passages_list.append(Chunker(q_id, i, "hi", hi_pass[i], hi_q))
                
    # Process Marathi file (contains Marathi)
    for _, row in df_mr.iterrows():
        q_id = row['query_id']
        mar_q = row['query']
        
        # We don't extract English passages from Marathi file because they are identical to Hindi file
        mar_pass = row['passages']['Translated_passages']
        for i in range(len(mar_pass)):
            passages_list.append(Chunker(q_id, i, "mr", mar_pass[i], mar_q))
            
    print(f"Extracted {len(passages_list)} total base passages across 3 languages.", flush=True)
    # Count breakdown
    en_cnt = sum(1 for p in passages_list if p.language == "en")
    hi_cnt = sum(1 for p in passages_list if p.language == "hi")
    mr_cnt = sum(1 for p in passages_list if p.language == "mr")
    print(f"  English base passages: {en_cnt}")
    print(f"  Hindi base passages:   {hi_cnt}")
    print(f"  Marathi base passages: {mr_cnt}")
    
    return passages_list

def evaluate_strategies(passages):
    strategies = ["naive", "sentence", "metadata_aware"]
    results = {}
    
    for strat in strategies:
        t0 = time.time()
        all_chunks = []
        
        for p in passages:
            if strat == "naive":
                chunks = p.naive_fixed_size(chunk_size=200, overlap=50)
            elif strat == "sentence":
                chunks = p.sentence_based(max_char_len=300)
            elif strat == "metadata_aware":
                chunks = p.metadata_aware(max_char_len=300)
            all_chunks.extend(chunks)
            
        duration = time.time() - t0
        sizes = [len(c['text']) for c in all_chunks]
        
        results[strat] = {
            "chunks": all_chunks,
            "count": len(all_chunks),
            "avg_size": np.mean(sizes) if sizes else 0,
            "min_size": np.min(sizes) if sizes else 0,
            "max_size": np.max(sizes) if sizes else 0,
            "time": duration
        }
        
    print("\n" + "="*50)
    print("COMPARATIVE CHUNKING ANALYSIS")
    print("="*50)
    
    for strat, res in results.items():
        print(f"\nStrategy: {strat.upper()}")
        print(f"  Total Chunks Generated: {res['count']}")
        print(f"  Average Chunk Size:      {res['avg_size']:.2f} chars")
        print(f"  Min/Max Chunk Size:      {res['min_size']} / {res['max_size']} chars")
        print(f"  Chunking Speed:          {res['time']:.4f} seconds")

    # Show example outputs for each strategy across languages
    print("\n" + "="*50)
    print("SAMPLE OUTPUTS PER STRATEGY AND LANGUAGE")
    print("="*50)
    
    languages = ["en", "hi", "mr"]
    
    for strat in strategies:
        print(f"\n>>> Strategy: {strat.upper()} <<<")
        for lang in languages:
            # Find first chunk of this language
            sample_chunk = None
            for c in results[strat]["chunks"]:
                if c["language"] == lang:
                    sample_chunk = c
                    break
            
            if sample_chunk:
                print(f"\n  Language: {lang.upper()}")
                print(f"  Chunk ID: {sample_chunk['chunk_id']}")
                print(f"  Chunk Text:\n    \"{sample_chunk['text']}\"")
                print(f"  Length: {len(sample_chunk['text'])} chars")
            else:
                print(f"\n  Language: {lang.upper()} - No sample found.")
                
if __name__ == "__main__":
    passages = load_base_passages()
    evaluate_strategies(passages)
