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

# Generic multilingual stop words for local keyword extraction
STOPWORDS = set([
    # English stop words
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", 
    "in", "on", "at", "to", "for", "of", "with", "by", "about", "as", "that", "this", "these", 
    "those", "it", "its", "they", "them", "their", "what", "which", "who", "whom", "how", "why", 
    "where", "when", "can", "could", "will", "would", "shall", "should", "may", "might", "must", 
    "has", "have", "had", "do", "does", "did", "also", "many", "some", "any", "other", "such", 
    "no", "not", "only", "first", "more", "most", "than", "then", "into", "out", "from", "up",
    # Hindi stop words (common Devanagari words)
    "है", "हैं", "था", "थी", "थे", "का", "की", "के", "को", "ने", "से", "में", "पर", "और", "या", 
    "भी", "ही", "तो", "यह", "वह", "जो", "कर", "करना", "करने", "किया", "दिये", "दिया", "इस", 
    "उस", "एक", "दो", "तीन", "चार", "पांच", "सकता", "सकते", "सकती", "हुए", "हुआ", "हुई",
    # Marathi stop words
    "आहे", "आहेत", "होता", "होती", "होते", "चा", "ची", "चे", "च्या", "ला", "ने", "साठी", 
    "मध्ये", "वर", "आणि", "किंवा", "पण", "परंतु", "या", "त्या", "हा", "ही", "हे", "ते", 
    "सर्व", "एक", "दोन", "तीन", "पासून", "पर्यंत"
])

# Local keyword extractor (Devanagari + English support) with length safety checks
def extract_keywords(text, num_keywords=3):
    if not text:
        return "general"
    # Find all words (support Latin alphabet and Devanagari script range \u0900-\u097F)
    words = re.findall(r'[a-zA-Z\u0900-\u097F]+', text.lower())
    # Filter stopwords, short terms, and also enforce max length of 25 characters to avoid run-on garbage words
    filtered = [w for w in words if w not in STOPWORDS and 2 < len(w) <= 25]
    if not filtered:
        return "information"
    
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
        
    sorted_words = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
    # Join and limit total keywords string length to 60 characters to keep prefix bounded
    kw_str = ", ".join(sorted_words[:num_keywords])
    return kw_str[:60]

# Helper: Split long texts to prevent outlier massive chunks
def split_long_text(text, max_len=400, overlap=50):
    if len(text) <= max_len:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += (max_len - overlap)
    return chunks

# Helper: Regex-based multilingual sentence splitter with fallback for long unpunctuated text
def split_sentences(text, max_sentence_len=400):
    if not text:
        return []
    # 1. Split on punctuation followed by space
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    processed = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # 2. Check for sentence outliers (no punctuation or massive run-on blocks)
        if len(s) > max_sentence_len:
            processed.extend(split_long_text(s, max_len=300, overlap=50))
        else:
            processed.append(s)
    return processed

# ----------------- Chunker class -----------------
class Chunker:
    def __init__(self, query_id, passage_index, language, text, query_text=None, query_type="general"):
        self.query_id = query_id
        self.passage_index = passage_index
        self.language = language
        self.text = text
        self.query_text = query_text
        self.query_type = query_type.upper() if query_type else "GENERAL"

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
        # Using the sentence splitter with built-in outlier fallback
        sentences = split_sentences(self.text, max_sentence_len=400)
        if not sentences:
            return []
            
        chunks = []
        current_chunk = []
        current_len = 0
        seq = 0
        
        for s in sentences:
            s_len = len(s)
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

    # 3. Metadata-Aware Chunking (Augmented prefix with generic, reusable categories and keywords)
    def metadata_aware(self, max_char_len=300):
        # Using the sentence splitter with built-in outlier fallback
        sentences = split_sentences(self.text, max_sentence_len=400)
        if not sentences:
            return []
            
        chunks = []
        current_chunk = []
        current_len = 0
        seq = 0
        
        # Build generic, reusable metadata (avoids literal query overfitting)
        lang_name = "English" if self.language == "en" else ("Hindi" if self.language == "hi" else "Marathi")
        keywords = extract_keywords(self.text, num_keywords=3)
        context_prefix = f"Language: {lang_name} | Category: {self.query_type} | Keywords: {keywords} | Passage: "
        
        for s in sentences:
            s_len = len(s)
            # We measure chunk size including the context prefix to stay bounded
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
                    "raw_body": chunk_body  # Clean raw text is kept for LLM injection
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
    seen_english = set()
    
    # Process Hindi file (contains English and Hindi)
    for _, row in df_hi.iterrows():
        q_id = row['query_id']
        eng_q = row['Eng_Query']
        hi_q = row['query']
        q_type = row.get('query_type', 'general')
        
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
    for _, row in df_mr.iterrows():
        q_id = row['query_id']
        mar_q = row['query']
        q_type = row.get('query_type', 'general')
        
        mar_pass = row['passages']['Translated_passages']
        for i in range(len(mar_pass)):
            passages_list.append(Chunker(q_id, i, "mr", mar_pass[i], mar_q, q_type))
            
    print(f"Extracted {len(passages_list)} total base passages across 3 languages.", flush=True)
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
    print("COMPARATIVE CHUNKING ANALYSIS (UPDATED - SAFE METADATA)")
    print("="*50)
    
    for strat, res in results.items():
        print(f"\nStrategy: {strat.upper()}")
        print(f"  Total Chunks Generated: {res['count']}")
        print(f"  Average Chunk Size:      {res['avg_size']:.2f} chars")
        print(f"  Min/Max Chunk Size:      {res['min_size']} / {res['max_size']} chars")
        print(f"  Chunking Speed:          {res['time']:.4f} seconds")

    print("\n" + "="*50)
    print("SAMPLE OUTPUTS PER STRATEGY AND LANGUAGE")
    print("="*50)
    
    languages = ["en", "hi", "mr"]
    
    for strat in strategies:
        print(f"\n>>> Strategy: {strat.upper()} <<<")
        for lang in languages:
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
