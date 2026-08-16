import os
import sys
import time
import random
import numpy as np
import pandas as pd
import lancedb
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ----------------- Configuration -----------------
LIMIT_ROWS = 5000
MODEL_ID = "intfloat/multilingual-e5-base"
DB_DIR = "data/lancedb"
TABLE_NAME = "multilingual_passages"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device configuration: {DEVICE.type.upper()} active for calibration.", flush=True)

# Mean pooling helper
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_text(text, tokenizer, model, is_query=True):
    prefix = "query: " if is_query else "passage: "
    prefixed = [prefix + text]
    encoded = tokenizer(prefixed, padding=True, truncation=True, max_length=512, return_tensors='pt')
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        if DEVICE.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model(**encoded)
        else:
            output = model(**encoded)
    emb = mean_pooling(output, encoded['attention_mask'])
    normalized = torch.nn.functional.normalize(emb, p=2, dim=1)
    return normalized.to(torch.float32).cpu().numpy()[0]

# Compile out-of-domain queries
out_of_domain_queries = [
    # English Factual
    "what is the capital of france",
    "who wrote the play hamlet",
    "how far is the moon from earth",
    "what is the boiling point of water",
    "who is the current president of the united states",
    "what is the capital of japan",
    "how many planets are in the solar system",
    "who painted the mona lisa",
    "what is the speed of light",
    "where are the pyramids of giza located",
    "who discovered gravity",
    "what is the largest ocean on earth",
    "how many bones are in the human body",
    "what is the chemical symbol for gold",
    "who was the first man on the moon",
    # English Chitchat & Random
    "tell me a joke",
    "how are you today",
    "what is your name",
    "can you sing a song",
    "tell me a story",
    "what is the meaning of life",
    "hello, who are you",
    "write a poem about love",
    "give me a recipe for chocolate chip cookies",
    "do you like sports",
    "what is 2 + 2",
    "recommend a good movie",
    "how do I learn python",
    "what is the weather like",
    "tell me a riddle",
    # Hindi Out-of-Domain
    "फ्रांस की राजधानी क्या है",
    "मुझे एक चुटकुला सुनाओ",
    "तुम्हारा नाम क्या है",
    "पानी का क्वथनांक क्या है",
    "ताज़महल कहाँ है",
    "अमेरिका का राष्ट्रपति कौन है",
    "पृथ्वी से चंद्रमा की दूरी कितनी है",
    "मुझे एक कहानी बताओ",
    "चाय बनाने की विधि क्या है",
    "गुरुत्वाकर्षण की खोज किसने की",
    # Marathi Out-of-Domain
    "फ्रान्सची राजधानी कोणती आहे",
    "मला एक विनोद सांग",
    "तुझे नाव काय आहे",
    "पाण्याचा उत्कलन बिंदू काय आहे",
    "ताजमहाल कोठे आहे",
    "अमेरिकेचे राष्ट्राध्यक्ष कोण आहेत",
    "मला एक गोष्ट सांग",
    "सूर्य कोणत्या दिशेला उगवतो",
    "चहा कसा बनवायचा",
    "गुरुत्वाकर्षणाचा शोध कोणी लावला"
]

def load_indomain_queries(limit_rows=5000):
    df_hi = pd.read_parquet("data/hinval_real_mini.parquet").head(limit_rows)
    df_mr = pd.read_parquet("data/marval_real_mini.parquet").head(limit_rows)
    
    in_domain = []
    # 25 English in-domain queries
    for idx, row in df_hi.iterrows():
        if len(in_domain) >= 25:
            break
        in_domain.append(row['Eng_Query'])
        
    # 15 Hindi in-domain queries
    for idx, row in df_hi.iterrows():
        if len(in_domain) >= 40:
            break
        in_domain.append(row['query'])
        
    # 10 Marathi in-domain queries
    for idx, row in df_mr.iterrows():
        if len(in_domain) >= 50:
            break
        in_domain.append(row['query'])
        
    return in_domain

def print_histogram(title, values, bins=10):
    print(f"\nHistogram: {title}")
    counts, edges = np.histogram(values, bins=bins, range=(0.50, 1.00))
    for i in range(bins):
        bar = "*" * int(counts[i] * 2)
        print(f"  [{edges[i]:.2f} - {edges[i+1]:.2f}): {counts[i]:<3} {bar}")

def main():
    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(TABLE_NAME)
    
    print("Loading embedding model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if DEVICE.type == "cuda":
        model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    else:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        model = ORTModelForFeatureExtraction.from_pretrained("./model_onnx_base")
        
    in_domain_queries = load_indomain_queries(LIMIT_ROWS)
    
    print("\nEvaluating Relevance Scores for In-Domain Queries...")
    in_domain_scores = []
    for q in tqdm(in_domain_queries):
        q_vector = embed_text(q, tokenizer, model, is_query=True)
        results = tbl.search(q_vector.tolist()).nprobes(80).limit(8).to_pandas()
        if not results.empty:
            max_sim = 1.0 - np.min(results['_distance'].values)
            in_domain_scores.append(max_sim)
            
    print("\nEvaluating Relevance Scores for Out-of-Domain Queries...")
    out_domain_scores = []
    for q in tqdm(out_of_domain_queries):
        q_vector = embed_text(q, tokenizer, model, is_query=True)
        results = tbl.search(q_vector.tolist()).nprobes(80).limit(8).to_pandas()
        if not results.empty:
            max_sim = 1.0 - np.min(results['_distance'].values)
            out_domain_scores.append(max_sim)
            
    # Print statistical summary
    print("\n" + "="*60)
    print("INPUT RELEVANCE SCORE DISTRIBUTIONS")
    print("="*60)
    
    id_stats = {
        "min": np.min(in_domain_scores),
        "max": np.max(in_domain_scores),
        "mean": np.mean(in_domain_scores),
        "median": np.median(in_domain_scores)
    }
    
    od_stats = {
        "min": np.min(out_domain_scores),
        "max": np.max(out_domain_scores),
        "mean": np.mean(out_domain_scores),
        "median": np.median(out_domain_scores)
    }
    
    print(f"{'Metric':<10} | {'In-Domain Queries':<20} | {'Out-of-Domain Queries':<20}")
    print("-" * 60)
    print(f"{'Min':<10} | {id_stats['min']:.4f}               | {od_stats['min']:.4f}")
    print(f"{'Max':<10} | {id_stats['max']:.4f}               | {od_stats['max']:.4f}")
    print(f"{'Mean':<10} | {id_stats['mean']:.4f}               | {od_stats['mean']:.4f}")
    print(f"{'Median':<10} | {id_stats['median']:.4f}               | {od_stats['median']:.4f}")
    print("-" * 60)
    
    print_histogram("In-Domain Relevance Scores", in_domain_scores)
    print_histogram("Out-of-Domain Relevance Scores", out_domain_scores)
    
    # Let's check Grounding Scores for Grounded vs. Ungrounded Answers
    print("\nEvaluating Grounding Scores...")
    grounded_scores = []
    ungrounded_scores = []
    
    # We evaluate 10 samples
    for i in range(10):
        # Retrieve context for in-domain query
        q = in_domain_queries[i]
        q_vector = embed_text(q, tokenizer, model, is_query=True)
        results = tbl.search(q_vector.tolist()).nprobes(80).limit(8).to_pandas()
        
        if len(results) >= 2:
            # 1. Genuinely Grounded Answer: Let's extract the actual passage raw_body and embed it!
            # Since the LLM output is derived directly from the passage, its embedding similarity to the passage will be close to the passage itself
            passage_text = results.iloc[0]['raw_body']
            # Truncate slightly to simulate an LLM summary
            grounded_answer = " ".join(passage_text.split()[:20])
            g_vector = embed_text(grounded_answer, tokenizer, model, is_query=False)
            
            # Compute similarity against all retrieved passages
            retrieved_vectors = np.vstack([np.array(v) for v in results['vector'].values])
            sims_g = np.dot(retrieved_vectors, g_vector)
            grounded_scores.append(np.max(sims_g))
            
            # 2. Genuinely Ungrounded Answer: Versailles or Grandma joke
            ungrounded_answers = [
                "Versailles.",
                "What's that wrinkly thing on Grandma?",
                "The capital of France is Paris.",
                "Sure, here is a joke: Why did the chicken cross the road? To get to the other side!"
            ]
            for ug_ans in ungrounded_answers:
                ug_vector = embed_text(ug_ans, tokenizer, model, is_query=False)
                sims_ug = np.dot(retrieved_vectors, ug_vector)
                ungrounded_scores.append(np.max(sims_ug))

    print("\n" + "="*60)
    print("ANSWER GROUNDING SCORE DISTRIBUTIONS")
    print("="*60)
    print(f"{'Metric':<10} | {'Grounded Answers':<20} | {'Ungrounded Answers':<20}")
    print("-" * 60)
    print(f"{'Min':<10} | {np.min(grounded_scores):.4f}               | {np.min(ungrounded_scores):.4f}")
    print(f"{'Max':<10} | {np.max(grounded_scores):.4f}               | {np.max(ungrounded_scores):.4f}")
    print(f"{'Mean':<10} | {np.mean(grounded_scores):.4f}               | {np.mean(ungrounded_scores):.4f}")
    print(f"{'Median':<10} | {np.median(grounded_scores):.4f}               | {np.median(ungrounded_scores):.4f}")
    print("-" * 60)
    
    print_histogram("Grounded Answer Scores", grounded_scores, bins=5)
    print_histogram("Ungrounded Answer Scores", ungrounded_scores, bins=5)

if __name__ == "__main__":
    main()
