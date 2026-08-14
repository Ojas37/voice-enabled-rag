import os
import sys
import time
import numpy as np
import torch
import lancedb
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Import modules
from llm_orchestrator import GroqProvider

# ----------------- Configuration -----------------
MODEL_DIR = "./model_onnx"
DB_DIR = "data/lancedb"
TABLE_NAME = "multilingual_passages"

# ----------------- Grounding Prompts -----------------
SYSTEM_PROMPT = """You are an accurate, grounded multilingual QA assistant.
You will be provided with several context passages in English, Hindi, or Marathi.
Your task is to answer the User Query using ONLY the provided context passages.

Strict Grounding Rules:
1. Base your answer solely on the facts directly mentioned in the Context. Do NOT make assumptions, extrapolate, or bring in external knowledge.
2. If the Context does not contain enough information to answer the Query, respond strictly with:
   - English: "I don't know based on the provided context."
   - Hindi: "दिए गए संदर्भ के आधार पर मुझे उत्तर नहीं पता है।"
   - Marathi: "दिलेल्या संदर्भाच्या आधारे मला उत्तर माहित नाही."
3. You must respond in the same language as the User Query. If the query is in Hindi, reply in Hindi. If Marathi, reply in Marathi. If English, reply in English.
4. Keep your answer concise, factual, and direct. Do not mention "based on the provided context" or "as per the context" in your answer unless failing to find the answer.
"""

USER_PROMPT_TEMPLATE = """Context Passages:
{context_text}

User Query: {query_text}

Answer:"""

# ----------------- Mean Pooling Helper -----------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class RAGPipeline:
    def __init__(self):
        print("Loading local ONNX model and tokenizer for retrieval...", flush=True)
        self.model = ORTModelForFeatureExtraction.from_pretrained(MODEL_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        
        print("Connecting to LanceDB...", flush=True)
        self.db = lancedb.connect(DB_DIR)
        if TABLE_NAME not in self.db.table_names():
            raise ValueError(f"LanceDB table '{TABLE_NAME}' not found. Please run index_and_retrieve.py first to build the index.")
        self.table = self.db.open_table(TABLE_NAME)
        
        print("Initializing Groq LLM Provider...", flush=True)
        self.llm = GroqProvider()
        
    def embed_query(self, query_text: str):
        # E5 query format prefix
        prefixed = ["query: " + query_text]
        encoded = self.tokenizer(prefixed, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**encoded)
        emb = mean_pooling(output, encoded['attention_mask'])
        normalized = torch.nn.functional.normalize(emb, p=2, dim=1)
        return normalized.cpu().numpy()[0]
        
    def retrieve_context(self, query_text: str, top_k: int = 5):
        t0 = time.perf_counter()
        q_vector = self.embed_query(query_text)
        t_emb = time.perf_counter() - t0
        
        t_search_start = time.perf_counter()
        results = self.table.search(q_vector.tolist()).limit(top_k).to_pandas()
        t_search = time.perf_counter() - t_search_start
        
        return results, t_emb, t_search

    def query(self, query_text: str, top_k: int = 5):
        print(f"\nUser Query: '{query_text}'", flush=True)
        
        # 1. Retrieve Context
        results, t_emb, t_search = self.retrieve_context(query_text, top_k)
        print(f"Retrieval complete in {((t_emb + t_search)*1000.0):.2f} ms (Embedding: {t_emb*1000.0:.2f} ms | Search: {t_search*1000.0:.2f} ms)", flush=True)
        
        # 2. Extract Context Text (using clean raw_body)
        context_blocks = []
        for idx, row in results.iterrows():
            lang = row['language'].upper()
            body = row['raw_body']
            context_blocks.append(f"[{lang} Context {idx+1}]:\n{body}")
            
        context_text = "\n\n".join(context_blocks)
        
        # 3. Format Prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(context_text=context_text, query_text=query_text)
        
        # 4. Stream Response from LLM
        print("\nStreaming grounded answer:", flush=True)
        t_first_token = None
        t_gen_start = time.perf_counter()
        
        stream = self.llm.generate_stream(SYSTEM_PROMPT, user_prompt)
        
        for chunk in stream:
            if t_first_token is None:
                t_first_token = time.perf_counter()
                ttft_ms = (t_first_token - t_gen_start) * 1000.0
                print(f" (TTFT: {ttft_ms:.2f} ms) ", end="", flush=True)
            print(chunk, end="", flush=True)
            
        t_total_gen = time.perf_counter() - t_gen_start
        print(f"\n\nGeneration complete in {t_total_gen:.2f} seconds.", flush=True)

if __name__ == "__main__":
    # Simple interactive command-line interface for testing
    try:
        pipeline = RAGPipeline()
        
        print("\n" + "="*50)
        print("Multilingual RAG Interactive CLI")
        print("="*50)
        print("Type 'exit' to quit.")
        
        while True:
            query = input("\nEnter query: ").strip()
            if not query or query.lower() == 'exit':
                break
            try:
                pipeline.query(query)
            except Exception as e:
                print(f"Error during query: {e}", flush=True)
                
    except Exception as e:
        print(f"Pipeline error: {e}", flush=True)
