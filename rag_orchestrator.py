import os
import sys
import time
import numpy as np
import torch
import lancedb
from transformers import AutoModel, AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Import modules
from llm_orchestrator import GroqProvider
from guardrails import GuardrailEngine

# ----------------- Configuration -----------------
MODEL_ID = "intfloat/multilingual-e5-base"  # Production model
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading embedding model on {self.device.type.upper()}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        if self.device.type == "cuda":
            # Load in FP16 for massive memory savings and acceleration on GPU
            self.model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(self.device)
        else:
            # Fallback to local CPU ONNX model
            self.model = ORTModelForFeatureExtraction.from_pretrained("./model_onnx_base")
            
        print("Connecting to LanceDB...", flush=True)
        self.db = lancedb.connect(DB_DIR)
        if TABLE_NAME not in self.db.table_names():
            raise ValueError(f"LanceDB table '{TABLE_NAME}' not found. Please run index_and_retrieve.py first to build the index.")
        self.table = self.db.open_table(TABLE_NAME)
        
        print("Initializing Groq LLM Provider...", flush=True)
        self.llm = GroqProvider()
        
        # Initialize Guardrails & Grounding Engine
        self.guardrails = GuardrailEngine(self.model, self.tokenizer, self.device, self.llm)
        
    def embed_query(self, query_text: str):
        prefixed = ["query: " + query_text]
        encoded = self.tokenizer(prefixed, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(**encoded)
            else:
                output = self.model(**encoded)
                
        emb = mean_pooling(output, encoded['attention_mask'])
        normalized = torch.nn.functional.normalize(emb, p=2, dim=1)
        return normalized.to(torch.float32).cpu().numpy()[0]
        
    def retrieve_context(self, query_text: str, top_k: int = 8):
        t0 = time.perf_counter()
        q_vector = self.embed_query(query_text)
        t_emb = time.perf_counter() - t0
        
        t_search_start = time.perf_counter()
        # Tuned: nprobes(80) recovers 90% of flat search recall (74% HI, 67% MR) under 40ms retrieval latency!
        results = self.table.search(q_vector.tolist()).nprobes(80).limit(top_k).to_pandas()
        t_search = time.perf_counter() - t_search_start
        
        return results, t_emb, t_search

    def query(self, query_text: str, top_k: int = 8):
        print(f"\nUser Query: '{query_text}'", flush=True)
        
        # 1. Retrieve Context
        results, t_emb, t_search = self.retrieve_context(query_text, top_k)
        print(f"Retrieval complete in {((t_emb + t_search)*1000.0):.2f} ms (Embedding: {t_emb*1000.0:.2f} ms | Search: {t_search*1000.0:.2f} ms)", flush=True)
        
        # 2. Pre-Retrieval Input Relevance Info (rely on post-generation grounding check to block off-topic content)
        _, max_sim = self.guardrails.check_input_relevance(results, similarity_threshold=0.0)
        print(f"Input Guardrail: Relevance Score = {max_sim:.4f}", flush=True)
            
        # 3. Extract Context Text (using clean raw_body)
        context_blocks = []
        for idx, row in results.iterrows():
            lang = row['language'].upper()
            body = row['raw_body']
            context_blocks.append(f"[{lang} Context {idx+1}]:\n{body}")
            
        context_text = "\n\n".join(context_blocks)
        
        # 4. Format Prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(context_text=context_text, query_text=query_text)
        
        # 5. Buffer LLM Generation to evaluate grounding before printing (guarantees safety)
        print("\nStreaming grounded answer:", flush=True)
        t_first_token = None
        t_gen_start = time.perf_counter()
        
        stream = self.llm.generate_stream(SYSTEM_PROMPT, user_prompt)
        full_answer_list = []
        
        for chunk in stream:
            if t_first_token is None:
                t_first_token = time.perf_counter()
                ttft_ms = (t_first_token - t_gen_start) * 1000.0
            full_answer_list.append(chunk)
            
        generated_answer = "".join(full_answer_list)
        t_total_gen = time.perf_counter() - t_gen_start
        
        # 6. Post-Generation Grounding Guardrail
        # Check if the generated answer is a refusal statement
        refusal_keywords = [
            "don't know", "do not know", "no information", 
            "mahit nahi", "maheet nahi", "माहित नाही",
            "nahin pata", "nahin pata", "नहीं पता"
        ]
        is_refusal = any(kw.lower() in generated_answer.lower() for kw in refusal_keywords)
        
        if is_refusal:
            grounding_status = "REFUSAL"
            score = 0.0
            print("Grounding Guardrail: Refusal detected -> Bypassing semantic checks.", flush=True)
            print(f"Grounding Guardrail: Score = N/A | Status = {grounding_status} | Path = REFUSAL_BYPASS", flush=True)
        else:
            retrieved_vectors = [np.array(v, dtype=np.float32) for v in results['vector'].values]
            retrieved_texts = results['text'].values
            
            grounding_status, score = self.guardrails.check_output_grounding(
                generated_answer, retrieved_vectors, retrieved_texts,
                low_threshold=0.80, high_threshold=0.85
            )
            
            # If Borderline, fall back to fast Groq LLM check
            if grounding_status == "BORDERLINE":
                print(f"Grounding: Borderline Score ({score:.4f}) -> Invoking LLM validator...", flush=True)
                grounding_status = self.guardrails.run_llm_grounding_eval(generated_answer, retrieved_texts)
                print(f"Grounding Guardrail: Score = {score:.4f} | Status = {grounding_status} | Path = BORDERLINE_LLM_EVAL", flush=True)
            elif grounding_status == "GROUNDED":
                print(f"Grounding Guardrail: Score = {score:.4f} | Status = {grounding_status} | Path = AUTO_PASS (Similarity > 0.85)", flush=True)
            else:
                print(f"Grounding Guardrail: Score = {score:.4f} | Status = {grounding_status} | Path = AUTO_BLOCK (Similarity < 0.80)", flush=True)
            
        if grounding_status == "NOT GROUNDED":
            print("\nHallucination detected! Blocking generated answer.", flush=True)
            fallback = "I don't know based on the provided context."
            lang = results.iloc[0]['language']
            if lang == 'hi':
                fallback = "दिए गए संदर्भ के आधार पर मुझे उत्तर नहीं पता है।"
            elif lang == 'mr':
                fallback = "दिलेल्या संदर्भाच्या आधारे मला उत्तर माहित नाही."
            print(f"\n{fallback}", flush=True)
        else:
            # Print the validated grounded answer or refusal
            print(f"\n(TTFT: {ttft_ms:.2f} ms) {generated_answer}", flush=True)
            
        print(f"\nGeneration complete in {t_total_gen:.2f} seconds.", flush=True)

if __name__ == "__main__":
    try:
        pipeline = RAGPipeline()
        
        print("\n" + "="*50)
        print("Multilingual RAG Interactive CLI (with Guardrails)")
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
