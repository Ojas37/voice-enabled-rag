import os
import sys
import time
import torch
import numpy as np

# Limit PyTorch to 1 thread safely (preventing RuntimeError if already set)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from rag_orchestrator import RAGPipeline

_pipeline = None

def get_pipeline_instance():
    global _pipeline
    if _pipeline is None:
        print("Initializing RAGPipeline wrapper for evaluation...", flush=True)
        # Check DEPLOY_ENV, if not set, fallback to default local configuration
        _pipeline = RAGPipeline()
    return _pipeline

def get_model():
    pipeline = get_pipeline_instance()
    return pipeline.model

def embed_one(text: str):
    pipeline = get_pipeline_instance()
    
    # Auto-detect or default to query instruction prefix for evaluation searches
    if text.startswith("query: ") or text.startswith("passage: "):
        prefixed = text
    else:
        prefixed = "query: " + text
        
    encoded = pipeline.tokenizer([prefixed], padding=True, truncation=True, max_length=512, return_tensors='pt')
    encoded = {k: v.to(pipeline.device) for k, v in encoded.items()}
    
    with torch.no_grad():
        if pipeline.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = pipeline.model(**encoded)
        else:
            output = pipeline.model(**encoded)
            
    from rag_orchestrator import mean_pooling
    emb = mean_pooling(output, encoded['attention_mask'])
    normalized = torch.nn.functional.normalize(emb, p=2, dim=1)
    return normalized.to(torch.float32).cpu().numpy()[0]

def embed(texts: list[str]):
    # The evaluation loop calls embed() to index document candidate passages
    prefixed_texts = []
    for t in texts:
        if t.startswith("query: ") or t.startswith("passage: "):
            prefixed_texts.append(t)
        else:
            prefixed_texts.append("passage: " + t)
            
    pipeline = get_pipeline_instance()
    embeddings = []
    
    for t in prefixed_texts:
        encoded = pipeline.tokenizer([t], padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded = {k: v.to(pipeline.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            if pipeline.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = pipeline.model(**encoded)
            else:
                output = pipeline.model(**encoded)
                
        from rag_orchestrator import mean_pooling
        emb = mean_pooling(output, encoded['attention_mask'])
        normalized = torch.nn.functional.normalize(emb, p=2, dim=1)
        embeddings.append(normalized.to(torch.float32).cpu().numpy()[0])
        
    return np.array(embeddings)

class AnswerObject:
    def __init__(self, text: str, grounded: bool, generation_ms: float, model: str):
        self.text = text
        self.grounded = grounded
        self.generation_ms = generation_ms
        self.model = model


def generate_answer(query: str, results: list) -> AnswerObject:
    pipeline = get_pipeline_instance()
    
    # 1. Pre-LLM Relevance Check: Embed query & passages to filter out unanswerable queries early
    query_vector = embed_one(query)
    retrieved_texts = [r.text for r in results]
    retrieved_vectors = []
    for t in retrieved_texts:
        retrieved_vectors.append(embed_one("passage: " + t))
        
    max_relevance = 0.0
    if retrieved_vectors:
        retrieved_np = np.vstack(retrieved_vectors)
        relevance_scores = np.dot(retrieved_np, query_vector)
        max_relevance = np.max(relevance_scores)
        
    # Threshold 0.77: anything below this is highly likely to be completely unrelated (decoy passages)
    RELEVANCE_THRESHOLD = 0.77
    if max_relevance < RELEVANCE_THRESHOLD:
        print(f"[Eval Guardrail] Low relevance detected ({max_relevance:.4f} < {RELEVANCE_THRESHOLD}). Refusing query.", flush=True)
        return AnswerObject(
            text="I don't know based on the provided context.",
            grounded=False,
            generation_ms=0.0,
            model=pipeline.llm.model_name
        )
        
    # 2. Format Context & Call LLM
    context_blocks = []
    for idx, r in enumerate(results):
        context_blocks.append(f"[Context {idx+1}] Source: {r.source}\n{r.text}")
    context_text = "\n\n".join(context_blocks)
    
    from rag_orchestrator import USER_PROMPT_TEMPLATE, SYSTEM_PROMPT
    user_prompt = USER_PROMPT_TEMPLATE.format(context_text=context_text, query_text=query)
    
    t_start = time.perf_counter()
    stream = pipeline.llm.generate_stream(SYSTEM_PROMPT, user_prompt)
    answer_tokens = []
    for chunk in stream:
        answer_tokens.append(chunk)
    generated_answer = "".join(answer_tokens)
    generation_ms = (time.perf_counter() - t_start) * 1000.0
    
    # 3. Post-Generation Grounding Validator
    grounding_status, score = pipeline.guardrails.check_output_grounding(
        generated_answer, retrieved_vectors, retrieved_texts
    )
    
    is_grounded = (grounding_status == "GROUNDED")
    final_text = generated_answer
    
    REFUSAL_PHRASES = [
        "i don't know", "i do not know", "no information", "not mentioned", "not provided", 
        "context does not", "cannot answer", "unable to answer", "does not contain", "no context",
        "मुझे उत्तर नहीं पता", "मला उत्तर माहित नाही", "संदर्भामध्ये", "संदर्भ में"
    ]
    
    is_refusal = False
    normalized = generated_answer.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in normalized:
            is_refusal = True
            break
            
    if not is_grounded and not is_refusal:
        final_text = "I don't know based on the provided context."
            
    return AnswerObject(
        text=final_text,
        grounded=is_grounded,
        generation_ms=generation_ms,
        model=pipeline.llm.model_name
    )
