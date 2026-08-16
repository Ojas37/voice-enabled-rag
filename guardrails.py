import os
import sys
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ----------------- Grounding Evaluator Prompt -----------------
LLM_GROUNDING_SYSTEM_PROMPT = "You are a strict factual consistency checker. Your job is to verify if an answer is fully supported by the provided context."
LLM_GROUNDING_PROMPT_TEMPLATE = """You will be provided with several Context passages and a proposed Answer.
Your task is to determine if the proposed Answer is fully supported by and grounded in the Context.

Rules:
1. Reply with EXACTLY "GROUNDED" if every fact in the proposed Answer is directly supported by the Context.
2. Reply with EXACTLY "NOT GROUNDED" if the proposed Answer contains facts, figures, or assumptions not present in the Context.
3. Do not include any other text, explanation, or conversational filler.

Context:
{context_text}

Proposed Answer:
{answer_text}

Evaluation:"""

class GuardrailEngine:
    def __init__(self, model, tokenizer, device, llm_provider):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.llm = llm_provider
        
    def check_input_relevance(self, lancedb_results, similarity_threshold: float = 0.60):
        """
        Pre-retrieval validation.
        Evaluates if the query is relevant/on-topic using search distance.
        LanceDB uses cosine distance (1 - similarity).
        Returns (is_relevant, max_similarity).
        """
        if lancedb_results.empty:
            return False, 0.0
            
        # In LanceDB, '_distance' for cosine metric is (1 - cosine_similarity)
        # So Cosine Similarity = 1 - Cosine Distance
        distances = lancedb_results['_distance'].values
        max_similarity = 1.0 - np.min(distances)
        
        is_relevant = max_similarity >= similarity_threshold
        return is_relevant, max_similarity
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed_text_passage(self, text: str):
        """Embeds a single passage (prefixed with 'passage: ')"""
        prefixed = ["passage: " + text]
        encoded = self.tokenizer(prefixed, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(**encoded)
            else:
                output = self.model(**encoded)
                
        emb = self._mean_pooling(output, encoded['attention_mask'])
        normalized = torch.nn.functional.normalize(emb, p=2, dim=1)
        return normalized.to(torch.float32).cpu().numpy()[0]

    def check_output_grounding(self, generated_answer: str, retrieved_vectors, retrieved_texts, 
                                low_threshold: float = 0.80, high_threshold: float = 0.85):
        """
        Post-generation validation.
        Compares generated answer vector to retrieved context vectors.
        Returns:
            status: "GROUNDED", "NOT GROUNDED", or "BORDERLINE"
            max_score: the highest cosine similarity score
        """
        if not generated_answer.strip():
            return "NOT GROUNDED", 0.0
            
        # 1. Embed the answer as a passage
        ans_vector = self._embed_text_passage(generated_answer)
        
        # 2. Compute Cosine Similarity against all retrieved context vectors
        # retrieved_vectors is a list of lists/numpy arrays of shape (K, 768)
        retrieved_np = np.vstack(retrieved_vectors)
        
        # Dot product of normalized vectors yields cosine similarity
        similarities = np.dot(retrieved_np, ans_vector)
        max_score = np.max(similarities)
        
        if max_score > high_threshold:
            return "GROUNDED", max_score
        elif max_score < low_threshold:
            return "NOT GROUNDED", max_score
        else:
            return "BORDERLINE", max_score

    def run_llm_grounding_eval(self, generated_answer: str, retrieved_texts):
        """Runs a fast LLM verification call on Groq for borderline cases."""
        context_text = "\n\n".join([f"[Passage {i+1}]:\n{t}" for i, t in enumerate(retrieved_texts)])
        user_prompt = LLM_GROUNDING_PROMPT_TEMPLATE.format(context_text=context_text, answer_text=generated_answer)
        
        try:
            # We request a stream but it's only 1-2 tokens (GROUNDED / NOT GROUNDED)
            token_list = []
            stream = self.llm.generate_stream(LLM_GROUNDING_SYSTEM_PROMPT, user_prompt)
            for token in stream:
                token_list.append(token)
            decision = "".join(token_list).strip().upper()
            
            if "NOT GROUNDED" in decision:
                return "NOT GROUNDED"
            return "GROUNDED"
        except Exception as e:
            print(f"Error during LLM grounding fallback evaluation: {e}", file=sys.stderr)
            # Default to block on error to guarantee safety
            return "NOT GROUNDED"
