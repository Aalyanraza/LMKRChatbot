# Security Guards - Input & Output Validation

import numpy as np
import faiss
from guardrails import Guard, OnFailAction
from guardrails.hub import DetectPII, ToxicLanguage, CompetitorCheck
import config
from embeddings_setup import embeddings, malicious_index

# Setup Input Guard
input_guard = Guard().use_many(
    DetectPII(
        pii_entities=config.PII_ENTITIES,
        on_fail=OnFailAction.FIX
    )
)

# Setup Output Guard
output_guard = Guard().use_many(
    ToxicLanguage(threshold=config.TOXIC_LANGUAGE_THRESHOLD, on_fail=OnFailAction.FIX),
    CompetitorCheck(competitors=config.COMPETITOR_LIST, on_fail=OnFailAction.REASK)
)

def detect_malicious_prompt(question: str) -> bool:
    """
    Detects if a question contains malicious injection attempts.
    Uses semantic similarity against known malicious patterns.
    
    Returns True if malicious pattern detected, False otherwise.
    """
    # Layer 1: Heuristic Check (Fast)
    blacklist_keywords = ["ignore previous", "system prompt", "developer mode"]
    if any(kw in question.lower() for kw in blacklist_keywords):
        return True
    
    # Layer 2: Semantic Similarity Check
    raw_embedding = embeddings.embed_query(question)
    query_vector = np.array([raw_embedding]).astype('float32')
    faiss.normalize_L2(query_vector)
    
    distances, indices = malicious_index.search(query_vector, k=1)
    similarity_score = distances[0][0]
    
    if similarity_score > config.MALICIOUS_SIMILARITY_THRESHOLD:
        print(f"🛑 Blocking injection attempt. Similarity: {similarity_score:.2f}")
        return True
    
    return False

def apply_input_guard(question: str) -> str:
    """
    Applies input guards (PII redaction, malicious prompt detection).
    Returns sanitized question or a rejection message.
    """
    try:
        validation_result = input_guard.validate(question)
        return validation_result.validated_output
    except:
        return question

def apply_output_guard(answer: str) -> str:
    """
    Applies output guards (toxic language, competitor check).
    Returns sanitized answer or a rejection message.
    """
    try:
        validation_result = output_guard.validate(answer)
        return validation_result.validated_output
    except Exception as e:
        print(f"⚠️ Output Blocked: {e}")
        return "I'm sorry, I cannot provide that information due to safety guidelines."
