# Security Guards - Input & Output Validation

import numpy as np
import faiss

from guardrails import Guard, OnFailAction
from guardrails.hub import DetectPII, ToxicLanguage, CompetitorCheck

import config
from embeddings_setup import embeddings, malicious_index

# --- Heuristic configuration ---

BLACKLIST_KEYWORDS = ["ignore previous", "system prompt", "developer mode"]

# Setup Input Guard
input_guard = Guard().use_many(
    DetectPII(
        pii_entities=config.PII_ENTITIES,
        on_fail=OnFailAction.FIX,
    )
)

# Setup Output Guard
output_guard = Guard().use_many(
    ToxicLanguage(threshold=config.TOXIC_LANGUAGE_THRESHOLD, on_fail=OnFailAction.FIX),
    CompetitorCheck(competitors=config.COMPONENT_LIST, on_fail=OnFailAction.REASK)
)


def detect_malicious_prompt(question: str) -> bool:
    """
    Detects if a question contains malicious injection attempts.
    Uses a fast heuristic keyword check and a semantic similarity
    check against known malicious patterns.

    Returns True if a malicious pattern is detected, False otherwise.
    """
    text = question.lower()

    # Layer 1: Heuristic Check (Fast)
    if any(kw in text for kw in BLACKLIST_KEYWORDS):
        print("🛑 Blocking injection attempt via blacklist keyword.")
        return True

    # Layer 2: Semantic Similarity Check
    try:
        query_vector = np.array([embeddings.embed_query(question)]).astype("float32")
        faiss.normalize_L2(query_vector)

        distances, _ = malicious_index.search(query_vector, k=1)
        similarity_score = float(distances[0][0])

        if similarity_score > config.MALICIOUS_SIMILARITY_THRESHOLD:
            print(f"🛑 Blocking injection attempt. Similarity: {similarity_score:.2f}")
            return True
    except Exception as e:
        # Fail-open on semantic check: do not block if the similarity check itself fails
        print(f"⚠️ Malicious prompt semantic check failed: {e}")

    return False


def apply_input_guard(question: str) -> str:
    """
    Applies Guardrails input validation (e.g., PII redaction).
    Malicious prompt detection is handled separately via detect_malicious_prompt().

    Returns sanitized question, or the original question if validation fails.
    """
    try:
        validation_result = input_guard.validate(question)
        return validation_result.validated_output
    except Exception as e:
        print(f"⚠️ Input guard failed: {e}")
        return question


def apply_output_guard(answer: str) -> str:
    """
    Applies output guards (toxic language, competitor check).
    Returns sanitized answer, or a safe fallback message if validation fails.
    """
    try:
        validation_result = output_guard.validate(answer)
        return validation_result.validated_output
    except Exception as e:
        print(f"⚠️ Output guard failed: {e}")
        return "I'm sorry, I cannot provide that information due to safety guidelines."
