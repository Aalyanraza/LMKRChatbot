# Graph Nodes - All workflow nodes

import os
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import FAISS
from models import (
    QueryAugmentation, GeneratedAnswer, ValidationResult, 
    RouteDecision, AgentState
)
from llm_helpers import query_llm_structured
from guards import detect_malicious_prompt, apply_input_guard, apply_output_guard
from tools import scrape_careers_tool, scrape_news_fast_tool
from utils import split_text_into_chunks, deduplicate_chunks, load_from_file, save_to_file
from embeddings_setup import vectorstore, embeddings
import config

# --- Node 1: INPUT GUARD ---

def input_guard_node(state: AgentState):
    """
    Security layer: Detects malicious prompts and redacts PII.
    """
    print("\n🛡️ Node: Custom Semantic Guard (Security Check)...")
    question = state["question"]
    
    # Layer 1 & 2: Malicious prompt detection
    if detect_malicious_prompt(question):
        return {
            "question": "Access Denied: Malicious pattern detected.",
            "destination": "conversational_node"
        }
    
    # Layer 3: PII Redaction
    question = apply_input_guard(question)
    
    return {"question": question, "destination": "router_node"}

# --- Node 2: ROUTER ---

def router_node(state: AgentState):
    """
    Routes the query to the appropriate retrieval path based on intent.
    """
    print("\n🚦 Router: Analyzing User Intent...")
    question = state["question"]
    
    parser = PydanticOutputParser(pydantic_object=RouteDecision)
    prompt = f"""
User Question: {question}

Role: You are a Router.

Task: Decide where to send this query.

Rules:

1. If the user asks about News, Announcements, Press Releases, or Recent Updates about LMKR, route to 'news_retrieve_node'.

2. If the user asks about Jobs, Careers, Vacancies, Internships about LMKR, route to 'career_retrieve_node'.

3. If the user uses greetings (Hi, Hello) or generic chat or anything not related to LMKR, route to 'conversational_node'.

4. For everything else (Company History, Software info, Contact, Services, Products), route to 'retrieve_node'.

"""
    
    decision = query_llm_structured(prompt, parser)
    
    if not decision:
        return {"destination": "retrieve_node"}
    
    print(f" 👉 Routing to: {decision.destination}")
    return {"destination": decision.destination}

# --- Node 3: GENERAL RETRIEVE ---

def retrieve_node(state: AgentState):
    """
    Retrieves context using adaptive multi-query augmentation.
    """
    print("\n🔍 Node 1: Retrieve (Augmenting & Searching)...")
    question = state["question"]
    
    # Adaptive logic: Widen search on retries
    current_retry = state.get("retry_count", 0)
    base_k = config.BASE_K_GENERAL
    dynamic_k = base_k + (current_retry * config.RETRY_K_INCREMENT)
    
    if current_retry > 0:
        print(f" 🔄 Retry #{current_retry} detected: Expanding search context to top-{dynamic_k} chunks.")
    
    # 1. Multi-Query Augmentation
    parser = PydanticOutputParser(pydantic_object=QueryAugmentation)
    prompt = f"""
User Question: {question}

Task: Generate 3 different search query variations to find relevant info in a corporate vector DB.

"""
    
    structured_aug = query_llm_structured(prompt, parser)
    queries = [question]
    
    if structured_aug:
        queries.extend(structured_aug.augmented_queries)
    
    # 2. Retrieve & Deduplicate
    all_docs = []
    for q in queries:
        docs = vectorstore.similarity_search(q, k=dynamic_k)
        all_docs.extend([d.page_content for d in docs])
    
    unique_context = list(set(all_docs))[:dynamic_k]
    
    print(f" Retrieved {len(unique_context)} unique context chunks (Target: {dynamic_k}).")
    
    # Debug log
    save_to_file("\n".join(unique_context), config.CONTEXT_DEBUG_FILE)
    
    return {"context_chunks": unique_context}

# --- Node 4: CAREER RETRIEVE ---

def career_retrieve_node(state: AgentState):
    """
    Retrieves career-related information by scraping and searching.
    """
    print("\n💼 Node: Career Retrieve (Adaptive)...")
    question = state["question"]
    
    current_retry = state.get("retry_count", 0)
    base_k = config.BASE_K_CAREER
    dynamic_k = base_k + (current_retry * config.RETRY_K_INCREMENT_CAREER)
    
    raw_text = ""
    
    # Check cache to avoid re-scraping
    if current_retry > 0 and os.path.exists(config.CAREERS_OUTPUT_FILE):
        print(f" 🔄 Retry #{current_retry}: Reading cached career data (Skipping Web Scrape)...")
        raw_text = load_from_file(config.CAREERS_OUTPUT_FILE)
    else:
        raw_text = scrape_careers_tool.invoke({})
    
    if not raw_text:
        print(" ⚠️ Warning: Scrape returned empty data.")
        return {"context_chunks": []}
    
    # 2. Chunk & Index
    chunks = split_text_into_chunks(raw_text)
    temp_vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # 3. Dynamic Search
    print(f" Searching career data with k={dynamic_k}...")
    retrieved_docs = temp_vectorstore.similarity_search(question, k=dynamic_k)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    
    print(f" Retrieved {len(retrieved_texts)} relevant career chunks.")
    return {"context_chunks": retrieved_texts}

# --- Node 5: NEWS RETRIEVE ---

def news_retrieve_node(state: AgentState):
    """
    Retrieves news-related information by scraping and searching.
    """
    print("\n🗞️ Node: News Retrieve (Fast & Adaptive)...")
    question = state["question"]
    
    current_retry = state.get("retry_count", 0)
    base_k = config.BASE_K_NEWS
    dynamic_k = base_k + (current_retry * config.RETRY_K_INCREMENT_NEWS)
    
    raw_text = ""
    
    # Check cache
    if current_retry > 0 and os.path.exists(config.NEWS_OUTPUT_FILE):
        print(f" 🔄 Retry #{current_retry}: Reading cached news data...")
        raw_text = load_from_file(config.NEWS_OUTPUT_FILE)
    else:
        raw_text = scrape_news_fast_tool.invoke({})
    
    if not raw_text:
        print(" ⚠️ Warning: News scrape returned empty data.")
        return {"context_chunks": []}
    
    # Chunk & Index
    chunks = split_text_into_chunks(raw_text)
    
    if not chunks:
        return {"context_chunks": []}
    
    temp_vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # Search
    print(f" Searching news data with k={dynamic_k}...")
    retrieved_docs = temp_vectorstore.similarity_search(question, k=dynamic_k)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    
    print(f" Retrieved {len(retrieved_texts)} relevant news chunks.")
    return {"context_chunks": retrieved_texts}

# --- Node 6: CONVERSATIONAL ---

def conversational_node(state: AgentState):
    """
    Handles conversational queries without retrieval context.
    """
    print("\n💬 Node: Conversational (Direct LLM)...")
    question = state["question"]
    
    parser = PydanticOutputParser(pydantic_object=GeneratedAnswer)
    prompt = f"""
User Input: {question}

Instructions:

1. You are a helpful corporate assistant for LMKR.

2. Respond naturally to the greeting or conversational question.

3. Do NOT make up technical facts. Just be polite.

4. Set 'sources_used' to ["Conversational"].

"""
    
    structured_response = query_llm_structured(prompt, parser)
    
    # Fallback
    if not structured_response:
        structured_response = GeneratedAnswer(
            answer="Hello! I am the LMKR AI Assistant. How can I help you with our software or services?",
            sources_used=["Conversational"]
        )
    
    return {"generated_answer": structured_response, "context_chunks": []}

# --- Node 7: GENERATE ---

def generate_node(state: AgentState):
    """
    Generates answer from retrieved context.
    """
    print("\n✍️ Node: Generate (Unified)...")
    question = state["question"]
    
    context_data = "\n---\n".join(state["context_chunks"])
    
    if not context_data:
        context_data = "No information found in the retrieved context."
    
    parser = PydanticOutputParser(pydantic_object=GeneratedAnswer)
    
    prompt = f"""
Context Data:

{context_data}

User Question: {question}

Instructions:

1. Answer the user's question using ONLY the provided Context Data.

2. If the context contains a list of items (like job openings, software features, or locations), present them clearly as a list.

3. If the answer is not in the context, state "I do not have enough information."

4. Do not hallucinate. Maintain a professional tone.

"""
    
    structured_response = query_llm_structured(prompt, parser)
    
    # Fallback
    if not structured_response:
        structured_response = GeneratedAnswer(
            answer="Error generating response.",
            sources_used=["None"]
        )
    
    return {
        "generated_answer": structured_response,
        "retry_count": state.get("retry_count", 0) + 1
    }

# --- Node 8: OUTPUT GUARD ---

def output_guard_node(state: AgentState):
    """
    Applies safety checks to generated output.
    """
    print("\n🛡️ Node: Output Guard (Safety Scan)...")
    generation = state["generated_answer"]
    
    generation.answer = apply_output_guard(generation.answer)
    
    return {"generated_answer": generation}

# --- Node 9: VALIDATE ---

def validate_node(state: AgentState):
    """
    Validates generated answer against retrieved context for hallucinations.
    """
    print("\n🛡️ Node 3: Robust Validation...")
    
    generation = state["generated_answer"]
    context_chunks = state["context_chunks"]
    context_text = "\n---\n".join(context_chunks)
    question = state["question"]
    
    # 1. Immediate Pass for Conversational/Fallbacks
    if "I do not have enough information" in generation.answer:
        return {
            "validation": ValidationResult(is_valid=True, reason="Honest fallback triggered.")
        }
    
    if "Conversational" in generation.sources_used:
        return {
            "validation": ValidationResult(is_valid=True, reason="Conversational turn.")
        }
    
    # 2. Stronger Validation Prompt
    parser = PydanticOutputParser(pydantic_object=ValidationResult)
    prompt = f"""
You are a strict Quality Control Auditor.

User Question: {question}

Generated Answer: {generation.answer}

Reference Context:

{context_text}

Instructions:

1. Break the Generated Answer into individual claims.

2. For EACH claim, attempt to find a supporting quote in the Reference Context.

3. If a claim exists in the Answer but NOT in the Context, it is a HALLUCINATION.

4. Ignore minor phrasing differences; look for semantic meaning.

Output JSON:

- set 'is_valid' to false if ANY unsupported claim is found.

- set 'reason' to a specific explanation of what fact was unsupported.

"""
    
    validation = query_llm_structured(prompt, parser)
    
    if not validation:
        validation = ValidationResult(is_valid=False, reason="Validation LLM failed to parse.")
    
    print(f" Evaluation: {'✅ PASS' if validation.is_valid else '❌ FAIL'} | Reason: {validation.reason}")
    
    return {"validation": validation}
