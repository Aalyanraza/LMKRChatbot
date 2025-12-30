# Graph Nodes - 1:1 Functional Match with OpenAI Refactor
import os
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from models import (
    QueryAugmentation, GeneratedAnswer, ValidationResult, 
    RouteDecision, AgentState, ProceduralRule
)
from llm_helpers import query_llm_structured
from guards import detect_malicious_prompt, apply_input_guard, apply_output_guard
from tools import scrape_careers_tool, scrape_news_fast_tool
from utils import split_text_into_chunks, load_from_file, save_to_file, is_file_fresh
from embeddings_setup import vectorstore, embeddings, memory_store
import config
from datetime import datetime

# --- Node 1: INPUT GUARD ---
def input_guard_node(state: AgentState):
    print("\n🛡️ Node: Custom Semantic Guard (Security Check)...")
    question = state["question"]
    
    # Layer 1 & 2: Malicious Detection (Heuristic + Semantic)
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
    print("\n🚦 Router: Analyzing User Intent...")
    question = state["question"]
    namespace = ("instructions", "global")
    stored_lessons = memory_store.search(namespace, limit=5)
    lessons_text = "\n".join([m.value["rule"] for m in stored_lessons])
    
    prompt = f"""
    User Question: {question}
    Existing Procedural Lessons:
    {lessons_text}
    Role: You are a Router. 
    Task: Decide where to send this query.
    
    Rules:
    1. If the user asks about News, Announcements, Press Releases, or Recent Updates about LMKR, route to 'news_retrieve_node'.
    2. If the user asks about Jobs, Careers, Vacancies, Internships about LMKR, route to 'career_retrieve_node'.
    3. If the user uses greetings (Hi, Hello) or generic chat or anything not related to LMKR, route to 'conversational_node'.
    4. For everything else (Company History, Software info, Contact, Services, Products), route to 'retrieve_node'.
    """
    
    decision = query_llm_structured(prompt, RouteDecision)
    destination = decision.destination if decision else "retrieve_node"
    print(f" 👉 Routing to: {destination}")
    return {"destination": destination}

# --- Node 3: GENERAL RETRIEVE ---
def retrieve_node(state: AgentState):
    print("\n🔍 Node 1: Retrieve (Augmenting & Searching)...")
    question = state["question"]
    current_retry = state.get("retry_count", 0)
    
    # Adaptive Logic: 5 -> 8 -> 11
    dynamic_k = config.BASE_K_GENERAL + (current_retry * config.RETRY_K_INCREMENT)
    if current_retry > 0:
        print(f"   🔄 Retry #{current_retry} detected: Expanding search context to top-{dynamic_k} chunks.")
    
    prompt = f"User Question: {question}\nTask: Generate 3 different search query variations to find relevant info in a corporate vector DB."
    
    structured_aug = query_llm_structured(prompt, QueryAugmentation)
    queries = [question] + (structured_aug.augmented_queries if structured_aug else [])
    
    all_docs = []
    for q in queries:
        docs = vectorstore.similarity_search(q, k=dynamic_k)
        all_docs.extend([d.page_content for d in docs])
    
    unique_context = list(set(all_docs))[:dynamic_k]
    print(f"   Retrieved {len(unique_context)} unique context chunks (Target: {dynamic_k}).")
    
    # Debug log as requested
    save_to_file("".join(unique_context), config.CONTEXT_DEBUG_FILE)
    
    return {"context_chunks": unique_context}

# --- Node 4: CAREER RETRIEVE ---
def career_retrieve_node(state: AgentState):
    print("\n💼 Node: Career Retrieve (Adaptive)...")
    question = state["question"]
    current_retry = state.get("retry_count", 0)
    dynamic_k = config.BASE_K_CAREER + (current_retry * config.RETRY_K_INCREMENT_CAREER)
    
    # Adaptive Caching Logic
    if is_file_fresh(config.CAREERS_OUTPUT_FILE, config.SCRAPE_CACHE_HOURS):
        print(f"   ✅ Using fresh cached career data ( < {config.SCRAPE_CACHE_HOURS}h old).")
        raw_text = load_from_file(config.CAREERS_OUTPUT_FILE)
    else:
        # 2. Re-scrape if file is old or missing
        print("   ⚠️ Cache expired or missing. Re-scraping live data...")
        raw_text = scrape_careers_tool.invoke({})

    if not raw_text: return {"context_chunks": []}
    
    temp_vectorstore = FAISS.from_texts(split_text_into_chunks(raw_text), embeddings)
    retrieved_docs = temp_vectorstore.similarity_search(question, k=dynamic_k)
    return {"context_chunks": [doc.page_content for doc in retrieved_docs]}

# --- Node 5: NEWS RETRIEVE ---
def news_retrieve_node(state: AgentState):
    print("\n🗞️ Node: News Retrieve (Fast & Adaptive)...")
    question = state["question"]
    current_retry = state.get("retry_count", 0)
    dynamic_k = config.BASE_K_NEWS + (current_retry * config.RETRY_K_INCREMENT_NEWS)
    
    raw_text = ""
    if current_retry > 0 and os.path.exists(config.NEWS_OUTPUT_FILE):
        print(f"   🔄 Retry #{current_retry}: Reading cached news data...")
        raw_text = load_from_file(config.NEWS_OUTPUT_FILE)
    else:
        raw_text = scrape_news_fast_tool.invoke({})

    if not raw_text: return {"context_chunks": []}

    temp_vectorstore = FAISS.from_texts(split_text_into_chunks(raw_text), embeddings)
    retrieved_docs = temp_vectorstore.similarity_search(question, k=dynamic_k)
    return {"context_chunks": [doc.page_content for doc in retrieved_docs]}

# --- Node 6: CONVERSATIONAL ---
def conversational_node(state: AgentState):
    print("\n💬 Node: Conversational (Direct LLM)...")
    question = state["question"]
    prompt = f"""
    User Input: {question}
    Instructions:
    1. You are a helpful corporate assistant for LMKR.
    2. Respond naturally to the greeting or conversational question.
    3. Do NOT make up technical facts. Just be polite.
    """
    response = query_llm_structured(prompt, GeneratedAnswer)
    # Ensure source is set correctly
    if response: response.sources_used = ["Conversational"]
    return {"generated_answer": response or GeneratedAnswer(answer="Hello!", sources_used=["Conversational"]), "context_chunks": []}

# --- Node 7: GENERATE ---
def generate_node(state: AgentState):
    print("\n✍️ Node: Generate (Unified)...")
    context_data = "\n---\n".join(state["context_chunks"])
    today = datetime.now().strftime("%B %d, %Y") # e.g., December 23, 2025

    namespace = ("memories", state["thread_id"], "episodes")
    past_episodes = memory_store.search(namespace, query=state["question"], limit=2)
    
    # Format episodes as context
    formatted_history = "\n".join([f"Past Question: {m.value['q']}\nPast Answer: {m.value['a']}" for m in past_episodes])
    
    prompt = f"""
    You are an expert assistant for LMKR, a global technology and software company. Use the provided Context Data to answer the user's question accurately.
    Past Conversations:
    {formatted_history}

    Context Data:
    {context_data}
    
    Current Date: {today}
    User Question: {state['question']}
    
    Instructions:
    1. Answer using ONLY the Context Data.
    2. The Context contains real dates up to {today}. Report them exactly as written.
    3. Use 'Past Experiences' to maintain consistency in tone or to handle follow-up pronouns.
    4. Answer using 'New Context' only.
    5. Any date in the context that is equal to or prior to {today} should be treated as a factual past event
    6. If information is missing, state "I do not have enough information."
    """
    response = query_llm_structured(prompt, GeneratedAnswer)
    print("   Generated Answer: ", response.answer if response else "No response.")

    # Safety Check for length-limit failures
    if response is None:
        return {"generated_answer": GeneratedAnswer(answer="I hit a processing limit. Please try a more specific question.", sources_used=["Error"])}
        
    return {"generated_answer": response, "retry_count": state.get("retry_count", 0) + 1}

# --- Node 8: OUTPUT GUARD ---
def output_guard_node(state: AgentState):
    print("\n🛡️ Node: Output Guard (Safety Scan)...")
    generation = state["generated_answer"]
    generation.answer = apply_output_guard(generation.answer)
    return {"generated_answer": generation}

# --- Node 9: VALIDATE ---
def validate_node(state: AgentState):
    print("\n🛡️ Node 3: Robust Validation...")
    generation = state["generated_answer"]
    context_text = "\n---\n".join(state["context_chunks"])
    today = datetime.now().strftime("%B %d, %Y")

    prompt = f"""
    Today's Date: {today}
    Generated Answer: {generation.answer}
    Reference Context: {context_text}
    
    Task: Check if the Answer is supported by Context.
    Note: Dates in the Answer that match dates in the Context (e.g., Dec 10, 2025) are VALID. 
    Only fail if the Answer mentions a fact OR a date NOT found in the Reference Context.
    """
    validation = query_llm_structured(prompt, ValidationResult)
    return {"validation": validation or ValidationResult(is_valid=False, reason="Validation failed.")}

# --- Node 10: MEMORY STORE UPDATE ---
def save_memory_node(state: AgentState):
    if state["validation"] and state["validation"].is_valid:
        namespace = ("memories", state["thread_id"], "episodes")
        memory_store.put(
            namespace, 
            key=str(datetime.now().timestamp()), # Unique key per turn
            value={"q": state["question"], "a": state["generated_answer"].answer}
        )
    return state

# --- Node 11: REFLECTION ---
def reflection_node(state: AgentState):
    """
    Analyzes the conversation to extract 'procedural lessons'.
    Only runs if a retry was needed and then succeeded.
    """
    if state["retry_count"] > 0 and state["validation"].is_valid:
        print("🧠 Node: Procedural Reflection (Learning from retries)...")
        
        # Analyze why the first attempt failed and how it was fixed
        prompt = f"""
        User Question: {state['question']}
        Route Taken: {state['destination']}
        Failure Reason: {state['validation'].reason}
        
        Task: Create a concise 'Instruction' for the bot to avoid this mistake next time.
        Example: 'Queries about GVERSE should be routed to retrieve_node, not news_retrieve_node.'
        """
        
        # We use a simple string output for the instruction
        instruction_obj = query_llm_structured(prompt, ProceduralRule)
        if instruction_obj:
            namespace = ("instructions", "global") 
            memory_store.put(namespace, key=f"rule_{datetime.now().timestamp()}", value={"rule": instruction_obj.rule})        
    return state
