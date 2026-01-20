# Graph Nodes - 1:1 Functional Match with OpenAI Refactor
import os
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from models import (
    QueryAugmentation, GeneratedAnswer, ValidationResult, 
    RouteDecision, AgentState, ProceduralRule
)
from langchain_core.messages import SystemMessage, HumanMessage
from llm_helpers import query_llm_structured, get_streaming_llm
from guards import detect_malicious_prompt, apply_input_guard, apply_output_guard
from tools import scrape_careers_tool, scrape_news_fast_tool, lookup_policy_tool
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

def decision_node(state: AgentState):
    print("\n🤖 Node: Decision (LLM Selecting Tools)...")
    question = state["question"]
    
    # 1. Bind Tools to the LLM
    llm = get_streaming_llm()
    tools = [lookup_policy_tool, scrape_careers_tool, scrape_news_fast_tool]
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. Invoke LLM to decide
    # We force it to decide: either call a tool or just chat.
    response = llm_with_tools.invoke(question)
    
    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
    
    if tool_calls:
        print(f"   👉 LLM decided to call {len(tool_calls)} tools: {[t['name'] for t in tool_calls]}")
    else:
        print("   👉 LLM decided this is conversational (No tools).")

    return {"tool_calls": tool_calls}

# --- UPDATED Node 3: TOOL EXECUTION NODE ---
def tool_execution_node(state: AgentState):
    print("\n🛠️ Node: Executing Tools...")
    tool_calls = state["tool_calls"]
    results = []
    question = state["question"]  # We need the question for the local search
    
    # Map tool names to actual functions
    tool_map = {
        "lookup_policy_tool": lookup_policy_tool,
        "scrape_careers_tool": scrape_careers_tool,
        "scrape_news_fast_tool": scrape_news_fast_tool
    }
    
    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        
        # --- SPECIAL HANDLING FOR CAREERS (Caching + Local Search) ---
        if tool_name == "scrape_careers_tool":
            print(f"   ▶️ Handling {tool_name} with Caching Strategy...")
            try:
                raw_text = ""
                
                # 1. Check Cache Freshness
                if is_file_fresh(config.CAREERS_OUTPUT_FILE, config.SCRAPE_CACHE_HOURS):
                    print(f"      ✅ Cache Hit: Loading careers data from file (Fresh < {config.SCRAPE_CACHE_HOURS}h).")
                    raw_text = load_from_file(config.CAREERS_OUTPUT_FILE)
                else:
                    print("      ⚠️ Cache Miss: Invoking Scraper Tool...")
                    # Invoke the tool directly to scrape and get text
                    raw_text = tool_map[tool_name].invoke(tool_args)

                # 2. Perform Local Vector Search (Context Filtering)
                # We do this to avoid dumping the whole page into the prompt
                if raw_text:
                    print(f"      🔍 Searching through careers text for: '{question}'")
                    chunks = split_text_into_chunks(raw_text)
                    temp_vectorstore = FAISS.from_texts(chunks, embeddings)
                    # Retrieve top 4 most relevant chunks
                    relevant_docs = temp_vectorstore.similarity_search(question, k=4)
                    search_results = "\n\n".join([d.page_content for d in relevant_docs])
                    results.append(f"CAREERS DATA (Filtered):\n{search_results}")
                else:
                    results.append("CAREERS DATA: No information found.")

            except Exception as e:
                print(f"      ❌ Error processing careers data: {e}")
                results.append(f"Error processing careers: {str(e)}")

        # --- STANDARD HANDLING FOR OTHER TOOLS ---
        elif tool_name in tool_map:
            print(f"   ▶️ Running {tool_name}...")
            try:
                output = tool_map[tool_name].invoke(tool_args)
                results.append(str(output))
            except Exception as e:
                print(f"   ❌ Error executing {tool_name}: {e}")
                results.append(f"Error executing {tool_name}: {str(e)}")
                
    return {"context_chunks": results}

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
    
    # Save all context retrieved to file
    context_output = "\n---CHUNK---\n".join(unique_context)
    save_to_file(context_output, config.CONTEXT_DEBUG_FILE)
    print(f"   ✅ Context saved to {config.CONTEXT_DEBUG_FILE}")
    
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
    context_chunks = [doc.page_content for doc in retrieved_docs]
    
    # Save all context retrieved to file
    context_output = "\n---CHUNK---\n".join(context_chunks)
    save_to_file(context_output, config.CAREERS_OUTPUT_FILE + ".context")
    print(f"   ✅ Career context saved to {config.CAREERS_OUTPUT_FILE}.context")
    
    return {"context_chunks": context_chunks}

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
    context_chunks = [doc.page_content for doc in retrieved_docs]
    
    # Save all context retrieved to file
    context_output = "\n---CHUNK---\n".join(context_chunks)
    save_to_file(context_output, config.NEWS_OUTPUT_FILE + ".context")
    print(f"   ✅ News context saved to {config.NEWS_OUTPUT_FILE}.context")
    
    return {"context_chunks": context_chunks}

# --- Node 6: CONVERSATIONAL ---
def conversational_node(state: AgentState):
    print("\n💬 Node: Conversational (Receptionist Persona)...")
    question = state["question"]
    
    # 1. Engineered System Prompt
    # Defines role, tone, and strict topic boundaries.
    system_prompt = """
    You are the Virtual Receptionist for LMKR, a global petroleum technology and software company.
    
    YOUR ROLE:
    - You are the first point of contact on the LMKR website.
    - Be professional, warm, concise, and helpful.
    - Assist with greetings, navigation, and high-level company inquiries.
    - Keep it mid-length (2-3 sentences) and to the point unless necessitated otherwise.
    
    BOUNDARIES & RESTRICTIONS:
    - You are NOT a general purpose AI assistant. Do NOT answer general trivia, math problems, or questions about biology, pop culture, politics or any question unrelated to LMKR.
    - If a user asks an off-topic question, politely decline and steer the conversation back to LMKR.
    
    RESPONSE EXAMPLES:
    - User: "What is the strongest animal?"
      You: "I am designed to assist with LMKR-related inquiries, so I don't have information on wildlife. However, I can help you with our GVERSE software or career opportunities!"
    - User: "Hi"
      You: "Hello! Welcome to LMKR. How can I assist you today? I can help with information about our software solutions, services, or job openings."
    """
    
    # Use standard streaming LLM
    llm = get_streaming_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    # .invoke() here waits for full completion BUT emits events we can catch in app.py
    response = llm.invoke(messages)
    
    return {
        "generated_answer": GeneratedAnswer(answer=response.content, sources_used=["Conversational"]),
        "context_chunks": []
    }

# --- Node 7: GENERATE ---
async def generate_node(state: AgentState):
    print("\n✍️ Node: Generate (Hybrid Mode)...")
    context_chunks = state.get("context_chunks", [])
    question = state["question"]
    today = datetime.now().strftime("%B %d, %Y")
    
    # MODE A: CONVERSATIONAL (No Context)
    if not context_chunks:
        print("   Model: Conversational Mode")
        system_prompt = """
        You are the Virtual Receptionist for LMKR.
        - The user's query did not trigger any database lookups, so it is likely a greeting or general chat.
        - Be professional, warm, and concise.
        - If the user asks a specific business question that YOU DO NOT know, admit it or suggest they ask about "Jobs", "News", or "GVERSE".
        - Do NOT hallucinate company data.
        - if the user asks about the number of employees or headcount, tell them "LMKR has over 700 employees worldwide."
        """
        user_content = question

    # MODE B: RAG (Context Present)
    else:
        print(f"   Model: RAG Mode ({len(context_chunks)} chunks)")
        context_data = "\n---\n".join(context_chunks)
        system_prompt = f"""
        You are an expert assistant for LMKR.
        
        CRITICAL INSTRUCTIONS:
        1. Answer using ONLY the provided Context Data below.
        2. If the answer is not in the context, state "I don't have that information in my current records."
        3. Do not mention competitors like Schlumberger.
        4. Be helpful and structure your answer clearly.
        5. If the user asks about the number of employees or headcount, tell them "LMKR has over 700 employees worldwide."
        
        """
        user_content = f"Context Data:\n{context_data}\n\nUser Question: {question}\nCurrent Date: {today}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    llm = get_streaming_llm()
    response = await llm.ainvoke(messages)
    
    return {
        "generated_answer": GeneratedAnswer(answer=response.content, sources_used=["Dynamic Tooling"]),
        "context_chunks": context_chunks # Pass through for UI
    }

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
