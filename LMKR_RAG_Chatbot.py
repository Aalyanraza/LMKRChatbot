import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import torch
from typing import List, Dict

# ---- HuggingFace & LangChain ----
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="LMKR RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Constants
DATA_PATH = "Data/lmkr_data/lmkr_combined.txt"
VECTOR_DB_PATH = "./vector_db/faiss_lmkr"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Tunable Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 3
SIMILARITY_THRESHOLD = 0.5

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .chat-user {background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; color: #000000;}
    .chat-bot {background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin: 10px 0; color: #000000;}
    .source-badge {background-color: #c8e6c9; padding: 5px 10px; border-radius: 5px; font-size: 0.85em;}
    .debug-info {background-color: #fff3cd; padding: 10px; border-radius: 5px; font-size: 0.85em; margin: 5px 0; color: #555;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CACHED RESOURCE LOADING
# ============================================================

@st.cache_resource

def load_embeddings():
    """Load and cache the embedding model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": False}
    )

@st.cache_resource
def setup_vector_db(_embeddings):
    """
    Load existing Vector DB or Create new one if it doesn't exist.
    """
    # 1. Try to load existing DB
    if os.path.exists(VECTOR_DB_PATH):
        try:
            vector_db = FAISS.load_local(VECTOR_DB_PATH, _embeddings, allow_dangerous_deserialization=True)
            return vector_db
        except Exception as e:
            st.error(f"Error loading existing DB: {e}")

    # 2. If not found, create new from data
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at: {DATA_PATH}")
        return None

    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            raw_data = f.read()

        # Preprocess
        import re
        cleaned_data = re.sub(r'\n+', '\n', raw_data).strip()
        
        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(cleaned_data)
        
        # Create DB
        vector_db = FAISS.from_texts(
            texts=chunks,
            embedding=_embeddings,
            metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))]
        )
        
        # Save
        os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
        vector_db.save_local(VECTOR_DB_PATH)
        
        return vector_db
    except Exception as e:
        st.error(f"Error creating vector DB: {e}")
        return None

@st.cache_resource
def setup_llm_client():
    """Initialize HuggingFace Client."""
    token = os.getenv("HF_API_TOKEN")
    if not token:
        st.error("HF_API_TOKEN not found in environment variables!")
        return None
        
    return InferenceClient(
        model=LLM_MODEL_NAME,
        token=token,
        timeout=60
    )

# ============================================================
# LOGIC FUNCTIONS
# ============================================================

def hf_generate(prompt: str, client) -> str:
    """Call HF API safely."""
    try:
        completion = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
            top_p=0.9,
        )
        msg = completion.choices[0].message
        return msg.content or ""
    except Exception as e:
        return f"Error communicating with LLM: {e}"

def get_rag_response(query, vector_store, client, history):
    """Orchestrate the RAG flow."""
    # 1. Retrieve - FIX: Used 'similarity_search_with_score' (singular)
    # Get docs with scores for filtering
    docs_and_scores = vector_store.similarity_search_with_score(query, k=RETRIEVER_K)
    
    # Filter by threshold 
    relevant_docs = []
    
    for doc, score in docs_and_scores:
        # FAISS L2 score: Lower is better (0 is identical).
        # We can accept everything for now or filter if score is too high (e.g., > 1.5)
        relevant_docs.append(doc)

    if not relevant_docs:
        return {
            "answer": "I couldn't find relevant information in the documents.",
            "sources": [],
            "confidence": "LOW"
        }

    # 2. Format Context
    context_text = "\n\n".join([d.page_content for d in relevant_docs])
    
    # 3. Format History
    history_text = ""
    for h in history[-2:]: # Keep last 2
        history_text += f"User: {h['query']}\nAssistant: {h['answer']}\n"

    # 4. Prompt
    system_prompt = f"""You are a helpful assistant for LMKR.
    
    CONTEXT INFORMATION:
    {context_text}
    
    CHAT HISTORY:
    {history_text}
    
    USER QUESTION: {query}
    
    INSTRUCTIONS:
    - Answer ONLY based on the context provided.
    - If the answer is not in the context, say "I don't have that information."
    - Keep answer to 2-3 sentences.
    
    ANSWER:"""

    # 5. Generate
    response = hf_generate(system_prompt, client)
    
    # Clean response
    if "ANSWER:" in response:
        response = response.split("ANSWER:")[-1].strip()

    return {
        "answer": response,
        "sources": relevant_docs,
        "confidence": "HIGH" if len(relevant_docs) > 0 else "LOW"
    }

# ============================================================
# MAIN APP UI
# ============================================================

def main():
    st.title("🤖 LMKR RAG Chatbot")
    st.caption("Powered by LangChain, FAISS & Mistral-7B")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
            
        st.divider()
        st.subheader("Debug")
        st.session_state.show_sources = st.checkbox("Show Sources", value=False)
        
        st.divider()
        st.info("Files loaded from: " + DATA_PATH)

    # Initialize State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load Resources (Cached)
    with st.spinner("Loading AI Models..."):
        embeddings = load_embeddings()
        vector_store = setup_vector_db(embeddings)
        client = setup_llm_client()

    if not vector_store or not client:
        st.error("Failed to initialize system. Check API Token and Data Path.")
        st.stop()

    # Display History
    for msg in st.session_state.chat_history:
        # User
        st.markdown(f"<div class='chat-user'><strong>👤 You:</strong><br>{msg['query']}</div>", unsafe_allow_html=True)
        # Bot
        st.markdown(f"<div class='chat-bot'><strong>🤖 Assistant:</strong><br>{msg['answer']}</div>", unsafe_allow_html=True)
        
        # Sources (if debug enabled)
        if st.session_state.show_sources and "sources" in msg:
            with st.expander(f"📚 Context Sources ({len(msg['sources'])})"):
                for doc in msg['sources']:
                    st.caption(f"...{doc.page_content[:200]}...")
                    st.divider()

    # Input Area
    prompt = st.chat_input("Ask about LMKR...")

    if prompt:
        with st.spinner("Thinking..."):
            result = get_rag_response(prompt, vector_store, client, st.session_state.chat_history)
            
            # Save to history
            st.session_state.chat_history.append({
                "query": prompt,
                "answer": result["answer"],
                "sources": result["sources"]
            })
            
        st.rerun()

if __name__ == "__main__":
    main()