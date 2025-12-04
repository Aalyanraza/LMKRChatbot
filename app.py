# LMKR RAG CHATBOT - STREAMLIT APP (OPTIMIZED RETRIEVAL)
# Run with: streamlit run app.py

import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
import logging

# ============================================================================
# LOGGING & SETUP
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ============================================================================
# CONFIG - OPTIMIZED FOR LARGER DATA
# ============================================================================
CONFIG = {
    "project_name": "LMKR RAG Chatbot",
    # CRITICAL UPDATE: Must match the model used in build_lmkr_rag_v2.py
    "embedding_model": "sentence-transformers/all-mpnet-base-v2", 
    "vector_db_path": "./vector_db/faiss_lmkr",
    "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
    "max_tokens": 256,
    "temperature": 0.5,
    "top_p": 0.9,
    "retriever_k": 5,              # INCREASED from 2 to 5 - Get more candidates
    "similarity_threshold": 0.3,   # LOWERED from 0.5 - Less strict filtering
    "max_history": 3,
}

SYSTEM_PROMPT = """You are a helpful customer assistant answering questions about LMKR company.

CONTEXT (Retrieved Documents):
{context}

CONVERSATION HISTORY:
{history}

QUESTION: {question}

Instructions:
1. Answer ONLY based on the provided context and conversation history.
2. If the context and conversation history doesn't have the answer, say "I don't have this information in my knowledge base."
3. Keep answers concise (1-2 sentences).
4. Be accurate and cite the context when possible.

Chain of thought:
1. Analyze the question.
2. Review the context and history for relevant info.
3. Check similarity scores to ensure relevance.
4. Disregard any unrelated context.
5. Formulate a concise, accurate answer that directly addresses the question.


Answer:"""

# ============================================================================
# STREAMLIT CONFIG
# ============================================================================
st.set_page_config(
    page_title="LMKR Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .chat-user {background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .chat-bot {background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .source-badge {background-color: #c8e6c9; padding: 5px 10px; border-radius: 5px; font-size: 0.85em;}
    .stat-box {background-color: #f0f2f6; padding: 15px; border-radius: 8px; text-align: center;}
    .debug-info {background-color: #fff3cd; padding: 10px; border-radius: 5px; font-size: 0.85em; margin: 5px 0;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 LMKR RAG Chatbot")
st.markdown("*Ask questions about LMKR. Powered by Mistral-7B + RAG*")

# ============================================================================
# SESSION STATE
# ============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "rag_components" not in st.session_state:
    st.session_state.rag_components = {}
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# ============================================================================
# RAG COMPONENTS INITIALIZATION
# ============================================================================
@st.cache_resource
def load_rag_components():
    """Load all RAG components once"""
    try:
        logger.info("Loading RAG components...")
        
        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG["embedding_model"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True} # MPNet benefits from normalization
        )
        
        # Vector store
        if not os.path.exists(CONFIG["vector_db_path"]):
             st.error(f"❌ Vector DB not found at {CONFIG['vector_db_path']}. Please run the builder script first.")
             return None

        vectorstore = FAISS.load_local(
            CONFIG["vector_db_path"],
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Retriever - OPTIMIZED
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": CONFIG["retriever_k"]}  # Get more results
        )
        
        # HF Client
        hf_token = os.getenv("HF_API_TOKEN")
        if not hf_token:
            # Fallback for Streamlit Cloud or local testing without .env
            # You can remove this warning if you are sure .env exists
            st.warning("⚠️ HF_API_TOKEN not found in environment variables. LLM generation may fail.")
        
        hf_client = InferenceClient(
            model=CONFIG["llm_model"],
            token=hf_token,
            timeout=60
        )
        
        logger.info("✅ RAG components loaded")
        return {
            "embeddings": embeddings,
            "vectorstore": vectorstore,
            "retriever": retriever,
            "hf_client": hf_client
        }
    except Exception as e:
        logger.error(f"❌ Error loading RAG: {str(e)}")
        st.error(f"Error loading RAG components: {e}")
        return None

# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def extract_text_from_response(response):
    """
    CRITICAL: Extract text from ANY response type.
    Handles: dict, object with .content, string, list
    """
    try:
        # If it's a string, return as-is
        if isinstance(response, str):
            return response.strip()
        
        # If it's a dict, look for content keys
        if isinstance(response, dict):
            # Try common keys
            for key in ["content", "text", "generated_text", "answer"]:
                if key in response:
                    value = response[key]
                    if isinstance(value, str):
                        return value.strip()
                    else:
                        return str(value).strip()
            # If no known key, convert entire dict
            return str(response).strip()
        
        # If it has .content attribute
        if hasattr(response, "content"):
            return str(response.content).strip()
        
        # If it's a list, try first element
        if isinstance(response, list) and len(response) > 0:
            return extract_text_from_response(response[0])
        
        # Fallback: convert to string
        return str(response).strip()
    
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return f"Error processing response: {str(e)}"

def query_hf_api(prompt: str) -> str:
    """
    Query HF InferenceClient safely.
    ALWAYS returns a clean string.
    """
    try:
        components = st.session_state.rag_components
        hf_client = components["hf_client"]
        
        # Call chat completion
        completion = hf_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=CONFIG["max_tokens"],
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
        )
        
        # Extract response (completion is an object)
        # Access: completion.choices[0].message.content
        try:
            response = completion.choices[0].message.content
        except:
            # Fallback for different response structures
            response = str(completion)
        
        # Final extraction using our utility
        return extract_text_from_response(response)
    
    except Exception as e:
        logger.error(f"HF API error: {e}")
        return f"Error: {str(e)}"

def get_relevant_sources(query: str) -> list:
    """Get source documents with metadata - IMPROVED"""
    try:
        components = st.session_state.rag_components
        retriever = components["retriever"]

        # asking llm for for similar questions to add to query
        '''SYSTEM_PROMPT1 = """You are an expert at finding similar questions for retrieval.
Given the user question, provide 2 similar questions that would help in retrieving relevant documents.
User Question: {query}
"""
        prompt = PromptTemplate(
            input_variables=["query"],
            template=SYSTEM_PROMPT1
        )
        similar_questions = query_hf_api(prompt.format(query=query))
        query += " " + similar_questions.replace("\n", " ")
        print(f"Expanded Query for Retrieval: {query}")'''

        # Get more documents
        docs = retriever.invoke(query)
        
        sources = [
            {
                "content": doc.page_content[:300],  # INCREASED from 250
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
                "full_content": doc.page_content  # Store full for debug
            }
            for doc in docs
        ]

        

        '''# reranking based on cross-encoder similarity scores 
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        query_embedding = cross_encoder.encode([query])
        scored_sources = []
        for src in sources:
            doc_embedding = cross_encoder.encode([src["full_content"]])
            score = cross_encoder.predict([query], [src["full_content"]])[0]
            if score >= CONFIG["similarity_threshold"] * 10:  # Scale threshold
                src["similarity_score"] = score
                scored_sources.append(src)
        # Sort by score descending
        scored_sources.sort(key=lambda x: x["similarity_score"], reverse=True)'''
        


        return sources
    except Exception as e:
        logger.error(f"Error getting sources: {e}")
        return []

def format_history(history_list: list) -> str:
    """Format chat history for context"""
    if not history_list:
        return "No previous conversation."
    
    recent = history_list[-CONFIG["max_history"]:]
    formatted = []
    for item in recent:
        formatted.append(f"Q: {item.get('query', '')}")
        formatted.append(f"A: {item.get('answer', '')}")
    return "\n".join(formatted)

def query_rag(user_query: str) -> dict:
    """Main RAG query function - IMPROVED"""
    try:
        # Get sources
        sources = get_relevant_sources(user_query)
        
        # IMPROVED: Combine all source content for better context
        context_parts = []
        for src in sources:
            context_parts.append(src["content"])
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Format history
        history = format_history(st.session_state.chat_history)
        
        # Create final prompt
        final_prompt = SYSTEM_PROMPT.format(
            context=context,
            history=history,
            question=user_query
        )
        
        # Query HF API
        answer = query_hf_api(final_prompt)
        
        result = {
            "query": user_query,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "num_sources_retrieved": len(sources)
        }
        
        logger.info(f"Retrieved {len(sources)} sources for query: {user_query}")
        return result
    
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        return {
            "query": user_query,
            "answer": f"Error: {str(e)}",
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "num_sources_retrieved": 0
        }

# ============================================================================
# INITIALIZE
# ============================================================================
if not st.session_state.initialized:
    with st.spinner("⏳ Loading RAG components..."):
        components = load_rag_components()
        if components:
            st.session_state.rag_components = components
            st.session_state.initialized = True
            st.success("✅ Ready to chat!")
        else:
            st.error("❌ Failed to load components. Check logs.")
            st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.initialized = False
            st.rerun()
    
    st.divider()
    
    st.subheader("📊 Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.chat_history))
    with col2:
        st.metric("Retriever K", CONFIG["retriever_k"])
    
    st.divider()
    
    st.subheader("🛠️ Model Config")
    st.caption(f"**LLM:** {CONFIG['llm_model']}")
    st.caption(f"**Temp:** {CONFIG['temperature']}")
    st.caption(f"**Max Tokens:** {CONFIG['max_tokens']}")
    st.caption(f"**K Retrieved:** {CONFIG['retriever_k']}")
    st.caption(f"**Similarity Threshold:** {CONFIG['similarity_threshold']}")
    
    st.divider()
    
    # DEBUG MODE
    st.subheader("🔍 Debug")
    st.session_state.debug_mode = st.checkbox("Show source details", value=False)

# ============================================================================
# MAIN CHAT AREA
# ============================================================================
st.subheader("💬 Conversation")

# Display history
for msg in st.session_state.chat_history:
    # User message
    st.markdown(f"""
    <div class='chat-user'>
        <strong>You:</strong><br/>
        {msg['query']}
    </div>
    """, unsafe_allow_html=True)
    
    # Bot response
    st.markdown(f"""
    <div class='chat-bot'>
        <strong>Bot:</strong><br/>
        {msg['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Debug info
    if st.session_state.debug_mode:
        st.markdown(f"""
        <div class='debug-info'>
        📊 Sources Retrieved: {msg.get('num_sources_retrieved', 0)}<br/>
        ⏱️ Timestamp: {msg['timestamp']}
        </div>
        """, unsafe_allow_html=True)
    
    # Sources
    if msg.get("sources"):
        with st.expander(f"📚 Sources ({len(msg['sources'])})"):
            for i, src in enumerate(msg["sources"], 1):
                st.markdown(f"**[Source {i}]**")
                st.write(src['content'])
                if st.session_state.debug_mode and src.get("full_content"):
                    with st.expander("Full content"):
                        st.write(src["full_content"])

# ============================================================================
# INPUT AREA
# ============================================================================
st.divider()

col_input, col_send = st.columns([6, 1])

with col_input:
    user_input = st.text_input(
        "Ask about LMKR...",
        placeholder="When was LMKR founded?",
        label_visibility="collapsed"
    )

with col_send:
    send_btn = st.button("Send", use_container_width=True, type="primary")

# Process input
if send_btn and user_input:
    with st.spinner("🤔 Thinking..."):
        result = query_rag(user_input)
        st.session_state.chat_history.append(result)
    
    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.8em;'>"
    "🚀 LMKR RAG Chatbot v2.0 | Mistral-7B + LangChain + Streamlit<br/>"
    f"Config: K={CONFIG['retriever_k']} | Threshold={CONFIG['similarity_threshold']}"
    "</p>",
    unsafe_allow_html=True
)