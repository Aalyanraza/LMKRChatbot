import streamlit as st
import requests
import time
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="LMKR Intelligent Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Professional & Modern) ---
st.markdown("""
<style>
    /* Main Color Scheme */
    :root {
        --primary-color: #1f6feb;
        --secondary-color: #0d47a1;
        --accent-color: #00d9ff;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --bg-light: #f8fafc;
        --bg-dark: #0f172a;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
    }
    
    /* Global Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body {
        height: 100%;
        width: 100%;
        overflow: hidden;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0d47a1 100%);
        background-attachment: fixed;
    }
    
    /* Main Container */
    .main {
        background: transparent;
        border-radius: 8px;
        padding: 0;
        height: 100vh;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    /* Scrollbar Styling */
    .main::-webkit-scrollbar {
        width: 8px;
    }
    
    .main::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    .main::-webkit-scrollbar-thumb {
        background: rgba(31, 111, 235, 0.5);
        border-radius: 4px;
    }
    
    .main::-webkit-scrollbar-thumb:hover {
        background: rgba(31, 111, 235, 0.8);
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, rgba(31, 111, 235, 0.95) 0%, rgba(13, 71, 161, 0.95) 100%);
        padding: 40px 30px;
        border-radius: 16px;
        color: white;
        margin: 30px 30px 20px 30px;
        box-shadow: 0 8px 32px rgba(31, 111, 235, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .header-container h1 {
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .header-container p {
        font-size: 1.1em;
        opacity: 0.95;
        margin: 0;
        font-weight: 300;
    }
    
    /* Content Wrapper */
    .content-wrapper {
        padding: 0 30px;
        margin-bottom: 30px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] div[data-testid="stTextInput"] input::placeholder {
    color: #64748b;
    opacity: 1;
}
    
    [data-testid="stSidebar"] > div > div {
        padding: 20px;
    }
    
    .sidebar-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }
    
    .sidebar-card:hover {
        box-shadow: 0 4px 16px rgba(31, 111, 235, 0.1);
        border-color: var(--primary-color);
    }
    
    .sidebar-card h3 {
        color: var(--primary-color);
        margin-bottom: 12px;
        font-size: 1.1em;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        margin: 12px 0;
        padding: 16px;
        border-left: 4px solid transparent;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .stChatMessage[data-testid*="user"] {
        border-left-color: var(--primary-color);
        background: linear-gradient(135deg, rgba(240, 244, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
    }
    
    .stChatMessage[data-testid*="assistant"] {
        border-left-color: var(--success-color);
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(31, 111, 235, 0.35);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Input Fields */
    [data-testid="stSidebar"] div[data-testid="stTextInput"] input {
        border-radius: 8px;
        border: 2px solid var(--border-color);
        padding: 12px 16px;
        font-size: 1em;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.95);
        color: #1e293b;
    }

    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.1);
    }
    
    /* Chat Input */
    .stChatInputContainer {
        padding: 20px 30px;
        background: rgba(255, 255, 255, 0.95);
        border-top: 1px solid var(--border-color);
        border-radius: 0;
        box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.1);
        margin: 0 -30px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(248, 250, 252, 0.95) 0%, rgba(240, 244, 255, 0.95) 100%);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(240, 244, 255, 0.95) 0%, rgba(224, 231, 255, 0.95) 100%);
        border-color: var(--primary-color);
    }
    
    /* Info/Error/Warning Boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
        padding: 16px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stAlert[data-testid*="info"] {
        background: linear-gradient(135deg, rgba(207, 250, 254, 0.95) 0%, rgba(236, 253, 245, 0.95) 100%);
        border-left-color: var(--accent-color);
    }
    
    .stAlert[data-testid*="error"] {
        background: linear-gradient(135deg, rgba(254, 226, 226, 0.95) 0%, rgba(254, 242, 242, 0.95) 100%);
        border-left-color: var(--error-color);
    }
    
    .stAlert[data-testid*="success"] {
        background: linear-gradient(135deg, rgba(220, 252, 231, 0.95) 0%, rgba(240, 253, 244, 0.95) 100%);
        border-left-color: var(--success-color);
    }
    
    .stAlert[data-testid*="warning"] {
        background: linear-gradient(135deg, rgba(254, 243, 199, 0.95) 0%, rgba(255, 251, 235, 0.95) 100%);
        border-left-color: var(--warning-color);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, var(--border-color), transparent);
        margin: 20px 0;
    }
    
    /* Spinner - Custom Animation */
    .stSpinner > div {
        border-color: var(--primary-color);
    }
    
    /* Caption & Small Text */
    .stCaption {
        color: var(--text-secondary);
        font-size: 0.9em;
    }
    
    /* Markdown Styling */
    p, li {
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    h2, h3, h4 {
        color: var(--text-primary);
    }
    
    /* Source Chunks Styling */
    .source-chunk {
        background: linear-gradient(135deg, rgba(240, 244, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        border-left: 4px solid var(--primary-color);
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 0.95em;
        line-height: 1.5;
        backdrop-filter: blur(10px);
    }
    
    /* Badge Style */
    .badge {
        display: inline-block;
        background: var(--accent-color);
        color: var(--bg-dark);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin-right: 8px;
    }
    
    /* Custom Loader Animation */
    @keyframes shimmer {
        0% {
            background: linear-gradient(90deg, rgba(31, 111, 235, 0.2) 0%, rgba(31, 111, 235, 0.4) 50%, rgba(31, 111, 235, 0.2) 100%);
            background-size: 200% 100%;
        }
        50% {
            background-position: 200% 0;
        }
        100% {
            background-position: -200% 0;
        }
    }
    
    @keyframes pulse-dot {
        0%, 100% {
            opacity: 0.3;
            transform: scale(1);
        }
        50% {
            opacity: 1;
            transform: scale(1.1);
        }
    }
    
    .custom-loader {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 12px 20px;
        background: linear-gradient(135deg, rgba(31, 111, 235, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%);
        border-radius: 8px;
        border: 1px solid rgba(31, 111, 235, 0.3);
        color: var(--primary-color);
        font-weight: 500;
        animation: shimmer 2s infinite;
    }
    
    .loader-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--primary-color);
        animation: pulse-dot 1.5s infinite;
    }
    
    .loader-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .loader-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    /* Welcome Card */
    .welcome-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 30px;
    }
    
    .welcome-card h2 {
        font-size: 2.2em;
        color: var(--primary-color);
        margin-bottom: 15px;
    }
    
    .welcome-card p {
        font-size: 1.1em;
        color: var(--text-secondary);
        margin-bottom: 25px;
    }
    
    .example-box {
        background: linear-gradient(135deg, rgba(240, 244, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        border-left: 4px solid var(--primary-color);
        padding: 20px;
        border-radius: 8px;
        text-align: left;
        display: inline-block;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .header-container {
            padding: 25px 20px;
            margin: 20px 20px 15px 20px;
        }
        
        .header-container h1 {
            font-size: 1.8em;
        }
        
        .content-wrapper {
            padding: 0 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### LMKR Assistant")
    
    # Logo with fallback
    try:
        st.image("static/lmkr.png", width=120)
    except:
        st.image("https://lmkr.com/wp-content/uploads/2019/03/lmkr-logo.png", width=120)
    
    st.markdown("---")
    
    # Session Info Card
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ Session Configuration")
    user_id = st.text_input(
        "User ID",
        value="dev_user_01",
        help="Unique identifier for this session",
        key="user_id_input"
    )

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Controls Card
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("#### 🎛️ Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Info Card
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("#### 📚 Topics Available")
    st.markdown("""
    - 🏢 LMKR Careers
    - 💼 GVERSE Software
    - 📢 Company Announcements
    - 🎯 General Information
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics Card
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Session Stats")
    message_count = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", message_count, delta=None)
    with col2:
        st.metric("Responses", len([m for m in st.session_state.get("messages", []) if m["role"] == "assistant"]), delta=None)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🔐 All data is processed securely | Powered by RAG")

# --- Main Content Area ---
# Header Section
st.markdown("""
<div class="header-container">
    <h1>🤖 LMKR Intelligent Assistant</h1>
</div>
""", unsafe_allow_html=True)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Conversation ---
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

if st.session_state.messages:
    st.markdown("### 💬 Conversation History")
    
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            
            # Display sources if they exist
            if "sources" in message and message["sources"]:
                with st.expander(f"📚 View {len(message['sources'])} Source(s)", expanded=False):
                    st.caption("The following text segments were used to generate this response:")
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f'<div class="source-chunk">**[{idx}]** {source}</div>', unsafe_allow_html=True)
else:
    # Welcome Message when no messages
    # Only show welcome card if no messages exist
    if len(st.session_state.get("messages", [])) == 0:
        st.markdown("""
        <div class="welcome-card">
        <h2>Welcome!</h2>
        <p>Ask me anything about LMKR, careers, GVERSE Software, or company announcements.</p>
        <div class="example-box">
            <p><strong>💡 Example questions:</strong></p>
            <ul style="margin: 10px 0; color: #1e293b; text-align: left;">
                <li>What career opportunities are available at LMKR?</li>
                <li>Tell me about GVERSE Software</li>
                <li>What are the latest company announcements?</li>
            </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True)

# --- Chat Input & Processing ---
if prompt := st.chat_input("How can I help you today?", key="chat_input"):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)
    
    # Call API and display response
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        loader_placeholder = st.empty()
        
        # Custom loader
        loader_placeholder.markdown("""
        <div class="custom-loader">
            <div class="loader-dot"></div>
            <div class="loader-dot"></div>
            <div class="loader-dot"></div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Make API call (same as original)
            payload = {"question": prompt, "user_id": user_id}
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json=payload,
                timeout=50
            )
            
            loader_placeholder.empty()
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer found.")
                sources = data.get("sources", [])
                
                # Display answer
                response_placeholder.markdown(answer)
                
                # Display sources if available
                if sources:
                    st.divider()
                    with st.expander(f"🔍 Citations & Evidence ({len(sources)} chunks)", expanded=False):
                        st.caption("The following text segments were used to generate this response:")
                        for idx, chunk in enumerate(sources, 1):
                            st.markdown(f'<div class="source-chunk">**[{idx}]** {chunk}</div>', unsafe_allow_html=True)
                
                # Update session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            else:
                loader_placeholder.empty()
                error_msg = f"⚠️ API Error ({response.status_code})"
                response_placeholder.error(error_msg)
                st.error(f"Server response: {response.text}")
                
        except requests.exceptions.Timeout:
            loader_placeholder.empty()
            error_msg = "⏱️ Request timed out. Please try again."
            response_placeholder.error(error_msg)
            
        except requests.exceptions.ConnectionError:
            loader_placeholder.empty()
            error_msg = "❌ Connection Failed"
            response_placeholder.error(error_msg)
            st.error("Ensure your FastAPI server is running on `http://127.0.0.1:8000`")
            
        except Exception as e:
            loader_placeholder.empty()
            error_msg = f"❌ Error: {str(e)}"
            response_placeholder.error(error_msg)

# --- Footer ---
st.markdown("""
<div style="text-align: center; color: rgba(255, 255, 255, 0.6); font-size: 0.9em; padding: 20px 30px; margin-top: 30px;">
    <p>🔐 <strong>Privacy Protected</strong> • ⚡ <strong>Powered by RAG</strong> • 🧠 <strong>AI-Driven</strong></p>
    <p style="margin-top: 10px;">© 2026 LMKR | All rights reserved</p>
</div>
""", unsafe_allow_html=True)