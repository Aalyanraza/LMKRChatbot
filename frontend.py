import streamlit as st
import requests

# Page Config
st.set_page_config(page_title="LMKR Assistant", page_icon="🤖", layout="centered")

# Custom CSS for dark mode adjustments
st.markdown("""
<style>
    .stChatMessage {
        background-color: ##FFFFFF; 
        border-radius: 10px;
    }
    /* Hide the "Deploy" button for cleaner look */
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 LMKR Intelligent Assistant")
st.markdown("Ask about *Careers*, *News*, or *Company Info*.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What would you like to know?"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Make request to your FastAPI backend
                response = requests.post(
                    "http://127.0.0.1:8000/chat", 
                    json={"question": prompt}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer received.")
                    st.markdown(answer)
                    
                    # Save context
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the backend.")