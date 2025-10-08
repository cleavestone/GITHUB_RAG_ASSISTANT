import streamlit as st
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# -------------------- PROJECT PATH -------------------- #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chat import agent  # 👈 your LangGraph agent lives here

from langchain_core.messages import HumanMessage

# -------------------- LOAD ENV -------------------- #
load_dotenv()

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(
    page_title="GitHub Portfolio Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- LOAD CSS -------------------- #
css_file = Path(__file__).parent / "static" / "css" / "style.css"
if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------- SESSION STATE -------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- HEADER -------------------- #
st.markdown("""
<div class="main-header">
    <h1>⚡ GitHub Portfolio Intelligence</h1>
    <p>AI-Powered Project Discovery & Analysis</p>
</div>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR -------------------- #
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-header" style="margin-top: 2rem;">📊 Session Analytics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.messages)}</div>
            <div class="metric-label">Messages</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        responses = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{responses}</div>
            <div class="metric-label">Responses</div>
        </div>
        """, unsafe_allow_html=True)

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.experimental_rerun()

# -------------------- CHAT -------------------- #
chat_container = st.container()
with chat_container:
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-container">
            <h2>👋 Welcome to Your Project Intelligence Hub</h2>
            <p>Ask me anything about your GitHub portfolio. I'll analyze your projects using advanced RAG technology to provide intelligent, context-aware responses.</p>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

# -------------------- INPUT -------------------- #
if prompt := st.chat_input("💬 Ask about projects, technologies, or implementations..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            with st.spinner("🔍 Analyzing project portfolio..."):
                # ---- LangGraph agent call ----
                state = {"messages": [HumanMessage(content=prompt)]}
                result = agent.invoke(state)

                # Get last message content
                last_msg = result["messages"][-1].content
                full_response = last_msg

                message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"❌ **Error:** Unable to process request\n\n`{str(e)}`"
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
