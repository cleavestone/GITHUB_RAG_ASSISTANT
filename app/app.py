# app.py

import streamlit as st
import sys
import os
from datetime import datetime
import json

# -------------------- PROJECT PATH -------------------- #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.chat import stream_agent_response_realtime

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(
    page_title="Portfolio AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- LOAD CSS -------------------- #
if os.path.exists("static/css/style.css"):
    with open("static/css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------- STATE -------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()

if "example_clicked" not in st.session_state:
    st.session_state.example_clicked = None

# -------------------- HEADER -------------------- #
st.markdown("""
<div class="main-header">
    <h1> Portfolio AI Assistant</h1>
    <p class="subtitle">Intelligent project discovery powered by LangGraph & RAG</p>
    <div class="status-badge">
        <span class="status-dot"></span>
        <span>AI Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR -------------------- #
with st.sidebar:
    st.markdown('<div class="sidebar-header"> Chat Controls</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_start = datetime.now()
            st.rerun()

    with col2:
        if st.button("📥 Export", use_container_width=True):
            if st.session_state.messages:
                export_data = {
                    "session_start": str(st.session_state.session_start),
                    "messages": st.session_state.messages
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

    st.markdown('<div class="sidebar-header" style="margin-top: 2rem;"> Retrieval Settings</div>', unsafe_allow_html=True)
    project_limit = st.slider("Projects to retrieve:", 1, 10, 5)

# -------------------- WELCOME SCREEN -------------------- #
if len(st.session_state.messages) == 0 and not st.session_state.example_clicked:
    st.markdown("""
    <div class="welcome-container">
        <h2>👋 Welcome to Your Portfolio Assistant</h2>
        <p>I'm an AI-powered assistant that can help you explore GitHub projects, discuss technical implementations, or just have a friendly chat. Ask me anything!</p>
        <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 1.5rem;">
             Try an example from the sidebar or type your own query below.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- CHAT HISTORY -------------------- #
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "action" in message:
            action_class = "action-retrieve" if message["action"] == "retrieve" else "action-chat"
            action_label = "📂 Retrieved Projects" if message["action"] == "retrieve" else "💬 General Chat"
            st.markdown(f'<div class="action-badge {action_class}">{action_label}</div>', unsafe_allow_html=True)
        st.markdown(message["content"])

# -------------------- CHAT INPUT -------------------- #
if prompt := st.chat_input(" Ask about projects, technologies, or chat normally..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Show immediate spinner and status
        status_container = st.empty()
        response_placeholder = st.empty()
        
        # Initial state: analyzing
        with status_container:
            with st.spinner("🔍 Analyzing your query..."):
                full_response = ""
                action_type = ""
                first_chunk_received = False

                try:
                    stream = stream_agent_response_realtime(prompt)

                    for chunk in stream:
                        if chunk["type"] == "metadata":
                            first_chunk_received = True
                            action_type = chunk["action"]
                            action_class = "action-retrieve" if action_type == "retrieve" else "action-chat"
                            
                            # Update status based on action
                            if action_type == "retrieve":
                                action_label = "📂 Retrieving Projects..."
                            else:
                                action_label = "💬 Thinking..."
                            
                            # Clear spinner and show action badge
                            status_container.markdown(
                                f'<div class="action-badge {action_class}">{action_label}</div>',
                                unsafe_allow_html=True
                            )

                        elif chunk["type"] == "content":
                            full_response += chunk["content"]
                            response_placeholder.markdown(full_response + "▌")

                        elif chunk["type"] == "done":
                            response_placeholder.markdown(full_response)

                            # Final status
                            if action_type:
                                action_class = "action-retrieve" if action_type == "retrieve" else "action-chat"
                                action_label = "📂 Retrieved Projects" if action_type == "retrieve" else "💬 General Chat"
                                status_container.markdown(
                                    f'<div class="action-badge {action_class}">{action_label}</div>',
                                    unsafe_allow_html=True
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "action": action_type
                    })

                except Exception as e:
                    error_msg = f"**Error:** {str(e)}"
                    response_placeholder.markdown(error_msg)
                    status_container.empty()  # Clear spinner on error
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })