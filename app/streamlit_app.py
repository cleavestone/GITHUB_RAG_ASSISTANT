import streamlit as st
import sys
import os
import pandas as pd

# -------------------- PROJECT PATH -------------------- #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retriever import Retriever

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(
    page_title="GitHub Portfolio Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS -------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; letter-spacing: -0.01em; }

.stApp {
    background: #0a0a0f;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(147, 51, 234, 0.15) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(14, 165, 233, 0.08) 0px, transparent 50%);
}

#MainMenu, footer, .stDeployButton, header {visibility: hidden;}

/* Header */
.main-header {padding: 2rem 0 1.5rem 0; text-align: center; margin-bottom: 2rem; border-bottom: 1px solid rgba(255, 255, 255, 0.06);}
.main-header h1 {font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; letter-spacing: -0.03em;}
.main-header p {color: #94a3b8; font-size: 1rem; font-weight: 400;}

/* Chat messages */
.stChatMessage {border-radius: 16px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; border: 1px solid rgba(255,255,255,0.06); background: rgba(255,255,255,0.02); backdrop-filter: blur(20px);}
.stChatMessage[data-testid="user-message"] {background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%); border: 1px solid rgba(99,102,241,0.3);}
.stChatMessage[data-testid="assistant-message"] {background: linear-gradient(135deg, rgba(14,165,233,0.1) 0%, rgba(6,182,212,0.08) 100%); border: 1px solid rgba(14,165,233,0.25);}
.stChatMessage p {color: #e2e8f0 !important; line-height: 1.7; margin: 0; font-size: 0.95rem;}

/* Welcome card */
.welcome-container {max-width: 800px; margin: 3rem auto; padding: 3rem; background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(139,92,246,0.08) 100%); border-radius: 24px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(20px); text-align: center;}
.welcome-container h2 {color: #f1f5f9; font-size: 2rem; font-weight: 700; margin-bottom: 1rem; letter-spacing: -0.02em;}
.welcome-container p {color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem; line-height: 1.6;}

/* Sidebar styling */
section[data-testid="stSidebar"] {background: rgba(10,10,15,0.95); backdrop-filter: blur(20px); border-right: 1px solid rgba(255,255,255,0.08);}
section[data-testid="stSidebar"] > div {padding-top: 2rem;}

/* Metric cards */
.metric-card {background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 1rem; text-align: center; transition: all 0.3s ease;}
.metric-card:hover {background: rgba(255,255,255,0.08); transform: translateY(-2px);}
.metric-value {font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #6366f1, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.metric-label {color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem;}

/* Buttons */
.stButton > button {width: 100%; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; color: #e2e8f0; padding: 0.75rem 1rem; font-weight: 500; transition: all 0.3s ease; font-size: 0.9rem;}
.stButton > button:hover {background: rgba(99,102,241,0.2); border-color: rgba(99,102,241,0.4); transform: translateX(4px);}

/* Scrollbar */
::-webkit-scrollbar {width: 8px; height: 8px;}
::-webkit-scrollbar-track {background: rgba(255,255,255,0.02);}
::-webkit-scrollbar-thumb {background: rgba(99,102,241,0.3); border-radius: 10px;}
::-webkit-scrollbar-thumb:hover {background: rgba(99,102,241,0.5);}
</style>
""", unsafe_allow_html=True)

# -------------------- STATE -------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    with st.spinner("⚡ Initializing Intelligence Engine..."):
        st.session_state.retriever = Retriever(index_name="github-index")

# -------------------- HEADER -------------------- #
st.markdown("""
<div class="main-header">
    <h1>⚡ GitHub Portfolio Intelligence</h1>
    <p><span class="status-indicator"></span>AI-Powered Project Discovery & Analysis</p>
</div>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR -------------------- #
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    top_k = st.slider(
        "Projects to Retrieve",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of relevant projects to search"
    )

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

# -------------------- HELPER: RENDER PROJECT JSON -------------------- #
def render_projects(projects: list):
    if not projects:
        st.warning("⚠️ No relevant projects found.")
        return
    # Normalize into DataFrame
    df = pd.DataFrame(projects)
    # Ensure proper formatting
    if "technologies" in df.columns:
        df["technologies"] = df["technologies"].apply(lambda t: ", ".join(t) if isinstance(t, list) else t)
    # Clickable URLs
    if "url" in df.columns:
        df["url"] = df["url"].apply(lambda x: f"[🔗 Link]({x})")
    #st.dataframe(df, width="stretch")
    st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)

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
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                render_projects(msg["content"])
            else:
                st.markdown(msg["content"])

# -------------------- INPUT -------------------- #
if prompt := st.chat_input("💬 Ask about projects, technologies, or implementations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            with st.spinner("🔍 Analyzing project portfolio..."):
                projects = st.session_state.retriever.retrieve_from_pinecone(prompt, top_k=top_k)
                render_projects(projects)
                st.session_state.messages.append({"role": "assistant", "content": projects})
        except Exception as e:
            error_msg = f"❌ **Error:** Unable to process request\n\n`{str(e)}`"
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
