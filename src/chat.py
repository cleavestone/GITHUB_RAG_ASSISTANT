# src/chat.py
"""
chat.py

LangGraph-like agent (but lightweight) with memory:
- Can chat normally or retrieve GitHub projects.
- Keeps conversation history (appends messages instead of overwriting).
- Streams output or returns synchronous responses for Streamlit.
Prompts are imported from src.prompts for clean separation.
"""

import os
import json
import time
from typing import TypedDict, Literal, List, Dict, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from src.retriever import github_project_retriever
from utils.custom_logger import get_logger
from src.prompts import (
    ROUTER_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    RESPONDER_SYSTEM_PROMPT,
    RESPONDER_PROMPT,
    CHAT_SYSTEM_PROMPT,
    STREAMING_RETRIEVER_PROMPT,
    STREAMING_CHAT_PROMPT,
)

load_dotenv()
logger = get_logger("chat")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", ".chat_memory")


# ----------------------------
# Types
# ----------------------------
class AgentState(TypedDict):
    messages: List[BaseMessage]       # conversation history as BaseMessage objects
    next_action: str                  # "chat" or "retrieve"
    retrieved_context: List[Dict]     # retrieved project list
    final_response: str               # assistant final string


# ----------------------------
# Helpers — Memory (file-backed)
# ----------------------------
def _ensure_memory_dir():
    os.makedirs(MEMORY_DIR, exist_ok=True)


def _memory_path(thread_id: str) -> str:
    _ensure_memory_dir()
    safe = thread_id.replace("/", "_")
    return os.path.join(MEMORY_DIR, f"{safe}.json")


def load_history(thread_id: str = "default") -> List[Dict]:
    """Load persisted messages for a thread. Returns list of dicts {role, content}."""
    p = _memory_path(thread_id)
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load history")
        return []


def save_history(thread_id: str, history: List[Dict]):
    """Persist messages list of dicts to disk."""
    p = _memory_path(thread_id)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to save history")


def to_base_messages(serialized: List[Dict]) -> List[BaseMessage]:
    """Convert stored dicts to langchain BaseMessage objects (HumanMessage/AIMessage/SystemMessage)."""
    out: List[BaseMessage] = []
    for m in serialized:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role in ("assistant", "ai", "bot"):
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def from_base_messages(messages: List[BaseMessage]) -> List[Dict]:
    """Convert BaseMessage objects to simple dicts for persistence."""
    out = []
    for m in messages:
        typ = "user"
        if isinstance(m, SystemMessage):
            typ = "system"
        elif isinstance(m, AIMessage):
            typ = "assistant"
        else:
            typ = "user"
        out.append({"role": typ, "content": getattr(m, "content", str(m))})
    return out


# ----------------------------
# LLM Setup
# ----------------------------
def get_llm(temperature: float = 0.7, streaming: bool = False):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment")
    return ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=temperature, streaming=streaming)


# ----------------------------
# Utility: append message into state (keeps list of BaseMessage objects)
# ----------------------------
def append_message(state: AgentState, message: BaseMessage) -> AgentState:
    if "messages" not in state or not isinstance(state["messages"], list):
        state["messages"] = []
    state["messages"].append(message)
    return state


# ----------------------------
# Router Node
# ----------------------------
def router_node(state: AgentState) -> AgentState:
    """Use an LLM to decide whether to retrieve or chat."""
    last = state["messages"][-1]
    user_query = getattr(last, "content", str(last))
    logger.info("Routing: %s", user_query)

    routing_prompt = ROUTER_PROMPT.format(query=user_query)
    messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=routing_prompt)]
    llm = get_llm(temperature=0.1)
    res = llm.invoke(messages)
    decision = (res.content or "").strip().upper()
    logger.info("Router decision: %s", decision)
    state["next_action"] = "retrieve" if "RETRIEVE" in decision else "chat"
    return state


# ----------------------------
# Retriever Node
# ----------------------------
def retriever_node(state: AgentState) -> AgentState:
    """Call the retriever to get GitHub projects (structured JSON)."""
    last = state["messages"][-1]
    user_query = getattr(last, "content", str(last))
    logger.info("Retrieving projects for: %s", user_query)
    results = github_project_retriever(query=user_query, top_k=5, min_score=0.0) or []
    state["retrieved_context"] = results
    logger.info("Retrieved %d projects", len(results))
    return state


# ----------------------------
# Response Generator Node (Fixed)
# ----------------------------
def response_generator_node(state: AgentState) -> AgentState:
    """Generate final response with strict JSON grounding to avoid hallucination."""

    last_message = state["messages"][-1]
    user_query = getattr(last_message, "content", str(last_message))

    if state["next_action"] == "retrieve":
        retrieved_projects = state.get("retrieved_context", [])

        # If nothing retrieved → return a safe fallback
        if not retrieved_projects:
            fallback = (
                "❌ I couldn’t find any projects related to your query.\n\n"
                "👉 Try rephrasing with specific technologies or keywords."
            )
            state["final_response"] = fallback
            return append_message(state, AIMessage(content=fallback))

        # Strict enforcement: Only respond based on retrieved JSON
        llm = get_llm(temperature=0.2)
        messages = [
            SystemMessage(content=RESPONDER_SYSTEM_PROMPT),
            HumanMessage(content=f"User query: {user_query}\n\nProjects JSON:\n{json.dumps(retrieved_projects, indent=2)}")
        ]

        response = llm.invoke(messages)
        state["final_response"] = response.content
        return append_message(state, AIMessage(content=response.content))

    else:  # Normal chat flow
        llm = get_llm(temperature=0.7)
        messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT), *state["messages"]]
        response = llm.invoke(messages)

        state["final_response"] = response.content
        return append_message(state, AIMessage(content=response.content))

    # Chat path
    system_prompt = CHAT_SYSTEM_PROMPT
    messages = [SystemMessage(content=system_prompt)] + state["messages"] + [HumanMessage(content=user_query)]
    llm = get_llm(temperature=0.7)
    res = llm.invoke(messages)
    out = (res.content or "").strip()
    state["final_response"] = out
    append_message(state, AIMessage(content=out))
    return state


# ----------------------------
# Public: Synchronous stream (simulated) used by UI when realtime streaming not needed
# ----------------------------
def stream_agent_response(user_query: str, thread_id: str = "default"):
    """
    Simulated streaming: runs router/retriever/response synchronously
    then yields metadata, then streams characters to simulate token streaming.
    Persists history to disk at the end.
    """
    # load history and convert to BaseMessage objects
    serialized = load_history(thread_id)
    history_msgs = to_base_messages(serialized)
    state: AgentState = {"messages": history_msgs, "next_action": "", "retrieved_context": [], "final_response": ""}

    append_message(state, HumanMessage(content=user_query))
    state = router_node(state)

    if state["next_action"] == "retrieve":
        state = retriever_node(state)

    state = response_generator_node(state)

    # Persist messages
    save_history(thread_id, from_base_messages(state["messages"]))

    # Metadata
    yield {"type": "metadata", "action": state["next_action"], "has_context": bool(state.get("retrieved_context"))}

    # Stream characters so front-end can progressively render
    for ch in state["final_response"]:
        yield {"type": "content", "content": ch}
        # tiny sleep optional to make UI spinner smoother
        time.sleep(0.005)

    yield {"type": "done", "full_response": state["final_response"], "messages": from_base_messages(state["messages"])}


# ----------------------------
# Public: Real-time LLM streaming (if ChatGroq supports .stream)
# ----------------------------
def stream_agent_response_realtime(user_query: str, thread_id: str = "default"):
    """
    Attempts real LLM token streaming.
    - Loads memory, appends user message, performs routing & retrieval.
    - Sends system+history+user messages to the LLM with streaming=True.
    - As chunks arrive yield them to caller; when finished, append assistant message to memory.
    """
    serialized = load_history(thread_id)
    history_msgs = to_base_messages(serialized)
    state: AgentState = {"messages": history_msgs, "next_action": "", "retrieved_context": [], "final_response": ""}

    append_message(state, HumanMessage(content=user_query))
    state = router_node(state)

    # If retrieve: get retrieved_context and include it in system prompt
    if state["next_action"] == "retrieve":
        state = retriever_node(state)
        yield {"type": "metadata", "action": "retrieve", "has_context": True}
        system_prompt = STREAMING_RETRIEVER_PROMPT.format(context=json.dumps(state["retrieved_context"], indent=2))
    else:
        yield {"type": "metadata", "action": "chat", "has_context": False}
        system_prompt = STREAMING_CHAT_PROMPT

    # Build messages: system prompt + history + current user
    messages = [SystemMessage(content=system_prompt)] + state["messages"] + [HumanMessage(content=user_query)]

    llm = get_llm(temperature=0.7, streaming=True)

    # Some ChatGroq streaming yields chunks with .content; we accumulate
    accumulated = ""
    try:
        for chunk in llm.stream(messages):
            # chunk may be dict-like or have .content
            content = getattr(chunk, "content", None) or (chunk.get("content") if isinstance(chunk, dict) else None)
            if not content:
                continue
            accumulated += content
            yield {"type": "content", "content": content}
    except Exception as e:
        logger.exception("Streaming LLM failed, falling back to synchronous invoke: %s", e)
        # fallback: synchronous invoke
        res = get_llm(temperature=0.7).invoke(messages)
        accumulated = res.content or ""
        for ch in accumulated:
            yield {"type": "content", "content": ch}

    # persist assistant message to memory
    state["final_response"] = accumulated
    append_message(state, AIMessage(content=accumulated))
    save_history(thread_id, from_base_messages(state["messages"]))

    yield {"type": "done", "full_response": accumulated, "messages": from_base_messages(state["messages"])}


# ----------------------------
# Simple synchronous chat helper (returns text and updated history)
# ----------------------------
def chat(user_query: str, thread_id: str = "default") -> (str, List[Dict]):
    """
    Synchronous convenience wrapper: runs the pipeline and returns final response and updated serialized history.
    """
    serialized = load_history(thread_id)
    history_msgs = to_base_messages(serialized)
    state: AgentState = {"messages": history_msgs, "next_action": "", "retrieved_context": [], "final_response": ""}
    append_message(state, HumanMessage(content=user_query))
    state = router_node(state)
    if state["next_action"] == "retrieve":
        state = retriever_node(state)
    state = response_generator_node(state)
    save_history(thread_id, from_base_messages(state["messages"]))
    return state["final_response"], from_base_messages(state["messages"])


# ----------------------------
# If run directly, quick smoke test
# ----------------------------
if __name__ == "__main__":
    print("=== Quick smoke test ===")
    r, hist = chat("Hi, my name is Cleave", thread_id="demo")
    print("Bot:", r)
    r, hist = chat("What is my name?", thread_id="demo")
    print("Bot:", r)
    r, hist = chat("Show me SQL projects", thread_id="demo")
    print("Bot:", r)
