import streamlit as st
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
import operator

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# ----------------------------
# Load env
# ----------------------------
load_dotenv()

# ----------------------------
# Import retriever
# ----------------------------
from src.retriever import github_project_retriever


# ----------------------------
# State definition
# ----------------------------
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# ----------------------------
# Setup LLM + Tool
# ----------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")

@tool
def github_tool(query: str) -> str:
    """Searches the GitHub project index for relevant information."""
    return github_project_retriever(query)

tools = [github_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


# ----------------------------
# Nodes
# ----------------------------
def llm_call(state: dict):
    """LLM node: handles both chat and retrieval, returns structured output."""

    system_prompt = """
    You are a helpful assistant answering questions about my GitHub projects.

    - If the user is asking about projects, call the `github_tool`.
    - Always return results in JSON format with the following fields:
        - name
        - url
        - technologies
        - description
    - If no projects match, return an empty list.
    - If it's normal chit-chat, just respond naturally.
    """

    # Ask LLM to respond or call tool
    response = llm_with_tools.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    # If tool was called, run it directly and return tool output
    if response.tool_calls:
        tool_outputs = []
        for tool_call in response.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            tool_outputs.append(observation)

        return {
            "messages": [
                response,
                ToolMessage(content=str(tool_outputs), tool_call_id=tool_call["id"])
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # Otherwise return the chat response
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ----------------------------
# Build Agent with LangGraph
# ----------------------------
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_edge("llm_call", END)
agent = agent_builder.compile()