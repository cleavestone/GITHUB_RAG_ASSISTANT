


# 🚀 GitHub RAG Portfolio Assistant

An intelligent **Retrieval-Augmented Generation (RAG)** system for managing and exploring your GitHub portfolio.  
This project combines **vector search, query fusion, reranking, and conversational AI** to make your repositories easily searchable and showcaseable through natural language queries.

---
![Demo GIF](static/demo.gif)

---

## 📝 Introduction

As your GitHub portfolio grows, it becomes harder to **track, organize, and present** your projects effectively.  
This assistant provides a smart interface to:

- 📥 **Fetch & summarize** GitHub repositories  
- 📊 **Embed & index** projects in a vector database  
- 🔍 **Retrieve projects** via conversational queries  
- 💬 Support both **normal chat** and **retrieval-augmented workflows**  

Think of it as your **personal AI assistant for your portfolio**.

---

## ✨ Features

- 🔄 **Conversational query preprocessing** → Improves search relevance by rewriting ambiguous queries  
- 🎯 **RAG Fusion** → Generates multiple query variations for broader retrieval coverage  
- 📚 **Reciprocal Rank Fusion (RRF)** → Combines results from multiple queries for robust ranking  
- ⚡ **Cross-Encoder reranking** → High-precision ranking using `cross-encoder/ms-marco-MiniLM-L-2-v2`  
- 🪶 **Lightweight embeddings** → `all-MiniLM-L6-v2` (384-dim) for a speed/accuracy trade-off  
- 🔀 **Dual chat modes** → Normal LLM chat & retrieval-augmented chat  
- 🤖 **Agentic workflows** → Routing between retrieval & normal chat with **LangGraph**  
- 📦 **JSON-serializable outputs** → Easy integration with other systems  
- 🛠 **Logging & config management** → Debuggable & reproducible workflows  
- 📓 **Example notebooks** → Data ingestion & LangGraph experimentation  

---

## 🏗️ Architecture

### 🔹 Data Ingestion Pipeline
- Fetches GitHub repos via API  
- Cleans and summarizes metadata  
- Embeds & stores in **Pinecone**  
- Orchestrated & schedulable for auto-updates  

### 🔹 Retrieval Pipeline
- Embedding-based search  
- Query rewriting + fusion  
- **Reciprocal Rank Fusion (RRF)**  
- Cross-encoder reranking for precision  

### 🔹 Chat Layer
- Normal **LLM conversations**  
- **Retrieval-augmented responses** when project context is required  
- Workflow routing via **LangGraph**  

### 🔹 Interface Layer
- **Streamlit UI** 
- Modular prompts (`prompts.py`) for maintainability  

---

## 🔎 RAG Details

- **Embedding model**: `all-MiniLM-L6-v2` (384 dimensions)  
  - Optimized for CPU inference  
  - Balance of speed & semantic accuracy  

- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-2-v2`  
  - Lightweight reranker for high-precision ranking  
  - Swappable with larger cross-encoders in GPU environments  

- **Query Fusion**:  
  - Multiple rewritten queries → retrieved separately  
  - Combined using **Reciprocal Rank Fusion (RRF)**  

---

## ⚙️ Tech Stack

- **Core**: Python, GitHub API  
- **RAG Framework**: LangGraph, Pinecone, Sentence Transformers  
- **Models**:  
  - Embedding → `all-MiniLM-L6-v2`  
  - Reranker → `cross-encoder/ms-marco-MiniLM-L-2-v2`  
- **Interface**: Streamlit  
- **Infrastructure**: Docker, CI/CD, AWS (ECR + ECS Fargate)  
- **Other**: Groq, Structured Logging, Config Management  

---

## 📌 Roadmap

- [x] Data ingestion & summarization  
- [x] Retrieval pipeline with query fusion + reranking  
- [x] Dual chat (retrieval & normal)  
- [x] Streamlit UI integration  
- [ ] Full Docker + AWS deployment pipeline  

---

### Project stucture

```
📦 project-root/
├── app/  
│   ├── app.py                # Streamlit/FastAPI entry point  
│   └── __init__.py  
│
├── src/  
│   ├── __init__.py  
│   ├── build_index.py        # Embedding/index pipeline  
│   ├── chat.py               # Chat agent logic with memory  
│   ├── fetch_github.py       # Fetch GitHub data  
│   ├── prompts.py            # Prompt templates  
│   └── retriever.py          # Retrieval logic  
│
├── utils/  
│   ├── __init__.py  
│   ├── config_loader.py  
│   ├── custom_logger.py  
│   └── llm_client.py  
│
├── Pipelines/  
│   └── Data_Ingestion.py     # Data ingestion pipeline  
│
├── data/  
│   ├── github/  
│   │   ├── projects.jsonl  
│   │   ├── github_readmes/   # Project README markdowns (.md files)  
│   │   └── summaries/        # JSON summaries of README files  
│
├── notebooks/                # Keep for experiments (optional in prod)  
│   ├── github_ingestion.ipynb  
│   └── langgraph_chat.ipynb  
│
├── static/  
│   ├── demo.gif  
│   └── css/  
│       └── style.css  
│
├── logs/                     # Keep logs, but rotate in prod  
│   ├── main.log  
│   ├── build_index.log  
│   ├── chat.log  
│   ├── rag.log  
│   └── retriever.log  
│
├── .dockerignore  
├── Dockerfile  
├── requirements.txt  
├── config.yaml  
├── README.md  
└── .gitignore  

```


