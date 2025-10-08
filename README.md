# GitHub RAG Portfolio Assistant

## 🎥 Demo

<video src="static/demo.mp4" controls width="600"></video>


An **intelligent portfolio assistant powered by Retrieval-Augmented Generation (RAG)**, designed to help you track and explore your GitHub projects effortlessly.  

⚠️ **Note:** Some parts of the project are still under active development, including the user interface and agentic features with LangGraph.

---

## 🔹 The Idea
When you’ve built many projects, it can be surprisingly hard to keep track of them all.  
This project creates a **smart assistant** that can search, summarize, and retrieve your repositories intelligently.  

---

## 🔹 What I’ve Built So Far
- Pulled repo **READMEs** using the GitHub API  
- Cleaned & summarized each project into:
  - **Name**
  - **Description**
  - **Tech Stack**
  - **Link**  
- Stored all project data in **structured JSONL format**  
- Embedded projects into **Pinecone** using open-source embedding models  
- Used an **open-source LLM** for generation and summaries  
- Incorporated **LangGraph** to support **agentic capabilities** (in progress)  

---

## 🔹 Data Ingestion Pipeline
The **ingestion pipeline is complete** and fully functional. It:
1. Fetches repos from GitHub  
2. Cleans & formats the data  
3. Summarizes metadata  
4. Embeds projects into a **vector store**  

This pipeline **can be scheduled to auto-update**, keeping your portfolio fresh.

---

## 🔹 Retrieval Pipeline
The **retrieval pipeline is already solid**, supporting natural language queries such as:
- “Show me my clustering projects”  
- “Which repos used KMeans?”  
- “Retrieve my machine learning projects”  

Enhancements implemented so far:
- **Cross-encoder reranking** for higher accuracy  
- **Query rewriting** to handle ambiguous or complex queries  

⚡ These improvements have already **brought impressive results**, and further refinements are minor.

---

## 🔹 User Interface
The **UI is still in progress**, with future plans to make interaction with the assistant intuitive and visually appealing.

---

## 💡 Why This Matters
As your repo count grows, finding the right project can be overwhelming.  
This system makes it easy to **search, retrieve, and showcase projects instantly**—almost like having a personal smart assistant for your GitHub portfolio.

---

## ⚡ Next Steps
- Complete the **user interface**  
- Refine **retrieval quality** further (minor improvements only)  
- Finalize **agentic behavior** with LangGraph  
- Deploy with **Docker, CI/CD, and AWS**  

---

## 🔗 Try It Yourself
Clone the repo and experiment with the current features:

```bash
git clone https://github.com/yourusername/github-rag-portfolio.git


