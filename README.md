# GitHub RAG Portfolio Assistant

An **intelligent portfolio assistant powered by Retrieval-Augmented Generation (RAG)**, designed to help you track and explore your GitHub projects effortlessly.  
⚠️ **Note:** This project is still a work in progress.

---

## 🔹 The Idea
When you’ve built many projects, it can be surprisingly hard to keep track of them all.  
This project aims to create a **smart assistant** that can search, summarize, and retrieve your repositories intelligently.  
⚠️ *Still under active development.*

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
- Used an **open-source LLM** for natural language generation and summaries  

⚠️ *Functionality is ongoing; features may evolve.*

---

## 🔹 Data Ingestion Pipeline
A **modular ingestion pipeline** that currently:
1. Fetches repos from GitHub  
2. Cleans & formats the data  
3. Summarizes metadata  
4. Embeds projects into a **vector store**  

This pipeline **can be scheduled to auto-update**, keeping your portfolio fresh.  
⚠️ *Pipeline enhancements are still in progress.*

---

## 🔹 Retrieval Pipeline
Supports **natural language queries** such as:
- “Show me my clustering projects”  
- “Which repos used KMeans?”  
- “Retrieve my machine learning projects”  

The system:
1. Retrieves the most relevant repositories (description, stack, link)  
2. Passes them to an LLM for **final output and summaries**  

⚠️ *Retrieval quality and UI are still being refined.*

---

## 🔹 Enhancements So Far
- End-to-end **ingestion → retrieval** flow  
- **Cross-encoder reranking** for improved accuracy  
- **Query rewriting** to handle ambiguous or complex searches  

⚠️ *Additional improvements planned.*

---

## 💡 Why This Matters
As your repo count grows, finding the right project can be overwhelming.  
This system makes it easy to **search, retrieve, and showcase projects instantly**—almost like having a personal smart assistant for your GitHub portfolio.  

⚠️ *Note: The project is still evolving; expect updates and improvements.*

---

## ⚡ Next Steps
- Refine retrieval quality  
- Improve the **UI**  
- Finalize deployment with **Docker, CI/CD, and AWS**  

---

## 🔗 Try It Yourself
Feel free to clone the repo and experiment with the current features:

```bash
git clone https://github.com/yourusername/github-rag-portfolio.git

