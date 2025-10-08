"""
src/retriever.py

Retriever (Fusion + tiny Cross-Encoder reranker) that returns JSON-serializable results.
Now includes streaming generator.
"""

import os
import json
import requests
import logging
from functools import lru_cache
from typing import List, Dict, Optional, Generator
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from dotenv import load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGPipeline")

# RAG Fusion constants
NUM_FUSION_QUERIES = 3


# ----------------------------
# Query Generator for RAG Fusion
# ----------------------------
class QueryGenerator:
    """Generate multiple query variations for RAG Fusion."""

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key

    def generate_fusion_queries(self, original_query: str, num_queries: int = NUM_FUSION_QUERIES) -> List[str]:
        prompt = f"""You are a query expansion assistant. Generate {num_queries} different variations of the following query.
Each variation should:
1. Preserve the original intent and meaning
2. Use different phrasings, synonyms, or perspectives
3. Be specific and searchable
4. Focus on different aspects of the query

Original query: {original_query}

Generate exactly {num_queries} query variations, one per line.
"""

        try:
            data = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a query expansion assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.8
            }
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json=data,
                timeout=30
            )
            response.raise_for_status()

            content = response.json()["choices"][0]["message"]["content"].strip()
            queries = [q.strip() for q in content.split("\n") if q.strip()]

            # Always include original as first element
            all_queries = [original_query] + queries[: max(0, num_queries - 1)]

            logger.info(f"✓ Generated {len(all_queries)} fusion queries")
            return all_queries

        except Exception as e:
            logger.warning(f"⚠ Query generation failed: {e}. Using original query only.")
            return [original_query]


# ----------------------------
# Retriever with Fusion + Reranker
# ----------------------------
class Retriever:
    def __init__(
        self,
        index_name: str,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
        namespace: Optional[str] = None,
        use_fusion: bool = True,
    ):
        self.namespace = namespace
        self.use_fusion = use_fusion
        self._initialize_pinecone(index_name)
        self._initialize_embedder(embed_model)
        self._initialize_reranker(rerank_model)
        if self.use_fusion:
            self.query_generator = QueryGenerator(GROQ_API_KEY)

    def _initialize_pinecone(self, index_name: str):
        if not PINECONE_API_KEY:
            raise ValueError("Missing Pinecone API key.")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(index_name)
        logger.info(f"✓ Connected to Pinecone index: {index_name}")

    def _initialize_embedder(self, embed_model: str):
        self.embedder = SentenceTransformer(embed_model)
        logger.info(f"✓ Loaded embedder: {embed_model} on {self.embedder.device}")

    def _initialize_reranker(self, rerank_model: str):
        # cross-encoder is intentionally lightweight; runs on CPU
        self.reranker = CrossEncoder(rerank_model)
        logger.info(f"✓ Loaded reranker: {rerank_model}")

    @lru_cache(maxsize=256)
    def embed_query(self, query: str) -> List[float]:
        return self.embedder.encode(query, convert_to_tensor=False).tolist()

    def retrieve_single_query(self, query: str, top_k: int = 10) -> List[Dict]:
        q_emb = self.embed_query(query)
        results = self.index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )
        matches = []
        for rank, match in enumerate(results.matches):
            matches.append(
                {
                    "id": match.id,
                    "score": match.score,
                    "rank": rank + 1,
                    "metadata": match.metadata or {},
                }
            )
        return matches

    def reciprocal_rank_fusion(self, all_results: List[List[Dict]], k: int = 60) -> List[Dict]:
        doc_scores = defaultdict(float)
        doc_data = {}
        for result_set in all_results:
            for match in result_set:
                doc_id = match["id"]
                rank = match["rank"]
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc_id] += rrf_score
                if doc_id not in doc_data or match["score"] > doc_data[doc_id]["original_score"]:
                    doc_data[doc_id] = {
                        "metadata": match["metadata"],
                        "original_score": match["score"],
                    }
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused = []
        for doc_id, rrf_score in ranked_docs:
            fused.append(
                {
                    "id": doc_id,
                    "fusion_score": rrf_score,
                    "original_score": doc_data[doc_id]["original_score"],
                    "metadata": doc_data[doc_id]["metadata"],
                }
            )
        return fused

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        if not docs:
            return []
        pairs = []
        for d in docs:
            meta = d.get("metadata", {}) or {}
            text = (
                (meta.get("description", "") or "")
                + " "
                + " ".join(meta.get("tech_stack", []) or [])
                + " "
                + " ".join(meta.get("skills", []) or [])
            ).strip()
            pairs.append((query, text if text else meta.get("repo_name", "")))
        try:
            scores = self.reranker.predict(pairs)
        except Exception as e:
            logger.warning(f"Reranker failed: {e}. Falling back to original scores.")
            for d in docs:
                d["rerank_score"] = float(d.get("original_score", d.get("score", 0)) or 0.0)
            return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return docs[:top_k]

    def retrieve_from_pinecone(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.use_fusion:
            matches = self.retrieve_single_query(query, top_k=top_k)
            return self._format_matches(matches[:top_k])
        fusion_queries = self.query_generator.generate_fusion_queries(query, NUM_FUSION_QUERIES)
        all_results = []
        for fq in fusion_queries:
            results = self.retrieve_single_query(fq, top_k=top_k * 3)
            if results:
                all_results.append(results)
        if not all_results:
            return []
        fused_results = self.reciprocal_rank_fusion(all_results)
        reranked = self.rerank(query, fused_results, top_k=top_k)
        return self._format_matches(reranked)

    def _format_matches(self, matches: List[Dict]) -> List[Dict]:
        formatted = []
        for match in matches:
            meta = (match.get("metadata") or {}) if isinstance(match, dict) else {}
            name = meta.get("repo_name") or meta.get("name") or meta.get("repo") or ""
            url = meta.get("repo_url") or meta.get("url") or meta.get("repo_url_https") or ""
            technologies = meta.get("tech_stack") or meta.get("technologies") or meta.get("tags") or []
            if isinstance(technologies, str):
                technologies = [t.strip() for t in technologies.split(",") if t.strip()]
            description = meta.get("description") or meta.get("summary") or ""
            score = match.get("rerank_score") or match.get("fusion_score") or match.get("original_score") or match.get("score")
            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None
            formatted.append(
                {
                    "name": name,
                    "url": url,
                    "technologies": technologies,
                    "description": description,
                    "score": score,
                }
            )
        return formatted


# ----------------------------
# Cached retriever helper
# ----------------------------
_CACHED_RETRIEVER: Optional[Retriever] = None

def get_retriever(index_name: str = "github-index", use_fusion: bool = True) -> Retriever:
    global _CACHED_RETRIEVER
    if _CACHED_RETRIEVER is None:
        _CACHED_RETRIEVER = Retriever(index_name=index_name, use_fusion=use_fusion)
    return _CACHED_RETRIEVER


# ----------------------------
# Public API (batch)
# ----------------------------
def github_project_retriever(query: str, top_k: int = 5) -> List[Dict]:
    retriever = get_retriever(index_name="github-index", use_fusion=True)
    return retriever.retrieve_from_pinecone(query, top_k=top_k)


# ----------------------------
# Public API (streaming generator)
# ----------------------------
def github_project_retriever(query: str, top_k: int = 5) -> List[Dict]:
    """
    GitHub project retriever (non-streaming).
    Returns a list of project dicts with keys:
    - name
    - url
    - technologies
    - description
    - score
    """
    retriever = get_retriever(index_name="github-index", use_fusion=True)

    # Step 1: Raw retrieval
    raw = retriever.retrieve_single_query(query, top_k=top_k * 3)

    # Step 2: Fusion
    fusion_queries = retriever.query_generator.generate_fusion_queries(query, NUM_FUSION_QUERIES)
    all_results = [retriever.retrieve_single_query(fq, top_k=top_k * 3) for fq in fusion_queries]
    fused = retriever.reciprocal_rank_fusion(all_results)

    # Step 3: Reranked final
    reranked = retriever.rerank(query, fused, top_k=top_k)

    return retriever._format_matches(reranked)



# ----------------------------
# Example usage (quick test)
# ----------------------------
if __name__ == "__main__":
    for event in github_project_retriever("SQL projects", top_k=3):
        print("\nStage:", event["stage"])
        print(json.dumps(event["results"], indent=2, ensure_ascii=False))
