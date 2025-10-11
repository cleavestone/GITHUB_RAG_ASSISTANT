"""
src/retriever.py

Retriever (Fusion + tiny Cross-Encoder reranker) that returns JSON-serializable results.
IMPROVED: Better handling of conversational queries.
"""

import os
import json
import re
import requests
import logging
from functools import lru_cache
from typing import List, Dict, Optional
from collections import defaultdict
from src.prompts import RAG_FUSION_PROMPT

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from dotenv import load_dotenv
from utils.llm_client import LLMClient

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
# Query Preprocessor
# ----------------------------
def extract_core_terms(query: str) -> str:
    """
    Extract core search terms from conversational queries.
    Removes filler words while preserving technical terms.
    """
    query_lower = query.lower().strip()
    
    # Remove common conversational patterns
    patterns = [
        r'^(show\s+me|find|search\s+for|get\s+me|give\s+me|list|display)\s+',
        r'^(do\s+you\s+have|are\s+there|can\s+you\s+show|what\s+about|tell\s+me\s+about)\s+',
        r'^(i\s+want|i\s+need|i\'m\s+looking\s+for)\s+',
        r'\s+(please|thanks?|thank\s+you)$',
        r'^(any|some|the|your)\s+',
        r'\s+(projects?|repositories?|repos?|examples?|samples?)$',
    ]
    
    cleaned = query_lower
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up spaces
    cleaned = ' '.join(cleaned.split())
    
    # If too short after cleaning, keep more of original
    if len(cleaned) < 3:
        # Just remove the most basic patterns
        basic_patterns = [
            r'^(show\s+me|find|do\s+you\s+have)\s+',
            r'\s+(please|thanks?)$',
        ]
        cleaned = query_lower
        for pattern in basic_patterns:
            cleaned = re.sub(pattern, '', cleaned)
        cleaned = ' '.join(cleaned.split())
    
    # Final check - if still too short, use original
    if len(cleaned) < 2:
        cleaned = query
    
    logger.info(f"Query extraction: '{query}' -> '{cleaned}'")
    return cleaned.strip()


# ----------------------------
# Query Generator for RAG Fusion
# ----------------------------
class QueryGenerator:
    """Generate multiple query variations for RAG Fusion."""

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key

    def generate_fusion_queries(self, original_query: str, num_queries: int = NUM_FUSION_QUERIES) -> List[str]:
        core_query = extract_core_terms(original_query)

        # Build prompt dynamically
        prompt = RAG_FUSION_PROMPT.format(num_queries=num_queries, core_query=core_query)

        try:
            # ✅ Centralized call
            content = self.client.chat(
                system_prompt="You are a technical search assistant. Generate SHORT keyword variations.",
                user_prompt=prompt
            )

            # Parse queries
            queries = []
            for line in content.split("\n"):
                clean_line = re.sub(r"^\d+[\.\)]\s*", "", line)
                clean_line = re.sub(r"^[-•]\s*", "", clean_line)
                clean_line = clean_line.strip().strip('"\'')
                if clean_line and len(clean_line) > 2:
                    queries.append(clean_line)

            all_queries = [core_query] + queries[: num_queries - 1]
            logger.info(f"✓ Generated fusion queries: {all_queries}")
            return all_queries

        except Exception as e:
            logger.warning(f"⚠ Query generation failed: {e}. Using core query only.")
            return [core_query]


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

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        if not docs:
            return []
        
        # Use core terms for reranking
        core_query = extract_core_terms(query)
        
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
            pairs.append((core_query, text if text else meta.get("repo_name", "")))

        try:
            scores = self.reranker.predict(pairs)
        except Exception as e:
            logger.warning(f"Reranker failed: {e}. Falling back to original scores.")
            for d in docs:
                d["rerank_score"] = float(d.get("original_score", d.get("score", 0)) or 0.0)
            reranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]

        # attach rerank scores
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)

        # filter out negatives (or below threshold)
        filtered = [d for d in docs if d["rerank_score"] >= min_score]

        # sort and slice
        reranked = sorted(filtered, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def retrieve_from_pinecone(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[Dict]:
        # Extract core terms first
        core_query = extract_core_terms(query)
        
        if not self.use_fusion:
            matches = self.retrieve_single_query(core_query, top_k=top_k)
            return self._format_matches(matches[:top_k])
        
        fusion_queries = self.query_generator.generate_fusion_queries(core_query, NUM_FUSION_QUERIES)
        all_results = []
        for fq in fusion_queries:
            results = self.retrieve_single_query(fq, top_k=top_k * 3)
            if results:
                all_results.append(results)
        
        if not all_results:
            logger.warning("No results from any fusion query!")
            return []
        
        fused_results = self.reciprocal_rank_fusion(all_results)
        reranked = self.rerank(core_query, fused_results, top_k=top_k, min_score=min_score)
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
# Public API (context-only for LangGraph)
# ----------------------------
def github_project_retriever(query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
    """
    GitHub project retriever for LangGraph.
    Returns structured JSON (list of projects) instead of free text.
    """
    retriever = get_retriever(index_name="github-index", use_fusion=True)
    results = retriever.retrieve_from_pinecone(query, top_k=top_k, min_score=min_score)

    if not results:
        return []

    formatted = []
    for idx, r in enumerate(results, start=1):
        formatted.append({
            "name": r["name"],
            "url": r["url"],
            "technologies": r["technologies"],
            "description": r["description"]
        })
    return formatted
# ----------------------------
# Example usage (quick test)
# ----------------------------
if __name__ == "__main__":
    test_queries = [
        "SQL projects",
        "show me SQL projects",
        "Do you have any SQL projects",
        "Python data analysis",
        "find Power BI dashboards"
    ]
    
    print("=== Testing Conversational Query Handling ===\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 70)
        
        context = github_project_retriever(query, top_k=3, min_score=0.0)
        
        if "No relevant projects found" in context:
            print("❌ No results\n")
        else:
            print(f"✅ Found projects:")
            # Show first project
            lines = context.split('\n')
            for line in lines[:6]:
                print(line)
            print("...\n")