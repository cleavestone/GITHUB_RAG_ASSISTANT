import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from utils.custom_logger import get_logger

logger = get_logger("build_index")


class PineconeIndexer:
    def __init__(self, index_name: str, model_name: str = "all-MiniLM-L6-v2", dim: int = 384, region: str = "us-east-1"):
        logger.info(f"Initializing PineconeIndexer for index '{index_name}' in region '{region}'")
        
        self.api_key = self._load_api_key()
        self.index_name = index_name
        self.model_name = model_name
        self.dim = dim
        self.region = region

        # Initialize Pinecone + model
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self._init_index()
        self.model = SentenceTransformer(model_name)

        logger.info(f"Model '{model_name}' loaded successfully")

    def _load_api_key(self):
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            logger.error("PINECONE_API_KEY not found in environment variables")
            raise ValueError("Missing Pinecone API key. Did you set it in .env?")
        logger.debug("Pinecone API key loaded successfully")
        return api_key

    def _init_index(self):
        logger.info(f"Checking if index '{self.index_name}' exists...")
        if self.index_name not in self.pc.list_indexes().names():
            logger.warning(f"Index '{self.index_name}' not found. Creating new index...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.region),
            )
            logger.info(f"Index '{self.index_name}' created successfully")
        else:
            logger.info(f"Index '{self.index_name}' already exists")
        return self.pc.Index(self.index_name)

    @staticmethod
    def load_docs(filepath: str):
        """Load JSONL documents into a list of dicts."""
        logger.info(f"Loading documents from {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                docs = [json.loads(line) for line in f]
            logger.info(f"Loaded {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Failed to load documents from {filepath}: {e}")
            raise

    @staticmethod
    def prepare_text_for_embedding(doc):
        """Convert project dict into a formatted text string for embeddings."""
        parts = []
        parts.append(f"Project: {doc.get('repo_name', 'Unknown')}")
        parts.append(f"Description: {doc.get('description', 'No description available')}")
        
        if doc.get('key_skills'):
            parts.append(f"Key Skills: {', '.join(doc['key_skills'])}")
        if doc.get('tech_stack'):
            parts.append(f"Technologies: {', '.join(doc['tech_stack'])}")
        if doc.get('use_cases'):
            parts.append(f"Use Cases: {'; '.join(doc['use_cases'])}")
        
        parts.append(f"Complexity: {doc.get('complexity_level', 'Unknown')}")
        if doc.get('tags'):
            parts.append(f"Tags: {', '.join(doc['tags'])}")
        
        return '\n'.join(parts)

    def create_embeddings(self, docs: list) -> list:
        """Generate embeddings and prepare Pinecone vectors."""
        logger.info(f"Creating embeddings for {len(docs)} documents using model '{self.model_name}'")
        vectors = []
        for i, doc in enumerate(docs):
            try:
                text = self.prepare_text_for_embedding(doc)
                emb = self.model.encode(text).tolist()
                vectors.append({
                    "id": doc.get("repo_name", f"doc-{i}"),
                    "values": emb,
                    "metadata": {
                        "repo_name": doc.get("repo_name", "unknown"),
                        "repo_url": doc.get("repo_url", "unknown"),
                        "description": doc.get("description", ""),
                        "tech_stack": doc.get("tech_stack", []),
                        "skills": doc.get("key_skills", []),
                        "use_cases": doc.get("use_cases", []),
                        "complexity": doc.get("complexity_level", "")
                    }
                })
                if (i + 1) % 10 == 0:
                    logger.debug(f"Processed {i+1} documents...")
            except Exception as e:
                logger.error(f"Failed to create embedding for doc {i}: {e}")
        logger.info(f"Successfully created {len(vectors)} embeddings")
        return vectors

    def upsert(self, vectors: list, batch_size: int = 100):
        """Upload vectors in batches."""
        logger.info(f"Upserting {len(vectors)} vectors into index '{self.index_name}' (batch size={batch_size})")
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i: i + batch_size]
            self.index.upsert(vectors=batch)
            logger.debug(f"Upserted batch {i // batch_size + 1} with {len(batch)} vectors")
        logger.info(f"✅ Uploaded all {len(vectors)} vectors to Pinecone index '{self.index_name}'")


if __name__ == "__main__":
    indexer = PineconeIndexer(index_name="github-index")
    docs = indexer.load_docs("data/github/projects.jsonl")
    vectors = indexer.create_embeddings(docs)
    indexer.upsert(vectors)
