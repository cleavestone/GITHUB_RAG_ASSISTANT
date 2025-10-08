from src.fetch_github import GitHubPipeline
from src.build_index import main
from src.build_index import PineconeIndexer


def run_ingestion_pipeline():
    # fetch and clean data
    pipe = GitHubPipeline("config.yaml")
    pipe.ingest_readmes()
    pipe.generate_embedding_summaries()
    pipe.combine_jsons_to_jsonl()
    # build index
    indexer = PineconeIndexer(index_name="github-index")
    docs = indexer.load_docs("data/github/projects.jsonl")
    vectors = indexer.create_embeddings(docs)
    indexer.upsert(vectors)


if __name__ == "__main__":
    run_ingestion_pipeline()