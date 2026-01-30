import os
import sys
from pathlib import Path
from data_loader import load_nfcorpus_dataset, embed_corpus, setup_qdrant


def main():
    """Prepare NFCorpus data: download, embed, and setup Qdrant."""
    print("Loading NFCorpus dataset...")
    corpus, queries, qrels = load_nfcorpus_dataset(cache_dir="./data_cache")
    print(f"Loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    print("Embedding corpus documents with OpenAI...")
    embeddings = embed_corpus(corpus, openai_api_key, batch_size=100)
    print(f"Generated {len(embeddings)} embeddings")
    
    print("Setting up Qdrant vector database...")
    client = setup_qdrant(corpus, embeddings, qdrant_path="./qdrant_data", collection_name="nfcorpus")
    print("Qdrant setup complete!")
    
    collection_info = client.get_collection("nfcorpus")
    print(f"Collection 'nfcorpus' has {collection_info.points_count} points")


if __name__ == "__main__":
    main()
