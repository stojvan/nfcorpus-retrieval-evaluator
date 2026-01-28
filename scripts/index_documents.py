#!/usr/bin/env python3
"""
Index NFCorpus documents into Qdrant vector database.
Generates embeddings using sentence-transformers/all-MiniLM-L6-v2.
"""

import json
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Batch
from tqdm import tqdm


def load_corpus(corpus_file: Path) -> List[Dict]:
    """Load corpus documents from JSONL file."""
    documents = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def index_documents(
    corpus_file: Path = Path("data/corpus.jsonl"),
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "nfcorpus",
    batch_size: int = 100
):
    """Index documents into Qdrant with embeddings."""
    
    print("Loading corpus documents...")
    documents = load_corpus(corpus_file)
    print(f"✓ Loaded {len(documents)} documents")
    
    print("\nInitializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f"✓ Model loaded (embedding dimension: {model.get_sentence_embedding_dimension()})")
    
    print(f"\nConnecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url)
    
    print(f"\nCreating collection '{collection_name}'...")
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"  (Deleted existing collection)")
    except Exception:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=Distance.COSINE
        )
    )
    print(f"✓ Collection created")
    
    print(f"\nGenerating embeddings and indexing documents (batch size: {batch_size})...")
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Indexing batches"):
        batch_docs = documents[i:i + batch_size]
        
        texts = [f"{doc['title']} {doc['text']}" for doc in batch_docs]
        embeddings = model.encode(texts, show_progress_bar=False)
        
        points = []
        for idx, doc in enumerate(batch_docs):
            point = PointStruct(
                id=hash(doc['_id']) & 0x7FFFFFFFFFFFFFFF,  # Convert to positive int
                vector=embeddings[idx].tolist(),
                payload={
                    'doc_id': doc['_id'],
                    'title': doc['title'],
                    'text': doc['text']
                }
            )
            points.append(point)
        
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    collection_info = client.get_collection(collection_name=collection_name)
    
    print("\n" + "="*60)
    print("Indexing complete!")
    print("="*60)
    print(f"Collection: {collection_name}")
    print(f"Documents indexed: {collection_info.points_count}")
    print(f"Vector dimension: {model.get_sentence_embedding_dimension()}")
    print(f"Distance metric: Cosine")
    print("="*60)
    
    print("\nTesting search functionality...")
    test_query = "What are the effects of calcium on bone health?"
    query_embedding = model.encode(test_query)
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=5
    )
    
    print(f"\nTest query: '{test_query}'")
    print(f"Top 5 results:")
    for idx, result in enumerate(results.points, 1):
        print(f"  {idx}. Doc ID: {result.payload['doc_id']}")
        print(f"     Score: {result.score:.4f}")
        print(f"     Title: {result.payload['title'][:80]}...")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index NFCorpus documents into Qdrant")
    parser.add_argument("--corpus-file", type=Path, default=Path("data/corpus.jsonl"),
                        help="Path to corpus JSONL file")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333",
                        help="Qdrant server URL")
    parser.add_argument("--collection-name", type=str, default="nfcorpus",
                        help="Collection name in Qdrant")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for indexing")
    
    args = parser.parse_args()
    
    index_documents(
        corpus_file=args.corpus_file,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        batch_size=args.batch_size
    )
