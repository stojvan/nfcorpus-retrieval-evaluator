from datasets import load_dataset
from pathlib import Path
import random
import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import Dict, List, Tuple
import json


def load_nfcorpus_dataset(cache_dir: str = "./data_cache"):
    """Load NFCorpus dataset from BEIR.
    
    Args:
        cache_dir: Directory to cache the dataset
        
    Returns:
        Tuple of (corpus, queries, qrels)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    dataset = load_dataset("BeIR/nfcorpus", "corpus", cache_dir=cache_dir, trust_remote_code=True)
    corpus_data = dataset["corpus"]
    
    queries_dataset = load_dataset("BeIR/nfcorpus", "queries", cache_dir=cache_dir, trust_remote_code=True)
    queries_data = queries_dataset["queries"]
    
    qrels_dataset = load_dataset("BeIR/nfcorpus-qrels", cache_dir=cache_dir, trust_remote_code=True)
    qrels_data = qrels_dataset["test"]
    
    corpus = {doc["_id"]: {"title": doc["title"], "text": doc["text"]} for doc in corpus_data}
    queries = {q["_id"]: q["text"] for q in queries_data}
    
    qrels = {}
    for item in qrels_data:
        query_id = item["query-id"]
        corpus_id = item["corpus-id"]
        score = item["score"]
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][corpus_id] = score
    
    return corpus, queries, qrels


def embed_corpus(corpus: Dict[str, Dict[str, str]], openai_api_key: str, batch_size: int = 100):
    """Embed corpus documents using OpenAI API.
    
    Args:
        corpus: Dictionary of document ID to document content
        openai_api_key: OpenAI API key
        batch_size: Number of documents to embed per batch
        
    Returns:
        Dictionary of document ID to embedding vector
    """
    client = OpenAI(api_key=openai_api_key)
    embeddings = {}
    
    doc_ids = list(corpus.keys())
    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i:i + batch_size]
        batch_texts = [f"{corpus[doc_id]['title']} {corpus[doc_id]['text']}" for doc_id in batch_ids]
        
        response = client.embeddings.create(
            input=batch_texts,
            model="text-embedding-3-small"
        )
        
        for j, doc_id in enumerate(batch_ids):
            embeddings[doc_id] = response.data[j].embedding
    
    return embeddings


def setup_qdrant(corpus: Dict[str, Dict[str, str]], embeddings: Dict[str, List[float]], 
                 qdrant_path: str = "./qdrant_data", collection_name: str = "nfcorpus"):
    """Setup Qdrant vector database with embedded corpus.
    
    Args:
        corpus: Dictionary of document ID to document content
        embeddings: Dictionary of document ID to embedding vector
        qdrant_path: Path to Qdrant storage
        collection_name: Name of the collection
        
    Returns:
        QdrantClient instance
    """
    client = QdrantClient(path=qdrant_path)
    
    vector_size = len(next(iter(embeddings.values())))
    
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    
    points = []
    for doc_id, embedding in embeddings.items():
        point = PointStruct(
            id=hash(doc_id) & 0x7FFFFFFFFFFFFFFF,
            vector=embedding,
            payload={
                "doc_id": doc_id,
                "title": corpus[doc_id]["title"],
                "text": corpus[doc_id]["text"]
            }
        )
        points.append(point)
    
    client.upsert(collection_name=collection_name, points=points)
    
    return client


def sample_queries(queries: Dict[str, str], num_queries: int, random_seed: int | None = None, 
                   qrels: Dict[str, Dict[str, int]] | None = None):
    """Sample queries from the query set.
    
    Args:
        queries: Dictionary of query ID to query text
        num_queries: Number of queries to sample
        random_seed: Optional seed for reproducibility
        qrels: Optional qrels dict - if provided, only sample queries that have qrels
        
    Returns:
        List of (query_id, query_text) tuples
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Only sample from queries that have qrels (test queries)
    if qrels is not None:
        query_ids = [qid for qid in queries.keys() if qid in qrels]
    else:
        query_ids = list(queries.keys())
    
    actual_num_queries = min(num_queries, len(query_ids))
    sampled_ids = random.sample(query_ids, actual_num_queries)
    
    return [(qid, queries[qid]) for qid in sampled_ids]
