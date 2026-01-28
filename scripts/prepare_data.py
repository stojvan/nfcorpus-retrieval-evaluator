#!/usr/bin/env python3
"""
Download and prepare NFCorpus dataset from BEIR/HuggingFace.
This script downloads the corpus, queries, and qrels (relevance judgments).
"""

import json
from pathlib import Path
from datasets import load_dataset


def prepare_nfcorpus_data(output_dir: Path = Path("data")):
    """Download and save NFCorpus dataset components."""
    
    output_dir.mkdir(exist_ok=True)
    
    print("Downloading NFCorpus corpus...")
    corpus_dataset = load_dataset('BeIR/nfcorpus', 'corpus', split='corpus')
    
    corpus_file = output_dir / "corpus.jsonl"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for item in corpus_dataset:
            doc = {
                '_id': item['_id'],
                'title': item['title'],
                'text': item['text']
            }
            f.write(json.dumps(doc) + '\n')
    print(f"✓ Saved {len(corpus_dataset)} documents to {corpus_file}")
    
    print("\nDownloading NFCorpus queries...")
    queries_dataset = load_dataset('BeIR/nfcorpus', 'queries', split='queries')
    
    queries_file = output_dir / "queries.jsonl"
    with open(queries_file, 'w', encoding='utf-8') as f:
        for item in queries_dataset:
            query = {
                '_id': item['_id'],
                'text': item['text']
            }
            f.write(json.dumps(query) + '\n')
    print(f"✓ Saved {len(queries_dataset)} queries to {queries_file}")
    
    print("\nDownloading NFCorpus qrels (relevance judgments)...")
    qrels_dataset = load_dataset('BeIR/nfcorpus-qrels', split='test')
    
    qrels_file = output_dir / "qrels.jsonl"
    with open(qrels_file, 'w', encoding='utf-8') as f:
        for item in qrels_dataset:
            qrel = {
                'query-id': item['query-id'],
                'corpus-id': item['corpus-id'],
                'score': item['score']
            }
            f.write(json.dumps(qrel) + '\n')
    print(f"✓ Saved {len(qrels_dataset)} qrels to {qrels_file}")
    
    print("\nFiltering test queries with qrels...")
    qrels_query_ids = set()
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            qrel = json.loads(line)
            qrels_query_ids.add(qrel['query-id'])
    
    test_queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line)
            if query['_id'] in qrels_query_ids:
                test_queries.append(query)
    
    test_queries_file = output_dir / "test_queries.jsonl"
    with open(test_queries_file, 'w', encoding='utf-8') as f:
        for query in test_queries:
            f.write(json.dumps(query) + '\n')
    print(f"✓ Saved {len(test_queries)} test queries to {test_queries_file}")
    
    test_qrels = []
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            qrel = json.loads(line)
            if qrel['query-id'] in qrels_query_ids:
                test_qrels.append(qrel)
    
    test_qrels_file = output_dir / "test_qrels.jsonl"
    with open(test_qrels_file, 'w', encoding='utf-8') as f:
        for qrel in test_qrels:
            f.write(json.dumps(qrel) + '\n')
    print(f"✓ Saved {len(test_qrels)} test qrels to {test_qrels_file}")
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"Documents: {len(corpus_dataset)}")
    print(f"Total queries: {len(queries_dataset)}")
    print(f"Test queries: {len(test_queries)}")
    print(f"Relevance judgments: {len(test_qrels)}")
    print("="*60)


if __name__ == "__main__":
    prepare_nfcorpus_data()
