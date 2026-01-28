"""
Metrics calculation for information retrieval evaluation.
Implements NDCG@k and related metrics.
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics import ndcg_score


def calculate_ndcg_at_k(
    retrieved_docs: List[str],
    ground_truth: Dict[str, int],
    k: int = 5
) -> float:
    """
    Calculate NDCG@k for a single query.
    
    Args:
        retrieved_docs: List of document IDs in ranked order
        ground_truth: Dict mapping doc_id -> relevance_score (0-2)
        k: Cutoff for NDCG@k calculation
        
    Returns:
        NDCG@k score (0.0 to 1.0)
    """
    if not retrieved_docs or not ground_truth:
        return 0.0
    
    retrieved_docs = retrieved_docs[:k]
    
    y_true = []
    y_score = []
    
    for rank, doc_id in enumerate(retrieved_docs):
        relevance = ground_truth.get(doc_id, 0)
        y_true.append(relevance)
        y_score.append(k - rank)
    
    if sum(y_true) == 0:
        return 0.0
    
    y_true = np.array([y_true])
    y_score = np.array([y_score])
    
    try:
        return float(ndcg_score(y_true, y_score, k=k))
    except Exception:
        return 0.0


def calculate_recall_at_k(
    retrieved_docs: List[str],
    ground_truth: Dict[str, int],
    k: int = 5
) -> float:
    """
    Calculate Recall@k - fraction of relevant docs in top-k.
    
    Args:
        retrieved_docs: List of document IDs in ranked order
        ground_truth: Dict mapping doc_id -> relevance_score (0-2)
        k: Cutoff for Recall@k calculation
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not ground_truth:
        return 0.0
    
    retrieved_docs = retrieved_docs[:k]
    relevant_docs = {doc_id for doc_id, score in ground_truth.items() if score > 0}
    
    if not relevant_docs:
        return 0.0
    
    retrieved_relevant = len(set(retrieved_docs) & relevant_docs)
    return retrieved_relevant / len(relevant_docs)


def calculate_precision_at_k(
    retrieved_docs: List[str],
    ground_truth: Dict[str, int],
    k: int = 5
) -> float:
    """
    Calculate Precision@k - fraction of top-k that are relevant.
    
    Args:
        retrieved_docs: List of document IDs in ranked order
        ground_truth: Dict mapping doc_id -> relevance_score (0-2)
        k: Cutoff for Precision@k calculation
        
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if not retrieved_docs:
        return 0.0
    
    retrieved_docs = retrieved_docs[:k]
    relevant_docs = {doc_id for doc_id, score in ground_truth.items() if score > 0}
    
    retrieved_relevant = len(set(retrieved_docs) & relevant_docs)
    return retrieved_relevant / len(retrieved_docs)


def count_relevant_retrieved(
    retrieved_docs: List[str],
    ground_truth: Dict[str, int]
) -> int:
    """
    Count how many relevant documents were retrieved.
    
    Args:
        retrieved_docs: List of document IDs
        ground_truth: Dict mapping doc_id -> relevance_score (0-2)
        
    Returns:
        Number of relevant documents retrieved
    """
    relevant_docs = {doc_id for doc_id, score in ground_truth.items() if score > 0}
    return len(set(retrieved_docs) & relevant_docs)


def calculate_mrr_at_k(
    retrieved_docs: List[str],
    ground_truth: Dict[str, int],
    k: int = 5
) -> float:
    """
    Calculate MRR@k (Mean Reciprocal Rank) - reciprocal of rank of first relevant doc.
    
    Args:
        retrieved_docs: List of document IDs in ranked order
        ground_truth: Dict mapping doc_id -> relevance_score (0-2)
        k: Cutoff for MRR@k calculation
        
    Returns:
        MRR@k score (0.0 to 1.0)
    """
    if not retrieved_docs or not ground_truth:
        return 0.0
    
    retrieved_docs = retrieved_docs[:k]
    relevant_docs = {doc_id for doc_id, score in ground_truth.items() if score > 0}
    
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / rank
    
    return 0.0


def count_total_relevant(ground_truth: Dict[str, int]) -> int:
    """
    Count total number of relevant documents.
    
    Args:
        ground_truth: Dict mapping doc_id -> relevance_score (0-2)
        
    Returns:
        Number of relevant documents
    """
    return sum(1 for score in ground_truth.values() if score > 0)
