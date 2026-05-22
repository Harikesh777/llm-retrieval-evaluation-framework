import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

import re

def normalize_text(text: str) -> str:
    """Remove all non-alphanumeric characters and lowercase the string for robust matching."""
    return re.sub(r'[^a-z0-9]', '', text.lower())

def get_relevance_scores(retrieved_texts: List[str], target_substrings: List[str]) -> List[int]:
    """
    Convert retrieved chunks into binary relevance scores by checking if they contain any target substring.
    """
    relevance = []
    normalized_targets = [normalize_text(t) for t in target_substrings]
    
    for text in retrieved_texts:
        normalized_text = normalize_text(text)
        is_relevant = any(target in normalized_text for target in normalized_targets)
        relevance.append(1 if is_relevant else 0)
    return relevance

def recall_at_k(relevance: List[int], total_relevant: int, k: int) -> float:
    """
    Calculate Recall@K.
    
    Args:
        relevance: List of binary relevance scores (1 for relevant, 0 for not).
        total_relevant: Total number of relevant items existing in the entire dataset for this query.
        k: The rank cutoff.
        
    Returns:
        Recall@K score.
    """
    if total_relevant == 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant

def precision_at_k(relevance: List[int], k: int) -> float:
    """
    Calculate Precision@K.
    
    Args:
        relevance: List of binary relevance scores.
        k: The rank cutoff.
        
    Returns:
        Precision@K score.
    """
    if k == 0:
        return 0.0
    k = min(k, len(relevance))
    if k == 0:
        return 0.0
    return sum(relevance[:k]) / k

def mrr(relevance: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a single query.
    
    Args:
        relevance: List of binary relevance scores.
        
    Returns:
        MRR score.
    """
    for i, rel in enumerate(relevance):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0

def dcg_at_k(relevance: List[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG) at K.
    """
    relevance = relevance[:k]
    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2) # i is 0-indexed, so i+2 starts at 2
    return dcg

def ndcg_at_k(relevance: List[int], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at K.
    Assuming ideal DCG is when all top K retrieved items are relevant (i.e. relevance = [1, 1, ...]).
    Wait, ideal DCG depends on the total number of actual relevant items.
    """
    dcg = dcg_at_k(relevance, k)
    
    num_relevant = sum(relevance)
    if num_relevant == 0:
        return 0.0
        
    ideal_relevance = [1] * num_relevant + [0] * max(0, k - num_relevant)
    idcg = dcg_at_k(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg
