# src/evaluate.py
import math
import csv
from typing import List

def dcg(scores: List[float], k: int):
    """DCG with relevance scores (higher better)."""
    scores = scores[:k]
    return sum((2**rel - 1) / math.log2(i+2) for i, rel in enumerate(scores))

def ndcg_at_k(predicted_authors: List[str], gold_authors: List[str], k: int = 10):
    """
    predicted_authors: list in rank order (top1,...)
    gold_authors: list of relevant authors (can be multiple)
    We give binary relevance (1 if in gold), so ideal DCG = sum_{i=0}^{min(len(gold),k)} (2^1-1)/log2(i+2)
    """
    rels = [1.0 if a in gold_authors else 0.0 for a in predicted_authors[:k]]
    dcg_val = dcg(rels, k)
    ideal_rels = sorted(rels, reverse=True)
    idcg = dcg(ideal_rels, k)
    return dcg_val / idcg if idcg > 0 else 0.0

def mrr_at_k(predicted_authors: List[str], gold_authors: List[str], k: int = 10):
    for i, a in enumerate(predicted_authors[:k], start=1):
        if a in gold_authors:
            return 1.0 / i
    return 0.0

def average_precision(predicted_authors: List[str], gold_authors: List[str], k: int = 10):
    hits = 0
    sum_prec = 0.0
    for i, a in enumerate(predicted_authors[:k], start=1):
        if a in gold_authors:
            hits += 1
            sum_prec += hits / i
    return sum_prec / max(hits, 1) if hits > 0 else 0.0

def evaluate(predictions: List[List[str]], golds: List[List[str]], k=10):
    """
    predictions: list of predicted author-lists (strings) per query
    golds: list of gold author-lists per query
    returns dict of averaged metrics
    """
    ndcgs, mrrs, aps = [], [], []
    for pred, gold in zip(predictions, golds):
        ndcgs.append(ndcg_at_k(pred, gold, k))
        mrrs.append(mrr_at_k(pred, gold, k))
        aps.append(average_precision(pred, gold, k))
    return {
        "NDCG@%d" % k: sum(ndcgs)/len(ndcgs),
        "MRR@%d" % k: sum(mrrs)/len(mrrs),
        "MAP@%d" % k: sum(aps)/len(aps)
    }

def load_gold_csv(path):
    """
    Expect CSV with columns: query,gold_authors
    gold_authors can be semicolon-separated if multiple.
    """
    queries, golds = [], []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            queries.append(r['query'])
            golds.append([a.strip() for a in r['gold_authors'].split(';') if a.strip()])
    return queries, golds
