import numpy as np

def precision_at_k(relevant, k=10):
    return sum(relevant[:k]) / k

def mrr(relevant):
    for i, rel in enumerate(relevant):
        if rel:
            return 1 / (i + 1)
    return 0

def ndcg(scores):
    return np.mean(scores)
