"""
Computes recall@K between the backend ids and oracle ids.

Args:
    backend_ids: list[int]
        IDs returned from chosen backend
    oracle_ids: list[int]
        Ground-truth IDs from bruteforce oracle
    K: int
        Cutoff for evaluation
"""
def compute_recall(backend_ids, oracle_ids, K) -> float:
    # Take top-K from each, but ignore padding (-1) if present
    backend_top = [i for i in backend_ids[:K] if i != -1]
    oracle_top = [i for i in oracle_ids[:K] if i != -1]

    # If oracle has fewer than K valid ids, we normalize by its length
    denom = max(1, min(K, len(oracle_top)))

    backend_set = set(backend_top)
    oracle_set = set(oracle_top)
    intersection = len(backend_set & oracle_set)

    return intersection / denom