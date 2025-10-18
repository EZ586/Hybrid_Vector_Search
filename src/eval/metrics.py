
"""
Computes recall@K between the backend ids and oracle ids

Args:
    backend_ids: list[int]
        IDs returned from chosen backend
    oracle_ids: list[int]
        Ground-truth IDs from bruteforce oracle
    K: int
        Cutoff for evaluation
"""
def compute_recall(backend_ids, oracle_ids, K) -> float:
    # convert into set for efficiency
    backend_set = set(backend_ids[:K])
    oracle_set = set(oracle_ids[:K])
    intersection = len(backend_set & oracle_set)
    return intersection / K
