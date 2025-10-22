
import time
import numpy as np
import pandas as pd
from src.selectivity import mask

def pre_filter_search(qvec: np.ndarray, filters: dict, df_metadata: pd.DataFrame, vectors: np.ndarray, K: int = 10):
    start = time.time()

    # Get allowed IDs
    allowed_mask = mask(filters, df_metadata)
    # get indices where value is true
    allowed_ids = np.where(allowed_mask)[0]

    if len(allowed_ids) == 0:
        return [], {"latency_ms": 0.0, "scored_vectors": 0}
    
    sub_vectors = vectors[allowed_ids]

    # Conduct Top-K inner product search
    similarity = np.dot(sub_vectors, qvec)
    top_idx = np.argpartition(-similarity, K)[:K]
    sorted_idx = top_idx[np.argsort(-similarity[top_idx])]
    results = allowed_ids[sorted_idx]

    latency_ms = (time.time() - start) * 1000
    stats = {"latency_ms": latency_ms, "scored_vectors": len(allowed_ids)}

    return results.tolist(), stats