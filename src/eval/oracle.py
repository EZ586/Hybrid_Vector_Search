import numpy as np
from typing import Optional
from pathlib import Path

base_dir_path = Path(__file__).resolve().parent.parent.parent
print(f"Base path: \n{base_dir_path}")

vector_path = base_dir_path / "artifacts" / "dev" / "v1"
print(f"Vector Path:\n{vector_path}")

VECTORS: Optional[np.ndarray] = None

def set_vectors(arr: np.ndarray):
    global VECTORS
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    VECTORS = arr


def load_vectors(version):
    """
    Loads vectors.npy from either dev or full

    Args:
        version: either dev or full
    """
    global VECTORS
    arr = []
    if version == "full":
        arr = np.load(base_dir_path / "artifacts" / "dev" / "v1" / "vectors.npy")
    else:
        arr = np.load(base_dir_path / "artifacts" / "full" / "v1" / "vectors.npy")
    # ensure that VECTOR has type float32
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    VECTORS = arr


def brute_force(qvec, allowed_ids, K) -> list[int]:
    # flatten qvec and allowed_ids into 1D array
    # ensures we are working with expected shape
    # converts into nparray if not already is nparray
    query = np.asarray(qvec, dtype=np.float32).reshape(-1)
    allowed = np.asarray(allowed_ids, dtype=np.int64).reshape(-1)
    if allowed.size == 0 or K <= 0:
        return []
    # considers case where K value set to be larger than available vectors
    K_val = min(K, allowed.size)

    candidates = VECTORS[allowed]
    scores = np.dot(candidates.astype(np.float64), query.astype(np.float64))
    indices = []
    # Case where K_val is same as number of scores (no need to partition)
    if K_val == scores.size:
        # -scores to get index positions from greatest to least
        # stable to preserve original order of equal elements
        indices = np.argsort(-scores, kind="stable")
    else:
        # use argpartition to find top K unordered indices for efficiency
        partitioned = np.argpartition(-scores, K_val - 1)[:K_val]
        top_scores = scores[partitioned]
        top_index = np.argsort(-top_scores, kind="stable")
        # done to get indices relative to all candidates
        indices = partitioned[top_index]
    top_k = allowed[indices][:K_val]
    return [int(x) for x in top_k]


# testing bruteforce search
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    load_vectors("dev")
    N, D = VECTORS.shape
    print(f"Shape:\n{VECTORS.shape}")
    # Create a random query and normalize
    q = rng.normal(size=(D,)).astype(np.float32)
    q = q / np.linalg.norm(q)
    # Select a random subset of allowed ids
    allowed = rng.choice(np.arange(N), size=50, replace=False)

    # Compute brute force using our function
    top5 = brute_force(q, allowed, K=5)
    print("Top-5 oracle ids:", top5)

    # Sanity check against explicit computation
    scores = VECTORS[allowed].dot(q)
    expected_order = allowed[np.argsort(-scores)][:5].tolist()
    print("Expected Top-5 ids:", expected_order)
    assert top5 == expected_order, "Mismatch in brute-force ranking"
    print("Self-check passed.")
