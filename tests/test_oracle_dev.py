import pandas as pd
from pathlib import Path
import sys
import json
from sentence_transformers import SentenceTransformer

# Set project root
path = str(Path(__file__).resolve().parents[1])
sys.path.append(path)
from src.selectivity import compute_selectivity, mask
from src.eval.oracle import brute_force
from src.eval.metrics import compute_recall

base_dir_path = Path(__file__).resolve().parent.parent
metadata_path = base_dir_path / "artifacts" / "dev" / "v1" / "metadata.parquet"
queries_path = base_dir_path / "artifacts" / "dev" / "v1" / "queries.parquet"

df_metadata = pd.read_parquet(metadata_path)
df_queries = pd.read_parquet(queries_path)
K = 10

model = SentenceTransformer("all-MiniLM-L6-v2")

first_query = df_queries.iloc[0]["filters"]
for index, query in df_queries.iterrows():
    qtext = query["qtext"]
    qvec = model.encode(qtext, normalize_embeddings=True)
    filter = json.loads(query["filters"])
    filter_mask = mask(filter, df_metadata)
    allowed_ids = df_metadata.index[filter_mask].tolist()
    oracle_ids = brute_force(qvec, allowed_ids, K)
    backend_ids = oracle_ids # set same for testing recall
    recall = compute_recall(backend_ids, oracle_ids, K)
    selectivity = compute_selectivity(filter, df_metadata)
    print(f"Query: {index} | Recall@{K}={recall:.2f} | Selectivity:{selectivity}")