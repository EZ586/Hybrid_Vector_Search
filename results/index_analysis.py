import faiss

index_path = "results/indexes/faiss_ivf.index"
index = faiss.read_index(index_path)

print("Index type:", type(index))
print("nlist:", index.nlist)
print("ntotal (number of vectors):", index.ntotal)
print("dimension:", index.d)