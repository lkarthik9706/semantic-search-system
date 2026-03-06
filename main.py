from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

from embeddings.vector_store import VectorStore
from clustering.fuzzy_cluster import perform_fuzzy_clustering
from cache.semantic_cache import SemanticCache


app = FastAPI()

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load dataset
dataset = fetch_20newsgroups(subset="train")

documents = dataset.data[:500]

# generate embeddings
doc_embeddings = model.encode(documents)

# vector store
vector_store = VectorStore(384)
vector_store.add_vectors(doc_embeddings)

# clustering
gmm, probabilities = perform_fuzzy_clustering(doc_embeddings, n_clusters=10)

# semantic cache
cache = SemanticCache()


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_api(request: QueryRequest):

    query = request.query

    query_embedding = model.encode(query)

    # check cache
    hit, entry, similarity = cache.lookup(query_embedding)

    if hit:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(similarity),
            "result": entry["result"],
            "dominant_cluster": None
        }

    # search documents
    distances, indices = vector_store.search(query_embedding)

    best_doc_index = indices[0][0]

    result = documents[best_doc_index]

    dominant_cluster = probabilities[best_doc_index].argmax()

    # store in cache
    cache.add(query, query_embedding, result)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result[:500],
        "dominant_cluster": int(dominant_cluster)
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}