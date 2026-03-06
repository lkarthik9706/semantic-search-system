from sklearn.datasets import fetch_20newsgroups
from embedder import generate_embeddings
from vector_store import VectorStore

print("Loading dataset...")

dataset = fetch_20newsgroups(subset='train')

documents = dataset.data[:100]

print("Generating embeddings...")

embeddings = generate_embeddings(documents)

print("Creating vector store...")

vector_store = VectorStore(384)

vector_store.add_vectors(embeddings)

print("Searching similar documents...")

query = "space mission nasa"

query_embedding = generate_embeddings([query])[0]

distances, indices = vector_store.search(query_embedding)

print("Top similar document indexes:", indices)