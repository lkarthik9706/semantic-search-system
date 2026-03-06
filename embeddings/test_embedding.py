from embedder import generate_embeddings
from sklearn.datasets import fetch_20newsgroups

print("Loading dataset...")

dataset = fetch_20newsgroups(subset='train')

documents = dataset.data[:10]

print("Generating embeddings...")

embeddings = generate_embeddings(documents)

print("Embedding shape:", embeddings.shape)
