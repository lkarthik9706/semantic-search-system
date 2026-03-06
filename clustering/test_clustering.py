from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from fuzzy_cluster import perform_fuzzy_clustering

print("Loading dataset...")

dataset = fetch_20newsgroups(subset='train')

documents = dataset.data[:500]

print("Generating embeddings...")

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(documents)

print("Running fuzzy clustering...")

gmm, probabilities = perform_fuzzy_clustering(embeddings, n_clusters=10)

print("Shape of probability matrix:", probabilities.shape)

print("\nCluster distribution for first document:")

print(probabilities[0])
