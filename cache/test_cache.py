from sentence_transformers import SentenceTransformer
from semantic_cache import SemanticCache

model = SentenceTransformer('all-MiniLM-L6-v2')

cache = SemanticCache()

query1 = "laws about guns"
query2 = "gun legislation rules"

emb1 = model.encode(query1)
emb2 = model.encode(query2)

# first query
hit, entry, score = cache.lookup(emb1)

if not hit:
    print("Cache miss")
    cache.add(query1, emb1, "Result about gun laws")

# second query
hit, entry, score = cache.lookup(emb2)

if hit:
    print("Cache hit!")
    print("Matched query:", entry["query"])
    print("Similarity:", score)