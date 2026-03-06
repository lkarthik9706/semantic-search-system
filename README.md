## System Architecture

The system consists of four main components:

1. **Embedding Layer**
   - Uses `SentenceTransformer (all-MiniLM-L6-v2)` to convert documents and queries into 384-dimensional vectors.
   - This model was selected for its balance between semantic quality and inference speed.

2. **Vector Database**
   - Uses **FAISS** for efficient nearest-neighbor search.
   - All document embeddings are stored in a FAISS index to enable fast semantic retrieval.

3. **Fuzzy Clustering**
   - Implemented using **Gaussian Mixture Models (GMM)**.
   - Unlike hard clustering, GMM produces probability distributions across clusters, allowing documents to belong to multiple topics.

4. **Semantic Cache**
   - Built from scratch without Redis or external caching libraries.
   - Uses cosine similarity between query embeddings.
   - If similarity exceeds a defined threshold (e.g., 0.85), the cached response is reused.

---

## Cache Design Decision

The semantic cache introduces a **similarity threshold parameter**.

- **Lower threshold**
  - More cache hits
  - Risk of returning less accurate results

- **Higher threshold**
  - More accurate responses
  - Fewer cache hits

This parameter demonstrates the trade-off between **system efficiency and semantic precision**.

---

## API Endpoints

| Endpoint | Method | Description |
|--------|--------|--------|
| `/query` | POST | Performs semantic search with cache lookup |
| `/cache/stats` | GET | Returns cache statistics |
| `/cache` | DELETE | Clears the semantic cache |

---

## Example Response

```json
{
 "query": "space exploration nasa",
 "cache_hit": true,
 "matched_query": "nasa space missions",
 "similarity_score": 0.91,
 "result": "...",
 "dominant_cluster": 3
}
User Query
↓
Embedding Model
↓
Semantic Cache Check
↓
Vector Search (FAISS)
↓
Cluster Analysis
↓
API Response

This makes the repo look **very professional**. ✨

---

Make sure your repo has:

- ✔ `main.py`
- ✔ `requirements.txt`
- ✔ folders: `data/`, `embeddings/`, `clustering/`, `cache/`
- ✔ `README.md` with architecture explanation
- ✔ API runs with:

```bash
uvicorn main:app --reload

