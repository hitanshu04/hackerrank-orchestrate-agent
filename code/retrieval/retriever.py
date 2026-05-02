import os
import pickle
import chromadb
from chromadb.utils import embedding_functions
from config import INDEX_DIR, BM25_TOP_K, VECTOR_TOP_K, FINAL_TOP_K
from models import EvidenceChunk


class HybridRetriever:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.bm25_indexes = {}
        self.collections = {}

        # Lazy load indexes
        self._load_domain("hackerrank")
        self._load_domain("claude")
        self._load_domain("visa")

    def _load_domain(self, domain: str):
        try:
            # Try to get existing collection, or create if doesn't exist
            try:
                self.collections[domain] = self.chroma_client.get_collection(
                    name=domain
                )
            except:
                self.collections[domain] = self.chroma_client.create_collection(
                    name=domain, embedding_function=self.emb_fn
                )

            bm25_path = os.path.join(INDEX_DIR, f"{domain}_bm25.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    self.bm25_indexes[domain] = pickle.load(f)
            else:
                print(f"Warning: BM25 index not found for {domain}")
        except Exception as e:
            print(f"Warning: Could not load index for {domain}. Error: {e}")

    def retrieve(self, query: str, domain: str) -> list[EvidenceChunk]:
        """
        Retrieves top evidence chunks using Reciprocal Rank Fusion (RRF)
        to combine BM25 and Vector search results.
        """
        if domain not in self.collections or domain not in self.bm25_indexes:
            return []

        # 1. Vector Search
        collection = self.collections[domain]
        vector_results = collection.query(
            query_texts=[query],
            n_results=VECTOR_TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        vector_chunks = []
        if vector_results["documents"] and len(vector_results["documents"]) > 0:
            docs = vector_results["documents"][0]
            metas = vector_results["metadatas"][0]
            dists = vector_results["distances"][0]

            for d, m, dist in zip(docs, metas, dists):
                # Distance in chroma is cosine distance (0 is exact match, 1 is orthogonal)
                # Convert distance to similarity score (1 - dist/2) approx
                sim = max(0.0, 1.0 - (dist / 2.0))
                vector_chunks.append({"text": d, "source": m["source"], "score": sim})

        # 2. BM25 Search
        bm25_data = self.bm25_indexes[domain]
        bm25 = bm25_data["bm25"]
        chunks = bm25_data["chunks"]

        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Get top K BM25 indices
        top_k_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:BM25_TOP_K]

        bm25_chunks = []
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        for i in top_k_indices:
            if bm25_scores[i] > 0:
                bm25_chunks.append(
                    {
                        "text": chunks[i]["text"],
                        "source": chunks[i]["source"],
                        "score": bm25_scores[i] / max_bm25,  # Normalize to 0-1
                    }
                )

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = 1 / (k + rank)  where k is usually 60
        k = 60
        rrf_scores = {}
        chunk_map = {}

        for rank, chunk in enumerate(vector_chunks):
            chunk_hash = hash(chunk["text"])
            chunk_map[chunk_hash] = chunk
            rrf_scores[chunk_hash] = rrf_scores.get(chunk_hash, 0) + (
                1.0 / (k + rank + 1)
            )

        for rank, chunk in enumerate(bm25_chunks):
            chunk_hash = hash(chunk["text"])
            chunk_map[chunk_hash] = chunk
            rrf_scores[chunk_hash] = rrf_scores.get(chunk_hash, 0) + (
                1.0 / (k + rank + 1)
            )

        # Sort by RRF score
        sorted_hashes = sorted(
            rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True
        )[:FINAL_TOP_K]

        final_chunks = []
        for h in sorted_hashes:
            c = chunk_map[h]
            final_chunks.append(
                EvidenceChunk(
                    text=c["text"],
                    source_file=c["source"],
                    # Keep ranking by RRF, but expose a 0-1 confidence-like score
                    # for downstream thresholding in L6.
                    score=float(c["score"]),
                )
            )

        return final_chunks
