import os
import glob
import re
import pickle
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from config import DATA_DIR, INDEX_DIR

def chunk_markdown(text: str, source_file: str, max_chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Splits text into chunks using a sliding window approach with overlap.
    Returns list of dicts: {"text": chunk_text, "source": source_file}
    """
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    current_chunk = []
    current_length = 0
    
    i = 0
    while i < len(words):
        word = words[i]
        word_len = len(word) + 1 # +1 for space
        
        if current_length + word_len > max_chunk_size and current_chunk:
            # Save chunk
            chunk_text = " ".join(current_chunk)
            chunks.append({"text": chunk_text, "source": source_file})
            
            # Backtrack for overlap
            overlap_length = 0
            overlap_words = []
            for w in reversed(current_chunk):
                if overlap_length + len(w) + 1 > overlap:
                    break
                overlap_words.insert(0, w)
                overlap_length += len(w) + 1
            
            current_chunk = overlap_words
            current_length = sum(len(w) + 1 for w in current_chunk)
            
        current_chunk.append(word)
        current_length += word_len
        i += 1
        
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if not chunks or chunks[-1]["text"] != chunk_text:
            chunks.append({"text": chunk_text, "source": source_file})
            
    return chunks

def build_indexes():
    """Reads all 772 files from data/ and builds ChromaDB and BM25 indexes."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
    
    # We use all-MiniLM-L6-v2 which runs locally and is extremely fast
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    domains = ["hackerrank", "claude", "visa"]
    
    for domain in domains:
        print(f"Indexing domain: {domain}...")
        domain_dir = os.path.join(DATA_DIR, domain)
        
        # 1. Read files
        filepaths = glob.glob(f"{domain_dir}/**/*.md", recursive=True)
        print(f"Found {len(filepaths)} files in {domain}")
        
        all_chunks = []
        for fp in filepaths:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Store relative path for cleaner output (e.g., hackerrank/screen/managing-tests.md)
                rel_path = os.path.relpath(fp, DATA_DIR)
                all_chunks.extend(chunk_markdown(content, rel_path))
                
        print(f"Total chunks for {domain}: {len(all_chunks)}")
        
        if not all_chunks:
            print(f"WARNING: No chunks found for {domain}")
            continue
            
        # 2. Build ChromaDB
        # Delete existing collection if it exists to ensure clean index
        try:
            chroma_client.delete_collection(name=domain)
        except Exception:
            pass
            
        collection = chroma_client.create_collection(name=domain, embedding_function=emb_fn)
        
        # Batch add to Chroma (Chroma handles batching internally, but we can do simple batching just in case)
        batch_size = 5000
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            collection.add(
                documents=[c["text"] for c in batch],
                metadatas=[{"source": c["source"]} for c in batch],
                ids=[f"{domain}_{i}_{j}" for j in range(len(batch))]
            )
            
        # 3. Build BM25
        tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Save BM25 index and chunk mapping using pickle
        bm25_data = {
            "bm25": bm25,
            "chunks": all_chunks
        }
        
        with open(os.path.join(INDEX_DIR, f"{domain}_bm25.pkl"), 'wb') as f:
            pickle.dump(bm25_data, f)
            
        print(f"Successfully indexed {domain}!\n")

if __name__ == "__main__":
    print("Starting Corpus Indexing...")
    build_indexes()
    print("Indexing Complete.")
