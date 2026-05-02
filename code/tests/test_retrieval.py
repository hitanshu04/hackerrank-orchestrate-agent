import pytest
import os
from retrieval.retriever import HybridRetriever
from config import INDEX_DIR

@pytest.fixture(scope="module")
def retriever():
    # Only run tests if indexes have been built
    if not os.path.exists(INDEX_DIR):
        pytest.skip("Index directory not found. Please run indexer.py first.")
    return HybridRetriever()

def test_hackerrank_retrieval(retriever):
    query = "adding extra time for candidates"
    results = retriever.retrieve(query, domain="hackerrank")
    assert len(results) > 0
    # Check if the correct file is near the top
    top_sources = [r.source_file for r in results]
    assert any("4811403281-adding-extra-time-for-candidates.md" in src for src in top_sources)

def test_visa_retrieval(retriever):
    query = "lost or stolen card"
    results = retriever.retrieve(query, domain="visa")
    assert len(results) > 0
    top_sources = [r.source_file for r in results]
    # Check if a relevant Visa article is retrieved
    assert any("card" in src.lower() or "support" in src.lower() for src in top_sources)

def test_claude_retrieval(retriever):
    query = "delete conversation"
    results = retriever.retrieve(query, domain="claude")
    assert len(results) > 0
    top_sources = [r.source_file for r in results]
    # In claude, privacy or settings often contain deletion instructions
    assert len(top_sources) > 0

def test_empty_query_handles_gracefully(retriever):
    results = retriever.retrieve("", domain="hackerrank")
    # BM25 might return 0 scores, Vector might return something
    assert isinstance(results, list)

def test_unknown_domain_returns_empty(retriever):
    results = retriever.retrieve("test", domain="unknown_domain")
    assert len(results) == 0
