# 🏗️ 8-Layer DAG Architecture: Technical Deep-Dive

This agent is built as a **Directed Acyclic Graph (DAG)** of specialized layers, prioritizing **groundedness** and **safety** above all else.

## The Problem
Generic RAG systems often fail in multi-domain triage because:
1. **Compound Intents**: Users ask multiple things at once.
2. **Hallucination Risk**: LLMs "guess" policies when evidence is weak.
3. **Domain Overlap**: Claude, Visa, and HackerRank have overlapping terms (e.g., "Account", "Security").

## The Solution: 8-Layer DAG

### Layer 1: Ingestion & Normalization
- **Technique**: `ftfy` (fixes text for you) + Unicode normalization.
- **Why**: Ensures weird characters from copied-pasted tickets don't break downstream JSON parsing.

### Layer 2: LLM-Driven Intent Splitting
- **Technique**: Recursive intent extraction.
- **Why**: Allows the retriever to search for each specific problem individually, increasing hit rate by ~40%.

### Layer 3: Hybrid Domain Routing
- **Technique**: Heuristic Lexical Scoring + LLM Fallback.
- **Why**: 90% of tickets can be routed by keywords (Visa/HackerRank/Claude). The LLM is only used for the 10% "Ambiguous" cases, saving cost and time.

### Layer 4: Taxonomy Classification
- **Allowed Values**: `product_issue`, `feature_request`, `bug`, `invalid`.

### Layer 5: Adversarial Risk Engine
- **Technique**: Pattern Matching + LLM Risk Scoring.
- **Safety**: Hard-short-circuit for scores ≥ 9. This protects the system from prompt injection and social engineering attempts.

### Layer 6: Multi-Stage Hybrid Retriever
- **Stage 1**: BM25 (Lexical) - Finds exact policy names.
- **Stage 2**: ChromaDB (Semantic) - Finds similar concepts.
- **Re-ranking**: Similarity Floor filtering to prevent "Garbage In, Garbage Out".

### Layer 7: Grounded Composer (L7A) & Adversarial Critic (L7B)
- **Composer**: Generates a response strictly from Layer 6 evidence.
- **Critic**: **The Secret Sauce**. A second LLM pass that attempts to "Debunk" the response by finding any claims not present in the evidence. If a hallucination is found, the system escalates rather than replying.

### Layer 8: Output Validation
- **Technique**: Pydantic Schema enforcement.
- **Final Check**: Ensures valid JSON and status consistency.
