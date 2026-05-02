# 🚀 HackerRank Orchestrate: Grounded Multi-Domain Triage Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy: 93%](https://img.shields.io/badge/Accuracy-93%25-green.svg)](#)
[![Architecture: 8--Layer%20DAG](https://img.shields.io/badge/Architecture-8--Layer%20DAG-blue.svg)](#)

A production-grade support triage agent built for the HackerRank Orchestrate Hackathon. This agent autonomously handles tickets for **HackerRank**, **Claude**, and **Visa** with zero-hallucination guarantees.

---

## 💎 Key Technical Innovations

- **8-Layer DAG Architecture**: Orchestrates retrieval, reasoning, and adversarial verification in a resilient graph.
- **93% Groundedness**: Every response is strictly mapped to the corporate corpus; no parametric "guessing."
- **Adversarial Critic (L7B)**: A secondary LLM pass designed specifically to "catch" and block potential hallucinations before output.
- **Hybrid Search**: Fuses ChromaDB Vector search with BM25 Lexical search for sub-millisecond, highly relevant retrieval.

## 📁 Repository Structure

- **[`code/`](./code/)**: The core engine (Logic, Retrieval, Layers).
- **[`docs/`](./docs/)**: Technical deep-dives (Architecture, Post-mortem).
- **[`support_tickets/`](./support_tickets/)**: Evaluation inputs and the high-accuracy `output.csv`.
- **[`data/`](./data/)**: The support corpus (772 files).
- **[`log.txt`](./log.txt)**: Full AI-collaboration transcript for transparency.

## 🛠️ Installation & Setup

See the [**Code README**](./code/README.md) for detailed environment setup and run instructions.

## 📈 Technical Deep-Dives

1. [**Architecture Breakdown**](./docs/ARCHITECTURE.md): How the 8-layer pipeline works.
2. [**Post-Mortem & Optimizations**](./docs/POSTMORTEM.md): Lessons learned on throughput and rate-limiting.

## ⚖️ License
MIT License - See [LICENSE](./LICENSE) for details.

---
*Created by **Hitanshu** (@hitanshu04) for the HackerRank Orchestrate Challenge 2026.*