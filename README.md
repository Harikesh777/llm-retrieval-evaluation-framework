# RAG Evaluation Framework

A production-grade Retrieval-Augmented Generation (RAG) benchmarking framework designed to evaluate chunking strategies, embedding models, and retrieval algorithms.

## Features

- **Chunking Benchmark**: Test 28+ chunking strategies (Fixed, Semantic, Recursive, Sentence-based, Sliding Window) against a benchmark dataset.
- **Embedding Benchmark**: Evaluate dense embedding models (like BGE, E5, MiniLM) for retrieval quality, testing Recall@K, MRR, NDCG, throughput, and memory footprint.
- **Retrieval Benchmark**: Compare diverse search strategies including Exact Cosine, FAISS Flat (L2 Normalized IP), FAISS IVF, FAISS HNSW, BM25 (Sparse), and a Hybrid fusion model.
- **Streamlit Dashboard**: A fully featured 3-tab dashboard for visualizing metrics and leaderboard comparisons natively.

## Project Structure

- `chunkers/`: Modular implementations for various chunking strategies.
- `embeddings/`: Wrapper implementations and registries for testing HuggingFace SentenceTransformer models.
- `retrieval/`: Base interfaces and specific implementations for dense, sparse, and hybrid retrieval indexes.
- `evaluation/`: The core benchmarking pipelines (`evaluate_chunking.py`, `evaluate_embeddings.py`, `evaluate_retrieval.py`).
- `leaderboard/`: Generated CSV leaderboards tracking the execution results of the pipelines.
- `logs/`: Edge-case evaluation logging (e.g. `retrieval_failures.json` where Recall drops to 0).

## Setup & Installation

1. Create a virtual environment and activate it:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Evaluate Chunking Strategies**:
   ```bash
   python evaluation/evaluate_chunking.py
   ```
2. **Evaluate Embedding Models**:
   ```bash
   python evaluation/evaluate_embeddings.py
   ```
3. **Evaluate Retrieval Algorithms**:
   ```bash
   python evaluation/evaluate_retrieval.py
   ```
4. **Launch Dashboard**:
   ```bash
   streamlit run app.py
   ```
