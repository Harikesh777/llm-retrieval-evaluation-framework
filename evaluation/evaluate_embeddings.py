import json
import time
import pandas as pd
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loaders import load_pdfs_from_directory
from chunkers.registry import CHUNKER_REGISTRY
from embeddings.registry import EMBEDDING_REGISTRY
from embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from retrieval.retriever import stateless_retrieve
from evaluation.metrics import get_relevance_scores, recall_at_k, precision_at_k, mrr, ndcg_at_k

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/raw_docs')
EVAL_DATASET = os.path.join(os.path.dirname(__file__), '../data/evaluation_dataset.json')
LEADERBOARD_PATH = os.path.join(os.path.dirname(__file__), '../leaderboard/embedding_leaderboard.csv')

FIXED_CHUNKER = "sliding_10_2"

def load_evaluation_dataset(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation_pipeline():
    logger.info("Starting Embedding Evaluation Pipeline...")
    
    documents = load_pdfs_from_directory(DATA_DIR)
    if not documents:
        logger.error("No documents found. Exiting.")
        return
        
    eval_dataset = load_evaluation_dataset(EVAL_DATASET)
    
    chunker = CHUNKER_REGISTRY.get(FIXED_CHUNKER)
    if not chunker:
        logger.error(f"Fixed chunker {FIXED_CHUNKER} not found in registry. Exiting.")
        return
        
    logger.info(f"Chunking {len(documents)} documents using fixed strategy: {FIXED_CHUNKER}...")
    chunks = []
    for doc in documents:
        text = doc.get("text", "")
        if text:
            chunks.extend(chunker.chunk(text, metadata=doc))
            
    chunk_texts = [c["text"] for c in chunks]
    logger.info(f"Generated {len(chunks)} total chunks.")
    
    results = []
    
    for model_name, hf_repo_id in EMBEDDING_REGISTRY.items():
        logger.info(f"--- Evaluating Embedding Model: {model_name} ({hf_repo_id}) ---")
        
        try:
            embedder = SentenceTransformerEmbedder(model_name=model_name, hf_repo_id=hf_repo_id)
            dim = embedder.get_embedding_dimension()
            
            start_emb = time.perf_counter()
            doc_embeddings = embedder.encode_documents(chunk_texts, batch_size=32)
            emb_time = time.perf_counter() - start_emb
            
            mem_usage_mb = (doc_embeddings.shape[0] * doc_embeddings.shape[1] * 4) / (1024 * 1024)
            
            metrics_accum = {
                "recall_5": [],
                "recall_10": [],
                "precision_5": [],
                "mrr": [],
                "ndcg_10": [],
                "retrieval_latency": []
            }
            
            query_times = []
            
            for item in eval_dataset:
                query = item.get("query", item.get("question"))
                target_substrings = item.get("relevant_chunks", [item.get("answer")])
                
                q_start = time.perf_counter()
                query_emb = embedder.encode_queries([query])
                q_time = time.perf_counter() - q_start
                query_times.append(q_time)
                
                r_start = time.perf_counter()
                retrieved_ids, scores, retrieved_texts = stateless_retrieve(query_emb, doc_embeddings, chunks, k=10)
                retrieval_latency = (time.perf_counter() - r_start) * 1000  # in ms
                
                relevance = get_relevance_scores(retrieved_texts, target_substrings)
                all_relevance = get_relevance_scores(chunk_texts, target_substrings)
                total_relevant = sum(all_relevance)
                
                r5 = recall_at_k(relevance, total_relevant, 5)
                r10 = recall_at_k(relevance, total_relevant, 10)
                p5 = precision_at_k(relevance, 5)
                m = mrr(relevance)
                n10 = ndcg_at_k(relevance, 10)
                
                metrics_accum["recall_5"].append(r5)
                metrics_accum["recall_10"].append(r10)
                metrics_accum["precision_5"].append(p5)
                metrics_accum["mrr"].append(m)
                metrics_accum["ndcg_10"].append(n10)
                metrics_accum["retrieval_latency"].append(retrieval_latency)
                
            num_queries = len(eval_dataset)
            avg_r5 = sum(metrics_accum["recall_5"]) / num_queries
            avg_r10 = sum(metrics_accum["recall_10"]) / num_queries
            avg_p5 = sum(metrics_accum["precision_5"]) / num_queries
            avg_mrr = sum(metrics_accum["mrr"]) / num_queries
            avg_ndcg10 = sum(metrics_accum["ndcg_10"]) / num_queries
            avg_ret_latency = sum(metrics_accum["retrieval_latency"]) / num_queries
            
            avg_emb_latency = (emb_time / len(chunk_texts)) * 1000  # ms per document
            avg_q_time = sum(query_times) / num_queries
            throughput = 1.0 / avg_q_time if avg_q_time > 0 else 0
            
            latency_score = 1 / (1 + avg_ret_latency)
            overall_score = (0.40 * avg_r10) + (0.25 * avg_mrr) + (0.20 * avg_ndcg10) + (0.10 * avg_p5) + (0.05 * latency_score)
            
            results.append({
                "Embedding Model": model_name,
                "Recall@5": avg_r5,
                "Recall@10": avg_r10,
                "Precision@5": avg_p5,
                "MRR": avg_mrr,
                "NDCG@10": avg_ndcg10,
                "Embedding Latency(ms)": avg_emb_latency,
                "Retrieval Latency(ms)": avg_ret_latency,
                "Throughput(req/sec)": throughput,
                "Embedding Dimension": dim,
                "Memory Usage(MB)": mem_usage_mb,
                "Overall Score": overall_score
            })
            
            del embedder
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            continue
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Overall Score", ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = "Rank"
        
        os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
        df.to_csv(LEADERBOARD_PATH)
        logger.info(f"Leaderboard saved to {LEADERBOARD_PATH}")
        print("\n=== Embedding Leaderboard ===")
        print(df)
    else:
        logger.warning("No results to build leaderboard.")

if __name__ == "__main__":
    run_evaluation_pipeline()
