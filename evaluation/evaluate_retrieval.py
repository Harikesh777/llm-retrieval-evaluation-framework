import json
import time
import pandas as pd
import logging
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loaders import load_pdfs_from_directory
from chunkers.registry import CHUNKER_REGISTRY
from embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from retrieval.registry import RETRIEVAL_REGISTRY
from evaluation.metrics import get_relevance_scores, recall_at_k, precision_at_k, mrr, ndcg_at_k

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/raw_docs')
EVAL_DATASET = os.path.join(os.path.dirname(__file__), '../data/evaluation_dataset.json')
LEADERBOARD_PATH = os.path.join(os.path.dirname(__file__), '../leaderboard/retrieval_leaderboard.csv')
FAILURES_LOG_PATH = os.path.join(os.path.dirname(__file__), '../logs/retrieval_failures.json')

FIXED_CHUNKER = "sliding_10_2"
FIXED_EMBEDDING = "intfloat/e5-small-v2"

def load_evaluation_dataset(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation_pipeline():
    logger.info("Starting Retrieval Evaluation Pipeline...")
    
    documents = load_pdfs_from_directory(DATA_DIR)
    if not documents:
        logger.error("No documents found. Exiting.")
        return
        
    eval_dataset = load_evaluation_dataset(EVAL_DATASET)
    
    chunker = CHUNKER_REGISTRY.get(FIXED_CHUNKER)
    if not chunker:
        logger.error(f"Fixed chunker {FIXED_CHUNKER} not found in registry.")
        return
        
    logger.info(f"Chunking {len(documents)} documents using {FIXED_CHUNKER}...")
    chunks = []
    for doc in documents:
        text = doc.get("text", "")
        if text:
            chunks.extend(chunker.chunk(text, metadata=doc))
            
    chunk_texts = [c["text"] for c in chunks]
    
    logger.info(f"Generating embeddings using {FIXED_EMBEDDING}...")
    embedder = SentenceTransformerEmbedder(model_name="e5-small", hf_repo_id=FIXED_EMBEDDING)
    doc_embeddings = embedder.encode_documents(chunk_texts, batch_size=32)
    
    total_relevants_map = []
    for item in eval_dataset:
        target_substrings = item.get("relevant_chunks", [item.get("answer")])
        all_relevance = get_relevance_scores(chunk_texts, target_substrings)
        total_relevants_map.append((target_substrings, sum(all_relevance)))
    
    results = []
    failures_log = []
    
    os.makedirs(os.path.dirname(FAILURES_LOG_PATH), exist_ok=True)
    
    for ret_name, ret_class in RETRIEVAL_REGISTRY.items():
        logger.info(f"--- Evaluating Retriever: {ret_name} ---")
        
        try:
            retriever = ret_class()
            
            retriever.build_index(doc_embeddings, chunks)
            stats = retriever.get_stats()
            indexing_time = stats.get("indexing_time_ms", 0.0)
            memory_usage = stats.get("memory_usage_mb", 0.0)
            
            metrics_accum = {
                "recall_5": [],
                "recall_10": [],
                "precision_5": [],
                "mrr": [],
                "ndcg_10": [],
                "retrieval_latency": []
            }
            
            query_times = []
            
            for idx, item in enumerate(eval_dataset):
                query = item.get("query", item.get("question"))
                target_substrings, total_relevant = total_relevants_map[idx]
                
                query_emb = embedder.encode_queries([query])
                
                r_start = time.perf_counter()
                retrieved_ids, scores, retrieved_chunks = retriever.retrieve(query_emb, query_text=query, top_k=10)
                r_latency = (time.perf_counter() - r_start) * 1000
                query_times.append(r_latency)
                
                retrieved_texts = [c.get("text", "") for c in retrieved_chunks]
                
                relevance = get_relevance_scores(retrieved_texts, target_substrings)
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
                metrics_accum["retrieval_latency"].append(r_latency)
                
                if r10 == 0:
                    failures_log.append({
                        "retriever": ret_name,
                        "query": query,
                        "expected_chunks": target_substrings,
                        "retrieved_chunks": retrieved_texts,
                        "similarity_scores": scores
                    })
                    
            num_queries = len(eval_dataset)
            avg_r5 = sum(metrics_accum["recall_5"]) / num_queries
            avg_r10 = sum(metrics_accum["recall_10"]) / num_queries
            avg_p5 = sum(metrics_accum["precision_5"]) / num_queries
            avg_mrr = sum(metrics_accum["mrr"]) / num_queries
            avg_ndcg10 = sum(metrics_accum["ndcg_10"]) / num_queries
            avg_ret_latency = sum(metrics_accum["retrieval_latency"]) / num_queries
            
            avg_q_time_sec = sum(query_times) / (1000 * num_queries)
            throughput = 1.0 / avg_q_time_sec if avg_q_time_sec > 0 else 0
            
            latency_score = 1 / (1 + avg_ret_latency)
            overall_score = (0.40 * avg_r10) + (0.25 * avg_mrr) + (0.20 * avg_ndcg10) + (0.10 * avg_p5) + (0.05 * latency_score)
            
            results.append({
                "Retriever": ret_name,
                "Recall@5": avg_r5,
                "Recall@10": avg_r10,
                "Precision@5": avg_p5,
                "MRR": avg_mrr,
                "NDCG@10": avg_ndcg10,
                "Index Build Time(ms)": indexing_time,
                "Retrieval Latency(ms)": avg_ret_latency,
                "Throughput(req/sec)": throughput,
                "Memory Usage(MB)": memory_usage,
                "Overall Score": overall_score
            })
            
        except Exception as e:
            logger.error(f"Error evaluating retriever {ret_name}: {e}")
            continue
            
    with open(FAILURES_LOG_PATH, 'w') as f:
        json.dump(failures_log, f, indent=4)
    logger.info(f"Saved {len(failures_log)} failure cases to {FAILURES_LOG_PATH}")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Overall Score", ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = "Rank"
        
        os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
        df.to_csv(LEADERBOARD_PATH)
        logger.info(f"Leaderboard saved to {LEADERBOARD_PATH}")
        print("\n=== Retrieval Leaderboard ===")
        print(df)
    else:
        logger.warning("No results to build leaderboard.")

if __name__ == "__main__":
    run_evaluation_pipeline()
