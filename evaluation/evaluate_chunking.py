import json
import time
import pandas as pd
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loaders import load_pdfs_from_directory
from chunkers.registry import CHUNKER_REGISTRY
from embeddings.generate_embeddings import EmbeddingGenerator
from retrieval.retriever import stateless_retrieve
from evaluation.metrics import get_relevance_scores, recall_at_k, precision_at_k, mrr, ndcg_at_k

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/raw_docs')
EVAL_DATASET = os.path.join(os.path.dirname(__file__), '../data/evaluation_dataset.json')
LEADERBOARD_PATH = os.path.join(os.path.dirname(__file__), '../leaderboard/chunking_leaderboard.csv')

def load_evaluation_dataset(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation_pipeline():
    logger.info("Starting Chunking Evaluation Pipeline...")
    
    documents = load_pdfs_from_directory(DATA_DIR)
    if not documents:
        logger.error("No documents found. Exiting.")
        return
        
    eval_dataset = load_evaluation_dataset(EVAL_DATASET)
    
    emb_generator = EmbeddingGenerator()
    results = []
    
    for name, chunker in CHUNKER_REGISTRY.items():
        logger.info(f"--- Evaluating Strategy: {name} ---")
        
        chunks = []
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
            doc_chunks = chunker.chunk(text, metadata=doc)
            chunks.extend(doc_chunks)
            
        if not chunks:
            logger.warning(f"Strategy {name} produced no chunks. Skipping.")
            continue
            
        start_time = time.time()
        doc_embeddings = emb_generator.encode_chunks(chunks)
        
        metrics_accum = {
            "recall_5": [],
            "recall_10": [],
            "precision_5": [],
            "mrr": [],
            "ndcg_10": [],
            "latency": []
        }
        
        for item in eval_dataset:
            query = item.get("query", item.get("question"))
            target_substrings = item.get("relevant_chunks", [item.get("answer")])
            
            q_start = time.time()
            query_emb = emb_generator.encode_queries([query])
            retrieved_ids, scores, retrieved_texts = stateless_retrieve(query_emb, doc_embeddings, chunks, k=10)
            latency_ms = (time.time() - q_start) * 1000
            
            relevance = get_relevance_scores(retrieved_texts, target_substrings)
            
            all_chunk_texts = [c["text"] for c in chunks]
            all_relevance = get_relevance_scores(all_chunk_texts, target_substrings)
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
            metrics_accum["latency"].append(latency_ms)
            
        avg_r5 = sum(metrics_accum["recall_5"]) / len(eval_dataset)
        avg_r10 = sum(metrics_accum["recall_10"]) / len(eval_dataset)
        avg_p5 = sum(metrics_accum["precision_5"]) / len(eval_dataset)
        avg_mrr = sum(metrics_accum["mrr"]) / len(eval_dataset)
        avg_ndcg10 = sum(metrics_accum["ndcg_10"]) / len(eval_dataset)
        avg_latency = sum(metrics_accum["latency"]) / len(eval_dataset)
        
        latency_score = 1 / (1 + avg_latency)
        overall_score = (0.3 * avg_r10) + (0.2 * avg_p5) + (0.2 * avg_mrr) + (0.2 * avg_ndcg10) + (0.1 * latency_score)
        
        results.append({
            "Chunk Strategy": name,
            "Chunk Size": getattr(chunker, "size", getattr(chunker, "token_size", getattr(chunker, "chunk_size", getattr(chunker, "num_sentences", getattr(chunker, "window_size", "Dynamic"))))),
            "Overlap": getattr(chunker, "overlap", getattr(chunker, "chunk_overlap", getattr(chunker, "stride", "Dynamic"))),
            "Recall@5": avg_r5,
            "Recall@10": avg_r10,
            "Precision@5": avg_p5,
            "MRR": avg_mrr,
            "NDCG@10": avg_ndcg10,
            "Latency(ms)": avg_latency,
            "Overall Score": overall_score
        })
        
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Overall Score", ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = "Rank"
        
        os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
        df.to_csv(LEADERBOARD_PATH)
        logger.info(f"Leaderboard saved to {LEADERBOARD_PATH}")
        print("\n=== Chunking Leaderboard ===")
        print(df)
    else:
        logger.warning("No results to build leaderboard.")

if __name__ == "__main__":
    run_evaluation_pipeline()
