import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="RAG Evaluation Leaderboards", layout="wide")

CHUNKING_LEADERBOARD_PATH = "leaderboard/chunking_leaderboard.csv"
EMBEDDING_LEADERBOARD_PATH = "leaderboard/embedding_leaderboard.csv"

def load_data(path, sort_col="Overall Score"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    return df

st.title("🏆 RAG Evaluation Leaderboards")
st.markdown("""
This dashboard evaluates different components of a Retrieval-Augmented Generation (RAG) pipeline.
""")

tab1, tab2, tab3 = st.tabs(["Chunking Strategies", "Embedding Models", "Retrieval Algorithms"])

with tab1:
    st.header("🧩 Chunking Strategy Leaderboard")
    st.markdown("Evaluating chunking strategies using a fixed `all-MiniLM-L6-v2` embedding model.")
    
    df_chunk = load_data(CHUNKING_LEADERBOARD_PATH)
    
    if df_chunk is None or df_chunk.empty:
        st.warning("Chunking leaderboard data not found. Please run the chunking evaluation pipeline first.")
    else:
        st.sidebar.header("Chunking Filters")
        strategies = ["All"] + df_chunk["Chunk Strategy"].unique().tolist()
        selected_strategy = st.sidebar.selectbox("Filter by Strategy Name", strategies, key="chunk_filter")
        
        filtered_df_chunk = df_chunk[df_chunk["Chunk Strategy"] == selected_strategy] if selected_strategy != "All" else df_chunk
            
        best_strategy = df_chunk.iloc[0]
        st.info(f"**{best_strategy['Chunk Strategy']}** is currently leading with an overall score of **{best_strategy['Overall Score']:.4f}**!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recall@10", f"{best_strategy['Recall@10']:.4f}")
        col2.metric("Precision@5", f"{best_strategy['Precision@5']:.4f}")
        col3.metric("MRR", f"{best_strategy['MRR']:.4f}")
        col4.metric("NDCG@10", f"{best_strategy['NDCG@10']:.4f}")

        st.subheader("📋 Leaderboard Table")
        st.dataframe(
            filtered_df_chunk.style.highlight_max(subset=['Overall Score', 'Recall@10', 'Recall@5', 'MRR', 'NDCG@10', 'Precision@5'], color='lightgreen')
                             .highlight_min(subset=['Latency(ms)'], color='lightgreen'),
            width='stretch'
        )

        st.subheader("📊 Performance Comparison")
        metric_to_plot_chunk = st.selectbox("Select Metric to Visualize", ["Overall Score", "Recall@10", "MRR", "NDCG@10", "Precision@5", "Latency(ms)"], key="chunk_metric")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df_chunk = filtered_df_chunk.sort_values(by=metric_to_plot_chunk, ascending=True)
        colors = ['#1f77b4' if stat != best_strategy['Chunk Strategy'] else '#ff7f0e' for stat in plot_df_chunk['Chunk Strategy']]
        ax.barh(plot_df_chunk['Chunk Strategy'], plot_df_chunk[metric_to_plot_chunk], color=colors)
        ax.set_xlabel(metric_to_plot_chunk)
        ax.set_title(f"Comparison of {metric_to_plot_chunk} across Chunking Strategies")
        for i, v in enumerate(plot_df_chunk[metric_to_plot_chunk]):
            ax.text(v, i, f" {v:.4f}", va='center')
        st.pyplot(fig)


with tab2:
    st.header("🧠 Embedding Model Leaderboard")
    st.markdown("Evaluating embedding models using the fixed optimal chunking strategy (`sliding_10_2`).")
    
    df_emb = load_data(EMBEDDING_LEADERBOARD_PATH)
    
    if df_emb is None or df_emb.empty:
        st.warning("Embedding leaderboard data not found. Please run the embedding evaluation pipeline first.")
    else:
        st.sidebar.header("Embedding Filters")
        models = ["All"] + df_emb["Embedding Model"].unique().tolist()
        selected_model = st.sidebar.selectbox("Filter by Model Name", models, key="emb_filter")
        
        filtered_df_emb = df_emb[df_emb["Embedding Model"] == selected_model] if selected_model != "All" else df_emb
            
        best_model = df_emb.iloc[0]
        st.info(f"**{best_model['Embedding Model']}** is currently leading with an overall score of **{best_model['Overall Score']:.4f}**!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recall@10", f"{best_model['Recall@10']:.4f}")
        col2.metric("MRR", f"{best_model['MRR']:.4f}")
        col3.metric("Throughput (req/sec)", f"{best_model['Throughput(req/sec)']:.2f}")
        col4.metric("Memory Usage (MB)", f"{best_model['Memory Usage(MB)']:.2f}")

        st.subheader("📋 Leaderboard Table")
        st.dataframe(
            filtered_df_emb.style.highlight_max(subset=['Overall Score', 'Recall@10', 'Recall@5', 'MRR', 'NDCG@10', 'Precision@5', 'Throughput(req/sec)'], color='lightgreen')
                             .highlight_min(subset=['Embedding Latency(ms)', 'Retrieval Latency(ms)', 'Memory Usage(MB)'], color='lightgreen'),
            width='stretch'
        )

        st.subheader("📊 Performance Comparison")
        metric_to_plot_emb = st.selectbox("Select Metric to Visualize", ["Overall Score", "Recall@10", "MRR", "Throughput(req/sec)", "Retrieval Latency(ms)", "Memory Usage(MB)", "Embedding Dimension"], key="emb_metric")
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        plot_df_emb = filtered_df_emb.sort_values(by=metric_to_plot_emb, ascending=True)
        colors2 = ['#2ca02c' if model != best_model['Embedding Model'] else '#d62728' for model in plot_df_emb['Embedding Model']]
        ax2.barh(plot_df_emb['Embedding Model'], plot_df_emb[metric_to_plot_emb], color=colors2)
        ax2.set_xlabel(metric_to_plot_emb)
        ax2.set_title(f"Comparison of {metric_to_plot_emb} across Embedding Models")
        for i, v in enumerate(plot_df_emb[metric_to_plot_emb]):
            if "Latency" in metric_to_plot_emb or "Memory" in metric_to_plot_emb or "Dimension" in metric_to_plot_emb or "Throughput" in metric_to_plot_emb:
                ax2.text(v, i, f" {v:.2f}", va='center')
            else:
                ax2.text(v, i, f" {v:.4f}", va='center')
        st.pyplot(fig2)

with tab3:
    st.header("🔍 Retrieval Search Algorithms")
    st.markdown("Evaluating retrieval algorithms using the optimal chunking (`sliding_10_2`) and embedding (`e5-small`).")
    
    RETRIEVAL_LEADERBOARD_PATH = "leaderboard/retrieval_leaderboard.csv"
    df_ret = load_data(RETRIEVAL_LEADERBOARD_PATH)
    
    if df_ret is None or df_ret.empty:
        st.warning("Retrieval leaderboard data not found. Please run the retrieval evaluation pipeline first.")
    else:
        st.sidebar.header("Retrieval Filters")
        retrievers = ["All"] + df_ret["Retriever"].unique().tolist()
        selected_retriever = st.sidebar.selectbox("Filter by Retriever Name", retrievers, key="ret_filter")
        
        filtered_df_ret = df_ret[df_ret["Retriever"] == selected_retriever] if selected_retriever != "All" else df_ret
            
        best_ret = df_ret.iloc[0]
        st.info(f"**{best_ret['Retriever']}** is currently leading with an overall score of **{best_ret['Overall Score']:.4f}**!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recall@10", f"{best_ret['Recall@10']:.4f}")
        col2.metric("MRR", f"{best_ret['MRR']:.4f}")
        col3.metric("Retrieval Latency(ms)", f"{best_ret['Retrieval Latency(ms)']:.2f}")
        col4.metric("Memory Usage (MB)", f"{best_ret['Memory Usage(MB)']:.2f}")

        st.subheader("📋 Leaderboard Table")
        st.dataframe(
            filtered_df_ret.style.highlight_max(subset=['Overall Score', 'Recall@10', 'Recall@5', 'MRR', 'NDCG@10', 'Precision@5', 'Throughput(req/sec)'], color='lightgreen')
                             .highlight_min(subset=['Index Build Time(ms)', 'Retrieval Latency(ms)', 'Memory Usage(MB)'], color='lightgreen'),
            width='stretch'
        )

        st.subheader("📊 Performance Comparison")
        metric_to_plot_ret = st.selectbox("Select Metric to Visualize", ["Overall Score", "Recall@10", "MRR", "Throughput(req/sec)", "Retrieval Latency(ms)", "Index Build Time(ms)", "Memory Usage(MB)"], key="ret_metric")
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        plot_df_ret = filtered_df_ret.sort_values(by=metric_to_plot_ret, ascending=True)
        colors3 = ['#9467bd' if r != best_ret['Retriever'] else '#e377c2' for r in plot_df_ret['Retriever']]
        ax3.barh(plot_df_ret['Retriever'], plot_df_ret[metric_to_plot_ret], color=colors3)
        ax3.set_xlabel(metric_to_plot_ret)
        ax3.set_title(f"Comparison of {metric_to_plot_ret} across Retrievers")
        for i, v in enumerate(plot_df_ret[metric_to_plot_ret]):
            if "Latency" in metric_to_plot_ret or "Memory" in metric_to_plot_ret or "Time" in metric_to_plot_ret or "Throughput" in metric_to_plot_ret:
                ax3.text(v, i, f" {v:.2f}", va='center')
            else:
                ax3.text(v, i, f" {v:.4f}", va='center')
        st.pyplot(fig3)

