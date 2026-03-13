"""
RAG Architecture Comparison Dashboard — Streamlit App

Visualizes results from the experiment runner.
Loads data from: logs/results.csv and logs/summary_metrics.csv

Run: streamlit run dashboard/app.py
"""
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from shared_config import RESULTS_CSV, SUMMARY_CSV

# ── Page Config ──
st.set_page_config(
    page_title="RAG Benchmark Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color Palette ──
COLORS = {
    "classic_rag": "#636EFA",
    "summarization_rag": "#EF553B",
    "scaledown_rag": "#00CC96",
}
LABELS = {
    "classic_rag": "Classic RAG",
    "summarization_rag": "RAG + Summarization",
    "scaledown_rag": "ScaleDown RAG",
}


def load_data():
    """Load results CSV files."""
    results_df = None
    summary_df = None

    if os.path.exists(RESULTS_CSV):
        results_df = pd.read_csv(RESULTS_CSV)
    if os.path.exists(SUMMARY_CSV):
        summary_df = pd.read_csv(SUMMARY_CSV)

    return results_df, summary_df



def render_kpi_cards(summary_df, results_df):
    """Render top-level KPI cards."""
    st.markdown("### 📊 Summary Metrics")

    if summary_df is not None and not summary_df.empty:
        data = summary_df
    else:
        # Compute from results
        data = results_df.groupby("pipeline").agg(
            avg_latency=("total_latency", "mean"),
            p95_latency=("total_latency", lambda x: np.percentile(x, 95)),
            avg_quality=("quality_score", "mean"),
            avg_compression_ratio=("compression_ratio", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            num_queries=("query", "count"),
        ).reset_index()

    cols = st.columns(len(data))
    for i, (_, row) in enumerate(data.iterrows()):
        pipe = row["pipeline"]
        label = LABELS.get(pipe, pipe)
        color = COLORS.get(pipe, "#888")

        with cols[i]:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {color}22, {color}11);
                            border-left: 4px solid {color};
                            padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{label}</h4>
                    <p style="margin: 0.2rem 0; font-size: 0.9rem;">
                        ⏱ Avg Latency: <b>{row.get('avg_latency', 0):.2f}s</b>
                    </p>
                    <p style="margin: 0.2rem 0; font-size: 0.9rem;">
                        ⏱ P95 Latency: <b>{row.get('p95_latency', 0):.2f}s</b>
                    </p>
                    <p style="margin: 0.2rem 0; font-size: 0.9rem;">
                        ⭐ Avg Quality: <b>{row.get('avg_quality', 0):.3f}</b>
                    </p>
                    <p style="margin: 0.2rem 0; font-size: 0.9rem;">
                        📦 Compression: <b>{row.get('avg_compression_ratio', 1.0):.2f}x</b>
                    </p>
                    <p style="margin: 0.2rem 0; font-size: 0.9rem;">
                        🪙 Avg Tokens: <b>{int(row.get('avg_total_tokens', 0))}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_latency_charts(results_df):
    """Render latency comparison charts."""
    st.markdown("### ⏱ Latency Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart: Average latency breakdown
        latency_data = results_df.groupby("pipeline").agg(
            retrieval=("retrieval_time", "mean"),
            generation=("generation_time", "mean"),
        ).reset_index()
        latency_data["pipeline"] = latency_data["pipeline"].map(lambda x: LABELS.get(x, x))

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Retrieval", x=latency_data["pipeline"], y=latency_data["retrieval"],
                             marker_color="#636EFA"))
        fig.add_trace(go.Bar(name="Generation", x=latency_data["pipeline"], y=latency_data["generation"],
                             marker_color="#00CC96"))
        fig.update_layout(
            title="Average Latency Breakdown",
            barmode="stack",
            yaxis_title="Time (seconds)",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot: Total latency distribution
        plot_df = results_df.copy()
        plot_df["pipeline_label"] = plot_df["pipeline"].map(lambda x: LABELS.get(x, x))
        fig = px.box(
            plot_df, x="pipeline_label", y="total_latency",
            color="pipeline_label",
            color_discrete_sequence=list(COLORS.values()),
            title="Latency Distribution",
            labels={"total_latency": "Total Latency (s)", "pipeline_label": "Pipeline"},
            template="plotly_dark",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_token_charts(results_df):
    """Render token usage comparison charts."""
    st.markdown("### 🪙 Token Usage Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart: Avg token counts
        token_data = results_df.groupby("pipeline").agg(
            input=("input_tokens", "mean"),
            output=("output_tokens", "mean"),
        ).reset_index()
        token_data["pipeline"] = token_data["pipeline"].map(lambda x: LABELS.get(x, x))

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Input Tokens", x=token_data["pipeline"], y=token_data["input"],
                             marker_color="#636EFA"))
        fig.add_trace(go.Bar(name="Output Tokens", x=token_data["pipeline"], y=token_data["output"],
                             marker_color="#AB63FA"))
        fig.update_layout(
            title="Average Token Usage",
            barmode="group",
            yaxis_title="Tokens",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bar chart: Original vs Optimized tokens
        comp_data = results_df.groupby("pipeline").agg(
            original=("original_tokens", "mean"),
            optimized=("optimized_tokens", "mean"),
        ).reset_index()
        comp_data["pipeline"] = comp_data["pipeline"].map(lambda x: LABELS.get(x, x))

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Original Tokens", x=comp_data["pipeline"], y=comp_data["original"],
                             marker_color="#FFA15A"))
        fig.add_trace(go.Bar(name="Optimized Tokens", x=comp_data["pipeline"], y=comp_data["optimized"],
                             marker_color="#00CC96"))
        fig.update_layout(
            title="Context Compression Effect",
            barmode="group",
            yaxis_title="Tokens",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_quality_charts(results_df):
    """Render quality comparison charts."""
    st.markdown("### ⭐ Quality Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart: Average quality
        qual_data = results_df.groupby("pipeline").agg(
            avg_quality=("quality_score", "mean"),
        ).reset_index()
        qual_data["pipeline_label"] = qual_data["pipeline"].map(lambda x: LABELS.get(x, x))

        fig = px.bar(
            qual_data, x="pipeline_label", y="avg_quality",
            color="pipeline_label",
            color_discrete_sequence=list(COLORS.values()),
            title="Average Answer Quality (0-1)",
            labels={"avg_quality": "Quality Score", "pipeline_label": "Pipeline"},
            template="plotly_dark",
        )
        fig.update_layout(height=400, showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Radar chart: Multi-dimensional comparison
        metrics = results_df.groupby("pipeline").agg(
            quality=("quality_score", "mean"),
            speed=("total_latency", lambda x: 1 / (1 + x.mean())),  # invert: lower is better
            compression=("compression_ratio", "mean"),
            token_efficiency=("total_tokens", lambda x: 1 / (1 + x.mean() / 1000)),  # invert
        ).reset_index()

        fig = go.Figure()
        categories = ["Quality", "Speed", "Compression", "Efficiency"]
        for _, row in metrics.iterrows():
            pipe = row["pipeline"]
            values = [row["quality"], row["speed"], row["compression"] / 5, row["token_efficiency"]]
            values.append(values[0])  # Close the shape
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=LABELS.get(pipe, pipe),
                line_color=COLORS.get(pipe, "#888"),
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Multi-Dimensional Comparison",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_compression_chart(results_df):
    """Render compression ratio comparison."""
    st.markdown("### 📦 Compression Ratio")

    comp_data = results_df.groupby("pipeline").agg(
        avg_ratio=("compression_ratio", "mean"),
    ).reset_index()
    comp_data["pipeline_label"] = comp_data["pipeline"].map(lambda x: LABELS.get(x, x))

    fig = px.bar(
        comp_data, x="pipeline_label", y="avg_ratio",
        color="pipeline_label",
        color_discrete_sequence=list(COLORS.values()),
        title="Average Compression Ratio (higher = more compression)",
        labels={"avg_ratio": "Compression Ratio", "pipeline_label": "Pipeline"},
        template="plotly_dark",
    )
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_per_query_table(results_df):
    """Render detailed per-query results table."""
    st.markdown("### 📋 Detailed Per-Query Results")

    # Pipeline filter
    pipelines = results_df["pipeline"].unique().tolist()
    selected = st.multiselect("Filter by pipeline:", pipelines, default=pipelines,
                              format_func=lambda x: LABELS.get(x, x))

    filtered = results_df[results_df["pipeline"].isin(selected)]

    # Display table
    display_cols = [
        "pipeline", "query", "total_latency", "retrieval_time",
        "compression_time", "generation_time",
        "total_tokens", "compression_ratio", "quality_score",
    ]
    available_cols = [c for c in display_cols if c in filtered.columns]

    styled = filtered[available_cols].copy()
    styled["pipeline"] = styled["pipeline"].map(lambda x: LABELS.get(x, x))
    styled = styled.rename(columns={
        "pipeline": "Pipeline",
        "query": "Query",
        "total_latency": "Total (s)",
        "retrieval_time": "Retrieval (s)",
        "compression_time": "Compress (s)",
        "generation_time": "Generate (s)",
        "total_tokens": "Tokens",
        "compression_ratio": "Compression",
        "quality_score": "Quality",
    })

    st.dataframe(styled, use_container_width=True, height=400)


def main():
    # ── Header ──
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="margin: 0;">🔬 RAG Architecture Comparison</h1>
            <p style="color: #888; font-size: 1.1rem; margin-top: 0.3rem;">
                Benchmarking Classic RAG vs Summarization vs ScaleDown Compression
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")
        st.markdown("### 🧪 Pipelines")
        st.markdown("""
        - 🔵 **Classic RAG**: Standard retrieve → generate
        - 🔴 **Summarization**: Map-reduce summaries → generate  
        - 🟢 **ScaleDown**: Context compression → generate
        """)

        st.markdown("---")
        st.markdown("### 📝 Experiment Config")
        st.markdown("""
        - **Chunk Size:** 500
        - **Chunk Overlap:** 50
        - **Top-K:** 5
        - **Embeddings:** all-MiniLM-L6-v2
        - **Vector Store:** FAISS
        """)

    # ── Load Data ──
    results_df, summary_df = load_data()
    if results_df is None:
        st.warning("⚠️ No experiment results found. Run the experiment first:")
        st.code("python experiments/run_all_pipelines.py", language="bash")
        st.stop()

    # ── Dashboard Sections ──
    render_kpi_cards(summary_df, results_df)
    st.markdown("---")
    render_latency_charts(results_df)
    st.markdown("---")
    render_token_charts(results_df)
    st.markdown("---")
    render_quality_charts(results_df)
    st.markdown("---")
    render_compression_chart(results_df)
    st.markdown("---")
    render_per_query_table(results_df)

    # Footer
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 1rem; color: #666;">
            <p>RAG Benchmark Platform — Built for Research Comparison</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
