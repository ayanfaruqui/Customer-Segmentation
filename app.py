"""
🛒 Customer Segmentation — Streamlit Web Application

A premium, interactive dashboard for exploring mall customer segments
using K-Means Clustering. Features EDA, model training, cluster
visualization, and real-time prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import load_data, encode_gender, select_features, scale_features
from src.model import (
    compute_elbow,
    compute_silhouette_scores,
    find_optimal_k,
    train_kmeans,
    get_cluster_labels,
    get_centroids,
    get_cluster_summary,
    predict_cluster,
    save_model,
    save_scaler,
    CLUSTER_NAMES,
    CLUSTER_DESCRIPTIONS,
)
from src.visualizations import (
    plot_elbow,
    plot_silhouette,
    plot_clusters_2d,
    plot_clusters_3d,
    plot_distribution,
    plot_gender_distribution,
    plot_cluster_distribution,
    plot_correlation_heatmap,
    plot_scatter_matrix,
    plot_cluster_profile_radar,
)


# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Segmentation | K-Means ML",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium Dark Theme with Glassmorphism
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
    /* ── Import Google Fonts ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ──────────────────────────────────── */
    :root {
        --bg-primary: #0F0F1A;
        --bg-secondary: #1A1A2E;
        --bg-card: rgba(30, 30, 60, 0.6);
        --text-primary: #E8E8F0;
        --text-secondary: #9D9DB5;
        --accent-purple: #6C5CE7;
        --accent-green: #00B894;
        --accent-coral: #E17055;
        --accent-blue: #0984E3;
        --accent-gold: #FDCB6E;
        --gradient-1: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 100%);
        --gradient-2: linear-gradient(135deg, #00B894 0%, #55EFC4 100%);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    /* ── Global ──────────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* ── Glass Card ──────────────────────────────────────── */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(108, 92, 231, 0.15);
    }

    /* ── Metric Cards ────────────────────────────────────── */
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: var(--accent-purple);
        box-shadow: 0 4px 24px rgba(108, 92, 231, 0.2);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Hero Header ─────────────────────────────────────── */
    .hero-header {
        text-align: center;
        padding: 40px 20px 30px;
        margin-bottom: 30px;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 40%, #00B894 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Section Headers ─────────────────────────────────── */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 30px 0 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(108, 92, 231, 0.3);
    }

    /* ── Cluster Badge ───────────────────────────────────── */
    .cluster-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        margin: 4px;
    }

    /* ── Prediction Result ───────────────────────────────── */
    .prediction-result {
        background: linear-gradient(135deg, rgba(108, 92, 231, 0.2), rgba(0, 184, 148, 0.2));
        border: 1px solid rgba(108, 92, 231, 0.4);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }

    .prediction-cluster {
        font-size: 2rem;
        font-weight: 800;
        color: #A29BFE;
        margin-bottom: 10px;
    }

    .prediction-desc {
        font-size: 1rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F0F1A 0%, #1A1A2E 100%);
    }

    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
    }

    /* ── Tabs ────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }

    /* ── Divider ─────────────────────────────────────────── */
    .fancy-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-purple), transparent);
        margin: 30px 0;
        border: none;
    }

    /* ── Hide default Streamlit elements ─────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING & CACHING
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def get_data():
    """Load and cache the dataset."""
    df = load_data()
    df_encoded = encode_gender(df)
    return df, df_encoded


@st.cache_data
def run_clustering(n_clusters: int):
    """Run the full clustering pipeline and cache results."""
    df, df_encoded = get_data()
    X = select_features(df_encoded)
    X_scaled, scaler = scale_features(X)

    # Elbow & Silhouette
    elbow_data = compute_elbow(X_scaled)
    silhouette_data = compute_silhouette_scores(X_scaled)

    # Train model
    model = train_kmeans(X_scaled, n_clusters=n_clusters)
    labels = get_cluster_labels(model)

    # Get centroids in original scale
    centroids_scaled = get_centroids(model)
    centroids_original = scaler.inverse_transform(centroids_scaled)

    # Cluster summary
    summary = get_cluster_summary(df, labels)

    return {
        "df": df,
        "df_encoded": df_encoded,
        "X": X,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "model": model,
        "labels": labels,
        "centroids_original": centroids_original,
        "elbow_data": elbow_data,
        "silhouette_data": silhouette_data,
        "cluster_summary": summary,
    }


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 3rem;">🛒</div>
            <div style="font-size: 1.3rem; font-weight: 700; 
                        background: linear-gradient(135deg, #6C5CE7, #00B894);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;">
                Customer Segmentation
            </div>
            <div style="font-size: 0.8rem; color: #9D9DB5; margin-top: 5px;">
                K-Means Clustering ML Project
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "🧭 Navigation",
        [
            "🏠 Overview",
            "📊 Exploratory Analysis",
            "🔬 Model Training",
            "🎯 Predict Segment",
            "📋 Cluster Profiles",
        ],
        index=0,
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    n_clusters = st.slider(
        "🔢 Number of Clusters (K)",
        min_value=2,
        max_value=10,
        value=5,
        help="Adjust K to see how clusters change. K=5 is optimal for this dataset.",
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0; font-size: 0.75rem; color: #666;">
            Built with ❤️ using<br>
            <strong>Scikit-learn</strong> · <strong>Streamlit</strong> · <strong>Plotly</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA & RUN MODEL
# ══════════════════════════════════════════════════════════════════════
results = run_clustering(n_clusters)

df = results["df"]
df_encoded = results["df_encoded"]
X = results["X"]
X_scaled = results["X_scaled"]
scaler = results["scaler"]
model = results["model"]
labels = results["labels"]
centroids_original = results["centroids_original"]
elbow_data = results["elbow_data"]
silhouette_data = results["silhouette_data"]
cluster_summary = results["cluster_summary"]


# ══════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    # Hero
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-title">🛒 Customer Segmentation</div>
            <div class="hero-subtitle">
                Discover hidden customer groups using K-Means Clustering.<br>
                Analyze spending patterns, income levels, and demographics<br>
                to create targeted marketing strategies.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Total Customers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{n_clusters}</div>
                <div class="metric-label">Clusters Found</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">${df['Annual Income (k$)'].mean():.0f}K</div>
                <div class="metric-label">Avg Income</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{df['Spending Score (1-100)'].mean():.0f}</div>
                <div class="metric-label">Avg Spending Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Quick cluster preview
    st.markdown('<div class="section-header">🔮 Quick Cluster Preview</div>', unsafe_allow_html=True)
    fig_2d = plot_clusters_2d(df, labels, centroids_original)
    st.plotly_chart(fig_2d, use_container_width=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Dataset preview
    st.markdown('<div class="section-header">📦 Dataset Preview</div>', unsafe_allow_html=True)

    with st.expander("🔽 View Raw Data", expanded=False):
        st.dataframe(
            df.style.background_gradient(cmap="viridis", subset=["Annual Income (k$)", "Spending Score (1-100)"]),
            use_container_width=True,
            height=400,
        )

    # Stats
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color: #A29BFE;">🎯 About This Project</h4>
                <p style="color: #9D9DB5; line-height: 1.8;">
                    This project uses <strong>K-Means Clustering</strong>, an unsupervised 
                    machine learning algorithm, to segment mall customers into distinct 
                    groups based on their <strong>Annual Income</strong> and 
                    <strong>Spending Score</strong>. Unlike supervised learning, the model 
                    discovers hidden patterns without labeled data.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color: #00B894;">⚙️ How K-Means Works</h4>
                <p style="color: #9D9DB5; line-height: 1.8;">
                    <strong>1.</strong> Initialize K centroids randomly (k-means++).<br>
                    <strong>2.</strong> Assign each point to the nearest centroid.<br>
                    <strong>3.</strong> Recalculate centroids as cluster means.<br>
                    <strong>4.</strong> Repeat steps 2-3 until convergence.<br>
                    <strong>5.</strong> Final clusters represent customer segments.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════
# PAGE: EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.markdown(
        '<div class="section-header">📊 Exploratory Data Analysis</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "👥 Gender Analysis", "🔗 Correlations", "🔍 Pair Analysis"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_distribution(df, "Age"), use_container_width=True)
        with col2:
            st.plotly_chart(
                plot_distribution(df, "Annual Income (k$)"), use_container_width=True
            )

        st.plotly_chart(
            plot_distribution(df, "Spending Score (1-100)"), use_container_width=True
        )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_gender_distribution(df), use_container_width=True)
        with col2:
            # Gender stats
            gender_stats = df.groupby("Gender").agg(
                {
                    "Age": "mean",
                    "Annual Income (k$)": "mean",
                    "Spending Score (1-100)": "mean",
                }
            ).round(1)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Gender-wise Averages")
            st.dataframe(gender_stats, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.plotly_chart(plot_correlation_heatmap(df_encoded), use_container_width=True)

        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color: #FDCB6E;">💡 Key Insight</h4>
                <p style="color: #9D9DB5;">
                    The correlation heatmap reveals that Annual Income and Spending Score 
                    have very low correlation, meaning high income doesn't necessarily 
                    translate to high spending — making these features ideal for 
                    customer segmentation!
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab4:
        st.plotly_chart(plot_scatter_matrix(df), use_container_width=True)

        # Descriptive stats table
        st.markdown("#### 📋 Descriptive Statistics")
        st.dataframe(
            df.describe().round(2).style.background_gradient(cmap="viridis"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════
elif page == "🔬 Model Training":
    st.markdown(
        '<div class="section-header">🔬 Model Training & Evaluation</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["📐 Optimal K", "🔮 Cluster Visualization", "📈 Cluster Stats"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_elbow(elbow_data), use_container_width=True)
        with col2:
            st.plotly_chart(plot_silhouette(silhouette_data), use_container_width=True)

        optimal_k = find_optimal_k(silhouette_data)

        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <h3 style="color: #A29BFE;">🏆 Recommended K = {optimal_k}</h3>
                <p style="color: #9D9DB5;">
                    Based on the <strong>Elbow Method</strong> (look for the bend) and the 
                    <strong>Silhouette Score</strong> (highest = {max(silhouette_data['silhouette_scores']):.4f}), 
                    the optimal number of clusters is <strong>{optimal_k}</strong>.<br>
                    Currently training with <strong>K = {n_clusters}</strong> 
                    (adjust in sidebar).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        st.plotly_chart(
            plot_clusters_2d(df, labels, centroids_original), use_container_width=True
        )

        st.plotly_chart(plot_clusters_3d(df, labels), use_container_width=True)

    with tab3:
        st.plotly_chart(
            plot_cluster_distribution(labels), use_container_width=True
        )

        st.plotly_chart(
            plot_cluster_profile_radar(cluster_summary), use_container_width=True
        )

        # Summary table
        st.markdown("#### 📊 Cluster Summary Table")
        summary_rows = []
        for cid, stats in cluster_summary.items():
            summary_rows.append(
                {
                    "Cluster": cid,
                    "Count": stats["count"],
                    "Avg Age": stats["avg_age"],
                    "Avg Income (k$)": stats["avg_income"],
                    "Avg Spending Score": stats["avg_spending"],
                    "Male": stats["gender_split"].get("Male", 0),
                    "Female": stats["gender_split"].get("Female", 0),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df.style.background_gradient(
                cmap="viridis",
                subset=["Avg Income (k$)", "Avg Spending Score"],
            ),
            use_container_width=True,
        )

    # Save model button
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    col_save1, col_save2, col_save3 = st.columns([1, 2, 1])
    with col_save2:
        if st.button("💾 Save Trained Model", use_container_width=True, type="primary"):
            model_path = save_model(model)
            scaler_path = save_scaler(scaler)
            st.success(f"✅ Model saved to `{model_path}`")
            st.success(f"✅ Scaler saved to `{scaler_path}`")


# ══════════════════════════════════════════════════════════════════════
# PAGE: PREDICT SEGMENT
# ══════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Segment":
    st.markdown(
        '<div class="section-header">🎯 Predict Your Customer Segment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <p style="color: #9D9DB5; font-size: 1rem;">
                Enter a customer's details below and the trained K-Means model will 
                predict which segment they belong to — in real time!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        input_gender = st.selectbox("👤 Gender", ["Male", "Female"])
        input_age = st.slider("🎂 Age", min_value=18, max_value=70, value=30)

    with col2:
        input_income = st.slider(
            "💰 Annual Income (k$)", min_value=15, max_value=140, value=60
        )
        input_spending = st.slider(
            "🛍️ Spending Score (1-100)", min_value=1, max_value=100, value=50
        )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    if st.button("🔮 Predict Segment", use_container_width=True, type="primary"):
        cluster_id = predict_cluster(model, scaler, input_income, input_spending)

        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        cluster_desc = CLUSTER_DESCRIPTIONS.get(
            cluster_id, "A distinct customer group with unique characteristics."
        )

        st.markdown(
            f"""
            <div class="prediction-result">
                <div style="font-size: 1rem; color: #9D9DB5; margin-bottom: 10px;">
                    This customer belongs to:
                </div>
                <div class="prediction-cluster">{cluster_name}</div>
                <div class="prediction-desc">{cluster_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show where this customer falls on the chart
        st.markdown("#### 📍 Customer Placement on Cluster Map")

        import plotly.graph_objects as go

        fig = plot_clusters_2d(df, labels, centroids_original)

        # Add the new customer point
        fig.add_trace(
            go.Scatter(
                x=[input_income],
                y=[input_spending],
                mode="markers+text",
                marker=dict(
                    size=22,
                    color="#FF6B6B",
                    symbol="star",
                    line=dict(width=3, color="#FFF"),
                ),
                text=["⭐ YOU"],
                textposition="top center",
                textfont=dict(size=14, color="#FF6B6B"),
                name="Your Customer",
                hovertemplate=(
                    f"Your Customer<br>"
                    f"Income: ${input_income}K<br>"
                    f"Spending Score: {input_spending}<br>"
                    f"Predicted: {cluster_name}<extra></extra>"
                ),
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Cluster comparison
        if cluster_id in cluster_summary:
            stats = cluster_summary[cluster_id]
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("📊 Cluster Size", f"{stats['count']} customers")
            with col_b:
                st.metric("💰 Avg Income", f"${stats['avg_income']}K")
            with col_c:
                st.metric("🛍️ Avg Spending", f"{stats['avg_spending']}")


# ══════════════════════════════════════════════════════════════════════
# PAGE: CLUSTER PROFILES
# ══════════════════════════════════════════════════════════════════════
elif page == "📋 Cluster Profiles":
    st.markdown(
        '<div class="section-header">📋 Detailed Cluster Profiles</div>',
        unsafe_allow_html=True,
    )

    # Radar chart overview
    st.plotly_chart(
        plot_cluster_profile_radar(cluster_summary), use_container_width=True
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Individual cluster cards
    for cluster_id, stats in cluster_summary.items():
        name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        desc = CLUSTER_DESCRIPTIONS.get(cluster_id, "")
        color = [
            "#6C5CE7",
            "#00B894",
            "#E17055",
            "#0984E3",
            "#FDCB6E",
            "#E84393",
            "#00CEC9",
        ][cluster_id % 7]

        male_count = stats["gender_split"].get("Male", 0)
        female_count = stats["gender_split"].get("Female", 0)

        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <h3 style="color: {color}; margin: 0;">{name}</h3>
                        <p style="color: #9D9DB5; margin-top: 8px; max-width: 500px;">{desc}</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.5rem; font-weight: 800; color: {color};">
                            {stats['count']}
                        </div>
                        <div style="color: #9D9DB5; font-size: 0.8rem;">customers</div>
                    </div>
                </div>
                <div style="display: flex; gap: 30px; margin-top: 15px; flex-wrap: wrap;">
                    <div>
                        <span style="color: #9D9DB5;">🎂 Avg Age:</span>
                        <strong style="color: #E8E8F0;"> {stats['avg_age']}</strong>
                    </div>
                    <div>
                        <span style="color: #9D9DB5;">💰 Avg Income:</span>
                        <strong style="color: #E8E8F0;"> ${stats['avg_income']}K</strong>
                    </div>
                    <div>
                        <span style="color: #9D9DB5;">🛍️ Avg Spending:</span>
                        <strong style="color: #E8E8F0;"> {stats['avg_spending']}</strong>
                    </div>
                    <div>
                        <span style="color: #9D9DB5;">👨 Male:</span>
                        <strong style="color: #E8E8F0;"> {male_count}</strong>
                        <span style="color: #9D9DB5; margin-left: 10px;">👩 Female:</span>
                        <strong style="color: #E8E8F0;"> {female_count}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Marketing recommendations
    st.markdown(
        '<div class="section-header">💡 Marketing Recommendations</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <th style="padding: 12px; text-align: left; color: #A29BFE;">Segment</th>
                    <th style="padding: 12px; text-align: left; color: #A29BFE;">Strategy</th>
                    <th style="padding: 12px; text-align: left; color: #A29BFE;">Channel</th>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px; color: #6C5CE7;">💰 High Income, High Spenders</td>
                    <td style="padding: 12px; color: #9D9DB5;">VIP loyalty program, exclusive launches</td>
                    <td style="padding: 12px; color: #9D9DB5;">Personal emails, premium events</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px; color: #00B894;">🎯 Moderate Spenders</td>
                    <td style="padding: 12px; color: #9D9DB5;">Seasonal deals, bundle offers</td>
                    <td style="padding: 12px; color: #9D9DB5;">Social media, newsletters</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px; color: #E17055;">📉 High Income, Low Spenders</td>
                    <td style="padding: 12px; color: #9D9DB5;">Personalized recommendations, quality focus</td>
                    <td style="padding: 12px; color: #9D9DB5;">Targeted ads, in-store experience</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px; color: #0984E3;">🛍️ Low Income, High Spenders</td>
                    <td style="padding: 12px; color: #9D9DB5;">Reward points, flash sales, discounts</td>
                    <td style="padding: 12px; color: #9D9DB5;">Mobile app push, SMS alerts</td>
                </tr>
                <tr>
                    <td style="padding: 12px; color: #FDCB6E;">💵 Low Income, Low Spenders</td>
                    <td style="padding: 12px; color: #9D9DB5;">Budget-friendly bundles, essentials</td>
                    <td style="padding: 12px; color: #9D9DB5;">In-store signage, flyers</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
