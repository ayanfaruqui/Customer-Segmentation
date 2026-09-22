"""
Visualization Module for Customer Segmentation.

All plotting functions return Plotly figure objects for
interactive rendering in Streamlit and Jupyter notebooks.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ── Color Palette ────────────────────────────────────────────────────
CLUSTER_COLORS = [
    "#6C5CE7",  # Purple
    "#00B894",  # Mint Green
    "#E17055",  # Coral
    "#0984E3",  # Blue
    "#FDCB6E",  # Gold
    "#E84393",  # Pink
    "#00CEC9",  # Teal
    "#D63031",  # Red
]

PLOTLY_TEMPLATE = "plotly_dark"


def plot_elbow(elbow_data: dict) -> go.Figure:
    """
    Create an interactive Elbow Method plot.

    Args:
        elbow_data: Dict with 'k_values' and 'wcss'.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=elbow_data["k_values"],
            y=elbow_data["wcss"],
            mode="lines+markers",
            marker=dict(size=12, color="#6C5CE7", symbol="diamond"),
            line=dict(width=3, color="#6C5CE7"),
            name="WCSS",
            hovertemplate="K=%{x}<br>WCSS=%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="📐 Elbow Method — Optimal K Selection",
            font=dict(size=22, color="#FFFFFF"),
        ),
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Within-Cluster Sum of Squares (WCSS)",
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2BEC3"),
        xaxis=dict(dtick=1, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        height=450,
    )

    return fig


def plot_silhouette(silhouette_data: dict) -> go.Figure:
    """
    Create an interactive Silhouette Score bar chart.

    Args:
        silhouette_data: Dict with 'k_values' and 'silhouette_scores'.

    Returns:
        Plotly Figure.
    """
    k_values = silhouette_data["k_values"]
    scores = silhouette_data["silhouette_scores"]
    best_k = k_values[np.argmax(scores)]

    colors = [
        "#6C5CE7" if k == best_k else "rgba(108, 92, 231, 0.4)" for k in k_values
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=k_values,
            y=scores,
            marker=dict(color=colors, line=dict(width=2, color="#6C5CE7")),
            text=[f"{s:.4f}" for s in scores],
            textposition="outside",
            textfont=dict(size=12, color="#FFFFFF"),
            hovertemplate="K=%{x}<br>Score=%{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"🎯 Silhouette Scores — Best K = {best_k}",
            font=dict(size=22, color="#FFFFFF"),
        ),
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score",
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2BEC3"),
        xaxis=dict(dtick=1, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        height=450,
    )

    return fig


def plot_clusters_2d(
    df: pd.DataFrame,
    labels: np.ndarray,
    centroids: np.ndarray = None,
    feature_x: str = "Annual Income (k$)",
    feature_y: str = "Spending Score (1-100)",
) -> go.Figure:
    """
    Create a 2D scatter plot of customer clusters.

    Args:
        df: DataFrame with feature columns.
        labels: Cluster labels.
        centroids: Cluster centroids (scaled or original).
        feature_x: X-axis feature name.
        feature_y: Y-axis feature name.

    Returns:
        Plotly Figure.
    """
    plot_df = df.copy()
    plot_df["Cluster"] = labels.astype(str)

    fig = px.scatter(
        plot_df,
        x=feature_x,
        y=feature_y,
        color="Cluster",
        color_discrete_sequence=CLUSTER_COLORS,
        template=PLOTLY_TEMPLATE,
        hover_data=plot_df.columns.tolist(),
    )

    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="#FFF")))

    # Add centroids if provided (in original feature space)
    if centroids is not None:
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                marker=dict(
                    size=20,
                    color="#FFFFFF",
                    symbol="x",
                    line=dict(width=3, color="#2D3436"),
                ),
                name="Centroids",
                hovertemplate="Centroid<br>Income=%{x:.1f}<br>Spending=%{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="🔮 Customer Segments — 2D Cluster View",
            font=dict(size=22, color="#FFFFFF"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2BEC3"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        height=550,
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
    )

    return fig


def plot_clusters_3d(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> go.Figure:
    """
    Create a 3D scatter plot using Age, Income, and Spending Score.

    Args:
        df: DataFrame with Age, Income, and Spending Score columns.
        labels: Cluster labels.

    Returns:
        Plotly Figure.
    """
    plot_df = df.copy()
    plot_df["Cluster"] = labels.astype(str)

    fig = px.scatter_3d(
        plot_df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        z="Age",
        color="Cluster",
        color_discrete_sequence=CLUSTER_COLORS,
        template=PLOTLY_TEMPLATE,
        opacity=0.85,
    )

    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="#FFF")))

    fig.update_layout(
        title=dict(
            text="🌐 3D Customer Segments",
            font=dict(size=22, color="#FFFFFF"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2BEC3"),
        height=600,
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
        ),
    )

    return fig


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    nbins: int = 20,
) -> go.Figure:
    """
    Create a histogram for a single feature's distribution.

    Args:
        df: DataFrame.
        column: Column name.
        nbins: Number of bins.

    Returns:
        Plotly Figure.
    """
    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        color_discrete_sequence=["#6C5CE7"],
        template=PLOTLY_TEMPLATE,
        marginal="box",
    )

    fig.update_layout(
        title=dict(
            text=f"📊 Distribution of {column}",
            font=dict(size=20, color="#FFFFFF"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2BEC3"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        height=400,
    )

    return fig


def plot_gender_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create a donut chart of gender distribution.

    Args:
        df: DataFrame with 'Gender' column.

    Returns:
        Plotly Figure.
    """
    gender_counts = df["Gender"].value_counts()

    fig = go.Figure(
        go.Pie(
            labels=gender_counts.index,
            values=gender_counts.values,
            hole=0.55,
            marker=dict(colors=["#6C5CE7", "#00B894"], line=dict(color="#FFF", width=2)),
            textfont=dict(size=14),
            textinfo="label+percent",
        )
    )

    fig.update_layout(
        title=dict(
            text="👥 Gender Distribution",
            font=dict(size=20, color="#FFFFFF"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2BEC3"),
        height=400,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )

    return fig


def plot_cluster_distribution(labels: np.ndarray) -> go.Figure:
    """
    Create a pie chart showing cluster size distribution.

    Args:
        labels: Cluster labels array.

    Returns:
        Plotly Figure.
    """
    unique, counts = np.unique(labels, return_counts=True)

    fig = go.Figure(
        go.Pie(
            labels=[f"Cluster {c}" for c in unique],
            values=counts,
            hole=0.5,
            marker=dict(
                colors=CLUSTER_COLORS[: len(unique)],
                line=dict(color="#FFF", width=2),
            ),
            textfont=dict(size=13),
            textinfo="label+percent+value",
        )
    )

    fig.update_layout(
        title=dict(
            text="📈 Cluster Size Distribution",
            font=dict(size=20, color="#FFFFFF"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2BEC3"),
        height=400,
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )

    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive correlation heatmap.

    Args:
        df: DataFrame with numeric columns.

    Returns:
        Plotly Figure.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Viridis",
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=14, color="#FFFFFF"),
            hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="🔗 Feature Correlation Heatmap",
            font=dict(size=20, color="#FFFFFF"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2BEC3"),
        height=450,
        xaxis=dict(side="bottom"),
    )

    return fig


def plot_scatter_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Create a scatter matrix (pairplot-style) of numeric features.

    Args:
        df: DataFrame.

    Returns:
        Plotly Figure.
    """
    numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    fig = px.scatter_matrix(
        df,
        dimensions=numeric_cols,
        color="Gender",
        color_discrete_sequence=["#6C5CE7", "#00B894"],
        template=PLOTLY_TEMPLATE,
        opacity=0.7,
    )

    fig.update_traces(
        diagonal_visible=True,
        marker=dict(size=4, line=dict(width=0.5, color="#FFF")),
    )

    fig.update_layout(
        title=dict(
            text="🔍 Feature Pair Analysis",
            font=dict(size=20, color="#FFFFFF"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2BEC3"),
        height=650,
    )

    return fig


def plot_cluster_profile_radar(cluster_summary: dict) -> go.Figure:
    """
    Create a radar chart comparing cluster profiles.

    Args:
        cluster_summary: Dict from model.get_cluster_summary().

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    categories = ["Avg Age", "Avg Income", "Avg Spending"]

    for cluster_id, stats in cluster_summary.items():
        values = [stats["avg_age"], stats["avg_income"], stats["avg_spending"]]
        # Close the radar
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                fill="toself",
                name=f"Cluster {cluster_id}",
                line=dict(color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]),
                opacity=0.6,
            )
        )

    fig.update_layout(
        title=dict(
            text="🕸️ Cluster Profile Comparison",
            font=dict(size=20, color="#FFFFFF"),
        ),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.15)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.15)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2BEC3"),
        height=500,
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )

    return fig
