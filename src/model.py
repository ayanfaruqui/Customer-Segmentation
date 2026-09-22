"""
K-Means Clustering Model Module for Customer Segmentation.

Provides Elbow Method, Silhouette Analysis, model training,
serialization, and prediction utilities.
"""

import os
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_elbow(X_scaled: np.ndarray, k_range: range = None) -> dict:
    """
    Compute WCSS (inertia) for a range of K values — Elbow Method.

    Args:
        X_scaled: Scaled feature array.
        k_range: Range of K values to try. Defaults to range(2, 11).

    Returns:
        Dictionary with 'k_values' and 'wcss' lists.
    """
    if k_range is None:
        k_range = range(2, 11)

    wcss = []
    k_values = list(k_range)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    return {"k_values": k_values, "wcss": wcss}


def compute_silhouette_scores(
    X_scaled: np.ndarray, k_range: range = None
) -> dict:
    """
    Compute silhouette scores for a range of K values.

    Args:
        X_scaled: Scaled feature array.
        k_range: Range of K values. Defaults to range(2, 11).

    Returns:
        Dictionary with 'k_values' and 'silhouette_scores' lists.
    """
    if k_range is None:
        k_range = range(2, 11)

    scores = []
    k_values = list(k_range)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(round(score, 4))

    return {"k_values": k_values, "silhouette_scores": scores}


def find_optimal_k(silhouette_data: dict) -> int:
    """
    Find the optimal K based on the highest silhouette score.

    Args:
        silhouette_data: Output from compute_silhouette_scores().

    Returns:
        Optimal number of clusters.
    """
    scores = silhouette_data["silhouette_scores"]
    k_values = silhouette_data["k_values"]
    best_idx = np.argmax(scores)
    return k_values[best_idx]


def train_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> KMeans:
    """
    Train a K-Means model with the specified number of clusters.

    Args:
        X_scaled: Scaled feature array.
        n_clusters: Number of clusters.
        random_state: Random seed for reproducibility.

    Returns:
        Fitted KMeans model.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    kmeans.fit(X_scaled)
    return kmeans


def get_cluster_labels(model: KMeans) -> np.ndarray:
    """Return the cluster labels from a fitted model."""
    return model.labels_


def get_centroids(model: KMeans) -> np.ndarray:
    """Return the cluster centroids from a fitted model."""
    return model.cluster_centers_


def save_model(model, filepath: str = None) -> str:
    """
    Save a trained model to disk using joblib.

    Args:
        model: The model object to save.
        filepath: Destination path. Defaults to models/kmeans_model.pkl.

    Returns:
        The filepath where the model was saved.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "models", "kmeans_model.pkl")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    return filepath


def save_scaler(scaler, filepath: str = None) -> str:
    """
    Save a fitted scaler to disk using joblib.

    Args:
        scaler: The scaler object to save.
        filepath: Destination path. Defaults to models/scaler.pkl.

    Returns:
        The filepath where the scaler was saved.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "models", "scaler.pkl")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scaler, filepath)
    return filepath


def load_model(filepath: str = None) -> KMeans:
    """Load a saved KMeans model from disk."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "models", "kmeans_model.pkl")
    return joblib.load(filepath)


def load_scaler(filepath: str = None):
    """Load a saved scaler from disk."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "models", "scaler.pkl")
    return joblib.load(filepath)


def predict_cluster(
    model: KMeans,
    scaler,
    annual_income: float,
    spending_score: float,
) -> int:
    """
    Predict which cluster a new customer belongs to.

    Args:
        model: Trained KMeans model.
        scaler: Fitted StandardScaler.
        annual_income: Customer's annual income in k$.
        spending_score: Customer's spending score (1-100).

    Returns:
        Cluster label (integer).
    """
    features = np.array([[annual_income, spending_score]])
    features_scaled = scaler.transform(features)
    cluster = model.predict(features_scaled)[0]
    return int(cluster)


def get_cluster_summary(df, labels: np.ndarray) -> dict:
    """
    Generate summary statistics for each cluster.

    Args:
        df: Original DataFrame with all columns.
        labels: Cluster labels array.

    Returns:
        Dictionary mapping cluster_id to summary stats.
    """
    df = df.copy()
    df["Cluster"] = labels

    summary = {}
    for cluster_id in sorted(df["Cluster"].unique()):
        cluster_df = df[df["Cluster"] == cluster_id]
        summary[cluster_id] = {
            "count": len(cluster_df),
            "avg_age": round(cluster_df["Age"].mean(), 1),
            "avg_income": round(cluster_df["Annual Income (k$)"].mean(), 1),
            "avg_spending": round(cluster_df["Spending Score (1-100)"].mean(), 1),
            "gender_split": cluster_df["Gender"].value_counts().to_dict(),
        }

    return summary


# ── Cluster Labels (Business-Friendly Names) ────────────────────────
CLUSTER_NAMES = {
    0: "💰 High Income, High Spenders",
    1: "🎯 Moderate Income, Moderate Spenders",
    2: "📉 High Income, Low Spenders",
    3: "🛍️ Low Income, High Spenders",
    4: "💵 Low Income, Low Spenders",
}

CLUSTER_DESCRIPTIONS = {
    0: "Premium customers who earn well and spend generously. Target with luxury products and VIP loyalty programs.",
    1: "Average customers with balanced income and spending. Engage with seasonal promotions and value bundles.",
    2: "Wealthy but cautious spenders. Convert with exclusive deals, premium quality messaging, and personalized offers.",
    3: "Budget-conscious enthusiasts who love to shop. Retain with flash sales, discounts, and rewards programs.",
    4: "Economy segment with limited spending power. Attract with budget-friendly essentials and basic loyalty cards.",
}
