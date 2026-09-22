"""
Data Preprocessing Module for Customer Segmentation.

Handles loading, cleaning, encoding, feature selection,
and scaling of the Mall Customer dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the Mall Customers dataset from CSV.

    Args:
        filepath: Path to the CSV file. Defaults to data/Mall_Customers.csv.

    Returns:
        Raw DataFrame with all columns.
    """
    if filepath is None:
        # Resolve relative to this file's parent directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "Mall_Customers.csv")

    df = pd.read_csv(filepath)
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """
    Return a summary dictionary with key statistics about the dataset.

    Args:
        df: Raw DataFrame.

    Returns:
        Dictionary with shape, dtypes, null counts, and descriptive stats.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "describe": df.describe(),
        "gender_counts": df["Gender"].value_counts().to_dict(),
    }
    return summary


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the 'Gender' column: Male=0, Female=1.

    Args:
        df: DataFrame with 'Gender' column.

    Returns:
        DataFrame with encoded 'Gender_Encoded' column added.
    """
    df = df.copy()
    le = LabelEncoder()
    df["Gender_Encoded"] = le.fit_transform(df["Gender"])
    return df


def select_features(
    df: pd.DataFrame,
    features: list = None,
) -> pd.DataFrame:
    """
    Select features for clustering.

    Args:
        df: Full DataFrame.
        features: List of column names to use. Defaults to
                  ['Annual Income (k$)', 'Spending Score (1-100)'].

    Returns:
        DataFrame with only the selected feature columns.
    """
    if features is None:
        features = ["Annual Income (k$)", "Spending Score (1-100)"]

    return df[features].copy()


def scale_features(X: pd.DataFrame) -> tuple:
    """
    Standardize features using StandardScaler.

    Args:
        X: DataFrame of numeric features.

    Returns:
        Tuple of (scaled_array, fitted_scaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def preprocess_pipeline(
    filepath: str = None,
    features: list = None,
) -> dict:
    """
    Run the full preprocessing pipeline.

    Args:
        filepath: Path to CSV file.
        features: Feature columns to use for clustering.

    Returns:
        Dictionary with keys:
            - 'df_raw': original DataFrame
            - 'df_encoded': DataFrame with Gender encoded
            - 'X': selected features DataFrame
            - 'X_scaled': scaled numpy array
            - 'scaler': fitted StandardScaler
            - 'feature_names': list of feature column names
    """
    df_raw = load_data(filepath)
    df_encoded = encode_gender(df_raw)

    if features is None:
        features = ["Annual Income (k$)", "Spending Score (1-100)"]

    X = select_features(df_encoded, features)
    X_scaled, scaler = scale_features(X)

    return {
        "df_raw": df_raw,
        "df_encoded": df_encoded,
        "X": X,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "feature_names": features,
    }
