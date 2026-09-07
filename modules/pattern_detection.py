import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Tuple, Any, Optional


def find_correlations(df: pd.DataFrame, numeric_cols: List[str], top_n: int = 5) -> List[Tuple[str, str, float]]:
    """Return the top N absolute Pearson correlations between numeric columns.
    Each entry is a tuple ``(col_a, col_b, r)`` where ``r`` is the correlation coefficient.
    Returns an empty list if fewer than 2 numeric columns are provided.
    """
    if len(numeric_cols) < 2:
        return []
    corr_matrix = df[numeric_cols].corr(method="pearson")
    # Unstack, drop self‑correlations, take absolute value for sorting
    corr_series = corr_matrix.abs().unstack()
    # Remove duplicate pairs (a,b) and (b,a) as well as self correlations
    mask = corr_series.index.get_level_values(0) < corr_series.index.get_level_values(1)
    corr_series = corr_series[mask]
    top = corr_series.nlargest(top_n)
    # Retrieve original signed values
    results = []
    for (col_a, col_b), _ in top.items():
        r = corr_matrix.loc[col_a, col_b]
        results.append((col_a, col_b, r))
    return results


def run_clustering(df: pd.DataFrame, numeric_cols: List[str], k_range: range = range(2, 6)) -> Optional[Dict[str, Any]]:
    """Cluster the numeric data using KMeans.
    The function scales the numeric columns, tries each k in ``k_range`` and selects the
    k with the highest silhouette score. Returns ``None`` if the data is insufficient.
    """
    if len(numeric_cols) < 2 or len(df) < 10:
        return None
    X = df[numeric_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_k = None
    best_score = -1
    best_labels = None
    for k in k_range:
        if k >= len(df):
            # Cannot have more clusters than samples
            continue
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        if len(set(labels)) == 1:
            # Silhouette undefined for a single cluster
            continue
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_centers = kmeans.cluster_centers_
    if best_k is None:
        return None
    return {
        "k": best_k,
        "labels": best_labels,
        "centers": best_centers,
        "silhouette": best_score,
    }


def find_outliers(df: pd.DataFrame, numeric_cols: List[str], contamination: float = 0.05) -> List[int]:
    """Detect outlier rows using IsolationForest.
    Returns a list of row indices flagged as outliers. Empty list if not enough rows.
    """
    if len(df) < 10:
        return []
    X = df[numeric_cols].values
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(X)
    # IsolationForest returns -1 for outliers, 1 for inliers
    outlier_idx = np.where(preds == -1)[0].tolist()
    return outlier_idx


def category_vs_numeric(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Identify categorical drivers for numeric metrics.
    For each (categorical, numeric) pair, compute the normalized spread of group means
    (|group_mean - overall_mean| / overall_std). Returns the top N pairs with the largest
    spread. Categorical columns with >15 or <2 unique values are skipped.
    """
    results: List[Dict[str, Any]] = []
    for cat_col in categorical_cols:
        n_unique = df[cat_col].nunique()
        if n_unique > 15 or n_unique < 2:
            continue
        for num_col in numeric_cols:
            overall_std = df[num_col].std()
            if overall_std == 0:
                continue
            overall_mean = df[num_col].mean()
            group_means = df.groupby(cat_col)[num_col].mean()
            # Compute the maximum absolute deviation normalized by overall std
            max_dev = (group_means - overall_mean).abs().max() / overall_std
            results.append({
                "categorical": cat_col,
                "numeric": num_col,
                "max_normalized_deviation": max_dev,
            })
    # Sort descending by deviation and keep top_n
    results.sort(key=lambda x: x["max_normalized_deviation"], reverse=True)
    return results[:top_n]
