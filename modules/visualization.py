import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional


def _add_chart(charts: List[Dict[str, Any]], title: str, fig: Any) -> None:
    """Append a chart dict to the list."""
    charts.append({"title": title, "figure": fig})


def build_charts(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    correlations: Optional[pd.DataFrame] = None,
    cluster_result: Optional[Dict[str, Any]] = None,
    datetime_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Automatically select and build Plotly charts based on data shape.

    The selection rules follow the design specification:
    * Single numeric column → histogram (up to first 4 numeric columns).
    * Single categorical column → bar chart of value counts (up to first 4).
    * Datetime + numeric → line chart (one per numeric column).
    * ≥2 numeric columns → correlation heatmap.
    * Clustering succeeded → scatter plot colored by cluster (first 2 numeric dims).
    * Categorical + numeric → box plot (only if categorical has ≤15 unique values).

    Parameters
    ----------
    df: pd.DataFrame
        Cleaned DataFrame.
    column_types: dict mapping column name to detected type ("numeric", "categorical", "datetime", ...).
    correlations: optional pre‑computed correlation DataFrame (if None, will compute from df).
    cluster_result: optional dict from ``run_clustering`` containing ``labels`` and ``silhouette``.
    datetime_col: optional explicit datetime column name (if not provided, the first detected datetime column is used).

    Returns
    -------
    List[dict]
        Each dict contains ``"title"`` and ``"figure"`` (a Plotly Figure).
    """
    charts: List[Dict[str, Any]] = []

    # Classify columns
    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in column_types.items() if t == "categorical"]
    datetime_cols = [c for c, t in column_types.items() if t == "datetime"]

    # 1. Single numeric → histogram (cap to 4)
    if len(numeric_cols) == 1:
        col = numeric_cols[0]
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        _add_chart(charts, f"Histogram of {col}", fig)
    elif len(numeric_cols) > 1:
        # 4. Correlation heatmap for >=2 numeric columns
        corr = correlations if correlations is not None else df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmid=0))
        fig.update_layout(title="Correlation Heatmap", xaxis_showgrid=False, yaxis_showgrid=False)
        _add_chart(charts, "Correlation Heatmap", fig)

    # 2. Single categorical → bar chart (cap to 4)
    if len(categorical_cols) == 1:
        col = categorical_cols[0]
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", title=f"Value Counts of {col}")
        _add_chart(charts, f"Bar chart of {col}", fig)

    # 3. Datetime + numeric → line chart per numeric column
    if datetime_cols and numeric_cols:
        dt_col = datetime_col if datetime_col else datetime_cols[0]
        # Ensure proper datetime dtype
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        for col in numeric_cols[:4]:  # cap to first 4 numeric for readability
            fig = px.line(df.sort_values(dt_col), x=dt_col, y=col, title=f"{col} over time")
            _add_chart(charts, f"Line chart of {col} by {dt_col}", fig)

    # 5. Clustering scatter plot (first 2 numeric dims)
    if cluster_result is not None and "labels" in cluster_result:
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[:2]
            df_scatter = df[[x_col, y_col]].copy()
            df_scatter["cluster"] = cluster_result["labels"]
            fig = px.scatter(df_scatter, x=x_col, y=y_col, color="cluster",
                             title="Clustering (KMeans) Scatter Plot",
                             hover_data=[x_col, y_col])
            _add_chart(charts, "Clustering Scatter Plot", fig)

    # 6. Categorical + numeric → box plots (limit to readable combos)
    for cat_col in categorical_cols:
        uniq = df[cat_col].nunique()
        if 2 <= uniq <= 15:
            for num_col in numeric_cols[:3]:  # cap number of box plots per cat
                fig = px.box(df, x=cat_col, y=num_col, points="all",
                             title=f"{num_col} distribution by {cat_col}")
                _add_chart(charts, f"Box plot of {num_col} by {cat_col}", fig)

    return charts
