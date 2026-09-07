import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Enterprise Light Theme Config
THEME_TEMPLATE = 'plotly_white'
COLOR_PRIMARY = '#0284C7'    # Sky Blue
COLOR_SUCCESS = '#059669'    # Emerald
COLOR_PURPLE = '#7C3AED'     # Violet
COLOR_DANGER = '#E11D48'     # Rose
COLOR_AMBER = '#D97706'      # Amber

CLUSTER_PALETTE = ['#059669', '#0284C7', '#7C3AED', '#EA580C', '#EC4899', '#2563EB']

def plot_numeric_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    """Generates a parametric histogram with marginal boxplot in enterprise light mode."""
    fig = px.histogram(
        df,
        x=col,
        marginal='box',
        title=f"NUMERIC PARAMETRIC ANALYSIS<br><b>{col} · Distribution Histogram</b>",
        color_discrete_sequence=[COLOR_PRIMARY],
        template=THEME_TEMPLATE,
        nbins=30
    )
    
    # Add mean line
    mean_val = df[col].mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color=COLOR_SUCCESS, annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=col),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title="Density / Count"),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_categorical_counts(df: pd.DataFrame, col: str, top_n: int = 7) -> go.Figure:
    """Generates a Top-N Volume & Share bar chart."""
    val_counts = df[col].value_counts().head(top_n).reset_index()
    val_counts.columns = [col, 'Count']
    total = len(df)
    val_counts['Share %'] = (val_counts['Count'] / total * 100).map('{:.1f}%'.format)
    
    fig = px.bar(
        val_counts,
        x='Count',
        y=col,
        text='Share %',
        orientation='h',
        title=f"CATEGORICAL CARDINALITY ANALYSIS<br><b>{col} · Top-{min(top_n, len(val_counts))} Volume & Share</b>",
        color='Count',
        color_continuous_scale='Blues',
        template=THEME_TEMPLATE
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        margin=dict(l=40, r=60, t=60, b=40),
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        coloraxis_showscale=False,
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_date_trend(df: pd.DataFrame, date_col: str, numeric_col: str = None) -> go.Figure:
    """Generates a monthly gross volume & rolling trend chart."""
    temp_df = df.dropna(subset=[date_col]).copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    
    date_range_days = (temp_df[date_col].max() - temp_df[date_col].min()).days
    freq = 'D' if date_range_days <= 60 else ('ME' if date_range_days <= 730 else 'YE')
    
    try:
        if numeric_col and numeric_col in temp_df.columns:
            trend = temp_df.set_index(date_col).resample(freq)[numeric_col].mean().reset_index()
            y_val = numeric_col
            title = f"TEMPORAL DECOMPOSITION<br><b>{numeric_col} Volume & Rolling Trend ({date_col})</b>"
        else:
            trend = temp_df.set_index(date_col).resample(freq).size().reset_index(name='Volume')
            y_val = 'Volume'
            title = f"TEMPORAL DECOMPOSITION<br><b>Transaction Volume Trend ({date_col})</b>"
    except Exception:
        # Defensive fallback for legacy or non-standard date ranges
        temp_df['Date_Group'] = temp_df[date_col].dt.date
        if numeric_col and numeric_col in temp_df.columns:
            trend = temp_df.groupby('Date_Group')[numeric_col].mean().reset_index()
            trend.columns = [date_col, numeric_col]
            y_val = numeric_col
            title = f"TEMPORAL DECOMPOSITION<br><b>{numeric_col} Daily Average ({date_col})</b>"
        else:
            trend = temp_df.groupby('Date_Group').size().reset_index(name='Volume')
            trend.columns = [date_col, 'Volume']
            y_val = 'Volume'
            title = f"TEMPORAL DECOMPOSITION<br><b>Daily Transaction Volume ({date_col})</b>"


    fig = px.area(
        trend,
        x=date_col,
        y=trend.columns[1],
        title=title,
        color_discrete_sequence=['#0284C7'],
        template=THEME_TEMPLATE
    )
    fig.update_traces(line=dict(width=3), fillcolor='rgba(2, 132, 199, 0.1)')
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title="Timeline"),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=y_val),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Generates an annotated correlation matrix heatmap matching Screenshot 3."""
    z_vals = corr_df.values
    x_labels = list(corr_df.columns)
    y_labels = list(corr_df.index)
    
    annotations = []
    for i, row in enumerate(y_labels):
        for j, col in enumerate(x_labels):
            val = z_vals[i][j]
            val_str = f"{val:+.2f}" if val != 1.0 else "1.00"
            
            # Color logic matching screenshot: Blue for positive, pink for negative
            font_col = "#0369A1" if val > 0.3 else ("#BE123C" if val < -0.2 else "#475569")
            
            annotations.append(dict(
                x=col, y=row,
                text=f"<b>{val_str}</b>",
                font=dict(color=font_col, size=11, family="JetBrains Mono, monospace"),
                showarrow=False
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z_vals, x=x_labels, y=y_labels,
        colorscale=[[0, '#FFE4E6'], [0.5, '#F8FAFC'], [1, '#E0F2FE']],
        zmin=-1, zmax=1,
        showscale=False
    ))
    
    fig.update_layout(
        annotations=annotations,
        title="MATRIX ANALYSIS<br><b>Pearson Correlation Matrix</b>",
        template=THEME_TEMPLATE,
        margin=dict(l=60, r=40, t=60, b=60),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_cluster_scatter(pca_df: pd.DataFrame) -> go.Figure:
    """Generates a 2D PCA Cluster Scatter Plot matching Screenshot 3."""
    fig = px.scatter(
        pca_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        hover_data=['Persona'] + [c for c in pca_df.columns if c not in ['PCA1', 'PCA2', 'Cluster', 'Persona']],
        title="UNSUPERVISED SEGMENTS<br><b>KMeans Behavioral Clustering - 2D PCA Space</b>",
        color_discrete_sequence=CLUSTER_PALETTE,
        template=THEME_TEMPLATE
    )
    
    # Calculate Centroids
    centroids = pca_df.groupby('Cluster')[['PCA1', 'PCA2']].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=centroids['PCA1'],
        y=centroids['PCA2'],
        mode='markers',
        marker=dict(size=14, color='white', line=dict(width=3, color='#0F172A'), symbol='circle'),
        name='Centroid Nodes (o)',
        hoverinfo='skip'
    ))

    fig.update_traces(marker=dict(size=9, opacity=0.85))
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title="PCA-1 Vector"),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title="PCA-2 Vector"),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_outlier_scatter(outlier_df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Generates an Isolation Forest Outlier highlight scatter plot."""
    fig = px.scatter(
        outlier_df,
        x=x_col,
        y=y_col,
        color='Is_Outlier',
        color_discrete_map={'Inlier': '#0284C7', 'Isolated Anomaly': '#E11D48'},
        title=f"UNSUPERVISED OUTLIER TELEMETRY<br><b>Isolation Forest Anomaly Detection ({x_col} vs {y_col})</b>",
        template=THEME_TEMPLATE
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=x_col),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=y_col),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_custom_user_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str = None, color_col: str = None, agg_func: str = "Mean", z_col: str = None) -> go.Figure:
    """
    Builds custom user-requested visualizations dynamically.
    Supports: Scatter, Bar (Aggregated), Line, Box, Violin, Histogram, Pie/Donut, 3D Scatter.
    """
    color_param = color_col if color_col and color_col != "None" else None
    y_param = y_col if y_col and y_col != "None" else None
    z_param = z_col if z_col and z_col != "None" else None

    title_str = f"CUSTOM EXPLORER<br><b>{chart_type}: {x_col}" + (f" vs {y_param}" if y_param else "") + (f" (Grouped by {color_param})" if color_param else "") + "</b>"

    if chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_param, color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)
        fig.update_traces(marker=dict(size=8, opacity=0.85))

    elif chart_type == "Bar Chart (Aggregated)":
        if y_param:
            if agg_func == "Sum":
                agg_df = df.groupby(x_col)[y_param].sum().reset_index()
            elif agg_func == "Count":
                agg_df = df.groupby(x_col)[y_param].count().reset_index()
            elif agg_func == "Median":
                agg_df = df.groupby(x_col)[y_param].median().reset_index()
            else:
                agg_df = df.groupby(x_col)[y_param].mean().reset_index()
            fig = px.bar(agg_df, x=x_col, y=y_param, color=color_param, title=f"{title_str} ({agg_func})", template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)
        else:
            val_counts = df[x_col].value_counts().reset_index()
            val_counts.columns = [x_col, 'Count']
            fig = px.bar(val_counts, x=x_col, y='Count', color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "Line Chart":
        fig = px.line(df, x=x_col, y=y_param, color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "Box Plot":
        fig = px.box(df, x=x_col, y=y_param, color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "Violin Plot":
        fig = px.violin(df, x=x_col, y=y_param, color=color_param, box=True, points="all", title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_col, y=y_param, color=color_param, marginal="rug", title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "Pie / Donut Chart":
        if y_param:
            fig = px.pie(df, names=x_col, values=y_param, color=color_param, hole=0.4, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)
        else:
            fig = px.pie(df, names=x_col, hole=0.4, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "3D Scatter Plot" and z_param:
        fig = px.scatter_3d(df, x=x_col, y=y_param, z=z_param, color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)
    else:
        fig = px.scatter(df, x=x_col, y=y_param, color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

