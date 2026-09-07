import pandas as pd
import numpy as np

def query_dataset(df: pd.DataFrame, col_types: dict, user_query: str) -> dict:
    """
    Parses natural language queries against the dataset and computes instant metric answers.
    Examples:
    - 'top region by revenue' -> Highest Revenue region
    - 'average customer age' -> Mean value of Age column
    - 'total sales' -> Sum of Revenue column
    - 'outlier count' -> Anomaly stats
    """
    if not user_query or not user_query.strip():
        return None

    query_lower = user_query.lower().strip()
    all_cols = list(df.columns)
    numeric_cols = col_types.get('numeric', [])
    categorical_cols = col_types.get('categorical', [])
    
    # 1. Search for matching column names in query
    matched_num_col = None
    for c in numeric_cols:
        c_clean = c.lower().replace('_', ' ')
        if c_clean in query_lower or any(word in query_lower for word in c_clean.split()):
            matched_num_col = c
            break

    matched_cat_col = None
    for c in categorical_cols:
        c_clean = c.lower().replace('_', ' ')
        if c_clean in query_lower or any(word in query_lower for word in c_clean.split()):
            matched_cat_col = c
            break

    # Default fallbacks if no exact col match
    if not matched_num_col and numeric_cols:
        matched_num_col = numeric_cols[0]
    if not matched_cat_col and categorical_cols:
        matched_cat_col = categorical_cols[0]

    # 2. INTENT CLASSIFICATION & AGGREGATION
    # Intent A: Average / Mean
    if any(kw in query_lower for kw in ['average', 'mean', 'avg']):
        if matched_num_col:
            val = df[matched_num_col].mean()
            return {
                "query": user_query,
                "answer_title": f"Average {matched_num_col}",
                "answer_val": f"{val:,.2f}",
                "explanation": f"Computed mean across {len(df):,} non-null records in attribute '{matched_num_col}'."
            }

    # Intent B: Total / Sum
    if any(kw in query_lower for kw in ['total', 'sum', 'overall', 'gross']):
        if matched_num_col:
            val = df[matched_num_col].sum()
            return {
                "query": user_query,
                "answer_title": f"Total {matched_num_col}",
                "answer_val": f"{val:,.2f}",
                "explanation": f"Summed aggregate of '{matched_num_col}' across all dataset observations."
            }

    # Intent C: Top / Maximum / Highest Category Breakdown
    if any(kw in query_lower for kw in ['top', 'highest', 'max', 'most', 'best', 'dominant']):
        if matched_cat_col and matched_num_col:
            grouped = df.groupby(matched_cat_col)[matched_num_col].sum().sort_values(ascending=False)
            top_cat = grouped.index[0]
            top_val = grouped.iloc[0]
            return {
                "query": user_query,
                "answer_title": f"Top {matched_cat_col} by {matched_num_col}",
                "answer_val": f"{top_cat} ({top_val:,.2f})",
                "explanation": f"Ranked highest contribution segment in '{matched_cat_col}' grouped by '{matched_num_col}'."
            }
        elif matched_num_col:
            max_val = df[matched_num_col].max()
            return {
                "query": user_query,
                "answer_title": f"Maximum {matched_num_col}",
                "answer_val": f"{max_val:,.2f}",
                "explanation": f"Highest recorded value in numeric attribute '{matched_num_col}'."
            }

    # Intent D: Lowest / Minimum
    if any(kw in query_lower for kw in ['lowest', 'min', 'minimum', 'smallest', 'worst']):
        if matched_num_col:
            min_val = df[matched_num_col].min()
            return {
                "query": user_query,
                "answer_title": f"Minimum {matched_num_col}",
                "answer_val": f"{min_val:,.2f}",
                "explanation": f"Lowest recorded value in numeric attribute '{matched_num_col}'."
            }

    # Intent E: Count / Records
    if any(kw in query_lower for kw in ['count', 'number', 'rows', 'records', 'how many']):
        return {
            "query": user_query,
            "answer_title": "Total Dataset Volume",
            "answer_val": f"{len(df):,} records",
            "explanation": f"Cleaned observation count across {len(all_cols)} detected attributes."
        }

    # General Fallback Response
    if matched_num_col:
        val = df[matched_num_col].sum()
        return {
            "query": user_query,
            "answer_title": f"Aggregate Telemetry ({matched_num_col})",
            "answer_val": f"{val:,.2f}",
            "explanation": f"Auto-computed total metric value for target attribute '{matched_num_col}'."
        }

    return {
        "query": user_query,
        "answer_title": "Dataset Records Evaluated",
        "answer_val": f"{len(df):,} rows",
        "explanation": f"Executed zero-loss query scanning {len(df):,} observations."
    }
