import pandas as pd
import numpy as np
import re

def filter_dataset_by_nl(df: pd.DataFrame, col_types: dict, user_query: str) -> dict:
    """
    Parses natural-language queries into pandas boolean filtering conditions.
    Examples:
    - 'show me the West region' -> df['Region'].str.contains('West', case=False)
    - 'orders above ₹5000' -> df['Sales'] > 5000
    - 'sales under 1000' -> df['Sales'] < 1000
    - 'female patients' -> df['Gender'].str.contains('Female', case=False)
    - 'category equals Electronics' -> df['Category'].str.contains('Electronics', case=False)
    """
    if df is None or df.empty:
        return {
            "query": user_query,
            "filtered_df": df,
            "original_count": 0,
            "filtered_count": 0,
            "percentage": 0.0,
            "applied_rules": [],
            "summary": "Empty dataset."
        }

    if not user_query or not user_query.strip():
        return {
            "query": "",
            "filtered_df": df,
            "original_count": len(df),
            "filtered_count": len(df),
            "percentage": 100.0,
            "applied_rules": [],
            "summary": f"Displaying full dataset ({len(df):,} rows)."
        }

    query_raw = user_query.strip()
    query_lower = query_raw.lower()
    
    numeric_cols = col_types.get('numeric', [])
    categorical_cols = col_types.get('categorical', []) + col_types.get('text', [])
    
    mask = pd.Series(True, index=df.index)
    applied_rules = []
    
    # -------------------------------------------------------------------
    # STEP 1: NUMERIC COMPARISON OPERATORS (above, below, greater, less, >, <, =, etc.)
    # -------------------------------------------------------------------
    num_op_regex = re.compile(
        r'(?:([a-zA-Z0-9_\s]+)\s+)?'  # Optional column name before operator
        r'(above|greater than|more than|over|>|>=|exceeding|higher than|below|less than|under|<|<=|lower than|smaller than|equal to|equals|=|==|is)\s*'
        r'(?:[₹$€£]\s*)?'              # Optional currency symbol
        r'([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)', # Numeric value
        re.IGNORECASE
    )
    
    matches = num_op_regex.findall(query_raw)
    
    matched_spans = []
    for match in matches:
        col_prefix, operator, num_str = match
        num_val = float(num_str.replace(',', ''))
        op_lower = operator.lower()
        
        # Determine target numeric column
        target_col = None
        if col_prefix and col_prefix.strip():
            prefix_clean = col_prefix.strip().lower()
            # Match against numeric columns
            for nc in numeric_cols:
                nc_clean = nc.lower().replace('_', ' ')
                if nc_clean in prefix_clean or prefix_clean in nc_clean or any(w in nc_clean for w in prefix_clean.split()):
                    target_col = nc
                    break
        
        if not target_col:
            # Look anywhere in query for a numeric column name
            for nc in numeric_cols:
                nc_clean = nc.lower().replace('_', ' ')
                if nc_clean in query_lower:
                    target_col = nc
                    break
        
        # Fallback to first numeric column if none matched
        if not target_col and numeric_cols:
            target_col = numeric_cols[0]
            
        if target_col and target_col in df.columns:
            if op_lower in ['above', 'greater than', 'more than', 'over', '>', '>=', 'exceeding', 'higher than']:
                cond = df[target_col] > num_val
                applied_rules.append(f"`{target_col}` > {num_val:,.2f}".rstrip('0').rstrip('.'))
            elif op_lower in ['below', 'less than', 'under', '<', '<=', 'lower than', 'smaller than']:
                cond = df[target_col] < num_val
                applied_rules.append(f"`{target_col}` < {num_val:,.2f}".rstrip('0').rstrip('.'))
            else:
                cond = df[target_col] == num_val
                applied_rules.append(f"`{target_col}` == {num_val:,.2f}".rstrip('0').rstrip('.'))
                
            mask = mask & cond

    # -------------------------------------------------------------------
    # STEP 2: CATEGORICAL / TEXT MATCHING (e.g., "West region", "female", "electronics")
    # -------------------------------------------------------------------
    # Clean query text by removing filler phrases
    clean_text_query = re.sub(
        r'\b(show me|filter|find|get|rows|where|display|the|dataset|records|with|that have|which have|list|all)\b',
        '', query_lower, flags=re.IGNORECASE
    ).strip()
    
    # Check categorical column values in df
    found_cat_match = False
    for cat_col in categorical_cols:
        if cat_col in df.columns:
            unique_vals = df[cat_col].dropna().unique()
            for u_val in unique_vals:
                u_val_str = str(u_val).strip()
                if len(u_val_str) > 1 and u_val_str.lower() in clean_text_query:
                    cond = df[cat_col].astype(str).str.contains(re.escape(u_val_str), case=False, na=False)
                    mask = mask & cond
                    applied_rules.append(f"`{cat_col}` == '{u_val_str}'")
                    found_cat_match = True

    # -------------------------------------------------------------------
    # STEP 3: FALLBACK FULL-TEXT SEARCH (if no specific filter matched)
    # -------------------------------------------------------------------
    if not applied_rules and clean_text_query:
        # Search across all object/string columns
        text_cols = [c for c in df.columns if df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c])]
        if text_cols:
            full_search_mask = pd.Series(False, index=df.index)
            search_terms = clean_text_query.split()
            for term in search_terms:
                if len(term) > 2:
                    for tc in text_cols:
                        full_search_mask = full_search_mask | df[tc].astype(str).str.contains(re.escape(term), case=False, na=False)
            
            if full_search_mask.any():
                mask = mask & full_search_mask
                applied_rules.append(f"Text Search: '{clean_text_query}'")

    # Apply final boolean mask
    filtered_df = df[mask].reset_index(drop=True)
    orig_len = len(df)
    filt_len = len(filtered_df)
    pct = round((filt_len / orig_len) * 100.0, 1) if orig_len > 0 else 0.0

    if applied_rules:
        summary_str = f"Filtered view: **{filt_len:,}** of **{orig_len:,}** rows ({pct}%) matching {' and '.join(applied_rules)}."
    else:
        summary_str = f"No specific filter matched. Displaying full dataset ({orig_len:,} rows)."

    return {
        "query": user_query,
        "filtered_df": filtered_df,
        "original_count": orig_len,
        "filtered_count": filt_len,
        "percentage": pct,
        "applied_rules": applied_rules,
        "summary": summary_str
    }
