import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Classifies columns of a dataframe into five distinct categories:
    - 'numeric': continuous or quantitative numbers
    - 'categorical': discrete groups or low-cardinality labels
    - 'date': datetime values or convertible date strings
    - 'id_high_cardinality': unique identifiers or keys
    - 'text': unstructured free-form text
    
    Returns structured details including null counts, unique percentages,
    sample values, and pipeline schema metadata.
    """
    classified = {
        'numeric': [],
        'categorical': [],
        'date': [],
        'id_high_cardinality': [],
        'text': [],
        'details': {},
        'telemetry_table': []
    }
    
    n_rows = len(df)
    if n_rows == 0:
        return classified

    for col in df.columns:
        col_type = _classify_single_column(df[col], col, n_rows)
        classified[col_type].append(col)
        classified['details'][col] = col_type
        
        # Calculate rich telemetry for table
        non_null_count = df[col].notnull().sum()
        null_count = df[col].isnull().sum()
        missing_pct = (null_count / max(n_rows, 1)) * 100
        n_unique = df[col].nunique(dropna=True)
        unique_pct = (n_unique / max(n_rows, 1)) * 100
        
        sample_vals = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(str(v) for v in sample_vals)
        if len(sample_str) > 40:
            sample_str = sample_str[:37] + "..."
            
        classified['telemetry_table'].append({
            "Feature Name": col,
            "Inferred Schema Type": col_type,
            "Non-Null Count": f"{non_null_count:,}",
            "Unique %": f"{unique_pct:.1f}%",
            "Missing %": f"{missing_pct:.1f}%" if missing_pct > 0 else "0.0%",
            "Sample Values": sample_str
        })

    return classified

def _classify_single_column(series: pd.Series, col_name: str, n_rows: int) -> str:
    try:
        col_lower = str(col_name).lower().strip()
        non_nulls = series.dropna()
        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / max(n_rows, 1)

        # 1. Direct Datetime check
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'date'

        # 2. String/Object Datetime Parsing check
        if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
            if any(date_word in col_lower for date_word in ['date', 'time', 'timestamp', 'dt', 'dob', 'created', 'updated']):
                try:
                    parsed = pd.to_datetime(non_nulls.head(200), errors='coerce')
                    if parsed.notnull().mean() > 0.6:
                        return 'date'
                except Exception:
                    pass
            else:
                if len(non_nulls) > 0:
                    sample_str = str(non_nulls.iloc[0]).strip()
                    if any(char in sample_str for char in ['-', '/', ':', ' ']) and len(sample_str) >= 6:
                        try:
                            parsed = pd.to_datetime(non_nulls.head(100), errors='coerce')
                            if parsed.notnull().mean() > 0.7:
                                return 'date'
                        except Exception:
                            pass

        # 3. ID / High Cardinality check
        id_keywords = ['id', 'uuid', 'guid', 'ssn', 'tax_id', 'account_num', 'index', 'key', 'code', 'hash']
        is_id_named = any(col_lower == kw or col_lower.endswith('_' + kw) or col_lower.startswith(kw + '_') for kw in id_keywords)
        
        if is_id_named and (unique_ratio > 0.3 or n_unique > 20):
            return 'id_high_cardinality'

        if unique_ratio >= 0.95 and n_rows >= 10:
            if not pd.api.types.is_float_dtype(series):
                return 'id_high_cardinality'

        # 4. Numeric check
        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= 5 and pd.api.types.is_integer_dtype(series):
                return 'categorical'
            return 'numeric'

        # 5. Categorical vs Text check
        if n_unique <= 30 or unique_ratio <= 0.05:
            return 'categorical'

        if len(non_nulls) > 0:
            avg_len = non_nulls.astype(str).str.len().mean()
            if avg_len > 40:
                return 'text'

        return 'categorical'
    except Exception:
        return 'categorical'
