import pandas as pd
from typing import Dict, Any


def _is_datetime_series(series: pd.Series, sample_size: int = 50, success_ratio: float = 0.8) -> bool:
    """Return True if a sufficient proportion of sampled values can be parsed as datetime.
    Args:
        series: pandas Series to test.
        sample_size: maximum number of non‑null values to sample.
        success_ratio: proportion of successful parses required to classify as datetime.
    """
    # Do not treat numeric columns as datetime
    if pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.iloc[:sample_size]
    # Attempt to parse using pandas to_datetime with errors='coerce'
    parsed = pd.to_datetime(sample, errors='coerce')
    success = parsed.notna().sum()
    return success / len(sample) >= success_ratio


def detect_column_types(
    df: pd.DataFrame,
    categorical_threshold: int = 20,
    id_uniqueness_ratio: float = 0.9,
) -> Dict[str, str]:
    """Detect column types for a DataFrame.

    The function follows the design specification:
    1. Datetime check – sample up to 50 values, classify as ``datetime`` if >=80% parse.
    2. Numeric check – ``pd.api.types.is_numeric_dtype``. Integer columns with <=20 unique values become ``categorical``;
       otherwise ``numeric``.
    3. Object/string columns – compute uniqueness ratio (unique / rows). If >=90% unique → ``identifier``, else ``categorical``.
    4. Empty columns → ``unknown``.

    Args:
        df: Input DataFrame.
        categorical_threshold: Max unique integer values to treat an integer column as categorical.
        id_uniqueness_ratio: Ratio of unique values to rows above which a column is considered an identifier.

    Returns:
        A dictionary mapping column names to detected type strings.
    """
    column_types: Dict[str, str] = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        # Empty column handling
        if series.isna().all():
            column_types[col] = "unknown"
            continue

        # 1. Datetime detection (before numeric)
        if _is_datetime_series(series):
            column_types[col] = "datetime"
            continue

        # 2. Numeric detection
        if pd.api.types.is_numeric_dtype(series):
            # Distinguish integer vs float for categorical decision
            if pd.api.types.is_integer_dtype(series):
                n_unique = series.nunique(dropna=True)
                if n_unique <= categorical_threshold:
                    column_types[col] = "categorical"
                    continue
            column_types[col] = "numeric"
            continue

        # 3. Object / string handling
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            n_unique = series.nunique(dropna=True)
            # Avoid division by zero; n_rows could be 0 only for empty df which we handled earlier
            uniq_ratio = n_unique / n_rows if n_rows > 0 else 0.0
            if uniq_ratio >= id_uniqueness_ratio:
                column_types[col] = "identifier"
            else:
                column_types[col] = "categorical"
            continue

        # Fallback for any other dtype
        column_types[col] = "unknown"

    return column_types
