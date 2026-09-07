import pandas as pd
import re
from typing import Tuple, List, Dict, Any


def _clean_string_value(val: Any) -> Any:
    """Normalize string values.
    - Strip surrounding whitespace.
    - Convert currency strings like "$1,200.50" to float 1200.5.
    - Convert percent strings like "45%" to float 0.45.
    Returns the original value if no conversion applies.
    """
    if isinstance(val, str):
        stripped = val.strip()
        # Currency pattern: optional $ sign, optional commas, optional decimals
        currency_match = re.fullmatch(r"\$?([0-9,]+(?:\.[0-9]+)?)", stripped)
        if currency_match:
            num_str = currency_match.group(1).replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                pass
        # Percent pattern: number followed by %
        percent_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)%", stripped)
        if percent_match:
            try:
                return float(percent_match.group(1)) / 100.0
            except ValueError:
                pass
        return stripped
    return val


def clean_dataset(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    missing_drop_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, List[str]]:
    """Clean the dataset according to the design specification.

    Steps:
    1. String normalization for all object columns.
    2. Duplicate removal.
    3. Missing‑value handling per detected type.
       - If > ``missing_drop_threshold`` missing, drop the column.
       - ``numeric`` → fill with median.
       - ``categorical`` → fill with mode.
       - ``datetime`` → coerce to datetime, then drop rows where NaT.
       - Others → fill with "Unknown".
    Returns the cleaned DataFrame and a list of human‑readable report strings.
    """
    report: List[str] = []
    # Work on a copy to avoid side‑effects
    clean_df = df.copy()

    # 1. String normalization
    obj_cols = clean_df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        original_nonnull = clean_df[col].notna().sum()
        clean_df[col] = clean_df[col].apply(_clean_string_value)
        changed = (clean_df[col].notna().sum() != original_nonnull)
        if changed:
            report.append(f"Normalized string values in column '{col}'.")

    # 2. Duplicate removal
    before_dup = len(clean_df)
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)
    after_dup = len(clean_df)
    if before_dup != after_dup:
        report.append(f"Removed {before_dup - after_dup} duplicate rows.")

    # 3. Missing‑value handling per column
    rows_before = len(clean_df)
    for col, col_type in column_types.items():
        if col not in clean_df.columns:
            # Column may have been dropped previously (e.g., unknown)
            continue
        missing_ratio = clean_df[col].isna().mean()
        if missing_ratio > missing_drop_threshold:
            clean_df = clean_df.drop(columns=[col])
            report.append(f"Dropped column '{col}' (>{missing_drop_threshold*100}% missing).")
            continue
        if col_type == "numeric":
            median_val = clean_df[col].median()
            clean_df[col] = clean_df[col].fillna(median_val)
            report.append(f"Filled missing values in '{col}' with median ({median_val}).")
        elif col_type == "categorical":
            mode_series = clean_df[col].mode()
            mode_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            clean_df[col] = clean_df[col].fillna(mode_val)
            report.append(f"Filled missing values in '{col}' with mode ('{mode_val}').")
        elif col_type == "datetime":
            # Coerce to datetime first (in case the column is still string)
            clean_df[col] = pd.to_datetime(clean_df[col], errors="coerce")
            # Drop rows where this datetime column is NaT
            rows_before_dt = len(clean_df)
            clean_df = clean_df[clean_df[col].notna()]
            dropped = rows_before_dt - len(clean_df)
            if dropped:
                report.append(f"Dropped {dropped} rows due to unparsable dates in '{col}'.")
        else:
            # Catch‑all for unknown/identifier etc.
            clean_df[col] = clean_df[col].fillna("Unknown")
            report.append(f"Filled missing values in '{col}' with placeholder 'Unknown'.")

    rows_after = len(clean_df)
    if rows_before != rows_after:
        report.append(f"Dataset size changed from {rows_before} to {rows_after} rows after cleaning.")

    return clean_df, report
