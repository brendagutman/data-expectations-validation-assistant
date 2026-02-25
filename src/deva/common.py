import pandas as pd
import html
import numpy as np
from typing import Optional, Dict


def infer_dtypes_prefer_int(df: pd.DataFrame) -> Dict[str, str]:
    """Infer dtypes for a DataFrame, preferring integers over floats when possible.
    
    Returns a dictionary suitable for passing to pd.astype().
    """
    dtype_map = {}
    for col in df.columns:
        series = df[col]
        # Skip if all NaN
        if series.isna().all():
            continue
        
        # If already object/string, keep as is
        if series.dtype == 'object' or isinstance(series.dtype, pd.StringDtype):
            dtype_map[col] = 'string'
            continue
        
        # If float, check if all non-NaN values are integers
        if pd.api.types.is_float_dtype(series):
            non_null_vals = series.dropna()
            if len(non_null_vals) > 0 and (non_null_vals == non_null_vals.astype(int)).all():
                dtype_map[col] = 'Int64'
            else:
                dtype_map[col] = 'float64'
        elif pd.api.types.is_integer_dtype(series):
            dtype_map[col] = 'Int64' 
        elif pd.api.types.is_bool_dtype(series):
            dtype_map[col] = 'bool'
        elif nullable_string_dtype := pd.api.types.is_string_dtype(series):
            dtype_map[col] = 'string'
        # else: keep current dtype
    
    return dtype_map


def convert_df_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply intelligent dtype conversion, preferring integers over floats."""
    dtype_map = infer_dtypes_prefer_int(df)
    if dtype_map:
        df = df.astype(dtype_map, errors='ignore')
    return df


def format_dtype_for_display(dtype_str: str) -> str:
    """Format dtype string for display, e.g. 'int64' -> 'integer'."""
    dtype_str = str(dtype_str).lower()
    if dtype_str in ('int64', 'int32', 'int16', 'int8', 'int'):
        return 'integer'
    elif dtype_str in ('float64', 'float32', 'float'):
        return 'float'
    elif dtype_str == 'object':
        return 'string'
    elif dtype_str == 'bool':
        return 'boolean'
    return dtype_str