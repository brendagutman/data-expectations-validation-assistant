import pandas as pd
import html
import numpy as np
from typing import Optional, Dict
from pathlib import Path
import yaml
import chardet
from deva import logger


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


def format_dtype_for_display(dtype_str) -> str:
    """Format dtype string for display, e.g. 'int64' -> 'integer'."""

    dtype_str = str(dtype_str).lower()
    if dtype_str in ('int64', 'int32', 'int16', 'int8', 'int'):
        return 'integer'
    elif dtype_str in ('float64', 'float32', 'float'):
        return "number"
    elif dtype_str == 'object':
        return 'string'
    elif dtype_str == 'bool':
        return 'boolean'
    return dtype_str


def infer_dtype_from_samples(samples) -> str:
    """A minimal dtype inference used when streaming.

    Samples may be a list or iterable of values. Tries integer, float,
    boolean-like, otherwise falls back to string.  Replicated from
    *characterize_datafile* logic so both modules can share it.
    """
    if not samples:
        return 'string'
    # try integer
    try:
        for v in samples:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            int(str(v))
        return 'integer'
    except Exception:
        pass
    # try float
    try:
        for v in samples:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            float(str(v))
        return 'float'
    except Exception:
        pass
    # boolean-like
    bool_set = {"true", "false", "True", "False", "0", "1", 0, 1}
    if all(str(v) in bool_set for v in samples if v is not None):
        return 'boolean'
    return 'string'

def deep_merge(existing: dict, incoming: dict) -> dict:
    for key, value in incoming.items():
        if (
            key in existing
            and isinstance(existing[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge(existing[key], value)
        else:
            existing[key] = value
    return existing

def detect_encoding(file_path, num_bytes=4096):
    with open(file_path, "rb") as f:
        raw = f.read(num_bytes)
    result = chardet.detect(raw)
    return result["encoding"]


def read_file(filepath, **kwargs):
    """
    Read a file and return its contents based on the file type.
    Supports YAML, CSV, Excel, SQL, and Markdown files.

    Handles encoding detection for text files.

    Optional kwargs are passed through to pandas readers when applicable
    (currently CSV only).
    """
    path = Path(filepath).resolve()

    if not path.exists():
        logger.warning(f"File does not exist: {path}")
        raise FileNotFoundError(f"File does not exist: {path}")

    file_ext = path.suffix.lower()

    # For text-based files, try to detect encoding
    text_exts = {".yaml", ".yml", ".sql", ".md", ".csv"}
    if file_ext in text_exts:
        try:
            encoding = detect_encoding(path)
            if encoding is None:
                encoding = "utf-8"  # fallback
        except Exception as e:
            logger.warning(f"Could not detect encoding for {path}: {e}")
            encoding = "utf-8"

    try:
        if file_ext in {".yaml", ".yml"}:
            with open(path, encoding=encoding) as f:
                return yaml.safe_load(f)
        elif file_ext == ".csv":
            csv_kwargs = {"header": 0, "encoding": encoding}
            csv_kwargs.update(kwargs)
            return pd.read_csv(path, **csv_kwargs)
        elif file_ext == ".xlsx":
            return pd.read_excel(path, header=0)
        elif file_ext in {".sql", ".md"}:
            return path.read_text(encoding=encoding)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    except UnicodeDecodeError as ude:
        logger.error(f"Unicode decode error for {path} with encoding {encoding}: {ude}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise


def write_file(
    filepath: Path,
    data,
    *,
    mode: str = "overwrite",
) -> None:
    filepath = Path(filepath).resolve()

    logger.debug(f"write_file '{mode}' mode. file: '{filepath}'")

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        suffix = filepath.suffix.lower()

        if mode not in {"create", "overwrite", "merge"}:
            raise ValueError(f"Invalid mode: {mode}")

        # CREATE
        if mode == "create" and filepath.exists():
            logger.debug(f"File exists, skipping create: {filepath}")
            return

        # YAML FILES
        if suffix in {".yml", ".yaml"}:
            if mode == "merge" and filepath.exists():
                with filepath.open("r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}

                if not isinstance(existing, dict):
                    raise ValueError(f"Cannot merge into non-dict YAML: {filepath}")

                data = deep_merge(existing, data)

            with filepath.open("w", encoding="utf-8") as f:
                yaml.safe_dump(
                    data,
                    f,
                    sort_keys=False,
                    default_flow_style=False,
                    indent=2,
                )
            return

        # TEXT FILES
        if suffix in {".md", ".sh", ".py"}:
            file_mode = "a" if mode == "merge" else "w"
            with filepath.open(file_mode, encoding="utf-8") as f:
                f.write(data)
            return

        # SQL FILES - Don't overwrite.
        if suffix in {".sql"} and filepath.exists():
            return
        elif suffix in {".sql"} and not filepath.exists():
            file_mode = "w"
            with filepath.open(file_mode, encoding="utf-8") as f:
                f.write(data)
            return

        # CSV
        if suffix == ".csv":
            data.to_csv(filepath, index=False)
            return

        raise ValueError(f"Unsupported file type: {suffix}")

    except Exception as e:
        logger.exception(
            "Unexpected error in write_file 'path': {filepath}, 'mode': {mode}"
        )
        raise
