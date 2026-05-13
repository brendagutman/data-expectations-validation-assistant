from collections import Counter
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from deva.common import (
    convert_df_dtypes,
    format_dtype_for_display,
    infer_dtype_from_samples,
    read_file,
)


DEFAULT_AUTO_ENUM_MAX = 50
DEFAULT_UNIQUE_CAP = 200


def parse_enum_filters(
    show_enums_arg: Optional[str],
    hide_enums_arg: Optional[str],
) -> Tuple[Optional[set], Optional[set]]:
    """Parse comma-separated enum filter args into sets.

    Raises:
        ValueError: if both show and hide filters are provided.
    """
    if show_enums_arg and hide_enums_arg:
        raise ValueError("--show-enums and --hide-enums cannot be used together")

    show_enums = (
        {c.strip() for c in show_enums_arg.split(",") if c.strip()}
        if show_enums_arg
        else None
    )
    hide_enums = (
        {c.strip() for c in hide_enums_arg.split(",") if c.strip()}
        if hide_enums_arg
        else None
    )
    return show_enums, hide_enums


def _build_enumerations(
    values,
    column_name: str,
    show_enums: Optional[set],
    hide_enums: Optional[set],
    auto_enum_max: int,
) -> str:
    unique_count = len(values)

    if show_enums and column_name in show_enums:
        try:
            return ";".join([str(v) for v in sorted(values)])
        except Exception:
            return ""

    if hide_enums and column_name in hide_enums:
        return ""

    if show_enums:
        return ""

    if unique_count > 0 and unique_count <= auto_enum_max and (not hide_enums or column_name not in hide_enums):
        try:
            return ";".join([str(v) for v in sorted(values)])
        except Exception:
            return ""

    return ""


def _numeric_bounds_from_series(series: pd.Series, enums_str: str):
    min_val = ""
    max_val = ""
    if pd.api.types.is_numeric_dtype(series) and not enums_str:
        try:
            min_val = series.min(skipna=True)
            max_val = series.max(skipna=True)
            if pd.isna(min_val):
                min_val = ""
            if pd.isna(max_val):
                max_val = ""
        except Exception:
            pass
    return min_val, max_val


def _build_row(variable_name: str, dtype: str, min_val, max_val, enums_str: str) -> dict:
    final_dtype = "enumeration" if enums_str else dtype
    return {
        "variable_name": variable_name,
        "description": "",
        "data_type": final_dtype,
        "min": min_val,
        "max": max_val,
        "units": "",
        "enumerations": enums_str,
        "comment": "",
    }


def generate_data_dictionary_from_dataframe(
    df: pd.DataFrame,
    show_enums: Optional[set] = None,
    hide_enums: Optional[set] = None,
    auto_enum_max: int = DEFAULT_AUTO_ENUM_MAX,
) -> pd.DataFrame:
    """Generate a data dictionary from an in-memory DataFrame."""
    rows = []
    for col in df.columns:
        series = df[col]
        dtype = format_dtype_for_display(str(series.dtype))
        unique_vals = list(pd.unique(series.dropna()))
        enums_str = _build_enumerations(
            unique_vals,
            col,
            show_enums,
            hide_enums,
            auto_enum_max,
        )
        min_val, max_val = _numeric_bounds_from_series(series, enums_str)
        rows.append(_build_row(col, dtype, min_val, max_val, enums_str))

    return pd.DataFrame(rows)


def _stream_column_stats(file_path: Path, chunksize: int, low_memory: bool, unique_cap: int):
    reader_obj = read_file(file_path, chunksize=chunksize, low_memory=low_memory)
    if isinstance(reader_obj, pd.DataFrame):
        reader = (reader_obj.iloc[i:i + chunksize] for i in range(0, len(reader_obj), chunksize))
    else:
        reader = reader_obj

    cols = None
    stats = {}

    for chunk in reader:
        if cols is None:
            cols = list(chunk.columns)
            for c in cols:
                stats[c] = {
                    "counter": Counter(),
                    "sample_vals": set(),
                    "min": None,
                    "max": None,
                    "unique_truncated": False,
                }

        for c in cols:
            s = chunk[c]
            try:
                vals = s.dropna().astype(str).tolist()
            except Exception:
                vals = [str(v) for v in s.dropna().tolist()]
            stats[c]["counter"].update(vals)

            if not stats[c]["unique_truncated"]:
                uniqs = pd.unique(s.dropna())
                for v in uniqs:
                    if len(stats[c]["sample_vals"]) < unique_cap:
                        stats[c]["sample_vals"].add(v)
                    else:
                        stats[c]["unique_truncated"] = True
                        break

            if pd.api.types.is_numeric_dtype(s):
                try:
                    mn = s.min(skipna=True)
                    mx = s.max(skipna=True)
                except Exception:
                    mn = None
                    mx = None
                if mn is not None and (stats[c]["min"] is None or mn < stats[c]["min"]):
                    stats[c]["min"] = mn
                if mx is not None and (stats[c]["max"] is None or mx > stats[c]["max"]):
                    stats[c]["max"] = mx

    if cols is None:
        raise ValueError(f"No rows found in input file: {file_path}")

    return cols, stats


def generate_data_dictionary_from_file(
    file_path: Path,
    show_enums: Optional[set] = None,
    hide_enums: Optional[set] = None,
    chunksize: int = 0,
    low_memory: bool = False,
    auto_enum_max: int = DEFAULT_AUTO_ENUM_MAX,
    unique_cap: int = DEFAULT_UNIQUE_CAP,
) -> pd.DataFrame:
    """Generate a data dictionary from file, with optional chunked processing."""
    if chunksize > 0:
        cols, stats = _stream_column_stats(file_path, chunksize, low_memory, unique_cap)
        rows = []
        for c in cols:
            sample_vals = list(stats[c]["sample_vals"])
            dtype = format_dtype_for_display(infer_dtype_from_samples(sample_vals[:20]))
            min_val = stats[c]["min"] if stats[c]["min"] is not None else ""
            max_val = stats[c]["max"] if stats[c]["max"] is not None else ""

            enums_str = _build_enumerations(
                sample_vals,
                c,
                show_enums,
                hide_enums,
                auto_enum_max,
            )
            rows.append(_build_row(c, dtype, min_val, max_val, enums_str))

        return pd.DataFrame(rows)

    df = read_file(file_path)
    df = convert_df_dtypes(df)
    return generate_data_dictionary_from_dataframe(
        df,
        show_enums=show_enums,
        hide_enums=hide_enums,
        auto_enum_max=auto_enum_max,
    )
