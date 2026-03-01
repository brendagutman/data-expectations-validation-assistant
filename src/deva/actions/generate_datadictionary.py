import argparse
import pandas as pd
import html
import numpy as np
import sys
from collections import Counter
from typing import Optional
from deva.common import convert_df_dtypes, format_dtype_for_display, infer_dtype_from_samples
from pathlib import Path
from datetime import date

def generate_data_dictionary(df: pd.DataFrame,
                             show_enums: Optional[set] = None,
                             hide_enums: Optional[set] = None) -> pd.DataFrame:
    """Generate a data dictionary from a DataFrame.

    The data dictionary includes variable name, description (empty), data type, min, max, units (empty), enumerations, and comment (empty).
    
    If show_enums is provided, populate enumerations for those columns with all unique values.
    If hide_enums is provided, skip enumeration population for those columns.
    """
    # When neither show_enums nor hide_enums are provided, auto-populate
    # enumerations for low-cardinality columns. Adjust this threshold as needed.
    AUTO_ENUM_MAX = 50

    data_dict = []
    for col in df.columns:
        series = df[col]
        dtype = format_dtype_for_display(str(series.dtype))
        
        enums_str = ""
        unique_count = int(series.dropna().nunique()) if series.notna().any() else 0

        if show_enums and col in show_enums:
            # show all unique values (sorted), separated by semicolon
            try:
                uniq = sorted(series.dropna().unique())
                enums_str = ";".join([str(v) for v in uniq])
            except Exception:
                enums_str = ""
        elif hide_enums and col in hide_enums:
            enums_str = ""
        else:
            # If explicit show_enums was provided we only populate those.
            # Otherwise auto-populate for low-cardinality columns, but respect hide_enums.
            if show_enums:
                enums_str = ""
            else:
                if unique_count > 0 and unique_count <= AUTO_ENUM_MAX and (not hide_enums or col not in hide_enums):
                    try:
                        uniq = sorted(series.dropna().unique())
                        enums_str = ";".join([str(v) for v in uniq])
                    except Exception:
                        enums_str = ""
                else:
                    enums_str = ""

        data_dict.append({
            "variable_name": col,
            "description": "",
            "data_type": dtype,
            "min": "",
            "max": "",
            "units": "",
            "enumerations": enums_str,
            "comment": "",
        })

    return pd.DataFrame(data_dict)
    
def main():
    parser = argparse.ArgumentParser(description="Generate a data dictionary from a datafile.")
    parser.add_argument("-df","--data_file", required=True, help="Path to the datafile (CSV)")
    parser.add_argument("-o", "--output", required=False, help="Output data dictionary CSV path")
    parser.add_argument("--show-enums", required=False,
                        help="Comma-separated columns for which enumerations should be included")
    parser.add_argument("--hide-enums", required=False,
                        help="Comma-separated columns for which enumerations should be suppressed")
    parser.add_argument("--chunksize", required=False, type=int, default=0,
                        help="Read the datafile in chunks (streaming) instead of loading whole file")
    parser.add_argument("--low-memory", action="store_true",
                        help="Pass low_memory=True to pandas.read_csv (uses less memory at cost of dtype inference)")
    args = parser.parse_args()

    file_path = args.data_file
    filename = Path(file_path).stem

    out_path = filename + "_dd" + str(date.today().strftime("%Y%m%d")) + ".csv" if not args.output else args.output

    # For now assume local CSV. In future, support file-location to read from other stores.
    chunksize = int(args.chunksize) if args.chunksize is not None else 0
    low_memory_flag = bool(args.low_memory)

    # process enumeration flags
    show_enums: Optional[set] = None
    hide_enums: Optional[set] = None
    if args.show_enums and args.hide_enums:
        parser.error("--show-enums and --hide-enums cannot be used together")
    if args.show_enums:
        show_enums = {c.strip() for c in args.show_enums.split(',') if c.strip()}
    if args.hide_enums:
        hide_enums = {c.strip() for c in args.hide_enums.split(',') if c.strip()}

    # if streaming requested, aggregate stats chunk-by-chunk
    if chunksize > 0:
        print(f"Streaming-generating data dictionary for '{file_path}' (chunksize={chunksize})...")
        reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=low_memory_flag)
        total_rows = 0
        stats = {}
        cols = None
        UNIQUE_CAP = 200
        AUTO_ENUM_MAX = 50

        for chunk in reader:
            if cols is None:
                cols = list(chunk.columns)
                for c in cols:
                    stats[c] = {
                        'counter': Counter(),
                        'sample_vals': set(),
                        'sample': None,
                        'min': None,
                        'max': None,
                        'unique_truncated': False,
                    }
            total_rows += len(chunk)
            for c in cols:
                s = chunk[c]
                # sample value
                if stats[c]['sample'] is None:
                    non_null = s.dropna()
                    if not non_null.empty:
                        stats[c]['sample'] = non_null.iloc[0]
                # counters for enums
                try:
                    vals = s.dropna().astype(str).tolist()
                except Exception:
                    vals = [str(v) for v in s.dropna().tolist()]
                stats[c]['counter'].update(vals)
                # unique sample values
                if not stats[c]['unique_truncated']:
                    uniqs = pd.unique(s.dropna())
                    for v in uniqs:
                        if len(stats[c]['sample_vals']) < UNIQUE_CAP:
                            stats[c]['sample_vals'].add(v)
                        else:
                            stats[c]['unique_truncated'] = True
                            break
                # min/max for numeric
                if pd.api.types.is_numeric_dtype(s):
                    try:
                        mn = s.min(skipna=True)
                        mx = s.max(skipna=True)
                    except Exception:
                        mn = None; mx = None
                    if mn is not None:
                        if stats[c]['min'] is None or mn < stats[c]['min']:
                            stats[c]['min'] = mn
                    if mx is not None:
                        if stats[c]['max'] is None or mx > stats[c]['max']:
                            stats[c]['max'] = mx
        # build dataframe
        rows = []
        for c in cols:
            sample_vals = list(stats[c]['sample_vals'])
            dtype = infer_dtype_from_samples(sample_vals[:20])
            min_val = stats[c]['min'] if stats[c]['min'] is not None else ""
            max_val = stats[c]['max'] if stats[c]['max'] is not None else ""
            unique_count = len(sample_vals) if not stats[c]['unique_truncated'] else len(sample_vals)

            enums_str = ""
            if show_enums and c in show_enums:
                try:
                    uniq = sorted(sample_vals)
                    enums_str = ";".join([str(v) for v in uniq])
                except Exception:
                    enums_str = ""
            elif hide_enums and c in hide_enums:
                enums_str = ""
            else:
                if show_enums:
                    enums_str = ""
                else:
                    if unique_count > 0 and unique_count <= AUTO_ENUM_MAX and (not hide_enums or c not in hide_enums):
                        try:
                            uniq = sorted(sample_vals)
                            enums_str = ";".join([str(v) for v in uniq])
                        except Exception:
                            enums_str = ""
                    else:
                        enums_str = ""

            rows.append({
                "variable_name": c,
                "description": "",
                "data_type": dtype,
                "min": min_val,
                "max": max_val,
                "units": "",
                "enumerations": enums_str,
                "comment": "",
            })

        data_dict_df = pd.DataFrame(rows)
        data_dict_df.to_csv(out_path, index=False)
        print(f"Wrote data dictionary to: {out_path}")
        return

    # non-streaming path
    df = pd.read_csv(file_path, low_memory=low_memory_flag)

    df = convert_df_dtypes(df)
    print(f"Generating data dictionary for '{file_path}' ({df.shape[0]} rows, {df.shape[1]} cols)...")
    data_dict_df = generate_data_dictionary(df, show_enums=show_enums, hide_enums=hide_enums)
    data_dict_df.to_csv(out_path, index=False)
    print(f"Wrote data dictionary to: {out_path}")


if __name__ == "__main__":
    main()