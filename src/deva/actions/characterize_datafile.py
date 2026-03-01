import argparse
import pandas as pd
import html
import numpy as np
import sys
from collections import Counter
from typing import Optional, Dict
from deva.common import convert_df_dtypes, format_dtype_for_display
from pathlib import Path
from datetime import date



def generate_html_report(df: pd.DataFrame, out_path: str, filename: str,
                         dd_dtype_map: Optional[Dict[str, str]] = None,
                         dd_info_map: Optional[Dict[str, Dict[str, any]]] = None,
                         show_enums: Optional[set] = None,
                         hide_enums: Optional[set] = None,
                         command: Optional[str] = None) -> None:
    """Generate a simple HTML characterization report for a DataFrame.

    The report includes an overview, a per-column summary, and numeric statistics.
    If dd_dtype_map is provided, it will include the DD datatype alongside the derived datatype.
    If dd_info_map is provided it will also insert all other DD columns (prefixed with
    "dd_") into the per-column summary.
    """
    n_rows, n_cols = df.shape

    overview = {
        "Filename": filename,
        "Rows": n_rows,
        "Columns": n_cols,
    }

    # Per-column summary
    cols = [] 
    for col in df.columns:
        series = df[col]
        non_null = int(series.notnull().sum())
        pct_missing = round(100.0 * (n_rows - non_null) / n_rows, 2) if n_rows else 0.0
        unique = int(series.nunique(dropna=True))
        dtype = format_dtype_for_display(str(series.dtype))
        sample = ""
        if non_null:
            sample_val = series.dropna().iloc[0]
            sample = html.escape(str(sample_val))

        # Determine enumeration display based on flags
        top_vals_str = ""
        if show_enums and col in show_enums:
            # show all unique values (sorted)
            try:
                uniq = sorted(series.dropna().unique())
                top_vals_str = "; ".join([html.escape(str(v)) for v in uniq])
            except Exception:
                top_vals_str = ""
        elif hide_enums and col in hide_enums:
            top_vals_str = "SKIP"
        else:
            try:
                top_vals = series.value_counts(dropna=True).head(5)
                top_vals_str = "; ".join([f"{html.escape(str(v))} ({int(c)})" for v, c in top_vals.items()])
            except Exception:
                top_vals_str = ""

        col_dict = {
            "column": col,
            "derived_dtype": dtype,
            "non_null": non_null,
            "pct_missing": pct_missing,
            "unique": unique,
            "sample": sample,
            "enums": top_vals_str,
        }

        # Add DD datatype if available
        if dd_dtype_map:
            dd_dtype = dd_dtype_map.get(col, "")
            col_dict["dd_dtype"] = dd_dtype

        # Add any additional DD fields (prefix with dd_)
        if dd_info_map and col in dd_info_map:
            for k, v in dd_info_map[col].items():
                if k == 'variable_name':
                    continue
                col_dict[f"dd_{k}"] = v

        cols.append(col_dict)

    summary_df = pd.DataFrame(cols) 

    # Numeric statistics for numeric columns
    numeric_df: Optional[pd.DataFrame] 
    try:
        numeric_df = df.select_dtypes(include=[np.number]).describe().T
    except Exception:
        numeric_df = None

    # Build HTML
    css = """
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      table { border-collapse: collapse; margin-bottom: 20px; }
      th, td { border: 1px solid #ddd; padding: 8px; }
      th { background: #f2f2f2; }
    </style>
    """

    html_parts = ["<html><head><meta charset='utf-8'><title>Datafile Characterization</title>", css, "</head><body>"]

    html_parts.append(f"<h1>Datafile Characterization</h1>")
    html_parts.append("<h2>Overview</h2>")
    html_parts.append("<ul>")
    html_parts.append(f"<li>Filename: {filename}</li>")
    html_parts.append(f"<li>Rows: {n_rows}</li>")
    html_parts.append(f"<li>Columns: {n_cols}</li>")
    if command:
        html_parts.append(f"<li>Run Command: <code>{html.escape(command)}</code></li>")
    html_parts.append("</ul>")

    html_parts.append("<h2>Per-column summary</h2>")
    html_parts.append(summary_df.to_html(index=False, escape=False))

    if numeric_df is not None and not numeric_df.empty:
        html_parts.append("<h2>Numeric column statistics</h2>")
        html_parts.append(numeric_df.to_html(escape=False))

    html_parts.append("</body></html>")

    full_html = "\n".join(html_parts)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)


def main():
    parser = argparse.ArgumentParser(description="Characterize a datafile and write a simple HTML report.")
    parser.add_argument("-df","--data_file", required=True, help="Path to the datafile (CSV)")
    parser.add_argument("-dd", "--data_dictionary", required=False, help="Path to the data dictionary (CSV)")
    parser.add_argument("--output", required=False, help="Output HTML path")
    parser.add_argument("--show-enums", required=False,
                        help="Comma-separated columns for which full enumerations should be shown")
    parser.add_argument("--hide-enums", required=False,
                        help="Comma-separated columns for which enumerations should be suppressed")
    parser.add_argument("--chunksize", required=False, type=int, default=0,
                        help="Read the datafile in chunks of this size (streaming mode when >0).")
    parser.add_argument("--low-memory", action="store_true", help="Pass low_memory=True to pandas.read_csv (less memory at cost of type inference).")
    args = parser.parse_args()

    file_path = args.data_file
    filename = Path(file_path).stem

    out_path = filename + "_characterization" + str(date.today().strftime("%Y%m%d")) + ".html" if not args.output else args.output

    # For now assume local CSV. In future, support file-location to read from other stores.
    chunksize = int(args.chunksize) if args.chunksize is not None else 0
    low_memory_flag = bool(args.low_memory)

    df = None
    if chunksize <= 0:
        df = pd.read_csv(file_path, low_memory=low_memory_flag)
        df = convert_df_dtypes(df)
    
    # Load data dictionary if provided
    dd_dtype_map: Optional[Dict[str, str]] = None
    dd_info_map: Optional[Dict[str, Dict[str, any]]] = None
    if args.data_dictionary:
        dd = pd.read_csv(args.data_dictionary)
        # Create a mapping from variable_name to data_type
        if 'variable_name' in dd.columns and 'data_type' in dd.columns:
            dd_dtype_map = dict(zip(dd['variable_name'], dd['data_type']))
        # also build a full info map keyed by variable_name
        if 'variable_name' in dd.columns:
            dd_info_map = dd.set_index('variable_name').to_dict(orient='index')

    # process enumeration flags
    show_enums: Optional[set] = None
    hide_enums: Optional[set] = None
    if args.show_enums and args.hide_enums:
        parser.error("--show-enums and --hide-enums cannot be used together")
    if args.show_enums:
        show_enums = {c.strip() for c in args.show_enums.split(',') if c.strip()}
    if args.hide_enums:
        hide_enums = {c.strip() for c in args.hide_enums.split(',') if c.strip()}

    # Build command string from sys.argv
    command_str = " ".join(sys.argv)

    # If streaming requested, compute summaries by chunking
    if chunksize > 0:
        print(f"Streaming-characterizing '{file_path}' with chunksize={chunksize}...")
        reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=low_memory_flag)
        total_rows = 0
        stats = {}
        cols = None
        UNIQUE_TRACK_CAP = 50000
        SAMPLE_CAP = 200

        def infer_dtype_from_samples(samples):
            # simple inference from sample strings
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

        for chunk in reader:
            if cols is None:
                cols = list(chunk.columns)
                for c in cols:
                    stats[c] = {
                        'non_null': 0,
                        'counter': Counter(),
                        'sample': None,
                        'sample_vals': set(),
                        'unique_truncated': False,
                    }
            total_rows += len(chunk)
            for c in cols:
                s = chunk[c]
                non_null = int(s.notna().sum())
                stats[c]['non_null'] += non_null

                if stats[c]['sample'] is None:
                    non_null_series = s.dropna()
                    if not non_null_series.empty:
                        stats[c]['sample'] = non_null_series.iloc[0]

                # update counters (stringified) for top values
                try:
                    vals = s.dropna().astype(str).tolist()
                except Exception:
                    vals = [str(v) for v in s.dropna().tolist()]
                stats[c]['counter'].update(vals)

                # update small set of unique samples
                if not stats[c]['unique_truncated']:
                    uniques_in_chunk = pd.unique(s.dropna())
                    for v in uniques_in_chunk:
                        if len(stats[c]['sample_vals']) < SAMPLE_CAP:
                            stats[c]['sample_vals'].add(v)
                        else:
                            stats[c]['unique_truncated'] = True
                            break

        # build summary rows
        summary_rows = []
        for c in cols:
            non_null = stats[c]['non_null']
            pct_missing = round(100.0 * (total_rows - non_null) / total_rows, 2) if total_rows else 0.0
            unique = len(stats[c]['sample_vals']) if not stats[c]['unique_truncated'] else len(stats[c]['sample_vals'])
            sample = html.escape(str(stats[c]['sample'])) if stats[c]['sample'] is not None else ""

            # enums/enumeration display
            enums_str = ""
            if show_enums and c in show_enums:
                try:
                    uniq = sorted([str(x) for x in stats[c]['sample_vals']])
                    enums_str = "; ".join([html.escape(str(v)) for v in uniq])
                except Exception:
                    enums_str = ""
            elif hide_enums and c in hide_enums:
                enums_str = "SKIP"
            else:
                # top 5 from counter
                try:
                    top_vals = stats[c]['counter'].most_common(5)
                    enums_str = "; ".join([f"{html.escape(str(v))} ({int(cnt)})" for v, cnt in top_vals])
                except Exception:
                    enums_str = ""

            # infer dtype from sample_vals (fallback to string)
            sample_vals = list(stats[c]['sample_vals'])[:20]
            dtype = infer_dtype_from_samples(sample_vals)

            row = {
                'column': c,
                'derived_dtype': dtype,
                'non_null': int(non_null),
                'pct_missing': pct_missing,
                'unique': unique,
                'sample': sample,
                'enums': enums_str,
            }

            # dd fields
            if dd_dtype_map:
                row['dd_dtype'] = dd_dtype_map.get(c, "")
            if dd_info_map and c in dd_info_map:
                for k, v in dd_info_map[c].items():
                    if k == 'variable_name':
                        continue
                    row[f'dd_{k}'] = v

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        numeric_df = None

        # call report generator with precomputed summary
        generate_html_report(None, out_path, filename, dd_dtype_map, dd_info_map,
                             show_enums=show_enums, hide_enums=hide_enums,
                             command=command_str, summary_df=summary_df, numeric_df=numeric_df)
        print(f"Wrote HTML report to: {out_path}")
        return

    # Non-streaming path
    print(f"Characterizing '{file_path}' ({df.shape[0]} rows, {df.shape[1]} cols)...")
    
    generate_html_report(df, out_path, filename, dd_dtype_map, dd_info_map,
                         show_enums=show_enums, hide_enums=hide_enums,
                         command=command_str)
    print(f"Wrote HTML report to: {out_path}")


if __name__ == "__main__":
    main()