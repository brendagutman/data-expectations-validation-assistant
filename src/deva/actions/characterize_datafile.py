import argparse
import pandas as pd
import html
import numpy as np
from typing import Optional, Dict
from deva.common import convert_df_dtypes, format_dtype_for_display
from pathlib import Path
from datetime import date



def generate_html_report(df: pd.DataFrame, out_path: str, filename: str) -> None:
    """Generate a simple HTML characterization report for a DataFrame.

    The report includes an overview, a per-column summary, and numeric statistics.
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

        # Top values (categorical/high-cardinality safe - show up to 5)
        try:
            top_vals = series.value_counts(dropna=True).head(5)
            top_vals_str = "; ".join([f"{html.escape(str(v))} ({int(c)})" for v, c in top_vals.items()])
        except Exception:
            top_vals_str = ""

        cols.append({
            "column": col,
            "dtype": dtype,
            "non_null": non_null,
            "pct_missing": pct_missing,
            "unique": unique,
            "sample": sample,
            "top_values": top_vals_str,
        })

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
    parser.add_argument("--output", required=False, help="Output HTML path")
    args = parser.parse_args()

    file_path = args.data_file
    filename=Path(file_path).stem

    out_path = filename + "_characterization" + str(date.today().strftime("%Y%m%d")) + ".html" if not args.output else args.output

    # For now assume local CSV. In future, support file-location to read from other stores.
    df = pd.read_csv(file_path)
    
    df = convert_df_dtypes(df)

    print(f"Characterizing '{file_path}' ({df.shape[0]} rows, {df.shape[1]} cols)...")
    generate_html_report(df, out_path, filename)
    print(f"Wrote HTML report to: {out_path}")


if __name__ == "__main__":
    main()