import argparse
import pandas as pd
import html
import numpy as np
import sys
from typing import Optional
from deva.common import convert_df_dtypes, format_dtype_for_display
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
    parser.add_argument("--output", required=False, help="Output data dictionary CSV path")
    parser.add_argument("--show-enums", required=False,
                        help="Comma-separated columns for which enumerations should be included")
    parser.add_argument("--hide-enums", required=False,
                        help="Comma-separated columns for which enumerations should be suppressed")
    args = parser.parse_args()

    file_path = args.data_file
    filename=Path(file_path).stem

    out_path = filename + "_dd" + str(date.today().strftime("%Y%m%d")) + ".csv" if not args.output else args.output

    # For now assume local CSV. In future, support file-location to read from other stores.
    df = pd.read_csv(file_path)

    df = convert_df_dtypes(df)
    
    # process enumeration flags
    show_enums: Optional[set] = None
    hide_enums: Optional[set] = None
    if args.show_enums and args.hide_enums:
        parser.error("--show-enums and --hide-enums cannot be used together")
    if args.show_enums:
        show_enums = {c.strip() for c in args.show_enums.split(',') if c.strip()}
    if args.hide_enums:
        hide_enums = {c.strip() for c in args.hide_enums.split(',') if c.strip()}

    print(f"Generating data dictionary for '{file_path}' ({df.shape[0]} rows, {df.shape[1]} cols)...")
    data_dict_df = generate_data_dictionary(df, show_enums=show_enums, hide_enums=hide_enums)
    data_dict_df.to_csv(out_path, index=False)
    print(f"Wrote data dictionary to: {out_path}")


if __name__ == "__main__":
    main()