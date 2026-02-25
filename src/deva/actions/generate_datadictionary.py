import argparse
import pandas as pd
import html
import numpy as np
from typing import Optional
from deva.common import convert_df_dtypes, format_dtype_for_display

def generate_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a data dictionary from a DataFrame.

    The data dictionary includes variable name, description (empty), data type, min, max, units (empty), enumerations (empty), and comment (empty).
    """
    data_dict = []
    for col in df.columns:
        series = df[col]
        dtype = format_dtype_for_display(str(series.dtype))

        data_dict.append({
            "variable_name": col,
            "description": "",
            "data_type": dtype,
            "min": "",
            "max": "",
            "units": "",
            "enumerations": "",
            "comment": "",
        })

    return pd.DataFrame(data_dict)
    
def main():
    parser = argparse.ArgumentParser(description="Characterize a datafile and write a simple HTML report.")
    parser.add_argument("-df","--data_file", required=True, help="Path to the datafile (CSV)")
    parser.add_argument("--output", required=False, help="Output data dictionary CSV path")
    args = parser.parse_args()

    file_path = args.data_file
    filename=Path(file_path).stem

    out_path = filename + "_dd" + str(date.today().strftime("%Y%m%d")) + ".csv" if not args.output else args.output

    # For now assume local CSV. In future, support file-location to read from other stores.
    df = pd.read_csv(file_path)

    df = convert_df_dtypes(df)

    print(f"Characterizing '{file_path}' ({df.shape[0]} rows, {df.shape[1]} cols)...")
    data_dict_df = generate_data_dictionary(df)
    data_dict_df.to_csv(out_path, index=False)
    print(f"Wrote data dictionary report to: {out_path}")


if __name__ == "__main__":
    main()