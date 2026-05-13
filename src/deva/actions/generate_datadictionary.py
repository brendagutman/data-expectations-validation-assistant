import argparse
import pandas as pd
from typing import Optional
from deva.common import write_file
from deva.datadictionary_utils import (
    generate_data_dictionary_from_dataframe,
    generate_data_dictionary_from_file,
    parse_enum_filters,
)
from pathlib import Path


def generate_data_dictionary(df: pd.DataFrame,
                             show_enums: Optional[set] = None,
                             hide_enums: Optional[set] = None) -> pd.DataFrame:
    """Backward-compatible wrapper around shared DD generation logic."""
    return generate_data_dictionary_from_dataframe(
        df,
        show_enums=show_enums,
        hide_enums=hide_enums,
    )

def main():
    parser = argparse.ArgumentParser(description="Generate a data dictionary from a datafile.")
    parser.add_argument("-df","--data_file", required=True, help="Path to the datafile (CSV or Excel)")
    parser.add_argument("-o", "--output", required=False, help="Output data dictionary CSV path")
    parser.add_argument("--show-enums", required=False,
                        help="Comma-separated columns for which enumerations should be included")
    parser.add_argument("--hide-enums", required=False,
                        help="Comma-separated columns for which enumerations should be suppressed")
    parser.add_argument("--chunksize", required=False, type=int, default=0,
                        help="Process the datafile in chunks instead of all at once")
    parser.add_argument("--low-memory", action="store_true",
                        help="Pass low_memory=True to pandas.read_csv (uses less memory at cost of dtype inference)")
    args = parser.parse_args()

    file_path = Path(args.data_file)
    filename = file_path.stem
    file_dir = file_path.parent
    file_ext = file_path.suffix.lower()

    out_path = args.output or file_dir / f"deva_files/{filename}_Dictionary.csv"
    output_dir = Path(out_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Supports local CSV and Excel files.
    chunksize = int(args.chunksize) if args.chunksize is not None else 0
    low_memory_flag = bool(args.low_memory)

    try:
        show_enums, hide_enums = parse_enum_filters(args.show_enums, args.hide_enums)
    except ValueError as exc:
        parser.error(str(exc))

    if chunksize > 0:
        print(
            f"Streaming-generating data dictionary for '{file_path}' (chunksize={chunksize})..."
        )
    else:
        print(f"Generating data dictionary for '{file_path}'...")

    data_dict_df = generate_data_dictionary_from_file(
        file_path,
        show_enums=show_enums,
        hide_enums=hide_enums,
        chunksize=chunksize,
        low_memory=low_memory_flag,
    )
    write_file(out_path, data_dict_df)
    print(f"Wrote data dictionary to: {out_path}")


if __name__ == "__main__":
    main()
