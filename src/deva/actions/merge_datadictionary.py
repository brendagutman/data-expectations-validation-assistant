import argparse
import pandas as pd
from pathlib import Path
from datetime import date
from typing import Optional
from deva.common import convert_df_dtypes
from deva.actions.generate_datadictionary import generate_data_dictionary

# Fields that are auto-generated and should be compared for differences.
GENERATED_FIELDS = ["data_type", "min", "max", "enumerations"]

# Canonical column name -> list of accepted aliases (after lowering + underscore normalization)
COLUMN_ALIASES = {
    "variable_name": ["variable_name", "variable", "field_name", "field", "column_name", "column"],
    "description":   ["description", "label", "desc"],
    "data_type":     ["data_type", "datatype", "type", "dtype"],
    "min":           ["min", "minimum"],
    "max":           ["max", "maximum"],
    "units":         ["units", "unit"],
    "enumerations":  ["enumerations", "enums", "enum", "enumeration", "permitted_values", "values"],
    "comment":       ["comment", "comments", "notes", "note"],
}

# Reverse lookup: alias -> canonical name
_ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}


def _normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_dd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns in an existing data dictionary to canonical names."""
    existing_cols = set(df.columns)
    rename_map = {}
    for col in df.columns:
        normalized = _normalize_col(col)
        canonical = _ALIAS_TO_CANONICAL.get(normalized)
        if canonical and canonical != col and canonical not in existing_cols:
            rename_map[col] = canonical
            existing_cols.add(canonical)
    if rename_map:
        print(f"Normalizing column names: {rename_map}")
    return df.rename(columns=rename_map)


def merge_data_dictionaries(generated: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Merge a generated data dictionary with an existing one.

    For each variable in the existing DD:
      - Keep the existing row as the primary.
      - If the generated DD has the same variable and any GENERATED_FIELDS differ,
        also include the generated row with column names suffixed with '_gen'.
    Variables in the generated DD that are not in the existing DD are appended at the end.
    """
    gen_indexed = generated.set_index("variable_name")
    existing_cols = list(existing.columns)
    # Columns in the generated DD that are missing from the existing DD — add as regular columns
    missing_gen_cols = [c for c in generated.columns if c != "variable_name" and c not in existing_cols]
    # _gen columns only for GENERATED_FIELDS that already exist in the existing DD
    gen_suffix_cols = {c: f"{c}_gen" for c in GENERATED_FIELDS if c in existing_cols}

    rows = []
    seen = set()
    any_varname_mismatch = False

    for _, existing_row in existing.iterrows():
        var = existing_row["variable_name"]
        seen.add(var)
        row = existing_row.to_dict()

        if var in gen_indexed.index:
            gen_row = gen_indexed.loc[var]
            # Fill in missing generated columns as regular columns
            for col in missing_gen_cols:
                row[col] = gen_row.get(col, "")
            # Add _gen value only when the existing field disagrees with generated
            for field, gen_col in gen_suffix_cols.items():
                existing_val = str(existing_row.get(field, ""))
                gen_val = str(gen_row.get(field, ""))
                if existing_val != gen_val:
                    row[gen_col] = gen_val
        else:
            # Variable not found in generated DD — flag it
            row["variable_name_gen"] = "NOT FOUND IN DATA"
            any_varname_mismatch = True
            for col in missing_gen_cols:
                row[col] = ""

        rows.append(row)

    # Append variables only in the generated DD — not in the existing DD at all
    for var in gen_indexed.index:
        if var not in seen:
            gen_row = gen_indexed.loc[var]
            new_entry = {"variable_name": var, "variable_name_gen": "NOT IN DICTIONARY"}
            any_varname_mismatch = True
            for col in existing_cols:
                if col != "variable_name":
                    new_entry[col] = gen_row.get(col, "")
            for col in missing_gen_cols:
                new_entry[col] = gen_row.get(col, "")
            rows.append(new_entry)

    # Only include _gen columns that have at least one disagreement
    gen_cols_used = []
    if any_varname_mismatch:
        gen_cols_used.append("variable_name_gen")
    gen_cols_used += [c for c in gen_suffix_cols.values() if any(c in r for r in rows)]

    # Column order: expected DD columns first, then extra columns from existing DD, then _gen columns
    expected_cols = list(generated.columns)  # variable_name, description, data_type, min, max, units, enumerations, comment
    extra_existing_cols = [c for c in existing_cols if c not in expected_cols]
    all_cols = expected_cols + extra_existing_cols + gen_cols_used
    return pd.DataFrame(rows, columns=all_cols)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a data dictionary from a datafile and optionally merge with an existing one."
    )
    parser.add_argument("-df", "--data_file", required=True, help="Path to the datafile (CSV)")
    parser.add_argument("-dd", "--data_dictionary", required=False, help="Path to an existing data dictionary CSV")
    parser.add_argument("-o", "--output", required=False, help="Output CSV path")
    parser.add_argument("--show-enums", required=False,
                        help="Comma-separated columns for which enumerations should be included")
    parser.add_argument("--hide-enums", required=False,
                        help="Comma-separated columns for which enumerations should be suppressed")
    args = parser.parse_args()

    file_path = args.data_file
    filename = Path(file_path).stem
    out_path = args.output or f"{filename}_merged_dd{date.today().strftime('%Y%m%d')}.csv"

    if args.show_enums and args.hide_enums:
        parser.error("--show-enums and --hide-enums cannot be used together")

    show_enums: Optional[set] = {c.strip() for c in args.show_enums.split(",") if c.strip()} if args.show_enums else None
    hide_enums: Optional[set] = {c.strip() for c in args.hide_enums.split(",") if c.strip()} if args.hide_enums else None

    df = pd.read_csv(file_path)
    df = convert_df_dtypes(df)
    print(f"Generating data dictionary for '{file_path}' ({df.shape[0]} rows, {df.shape[1]} cols)...")
    generated = generate_data_dictionary(df, show_enums=show_enums, hide_enums=hide_enums)

    if args.data_dictionary:
        existing = pd.read_csv(args.data_dictionary)
        existing = normalize_dd_columns(existing)
        if "variable_name" not in existing.columns:
            parser.error(
                f"Could not find a variable name column in '{args.data_dictionary}'. "
                f"Expected one of: {COLUMN_ALIASES['variable_name']}"
            )
        print(f"Merging with existing data dictionary '{args.data_dictionary}'...")
        result = merge_data_dictionaries(generated, existing)
    else:
        result = generated

    result.to_csv(out_path, index=False)
    print(f"Wrote merged data dictionary to: {out_path}")


if __name__ == "__main__":
    main()
