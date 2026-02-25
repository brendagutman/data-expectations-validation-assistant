import argparse
import pandas as pd
import numpy as np
import html
from typing import Dict, List, Tuple, Any
from deva.common import convert_df_dtypes, format_dtype_for_display


def parse_enumerations(enum_str: str) -> set:
    """Parse enumeration string (comma or semicolon separated) into a set."""
    if not enum_str or pd.isna(enum_str):
        return set()
    enum_str = str(enum_str).strip()
    if not enum_str:
        return set()
    # Try semicolon first, then comma
    if ';' in enum_str:
        return {v.strip() for v in enum_str.split(';')}
    else:
        return {v.strip() for v in enum_str.split(',')}


def validate_column_exists(dd_row: pd.Series, df: pd.DataFrame) -> Tuple[bool, str]:
    """Check if column exists in datafile."""
    col_name = dd_row['variable_name']
    if col_name not in df.columns:
        return False, f"Column '{col_name}' not found in datafile"
    return True, ""


def can_cast_to_dtype(series: pd.Series, target_dtype: str) -> bool:
    """Check if a series can be cast to the target datatype without losing data."""
    target_dtype = target_dtype.lower().strip()
    
    try:
        if target_dtype in ('integer', 'int', 'int64'):
            # Try to cast to int64
            non_null = series.dropna()
            if len(non_null) == 0:
                return True
            non_null.astype('int64')
            return True
        elif target_dtype in ('float', 'float64'):
            non_null = series.dropna()
            if len(non_null) == 0:
                return True
            non_null.astype('float64')
            return True
        elif target_dtype in ('string', 'object', 'str'):
            # Everything can be cast to string
            return True
        elif target_dtype in ('boolean', 'bool'):
            non_null = series.dropna()
            if len(non_null) == 0:
                return True
            # Check if values are boolean-like
            valid_bool = non_null.isin([True, False, 'true', 'false', 'True', 'False', 'TRUE', 'FALSE', 0, 1, '0', '1']).all()
            return valid_bool
        else:
            return True
    except (ValueError, TypeError):
        return False


def validate_dtype(dd_row: pd.Series, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], int]:
    """Check if column can be cast to expected dtype."""
    col_name = dd_row['variable_name']
    expected_dtype = str(dd_row.get('data_type', '')).lower().strip()
    
    if not expected_dtype or col_name not in df.columns:
        return [], 0
    
    series = df[col_name]
    actual_dtype = format_dtype_for_display(str(series.dtype)).lower()
    
    failures = []
    # Only fail if the column cannot be cast to the expected dtype
    if actual_dtype != expected_dtype and not can_cast_to_dtype(series, expected_dtype):
        failures.append({
            'column': col_name,
            'check': 'data_type',
            'expected': expected_dtype,
            'actual': actual_dtype,
            'detail': f"Cannot cast {actual_dtype} to {expected_dtype}"
        })
    
    return failures, len(failures)


def validate_enumerations(dd_row: pd.Series, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], int]:
    """Check if all values in column are in allowed enumerations."""
    col_name = dd_row['variable_name']
    enum_str = dd_row.get('enumerations', '')
    
    if not enum_str or pd.isna(enum_str) or col_name not in df.columns:
        return [], 0
    
    allowed_values = parse_enumerations(enum_str)
    if not allowed_values:
        return [], 0
    
    series = df[col_name]
    # Get non-null unique values
    unique_vals = series.dropna().unique()
    
    failures = []
    invalid_vals = []
    for val in unique_vals:
        val_str = str(val)
        if val_str not in allowed_values:
            invalid_vals.append(val_str)
    
    if invalid_vals:
        count = sum(series.isin([float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v for v in invalid_vals]).sum() 
                   for v in invalid_vals 
                   if not pd.isna(v))
        count = series.isin(invalid_vals).sum()
        failures.append({
            'column': col_name,
            'check': 'enumerations',
            'expected': enum_str,
            'actual': ', '.join(invalid_vals),
            'detail': f"Found {count} rows with invalid enumeration values: {', '.join(invalid_vals)}"
        })
    
    return failures, len(failures)


def validate_range(dd_row: pd.Series, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], int]:
    """Check if numeric values are within min/max range."""
    col_name = dd_row['variable_name']
    
    if col_name not in df.columns:
        return [], 0
    
    series = df[col_name].dropna()
    if series.empty or not pd.api.types.is_numeric_dtype(series):
        return [], 0
    
    failures = []
    min_val = dd_row.get('min', '')
    max_val = dd_row.get('max', '')
    
    if min_val and not pd.isna(min_val):
        try:
            min_threshold = float(min_val)
            too_small = (series < min_threshold).sum()
            if too_small > 0:
                failures.append({
                    'column': col_name,
                    'check': 'min_value',
                    'expected': f">= {min_val}",
                    'actual': f"min: {series.min()}",
                    'detail': f"Found {too_small} rows below minimum value {min_val}"
                })
        except (ValueError, TypeError):
            pass
    
    if max_val and not pd.isna(max_val):
        try:
            max_threshold = float(max_val)
            too_large = (series > max_threshold).sum()
            if too_large > 0:
                failures.append({
                    'column': col_name,
                    'check': 'max_value',
                    'expected': f"<= {max_val}",
                    'actual': f"max: {series.max()}",
                    'detail': f"Found {too_large} rows above maximum value {max_val}"
                })
        except (ValueError, TypeError):
            pass
    
    return failures, len(failures)


def validate_dd_columns_not_in_datafile(dd_df: pd.DataFrame, data_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check for columns in data dictionary that are not in the datafile."""
    failures = []
    dd_cols = set(dd_df['variable_name'])
    df_cols = set(data_df.columns)
    
    missing_cols = dd_cols - df_cols
    for col in sorted(missing_cols):
        failures.append({
            'column': col,
            'check': 'dd_columns_missing_in_datafile',
            'expected': f"Column '{col}' in datafile",
            'actual': 'Not found',
            'detail': f"Column '{col}' is defined in data dictionary but not found in datafile"
        })
    
    return failures


def validate_datafile_columns_not_in_dd(dd_df: pd.DataFrame, data_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check for columns in datafile that are not in the data dictionary."""
    failures = []
    dd_cols = set(dd_df['variable_name'])
    df_cols = set(data_df.columns)
    
    extra_cols = df_cols - dd_cols
    for col in sorted(extra_cols):
        failures.append({
            'column': col,
            'check': 'datafile_columns_not_in_dd',
            'expected': f"Column '{col}' in data dictionary",
            'actual': 'Not defined',
            'detail': f"Column '{col}' found in datafile but not defined in data dictionary"
        })
    
    return failures


def generate_validation_html(results: Dict[str, Any], out_path: str) -> None:
    """Generate HTML validation report."""
    css = """
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }
      .summary { background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
      .summary h2 { margin: 0 0 10px 0; }
      .summary-item { display: inline-block; margin-right: 20px; }
      .summary-item strong { color: #d9534f; }
      table { border-collapse: collapse; margin-bottom: 20px; background: white; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
      th { background: #34495e; color: white; }
      tr:nth-child(even) { background: #f2f2f2; }
      .pass { color: #27ae60; }
      .fail { color: #d9534f; font-weight: bold; }
      h1 { color: #34495e; }
      h2 { color: #34495e; margin-top: 30px; }
      .detail { font-size: 0.9em; color: #555; }
    </style>
    """
    
    total_failures = sum(len(f) for f in results['all_failures'].values())
    total_checks = results['total_checks']
    
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Data Alignment Validation Report</title>",
        css,
        "</head><body>",
        "<h1>Data Alignment Validation Report</h1>",
        "<div class='summary'>",
        "<h2>Summary</h2>",
        f"<div class='summary-item'>Total Checks: {total_checks}</div>",
        f"<div class='summary-item'>Total Failures: <strong>{total_failures}</strong></div>",
        f"<div class='summary-item'>Status: {'<span class=\"pass\">✓ PASS</span>' if total_failures == 0 else '<span class=\"fail\">✗ FAIL</span>'}</div>",
        "</div>",
    ]
    
    # Group failures by check type
    failures_by_check = {}
    for col_name in results['all_failures'].keys():
        for failure in results['all_failures'][col_name]:
            check_type = failure['check']
            if check_type not in failures_by_check:
                failures_by_check[check_type] = []
            failures_by_check[check_type].append(failure)
    
    # Define all check types
    check_type_labels = {
        'dd_columns_missing_in_datafile': 'Data Dictionary Columns Missing in Datafile',
        'datafile_columns_not_in_dd': 'Datafile Columns Not in Data Dictionary',
        'data_type': 'Matching Datatypes',
        'enumerations': 'Enumeration Constraints',
        'min_value': 'Minimum Values',
        'max_value': 'Maximum Values',
    }
    
    # Create a table for each check type, even if no failures
    for check_type in sorted(check_type_labels.keys()):
        failures = failures_by_check.get(check_type, [])
        check_label = check_type_labels[check_type]
        status = '<span class="pass">✓ PASS</span>' if not failures else '<span class="fail">✗ FAIL</span>'
        html_parts.append(f"<h2>{check_label} ({len(failures)}) {status}</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Column</th><th>Expected</th><th>Actual</th><th>Detail</th></tr>")
        
        if failures:
            for failure in sorted(failures, key=lambda x: x['column']):
                html_parts.append(
                    f"<tr>"
                    f"<td>{html.escape(failure['column'])}</td>"
                    f"<td>{html.escape(failure['expected'])}</td>"
                    f"<td>{html.escape(failure['actual'])}</td>"
                    f"<td class='detail'>{html.escape(failure['detail'])}</td>"
                    f"</tr>"
                )
        else:
            html_parts.append(f"<tr><td colspan='4' style='text-align: center; color: #27ae60;'>No issues found</td></tr>")
        
        html_parts.append("</table>")
    
    html_parts.append("</body></html>")
    full_html = "\n".join(html_parts)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)


def main():
    parser = argparse.ArgumentParser(
        description="Validate alignment between a data dictionary and datafile."
    )
    parser.add_argument(
        "-dd",
        "--data-dictionary",
        required=True,
        help="Path to data dictionary CSV"
    )
    parser.add_argument(
        "-df",
        "--datafile",
        required=True,
        help="Path to datafile CSV"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="Output HTML report path",
        default="alignment_validation_report.html"
    )
    args = parser.parse_args()
    
    # Read files
    dd_df = pd.read_csv(args.data_dictionary)
    data_df = pd.read_csv(args.datafile)
    data_df = convert_df_dtypes(data_df)
    
    print(f"Validating alignment between '{args.data_dictionary}' and '{args.datafile}'...")
    
    # Run validations
    all_failures = {}
    total_checks = 0
    
    # Check for DD columns missing in datafile
    dd_missing_failures = validate_dd_columns_not_in_datafile(dd_df, data_df)
    total_checks += 1
    for failure in dd_missing_failures:
        col_name = failure['column']
        if col_name not in all_failures:
            all_failures[col_name] = []
        all_failures[col_name].append(failure)
    
    # Check for datafile columns not in DD
    df_extra_failures = validate_datafile_columns_not_in_dd(dd_df, data_df)
    total_checks += 1
    for failure in df_extra_failures:
        col_name = failure['column']
        if col_name not in all_failures:
            all_failures[col_name] = []
        all_failures[col_name].append(failure)
    
    for idx, dd_row in dd_df.iterrows():
        col_name = dd_row['variable_name']
        col_failures = []
        
        # Skip columns that don't exist in datafile (already caught by validate_dd_columns_not_in_datafile)
        exists, msg = validate_column_exists(dd_row, data_df)
        if not exists:
            continue
        
        # Check 1: Data type
        total_checks += 1
        dtype_failures, _ = validate_dtype(dd_row, data_df)
        col_failures.extend(dtype_failures)
        
        # Check 2: Enumerations
        total_checks += 1
        enum_failures, _ = validate_enumerations(dd_row, data_df)
        col_failures.extend(enum_failures)
        
        # Check 3: Range (min/max)
        total_checks += 1
        range_failures, _ = validate_range(dd_row, data_df)
        col_failures.extend(range_failures)
        
        if col_failures:
            all_failures[col_name] = col_failures
    
    # Generate report
    results = {
        'all_failures': all_failures,
        'total_checks': total_checks
    }
    
    generate_validation_html(results, args.output)
    
    total_failures = sum(len(f) for f in all_failures.values())
    print(f"Validation complete. {total_failures} failure(s) found.")
    print(f"Wrote validation report to: {args.output}")


if __name__ == "__main__":
    main()