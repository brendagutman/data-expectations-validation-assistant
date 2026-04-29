# data-expectations-validation-assistant

**DEVA (Data Expectations Validation Assistant)** — a command-line toolkit for profiling, validating, and documenting data files and data dictionaries.

---

## Requirements

- Python >= 3.10

Dependencies (installed automatically):
- [`pandas`](https://pandas.pydata.org/) — data loading and manipulation
- [`numpy`](https://numpy.org/) — numeric type inference

---

## Installation

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install from GitHub:

```bash
pip install git+https://github.com/brendagutman/data-expectations-validation-assistant.git
```

Or install locally from source:

```bash
pip install .
```

---

## Commands

### `characterize_datafile` — Profile a Data File

Reads a data file and produces an HTML report with an overview, per-column summary (data type, null rate, unique count, sample values, and enumerations), and numeric statistics. Optionally cross-references a data dictionary to include expected data types and DD metadata alongside observed values.

```bash
characterize_datafile -df <data_file> [-dd <data_dictionary>] [-o <output>] [options]
```

#### Arguments

| Argument | Required | Description |
|---|---|---|
| `-df` / `--data_file` | Yes | Path to the data file (CSV). |
| `-dd` / `--data_dictionary` | No | Path to a data dictionary CSV. When provided, expected data types and all DD fields are shown alongside observed values in the report. |
| `-o` / `--output` | No | Output HTML path. Defaults to `<stem>_characterization<YYYYMMDD>.html`. |
| `--show-enums` | No | Comma-separated column names for which all unique values should be shown. |
| `--hide-enums` | No | Comma-separated column names for which enumerations should be suppressed. Cannot be used with `--show-enums`. |
| `--chunksize` | No | Read the file in streaming chunks of this many rows (default: `100000`). Useful for large files. |
| `--low-memory` | No | Pass `low_memory=True` to pandas, reducing memory at the cost of dtype inference accuracy. |

#### Examples

```bash
# Basic profiling
characterize_datafile -df data/mydata.csv -o data/mydata_profile.html

# With a data dictionary for cross-reference
characterize_datafile -df data/mydata.csv -dd data/mydata_dd.csv -o data/mydata_profile.html

# Suppress enumerations for high-cardinality columns
characterize_datafile -df data/mydata.csv --hide-enums patient_id,notes
```

---

### `generate_datadictionary` — Generate a Data Dictionary

Inspects a data file and generates a data dictionary CSV with inferred data types, min/max values for numeric columns, and enumerations for low-cardinality columns (≤50 unique values by default).

```bash
generate_datadictionary -df <data_file> [-o <output>] [options]
```

#### Arguments

| Argument | Required | Description |
|---|---|---|
| `-df` / `--data_file` | Yes | Path to the data file (CSV). |
| `-o` / `--output` | No | Output CSV path. Defaults to `<stem>_dd<YYYYMMDD>.csv`. |
| `--show-enums` | No | Comma-separated column names for which enumerations should be populated regardless of cardinality. |
| `--hide-enums` | No | Comma-separated column names for which enumerations should be suppressed. Cannot be used with `--show-enums`. |
| `--chunksize` | No | Read the file in streaming chunks of this many rows. |
| `--low-memory` | No | Pass `low_memory=True` to pandas. |

#### Output Columns

| Column | Description |
|---|---|
| `variable_name` | Column name from the data file. |
| `description` | Empty — intended for manual entry. |
| `data_type` | Inferred data type (`string`, `integer`, `float`, `boolean`). |
| `min` | Minimum value for numeric columns. |
| `max` | Maximum value for numeric columns. |
| `units` | Empty — intended for manual entry. |
| `enumerations` | Semicolon-separated unique values for low-cardinality columns. |
| `comment` | Empty — intended for manual entry. |

#### Examples

```bash
generate_datadictionary -df data/mydata.csv -o data/mydata_dd.csv

# Show enumerations for specific columns only
generate_datadictionary -df data/mydata.csv --show-enums status,visit_type
```

---

### `merge_datadictionary` — Merge a Generated DD with an Existing One

Generates a data dictionary from a data file and merges it with an existing data dictionary. The existing DD is treated as the primary source. Where the generated DD disagrees with an existing field, a `<field>_gen` column is added on the same row. Fields that appear in the generated DD but are absent from the existing DD are added as regular columns. Variables in the existing DD that have no matching column in the data file are flagged in a `variable_name_gen` column, and variables in the data file that are missing from the existing DD are appended at the end.

```bash
merge_datadictionary -df <data_file> [-dd <data_dictionary>] [-o <output>] [options]
```

#### Arguments

| Argument | Required | Description |
|---|---|---|
| `-df` / `--data_file` | Yes | Path to the data file (CSV). |
| `-dd` / `--data_dictionary` | No | Path to an existing data dictionary CSV. If omitted, behaves like `generate_datadictionary`. |
| `-o` / `--output` | No | Output CSV path. Defaults to `<stem>_merged_dd<YYYYMMDD>.csv`. |
| `--show-enums` | No | Comma-separated column names for which enumerations should be populated. |
| `--hide-enums` | No | Comma-separated column names for which enumerations should be suppressed. Cannot be used with `--show-enums`. |
| `--token` | No | Synapse personal access token (or set `SYN_AUTH_TOKEN` env var). |

#### Column Name Normalization

Existing data dictionaries often use non-standard column names. DEVA normalizes column names by lowercasing and replacing spaces/hyphens with underscores, then maps common aliases to canonical names:

| Canonical Name | Accepted Aliases |
|---|---|
| `variable_name` | `variable`, `field_name`, `field`, `column_name`, `column` |
| `description` | `label`, `desc` |
| `data_type` | `datatype`, `type`, `dtype` |
| `min` | `minimum` |
| `max` | `maximum` |
| `units` | `unit` |
| `enumerations` | `enums`, `enum`, `enumeration`, `permitted_values`, `values` |
| `comment` | `comments`, `notes`, `note` |

If a canonical name already exists as a column, any alias for it is left with its original name to avoid duplicates.

#### Output Behavior

| Scenario | Output |
|---|---|
| Field exists in both and values agree | Original column only, no `_gen` column |
| Field exists in both and values **disagree** | Original column kept; `<field>_gen` column added with generated value |
| Field missing from existing DD | Added as a regular column populated from generated DD |
| Variable in existing DD **not found in data file** | `variable_name_gen` = `NOT FOUND IN DATA` |
| Variable in data file **not in existing DD** | Row appended at end; `variable_name_gen` = `NOT IN DICTIONARY` |

`_gen` columns only appear in the output if at least one row has a disagreement for that field.

#### Examples

```bash
# Generate a DD from scratch
merge_datadictionary -df data/mydata.csv -o data/mydata_dd.csv

# Merge with an existing DD
merge_datadictionary -df data/mydata.csv -dd data/existing_dd.csv -o data/merged_dd.csv
```

---

### `validate_alignment` — Validate Alignment Between a DD and Data File

Validates that a data file conforms to a data dictionary. Checks column presence, data types, allowed values (enumerations), and numeric ranges. Results are written to an HTML report.

```bash
validate_alignment -dd <data_dictionary> -df <datafile> [-o <output>]
```

#### Arguments

| Argument | Required | Description |
|---|---|---|
| `-dd` / `--data-dictionary` | Yes | Path to the data dictionary CSV. |
| `-df` / `--datafile` | Yes | Path to the data file CSV. |
| `-o` / `--output` | No | Output HTML report path. Defaults to `<stem>_validation<YYYYMMDD>.html`. |

#### Validation Checks

| Check | Description |
|---|---|
| Column presence | Columns defined in the DD but absent from the data file are flagged. Case-insensitive matching with a warning when case differs. |
| Extra columns | Columns in the data file not defined in the DD are flagged. |
| Data type | Values that cannot be cast to the declared `data_type` are reported. |
| Enumerations | Values not in the `enumerations` list defined in the DD are reported. |
| Range | Numeric values outside `min`/`max` constraints are reported. |
| Nullability | Fields marked `required: yes` that contain null or empty values are reported. |

#### Examples

```bash
validate_alignment -dd data/mydata_dd.csv -df data/mydata.csv -o data/alignment_report.html
```
