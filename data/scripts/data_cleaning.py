import pandas as pd
from pathlib import Path

def clean_csv(
        input_path: str,
        output_path: str = None,
        drop_columns: list = None,
        keep_columns: list = None,
        drop_duplicates: bool = True,
        fill_missing: bool = True,
        lowercase_columns: bool = True,
        trim_strings: bool = True,
        list_columns: list = None,
        list_separators: dict = None,
        newline_to_space_columns: list = None,
        save_format: str = "csv"  # options: "csv" or "parquet"
):
    """
    Cleans a CSV file by removing/keeping selected columns, performing preprocessing,
    converting specified columns into lists, and replacing newlines with spaces.

    Parameters:
        input_path (str): Path to the input CSV file.
        output_path (str, optional): Path to save cleaned file. Defaults to '<original>_cleaned.csv'.
        drop_columns (list, optional): Columns to remove.
        keep_columns (list, optional): Columns to keep; all others are removed if specified.
        drop_duplicates (bool): Remove duplicate rows.
        fill_missing (bool): Fill missing values (mean for numeric, mode for text).
        lowercase_columns (bool): Lowercase column names.
        trim_strings (bool): Strip whitespace in string columns.
        list_columns (list, optional): Columns to convert into Python lists.
        list_separators (dict, optional): Dict mapping column names to separator (`','` or `';'`). Defaults to ','.
        newline_to_space_columns (list, optional): Columns in which to replace newlines with spaces.
        save_format (str): "csv" or "parquet".

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Load CSV
    df = pd.read_csv(input_path)
    print(f"Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")

    # Normalize column names
    if lowercase_columns:
        df.columns = df.columns.str.strip().str.lower()

    # Determine columns to keep or drop
    if keep_columns:
        keep_columns = [c.lower() if lowercase_columns else c for c in keep_columns]
        missing_keep = [c for c in keep_columns if c not in df.columns]
        if missing_keep:
            print(f"Warning: These keep_columns were not found and skipped: {missing_keep}")
        df = df[[c for c in keep_columns if c in df.columns]]
        print(f"Kept {len(df.columns)} columns (based on keep_columns list)")
    elif drop_columns:
        drop_columns = [c.lower() if lowercase_columns else c for c in drop_columns]
        missing_drop = [c for c in drop_columns if c not in df.columns]
        if missing_drop:
            print(f"Warning: These drop_columns were not found and skipped: {missing_drop}")
        df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")
        print(f"Dropped specified columns; remaining {len(df.columns)} columns")

    # Remove duplicates
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        print(f"Removed {before - len(df)} duplicate rows")

    # Trim whitespace in string columns
    if trim_strings:
        string_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
        if len(string_cols) > 0:
            print(f"Trimmed whitespace from {len(string_cols)} string columns")

    # Fill missing values
    if fill_missing:
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == "O":
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].mean())
        print("Filled missing values")

    # Convert specified columns to lists
    if list_columns:
        list_columns = [c.lower() if lowercase_columns else c for c in list_columns]
        for col in list_columns:
            if col in df.columns:
                sep = list_separators.get(col, ",") if list_separators else ","
                df[col] = df[col].apply(lambda x: [s.strip().strip('"') for s in str(x).split(sep)])
        print(f"Converted {len(list_columns)} columns into lists")

    # Replace newlines with spaces in specified columns (before any other cleaning)
    if newline_to_space_columns:
        newline_to_space_columns = [c.lower() if lowercase_columns else c for c in newline_to_space_columns]
        for col in newline_to_space_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[\r\n]+', ' ', regex=True)
        print(f"Replaced newlines with spaces in {len(newline_to_space_columns)} columns")

    # Output path
    if output_path is None:
        suffix = "_cleaned.parquet" if save_format == "parquet" else "_cleaned.csv"
        output_path = Path(input_path).with_name(Path(input_path).stem + suffix)

    # Save cleaned data
    if save_format == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"Cleaned file saved to: {output_path}")
    return df



list_cols = ["Categories", "Route of Exposure", "Symptoms", "Treatment", "Minimum Risk Level", "Health Effects"]
list_seps = {
    "Categories": ",",
    "Route of Exposure": ";",
    "Symptoms": ".",
    "Treatment": ".",
    "Minimum Risk Level": "\n",
    "Health Effects": "."
}
newline_cols = ["Mechanism of Toxicity", "Metabolism", "Health effects", "Lethal Dose", "Uses/Sources"]
clean_csv(
    input_path="../t3db.csv",
    keep_columns=["Categories", "Route of Exposure", "Mechanism of Toxicity", "Metabolism", "Lethal Dose",
                  "Carcinogenicity", "Uses/Sources", "Minimum Risk Level", "Health Effects", "Symptoms",
                  "Treatment", "PubChem Compound ID"],
    drop_duplicates=True,
    trim_strings=True,
    fill_missing=False,
    list_columns=list_cols,
    list_separators=list_seps,
    newline_to_space_columns=newline_cols,
    lowercase_columns=False,
    save_format="csv"
)