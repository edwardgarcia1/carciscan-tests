import duckdb
from pathlib import Path


def csv_to_parquet(
        input_path: str,
        output_path: str = None,
        delim: str = '\t',
        header: bool = False,
        columns: dict = None,
        compression: str = 'SNAPPY',
        quote: str = ''
):
    """
    Converts a CSV or TXT file to Parquet using DuckDB.

    Parameters:
        input_path (str): Path to the input CSV/TXT file.
        output_path (str): Path to the output Parquet file (optional, defaults to same name with .parquet).
        delim (str): Field delimiter (default: tab).
        header (bool): Whether the file includes a header row (default: False).
        columns (dict): Optional schema mapping (e.g. {'CID': 'BIGINT', 'SMILES': 'TEXT'}).
        compression (str): Parquet compression codec (e.g., 'SNAPPY', 'ZSTD', 'GZIP').
        quote (str): Quote character (set to '' to disable).
    """

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('.parquet')

    # Build DuckDB SQL command
    sql = f"""
    COPY (
        SELECT * FROM read_csv_auto(
            '{input_path}',
            delim='{delim}',
            header={'TRUE' if header else 'FALSE'},
            quote='{quote}'
            {f", columns={columns}" if columns else ""}
        )
    ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION {compression});
    """

    # Execute conversion
    print(f"Converting: {input_path} → {output_path}")
    duckdb.sql(sql)
    print("Conversion complete.")

    # Display schema for verification
    print("\nSchema:")
    duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{output_path}');").show()
    print()

### Example usage ###
# csv_to_parquet(
#     input_path='../cid_smiles.txt',
#     columns={'CID': 'BIGINT', 'SMILES': 'TEXT'},
# )
#
# csv_to_parquet(
#     input_path='../cid_synonym_filtered.txt',
#     columns={'CID': 'BIGINT', 'Synonyms': 'TEXT'},
# )

csv_to_parquet(
    input_path='../t3db_cleaned.csv',
    output_path='../../db/t3db.parquet',
    header=True,
    delim=',',
    quote='"'
)