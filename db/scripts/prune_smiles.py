import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd

def filter_smiles_by_synonyms_optimized():
    """
    Memory and speed optimized version to filter cid_smiles.parquet
    by keeping only rows where CID exists in synonyms.parquet
    """
    print("Loading synonyms.parquet to get valid CIDs...")
    # Load only the CID column from synonyms to save memory [[1]]
    synonyms_table = pq.read_table("../synonyms.parquet", columns=['CID'])
    valid_cids = set(synonyms_table['CID'].to_pylist())
    print(f"Found {len(valid_cids)} unique CIDs in synonyms dataset")

    print("Reading cid_smiles.parquet...")
    # Read the entire cid_smiles table
    cid_smiles_table = pq.read_table("../cid_smiles.parquet")

    print(f"Original dataset has {cid_smiles_table.num_rows} rows")

    # Convert the set of valid CIDs to a PyArrow array for efficient computation
    valid_cids_array = pa.array(list(valid_cids))

    # Use PyArrow compute to create a boolean mask efficiently
    mask = pc.is_in(cid_smiles_table['CID'], value_set=valid_cids_array)

    # Filter the table using the mask
    filtered_table = cid_smiles_table.filter(mask)

    print(f"After filtering, {filtered_table.num_rows} rows remain")

    # Write the filtered table to parquet [[5]]
    pq.write_table(filtered_table, "../smiles.parquet")

    print("Filtering completed successfully!")
    print(f"Filtered data saved to ../smiles.parquet")


def filter_smiles_by_synonyms_chunked():
    """
    Memory-efficient version that processes the file in chunks
    """
    print("Loading synonyms.parquet to get valid CIDs...")
    # Load only the CID column from synonyms to save memory [[1]]
    synonyms_table = pq.read_table("../synonyms.parquet", columns=['CID'])
    valid_cids = set(synonyms_table['CID'].to_pylist())
    print(f"Found {len(valid_cids)} unique CIDs in synonyms dataset")

    print("Processing cid_smiles.parquet in chunks...")

    # Open the Parquet file for reading in batches [[9]]
    parquet_file = pq.ParquetFile("../cid_smiles.parquet")

    # Prepare output file
    output_path = "../smiles.parquet"
    first_batch = True

    processed_rows = 0
    kept_rows = 0

    # Process the file in row groups or batches
    for batch in parquet_file.iter_batches(batch_size=100000):  # Process 100k rows at a time
        print(f"Processing batch, rows {processed_rows} to {processed_rows + batch.num_rows}...")

        # Convert batch to table
        batch_table = pa.Table.from_pydict(batch)

        # Create boolean mask using PyArrow compute
        mask = pc.is_in(batch_table['CID'], value_set=pa.array(list(valid_cids)))

        # Filter the current batch
        filtered_batch = batch_table.filter(mask)

        if filtered_batch.num_rows > 0:
            # Write to parquet file
            if first_batch:
                # Write with schema for the first batch
                pq.write_table(filtered_batch, output_path)
                first_batch = False
            else:
                # For subsequent batches, we need to append.
                # This requires reading the existing file, appending, and rewriting,
                # which can be inefficient. A better approach might be to collect batches
                # and write them all at once, or use a different library for appending.
                # For now, we'll concatenate with the existing file if it exists.
                try:
                    existing_table = pq.read_table(output_path)
                    combined_table = pa.concat_tables([existing_table, filtered_batch])
                    pq.write_table(combined_table, output_path)
                except:
                    # If file doesn't exist yet, just write the batch
                    pq.write_table(filtered_batch, output_path)

        processed_rows += batch.num_rows
        kept_rows += filtered_batch.num_rows

        print(f"  Kept {filtered_batch.num_rows} out of {batch.num_rows} rows in this batch")
        print(f"  Total processed: {processed_rows}, Total kept: {kept_rows}")

    print(f"Filtering completed! Processed {processed_rows} total rows, kept {kept_rows} rows.")
    print(f"Filtered data saved to {output_path}")


def filter_smiles_by_synonyms_pandas_chunks():
    """
    Memory-efficient version using pandas with chunking and the pyarrow engine
    """
    print("Loading synonyms.parquet to get valid CIDs...")
    synonyms_df = pq.read_table("../synonyms.parquet", columns=['CID']).to_pandas()  # [[10]]
    valid_cids = set(synonyms_df['CID'].unique())
    print(f"Found {len(valid_cids)} unique CIDs in synonyms dataset")

    print("Processing cid_smiles.parquet in chunks using pandas...")

    chunk_size = 100000
    first_chunk = True
    processed_rows = 0
    kept_rows = 0

    # Read the large parquet file in chunks using pandas
    for chunk in pd.read_parquet("../cid_smiles.parquet", chunksize=chunk_size):
        print(f"Processing chunk, rows {processed_rows} to {processed_rows + len(chunk)}...")

        # Filter the current chunk
        filtered_chunk = chunk[chunk['CID'].isin(valid_cids)]

        if not filtered_chunk.empty:
            # Write to parquet file
            if first_chunk:
                # Write with schema for the first chunk
                filtered_chunk.to_parquet("../smiles.parquet", index=False)
                first_chunk = False
            else:
                # Append to existing file using pandas
                filtered_chunk.to_parquet("../smiles.parquet", index=False, mode='a')

        processed_rows += len(chunk)
        kept_rows += len(filtered_chunk)

        print(f"  Kept {len(filtered_chunk)} out of {len(chunk)} rows in this chunk")
        print(f"  Total processed: {processed_rows}, Total kept: {kept_rows}")

    print(f"Filtering completed! Processed {processed_rows} total rows, kept {kept_rows} rows.")
    print(f"Filtered data saved to ../smiles.parquet")


if __name__ == "__main__":
    # For a 3.5GB file, start with the first version which loads everything into memory at once
    # but uses efficient PyArrow operations
    filter_smiles_by_synonyms_optimized()

    # If memory is an issue, use one of the chunked versions instead:
    # filter_smiles_by_synonyms_chunked() # Uses PyArrow
    # filter_smiles_by_synonyms_pandas_chunks() # Uses pandas