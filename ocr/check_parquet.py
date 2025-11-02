import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


def explore_parquet_file(parquet_path: str):
    """Explore the structure and content of a parquet file"""
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        print(f"File not found: {parquet_path}")
        return

    print(f"=== Parquet File Explorer ===")
    print(f"File: {parquet_path}")
    print(f"Size: {parquet_path.stat().st_size / (1024 ** 2):.2f} MB")

    # Read metadata
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        print(f"Total rows: {parquet_file.metadata.num_rows:,}")
        print(f"Total columns: {parquet_file.metadata.num_columns}")
        print(f"Number of row groups: {parquet_file.metadata.num_row_groups}")

        # Get column names
        columns = parquet_file.schema.names
        print(f"\n=== Columns ({len(columns)} total) ===")

        # Show first 20 columns
        for i, col in enumerate(columns[:20]):
            print(f"{i + 1:3d}. {col}")

        if len(columns) > 20:
            print(f"... and {len(columns) - 20} more columns")

        # Read first few rows to see data
        print(f"\n=== Sample Data (first 3 rows) ===")
        first_batch = next(parquet_file.iter_batches(batch_size=3))
        df_sample = first_batch.to_pandas()
        print(df_sample.to_string())

        # Check specific columns of interest
        print(f"\n=== Country Information ===")
        if 'countries_tags' in columns:
            # Get a larger sample to check countries
            sample_batch = next(parquet_file.iter_batches(batch_size=1000))
            sample_df = sample_batch.to_pandas()

            if 'countries_tags' in sample_df.columns:
                countries = sample_df['countries_tags'].dropna()
                print(f"Rows with country data: {len(countries)} out of 1000 sampled")

                if len(countries) > 0:
                    # Show unique country values
                    unique_countries = set()
                    for country_val in countries.head(50):  # Check first 50
                        if isinstance(country_val, str):
                            if ',' in country_val:
                                unique_countries.update(c.strip() for c in country_val.split(','))
                            else:
                                unique_countries.add(country_val.strip())
                        elif isinstance(country_val, list):
                            unique_countries.update(str(c).strip() for c in country_val)

                    print(f"Sample country values (first 10):")
                    for i, country in enumerate(list(unique_countries)[:10]):
                        print(f"  {i + 1}. {country}")
                else:
                    print("No country data in sample")
            else:
                print("'countries_tags' column exists but not in sample")
        else:
            print("No 'countries_tags' column found")
            # Show columns that might contain country info
            country_like_cols = [col for col in columns if 'country' in col.lower() or 'nation' in col.lower()]
            if country_like_cols:
                print(f"Possible country columns: {country_like_cols[:5]}")

        # Check for image/ingredient columns
        print(f"\n=== Image and Ingredient Columns ===")
        image_cols = [col for col in columns if 'image' in col.lower()]
        ingredient_cols = [col for col in columns if 'ingredient' in col.lower()]

        print(f"Image-related columns ({len(image_cols)}):")
        for col in image_cols[:10]:
            print(f"  - {col}")
        if len(image_cols) > 10:
            print(f"  ... and {len(image_cols) - 10} more")

        print(f"\nIngredient-related columns ({len(ingredient_cols)}):")
        for col in ingredient_cols[:10]:
            print(f"  - {col}")
        if len(ingredient_cols) > 10:
            print(f"  ... and {len(ingredient_cols) - 10} more")

        # Check for Philippines specifically in the sample
        print(f"\n=== Philippines Search ===")
        ph_found = False
        if 'countries_tags' in sample_df.columns:
            for idx, row in sample_df.head(1000).iterrows():
                if pd.notna(row['countries_tags']):
                    country_str = str(row['countries_tags']).lower()
                    if 'philippine' in country_str or 'ph' in country_str or 'pilipinas' in country_str:
                        print(f"Found potential Philippines entry at row {idx}: {row['countries_tags']}")
                        ph_found = True
                        break

        if not ph_found:
            print("No Philippines entries found in first 1000 rows with country data")

    except Exception as e:
        print(f"Error reading parquet file: {e}")
        # Try with pandas as fallback
        try:
            print("Trying with pandas (may use more memory)...")
            df = pd.read_parquet(parquet_path, engine='pyarrow')
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)[:10]}...")
            print("First few rows:")
            print(df.head(3).to_string())
        except Exception as e2:
            print(f"Pandas also failed: {e2}")


if __name__ == "__main__":
    explore_parquet_file("../data/food.parquet")