import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import sys
from normalization import normalize_carcinogenicity, normalize_route

SMILES_PATH = '../carcinogen_smiles.csv'
T3DB_PATH = '../t3db_cleaned.csv'
OUTPUT_PATH = '../dataset.csv'


def calculate_rdkit_descriptors(smiles):
    """Calculates all available RDKit descriptors for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {name: None for name, _ in Descriptors._descList}
        descriptors = {name: fn(mol) for name, fn in Descriptors._descList}
        return descriptors
    except Exception as e:
        print(f"Error calculating descriptors for SMILES '{smiles}': {e}", file=sys.stderr)
        return {name: None for name, _ in Descriptors._descList}


def main():
    """Main function to load data, calculate descriptors, merge, normalize, and save."""
    print("Step 1: Loading datasets...")
    try:
        smiles_df = pd.read_csv(SMILES_PATH)
        t3db_df = pd.read_csv(T3DB_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.")
        sys.exit(1)

    print("Step 2: Preparing data for merge...")
    t3db_df['CID'] = pd.to_numeric(t3db_df['CID'], errors='coerce').astype('Int64')
    smiles_df.rename(columns={'Compound_CID': 'CID'}, inplace=True)

    print("Step 3: Calculating RDKit molecular descriptors...")
    tqdm.pandas(desc="Calculating Descriptors")
    descriptor_list = smiles_df['SMILES'].progress_apply(calculate_rdkit_descriptors)
    descriptor_df = pd.DataFrame(list(descriptor_list))
    smiles_with_descriptors_df = pd.concat([smiles_df, descriptor_df], axis=1)

    print("Step 4: Merging with T3DB data...")
    merged_df = pd.merge(
        smiles_with_descriptors_df,
        t3db_df[['CID', 'Categories', 'Route of Exposure']],
        on='CID',
        how='left'
    )

    print("Step 5: Normalizing 'Carcinogenicity', 'Categories', and 'Route of Exposure' columns...")
    # Normalize empty strings and NaNs in the 'Categories' column for consistency
    merged_df['Categories'] = merged_df['Categories'].apply(
        lambda x: ['No data'] if pd.isna(x) or x == '' else x
    )

    # Apply the normalization functions
    merged_df['Carcinogenicity'] = merged_df['Carcinogenicity'].astype(str).apply(normalize_carcinogenicity)

    merged_df['Route'] = merged_df['Route of Exposure'].apply(normalize_route)

    # Drop the original 'Route of Exposure' and 'SMILES' columns
    merged_df.drop(columns=['Route of Exposure', 'SMILES'], inplace=True)

    print("Step 6: Dropping rows with no data in all three key columns...")
    initial_shape = merged_df.shape[0]

    # Create boolean masks for each condition
    is_no_data_carcinogenicity = merged_df['Carcinogenicity'] == 'No data'
    is_no_data_categories = merged_df['Categories'].apply(lambda x: x == ['No data'])
    is_no_data_route = merged_df['Route'].apply(lambda x: x == ['No data'])

    # Combine the conditions to find rows where all three are 'No data'
    condition = is_no_data_carcinogenicity & is_no_data_categories & is_no_data_route

    final_df = merged_df[~condition]
    final_shape = final_df.shape[0]
    print(f"Dropped {initial_shape - final_shape} rows. New shape: {final_df.shape}")

    print("Step 7: Finalizing and saving the dataset in CSV and Parquet formats...")
    descriptor_names = [name for name, _ in Descriptors._descList]
    final_columns = ['CID', 'Carcinogenicity', 'Categories', 'Route'] + descriptor_names
    final_df = final_df[final_columns]

    # Save to CSV
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"  - Successfully saved to CSV: {OUTPUT_PATH}")


    print("\n" + "=" * 50)
    print(f"Successfully created the final dataset!")
    print(f"Final dataset shape: {final_df.shape}")
    print("=" * 50)


if __name__ == "__main__":
    main()