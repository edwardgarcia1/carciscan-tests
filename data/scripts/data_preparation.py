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
    """
    Calculates all available RDKit descriptors for a given SMILES string.
    Returns a tuple: (success, descriptors_dict).
    - success: True if calculation was successful, False otherwise.
    - descriptors_dict: Dictionary of descriptors if successful, or a dict of Nones if failed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # SMILES string is invalid, return failure with Nones
            return (False, {name: None for name, _ in Descriptors._descList})

        # SMILES is valid, calculate descriptors and return success
        descriptors = {name: fn(mol) for name, fn in Descriptors._descList}
        return (True, descriptors)
    except Exception as e:
        # Any other error during calculation
        print(f"Error calculating descriptors for SMILES '{smiles}': {e}", file=sys.stderr)
        return (False, {name: None for name, _ in Descriptors._descList})


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

    # --- MODIFICATION START ---
    # Apply the function which now returns a (success, descriptors) tuple
    result_series = smiles_df['SMILES'].progress_apply(calculate_rdkit_descriptors)

    # Unpack the tuple into two separate columns
    smiles_df['desc_calc_success'] = result_series.apply(lambda x: x[0])
    smiles_df['descriptors_dict'] = result_series.apply(lambda x: x[1])

    # Create the descriptor DataFrame from the dictionary column
    descriptor_df = pd.DataFrame(list(smiles_df['descriptors_dict']))

    # Concatenate with the original dataframe and drop the temporary dictionary column
    smiles_with_descriptors_df = pd.concat([smiles_df.drop(columns=['descriptors_dict']), descriptor_df], axis=1)
    # --- MODIFICATION END ---

    print("Step 4: Merging with T3DB data...")
    merged_df = pd.merge(
        smiles_with_descriptors_df,
        t3db_df[['CID', 'Categories', 'Route of Exposure']],
        on='CID',
        how='left'
    )

    print("Step 5: Normalizing 'Carcinogenicity', 'Categories', and 'Route of Exposure' columns...")
    merged_df['Categories'] = merged_df['Categories'].apply(
        lambda x: ['No data'] if pd.isna(x) or x == '' else x
    )
    merged_df['Carcinogenicity'] = merged_df['Carcinogenicity'].astype(str).apply(normalize_carcinogenicity)
    merged_df['Route'] = merged_df['Route of Exposure'].apply(normalize_route)
    merged_df.drop(columns=['Route of Exposure', 'SMILES'], inplace=True)

    print("Step 6: Filtering out rows with failed calculations or no key data...")
    initial_shape = merged_df.shape[0]

    # --- MODIFICATION START ---
    # Condition 1: Keep only rows where descriptor calculation was successful
    condition_desc_success = merged_df['desc_calc_success'] == True

    # Condition 2: Keep only rows that are not all 'No data' in toxicological fields
    is_no_data_carcinogenicity = merged_df['Carcinogenicity'] == 'No data'
    is_no_data_categories = merged_df['Categories'].apply(lambda x: x == ['No data'])
    is_no_data_route = merged_df['Route'].apply(lambda x: x == ['No data'])
    condition_tox_data = ~(is_no_data_carcinogenicity & is_no_data_categories & is_no_data_route)

    # Combine both conditions
    final_condition = condition_desc_success & condition_tox_data

    # Apply the filter
    final_df = merged_df[final_condition]

    # Report on filtering
    rows_dropped = initial_shape - final_df.shape[0]
    desc_failures = initial_shape - condition_desc_success.sum()
    tox_data_drops = condition_desc_success.sum() - final_df.shape[0]
    print(f"  - Dropped {desc_failures} rows due to descriptor calculation failures.")
    print(f"  - Dropped {tox_data_drops} rows because all key toxicological fields were 'No data'.")
    print(f"Total rows dropped: {rows_dropped}. New shape: {final_df.shape}")
    # --- MODIFICATION END ---

    print("Step 7: Finalizing and saving the dataset...")
    descriptor_names = [name for name, _ in Descriptors._descList]

    # --- MODIFICATION START ---
    # Drop the temporary success flag before saving
    final_df.drop(columns=['desc_calc_success'], inplace=True)

    # Define the final column order
    final_columns = ['CID', 'Carcinogenicity', 'Categories', 'Route'] + descriptor_names
    final_df = final_df[final_columns]
    # --- MODIFICATION END ---

    # Save to CSV
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"  - Successfully saved to CSV: {OUTPUT_PATH}")

    print("\n" + "=" * 50)
    print(f"Successfully created the final dataset!")
    print(f"Final dataset shape: {final_df.shape}")
    print("=" * 50)


if __name__ == "__main__":
    main()