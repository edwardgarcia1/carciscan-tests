import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
from normalization import normalize_carcinogenicity

warnings.filterwarnings('ignore')


def get_all_rdkit_descriptors(smiles):
    """
    Calculate all available RDKit molecular descriptors from SMILES.
    Returns a dictionary of all descriptors or None if molecule is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get all descriptor names and functions
        desc_list = Descriptors.descList

        # Calculate all descriptors
        descriptors = {}
        for desc_name, desc_func in desc_list:
            try:
                descriptors[desc_name] = desc_func(mol)
            except Exception as e:
                # Handle any descriptor calculation errors
                descriptors[desc_name] = None
                print(f"Warning: Could not calculate {desc_name} for SMILES {smiles}: {str(e)}")

        return descriptors

    except Exception as e:
        print(f"Error processing SMILES {smiles}: {str(e)}")
        return None




def main():
    INPUT_FILE = "../carcinogen_smiles.csv"
    OUTPUT_FILE = "../carcinogen_properties.csv"

    # Read the input file
    df = pd.read_csv(INPUT_FILE)

    # Check required columns exist
    required_cols = ['Compound_CID', 'SMILES', 'Carcinogenicity']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input file")

    print(f"Processing {len(df)} compounds...")
    print(f"Calculating {len(Descriptors.descList)} RDKit descriptors...")

    # Get all descriptor names for consistent column ordering
    all_descriptor_names = [desc_name for desc_name, _ in Descriptors.descList]

    # Initialize list to store results
    results_list = []

    valid_count = 0
    invalid_count = 0

    for idx, row in df.iterrows():
        cid = row['Compound_CID']
        smiles = row['SMILES']
        carcinogenicity = row['Carcinogenicity']

        descriptors = get_all_rdkit_descriptors(smiles)

        if descriptors is not None:
            normalized_carcinogenicity = normalize_carcinogenicity(carcinogenicity)

            if normalized_carcinogenicity != "other":
                descriptors['Compound_CID'] = cid
                descriptors['Carcinogenicity'] = normalized_carcinogenicity
                results_list.append(descriptors)
                valid_count += 1
            else:
                print(f"Skipping CID {cid}: Carcinogenicity classified as 'other'")
                invalid_count += 1
        else:
            print(f"Invalid SMILES for CID {cid}: {smiles}")
            invalid_count += 1

        # Progress indicator (optional)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} compounds...")

    print(f"\nSuccessfully processed: {valid_count} compounds")
    print(f"Failed to process: {invalid_count} compounds")

    if results_list:
        # Create DataFrame
        result_df = pd.DataFrame(results_list)

        # Reorder columns: CID, Carcinogenicity, then all descriptors in consistent order
        column_order = ['Compound_CID', 'Carcinogenicity'] + all_descriptor_names
        # Only include columns that actually exist in the DataFrame
        column_order = [col for col in column_order if col in result_df.columns]
        result_df = result_df[column_order]

        # Save to output file
        result_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nResults saved to {OUTPUT_FILE}")
        print(f"Final dataset shape: {result_df.shape}")
        print(f"Number of descriptor columns: {len(all_descriptor_names)}")

        # Show sample of the data
        print("\nFirst few rows (showing first 10 columns):")
        display_cols = ['Compound_CID', 'Carcinogenicity'] + all_descriptor_names[:8]
        print(result_df[display_cols].head())

    else:
        print("No valid compounds processed. Output file not created.")


if __name__ == "__main__":
    main()