import requests
import pandas as pd
import time

INPUT_FILE = "../pubchem_carcinogen.csv"
OUTPUT_FILE = "../carcinogen_smiles.csv"
FAILED_LOG_FILE = "failed_cids.csv"

def clean_dataset(df):
    property_columns = [
        "Compound_CID",
        "SMILES"
    ]

    # Create separate dataframes
    df_properties = df[property_columns].copy()

    return df_properties

def fetch_carcinogenicity(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?heading=Carcinogen+Classification"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception:
        return "Not Available", "Unknown"

    data = r.json()

    carcinogen_references = data.get(
        "Record", {}).get(
        "Reference", [])

    carcinogen_classifications = data.get(
        "Record", {}).get(
        "Section", {})[0].get(
        "Section", {})[0].get(
        "Section", {})[0].get(
        "Information", {})

    classification_reference = "Unknown"
    classification_reference_no = None
    classification_found = "Not Available"

    usable_references = [
        "International Agency for Research on Cancer (IARC)",
        "Toxin and Toxin Target Database (T3DB)"
    ]

    for ref in carcinogen_references:
        ref_no = ref.get("ReferenceNumber", 0)
        ref_name = ref.get("SourceName", "Unknown")
        if ref_name == usable_references[0]: # IARC has higher priority:
            classification_reference = ref_name
            classification_reference_no = ref_no
            break
        elif ref_name == usable_references[1]: # T3DB is second priority:
            classification_reference = ref_name
            classification_reference_no = ref_no

    for classifications in carcinogen_classifications:
        if classification_reference == usable_references[0]:
            if classifications.get("Name") == "IARC Carcinogenic Classes":
                classification_found = classifications.get("Value", {}).get("StringWithMarkup", [])[0].get("String", "")
                break
        elif classification_reference == usable_references[1]:
            if classifications.get("ReferenceNumber", 0) == classification_reference_no:
                classification_found = classifications.get("Value", {}).get("StringWithMarkup", [])[0].get("String", "")
                break

    return classification_found, classification_reference

def main():
    dataset = pd.read_csv(INPUT_FILE)
    df = clean_dataset(dataset)

    if "Compound_CID" not in df.columns:
        raise ValueError("Dataset must contain a 'Compound_CID' column")

    carcinogenicity = []
    sources = []
    skipped_cids = []

    # --- First pass ---
    for i, cid in enumerate(df["Compound_CID"]):
        try:
            cls, src = fetch_carcinogenicity(cid)
            if cls == "Not Available" or src == "Unknown":
                skipped_cids.append(cid)
            carcinogenicity.append(cls)
            sources.append(src)
            print(f"[{i+1}/{len(df)}] CID {cid}: {cls} ({src})")
            time.sleep(0.25)
        except Exception as e:
            print(f"Error with CID {cid}: {e}")
            skipped_cids.append(cid)
            carcinogenicity.append("Error")
            sources.append("Error")

    # --- Retry skipped CIDs ---
    if skipped_cids:
        print(f"\nRetrying {len(skipped_cids)} skipped CIDs...")
        for cid in skipped_cids[:]:  # iterate copy to allow removal
            try:
                cls, src = fetch_carcinogenicity(cid)
                idx = df.index[df["Compound_CID"] == cid][0]
                if cls != "Not Available" and src != "Unknown":
                    df.at[idx, "Carcinogenicity"] = cls
                    df.at[idx, "Carcinogenicity_Source"] = src
                    skipped_cids.remove(cid)
                print(f"Retry CID {cid}: {cls} ({src})")
                time.sleep(0.2)
            except Exception as e:
                print(f"Retry Error CID {cid}: {e}")

    # Save enriched dataset
    df["Carcinogenicity"] = carcinogenicity
    df["Carcinogenicity_Source"] = sources
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved enriched dataset → {OUTPUT_FILE}")

    # Save failed CIDs
    if skipped_cids:
        pd.DataFrame({"Failed_CID": skipped_cids}).to_csv(FAILED_LOG_FILE, index=False)
        print(f"Saved failed CIDs → {FAILED_LOG_FILE}")
    else:
        print("All CIDs fetched successfully.")

if __name__ == "__main__":
    main()