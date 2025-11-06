import re
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc

# Define regex patterns for various chemical identifiers and systematic names
# CAS: d{2,7}-d{2}-d{1} e.g., 14992-62-2, 870-77-9
CAS_PATTERN = re.compile(r'^\d{2,7}-\d{2}-\d$')

EMBEDDED_CAS_PATTERN = re.compile(r'^CAS-\d{2,7}-\d{2}-\d$|Caswell No.', re.IGNORECASE)

# UNII: UNII- followed by alphanumeric e.g., UNII-07OP6H4V4A
UNII_PATTERN = re.compile(r'^UNII-[A-Z0-9]+$')

# InChI Key: Q + 23 digits e.g., Q27140241 (though format might vary slightly)
INCHIKEY_PATTERN = re.compile(r'^Q\d{23}$')

# ChEBI: CHEBI: followed by numbers e.g., CHEBI:73024
CHEBI_PATTERN = re.compile(r'^CHEBI:\d+$')

# DrugBank: DB followed by numbers e.g., DB11567
DRUGBANK_PATTERN = re.compile(r'^DB\d+$')

# KEGG: C, D, G, R etc. + 5 digits e.g., C06427
KEGG_PATTERN = re.compile(r'^[CDGR]\d{5}$')

# HMDB: HMDB followed by numbers e.g., HMDB0000094
HMDB_PATTERN = re.compile(r'^HMDB\d+$')

# LipidMaps: LM + various formats e.g., LMFA07070060
LIPIDMAPS_PATTERN = re.compile(r'^LM[A-Z0-9]+$')

# PDB: 4 characters (letters/numbers) e.g., 1C9U
PDB_PATTERN = re.compile(r'^[A-Z0-9]{4}$')

# ChemSpider: CS- + numbers e.g., CS-0102945
CHEMSPIDER_PATTERN = re.compile(r'^CS-\d+$')

# Beilstein: BRN + numbers e.g., BRN 0605275 (allowing optional space)
BEILSTEIN_PATTERN = re.compile(r'^BRN\s?\d+$')

EMBEDDED_BEILSTEIN_REF_PATTERN = re.compile(r'^\d{1,2}-\d{2}-\d{2}-\d{5}')

# Reaxys: RX + numbers e.g., RX81621
REAXYS_PATTERN = re.compile(r'^RX\d+$')

# ZINC: ZINC + numbers e.g., ZINC000000005157
ZINC_PATTERN = re.compile(r'^ZINC\d+$')

# PubChem SID/SCID: SID/SCID + numbers e.g., SID105780537
PUBCHEM_SID_PATTERN = re.compile(r'^(SID|SCID)\d+$')

# DTXSID: DTXSID + numbers e.g., DTXSID2048117
DTXSID_PATTERN = re.compile(r'^DTXSID\d+$')

# SCHEMBL: SCHEMBL + numbers e.g., SCHEMBL69781
SCHEMBL_PATTERN = re.compile(r'^SCHEMBL\d+$')

# NSC: NSC or NSC- + numbers e.g., NSC-3188, NSC00074255
NSC_PATTERN = re.compile(r'^NSC-?\d+$')

# AKOS: AKOS + numbers e.g., AKOS024284761
AKOS_PATTERN = re.compile(r'^AKOS\d+$')

# MFCD: MFCD + numbers e.g., MFCD00000005
MFCD_PATTERN = re.compile(r'^MFCD\d+$')

# EINECS: EINECS + numbers e.g., EINECS 201-162-7 (allowing optional space)
EINECS_PATTERN = re.compile(r'^EINECS\s?\d+-\d+-\d+$')

# Molecular Formula: e.g., C7H15NO3 (simplified pattern, can be more complex)
# Matches sequences starting with an uppercase letter, followed by numbers and other elements
FORMULA_PATTERN = re.compile(r'^[A-Z][a-z]?\d*[A-Z][a-z]?\d*.*$')  # Basic check, needs refinement for accuracy

# DTXCID: DTXCID + numbers e.g., DTXCID5028089, DTXCID60225732
DTXCID_PATTERN = re.compile(r'^DTXCID\d+$')

# CHEMBL: CHEMBL + numbers e.g., CHEMBL1625375
CHEMBL_PATTERN = re.compile(r'^CHEMBL\d+$')

# Generic database prefixes with numbers (bmse, orb, MSK, DAA, EIA, A, BCP, Tox21, BBL, SB, NCGC, DB)
GENERIC_DB_PREFIX_PATTERN = re.compile(r'^(?:bmse\d{6}|orb|MSK|DAA|EIA|NS\d{7,}|A\d{4,}|BCP\d+|Tox21_\d+|BBL\d+|SB\d+|NCGC\d+|DB-\d+|[A-Z]{2,3}\d{3,})$', re.IGNORECASE)

# EC Number / EC Registry Number: EC followed by numbers separated by hyphens e.g., EC 201-162-7, 201-162-7
EC_PATTERN = re.compile(r'^(EC\s?)?\d{3}-\d{3}-\d$', re.IGNORECASE)

# FEMA Number: FEMA NO. followed by numbers e.g., FEMA NO. 3965
FEMA_PATTERN = re.compile(r'^FEMA NO\. \d+$')

# Beilstein Handbook Reference: d-d-d-d (e.g., 4-04-00-01665)
BEILSTEIN_REF_PATTERN = re.compile(r'^\d{1,2}-\d{2}-\d{2}-\d{5}$')

# WLN (Wolff Linear Notation): Often starts with letters followed by numbers e.g., WLN: Z1YQ1
WLN_PATTERN = re.compile(r'^WLN: [A-Z0-9]+$')

# Generic UNII-like code (4-letter block, hyphen, 4-character block, hyphen, 4-character block, e.g., 07OP6H4V4A)
GENERIC_UNII_LIKE_PATTERN = re.compile(r'^(?=.*\d.*\d)[A-Z0-9]{8,10}$')

IUPAC_SYST_PATTERN = re.compile(
    r'(?:\(\+/-\)|\(\+\)-|\(-\)-|\([RS]\)-|\([2S]\)-|\([2RS]\)-|rac-|racemic|DL-|\.alpha\.|\.beta\.|\.gamma\.|alpha-|beta-|gamma-)'
    r'|.*[Ss]odium.*|.*[Pp]otassium.*|.*[Cc]alcium.*|.*[Aa]cid,.*|.*[Ii]um$|.*[Ii]nium$',
    re.IGNORECASE
)

# General database prefixes
DB_PREFIX_PATTERN = re.compile(r'^(RefChem:|NSC|AKOS|MFCD|ALBB|ST|BMSE|HY|AS|CCRIS|HSDB|AI3-|UE40BY1BZW)', re.IGNORECASE)

GENERIC_CODE_LIKE_PATTERN = re.compile(r'^(?=.*\d.*\d)[A-Z0-9]{6,10}$')


# Combined identifier check function - adapted to work on individual string values
def is_identifier_or_systematic(synonym_str):
    """Checks if a synonym string is likely an identifier or systematic/IUPAC name."""
    if synonym_str is None:
        return True # Treat nulls as identifiers to filter them out

    synonym = synonym_str # No need for .strip() here as PyArrow handles it efficiently
    syn_lower = synonym.lower()
    upper_count = sum(1 for c in synonym if c.isupper())

    num_digits = sum(1 for c in synonym if c.isdigit())
    total_chars = len(synonym)
    if total_chars > 0 and (num_digits / total_chars) > 0.4:
        return True

    if len(synonym) <= 7 and any(c.isdigit() for c in synonym):
        return True

    # Check for specific patterns
    if (CAS_PATTERN.match(synonym) or
            EMBEDDED_CAS_PATTERN.match(synonym) or
            UNII_PATTERN.match(synonym) or
            INCHIKEY_PATTERN.match(synonym) or
            CHEBI_PATTERN.match(synonym) or
            DRUGBANK_PATTERN.match(synonym) or
            KEGG_PATTERN.match(synonym) or
            HMDB_PATTERN.match(synonym) or
            LIPIDMAPS_PATTERN.match(synonym) or
            PDB_PATTERN.match(synonym) or
            CHEMSPIDER_PATTERN.match(synonym) or
            BEILSTEIN_PATTERN.match(synonym) or
            EMBEDDED_BEILSTEIN_REF_PATTERN.match(synonym) or
            REAXYS_PATTERN.match(synonym) or
            ZINC_PATTERN.match(synonym) or
            PUBCHEM_SID_PATTERN.match(synonym) or
            DTXSID_PATTERN.match(synonym) or
            SCHEMBL_PATTERN.match(synonym) or
            NSC_PATTERN.match(synonym) or
            AKOS_PATTERN.match(synonym) or
            MFCD_PATTERN.match(synonym) or
            EINECS_PATTERN.match(synonym) or
            DB_PREFIX_PATTERN.match(synonym) or
            DTXCID_PATTERN.match(synonym) or
            CHEMBL_PATTERN.match(synonym) or
            GENERIC_DB_PREFIX_PATTERN.match(synonym) or
            GENERIC_CODE_LIKE_PATTERN.match(synonym) or
            EC_PATTERN.match(synonym) or
            FEMA_PATTERN.match(synonym) or
            BEILSTEIN_REF_PATTERN.match(synonym) or
            WLN_PATTERN.match(synonym) or
            GENERIC_UNII_LIKE_PATTERN.match(synonym) or
            ('[' in synonym) or
            # Check for potential SMILES/InChI indicators (basic)
            ('InChI=' in synonym) or ('/' in synonym and '\\' in synonym) or ('@' in synonym) or (
                    'mol' in syn_lower) or ('smiles' in syn_lower) or
            # Check for potential molecular formula (very basic, can match common names too)
            (FORMULA_PATTERN.match(synonym) and not any(
                char.isdigit() for char in syn_lower)) or  # Avoid matching names with digits
            # Check for systematic name patterns (basic)
            IUPAC_SYST_PATTERN.match(synonym)):
        return True

    # Heuristic: Very long names with many capitals or specific suffixes might be systematic
    if len(synonym) > 40 or upper_count > 3:
        return True

    return False


# --- Main Script Execution (Filter All Rows - Optimized) ---

input_parquet_file_path = '../cid_synonym_filtered.parquet'
output_parquet_file_path = '../synonyms.parquet'

# Use PyArrow to open the Parquet file
parquet_file = pq.ParquetFile(input_parquet_file_path)

total_rows = parquet_file.metadata.num_rows
print(f"Total rows in input file: {total_rows}")

# Initialize counters
kept_count = 0
filtered_count = 0

# Prepare to write the output file using PyArrow
schema = pa.schema([
    ('CID', pa.int64()),
    ('Synonyms', pa.string())
])
all_filtered_tables = []

for i in range(parquet_file.num_row_groups):
    print(f"Processing row group {i+1}/{parquet_file.num_row_groups}...")

    row_group_table = parquet_file.read_row_group(i, use_pandas_metadata=False)
    synonyms_array = row_group_table.column('Synonyms')

    is_id_list = [is_identifier_or_systematic(syn_val.as_py()) for syn_val in synonyms_array]
    mask_array = pa.array(is_id_list, type=pa.bool_())
    keep_mask_array = pc.invert(mask_array)
    filtered_table = row_group_table.filter(keep_mask_array)

    kept_in_group = filtered_table.num_rows
    filtered_in_group = row_group_table.num_rows - kept_in_group
    kept_count += kept_in_group
    filtered_count += filtered_in_group

    if kept_in_group > 0:
        all_filtered_tables.append(filtered_table)

# Combine all filtered tables
if all_filtered_tables:
    final_table = pa.concat_tables(all_filtered_tables)
    # Ensure CID column type matches input if needed
    # final_table = final_table.cast(pa.schema([('CID', pa.int64()), ('Synonyms', pa.string())]))
    pq.write_table(final_table, output_parquet_file_path)
else:
    # Create an empty table with the correct schema if no rows pass the filter
    empty_table = pa.table({'CID': pa.array([], type=pa.int64()), 'Synonyms': pa.array([], type=pa.string())})
    pq.write_table(empty_table, output_parquet_file_path)

print(f"\n--- Filtering Summary ---")
print(f"Total rows processed: {total_rows}")
print(f"Rows kept (common names): {kept_count}")
print(f"Rows filtered out: {filtered_count}")
print(f"Output saved to: {output_parquet_file_path}")