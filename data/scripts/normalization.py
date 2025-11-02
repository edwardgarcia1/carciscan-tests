import re
import pandas as pd

def normalize_carcinogenicity(text):
    """
    Normalizes text descriptions of carcinogenicity into a unified numerical code.

    This function uses a two-tiered logic:
    1.  It first checks for high-priority "overriding" patterns. If any are found,
        their code is returned immediately.
    2.  If no overriding patterns match, it finds the pattern that appears
        earliest in the text and returns its code.

    Codes:
    0 : Group 1 (Carcinogenic to humans)
    1 : Group 2A (Probably carcinogenic to humans)
    2 : Group 2B (Possibly carcinogenic to humans)
    3 : Group 3 (Not classifiable as to its carcinogenicity to humans)
    "other" : If no pattern matches.

    Args:
        text (str): The input text description of carcinogenicity.
    """
    if not text or not isinstance(text, str):
        return "other"

    text = text.strip().lower()

    # --- TIER 1: OVERRIDING PATTERNS ---
    # Add patterns here that should ALWAYS take precedence, regardless of position.
    # The script checks this list first. The first match in this list wins.
    overriding_rules = [

        (r"not listed by iarc", "other"),
        (r"not directly listed by iarc", "other"),

    ]
    # --- TIER 2: FIRST-MATCH PATTERNS ---
    # These patterns are only checked if no overriding patterns are found.
    # The one that appears earliest in the text will be used.
    first_match_rules = [
        # --- Specific Exclusion Patterns ---
        (r"are not classifiable", 3),
        (r"is not classifiable", 3),
        (r"no indication", 3),

        # --- IARC Group Patterns ---
        (r"not listed by iarc", 3),
        (r"\bgroup\s*3\b", 3),
        (r"^3[\s,:] ", 3),
        (r"\bgroup\s*2b\b", 2),
        (r"2b[\s,:]", 2),
        (r"\bgroup\s*2a\b", 1),
        (r"^2a[\s,:]", 1),
        (r"\bgroup\s*1\b", 0),
        (r"^1[\s,:]", 0),
    ]

    # --- LOGIC EXECUTION ---

    # Step 1: Check for overriding patterns first.
    for pattern, code in overriding_rules:
        if re.search(pattern, text, flags=re.IGNORECASE):
            # Found an overriding pattern, return its code immediately.
            return code

    # Step 2: If no overriding patterns were found, use the "first match" logic.
    first_match = None
    first_match_pos = len(text) + 1 # Initialize with a position beyond the text

    for pattern, code in first_match_rules:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.start() < first_match_pos:
            first_match = (match.start(), code)
            first_match_pos = match.start()

    # Step 3: Return the result from the first match logic, or "other".
    if first_match:
        return first_match[1]

    return "other"

def main():
    """
    Loads the dataset, applies the normalization function, and prints
    a sorted list of unique values and their counts for each group.
    """
    csv_path = '../carcinogen_smiles.csv'

    print(f"Loading dataset from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}")
        print("Please check the file path and try again.")
        return

    # Initialize a dictionary to store the raw text for each group
    grouped_results = {
        0: [],  # Group 1
        1: [],  # Group 2A
        2: [],  # Group 2B
        3: [],  # Group 3
        "other": []
    }

    # Create a human-readable mapping for printing
    group_names = {
        0: "Group 1 (Carcinogenic)",
        1: "Group 2A (Probably Carcinogenic)",
        2: "Group 2B (Possibly Carcinogenic)",
        3: "Group 3 (Not Classifiable)",
        "other": "Other / Unmatched"
    }

    print("\nProcessing carcinogenicity entries...")
    # Drop rows where the 'carcinogenicity' column is missing
    df.dropna(subset=['Carcinogenicity'], inplace=True)

    for text in df['Carcinogenicity']:
        normalized_code = normalize_carcinogenicity(text)
        grouped_results[normalized_code].append(text)

    # --- Print the results ---
    print("\n--- Normalization Summary ---")
    for code, name in group_names.items():
        count = len(grouped_results[code])
        print(f"{name} (Code: {code}): {count} entries")
    print("\n--- Normalization Results (Sorted by Count) ---")
    for code, name in group_names.items():
        entries = grouped_results[code]
        print(f"\n{name} (Code: {code}) - Total Unique Entries: {len(set(entries))}")
        print("=" * 80)

        if not entries:
            print("  (No entries found in this category)")
            continue

        # --- MODIFICATION: Use pandas to count and sort unique values ---
        # Convert the list of strings to a pandas Series
        series = pd.Series(entries)
        # value_counts() gets unique values and their counts, sorted in descending order
        counted_entries = series.value_counts()

        print(f"  {'Count':<6} | Original Text")
        print(f"  {'------':<6} | --------------------------------------------------")
        text_len = 200
        for text, count in counted_entries.items():
            # Truncate very long text for better readability in the console
            display_text = (text[:text_len] + '...') if len(text) > text_len else text
            print(f"  {count:<6} | {display_text}")


if __name__ == "__main__":
    main()