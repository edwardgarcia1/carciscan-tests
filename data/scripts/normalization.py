import ast
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
    "Group 1" : Group 1 (Carcinogenic to humans)
    "Group 2A" : Group 2A (Probably carcinogenic to humans)
    "Group 2B" : Group 2B (Possibly carcinogenic to humans)
    "Group 3" : Group 3 (Not classifiable as to its carcinogenicity to humans)
    "No data" : If no pattern matches.

    Args:
        text (str): The input text description of carcinogenicity.
    """
    if not text or not isinstance(text, str):
        return "No data"

    text = text.strip().lower()

    # --- TIER 1: OVERRIDING PATTERNS ---
    # Add patterns here that should ALWAYS take precedence, regardless of position.
    # The script checks this list first. The first match in this list wins.
    overriding_rules = [
        (r"not listed by iarc", "No data"),
        (r"not directly listed by iarc", "No data"),
    ]
    # --- TIER 2: FIRST-MATCH PATTERNS ---
    # These patterns are only checked if no overriding patterns are found.
    # The one that appears earliest in the text will be used.
    first_match_rules = [
        # --- Specific Exclusion Patterns ---
        (r"are not classifiable", "Group 3"),
        (r"is not classifiable", "Group 3"),
        (r"no indication", "Group 3"),

        # --- IARC Group Patterns ---
        (r"not listed by iarc", "Group 3"),
        (r"\bgroup\s*3\b", "Group 3"),
        (r"^3[\s,:] ", "Group 3"),
        (r"\bgroup\s*2b\b", "Group 2B"),
        (r"2b[\s,:]", "Group 2B"),
        (r"\bgroup\s*2a\b", "Group 2A"),
        (r"^2a[\s,:]", "Group 2A"),
        (r"\bgroup\s*1\b", "Group 1"),
        (r"^1[\s,:]", "Group 1"),
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

    # Step 3: Return the result from the first match logic, or "No data".
    if first_match:
        return first_match[1]

    return "No data"

def normalize_route(route_list):
    """
    Normalize a list of route-of-exposure strings by:
    - Filtering out non-route text (e.g., absorption %, bioavailability, full sentences)
    - Mapping valid exposure keywords to standardized labels
    - Only including valid routes in the returned list
    - Returning "No data" as a string if no valid routes are found in the entire list

    Args:
        route_list (list): List of raw exposure strings

    Returns:
        list or str: List of normalized routes (e.g., 'oral', 'inhalation', 'dermal', 'ocular', 'injection'),
                    or 'No data' as a string if no valid routes are found
    """
    # Input validation
    if pd.isna(route_list):
        return ['No data']

    if not isinstance(route_list, list):
        if route_list is not None:
            route_list = ast.literal_eval(route_list)

    # Helper function to normalize a single route
    def _normalize_single_route(text):
        if not isinstance(text, str):
            return None

        text = text.strip()
        if not text or text.lower() in ['nan', 'none', 'null', '-', '']:
            return None

        # Remove content in parentheses (like (L2), (L3), etc.)
        text = re.sub(r'\s*\([^)]*\)', '', text)

        # Convert to lowercase for case-insensitive matching
        lower_text = text.lower()

        # --- Step 1: Discard obviously non-route strings ---
        # Skip strings that are clearly pharmacokinetic descriptions, percentages, or fragments
        if (
                re.search(r'\d+%', lower_text) or
                re.search(
                    r'absorbed|absorption|bioavailability|metabolism|excreted|tmax|cmax|g/ml|virtually complete|poor absorbed|well absorbed|rapidly absorbed|first[- ]?pass|enteral|gastrointestinal',
                    lower_text) or
                'hours' in lower_text or
                'thus,' in lower_text or
                'however,' in lower_text or
                'because of' in lower_text or
                'administered' in lower_text or
                len(lower_text.split()) > 6  # Likely a sentence, not a route label
        ):
            return None

        # --- Step 2: Standardize known route keywords ---
        # Handle eye-related terms
        if any(term in lower_text for term in ['eye contact', 'eyes', 'eye']):
            return 'ocular'

        # Handle injection-related terms
        if any(term in lower_text for term in
               ['injection', 'intravenous', 'subcutaneous', 'intravesical', 'parenteral']):
            return 'injection'

        # Handle radiation
        if 'radiation' in lower_text:
            return 'radiation'

        # General route keywords (must appear as whole words or clear substrings)
        if 'oral' in lower_text or 'ingestion' in lower_text:
            return 'oral'

        if 'inhalation' in lower_text:
            return 'inhalation'

        if 'dermal' in lower_text or 'skin' in lower_text:
            return 'dermal'

        # --- Step 3: Final fallback for very short, clean terms ---
        # If it's a single word and matches a known route, accept it
        words = [w.strip('.,;:') for w in lower_text.split()]
        if len(words) == 1:
            if words[0] in ['oral', 'inhalation', 'dermal', 'injection', 'ocular', 'radiation']:
                return words[0]

        # If nothing matches, return None
        return None

    # Process each item in the list
    normalized_routes = [_normalize_single_route(item) for item in route_list]

    # Filter out None values
    valid_routes = [route for route in normalized_routes if route is not None]

    # If no valid routes found, return 'No data' as a string
    if not valid_routes:
        return ['No data']

    # Otherwise, return the list of valid routes
    return valid_routes

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
        "Group 1": [],
        "Group 2A": [],
        "Group 2B": [],
        "Group 3": [],
        "No data": []
    }

    # Create a human-readable mapping for printing
    group_names = {
        "Group 1": "Group 1 (Carcinogenic)",
        "Group 2A": "Group 2A (Probably Carcinogenic)",
        "Group 2B": "Group 2B (Possibly Carcinogenic)",
        "Group 3": "Group 3 (Not Classifiable)",
        "No data": "No Data / Unmatched"
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