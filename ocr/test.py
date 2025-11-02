import requests
from pathlib import Path


def get_off_image_folder(barcode: str) -> str:
    """Compute the Open Food Facts image folder path from a barcode."""
    # Pad to 13 digits with leading zeros
    padded = barcode.zfill(13)
    # Split as (3)(3)(3)(remaining)
    part1 = padded[0:3]
    part2 = padded[3:6]
    part3 = padded[6:9]
    part4 = padded[9:]
    return f"{part1}/{part2}/{part3}/{part4}"


# Config
DOWNLOAD_DIR = Path("ingredient_images")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Real product barcodes (13-digit EANs)
product_codes = [
    "3017620422034",  # Lesieur Margarine
    "3270190113057",  # Biscuit BN
    "3029330020760",  # Coca-Cola
    "3270190113118",  # Another BN variant
    "3017620429552",  # Butter
    "4012359114303"
]

for code in product_codes:
    folder = get_off_image_folder(code)
    url = f"https://images.openfoodfacts.org/images/products/{folder}/1.400.jpg"
    filepath = DOWNLOAD_DIR / f"{code}.jpg"

    print(f"Trying: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded: {filepath.name}")
    else:
        print(f"❌ Failed (HTTP {response.status_code}) for {code}")

print(f"\n✅ Done! Check '{DOWNLOAD_DIR}'")