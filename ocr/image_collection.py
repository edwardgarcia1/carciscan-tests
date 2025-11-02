import pandas as pd
import requests
import os
from pathlib import Path
from tqdm import tqdm
import time
import logging
import gc
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_off_image_folder(barcode: str) -> str:
    """Compute the Open Food Facts image folder path from a barcode."""
    padded = str(barcode).zfill(13)
    return f"{padded[0:3]}/{padded[3:6]}/{padded[6:9]}/{padded[9:]}"


class IngredientImageDownloader:
    def __init__(self, parquet_path: str, download_dir: str = "../data/ingredient_images"):
        self.parquet_path = Path(parquet_path)
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.ground_truth_file = self.download_dir / "ground_truth.json"

        file_size = self.parquet_path.stat().st_size / (1024 ** 3)  # GB
        logger.info(f"Parquet file size: {file_size:.2f} GB")

    def parse_ingredient_text(self, ingredient_text_field) -> str:
        """Extract ingredient text; prefer 'main', then 'en'."""
        if ingredient_text_field.size == 0:
            return ""
        try:
            entries = json.loads(ingredient_text_field) if isinstance(ingredient_text_field,
                                                                      str) else ingredient_text_field
            if not isinstance(entries, list):
                return ""
            for entry in entries:
                if entry.get("lang") == "main" and entry.get("text"):
                    return str(entry["text"]).strip()
            for entry in entries:
                if entry.get("lang") == "en" and entry.get("text"):
                    return str(entry["text"]).strip()
            return ""
        except Exception as e:
            logger.debug(f"Failed to parse ingredients_text: {e}")
            return ""

    def has_valid_ingredient_image(self, images_field) -> bool:
        """Check if images list contains an ingredient image (key contains 'ingredients')."""
        if images_field.size <= 0:
            return False
        try:
            images = json.loads(images_field) if isinstance(images_field, str) else images_field
            if not isinstance(images, list):
                return False
            return any(
                isinstance(img.get("key"), str) and "ingredients" in img["key"]
                for img in images
            )
        except Exception as e:
            logger.debug(f"Failed to parse images: {e}")
            return False

    def get_ingredient_image_url(self, product_code: str, images_field) -> str:
        """Build the correct OFF ingredient image URL using folder structure."""
        try:
            images = json.loads(images_field) if isinstance(images_field, str) else images_field
            if not isinstance(images, list):
                return ""

            # Prefer 'ingredients_en', then any 'ingredients_*'
            target_key = None
            target_rev = None
            for img in images:
                key = img.get("key")
                if key == "ingredients_en":
                    target_key = key
                    target_rev = img.get("rev")
                    break
            if not target_key:
                for img in images:
                    key = img.get("key")
                    if isinstance(key, str) and "ingredients" in key:
                        target_key = key
                        target_rev = img.get("rev")
                        break
            if not target_key:
                return ""

            folder = get_off_image_folder(product_code)
            return f"https://images.openfoodfacts.org/images/products/{folder}/{target_key}/{target_rev}.400.jpg"
        except Exception as e:
            logger.debug(f"Failed to build image URL for {product_code}: {e}")
            return ""

    def process_parquet_in_chunks(self, chunk_size: int = 10000):
        """Process parquet file in chunks and collect URLs + ground truth."""
        logger.info("Processing parquet file in chunks...")

        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(self.parquet_path)
        total_rows = parquet_file.metadata.num_rows
        logger.info(f"Total rows in dataset: {total_rows:,}")

        ground_truth_data = {}
        url_to_product = {}

        required_cols = {'lang', 'ingredients_text', 'images', 'code'}
        for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size),
                          total=(total_rows // chunk_size) + 1,
                          desc="Processing chunks"):
            try:
                chunk_df = batch.to_pandas()

                missing_cols = required_cols - set(chunk_df.columns)
                if missing_cols:
                    logger.warning(f"Missing columns: {missing_cols}")
                    continue

                for idx, row in chunk_df.iterrows():
                    # Check language
                    if row.get('lang') != 'en':
                        continue

                    product_code = row.get('code')
                    if pd.isna(product_code):
                        continue
                    product_code = str(product_code).strip()
                    if not product_code.isdigit():
                        continue

                    # Parse ingredient text
                    ingredient_text = self.parse_ingredient_text(row.get('ingredients_text'))
                    if not ingredient_text:
                        continue

                    # Check for valid ingredient image metadata
                    if not self.has_valid_ingredient_image(row.get('images')):
                        continue

                    # Build image URL
                    image_url = self.get_ingredient_image_url(product_code, row.get('images'))
                    if not image_url:
                        continue

                    filename = f"product_{product_code}"
                    url_to_product[image_url] = filename
                    ground_truth_data[filename] = ingredient_text

                del chunk_df, batch
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                continue

        logger.info(
            f"Found {len(ground_truth_data)} products with lang='en', valid ingredients_text, and ingredient images")
        return url_to_product, ground_truth_data

    def download_image(self, url: str, filename: str, timeout: int = 30) -> bool:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Open Food Facts Data Collector)'
            }
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            if response.status_code == 200:
                ext = '.jpg'  # OFF uses .jpg for .400.jpg
                filepath = self.download_dir / f"{filename}{ext}"
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            return False
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            return False

    def save_ground_truth(self, ground_truth_data: dict):
        with open(self.ground_truth_file, 'w', encoding='utf-8') as f:
            json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Ground truth saved to {self.ground_truth_file}")

    def download_all_with_ground_truth(self, max_images: int = None, delay: float = 0.5):
        url_to_product, ground_truth_data = self.process_parquet_in_chunks(chunk_size=15000)

        if not ground_truth_data:
            logger.error("No valid products found!")
            return

        if max_images:
            limited = list(ground_truth_data.items())[:max_images]
            ground_truth_data = dict(limited)
            valid_files = set(ground_truth_data.keys())
            url_to_product = {u: f for u, f in url_to_product.items() if f in valid_files}

        self.save_ground_truth(ground_truth_data)

        logger.info(f"Starting download of {len(url_to_product)} images")
        success_count = 0
        downloaded = []

        for url, filename in tqdm(url_to_product.items(), desc="Downloading"):
            if filename not in ground_truth_data:
                continue
            if self.download_image(url, filename):
                success_count += 1
                downloaded.append(filename)
            else:
                logger.warning(f"Failed to download image for {filename}")
            time.sleep(delay)

        final_gt = {k: v for k, v in ground_truth_data.items() if k in downloaded}
        self.save_ground_truth(final_gt)

        logger.info(f"✅ Completed: {success_count} images downloaded")
        logger.info(f"Final ground truth entries: {len(final_gt)}")


def main():
    downloader = IngredientImageDownloader(
        parquet_path="../data/food.parquet",
        download_dir="../data/ingredient_images"
    )
    downloader.download_all_with_ground_truth(max_images=20, delay=0.3)


if __name__ == "__main__":
    main()