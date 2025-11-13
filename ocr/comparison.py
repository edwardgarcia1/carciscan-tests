import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import pandas as pd

# OCR imports
from paddleocr import PaddleOCR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pytesseract


class OCRComparator:
    def __init__(self):
        # Initialize PaddleOCR
        self.paddle_ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=480, #480: 0.1461 0.2554, 0.5879s
            text_det_limit_type='max',
            lang="en",
        )

        # Initialize DocTR
        self.doctr_model = ocr_predictor(
            pretrained=True,
            det_arch='fast_base',
            reco_arch='parseq',
        )

        # Tesseract configuration
        self.tesseract_config = '--psm 6'

    def normalize_text(self, text: str) -> str:
        """Normalize text by converting to lowercase and replacing tabs/newlines with spaces"""
        if not text:
            return ""
        # Replace tabs and newlines with spaces, then normalize whitespace
        normalized = ' '.join(text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').split())
        return normalized.lower()

    def calculate_cer(self, predicted: str, ground_truth: str) -> float:
        """Calculate Character Error Rate (CER)"""
        if not ground_truth:
            return 1.0  # If no ground truth, CER is 100%

        # Normalize both texts
        pred = self.normalize_text(predicted)
        gt = self.normalize_text(ground_truth)

        # Calculate edit distance using dynamic programming
        n, m = len(pred), len(gt)
        if n == 0:
            return 1.0 if m > 0 else 0.0
        if m == 0:
            return 1.0

        # Create DP table
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        # Initialize base cases
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        # Fill the DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if pred[i - 1] == gt[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        # Calculate CER
        edit_distance = dp[n][m]
        cer = edit_distance / max(len(gt), 1)
        return round(cer, 4)

    def calculate_wer(self, predicted: str, ground_truth: str) -> float:
        """Calculate Word Error Rate (WER)"""
        if not ground_truth:
            return 1.0  # If no ground truth, WER is 100%

        # Normalize both texts
        pred_words = self.normalize_text(predicted).split()
        gt_words = self.normalize_text(ground_truth).split()

        n, m = len(pred_words), len(gt_words)
        if n == 0:
            return 1.0 if m > 0 else 0.0
        if m == 0:
            return 1.0

        # Create DP table for WER
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        # Initialize base cases
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        # Fill the DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if pred_words[i - 1] == gt_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        # Calculate WER
        edit_distance = dp[n][m]
        wer = edit_distance / max(len(gt_words), 1)
        return round(wer, 4)

    def run_paddleocr(self, image_path: str) -> Tuple[str, float]:
        """Run PaddleOCR and return text and processing time"""
        start_time = time.time()
        result = self.paddle_ocr.predict(image_path)
        end_time = time.time()

        if not result or not result[0]:
            return "", end_time - start_time

        for res in result:
            texts = res['rec_texts']

            single_line = " ".join(texts)
        return single_line, round(end_time - start_time, 4)

    def run_doctr(self, image_path: str) -> Tuple[str, float]:
        """Run DocTR and return text and processing time"""
        start_time = time.time()

        # Load document
        doc = DocumentFile.from_images(image_path)
        result = self.doctr_model(doc)

        end_time = time.time()

        # Extract text
        full_text = ""
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ""
                    for word in line.words:
                        line_text += word.value + " "
                    full_text += line_text.strip() + "\n"

        return full_text.strip(), round(end_time - start_time, 4)

    def run_tesseract(self, image_path: str) -> Tuple[str, float]:
        """Run Tesseract and return text and processing time"""
        start_time = time.time()

        try:
            text = pytesseract.image_to_string(
                image_path,
                config=self.tesseract_config
            ).strip()
        except Exception as e:
            print(f"Tesseract error: {e}")
            text = ""

        end_time = time.time()
        return text, round(end_time - start_time, 4)

    def compare_on_image(self, image_path: str, ground_truth: str = None) -> Dict:
        """Compare all three OCR engines on a single image"""
        results = {}

        # PaddleOCR
        try:
            text, time_taken = self.run_paddleocr(image_path)
            results['paddleocr'] = {
                'text': text,
                'time': time_taken,
                'cer': self.calculate_cer(text, ground_truth) if ground_truth else None,
                'wer': self.calculate_wer(text, ground_truth) if ground_truth else None
            }
        except Exception as e:
            results['paddleocr'] = {'error': str(e)}

        # DocTR
        try:
            text, time_taken = self.run_doctr(image_path)
            results['doctr'] = {
                'text': text,
                'time': time_taken,
                'cer': self.calculate_cer(text, ground_truth) if ground_truth else None,
                'wer': self.calculate_wer(text, ground_truth) if ground_truth else None
            }
        except Exception as e:
            results['doctr'] = {'error': str(e)}

        # Tesseract
        try:
            text, time_taken = self.run_tesseract(image_path)
            results['tesseract'] = {
                'text': text,
                'time': time_taken,
                'cer': self.calculate_cer(text, ground_truth) if ground_truth else None,
                'wer': self.calculate_wer(text, ground_truth) if ground_truth else None
            }
        except Exception as e:
            results['tesseract'] = {'error': str(e)}

        return results

    def batch_compare(self, image_paths: List[str], ground_truths: Dict[str, str] = None) -> Dict:
        """Compare on multiple images"""
        batch_results = {}

        for i, img_path in enumerate(image_paths):
            # Extract image ID from filename (remove extension)
            img_id = Path(img_path).stem
            gt = ground_truths.get(img_id) if ground_truths else None

            print(f"Processing image {i + 1}/{len(image_paths)}: {Path(img_path).name}")
            batch_results[img_id] = self.compare_on_image(img_path, gt)

        return batch_results

    def save_results_to_csv(self, batch_results: Dict, ground_truths: Dict[str, str], output_dir: str):
        """Save results to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Prepare performance comparison data
        perf_data = []
        text_comparison_data = []

        for img_id, results in batch_results.items():
            gt_text = ground_truths.get(img_id, "") if ground_truths else ""
            normalized_gt_text = self.normalize_text(gt_text)

            for engine in ['paddleocr', 'doctr', 'tesseract']:
                if engine in results and 'error' not in results[engine]:
                    result = results[engine]
                    perf_data.append({
                        'image_id': img_id,
                        'engine': engine,
                        'cer': result['cer'],
                        'wer': result['wer'],
                        'time': result['time']
                    })

                    # Normalize the predicted text for the text comparison
                    normalized_pred_text = self.normalize_text(result['text'])
                    text_comparison_data.append({
                        'image_id': img_id,
                        'engine': engine,
                        'predicted_text': normalized_pred_text,
                        'actual_text': normalized_gt_text
                    })
                else:
                    perf_data.append({
                        'image_id': img_id,
                        'engine': engine,
                        'cer': None,
                        'wer': None,
                        'time': None
                    })

                    text_comparison_data.append({
                        'image_id': img_id,
                        'engine': engine,
                        'predicted_text': "",
                        'actual_text': normalized_gt_text
                    })

        # Save performance comparison
        perf_df = pd.DataFrame(perf_data)
        perf_df.to_csv(output_path / 'performance_comparison.csv', index=False, float_format='%.4f')

        # Save text comparison
        text_df = pd.DataFrame(text_comparison_data)
        text_df.to_csv(output_path / 'text_comparison.csv', index=False)

        # Calculate and save summary
        engines = ['paddleocr', 'doctr', 'tesseract']
        summary_data = []

        for engine in engines:
            valid_results = []
            for img_id, results in batch_results.items():
                if engine in results and 'error' not in results[engine]:
                    valid_results.append(results[engine])

            if valid_results:
                avg_cer = np.mean([r['cer'] for r in valid_results if r['cer'] is not None])
                avg_wer = np.mean([r['wer'] for r in valid_results if r['wer'] is not None])
                avg_time = np.mean([r['time'] for r in valid_results if r['time'] is not None])

                summary_data.append({
                    'engine': engine,
                    'avg_cer': round(avg_cer, 4),
                    'avg_wer': round(avg_wer, 4),
                    'avg_time': round(avg_time, 4)
                })
            else:
                summary_data.append({
                    'engine': engine,
                    'avg_cer': None,
                    'avg_wer': None,
                    'avg_time': None
                })

        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / 'summary.csv', index=False, float_format='%.4f')

        print(f"Results saved to {output_path}/")


def load_ground_truth(data_dir: str) -> Dict[str, str]:
    """Load ground truth from JSON file"""
    gt_path = Path(data_dir) / "ground_truth.json"
    with open(gt_path, 'r') as f:
        return json.load(f)


def main():
    # Define data directory
    data_dir = Path("../data/ingredients_images_ph")
    results_dir = Path("results")

    # Load ground truth
    ground_truths = load_ground_truth(str(data_dir))

    # Create list of image paths
    image_paths = []
    for i in range(1, 23):  # Images from 001.jpg to 020.jpg
        img_path = data_dir / f"{i:03d}.jpg"
        if img_path.exists():
            image_paths.append(str(img_path))
        else:
            print(f"Warning: {img_path} not found")

    print(f"Found {len(image_paths)} images")

    # Initialize comparator
    comparator = OCRComparator()

    # Run batch comparison
    print("Starting batch comparison...")
    batch_results = comparator.batch_compare(image_paths, ground_truths)

    # Save results to CSV files
    comparator.save_results_to_csv(batch_results, ground_truths, str(results_dir))

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    engines = ['paddleocr', 'doctr', 'tesseract']
    for engine in engines:
        valid_results = []
        for img_id, results in batch_results.items():
            if engine in results and 'error' not in results[engine]:
                valid_results.append(results[engine])

        if valid_results:
            avg_cer = np.mean([r['cer'] for r in valid_results if r['cer'] is not None])
            avg_wer = np.mean([r['wer'] for r in valid_results if r['wer'] is not None])
            avg_time = np.mean([r['time'] for r in valid_results if r['time'] is not None])

            print(f"\n{engine.upper()}:")
            print(f"  Average CER: {avg_cer:.4f}")
            print(f"  Average WER: {avg_wer:.4f}")
            print(f"  Average Time: {avg_time:.4f}s")


if __name__ == "__main__":
    main()