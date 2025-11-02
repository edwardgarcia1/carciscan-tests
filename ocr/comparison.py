import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from PIL import Image

# OCR imports
from paddleocr import PaddleOCR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pytesseract


class OCRComparator:
    def __init__(self):
        # Initialize PaddleOCR
        self.paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,  # Set to True if you have GPU
            show_log=False
        )

        # Initialize DocTR
        self.doctr_model = ocr_predictor(
            pretrained=True,
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn'
        )

        # Tesseract configuration
        self.tesseract_config = '--psm 6'  # Assume uniform block of text

    def run_paddleocr(self, image_path: str) -> Tuple[str, float, List]:
        """Run PaddleOCR and return text, confidence, and boxes"""
        start_time = time.time()
        result = self.paddle_ocr.ocr(image_path, cls=True)
        end_time = time.time()

        if not result or not result[0]:
            return "", end_time - start_time, []

        # Extract text and confidence
        text_parts = []
        confidences = []
        boxes = []

        for line in result[0]:
            box = line[0]
            text = line[1][0]
            confidence = line[1][1]

            text_parts.append(text)
            confidences.append(confidence)
            boxes.append(box)

        avg_confidence = np.mean(confidences) if confidences else 0
        full_text = "\n".join(text_parts)

        return full_text, end_time - start_time, avg_confidence

    def run_doctr(self, image_path: str) -> Tuple[str, float, float]:
        """Run DocTR and return text, processing time, and confidence"""
        start_time = time.time()

        # Load document
        doc = DocumentFile.from_images(image_path)
        result = self.doctr_model(doc)

        end_time = time.time()

        # Extract text and confidence
        full_text = ""
        all_confidences = []

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ""
                    line_confidences = []
                    for word in line.words:
                        line_text += word.value + " "
                        line_confidences.append(word.confidence)
                    full_text += line_text.strip() + "\n"
                    all_confidences.extend(line_confidences)

        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        return full_text.strip(), end_time - start_time, avg_confidence

    def run_tesseract(self, image_path: str) -> Tuple[str, float, float]:
        """Run Tesseract and return text, processing time, and confidence"""
        start_time = time.time()

        # Get text and confidence data
        try:
            # Get detailed data including confidence
            data = pytesseract.image_to_data(
                Image.open(image_path),
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )

            # Extract text and confidence
            text_parts = []
            confidences = []

            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Valid confidence
                    text_parts.append(data['text'][i])
                    confidences.append(int(data['conf'][i]) / 100.0)

            full_text = " ".join([t for t in text_parts if t.strip()])
            avg_confidence = np.mean(confidences) if confidences else 0

        except Exception as e:
            print(f"Tesseract error: {e}")
            full_text = ""
            avg_confidence = 0

        end_time = time.time()
        return full_text, end_time - start_time, avg_confidence

    def calculate_accuracy(self, predicted: str, ground_truth: str) -> float:
        """Calculate character-level accuracy"""
        if not ground_truth:
            return 0.0

        # Simple character-level accuracy
        correct_chars = sum(1 for p, g in zip(predicted.lower(), ground_truth.lower())
                            if p == g)
        total_chars = len(ground_truth)
        return correct_chars / total_chars if total_chars > 0 else 0.0

    def compare_on_image(self, image_path: str, ground_truth: str = None) -> Dict:
        """Compare all three OCR engines on a single image"""
        results = {}

        # PaddleOCR
        try:
            text, time_taken, confidence = self.run_paddleocr(image_path)
            results['paddleocr'] = {
                'text': text,
                'time': time_taken,
                'confidence': confidence,
                'accuracy': self.calculate_accuracy(text, ground_truth) if ground_truth else None
            }
        except Exception as e:
            results['paddleocr'] = {'error': str(e)}

        # DocTR
        try:
            text, time_taken, confidence = self.run_doctr(image_path)
            results['doctr'] = {
                'text': text,
                'time': time_taken,
                'confidence': confidence,
                'accuracy': self.calculate_accuracy(text, ground_truth) if ground_truth else None
            }
        except Exception as e:
            results['doctr'] = {'error': str(e)}

        # Tesseract
        try:
            text, time_taken, confidence = self.run_tesseract(image_path)
            results['tesseract'] = {
                'text': text,
                'time': time_taken,
                'confidence': confidence,
                'accuracy': self.calculate_accuracy(text, ground_truth) if ground_truth else None
            }
        except Exception as e:
            results['tesseract'] = {'error': str(e)}

        return results

    def batch_compare(self, image_paths: List[str], ground_truths: List[str] = None) -> Dict:
        """Compare on multiple images"""
        if ground_truths is None:
            ground_truths = [None] * len(image_paths)

        batch_results = {}

        for i, (img_path, gt) in enumerate(zip(image_paths, ground_truths)):
            print(f"Processing image {i + 1}/{len(image_paths)}: {Path(img_path).name}")
            batch_results[img_path] = self.compare_on_image(img_path, gt)

        return batch_results

    def generate_summary_report(self, batch_results: Dict) -> Dict:
        """Generate summary statistics"""
        summary = {
            'paddleocr': {'avg_time': 0, 'avg_confidence': 0, 'avg_accuracy': 0, 'success_rate': 0},
            'doctr': {'avg_time': 0, 'avg_confidence': 0, 'avg_accuracy': 0, 'success_rate': 0},
            'tesseract': {'avg_time': 0, 'avg_confidence': 0, 'avg_accuracy': 0, 'success_rate': 0}
        }

        engines = ['paddleocr', 'doctr', 'tesseract']
        metrics = ['time', 'confidence', 'accuracy']

        for engine in engines:
            valid_results = []
            total_images = len(batch_results)
            successful = 0

            for img_results in batch_results.values():
                if engine in img_results and 'error' not in img_results[engine]:
                    valid_results.append(img_results[engine])
                    successful += 1

            if valid_results:
                for metric in metrics:
                    values = [r[metric] for r in valid_results if r[metric] is not None]
                    if values:
                        summary[engine][f'avg_{metric}'] = np.mean(values)

                summary[engine]['success_rate'] = successful / total_images

        return summary


# Usage example
def main():
    # Initialize comparator
    comparator = OCRComparator()

    # Test on single image
    test_image = "path/to/your/test_image.jpg"
    results = comparator.compare_on_image(test_image)

    print("Single Image Results:")
    for engine, result in results.items():
        if 'error' not in result:
            print(f"\n{engine.upper()}:")
            print(f"  Time: {result['time']:.2f}s")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Text preview: {result['text'][:100]}...")
        else:
            print(f"{engine.upper()}: Error - {result['error']}")

    # Batch comparison (if you have ground truth)
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    ground_truths = ["ground truth 1", "ground truth 2", "ground truth 3"]  # Optional

    batch_results = comparator.batch_compare(image_paths, ground_truths)
    summary = comparator.generate_summary_report(batch_results)

    print("\nSummary Report:")
    for engine, stats in summary.items():
        print(f"\n{engine.upper()}:")
        print(f"  Avg Time: {stats['avg_time']:.2f}s")
        print(f"  Avg Confidence: {stats['avg_confidence']:.3f}")
        print(f"  Avg Accuracy: {stats['avg_accuracy']:.3f}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")


if __name__ == "__main__":
    main()