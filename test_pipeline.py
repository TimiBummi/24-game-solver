"""
End-to-end pipeline test: card_detector.tflite → rank_classifier.tflite.

Mirrors the Dart MlPipeline exactly (same preprocessing, NMS, class mapping).

Usage:
    # Single image, check predictions only:
    python test_pipeline.py path/to/image.jpg

    # Single image with expected cards:
    python test_pipeline.py path/to/image.jpg "A 3 7 K"

    # Batch from CSV (columns: image_path, expected_cards):
    python test_pipeline.py --batch tests.csv

    # Save all detected corner crops for fine-tuning (requires ground truth):
    python test_pipeline.py --batch tests.csv --save-crops crops/

    # Override model paths:
    python test_pipeline.py image.jpg --detector path/to/card_detector.tflite \
                                       --classifier path/to/rank_classifier.tflite

Save-crops output layout (ready for Keras ImageDataGenerator / tf.data):
    crops/
      A/  2/  3/  4/  5/  6/  7/  8/  9/  10/  J/  Q/  K/
        <image_id>_det<n>.jpg   ← one file per detected corner
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    def make_interpreter(path):
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        return interp
except ImportError:
    import tflite_runtime.interpreter as tflite
    def make_interpreter(path):
        interp = tflite.Interpreter(model_path=path)
        interp.allocate_tensors()
        return interp

# ---------------------------------------------------------------------------
# Constants — must match Dart code exactly
# ---------------------------------------------------------------------------

DETECTOR_INPUT_SIZE = 320
CLASSIFIER_INPUT_SIZE = 128
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.5
MAX_CARDS = 4

# Class names sorted alphabetically (matches training dataset directory names).
# Index → card label → card value (1=Ace, 11=J, 12=Q, 13=K)
CLASS_NAMES = sorted(['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'K', 'Q'])
INDEX_TO_VALUE = [10, 2, 3, 4, 5, 6, 7, 8, 9, 1, 11, 13, 12]

# Label → value for parsing expected-cards strings
LABEL_TO_VALUE = {
    'A': 1, 'a': 1,
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 11, 'j': 11, 'B': 11, 'b': 11,  # B = Bube (German)
    'Q': 12, 'q': 12, 'D': 12, 'd': 12,  # D = Dame (German)
    'K': 13, 'k': 13,
}
VALUE_TO_LABEL = {
    1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
    7: '7', 8: '8', 9: '9', 10: '10',
    11: 'J', 12: 'Q', 13: 'K',
}

# Default model paths relative to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DETECTOR = os.path.join(
    _SCRIPT_DIR, 'flutter_app', 'assets', 'models', 'card_detector.tflite')
DEFAULT_CLASSIFIER = os.path.join(
    _SCRIPT_DIR, 'flutter_app', 'assets', 'models', 'rank_classifier.tflite')


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    x: float       # normalized top-left x [0,1]
    y: float       # normalized top-left y [0,1]
    w: float       # normalized width
    h: float       # normalized height
    conf: float


@dataclass
class CardResult:
    value: int
    confidence: float
    det: Detection


# ---------------------------------------------------------------------------
# YOLO detector (mirrors yolo_detector.dart)
# ---------------------------------------------------------------------------

def _run_detector(interpreter, image_rgb: np.ndarray) -> List[Detection]:
    """Run card_detector.tflite on an RGB image array."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize + normalize to [0, 1]
    pil = Image.fromarray(image_rgb).resize(
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE), Image.BILINEAR)
    inp = np.array(pil, dtype=np.float32) / 255.0            # [320, 320, 3]
    inp = inp[np.newaxis]                                      # [1, 320, 320, 3]

    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()

    # Output shape: [1, 5, N]  rows = [cx, cy, w, h, conf]
    out = interpreter.get_tensor(output_details[0]['index'])   # [1, 5, N]
    out = out[0]                                               # [5, N]

    conf_row = out[4]
    max_conf = float(conf_row.max()) if len(conf_row) else 0.0
    print(f'  [DETECTOR] output shape: {out.shape}, '
          f'candidates above threshold: {(conf_row >= CONF_THRESHOLD).sum()}, '
          f'max conf: {max_conf:.3f}')

    detections: List[Detection] = []
    for i in range(out.shape[1]):
        conf = float(out[4, i])
        if conf < CONF_THRESHOLD:
            continue
        cx, cy, w, h = float(out[0, i]), float(out[1, i]), float(out[2, i]), float(out[3, i])

        # Model outputs normalized [0,1] center-format — convert to top-left only
        nx = cx - w / 2
        ny = cy - h / 2
        nw = w
        nh = h

        detections.append(Detection(
            x=float(np.clip(nx, 0, 1)),
            y=float(np.clip(ny, 0, 1)),
            w=float(np.clip(nw, 0, 1)),
            h=float(np.clip(nh, 0, 1)),
            conf=conf,
        ))

    return _nms(detections)


def _iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (a.w * a.h + b.w * b.h - inter)


def _nms(detections: List[Detection]) -> List[Detection]:
    detections = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept: List[Detection] = []
    for det in detections:
        if not any(_iou(det, k) > NMS_IOU_THRESHOLD for k in kept):
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# Rank classifier (mirrors rank_classifier.dart)
# ---------------------------------------------------------------------------

def _run_classifier(interpreter, image_rgb: np.ndarray) -> Tuple[int, float]:
    """Classify a card crop. Returns (card_value, confidence)."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    pil = Image.fromarray(image_rgb).resize(
        (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE), Image.BILINEAR)
    inp = np.array(pil, dtype=np.float32) / 255.0             # [128, 128, 3]
    inp = inp[np.newaxis]                                      # [1, 128, 128, 3]

    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]['index'])[0]  # [13]
    best_idx = int(np.argmax(probs))
    return INDEX_TO_VALUE[best_idx], float(probs[best_idx])


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    image_path: str,
    detector_interp,
    classifier_interp,
    save_crops_dir: Optional[str] = None,
    expected_values: Optional[List[int]] = None,
) -> List[CardResult]:
    """Run the full two-stage pipeline on one image.

    If save_crops_dir is set, each detected corner crop is saved to
    save_crops_dir/<label>/<stem>_det<n>.jpg using the predicted label.
    When expected_values is also provided, saves under the TRUE label instead,
    so the dataset is correctly labelled for fine-tuning.
    """
    img_pil = Image.open(image_path).convert('RGB')
    img_arr = np.array(img_pil)  # [H, W, 3] uint8
    h, w = img_arr.shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]

    detections = _run_detector(detector_interp, img_arr)
    print(f'  [DETECTOR] {len(detections)} card(s) after NMS')

    # Classify every detected corner region.
    # Each physical card has two rank indicators (upper-left + lower-right).
    # Classify all, then deduplicate by rank keeping highest confidence read.
    all_preds: List[CardResult] = []
    crops: List[np.ndarray] = []
    for i, det in enumerate(detections):
        px = int(round(det.x * w))
        py = int(round(det.y * h))
        pw = max(1, int(round(det.w * w)))
        ph = max(1, int(round(det.h * h)))
        px = min(px, w - 1)
        py = min(py, h - 1)
        pw = min(pw, w - px)
        ph = min(ph, h - py)

        crop = img_arr[py:py + ph, px:px + pw]
        crops.append(crop)
        value, conf = _run_classifier(classifier_interp, crop)
        all_preds.append(CardResult(value=value, confidence=conf, det=det))
        print(f'  [CLASSIFIER] det {i+1}: {VALUE_TO_LABEL[value]:>2}  '
              f'(conf {conf:.3f})  bbox=({det.x:.2f},{det.y:.2f},'
              f'{det.w:.2f},{det.h:.2f})')

    # Deduplicate: keep highest-confidence prediction per rank value.
    best: dict[int, CardResult] = {}
    for pred in all_preds:
        if pred.value not in best or pred.confidence > best[pred.value].confidence:
            best[pred.value] = pred

    results = sorted(best.values(), key=lambda r: r.confidence, reverse=True)[:MAX_CARDS]
    print(f'  [DEDUP] {len(all_preds)} detections → {len(results)} unique rank(s)')

    # Save crops if requested.
    if save_crops_dir and expected_values:
        _save_crops(save_crops_dir, stem, crops, all_preds, expected_values)

    return results


def _save_crops(
    out_dir: str,
    stem: str,
    crops: List[np.ndarray],
    preds: List[CardResult],
    expected_values: List[int],
) -> None:
    """Save each crop under the correct ground-truth label directory.

    Matching strategy: for each detected corner, assign the true label by
    finding the closest expected value to the predicted value (by rank
    distance). This is imperfect for wrong predictions, but gives correctly
    labelled crops when the model predicts right, and usable near-misses
    otherwise — far better than using predicted labels directly.
    """
    remaining = list(expected_values)  # may have duplicates (e.g., two Kings)

    saved = 0
    for i, (crop, pred) in enumerate(zip(crops, preds)):
        if not remaining:
            break

        # Assign the expected value whose distance to the prediction is smallest.
        best_match = min(remaining, key=lambda v: abs(v - pred.value))
        remaining.remove(best_match)
        label = VALUE_TO_LABEL[best_match]

        label_dir = os.path.join(out_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        out_path = os.path.join(label_dir, f'{stem}_det{i}.jpg')
        Image.fromarray(crop).save(out_path)
        saved += 1

    print(f'  [CROPS] saved {saved} crop(s) to {out_dir}/')


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_expected(label_str: str) -> Optional[List[int]]:
    if not label_str or not label_str.strip():
        return None
    values = []
    for token in label_str.strip().split():
        if token not in LABEL_TO_VALUE:
            print(f'  WARNING: unknown card label "{token}", skipping')
            continue
        values.append(LABEL_TO_VALUE[token])
    return values if values else None


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def test_image(
    image_path: str,
    expected_str: Optional[str],
    detector_interp,
    classifier_interp,
    save_crops_dir: Optional[str] = None,
) -> bool:
    """Run pipeline on one image. Returns True if PASS (or no expected given)."""
    print(f'\nImage: {image_path}')
    if not os.path.exists(image_path):
        print('  ERROR: file not found')
        return False

    expected_values = parse_expected(expected_str) if expected_str else None
    results = run_pipeline(
        image_path, detector_interp, classifier_interp,
        save_crops_dir=save_crops_dir,
        expected_values=expected_values,
    )
    detected_values = sorted(r.value for r in results)
    detected_labels = [VALUE_TO_LABEL[v] for v in sorted(detected_values)]

    if expected_str is None:
        print(f'  Detected: {detected_labels}')
        return True

    if expected_values is None:
        print(f'  Detected: {detected_labels}  (no valid expected labels)')
        return True

    expected_sorted = sorted(expected_values)
    expected_labels = [VALUE_TO_LABEL[v] for v in expected_sorted]

    passed = detected_values == expected_sorted
    status = 'PASS' if passed else 'FAIL'
    print(f'  {status}  expected={expected_labels}  detected={detected_labels}')
    return passed


def run_batch(
    csv_path: str,
    detector_interp,
    classifier_interp,
    save_crops_dir: Optional[str] = None,
) -> None:
    """Run all rows from a CSV file (columns: image_path, expected_cards)."""
    passed = 0
    failed = 0
    errors = 0

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        rows = [(r[0].strip(), r[1].strip() if len(r) > 1 else None)
                for r in reader if r and r[0].strip()]

    for image_path, expected_str in rows:
        ok = test_image(image_path, expected_str, detector_interp, classifier_interp,
                        save_crops_dir=save_crops_dir)
        if expected_str:
            if ok:
                passed += 1
            else:
                failed += 1
        else:
            errors += 1  # ran, but no ground truth

    total = passed + failed
    print(f'\n{"="*50}')
    print(f'Results: {passed}/{total} passed', end='')
    if errors:
        print(f'  ({errors} without expected labels)', end='')
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Test the full card recognition pipeline.')
    parser.add_argument('image', nargs='?', help='Image file to test')
    parser.add_argument('expected', nargs='?', default=None,
                        help='Expected cards, e.g. "A 3 7 K"')
    parser.add_argument('--batch', metavar='CSV',
                        help='CSV file: image_path,expected_cards')
    parser.add_argument('--save-crops', metavar='DIR', dest='save_crops',
                        help='Save detected corner crops to DIR/<label>/*.jpg for fine-tuning')
    parser.add_argument('--detector', default=DEFAULT_DETECTOR,
                        help=f'card_detector.tflite path (default: {DEFAULT_DETECTOR})')
    parser.add_argument('--classifier', default=DEFAULT_CLASSIFIER,
                        help=f'rank_classifier.tflite path (default: {DEFAULT_CLASSIFIER})')
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.print_help()
        sys.exit(1)

    print(f'Loading detector:   {args.detector}')
    detector_interp = make_interpreter(args.detector)
    det_in = detector_interp.get_input_details()[0]
    det_out = detector_interp.get_output_details()[0]
    print(f'  input:  shape={det_in["shape"]}, dtype={det_in["dtype"]}')
    print(f'  output: shape={det_out["shape"]}, dtype={det_out["dtype"]}')

    print(f'Loading classifier: {args.classifier}')
    classifier_interp = make_interpreter(args.classifier)
    cls_in = classifier_interp.get_input_details()[0]
    cls_out = classifier_interp.get_output_details()[0]
    print(f'  input:  shape={cls_in["shape"]}, dtype={cls_in["dtype"]}')
    print(f'  output: shape={cls_out["shape"]}, dtype={cls_out["dtype"]}')

    if args.save_crops:
        print(f'Saving crops to:    {args.save_crops}/')

    if args.batch:
        run_batch(args.batch, detector_interp, classifier_interp,
                  save_crops_dir=args.save_crops)
    else:
        test_image(args.image, args.expected, detector_interp, classifier_interp,
                   save_crops_dir=args.save_crops)


if __name__ == '__main__':
    main()
