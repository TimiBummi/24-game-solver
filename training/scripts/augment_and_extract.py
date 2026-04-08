"""
Generate labeled corner crops for rank classifier training.

Two-phase augmentation:
  Phase A (before YOLO): positional transforms — rotation, crop+pad.
                         YOLO must locate the card despite these changes.
  Phase B (after crop):  pixel transforms — brightness, contrast, noise, blur.
                         Applied to each 128×128 corner crop independently.

Input:  CSV with columns: image_path, suit_rank_label  (e.g. "2d", "Jh")
Output: training/data/real_crops_v2/{rank}/{suitrank}_aug{i:03d}_p{j:02d}_{tl|br}.jpg

Folder = rank only (used as class label by Keras image_dataset_from_directory).
Filename encodes full suit+rank for deduplication when training.

Usage:
    python augment_and_extract.py
    python augment_and_extract.py --csv training/test_data_labels.csv \
                                  --out training/data/real_crops_v2 \
                                  --pos-variants 10 --pix-variants 3
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ── locate project root & models ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _SCRIPT_DIR)   # so we can import from test_pipeline

# Import YOLO helpers from test_pipeline (avoids code duplication)
from test_pipeline import (
    make_interpreter,
    _run_detector,
    _find_br_indices,
    DEFAULT_DETECTOR,
    CLASSIFIER_INPUT_SIZE,
)

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CSV = os.path.join(_PROJECT_DIR, 'training', 'test_data_labels.csv')
# Photos live at training/data/photos/ (moved from training/test_data/)
DEFAULT_OUT = os.path.join(_PROJECT_DIR, 'training', 'data', 'real_crops_v2')

_SUITS = {'d', 'h', 's', 'c', 'D', 'H', 'S', 'C'}

# Rank-only label → folder name (matches training CLASS_NAMES sort order)
_RANK_MAP = {
    'A': 'A', 'a': 'A',
    '2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
    '7': '7', '8': '8', '9': '9', '10': '10',
    'J': 'J', 'j': 'J', 'B': 'J', 'b': 'J',
    'Q': 'Q', 'q': 'Q', 'D': 'Q',
    'K': 'K', 'k': 'K',
}


# ── label parsing ─────────────────────────────────────────────────────────────

def parse_label(label: str):
    """Return (rank_folder, suit_rank_key) from a label like '2d' or 'Jh'.

    rank_folder: classifier class name, e.g. '2', 'J', 'K'
    suit_rank_key: full label for filename, lowercased, e.g. '2d', 'jh', 'kc'
    """
    label = label.strip()
    if len(label) > 1 and label[-1].lower() in _SUITS:
        rank_token = label[:-1]
        suit_char = label[-1].lower()
    else:
        rank_token = label
        suit_char = 'x'   # unknown suit

    rank_folder = _RANK_MAP.get(rank_token) or _RANK_MAP.get(rank_token.upper())
    if rank_folder is None:
        return None, None
    return rank_folder, f'{rank_token.upper()}{suit_char}'


# ── Phase A: positional augmentations ─────────────────────────────────────────

def positional_augment(img_pil: Image.Image, seed: int) -> Image.Image:
    """Apply position-changing transforms. YOLO must still find the card after."""
    rng = random.Random(seed)

    # 1. Rotation ±30°
    angle = rng.uniform(-30, 30)
    bg = _median_border_color(img_pil)
    img_pil = img_pil.rotate(angle, expand=False, fillcolor=bg)

    # 2. Random crop + re-pad (simulate different framing / zoom)
    w, h = img_pil.size
    # Crop 5–15% from each edge independently
    left   = int(rng.uniform(0.00, 0.12) * w)
    right  = int(rng.uniform(0.00, 0.12) * w)
    top    = int(rng.uniform(0.00, 0.12) * h)
    bottom = int(rng.uniform(0.00, 0.12) * h)
    cropped = img_pil.crop((left, top, w - right, h - bottom))
    # Pad back to original size so YOLO sees the expected resolution
    img_pil = Image.new('RGB', (w, h), color=bg)
    img_pil.paste(cropped, (left, top))

    return img_pil


def _median_border_color(img_pil: Image.Image):
    """Estimate background color from image border pixels."""
    arr = np.array(img_pil)
    border = np.concatenate([
        arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]
    ])
    return tuple(int(x) for x in np.median(border, axis=0).astype(int))


# ── Phase B: pixel augmentations ──────────────────────────────────────────────

def pixel_augment(crop_arr: np.ndarray, seed: int) -> np.ndarray:
    """Apply non-positional pixel transforms to a 128×128 corner crop."""
    rng = random.Random(seed)
    img = Image.fromarray(crop_arr)

    # Brightness
    factor = rng.uniform(0.6, 1.4)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast
    factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Gaussian noise (30% chance)
    if rng.random() < 0.3:
        arr = np.array(img, dtype=np.float32)
        noise = rng.uniform(2, 12)
        arr = arr + np.random.default_rng(seed).normal(0, noise, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Gaussian blur (20% chance)
    if rng.random() < 0.2:
        radius = rng.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    return np.array(img, dtype=np.uint8)


# ── corner extraction ─────────────────────────────────────────────────────────

def extract_corners(img_arr: np.ndarray, detector_interp) -> list[np.ndarray]:
    """Run YOLO and return list of 128×128 corner crops (BR already rotated 180°).

    Returns empty list if no card detected.
    """
    detections = _run_detector(detector_interp, img_arr)
    if not detections:
        return []

    h, w = img_arr.shape[:2]
    br_indices = _find_br_indices(detections)
    crops = []

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
        if i in br_indices:
            crop = np.rot90(crop, 2)           # rotate 180°
        crop = np.array(
            Image.fromarray(crop).resize(
                (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE), Image.BILINEAR
            )
        )
        tag = 'br' if i in br_indices else 'tl'
        crops.append((crop, tag))

    return crops


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Augment photos and extract corner crops.')
    parser.add_argument('--csv', default=DEFAULT_CSV,
                        help=f'Input CSV (default: {DEFAULT_CSV})')
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help=f'Output directory (default: {DEFAULT_OUT})')
    parser.add_argument('--detector', default=DEFAULT_DETECTOR,
                        help='card_detector.tflite path')
    parser.add_argument('--pos-variants', type=int, default=10, metavar='N',
                        help='Positional augmentation variants per image (default: 10)')
    parser.add_argument('--pix-variants', type=int, default=3, metavar='M',
                        help='Pixel augmentation variants per corner crop (default: 3)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f'Loading detector: {args.detector}')
    detector = make_interpreter(args.detector)

    # Read CSV
    rows = []
    with open(args.csv, newline='') as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                rows.append((row[0].strip(), row[1].strip() if len(row) > 1 else ''))
    print(f'Found {len(rows)} images in CSV.\n')

    total_crops = 0
    skipped = 0

    for img_idx, (img_path, label_str) in enumerate(rows):
        rank_folder, suit_rank_key = parse_label(label_str)
        if rank_folder is None:
            print(f'  SKIP  {img_path}  (unknown label "{label_str}")')
            skipped += 1
            continue

        if not os.path.exists(img_path):
            print(f'  SKIP  {img_path}  (file not found)')
            skipped += 1
            continue

        img_pil = Image.open(img_path).convert('RGB')
        out_rank_dir = os.path.join(args.out, rank_folder)
        os.makedirs(out_rank_dir, exist_ok=True)

        img_crops = 0

        # Unaugmented original first (pos_variant 0)
        for pos_var in range(args.pos_variants):
            pos_seed = rng.randint(0, 2**31)

            if pos_var == 0:
                aug_img = img_pil          # keep original for variant 0
            else:
                aug_img = positional_augment(img_pil, seed=pos_seed)

            aug_arr = np.array(aug_img, dtype=np.uint8)
            corner_crops = extract_corners(aug_arr, detector)

            if not corner_crops:
                continue

            for crop_arr, corner_tag in corner_crops:
                for pix_var in range(args.pix_variants):
                    pix_seed = rng.randint(0, 2**31)

                    if pix_var == 0:
                        final_crop = crop_arr          # keep raw crop for variant 0
                    else:
                        final_crop = pixel_augment(crop_arr, seed=pix_seed)

                    fname = (
                        f'{suit_rank_key}_'
                        f'aug{pos_var:03d}_'
                        f'p{pix_var:02d}_'
                        f'{img_idx:04d}_'
                        f'{corner_tag}.jpg'
                    )
                    out_path = os.path.join(out_rank_dir, fname)
                    Image.fromarray(final_crop).save(out_path, quality=90)
                    img_crops += 1

        total_crops += img_crops
        if (img_idx + 1) % 20 == 0 or img_idx == len(rows) - 1:
            print(f'  [{img_idx+1:3d}/{len(rows)}]  {img_path}  '
                  f'label={label_str}  crops_this_img={img_crops}  '
                  f'total={total_crops}')

    print(f'\nDone — {total_crops} crops saved to {args.out}/')
    print(f'Skipped {skipped} images.')

    # Summary by rank
    print('\nCrops per rank:')
    for rank in sorted(os.listdir(args.out)):
        rank_dir = os.path.join(args.out, rank)
        if os.path.isdir(rank_dir):
            n = len(os.listdir(rank_dir))
            print(f'  {rank:>3}: {n}')


if __name__ == '__main__':
    main()
