"""
Extract rank-indicator corner crops from Kaggle playing-card photos.

Input : data/kaggle-playing-cards/cards-{SUIT}{RANK}-001.jpg
Output: data/real_crops/{rank}/<stem>_tl.jpg  (top-left corner, 128×128)
         data/real_crops/{rank}/<stem>_br.jpg  (bottom-right corner, 128×128)

Rank mapping in filenames: 0→10, 2-9, A, J, Q, K  (suit C/D/H/S ignored)
"""

import os, sys, glob
import cv2
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
INPUT_GLOB  = "data/kaggle-playing-cards/cards-*.jpg"
OUTPUT_DIR  = "data/real_crops"
CROP_SIZE   = 128          # final output resolution (pixels)
CORNER_FRAC = 0.20         # top-left/bottom-right fraction of card to crop

# Filename rank token → classifier label
RANK_MAP = {
    "0": "10", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7",  "8": "8", "9": "9", "A": "A",
    "J": "J",  "Q": "Q", "K": "K",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def order_corners(pts):
    """Return corners in order: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1)
    tl  = pts[np.argmin(s)]
    br  = pts[np.argmax(s)]
    tr  = pts[np.argmin(d)]
    bl  = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_card_corners(img_bgr):
    """Return the 4 card corners as (4,2) float32, or None on failure."""
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (7, 7), 0)
    # Card is white/light on a wooden background → threshold high values
    _, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)
    # Morphological close to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Largest contour by area
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.10 * img_bgr.shape[0] * img_bgr.shape[1]:
        return None   # sanity: card should cover >10% of image
    peri   = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        return order_corners(approx)
    # Fall back: minimum-area bounding rect
    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect)
    return order_corners(box)


def perspective_warp(img_bgr, corners, out_w=600, out_h=400):
    """Warp card to a flat rectangle (landscape orientation)."""
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img_bgr, M, (out_w, out_h))


def crop_corner(flat, position="tl", frac=CORNER_FRAC, size=CROP_SIZE):
    """Crop top-left or bottom-right corner region and resize to size×size."""
    h, w = flat.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    if position == "tl":
        region = flat[0:ch, 0:cw]
    else:  # br
        region = flat[h - ch:h, w - cw:w]
    return cv2.resize(region, (size, size), interpolation=cv2.INTER_LANCZOS4)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_rank(filename):
    """Extract rank label from filename like 'cards-C2-001.jpg' → '2'."""
    stem = os.path.splitext(os.path.basename(filename))[0]  # cards-C2-001
    parts = stem.split("-")  # ['cards', 'C2', '001']
    if len(parts) < 2:
        return None
    code = parts[1]   # e.g. 'C2', 'H10', 'SA'
    rank_token = code[1:]  # drop suit letter
    return RANK_MAP.get(rank_token)


def process(src_path, only_ranks=None):
    rank = parse_rank(src_path)
    if rank is None:
        print(f"  SKIP  {src_path}  (unrecognised rank)")
        return 0
    if only_ranks and rank not in only_ranks:
        return 0

    img = cv2.imread(src_path)
    if img is None:
        print(f"  ERROR  cannot read {src_path}")
        return 0

    corners = find_card_corners(img)
    if corners is None:
        print(f"  ERROR  no card found in {src_path}")
        return 0

    flat = perspective_warp(img, corners)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out_dir = os.path.join(OUTPUT_DIR, rank)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for pos in ("tl", "br"):
        crop = crop_corner(flat, position=pos)
        # Bottom-right indicator is upside-down.
        # The pipeline (ml_pipeline.dart / test_pipeline.py) rotates BR crops
        # 180° before classifying, so training crops must also be upright.
        if pos == "br":
            crop = cv2.rotate(crop, cv2.ROTATE_180)
        out_path = os.path.join(out_dir, f"{stem}_{pos}.jpg")
        cv2.imwrite(out_path, crop)
        saved += 1

    print(f"  OK    {os.path.basename(src_path)}  rank={rank:>2}  ({saved} crops)")
    return saved


if __name__ == "__main__":
    # Optional: pass rank filter on command line, e.g. "2 3 4 5 6"
    only_ranks = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    if only_ranks:
        print(f"Filtering to ranks: {sorted(only_ranks)}")

    files = sorted(glob.glob(INPUT_GLOB))
    print(f"Found {len(files)} Kaggle card images.\n")

    total = 0
    for f in files:
        total += process(f, only_ranks)

    print(f"\nDone — {total} corner crops saved to {OUTPUT_DIR}/")
