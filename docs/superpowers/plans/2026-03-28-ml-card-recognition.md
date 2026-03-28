# ML Card Recognition Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the OCR-based card recognition pipeline in the 24 Game Solver Flutter app with a two-stage ML pipeline (YOLOv8-nano detector + MobileNetV3-Small classifier), trained in Colab and running on-device via TFLite.

**Architecture:** Camera photo → YOLOv8-nano detects card bounding boxes → crop each card → MobileNetV3-Small classifies rank (A–K, 13 classes) → values fed to solver. Both models exported to TFLite FP16, running on-device.

**Tech Stack:** Python (ultralytics, tensorflow), Google Colab, Flutter (tflite_flutter, camera), Dart

**Spec:** `docs/superpowers/specs/2026-03-28-24-game-ml-pipeline-design.md`

**Repo:** `/home/tim/24`

---

## File Map

### New files (training pipeline)
| File | Responsibility |
|------|---------------|
| `notebooks/01_yolo_card_detector.ipynb` | YOLO detection training: data download, class remapping, training, evaluation, TFLite export |
| `notebooks/02_rank_classifier.ipynb` | Rank classification training: dataset prep, MobileNetV3-Small, evaluation, TFLite export |
| `notebooks/requirements.txt` | Python deps for Colab notebooks |
| `notebooks/.gitignore` | Ignore data/ and outputs/ directories |

### New files (Flutter)
| File | Responsibility |
|------|---------------|
| `flutter_app/lib/recognition/yolo_detector.dart` | TFLite YOLO inference: load model, preprocess frame, decode output, NMS |
| `flutter_app/lib/recognition/rank_classifier.dart` | TFLite classifier: load model, preprocess crop, classify rank |
| `flutter_app/lib/recognition/ml_pipeline.dart` | Orchestrate detector → crop → classifier → results |
| `flutter_app/assets/models/` | Directory for `.tflite` model files |

### Modified files
| File | Changes |
|------|---------|
| `flutter_app/pubspec.yaml` | Remove `google_mlkit_text_recognition`, add `tflite_flutter`, add model assets |
| `flutter_app/lib/recognition/card_recognizer.dart` | Rewrite to use `MlPipeline` for capture-only mode |
| `flutter_app/lib/recognition/card_parser.dart` | Remove OCR-specific methods, keep `valueToLabel` and `_rankMap` |
| `flutter_app/lib/screens/camera_screen.dart` | Remove live-stream processing, simplify to capture-only, update debug overlay |
| `flutter_app/test/card_parser_test.dart` | Remove `extractFromRecognizedText` tests (depends on ML Kit types), keep `extractCards` and `valueToLabel` tests |

### Deleted files
| File | Reason |
|------|--------|
| `flutter_app/lib/recognition/pip_counter.dart` | Replaced by ML classifier |
| `flutter_app/lib/recognition/image_preprocessor.dart` | Replaced by ML classifier |
| `flutter_app/lib/recognition/recognition_fusion.dart` | Replaced by ML classifier |
| `flutter_app/lib/recognition/card_detector.dart` | Replaced by YOLO detector |

---

## Task 1: Training Infrastructure Setup

**Files:**
- Create: `notebooks/requirements.txt`
- Create: `notebooks/.gitignore`

- [ ] **Step 1: Create notebooks directory and requirements.txt**

```bash
cd /home/tim/24
mkdir -p notebooks
```

Write `notebooks/requirements.txt`:

```text
ultralytics>=8.3.0
tensorflow>=2.16.0
roboflow>=1.1.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

- [ ] **Step 2: Create notebooks/.gitignore**

Write `notebooks/.gitignore`:

```text
data/
outputs/
*.pyc
__pycache__/
.ipynb_checkpoints/
```

- [ ] **Step 3: Commit**

```bash
cd /home/tim/24
git add notebooks/requirements.txt notebooks/.gitignore
git commit -m "feat: add training pipeline infrastructure"
```

---

## Task 2: YOLO Card Detector Training Notebook

**Files:**
- Create: `notebooks/01_yolo_card_detector.ipynb`

This notebook is designed for Google Colab. It downloads a playing card detection dataset from Roboflow, remaps all classes to single "card" class, trains YOLOv8-nano, evaluates, and exports to TFLite.

- [ ] **Step 1: Create the notebook**

Write `notebooks/01_yolo_card_detector.ipynb` with the following cells:

**Cell 1 — Setup** (code):
```python
# Install dependencies
!pip install -q ultralytics roboflow

import os
import glob
from pathlib import Path
from ultralytics import YOLO
```

**Cell 2 — Download Dataset** (code):
```python
# Download a playing card detection dataset from Roboflow
# Search Roboflow Universe for "playing cards" → pick one with bounding box annotations
# Example: https://universe.roboflow.com/ — search "playing cards object detection"
#
# Replace ROBOFLOW_API_KEY and dataset details with actual values after selecting a dataset.
# Alternatively, download manually and upload to Colab.

from roboflow import Roboflow

# TODO: Replace with your actual Roboflow API key and dataset details
# You can get a free API key at https://app.roboflow.com/
ROBOFLOW_API_KEY = "YOUR_API_KEY"  # <-- replace before running
WORKSPACE = "YOUR_WORKSPACE"       # <-- replace before running
PROJECT = "YOUR_PROJECT"           # <-- replace before running
VERSION = 1                        # <-- replace before running

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download("yolov8", location="data/cards_detection")

print(f"Dataset downloaded to: data/cards_detection")
print(f"Train images: {len(os.listdir('data/cards_detection/train/images'))}")
print(f"Val images: {len(os.listdir('data/cards_detection/valid/images'))}")
```

**Cell 3 — Remap Classes** (code):
```python
# Remap all class labels to single class 0 ("card")
# YOLO format: class_id center_x center_y width height (normalized)

def remap_labels_to_single_class(labels_dir: str) -> int:
    """Rewrite all YOLO label files to use class 0 for every object."""
    count = 0
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(label_file, "r") as f:
            lines = f.readlines()

        remapped = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Replace class ID (first field) with 0
                parts[0] = "0"
                remapped.append(" ".join(parts))

        with open(label_file, "w") as f:
            f.write("\n".join(remapped) + "\n" if remapped else "")
        count += 1
    return count

train_count = remap_labels_to_single_class("data/cards_detection/train/labels")
val_count = remap_labels_to_single_class("data/cards_detection/valid/labels")
print(f"Remapped {train_count} train + {val_count} val label files to single class 'card'")
```

**Cell 4 — Create data.yaml** (code):
```python
# Create YOLO data config
data_yaml = """
path: /content/data/cards_detection
train: train/images
val: valid/images

nc: 1
names:
  0: card
"""

with open("data/cards_detection/data.yaml", "w") as f:
    f.write(data_yaml.strip())

print("Created data.yaml")
print(open("data/cards_detection/data.yaml").read())
```

**Cell 5 — Train** (code):
```python
# Train YOLOv8-nano
model = YOLO("yolov8n.pt")

results = model.train(
    data="data/cards_detection/data.yaml",
    epochs=100,
    imgsz=640,       # train at 640 for better features
    batch=16,
    patience=15,     # early stopping
    project="outputs",
    name="yolo_card_detector",
    exist_ok=True,
)

print("Training complete!")
print(f"Best model: outputs/yolo_card_detector/weights/best.pt")
```

**Cell 6 — Evaluate** (code):
```python
# Evaluate on validation set
best_model = YOLO("outputs/yolo_card_detector/weights/best.pt")
metrics = best_model.val(data="data/cards_detection/data.yaml")

print(f"mAP@0.5:     {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")

# Target: mAP@0.5 > 0.90
assert metrics.box.map50 > 0.85, f"mAP@0.5 too low: {metrics.box.map50:.4f}"
```

**Cell 7 — Visualize Predictions** (code):
```python
import matplotlib.pyplot as plt
from PIL import Image

# Run predictions on sample validation images
val_images = glob.glob("data/cards_detection/valid/images/*.jpg")[:6]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, img_path in zip(axes.flat, val_images):
    results = best_model.predict(img_path, conf=0.5, verbose=False)
    annotated = results[0].plot()
    ax.imshow(annotated[:, :, ::-1])  # BGR to RGB
    ax.axis("off")
    ax.set_title(Path(img_path).name)

plt.suptitle("YOLOv8-nano Card Detector — Validation Predictions", fontsize=16)
plt.tight_layout()
plt.savefig("outputs/yolo_card_detector/val_predictions.png", dpi=150)
plt.show()
```

**Cell 8 — Export to TFLite** (code):
```python
# Export to TFLite at inference size 320x320
# FP16 quantization for good accuracy + small size
exported_path = best_model.export(
    format="tflite",
    imgsz=320,
    int8=False,
    half=True,  # FP16
)

import os
tflite_files = glob.glob("outputs/yolo_card_detector/weights/*float16.tflite") + \
               glob.glob("outputs/yolo_card_detector/weights/*.tflite")
for f in tflite_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"{f}: {size_mb:.1f} MB")

print("\nCopy the .tflite file to flutter_app/assets/models/card_detector.tflite")
```

- [ ] **Step 2: Commit**

```bash
cd /home/tim/24
git add notebooks/01_yolo_card_detector.ipynb
git commit -m "feat: add YOLO card detector training notebook"
```

---

## Task 3: Rank Classifier Training Notebook

**Files:**
- Create: `notebooks/02_rank_classifier.ipynb`

- [ ] **Step 1: Create the notebook**

Write `notebooks/02_rank_classifier.ipynb` with the following cells:

**Cell 1 — Setup** (code):
```python
# Install dependencies
!pip install -q tensorflow matplotlib seaborn

import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
```

**Cell 2 — Download & Organize Dataset** (code):
```python
# Download a playing card classification dataset from Kaggle
# Example: "Cards Image Dataset for Classification" — cropped card images by suit+rank
#
# Option A: Kaggle API
# !pip install kaggle
# !kaggle datasets download -d USERNAME/DATASET_NAME -p data/raw_cards
# !unzip data/raw_cards/DATASET_NAME.zip -d data/raw_cards
#
# Option B: Manual upload to Colab
# Upload a zip of card images organized by class (e.g., ace_of_hearts/, king_of_spades/)

# TODO: Replace with actual dataset download command
# After downloading, organize into 13 rank-only folders

RAW_DIR = "data/raw_cards"    # source: 52 suit+rank folders
RANK_DIR = "data/rank_cards"  # target: 13 rank-only folders

# Mapping from common dataset folder names to our 13 rank classes
RANK_MAPPING = {
    "ace": "A", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "jack": "J", "queen": "Q", "king": "K",
    # Numeric variants
    "1": "A", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "10": "10",
    "11": "J", "12": "Q", "13": "K",
}

def organize_by_rank(raw_dir: str, rank_dir: str):
    """Collapse suit+rank folders into 13 rank-only folders."""
    os.makedirs(rank_dir, exist_ok=True)

    for rank_name in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]:
        os.makedirs(os.path.join(rank_dir, rank_name), exist_ok=True)

    copied = 0
    for folder in sorted(os.listdir(raw_dir)):
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Try to extract rank from folder name
        folder_lower = folder.lower().replace("_", " ").replace("-", " ")
        rank = None
        for key, val in RANK_MAPPING.items():
            if key in folder_lower:
                rank = val
                break

        if rank is None:
            print(f"  Skipping unrecognized folder: {folder}")
            continue

        dest = os.path.join(rank_dir, rank)
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(folder_path, img_file)
                dst = os.path.join(dest, f"{folder}_{img_file}")
                shutil.copy2(src, dst)
                copied += 1

    print(f"Copied {copied} images into rank folders")
    for rank in sorted(os.listdir(rank_dir)):
        count = len(os.listdir(os.path.join(rank_dir, rank)))
        print(f"  {rank}: {count} images")

organize_by_rank(RAW_DIR, RANK_DIR)
```

**Cell 3 — Data Loading & Augmentation** (code):
```python
IMG_SIZE = 128
BATCH_SIZE = 32
RANK_DIR = "data/rank_cards"

# Class names in sorted order — this determines the index→rank mapping
CLASS_NAMES = sorted(os.listdir(RANK_DIR))
print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
assert len(CLASS_NAMES) == 13, f"Expected 13 classes, got {len(CLASS_NAMES)}"

# Load datasets with augmentation for training
train_ds = tf.keras.utils.image_dataset_from_directory(
    RANK_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    RANK_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

# Normalize to [0, 1]
normalization = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

# Data augmentation (applied only during training)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.04),          # ±15 degrees
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.1),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Prefetch for performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

print(f"Training batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
```

**Cell 4 — Build Model** (code):
```python
# MobileNetV3-Small with custom classification head
base = tf.keras.applications.MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False  # freeze backbone initially

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(13, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
print(f"\nTotal params: {model.count_params():,}")
```

**Cell 5 — Train Phase 1: Frozen Backbone** (code):
```python
# Phase 1: Train only the classification head (backbone frozen)
history_frozen = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
)

print(f"Phase 1 val accuracy: {history_frozen.history['val_accuracy'][-1]:.4f}")
```

**Cell 6 — Train Phase 2: Fine-tune** (code):
```python
# Phase 2: Unfreeze backbone and fine-tune at lower learning rate
base.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
    ],
)

print(f"Phase 2 best val accuracy: {max(history_finetune.history['val_accuracy']):.4f}")
```

**Cell 7 — Evaluate: Confusion Matrix** (code):
```python
# Confusion matrix to catch systematic errors (6/9, J/Q/K, B/D)
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
accuracy = np.mean(y_true == y_pred)
print(f"Overall accuracy: {accuracy:.4f}")
print(f"\nPer-class report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Rank Classifier Confusion Matrix (accuracy: {accuracy:.2%})")
plt.tight_layout()
plt.savefig("outputs/classifier/confusion_matrix.png", dpi=150)
plt.show()

# Flag common confusions
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        if i != j and cm[i][j] > 2:
            print(f"  WARNING: {CLASS_NAMES[i]} confused with {CLASS_NAMES[j]} ({cm[i][j]} times)")
```

**Cell 8 — Export to TFLite** (code):
```python
# Export to TFLite with FP16 quantization
os.makedirs("outputs/classifier", exist_ok=True)
model.save("outputs/classifier/rank_classifier.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

tflite_path = "outputs/classifier/rank_classifier.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"TFLite model saved: {tflite_path} ({size_mb:.1f} MB)")

# Save class name mapping for reference
import json
class_map = {i: name for i, name in enumerate(CLASS_NAMES)}
with open("outputs/classifier/class_names.json", "w") as f:
    json.dump(class_map, f, indent=2)
print(f"Class mapping: {class_map}")
print("\nCopy rank_classifier.tflite to flutter_app/assets/models/rank_classifier.tflite")
```

**Cell 9 — Verify TFLite Model** (code):
```python
# Quick sanity check: run TFLite model on a few validation images
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input:  shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
print(f"Output: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

# Test on 5 validation images
correct = 0
total = 0
for images, labels in val_ds.take(1):
    for i in range(min(5, len(images))):
        img = images[i:i+1].numpy().astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        pred_class = np.argmax(output[0])
        true_class = labels[i].numpy()
        match = "✓" if pred_class == true_class else "✗"
        print(f"  {match} True: {CLASS_NAMES[true_class]}, Predicted: {CLASS_NAMES[pred_class]} "
              f"(conf: {output[0][pred_class]:.2f})")
        if pred_class == true_class:
            correct += 1
        total += 1

print(f"\nTFLite quick check: {correct}/{total} correct")
```

- [ ] **Step 2: Commit**

```bash
cd /home/tim/24
git add notebooks/02_rank_classifier.ipynb
git commit -m "feat: add rank classifier training notebook"
```

---

## Task 4: Run Training Notebooks in Colab

This task is manual — Tim runs the notebooks in Google Colab and downloads the exported `.tflite` files.

- [ ] **Step 1: Open Notebook 1 in Colab**

Upload `notebooks/01_yolo_card_detector.ipynb` to Google Colab. Set runtime to GPU.

Find a suitable Roboflow dataset:
1. Go to https://universe.roboflow.com/
2. Search "playing cards object detection"
3. Pick a dataset with bounding box annotations and at least 2000 images
4. Get your Roboflow API key from https://app.roboflow.com/
5. Update the dataset download cell with your API key and dataset details

Run all cells. Verify:
- mAP@0.5 > 0.85 (ideally > 0.90)
- Visual predictions look correct on sample images
- TFLite file exported successfully

- [ ] **Step 2: Open Notebook 2 in Colab**

Upload `notebooks/02_rank_classifier.ipynb` to Google Colab. Set runtime to GPU.

Find a suitable Kaggle dataset:
1. Go to https://www.kaggle.com/
2. Search "playing cards image classification"
3. Pick a dataset with cropped card images organized by class
4. Update the dataset download cell

Run all cells. Verify:
- Overall accuracy > 0.90
- No systematic confusions in the confusion matrix (especially 6/9, J/Q/K)
- TFLite model works in the verification cell

- [ ] **Step 3: Download TFLite models**

```bash
cd /home/tim/24/flutter_app
mkdir -p assets/models

# Copy the exported models from Colab downloads
# (exact filenames depend on Colab output)
cp ~/Downloads/card_detector.tflite assets/models/card_detector.tflite
cp ~/Downloads/rank_classifier.tflite assets/models/rank_classifier.tflite

ls -la assets/models/
```

- [ ] **Step 4: Commit models**

```bash
cd /home/tim/24
git add flutter_app/assets/models/card_detector.tflite flutter_app/assets/models/rank_classifier.tflite
git commit -m "feat: add trained TFLite models (card detector + rank classifier)"
```

---

## Task 5: Update Flutter Dependencies

**Files:**
- Modify: `flutter_app/pubspec.yaml`

- [ ] **Step 1: Update pubspec.yaml**

In `flutter_app/pubspec.yaml`, make these changes:

Replace:
```yaml
  google_mlkit_text_recognition: ^0.15.1
```
With:
```yaml
  tflite_flutter: ^0.11.0
```

Add assets section — replace the commented-out assets block (lines 63-66) with:
```yaml
  assets:
    - assets/models/
```

- [ ] **Step 2: Run pub get**

```bash
cd /home/tim/24/flutter_app
flutter pub get
```

Expected: resolves successfully, `tflite_flutter` appears in `pubspec.lock`.

- [ ] **Step 3: Commit**

```bash
cd /home/tim/24
git add flutter_app/pubspec.yaml flutter_app/pubspec.lock
git commit -m "feat: swap google_mlkit_text_recognition for tflite_flutter"
```

---

## Task 6: YOLO Detector Dart Wrapper

**Files:**
- Create: `flutter_app/lib/recognition/yolo_detector.dart`

- [ ] **Step 1: Create yolo_detector.dart**

Write `flutter_app/lib/recognition/yolo_detector.dart`:

```dart
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// A detected card bounding box from YOLO inference.
class YoloDetection {
  final double x, y, width, height;
  final double confidence;

  const YoloDetection({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.confidence,
  });

  /// Convert to integer pixel coordinates clamped to image bounds.
  ({int x, int y, int w, int h}) toPixelRect(int imgWidth, int imgHeight) {
    final px = (x * imgWidth).round().clamp(0, imgWidth - 1);
    final py = (y * imgHeight).round().clamp(0, imgHeight - 1);
    final pw = (width * imgWidth).round().clamp(1, imgWidth - px);
    final ph = (height * imgHeight).round().clamp(1, imgHeight - py);
    return (x: px, y: py, w: pw, h: ph);
  }
}

/// Runs YOLOv8-nano TFLite inference for card detection.
class YoloDetector {
  static const int inputSize = 320;
  static const double defaultConfThreshold = 0.5;
  static const double nmsIouThreshold = 0.5;

  Interpreter? _interpreter;

  /// Load the TFLite model from assets.
  Future<void> load() async {
    _interpreter = await Interpreter.fromAsset('models/card_detector.tflite');
  }

  /// Detect cards in an image. Returns normalized bounding boxes.
  List<YoloDetection> detect(img.Image image, {double confThreshold = defaultConfThreshold}) {
    if (_interpreter == null) return [];

    // 1. Preprocess: resize to 320x320, normalize to [0, 1] float32.
    final resized = img.copyResize(image, width: inputSize, height: inputSize);
    final input = Float32List(1 * inputSize * inputSize * 3);

    int idx = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        input[idx++] = pixel.r / 255.0;
        input[idx++] = pixel.g / 255.0;
        input[idx++] = pixel.b / 255.0;
      }
    }

    final inputTensor = input.reshape([1, inputSize, inputSize, 3]);

    // 2. Run inference.
    // YOLOv8 TFLite output shape: [1, 5, N] where 5 = [x, y, w, h, conf]
    // N = number of detection candidates (depends on input size).
    final outputShape = _interpreter!.getOutputTensor(0).shape;
    final outputSize = outputShape.reduce((a, b) => a * b);
    final outputBuffer = Float32List(outputSize);
    final outputTensor = outputBuffer.reshape(outputShape);

    _interpreter!.run(inputTensor, outputTensor);

    // 3. Decode output.
    // Output is [1, 5, N] — transpose to get N detections of [x, y, w, h, conf].
    final numDetections = outputShape.last;
    final numFields = outputShape[1]; // should be 5 for single-class

    final detections = <YoloDetection>[];
    for (int i = 0; i < numDetections; i++) {
      final conf = outputBuffer[4 * numDetections + i]; // confidence at row 4
      if (conf < confThreshold) continue;

      final cx = outputBuffer[0 * numDetections + i]; // center x
      final cy = outputBuffer[1 * numDetections + i]; // center y
      final w = outputBuffer[2 * numDetections + i];  // width
      final h = outputBuffer[3 * numDetections + i];  // height

      // Convert from center-wh to top-left-wh, normalized by input size.
      final nx = (cx - w / 2) / inputSize;
      final ny = (cy - h / 2) / inputSize;
      final nw = w / inputSize;
      final nh = h / inputSize;

      detections.add(YoloDetection(
        x: nx.clamp(0, 1),
        y: ny.clamp(0, 1),
        width: nw.clamp(0, 1),
        height: nh.clamp(0, 1),
        confidence: conf,
      ));
    }

    // 4. Non-maximum suppression.
    return _nms(detections);
  }

  /// Greedy NMS: keep highest-confidence detections, suppress overlaps.
  List<YoloDetection> _nms(List<YoloDetection> detections) {
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));
    final kept = <YoloDetection>[];

    for (final det in detections) {
      bool suppressed = false;
      for (final k in kept) {
        if (_iou(det, k) > nmsIouThreshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) kept.add(det);
    }

    return kept;
  }

  double _iou(YoloDetection a, YoloDetection b) {
    final x1 = math.max(a.x, b.x);
    final y1 = math.max(a.y, b.y);
    final x2 = math.min(a.x + a.width, b.x + b.width);
    final y2 = math.min(a.y + a.height, b.y + b.height);

    if (x2 <= x1 || y2 <= y1) return 0;

    final intersection = (x2 - x1) * (y2 - y1);
    final areaA = a.width * a.height;
    final areaB = b.width * b.height;
    return intersection / (areaA + areaB - intersection);
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/tim/24/flutter_app
dart analyze lib/recognition/yolo_detector.dart
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
cd /home/tim/24
git add flutter_app/lib/recognition/yolo_detector.dart
git commit -m "feat: add YOLO TFLite detector wrapper"
```

---

## Task 7: Rank Classifier Dart Wrapper

**Files:**
- Create: `flutter_app/lib/recognition/rank_classifier.dart`

- [ ] **Step 1: Create rank_classifier.dart**

Write `flutter_app/lib/recognition/rank_classifier.dart`:

```dart
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Result of classifying a single card crop.
class RankPrediction {
  /// Card value: 1 (Ace) through 13 (King).
  final int value;

  /// Model confidence (0.0–1.0).
  final double confidence;

  const RankPrediction({required this.value, required this.confidence});
}

/// Runs MobileNetV3-Small TFLite inference for rank classification.
class RankClassifier {
  static const int inputSize = 128;

  /// Maps class index to card value.
  /// Class names are sorted alphabetically by the training dataset:
  /// ["10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q"]
  /// So index 0 = "10" (value 10), index 9 = "A" (value 1), etc.
  static const List<int> _indexToValue = [
    10, // 0: "10"
    2,  // 1: "2"
    3,  // 2: "3"
    4,  // 3: "4"
    5,  // 4: "5"
    6,  // 5: "6"
    7,  // 6: "7"
    8,  // 7: "8"
    9,  // 8: "9"
    1,  // 9: "A"
    11, // 10: "J"
    13, // 11: "K"
    12, // 12: "Q"
  ];

  Interpreter? _interpreter;

  /// Load the TFLite model from assets.
  Future<void> load() async {
    _interpreter = await Interpreter.fromAsset('models/rank_classifier.tflite');
  }

  /// Classify a cropped card image into one of 13 ranks.
  RankPrediction? classify(img.Image cardCrop) {
    if (_interpreter == null) return null;

    // 1. Preprocess: resize to 128x128, normalize to [0, 1].
    final resized = img.copyResize(cardCrop, width: inputSize, height: inputSize);
    final input = Float32List(1 * inputSize * inputSize * 3);

    int idx = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        input[idx++] = pixel.r / 255.0;
        input[idx++] = pixel.g / 255.0;
        input[idx++] = pixel.b / 255.0;
      }
    }

    final inputTensor = input.reshape([1, inputSize, inputSize, 3]);

    // 2. Run inference — output is [1, 13] softmax.
    final output = Float32List(13).reshape([1, 13]);
    _interpreter!.run(inputTensor, output);

    // 3. Find argmax.
    final probs = output[0] as List<double>;
    int bestIdx = 0;
    double bestConf = probs[0];
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > bestConf) {
        bestConf = probs[i];
        bestIdx = i;
      }
    }

    return RankPrediction(
      value: _indexToValue[bestIdx],
      confidence: bestConf,
    );
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/tim/24/flutter_app
dart analyze lib/recognition/rank_classifier.dart
```

- [ ] **Step 3: Commit**

```bash
cd /home/tim/24
git add flutter_app/lib/recognition/rank_classifier.dart
git commit -m "feat: add rank classifier TFLite wrapper"
```

---

## Task 8: ML Pipeline Orchestrator

**Files:**
- Create: `flutter_app/lib/recognition/ml_pipeline.dart`

- [ ] **Step 1: Create ml_pipeline.dart**

Write `flutter_app/lib/recognition/ml_pipeline.dart`:

```dart
import 'dart:io';

import 'package:image/image.dart' as img;

import 'rank_classifier.dart';
import 'yolo_detector.dart';

/// A recognized card with its rank, confidence, and bounding box.
class PipelineCard {
  final int value;
  final double confidence;
  final double x, y, width, height; // normalized [0,1]

  const PipelineCard({
    required this.value,
    required this.confidence,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });
}

/// Result of the full ML pipeline on a captured image.
class PipelineResult {
  final List<PipelineCard> cards;
  final int imageWidth;
  final int imageHeight;
  final int detectionsFound;

  const PipelineResult({
    required this.cards,
    required this.imageWidth,
    required this.imageHeight,
    required this.detectionsFound,
  });

  bool get hasCards => cards.isNotEmpty;
  List<int> get values => cards.map((c) => c.value).toList();
}

/// Orchestrates the two-stage ML pipeline: YOLO detection → rank classification.
class MlPipeline {
  final YoloDetector _detector = YoloDetector();
  final RankClassifier _classifier = RankClassifier();
  bool _loaded = false;

  /// Load both TFLite models. Call once before using [processCapture].
  Future<void> load() async {
    await _detector.load();
    await _classifier.load();
    _loaded = true;
  }

  bool get isLoaded => _loaded;

  /// Process a captured photo through the full pipeline.
  ///
  /// 1. Load image, apply EXIF orientation.
  /// 2. Run YOLO detector to find card bounding boxes.
  /// 3. Crop each detection from the original image.
  /// 4. Classify each crop to determine the card rank.
  Future<PipelineResult> processCapture(String imagePath) async {
    final bytes = await File(imagePath).readAsBytes();
    var decoded = img.decodeImage(bytes);
    if (decoded == null) {
      return const PipelineResult(
        cards: [],
        imageWidth: 0,
        imageHeight: 0,
        detectionsFound: 0,
      );
    }
    decoded = img.bakeOrientation(decoded);

    return processImage(decoded);
  }

  /// Process an already-decoded image through the pipeline.
  PipelineResult processImage(img.Image image) {
    if (!_loaded) {
      return PipelineResult(
        cards: const [],
        imageWidth: image.width,
        imageHeight: image.height,
        detectionsFound: 0,
      );
    }

    // 1. Detect cards.
    final detections = _detector.detect(image);

    // 2. For each detection, crop and classify.
    final cards = <PipelineCard>[];
    for (final det in detections) {
      if (cards.length >= 4) break; // max 4 cards for the game

      final rect = det.toPixelRect(image.width, image.height);
      final crop = img.copyCrop(
        image,
        x: rect.x,
        y: rect.y,
        width: rect.w,
        height: rect.h,
      );

      final prediction = _classifier.classify(crop);
      if (prediction != null) {
        cards.add(PipelineCard(
          value: prediction.value,
          confidence: prediction.confidence,
          x: det.x,
          y: det.y,
          width: det.width,
          height: det.height,
        ));
      }
    }

    return PipelineResult(
      cards: cards,
      imageWidth: image.width,
      imageHeight: image.height,
      detectionsFound: detections.length,
    );
  }

  void dispose() {
    _detector.dispose();
    _classifier.dispose();
  }
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/tim/24/flutter_app
dart analyze lib/recognition/ml_pipeline.dart
```

- [ ] **Step 3: Commit**

```bash
cd /home/tim/24
git add flutter_app/lib/recognition/ml_pipeline.dart
git commit -m "feat: add ML pipeline orchestrator (detector + classifier)"
```

---

## Task 9: Rewrite CardRecognizer for Capture-Only Mode

**Files:**
- Modify: `flutter_app/lib/recognition/card_recognizer.dart`

- [ ] **Step 1: Rewrite card_recognizer.dart**

Replace the entire contents of `flutter_app/lib/recognition/card_recognizer.dart` with:

```dart
import 'ml_pipeline.dart';

/// Orchestrates card recognition using the ML pipeline.
///
/// V1: Capture-only mode. The user takes a photo, and the pipeline
/// runs YOLO detection + rank classification on it.
class CardRecognizer {
  final MlPipeline _pipeline = MlPipeline();

  /// Load the ML models. Call once during initialization.
  Future<void> load() => _pipeline.load();

  bool get isLoaded => _pipeline.isLoaded;

  /// Process a captured photo. Returns recognized card values and bounding boxes.
  Future<PipelineResult> processCapture(String imagePath) =>
      _pipeline.processCapture(imagePath);

  void dispose() => _pipeline.dispose();
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/tim/24/flutter_app
dart analyze lib/recognition/card_recognizer.dart
```

- [ ] **Step 3: Commit**

```bash
cd /home/tim/24
git add flutter_app/lib/recognition/card_recognizer.dart
git commit -m "refactor: rewrite CardRecognizer for capture-only ML pipeline"
```

---

## Task 10: Trim card_parser.dart

**Files:**
- Modify: `flutter_app/lib/recognition/card_parser.dart`

- [ ] **Step 1: Remove OCR-specific code**

Replace the entire contents of `flutter_app/lib/recognition/card_parser.dart` with:

```dart
/// Maps card values to display labels and vice versa.
class CardParser {
  static const Map<String, int> rankMap = {
    'A': 1, '1': 1,
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'B': 11, 'J': 11, // Bube / Jack
    'D': 12, 'Q': 12, // Dame / Queen
    'K': 13,           // König / King
  };

  static const Map<int, String> valueToLabel = {
    1: 'A', 2: '2', 3: '3', 4: '4', 5: '5',
    6: '6', 7: '7', 8: '8', 9: '9', 10: '10',
    11: 'B', 12: 'D', 13: 'K',
  };
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/tim/24/flutter_app
dart analyze lib/recognition/card_parser.dart
```

- [ ] **Step 3: Commit**

```bash
cd /home/tim/24
git add flutter_app/lib/recognition/card_parser.dart
git commit -m "refactor: trim card_parser.dart to value/label maps only"
```

---

## Task 11: Update Camera Screen for Capture-Only ML Pipeline

**Files:**
- Modify: `flutter_app/lib/screens/camera_screen.dart`

This is the largest change. The camera screen needs to:
1. Remove live-stream processing (no more `processFrame`, spatial voting)
2. Load ML models at init
3. Use `PipelineResult` instead of `FusedRecognition`/`CaptureDebugInfo`
4. Simplify the debug overlay

- [ ] **Step 1: Rewrite camera_screen.dart**

Replace the entire contents of `flutter_app/lib/screens/camera_screen.dart` with:

```dart
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../recognition/card_parser.dart';
import '../recognition/card_recognizer.dart';
import '../recognition/ml_pipeline.dart';
import '../widgets/card_slots.dart';
import 'results_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  final CardRecognizer _recognizer = CardRecognizer();
  List<int?> _cards = [null, null, null, null];
  List<double> _confidences = [0, 0, 0, 0];
  bool _isCapturing = false;
  String? _error;
  bool _cameraReady = false;
  bool _modelsLoaded = false;

  /// Last pipeline result for debug overlay.
  PipelineResult? _lastResult;
  bool _showDebug = false;
  String _pipelineStatus = '';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
    _loadModels();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    _recognizer.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _controller?.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _loadModels() async {
    try {
      await _recognizer.load();
      if (mounted) setState(() => _modelsLoaded = true);
    } catch (e) {
      if (mounted) {
        setState(() => _error = 'Failed to load ML models: $e');
      }
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _error = 'No camera available on this device.');
        return;
      }

      final backCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        backCamera,
        ResolutionPreset.high,
        enableAudio: false,
      );

      await controller.initialize();
      if (!mounted) {
        controller.dispose();
        return;
      }

      _controller = controller;
      setState(() => _cameraReady = true);
    } catch (e) {
      setState(() => _error = 'Camera error: $e');
    }
  }

  // ── Capture ──────────────────────────────────────────────────────────

  Future<void> _onCapture() async {
    if (_controller == null || _isCapturing || !_modelsLoaded) return;

    setState(() {
      _isCapturing = true;
      _pipelineStatus = 'Detecting cards...';
    });

    try {
      final file = await _controller!.takePicture();

      if (mounted) setState(() => _pipelineStatus = 'Classifying...');
      final result = await _recognizer.processCapture(file.path);

      if (!mounted) return;

      setState(() {
        _lastResult = result;
        _pipelineStatus = result.hasCards
            ? 'Found ${result.cards.length} card(s) (${result.detectionsFound} detected)'
            : 'No cards found (${result.detectionsFound} detected)';

        for (int i = 0; i < result.cards.length && i < 4; i++) {
          _cards[i] = result.cards[i].value;
          _confidences[i] = result.cards[i].confidence;
        }
      });

      // Clean up temp file.
      try {
        File(file.path).deleteSync();
      } catch (_) {}
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Capture failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isCapturing = false);
    }
  }

  // ── Slot actions ──────────────────────────────────────────────────────

  void _onSlotTapped(int index) {
    showCardPicker(context, (value) {
      setState(() {
        _cards[index] = value;
        _confidences[index] = 1.0;
      });
    });
  }

  void _onSolve() {
    final filled = _cards.whereType<int>().toList();
    if (filled.length != 4) return;
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => ResultsScreen(cards: filled)),
    );
  }

  void _onClear() {
    setState(() {
      _cards = [null, null, null, null];
      _confidences = [0, 0, 0, 0];
      _lastResult = null;
      _pipelineStatus = '';
    });
  }

  // ── Build ─────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Expanded(
            child: Stack(
              fit: StackFit.expand,
              children: [
                _buildPreview(),
                _buildGuideOverlay(),
                _buildDetectionOverlay(),
                _buildStatusBanner(),
                _buildDebugToggle(),
                _buildCaptureButton(),
              ],
            ),
          ),
          CardSlots(
            cards: _cards,
            onSlotTapped: _onSlotTapped,
            onSolve: _onSolve,
            onClear: _onClear,
          ),
        ],
      ),
    );
  }

  Widget _buildPreview() {
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.videocam_off, size: 64, color: Colors.grey),
              const SizedBox(height: 16),
              Text(_error!, textAlign: TextAlign.center),
              const SizedBox(height: 16),
              const Text(
                'You can still enter cards manually by tapping the slots below.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.grey),
              ),
            ],
          ),
        ),
      );
    }

    if (!_cameraReady || _controller == null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(),
            if (!_modelsLoaded) ...[
              const SizedBox(height: 16),
              const Text('Loading ML models...', style: TextStyle(color: Colors.grey)),
            ],
          ],
        ),
      );
    }

    return ClipRect(
      child: OverflowBox(
        alignment: Alignment.center,
        child: FittedBox(
          fit: BoxFit.cover,
          child: SizedBox(
            width: _controller!.value.previewSize!.height,
            height: _controller!.value.previewSize!.width,
            child: CameraPreview(_controller!),
          ),
        ),
      ),
    );
  }

  Widget _buildGuideOverlay() {
    if (!_cameraReady) return const SizedBox.shrink();

    return LayoutBuilder(
      builder: (context, constraints) {
        final w = constraints.maxWidth;
        final h = constraints.maxHeight;

        const cardCount = 4;
        final totalPadding = w * 0.1;
        final spacing = w * 0.03;
        final cardWidth =
            (w - totalPadding - spacing * (cardCount - 1)) / cardCount;
        final cardHeight = cardWidth / 0.714;
        final startX = totalPadding / 2;
        final startY = (h - cardHeight) / 2;

        return CustomPaint(
          size: Size(w, h),
          painter: _GuideOverlayPainter(
            cardWidth: cardWidth,
            cardHeight: cardHeight,
            startX: startX,
            startY: startY,
            spacing: spacing,
            cardCount: cardCount,
            filledSlots: _cards,
          ),
        );
      },
    );
  }

  Widget _buildDetectionOverlay() {
    final result = _lastResult;
    if (result == null ||
        result.cards.isEmpty ||
        !_cameraReady ||
        _controller == null) {
      return const SizedBox.shrink();
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        return CustomPaint(
          size: Size(constraints.maxWidth, constraints.maxHeight),
          painter: _DetectionOverlayPainter(
            result: result,
            widgetSize: Size(constraints.maxWidth, constraints.maxHeight),
            cards: _cards,
            confidences: _confidences,
            showDebug: _showDebug,
          ),
        );
      },
    );
  }

  Widget _buildStatusBanner() {
    if (_pipelineStatus.isEmpty) return const SizedBox.shrink();

    return Positioned(
      top: 0,
      left: 0,
      right: 0,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
        color: Colors.black.withValues(alpha: 0.6),
        child: SafeArea(
          bottom: false,
          child: Text(
            _pipelineStatus,
            style: const TextStyle(color: Colors.white, fontSize: 13),
          ),
        ),
      ),
    );
  }

  Widget _buildDebugToggle() {
    if (!_cameraReady) return const SizedBox.shrink();

    return Positioned(
      top: 8,
      right: 8,
      child: SafeArea(
        child: GestureDetector(
          onTap: () => setState(() => _showDebug = !_showDebug),
          child: Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _showDebug
                  ? Colors.deepPurple.withValues(alpha: 0.8)
                  : Colors.black.withValues(alpha: 0.4),
            ),
            child: Icon(
              Icons.bug_report,
              size: 20,
              color: _showDebug ? Colors.white : Colors.white70,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildCaptureButton() {
    if (!_cameraReady) return const SizedBox.shrink();

    final canCapture = _modelsLoaded && !_isCapturing;

    return Positioned(
      bottom: 20,
      left: 0,
      right: 0,
      child: Center(
        child: GestureDetector(
          onTap: canCapture ? _onCapture : null,
          child: Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: canCapture
                  ? Colors.white.withValues(alpha: 0.9)
                  : Colors.grey,
              border: Border.all(color: Colors.white, width: 4),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.3),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: _isCapturing
                ? const Padding(
                    padding: EdgeInsets.all(20),
                    child: CircularProgressIndicator(strokeWidth: 3),
                  )
                : Icon(
                    _modelsLoaded ? Icons.camera_alt : Icons.hourglass_empty,
                    size: 32,
                    color: canCapture ? Colors.deepPurple : Colors.white54,
                  ),
          ),
        ),
      ),
    );
  }
}

// ── Guide Overlay ─────────────────────────────────────────────────────

class _GuideOverlayPainter extends CustomPainter {
  final double cardWidth, cardHeight, startX, startY, spacing;
  final int cardCount;
  final List<int?> filledSlots;

  _GuideOverlayPainter({
    required this.cardWidth,
    required this.cardHeight,
    required this.startX,
    required this.startY,
    required this.spacing,
    required this.cardCount,
    required this.filledSlots,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final guidePaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.4)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final filledPaint = Paint()
      ..color = Colors.green.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final cornerLength = cardWidth * 0.2;

    for (int i = 0; i < cardCount; i++) {
      final x = startX + i * (cardWidth + spacing);
      final y = startY;
      final rect = Rect.fromLTWH(x, y, cardWidth, cardHeight);
      final paint = filledSlots[i] != null ? filledPaint : guidePaint;
      _drawCornerBrackets(canvas, rect, paint, cornerLength);
    }

    final textPainter = TextPainter(
      text: TextSpan(
        text: 'Place 4 cards in view, then tap capture',
        style: TextStyle(
          color: Colors.white.withValues(alpha: 0.8),
          fontSize: 14,
          shadows: [
            Shadow(
              color: Colors.black.withValues(alpha: 0.7),
              blurRadius: 4,
            ),
          ],
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset((size.width - textPainter.width) / 2, startY - 30),
    );
  }

  void _drawCornerBrackets(Canvas canvas, Rect rect, Paint paint, double len) {
    final path = Path()
      ..moveTo(rect.left, rect.top + len)
      ..lineTo(rect.left, rect.top)
      ..lineTo(rect.left + len, rect.top)
      ..moveTo(rect.right - len, rect.top)
      ..lineTo(rect.right, rect.top)
      ..lineTo(rect.right, rect.top + len)
      ..moveTo(rect.right, rect.bottom - len)
      ..lineTo(rect.right, rect.bottom)
      ..lineTo(rect.right - len, rect.bottom)
      ..moveTo(rect.left + len, rect.bottom)
      ..lineTo(rect.left, rect.bottom)
      ..lineTo(rect.left, rect.bottom - len);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(_GuideOverlayPainter old) =>
      old.filledSlots != filledSlots;
}

// ── Detection Overlay ─────────────────────────────────────────────────

class _DetectionOverlayPainter extends CustomPainter {
  final PipelineResult result;
  final Size widgetSize;
  final List<int?> cards;
  final List<double> confidences;
  final bool showDebug;

  _DetectionOverlayPainter({
    required this.result,
    required this.widgetSize,
    required this.cards,
    required this.confidences,
    required this.showDebug,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final imgW = result.imageWidth.toDouble();
    final imgH = result.imageHeight.toDouble();
    if (imgW == 0 || imgH == 0) return;

    final imageAspect = imgW / imgH;
    final widgetAspect = widgetSize.width / widgetSize.height;

    double scale, offsetX, offsetY;
    if (imageAspect > widgetAspect) {
      scale = widgetSize.height / imgH;
      offsetX = (widgetSize.width - imgW * scale) / 2;
      offsetY = 0;
    } else {
      scale = widgetSize.width / imgW;
      offsetX = 0;
      offsetY = (widgetSize.height - imgH * scale) / 2;
    }

    Rect mapNormalized(double nx, double ny, double nw, double nh) {
      return Rect.fromLTWH(
        nx * imgW * scale + offsetX,
        ny * imgH * scale + offsetY,
        nw * imgW * scale,
        nh * imgH * scale,
      );
    }

    final bboxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    for (int i = 0; i < result.cards.length && i < 4; i++) {
      final card = result.cards[i];
      final recognized = i < cards.length && cards[i] != null;

      bboxPaint.color = recognized
          ? Colors.green.withValues(alpha: 0.8)
          : Colors.orange.withValues(alpha: 0.8);

      final rect = mapNormalized(card.x, card.y, card.width, card.height);
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(4)),
        bboxPaint,
      );

      // Label
      if (recognized && i < confidences.length) {
        final conf = (confidences[i] * 100).round();
        final label = CardParser.valueToLabel[cards[i]] ?? '?';
        final textPainter = TextPainter(
          text: TextSpan(
            text: '$label ($conf%)',
            style: TextStyle(
              color: Colors.white,
              fontSize: 12,
              backgroundColor: bboxPaint.color,
            ),
          ),
          textDirection: TextDirection.ltr,
        );
        textPainter.layout();
        textPainter.paint(canvas, Offset(rect.left, rect.top - 16));
      }

      // Debug info
      if (showDebug) {
        final debugText = 'YOLO conf: ${(card.confidence * 100).round()}%\n'
            'Rank conf: ${(confidences[i] * 100).round()}%';
        final tp = TextPainter(
          text: TextSpan(
            text: debugText,
            style: TextStyle(
              color: Colors.white,
              fontSize: 10,
              backgroundColor: Colors.black.withValues(alpha: 0.6),
            ),
          ),
          textDirection: TextDirection.ltr,
        );
        tp.layout(maxWidth: rect.width + 40);
        tp.paint(canvas, Offset(rect.left, rect.bottom + 4));
      }
    }
  }

  @override
  bool shouldRepaint(_DetectionOverlayPainter old) => true;
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/tim/24/flutter_app
dart analyze lib/screens/camera_screen.dart
```

- [ ] **Step 3: Commit**

```bash
cd /home/tim/24
git add flutter_app/lib/screens/camera_screen.dart
git commit -m "refactor: simplify camera screen for capture-only ML pipeline"
```

---

## Task 12: Delete Old Recognition Files

**Files:**
- Delete: `flutter_app/lib/recognition/pip_counter.dart`
- Delete: `flutter_app/lib/recognition/image_preprocessor.dart`
- Delete: `flutter_app/lib/recognition/recognition_fusion.dart`
- Delete: `flutter_app/lib/recognition/card_detector.dart`

- [ ] **Step 1: Delete old files**

```bash
cd /home/tim/24/flutter_app
rm lib/recognition/pip_counter.dart
rm lib/recognition/image_preprocessor.dart
rm lib/recognition/recognition_fusion.dart
rm lib/recognition/card_detector.dart
```

- [ ] **Step 2: Verify the app still compiles**

```bash
cd /home/tim/24/flutter_app
dart analyze lib/
```

Expected: no errors. The deleted files should not be imported anywhere after the rewrites in Tasks 9–11.

- [ ] **Step 3: Commit**

```bash
cd /home/tim/24
git add -A flutter_app/lib/recognition/
git commit -m "refactor: remove old OCR recognition pipeline files"
```

---

## Task 13: Update Tests

**Files:**
- Modify: `flutter_app/test/card_parser_test.dart`

- [ ] **Step 1: Rewrite card_parser_test.dart**

The old tests depend on `google_mlkit_text_recognition` types (`RecognizedText`, `TextElement`, etc.) which are no longer available. Keep only the `valueToLabel` tests.

Replace the entire contents of `flutter_app/test/card_parser_test.dart` with:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:twenty_four_solver/recognition/card_parser.dart';

void main() {
  group('CardParser.valueToLabel', () {
    test('maps all 13 ranks correctly', () {
      expect(CardParser.valueToLabel[1], 'A');
      expect(CardParser.valueToLabel[2], '2');
      expect(CardParser.valueToLabel[3], '3');
      expect(CardParser.valueToLabel[4], '4');
      expect(CardParser.valueToLabel[5], '5');
      expect(CardParser.valueToLabel[6], '6');
      expect(CardParser.valueToLabel[7], '7');
      expect(CardParser.valueToLabel[8], '8');
      expect(CardParser.valueToLabel[9], '9');
      expect(CardParser.valueToLabel[10], '10');
      expect(CardParser.valueToLabel[11], 'B');
      expect(CardParser.valueToLabel[12], 'D');
      expect(CardParser.valueToLabel[13], 'K');
    });

    test('uses German labels for face cards', () {
      expect(CardParser.valueToLabel[11], 'B'); // Bube, not J
      expect(CardParser.valueToLabel[12], 'D'); // Dame, not Q
      expect(CardParser.valueToLabel[13], 'K'); // König
    });
  });

  group('CardParser.rankMap', () {
    test('maps German face card labels', () {
      expect(CardParser.rankMap['B'], 11);
      expect(CardParser.rankMap['D'], 12);
      expect(CardParser.rankMap['K'], 13);
    });

    test('maps international face card labels', () {
      expect(CardParser.rankMap['J'], 11);
      expect(CardParser.rankMap['Q'], 12);
    });

    test('maps number cards', () {
      expect(CardParser.rankMap['A'], 1);
      expect(CardParser.rankMap['1'], 1);
      expect(CardParser.rankMap['10'], 10);
      expect(CardParser.rankMap['5'], 5);
    });
  });
}
```

- [ ] **Step 2: Run tests**

```bash
cd /home/tim/24/flutter_app
flutter test test/card_parser_test.dart test/solver_test.dart
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
cd /home/tim/24
git add flutter_app/test/card_parser_test.dart
git commit -m "test: update card_parser tests for trimmed API"
```

---

## Task 14: End-to-End Verification

- [ ] **Step 1: Full static analysis**

```bash
cd /home/tim/24/flutter_app
dart analyze lib/ test/
```

Expected: no errors or warnings.

- [ ] **Step 2: Run all tests**

```bash
cd /home/tim/24/flutter_app
flutter test
```

Expected: all tests pass (solver_test.dart + card_parser_test.dart).

- [ ] **Step 3: Build the app**

```bash
cd /home/tim/24/flutter_app
flutter build apk --debug
```

Expected: builds successfully. The `.tflite` model files are bundled as assets.

- [ ] **Step 4: Test on device**

Install on a real Android device and test:
1. App launches, camera preview shows
2. "Loading ML models..." appears briefly, then capture button activates
3. Place 4 cards, tap capture
4. Cards recognized and values populated in slots
5. Tap Solve → correct solutions shown
6. Manual card entry still works via slot tapping
7. Debug toggle shows YOLO + classifier confidence

- [ ] **Step 5: Final commit**

If any fixes were needed, commit them:

```bash
cd /home/tim/24
git add -A
git commit -m "fix: address issues found during end-to-end testing"
```
