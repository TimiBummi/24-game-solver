# 24 Game Solver — ML Card Recognition Pipeline

## Context

The 24 Game Solver app currently uses a handcrafted recognition pipeline (Google ML Kit OCR + contour-based detection + pip counting + multi-signal fusion, ~1400 lines of Dart). This works but is fragile, hard to improve, and doesn't showcase ML skills.

Tim is looking for ML Engineer roles and needs portfolio projects that demonstrate model training, pipeline design, and deployment. Replacing the OCR pipeline with a trained two-stage ML pipeline serves both as a better product and a stronger portfolio piece.

## Architecture

Two-stage pipeline, both models running on-device via TFLite:

```
Camera photo (JPEG)
  → [YOLOv8-nano] → bounding boxes (up to 4 cards)
  → [crop each card at full resolution]
  → [MobileNetV3-Small classifier] → rank per card (A–K, 13 classes)
  → card values fed to solver
```

### Stage 1: Card Detector (YOLOv8-nano)
- **Task**: Localize playing cards in the image
- **Classes**: 1 ("card")
- **Input**: 320×320 RGB, normalized [0,1]
- **Output**: bounding boxes + confidence scores
- **Model size**: ~6 MB (FP16 TFLite)
- **Inference**: 20-40ms on mid-range phone (GPU delegate)

### Stage 2: Rank Classifier (MobileNetV3-Small)
- **Task**: Classify a cropped card image into one of 13 ranks
- **Classes**: 13 (A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K)
- **Input**: 128×128 RGB crop, normalized [0,1]
- **Output**: 13-class softmax
- **Model size**: ~2.5 MB (FP16 TFLite)
- **Inference**: 5-10ms per crop

### Why two stages over one
- Portfolio value: demonstrates multi-model pipeline design
- Flexibility: improve detection and classification independently
- On-device friendly: two small specialized models > one large general model
- Learning: experience with both object detection AND image classification

## Training Pipeline (Google Colab)

### Data Strategy

**Detection data**:
- Source: Roboflow playing card detection datasets (~3000-5000 images)
- Remap all class labels to single class 0 ("card")
- YOLO format: `class_id center_x center_y width height` (normalized), one `.txt` per image
- Split: ~80% train, ~20% val

**Classification data**:
- Source: Kaggle card classification datasets, collapsed from 52 suit+rank classes to 13 rank-only classes
- Target: 300-500 images per class (3900-6500 total)
- Format: folder-per-class (`A/`, `2/`, ..., `K/`)
- Augment with crops from YOLO detector on full-table images

**Custom data**: 50-100 photos of Tim's German deck (B/D/K face cards) in varied conditions:
- Backgrounds: wood, felt, carpet, white, dark
- Lighting: daylight, warm lamp, fluorescent, dim
- Angles: top-down, 30° tilt, slight perspective

### Notebook 1: YOLOv8 Card Detector
**File**: `notebooks/01_yolo_card_detector.ipynb`

1. Install `ultralytics>=8.3.0`
2. Download dataset from Roboflow (Python API)
3. Remap classes to single "card" class
4. Create `data.yaml`
5. Train: `YOLO('yolov8n.pt').train(data='data.yaml', epochs=100, imgsz=640, batch=16)`
6. Evaluate: mAP@0.5, precision/recall curves
7. Export: `model.export(format='tflite', imgsz=320, int8=False)` (FP16)
8. Visualize sample predictions

### Notebook 2: Rank Classifier
**File**: `notebooks/02_rank_classifier.ipynb`

1. Install `tensorflow>=2.16.0`
2. Organize dataset into 13 class folders
3. Augmentation: rotation (±15°), brightness/contrast jitter, color jitter, perspective transform
4. Architecture:
   ```python
   base = tf.keras.applications.MobileNetV3Small(
       input_shape=(128, 128, 3), include_top=False, weights='imagenet'
   )
   model = tf.keras.Sequential([
       base,
       tf.keras.layers.GlobalAveragePooling2D(),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(13, activation='softmax')
   ])
   ```
5. Training: freeze backbone 5 epochs, unfreeze + fine-tune 20-30 epochs at lower LR
6. Evaluate: per-class accuracy, confusion matrix (watch 6/9 confusion, B/J/Q/D)
7. Export to TFLite with FP16 quantization

### Notebook 3 (optional): End-to-End Evaluation
**File**: `notebooks/03_e2e_evaluation.ipynb`

Run full two-stage pipeline on held-out full-table images. Measure end-to-end accuracy (correct 4-card identification rate).

## Flutter Integration

### Dependencies

**Remove**: `google_mlkit_text_recognition: ^0.15.1`
**Add**: `tflite_flutter: ^0.11.0`
**Keep**: `camera: ^0.11.4`, `image: ^4.3.0`

**Assets**:
```yaml
flutter:
  assets:
    - assets/models/card_detector.tflite
    - assets/models/rank_classifier.tflite
```

### File Changes

**New files**:
- `lib/recognition/yolo_detector.dart` — TFLite YOLO inference, preprocessing, NMS, output decoding
- `lib/recognition/rank_classifier.dart` — TFLite classifier inference, index-to-value mapping
- `lib/recognition/ml_pipeline.dart` — orchestrates detector → crop → classifier

**Rewrite**:
- `lib/recognition/card_recognizer.dart` — simplified to use `ml_pipeline.dart` for capture mode only (V1)

**Delete**:
- `lib/recognition/pip_counter.dart`
- `lib/recognition/image_preprocessor.dart`
- `lib/recognition/recognition_fusion.dart`

**Keep**:
- `lib/recognition/card_parser.dart` — `valueToLabel` map still used by UI

### Recognition UX

**V1: Capture mode only**
- User points camera, taps capture button
- App takes high-res photo
- Runs YOLO → crop → classify (~70-130ms total)
- Shows results on results screen

**V2 (stretch goal): Add live scanning**
- Continuous frame processing at 3-5 FPS
- Spatial voting across frames for stability
- Cards auto-populate slots
- Port existing voting logic from current `camera_screen.dart`

### Inference Pipeline Detail

```
Camera photo (JPEG)
  |
  [Load + EXIF orientation]
  [Resize to 320×320, normalize to float32 [0,1]]
  |
  [YOLOv8n TFLite] → output shape [1, 5, N]
  [Decode: transpose, extract boxes + confidence]
  [NMS, filter confidence > 0.5]
  → List<BoundingBox> (up to 4 cards)
  |
  For each box:
    [Crop from original resolution]
    [Resize to 128×128, normalize]
    [MobileNetV3-Small TFLite] → [1, 13] softmax
    [argmax → rank, confidence = max prob]
  |
  → List<(rank, confidence, boundingBox)>
```

### Performance Budget (capture mode)

| Step | Time |
|------|------|
| YOLO inference (320×320) | 20-40ms |
| 4× classifier (128×128) | 20-40ms |
| Image loading + preprocessing | 30-50ms |
| **Total** | **70-130ms** |

## Repository Structure

```
/home/tim/24/
  notebooks/
    01_yolo_card_detector.ipynb
    02_rank_classifier.ipynb
    03_e2e_evaluation.ipynb        # optional
    requirements.txt
    data/                          # .gitignore'd
    outputs/                       # training checkpoints
  flutter_app/
    assets/models/
      card_detector.tflite         # ~6 MB
      rank_classifier.tflite       # ~2.5 MB
    lib/
      recognition/
        yolo_detector.dart         # NEW
        rank_classifier.dart       # NEW
        ml_pipeline.dart           # NEW
        card_recognizer.dart       # REWRITTEN
        card_parser.dart           # KEPT (trimmed)
      solver/                      # UNCHANGED
      screens/
        camera_screen.dart         # MINOR CHANGES
        results_screen.dart        # UNCHANGED
```

Total model payload shipped with app: ~8.5 MB.

## Build Phases

### Phase 1: YOLO Card Detector Training (Colab)
- Set up notebook, download data, remap classes
- Train YOLOv8n, evaluate, export to TFLite
- Validate detection quality on sample images

### Phase 2: Rank Classifier Training (Colab)
- Set up notebook, prepare 13-class dataset
- Include custom German deck photos
- Train MobileNetV3-Small, evaluate confusion matrix
- Export to TFLite

### Phase 3: Flutter Detector Integration
- Add `tflite_flutter` to pubspec
- Create `yolo_detector.dart`
- Wire into camera screen — show bounding boxes on preview
- Keep existing OCR as temporary fallback

### Phase 4: Flutter Full Pipeline
- Create `rank_classifier.dart` and `ml_pipeline.dart`
- Rewrite `card_recognizer.dart` for capture-only mode
- Remove OCR dependency and deleted files
- End-to-end test: photo → card values → solver → solutions

### Phase 5: Polish
- Tune confidence thresholds
- Test varied decks and conditions
- Update tests
- Clean up camera screen UI for capture-only flow

### Phase 6 (stretch): Live Scanning
- Add frame processing loop
- Port spatial voting logic
- Performance tune for 3-5 FPS
