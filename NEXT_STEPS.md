# Next Steps — Card Recognition Pipeline

## Current state (as of 2026-04-01)

The two-stage ML pipeline (YOLO detector → MobileNetV3 rank classifier) is working
end-to-end. Two major bugs were fixed in this session:

1. **Rank classifier output buffer** (`rank_classifier.dart`):
   `List.filled()` replaced with `Float32List` — tflite_flutter only writes into
   typed buffers.

2. **YOLO bounding box coordinates** (`yolo_detector.dart`):
   The model outputs already-normalized [0,1] coords. The `/inputSize` division
   was removed; bboxes are now correct.

3. **Corner deduplication** (`ml_pipeline.dart`):
   The detector finds both rank-indicator corners of each card (upper-left +
   lower-right). The pipeline now classifies all corners and deduplicates by rank,
   keeping the highest-confidence read per unique rank value.

**Accuracy on 50 real photos: 12/50 exact match (24%).**
Per-class accuracy: 57% overall. Worst classes: 9 (14%), J (20%), Q (38%), K (40%).

---

## Task 1 — Fine-tune the rank classifier on real data  ← START HERE

Real corner crops have been extracted from the 50 test photos and saved to
`data/real_crops/<label>/` (~180 crops, 7–20 per class).

> ⚠️ `data/` is in `.gitignore` — re-generate crops on your laptop with:
> ```bash
> uv run --python 3.11 --with "numpy<2" --with pillow --with tflite-runtime \
>   python3 test_pipeline.py --batch <csv> --save-crops data/real_crops
> ```
> See "Running the test script" below for how to build the CSV.

### Steps

1. **Open the training notebook** (whichever trained `rank_classifier.tflite`).

2. **Add real data to the pipeline:**
   ```python
   real_ds = tf.keras.utils.image_dataset_from_directory(
       'data/real_crops',
       image_size=(128, 128),
       batch_size=32,
       label_mode='int',
   )
   # Mix with existing synthetic data (e.g. 50/50 or weight real samples higher)
   ```

3. **Freeze backbone, retrain head only** (prevents catastrophic forgetting):
   ```python
   base_model.trainable = False          # freeze MobileNetV3 backbone
   model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   model.fit(mixed_ds, epochs=10, validation_data=val_ds)
   ```

4. **Then optionally unfreeze and fine-tune end-to-end** at a lower LR (1e-5).

5. **Export to TFLite** and drop into `flutter_app/assets/models/rank_classifier.tflite`.

6. **Re-run the test script** — target: >80% exact match on the 50 photos.

---

## Task 2 — Collect more real photos (if accuracy is still low after Task 1)

~180 crops from 50 photos is a thin dataset. To get to 90%+:

- Photograph each individual card rank (one card per photo) from multiple angles/
  lighting conditions → run `test_pipeline.py --save-crops` to extract and label.
- Aim for 50+ real crops per class (650 total).

---

## Task 3 — Verify the Flutter app end-to-end

Once the new `.tflite` is in place:

1. Build a release APK:
   ```bash
   cd flutter_app
   flutter build apk --release
   ```
   ProGuard rule already added: `-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options`

2. Test live on device — point camera at 4 cards, tap capture, verify recognised values.

3. The manual-slot picker (`_onSlotTapped`) is the fallback if a card is mis-recognised.

---

## Running the test script

```bash
# Build the CSV from ground_truth.json (run once):
python3 -c "
import json, csv
gt = json.load(open('data/real_photos/ground_truth.json'))
rows = [[f'data/real_photos/{k}.jpg', ' '.join(v[\"ranks\"])]
        for k, v in gt.items() if not k.startswith('_')]
csv.writer(open('data/real_photos/test.csv','w')).writerows(rows)
print(len(rows), 'rows written')
"

# Run all 50 images:
uv run --python 3.11 --with "numpy<2" --with pillow --with tflite-runtime \
  python3 test_pipeline.py --batch data/real_photos/test.csv

# Run + save crops for fine-tuning:
uv run --python 3.11 --with "numpy<2" --with pillow --with tflite-runtime \
  python3 test_pipeline.py --batch data/real_photos/test.csv \
  --save-crops data/real_crops

# Single image:
uv run --python 3.11 --with "numpy<2" --with pillow --with tflite-runtime \
  python3 test_pipeline.py data/real_photos/1000013751.jpg "4 7 8 K"
```

---

## Key files changed in this session

| File | What changed |
|------|-------------|
| `flutter_app/lib/recognition/rank_classifier.dart` | Output buffer: `List.filled` → `Float32List` |
| `flutter_app/lib/recognition/yolo_detector.dart` | Removed `/inputSize` from bbox coords |
| `flutter_app/lib/recognition/ml_pipeline.dart` | Classify all corners, deduplicate by rank |
| `flutter_app/android/app/proguard-rules.pro` | Added tflite_gpu dontwarn rule |
| `test_pipeline.py` | New end-to-end test script |
| `test_classifier.py` | Pre-existing single-model diagnostic script |
| `flutter_app/assets/models/*.tflite` | Models now committed |
