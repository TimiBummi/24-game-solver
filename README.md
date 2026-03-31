# 24 Game Solver

A 24 Game solver with **automatic card recognition** — point your phone camera at up to 4 playing cards and get all solutions instantly.

## What is the 24 Game?

Given 4 playing cards, use addition, subtraction, multiplication, and division to make the number 24. For example: `1, 6, 11, 13` → `(11 - 1 - 6) * 13 = ... ` no wait — `(13 - 11) * (6 + 1)` ... the solver will figure it out.

## Architecture

```
Camera photo
  → [YOLOv8-nano]           detect & crop each card (~6 MB, FP16 TFLite)
  → [MobileNetV3-Small]     classify rank A–K per crop (~2.5 MB, FP16 TFLite)
  → card values → solver    enumerate all valid expressions → display solutions
```

Both models run **on-device** via TFLite — no internet required.

## Project Structure

```
├── main.py              # Python CLI solver (standalone, no ML)
├── game.py              # Core solver logic
├── formula.py           # Expression tree representation
├── Operations/          # +, -, *, / operator classes
├── notebooks/
│   ├── 01_yolo_card_detector.ipynb    # Train YOLOv8-nano card detector
│   ├── 02_german_card_synth.ipynb     # Synthetic data generator for German cards (B/D indices)
│   └── 03_rank_classifier.ipynb       # Train MobileNetV3-Small rank classifier
└── flutter_app/         # Flutter mobile app with TFLite inference
```

## ML Pipeline

The recognition pipeline replaces brittle OCR-based detection with two trained models:

| Stage | Model | Task | Input | Latency |
|-------|-------|------|-------|---------|
| 1 | YOLOv8-nano | Detect cards in frame | 320×320 | 20–40ms |
| 2 | MobileNetV3-Small | Classify rank (A–K) | 128×128 crop | 5–10ms per card |

Training runs on **Google Colab** (free GPU). See the notebooks for data sources, training configs, and export steps.

**German card support:** The synthetic data generator (`02_german_card_synth.ipynb`) creates training images with German corner indices (B for Bube/Jack, D for Dame/Queen) to handle German playing card decks.

## Python CLI

No Flutter needed — run the solver directly:

```bash
python main.py
```

Edit the `rounds` list in `main.py` to try different card combinations.

## Flutter App

The mobile app lives in `flutter_app/`. It uses the camera to capture cards, runs both TFLite models, and displays all valid solutions.

```bash
cd flutter_app
flutter run
```

Requires Flutter 3.x and Android/iOS device or emulator.

## Tech Stack

- **Python** — solver logic, ML training notebooks
- **YOLOv8** (Ultralytics) — card detection
- **MobileNetV3** (TensorFlow/Keras) — rank classification
- **TFLite** — on-device inference
- **Flutter** — mobile app
