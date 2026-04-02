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

    // 2. Classify every detected corner region.
    // Each physical card produces two detections (upper-left and lower-right
    // rank indicators). BR crops are rotated 180° so the rank digit is always
    // upright before classification. Deduplicate by rank keeping highest
    // confidence read per rank value.
    final brIndices = _findBrIndices(detections);
    final allPredictions = <PipelineCard>[];
    for (int i = 0; i < detections.length; i++) {
      final det = detections[i];
      final rect = det.toPixelRect(image.width, image.height);
      var crop = img.copyCrop(
        image,
        x: rect.x,
        y: rect.y,
        width: rect.w,
        height: rect.h,
      );

      // Rotate bottom-right corner crops to upright orientation.
      if (brIndices.contains(i)) {
        crop = img.copyRotate(crop, angle: 180);
      }

      final prediction = _classifier.classify(crop);
      if (prediction != null) {
        allPredictions.add(PipelineCard(
          value: prediction.value,
          confidence: prediction.confidence,
          x: det.x,
          y: det.y,
          width: det.width,
          height: det.height,
        ));
      }
    }

    // Deduplicate: keep the highest-confidence detection per rank value.
    final best = <int, PipelineCard>{};
    for (final card in allPredictions) {
      final existing = best[card.value];
      if (existing == null || card.confidence > existing.confidence) {
        best[card.value] = card;
      }
    }

    // Sort by confidence descending, take up to 4.
    final cards = best.values.toList()
      ..sort((a, b) => b.confidence.compareTo(a.confidence));
    if (cards.length > 4) cards.length = 4;

    return PipelineResult(
      cards: cards,
      imageWidth: image.width,
      imageHeight: image.height,
      detectionsFound: detections.length,
    );
  }

  /// Pair detections by nearest neighbour; return indices of BR corners.
  ///
  /// Within each pair the detection with the higher cx+cy is the bottom-right
  /// rank indicator and must be rotated 180° before classification.
  Set<int> _findBrIndices(List<YoloDetection> detections) {
    final n = detections.length;
    if (n == 0) return {};

    double cx(YoloDetection d) => d.x + d.width / 2;
    double cy(YoloDetection d) => d.y + d.height / 2;
    double diag(int i) => cx(detections[i]) + cy(detections[i]);

    final used = List<bool>.filled(n, false);
    final brSet = <int>{};

    // Process in order of increasing cx+cy (top-left detections first).
    final order = List<int>.generate(n, (i) => i)
      ..sort((a, b) => diag(a).compareTo(diag(b)));

    for (final i in order) {
      if (used[i]) continue;
      double bestDist = double.infinity;
      int bestJ = -1;
      for (int j = 0; j < n; j++) {
        if (j == i || used[j]) continue;
        final dx = cx(detections[i]) - cx(detections[j]);
        final dy = cy(detections[i]) - cy(detections[j]);
        final d2 = dx * dx + dy * dy;
        if (d2 < bestDist) {
          bestDist = d2;
          bestJ = j;
        }
      }
      used[i] = true;
      if (bestJ >= 0) {
        used[bestJ] = true;
        // The one with higher cx+cy is the bottom-right corner.
        brSet.add(diag(bestJ) >= diag(i) ? bestJ : i);
      }
      // Lone detection → assume TL, no rotation.
    }
    return brSet;
  }

  void dispose() {
    _detector.dispose();
    _classifier.dispose();
  }
}
