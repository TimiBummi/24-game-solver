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
  static const double defaultConfThreshold = 0.25;
  static const double nmsIouThreshold = 0.5;

  Interpreter? _interpreter;

  /// Load the TFLite model from assets.
  Future<void> load() async {
    _interpreter = await Interpreter.fromAsset('assets/models/card_detector.tflite');
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
    final numDetections = outputShape.last;
    // Use a nested list as output — tflite_flutter writes into this structure,
    // not into the flat buffer that reshape() is derived from.
    final outputTensor = List.generate(1, (_) =>
        List.generate(5, (_) => List.filled(numDetections, 0.0)));

    _interpreter!.run(inputTensor, outputTensor);

    // 3. Decode output: outputTensor[0][row][i]
    final rows = outputTensor[0] as List;
    final confRow = rows[4] as List;
    final maxConf = confRow.fold<double>(0, (m, v) => (v as double) > m ? v : m);
    // ignore: avoid_print
    print('[YOLO] output shape: $outputShape, numDetections: $numDetections, max conf: ${maxConf.toStringAsFixed(3)}');

    final detections = <YoloDetection>[];
    for (int i = 0; i < numDetections; i++) {
      final conf = (confRow[i] as double);
      if (conf < confThreshold) continue;

      final cx = (rows[0] as List)[i] as double; // center x
      final cy = (rows[1] as List)[i] as double; // center y
      final w  = (rows[2] as List)[i] as double; // width
      final h  = (rows[3] as List)[i] as double; // height

      // Model outputs normalized [0,1] center-format — convert to top-left only.
      final nx = cx - w / 2;
      final ny = cy - h / 2;
      final nw = w;
      final nh = h;

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
