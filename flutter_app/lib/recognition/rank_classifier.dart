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
    _interpreter = await Interpreter.fromAsset('assets/models/rank_classifier.tflite');
    final inT = _interpreter!.getInputTensor(0);
    final outT = _interpreter!.getOutputTensor(0);
    // ignore: avoid_print
    print('[RANK] input: shape=${inT.shape}, type=${inT.type}');
    // ignore: avoid_print
    print('[RANK] output: shape=${outT.shape}, type=${outT.type}');
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
    // Float32List is required: tflite_flutter writes into typed data buffers only.
    // Plain List<double> (List.filled) is not written back by the runtime.
    final outputBuffer = Float32List(13);
    _interpreter!.run(inputTensor, [outputBuffer]);

    // 3. Find argmax.
    final probs = outputBuffer;
    // ignore: avoid_print
    print('[RANK] probs: ${probs.map((v) => v.toStringAsFixed(3)).toList()}');
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
