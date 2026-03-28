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
