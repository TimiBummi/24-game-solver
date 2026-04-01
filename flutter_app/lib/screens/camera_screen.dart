import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../recognition/card_recognizer.dart';
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

  // ── Slot actions ──────────────────────────────────────────────────

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
      _pipelineStatus = '';
    });
  }

  // ── Build ────────────────────────────────────────────────────────────

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
                _buildStatusBanner(),
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
