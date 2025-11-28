import 'package:breathing_app/utils/logger.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:typed_data';
import 'package:breathing_app/enums/enums.dart';
import 'package:logging/logging.dart';

class BreathClassifier {
  static const MethodChannel _channel = MethodChannel('breathing_classifier');
  bool _isInitialized = false;
  int _initAttempts = 0;
  static const int maxInitAttempts = 3;
  final Logger _logger = LoggerService.getLogger('BreathClassifier');

  static const int sampleRate = 44100;
  ModelType _currentModel = ModelType.standard;

  Future<void> initialize({ModelType model = ModelType.standard}) async {
    if (_isInitialized && _currentModel == model) return;

    try {
      _currentModel = model;
      final String nativeModelType = _mapModelTypeToNative(model);
      _isInitialized =
          await _channel.invokeMethod<bool>('initializeModel', {
            'modelType': nativeModelType,
          }) ??
          false;

      if (_isInitialized) {
        _logger.info(
          'Breath classifier successfully initialized with model: $model',
        );
      } else {
        _logger.warning(
          'Classifier was not properly initialized on the native side.',
        );

        if (_initAttempts < maxInitAttempts) {
          _initAttempts++;
          _logger.info(
            'Attempting reinitialization $_initAttempts of $maxInitAttempts...',
          );

          await Future.delayed(const Duration(seconds: 2));
          await initialize(model: model);
        } else {
          _logger.severe(
            'Maximum number of classifier initialization attempts exceeded.',
          );
          throw Exception(
            'Failed to initialize breath classifier after $maxInitAttempts attempts',
          );
        }
      }
    } catch (e) {
      _logger.severe('Error during classifier initialization: $e');
      _isInitialized = false;
      throw Exception('Failed to initialize breath classifier');
    }
  }

  String _mapModelTypeToNative(ModelType model) {
    switch (model) {
      case ModelType.exhaleOnly:
        return 'exhale_only';
      case ModelType.standard:
        return 'standard';
    }
  }

  Future<void> switchModel(ModelType modelType) async {
    if (_currentModel == modelType) return;

    _logger.info('Switching to model: $modelType');
    _isInitialized = false;
    _initAttempts = 0;

    await initialize(model: modelType);
  }

  Future<bool> checkInitialized() async {
    try {
      _isInitialized =
          await _channel.invokeMethod<bool>('isInitialized') ?? false;
      return _isInitialized;
    } catch (e) {
      _logger.warning('Error while checking initialization status: $e');
      _isInitialized = false;
      return false;
    }
  }

  Future<BreathPhase> classify(List<int> audioData) async {
    if (!_isInitialized) {
      _isInitialized = await checkInitialized();

      if (!_isInitialized) {
        _logger.warning(
          'Classifier is not initialized. Attempting reinitialization...',
        );
        try {
          await initialize();
        } catch (e) {
          _logger.severe('Reinitialization failed: $e');
          return BreathPhase.silence;
        }
      }
    }

    try {
      // TODO: check if this is neccessary
      // possibly we could just pass the audioData as is
      final Int16List audioInt16 = Int16List(audioData.length);
      for (int i = 0; i < audioData.length; i++) {
        audioInt16[i] = audioData[i];
      }

      final ByteData byteData = audioInt16.buffer.asByteData();
      final Uint8List byteList = byteData.buffer.asUint8List();

      final int classIndex =
          await _channel.invokeMethod<int>('classifyAudio', {
            'audioData': byteList,
          }) ??
          2;

      _logger.fine(
        'Native classification result: $classIndex (${_indexToBreathPhase(classIndex)})',
      );
      return _indexToBreathPhase(classIndex);
    } catch (e) {
      _logger.warning('Error during native classification: $e');

      if (e.toString().contains('INIT_FAILED')) {
        _isInitialized = false;
      }

      return BreathPhase.silence;
    }
  }

  static BreathPhase _indexToBreathPhase(int index) {
    switch (index) {
      case 0:
        return BreathPhase.exhale;
      case 1:
        return BreathPhase.inhale;
      case 2:
      default:
        return BreathPhase.silence;
    }
  }

  static Color getColorForPhase(BreathPhase phase) {
    switch (phase) {
      case BreathPhase.inhale:
        return Colors.red;
      case BreathPhase.exhale:
        return Colors.green;
      case BreathPhase.silence:
        return Colors.blue;
    }
  }

  static String getLabelForPhase(BreathPhase phase) {
    switch (phase) {
      case BreathPhase.inhale:
        return 'Inhale';
      case BreathPhase.exhale:
        return 'Exhale';
      case BreathPhase.silence:
        return 'Silence';
    }
  }

  void dispose() {
    _isInitialized = false;
  }
}
