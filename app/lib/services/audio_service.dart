import 'dart:async';
import 'package:breathing_app/enums/enums.dart';
import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import '../models/breath_classifier.dart';
import './audio_recording_service.dart';
import './breath_tracking_service.dart';
import 'package:logging/logging.dart';
import 'package:breathing_app/utils/logger.dart';

class AudioService extends ChangeNotifier {
  final Logger _logger = LoggerService.getLogger('AudioService');

  final AudioRecordingService _recordingService;
  final BreathTrackingService _breathTrackingService;
  final BreathClassifier _classifier;

  Timer? _audioProcessingTimer;

  bool _onlyExhaleMode = false;
  bool get onlyExhaleMode => _onlyExhaleMode;

  AudioRecordingService get recordingService => _recordingService;
  BreathTrackingService get breathTrackingService => _breathTrackingService;

  bool get isRecording => _recordingService.isRecording;
  Stream<BreathPhase> get breathPhasesStream =>
      _breathTrackingService.breathPhasesStream;
  int get inhaleCount => _breathTrackingService.inhaleCount;
  int get exhaleCount => _breathTrackingService.exhaleCount;

  Stream<Duration> get durationStream => _breathTrackingService.durationStream;
  Stream<double> get tempoStream => _breathTrackingService.tempoStream;

  bool get isSaving => false;
  String? get lastSavedFilePath => null;
  List<InputDevice> get inputDevices => _recordingService.inputDevices;
  InputDevice? get selectedDevice => _recordingService.selectedDevice;
  bool get isLoadingDevices => _recordingService.isLoadingDevices;
  List<int> get audioBuffer => _recordingService.audioBuffer;

  static const int audioProcessingInterval = 300; // ms

  static const int bufferSize = 154350;

  final List<int> _audioBufferForClassification = [];
  bool _isProcessing = false;
  int _classificationErrors = 0;
  static const int maxConsecutiveErrors = 5;

  AudioService({
    AudioRecordingService? recordingService,
    BreathTrackingService? breathTrackingService,
    BreathClassifier? classifier,
  }) : _recordingService = recordingService ?? AudioRecordingService(),
       _breathTrackingService =
           breathTrackingService ?? BreathTrackingService(),
       _classifier = classifier ?? BreathClassifier() {
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      await _classifier.initialize(model: ModelType.standard);
      _logger.info('Breath classifier successfully initialized');
    } catch (e) {
      _logger.severe('Error during classifier initialization: $e');
    }

    await _recordingService.loadInputDevices();

    _setupAudioProcessing();

    notifyListeners();
  }

  Future<void> setOnlyExhaleMode(bool enabled) async {
    _onlyExhaleMode = enabled;
    _logger.info(
      'Switching to mode: ${enabled ? ModelType.exhaleOnly.name : ModelType.standard.name}',
    );

    try {
      await _classifier.switchModel(
        enabled ? ModelType.exhaleOnly : ModelType.standard,
      );
      notifyListeners();
    } catch (e) {
      _logger.severe('Error switching model: $e');
      _onlyExhaleMode = !enabled;
      notifyListeners();
    }
  }

  void _setupAudioProcessing() {
    final List<int> _chunkBuffer = [];
    int _samplesPerInterval =
        (audioProcessingInterval * 44100 / 1000).toInt(); // 13,230 samples

    _recordingService.audioStream.listen((audioData) {
      if (isRecording) {
        _chunkBuffer.addAll(audioData);
        _logger.info(
          'Chunk buffer: ${_chunkBuffer.length}/$_samplesPerInterval samples',
        );

        // Czekaj aż zbierze się pełny interwał (300ms)
        if (_chunkBuffer.length >= _samplesPerInterval) {
          // Weź dokładnie 13,230 samples i dodaj do bufora klasyfikacji
          final chunk = _chunkBuffer.sublist(0, _samplesPerInterval);
          _audioBufferForClassification.addAll(chunk);

          // Usuń przetworzone samples z chunk bufora
          _chunkBuffer.removeRange(0, _samplesPerInterval);

          _logger.info(
            'Adding chunk to classification buffer: ${_audioBufferForClassification.length}/$bufferSize samples',
          );

          // Sliding window: jeśli bufor przekroczy maksymalny rozmiar
          if (_audioBufferForClassification.length > bufferSize) {
            final excessSize =
                _audioBufferForClassification.length - bufferSize;
            _audioBufferForClassification.removeRange(0, excessSize);
            _logger.info('Sliding window: removed $excessSize old samples');
          }

          // Gdy bufor jest pełny, klasyfikuj
          if (_audioBufferForClassification.length == bufferSize &&
              !_isProcessing) {
            _isProcessing = true;
            _logger.info('Buffer full! Starting classification...');

            final samplesToProcess = List<int>.from(
              _audioBufferForClassification,
            );

            _processAudioData(samplesToProcess).then((_) {
              _isProcessing = false;
            });
          }
        }
      }
    });
  }

  Future<void> _processAudioData(List<int> audioData) async {
    final dataLength = audioData.length;
    _logger.fine('Processing audio buffer of length: $dataLength');

    if (dataLength != bufferSize) return;

    try {
      final phase = await _classifier.classify(audioData);

      _breathTrackingService.addBreathPhase(phase);

      _classificationErrors = 0;

      notifyListeners();

      _logger.fine(
        'Breath classification: ${BreathClassifier.getLabelForPhase(phase)}',
      );
    } catch (e) {
      _classificationErrors++;

      _logger.warning(
        'Error during audio processing ($e). Error #$_classificationErrors',
      );

      if (_classificationErrors >= maxConsecutiveErrors) {
        _logger.warning(
          'Too many classification errors in a row. Attempting to reinitialize the classifier...',
        );

        try {
          await _classifier.initialize();
          _classificationErrors = 0;
        } catch (reinitError) {
          _logger.severe(
            'Reinitialization of the classifier failed: $reinitError',
          );
        }
      }
    }
  }

  StreamSubscription<List<int>> subscribeToAudioStream(
    void Function(List<int> audioData) onData,
  ) {
    return _recordingService.audioStream.listen(onData);
  }

  Future<void> startRecording() async {
    if (isRecording) return;

    _audioBufferForClassification.clear();
    _isProcessing = false;
    _classificationErrors = 0;

    await _recordingService.startRecording();
    _breathTrackingService.startTimer();
    notifyListeners();
  }

  Future<void> stopRecording() async {
    if (!isRecording) return;

    await _recordingService.stopRecording();
    _breathTrackingService.pauseTimer();
    notifyListeners();
  }

  void resetCounters() {
    _breathTrackingService.resetTimer();
    notifyListeners();
  }

  void selectDevice(InputDevice device) {
    _recordingService.selectDevice(device);
    notifyListeners();
  }

  Future<void> loadInputDevices() async {
    await _recordingService.loadInputDevices();
    notifyListeners();
  }

  @override
  void dispose() {
    _logger.info('Disposing AudioService');
    _audioProcessingTimer?.cancel();
    _recordingService.dispose();
    _breathTrackingService.dispose();
    _classifier.dispose();
    super.dispose();
  }
}
