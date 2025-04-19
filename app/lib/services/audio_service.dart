import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import '../models/breath_classifier.dart';
import './audio_recording_service.dart';
import './breath_tracking_service.dart';
import './audio_file_service.dart';

class AudioService extends ChangeNotifier {
  // Usługi
  final AudioRecordingService _recordingService;
  final BreathTrackingService _breathTrackingService;
  final AudioFileService _fileService;
  final BreathClassifier _classifier;

  // Timer
  Timer? _audioProcessingTimer;

  // Gettery dla dostępu do serwisów
  AudioRecordingService get recordingService => _recordingService;
  BreathTrackingService get breathTrackingService => _breathTrackingService;
  AudioFileService get fileService => _fileService;

  // Przekazywanie kluczowych właściwości z serwisów
  bool get isRecording => _recordingService.isRecording;
  Stream<BreathPhase> get breathPhasesStream => _breathTrackingService.breathPhasesStream;
  int get inhaleCount => _breathTrackingService.inhaleCount;
  int get exhaleCount => _breathTrackingService.exhaleCount;
  bool get isSaving => _fileService.isSaving;
  String? get lastSavedFilePath => _fileService.lastSavedFilePath;
  List<InputDevice> get inputDevices => _recordingService.inputDevices;
  InputDevice? get selectedDevice => _recordingService.selectedDevice;
  bool get isLoadingDevices => _recordingService.isLoadingDevices;
  List<int> get audioBuffer => _recordingService.audioBuffer;

  // Konfiguracja stała
  static const int audioProcessingInterval = 300; // milisekundy
  static const int bufferSize = 13230; // 0.3 sekundy

  // Dodaj te zmienne członkowskie klasy
  final List<int> _audioBufferForClassification = [];
  bool _isProcessing = false;

  AudioService({
    AudioRecordingService? recordingService,
    BreathTrackingService? breathTrackingService,
    AudioFileService? fileService,
    BreathClassifier? classifier,
  }) :
    _recordingService = recordingService ?? AudioRecordingService(),
    _breathTrackingService = breathTrackingService ?? BreathTrackingService(),
    _fileService = fileService ?? AudioFileService(),
    _classifier = classifier ?? BreathClassifier() {
    _initialize();
  }

  Future<void> _initialize() async {
    // Inicjalizacja klasyfikatora
    await _classifier.initialize();

    // Wczytanie dostępnych urządzeń wejściowych
    await _recordingService.loadInputDevices();

    // Ustawienie przetwarzania audio
    _setupAudioProcessing();

    notifyListeners();
  }

  void _setupAudioProcessing() {
    _recordingService.audioStream.listen((audioData) {
      if (isRecording) {
        // Dodaj dane do bufora klasyfikacji
        _audioBufferForClassification.addAll(audioData);
        
        // Sprawdź czy uzbieraliśmy wystarczająco dużo próbek i nie trwa przetwarzanie
        if (_audioBufferForClassification.length >= bufferSize && !_isProcessing) {
          _isProcessing = true;
          
          // Weź dokładnie 0.3s najnowszych danych
          final samplesToProcess = _audioBufferForClassification.sublist(
            _audioBufferForClassification.length - bufferSize
          );
          
          // Wyczyść bufor z próbkami
          _audioBufferForClassification.clear();

          // Przetwarzaj dane
          _processAudioData(samplesToProcess).then((_) {
            // Po zakończeniu przetwarzania, zezwól na kolejne
            _isProcessing = false;
          });
        }
      }
    });
  }

  Future<void> _processAudioData(List<int> audioData) async {
    final dataLength = audioData.length;
    if (kDebugMode) {
      print('Przetwarzam bufor o długości: $dataLength');
    }
    
    if (dataLength != bufferSize) return;

    try {

      // Klasyfikacja danych audio
      final phase = await _classifier.classify(audioData);

      // Dodanie wyniku klasyfikacji do systemu śledzenia oddechów
      _breathTrackingService.addBreathPhase(phase);
      notifyListeners();

      if (kDebugMode) {
        print('Lokalna klasyfikacja: ${_classifier.getLabelForPhase(phase)}');
      }
    } catch (e) {
      if (kDebugMode) {
        print('Błąd podczas przetwarzania audio: $e');
      }
    }
  }

  StreamSubscription<List<int>> subscribeToAudioStream(void Function(List<int> audioData) onData) {
    return _recordingService.audioStream.listen(onData);
  }

  Future<void> startRecording() async {
    if (isRecording) return;

    // Wyczyść bufor przy rozpoczęciu nagrywania
    _audioBufferForClassification.clear();
    _isProcessing = false;
    
    await _recordingService.startRecording();
    notifyListeners();
  }

  Future<void> stopRecording() async {
    if (!isRecording) return;

    await _recordingService.stopRecording();

    notifyListeners();
  }

  void resetCounters() {
    _breathTrackingService.resetCounters();
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

  Future<String?> saveRecording() async {
    final result = await _fileService.saveRecording(
      _recordingService.audioBuffer,
      AudioRecordingService.sampleRate
    );
    notifyListeners();
    return result;
  }

  @override
  void dispose() {
    if (kDebugMode) {
      print('Disposing AudioService');
    }
    _audioProcessingTimer?.cancel();
    _recordingService.dispose();
    _breathTrackingService.dispose();
    _classifier.dispose();
    super.dispose();
  }
}