import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import '../models/breath_classifier.dart';
import './socket_service.dart';
import './audio_recording_service.dart';
import './breath_tracking_service.dart';
import './audio_file_service.dart';

class AudioService extends ChangeNotifier {
  // Services
  final AudioRecordingService _recordingService;
  final BreathTrackingService _breathTrackingService;
  final AudioFileService _fileService;
  final SocketService _socketService;
  final BreathClassifier _classifier;

  // Timer
  Timer? _audioProcessingTimer;
  
  // Getters for accessing services
  AudioRecordingService get recordingService => _recordingService;
  BreathTrackingService get breathTrackingService => _breathTrackingService;
  AudioFileService get fileService => _fileService;
  SocketService get socketService => _socketService;
  
  // Forward key properties from services
  bool get isRecording => _recordingService.isRecording;
  List<BreathPhase> get breathPhases => _breathTrackingService.breathPhases;
  int get inhaleCount => _breathTrackingService.inhaleCount;
  int get exhaleCount => _breathTrackingService.exhaleCount;
  bool get isSaving => _fileService.isSaving;
  String? get lastSavedFilePath => _fileService.lastSavedFilePath;
  List<InputDevice> get inputDevices => _recordingService.inputDevices;
  InputDevice? get selectedDevice => _recordingService.selectedDevice;
  bool get isLoadingDevices => _recordingService.isLoadingDevices;
  List<int> get audioBuffer => _recordingService.audioBuffer;

  // Configuration constants
  static const int audioProcessingInterval = 300; // milliseconds

  AudioService({
    AudioRecordingService? recordingService,
    BreathTrackingService? breathTrackingService,
    AudioFileService? fileService,
    SocketService? socketService,
    BreathClassifier? classifier,
  }) : 
    _recordingService = recordingService ?? AudioRecordingService(),
    _breathTrackingService = breathTrackingService ?? BreathTrackingService(),
    _fileService = fileService ?? AudioFileService(),
    _socketService = socketService ?? SocketService(),
    _classifier = classifier ?? BreathClassifier() {
    _initialize();
  }

  Future<void> _initialize() async {
    // Initialize classifier
    await _classifier.initialize();
    
    // Load available input devices
    await _recordingService.loadInputDevices();
    
    // Setup audio processing and socket communication
    _setupAudioProcessing();
    
    notifyListeners();
  }

  void _setupAudioProcessing() {
    // Listen to predictions from socket
    _socketService.predictionStream.listen((phase) {
      _breathTrackingService.addBreathPhase(phase);
      notifyListeners();
    });

    // Connect audio stream to socket
    _recordingService.audioStream.listen((audioData) {
      if (isRecording) {
        _socketService.sendAudioData(audioData);
      }
    });
  }

  StreamSubscription<List<int>> subscribeToAudioStream(void Function(List<int> audioChunk) onData) {
    return _recordingService.audioStream.listen(onData);
  }

  Future<void> startRecording() async {
    if (isRecording) return;
    
    await _socketService.connect();
    await _recordingService.startRecording();
    
    notifyListeners();
  }

  Future<void> stopRecording() async {
    if (!isRecording) return;
    
    await _recordingService.stopRecording();
    _socketService.disconnect();
    
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
    _socketService.dispose();
    super.dispose();
  }
}
