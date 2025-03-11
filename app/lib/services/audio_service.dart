import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';
import '../models/breath_classifier.dart';

class AudioService extends ChangeNotifier {
  // Constants similar to the Python version
  static const int sampleRate = 44100;
  static const double refreshTime = 0.3;
  static final int chunkSize = (sampleRate * refreshTime).round();
  
  // Audio state
  bool _isRecording = false;
  bool get isRecording => _isRecording;
  
  // Audio data storage
  final List<double> _audioBuffer = [];
  List<double> get audioBuffer => _audioBuffer;
  
  // Breath phase tracking
  final List<BreathPhase> _breathPhases = [];
  List<BreathPhase> get breathPhases => _breathPhases;
  
  // Counters
  int _inhaleCount = 0;
  int _exhaleCount = 0;
  int get inhaleCount => _inhaleCount;
  int get exhaleCount => _exhaleCount;
  
  // Maximum history to keep
  final int _maxHistorySeconds = 5;
  int get maxBufferSize => sampleRate * _maxHistorySeconds;
  int get maxPhaseHistory => (_maxHistorySeconds / refreshTime).round();

  // Audio generation timer
  Timer? _simulationTimer;
  final BreathClassifier _classifier = BreathClassifier();

  AudioService() {
    _classifier.initialize();
  }

  Future<bool> requestMicrophonePermission() async {
    final status = await Permission.microphone.request();
    return status == PermissionStatus.granted;
  }

  void startRecording() async {
    if (_isRecording) return;
    
    final bool hasPermission = await requestMicrophonePermission();
    if (!hasPermission) {
      if (kDebugMode) {
        print('Microphone permission denied');
      }
      return;
    }
    
    _isRecording = true;
    notifyListeners();
    
    // For now, we'll generate simulated audio data
    _startSimulation();
  }

  void stopRecording() {
    if (!_isRecording) return;
    
    _isRecording = false;
    _simulationTimer?.cancel();
    notifyListeners();
  }

  void resetCounters() {
    _inhaleCount = 0;
    _exhaleCount = 0;
    notifyListeners();
  }

  void _startSimulation() {
    // Generate fake audio data similar to a breathing pattern
    _simulationTimer = Timer.periodic(
      const Duration(milliseconds: (refreshTime * 1000 ~/ 3)),
      (_) => _generateAudioData(),
    );
  }

  void _generateAudioData() async {
    // Generate synthetic breathing waveform
    final int currentTime = DateTime.now().millisecondsSinceEpoch;
    final List<double> newSamples = List.generate(
      chunkSize ~/ 3,
      (i) {
        final double t = (currentTime / 1000.0) + (i / sampleRate);
        
        // Base sine wave with frequency of approximately 0.3Hz (1 breath every ~3 seconds)
        double baseBreathing = math.sin(2 * math.pi * 0.3 * t) * 0.5;
        
        // Add some higher frequency components to make it look like real audio
        double noise = math.sin(2 * math.pi * 100 * t) * 0.05 + 
                     math.sin(2 * math.pi * 220 * t) * 0.03;
                     
        return baseBreathing + noise;
      },
    );
    
    // Add to buffer and trim if needed
    _audioBuffer.addAll(newSamples);
    if (_audioBuffer.length > maxBufferSize) {
      _audioBuffer.removeRange(0, _audioBuffer.length - maxBufferSize);
    }
    
    // Every refreshTime, classify the breath
    if (_audioBuffer.length >= chunkSize) {
      final segment = _audioBuffer.sublist(_audioBuffer.length - chunkSize);
      
      // Get the classification
      final phase = await _classifier.classify(segment);
      
      // Update phases list
      _breathPhases.add(phase);
      if (_breathPhases.length > maxPhaseHistory) {
        _breathPhases.removeAt(0);
      }
      
      // Update counters
      if (phase == BreathPhase.inhale) {
        _inhaleCount++;
      } else if (phase == BreathPhase.exhale) {
        _exhaleCount++;
      }
      
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _simulationTimer?.cancel();
    super.dispose();
  }
}
