import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:record_platform_interface/record_platform_interface.dart';
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

  // Temporary buffer for audio data
  final List<int> _pcmBuffer = [];
  
  // First 10 PCM samples for UI display
  List<int> _first10PcmSamples = [];
  List<int> get first10PcmSamples => _first10PcmSamples;
  
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
  // Timer do odświeżania amplitudy
  Timer? _amplitudeRefreshTimer;
  final BreathClassifier _classifier = BreathClassifier();

  // Record plugin
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _amplitudeStreamSubscription;
  double _currentAmplitude = 0.0;
  double get currentAmplitude => _currentAmplitude;

  

  AudioService(){
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
    await printInputDevices();

    // Record plugin
    _startRecordingMicrophoneAmplitude();
    
    // For now, we'll generate simulated audio data
    _startSimulation();
  }

  Future<void> printInputDevices() async {
          final devices = await _recorder.listInputDevices();
        for (final device in devices) {
          print('Device id: ${device.id}, label: ${device.label}');
        }
  }

  void stopRecording() async {
    if (!_isRecording) return;
    
    _isRecording = false;
    _simulationTimer?.cancel();

    await _amplitudeStreamSubscription?.cancel();
    await _recorder.stop();

    _pcmBuffer.clear();

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

  void _startRecordingMicrophoneAmplitude() async {
    final devices = await _recorder.listInputDevices();
    final device = devices[0];
    print("Used device: ${device.label}");
    
    // Używamy startStream z konfiguracją PCM 16-bit
    final config = RecordConfig(encoder: AudioEncoder.pcm16bits, device: device);
    final audioStream = await _recorder.startStream(config);
    
    // Ustawienie timera odświeżania amplitudy
    _amplitudeRefreshTimer = Timer.periodic(
      Duration(milliseconds: (refreshTime * 1000).round()), 
      (_) => _updateAmplitude()
    );
    
    // Nasłuchiwanie strumienia audio i buforowanie próbek
    _amplitudeStreamSubscription = audioStream.listen((data) {
      // Log surowych danych audio
      if (kDebugMode) {
        print("Raw audio data length: ${data.length} bytes");
      }
      
      // Konwersja odebranych bajtów na PCM 16-bit
      final pcmSamples = _recorder.convertBytesToInt16(data);
      
      // Dodanie próbek do bufora
      synchronized(() {
        _pcmBuffer.addAll(pcmSamples);
      });
      
      // Zapisz pierwsze 10 próbek do wyświetlenia w UI
      if (pcmSamples.isNotEmpty) {
        _first10PcmSamples = pcmSamples.sublist(4, 14);
        notifyListeners();
      }
      
      // Debug: wypisanie pierwszych kilku wartości PCM
      if (kDebugMode) {
        print("First 10 PCM samples: ${pcmSamples.take(10).toList()}");
      }
    });
  }

  void synchronized(Function() action) {
    action();
  }

    // Metoda aktualizująca amplitudę na podstawie zawartości bufora
  void _updateAmplitude() {
    // Kopiujemy bufor i czyścimy oryginalny
    List<int> currentPcmSamples = [];
    synchronized(() {
      if (_pcmBuffer.isEmpty) return;
      currentPcmSamples = List.from(_pcmBuffer);
      _pcmBuffer.clear();
    });
    
    if (currentPcmSamples.isEmpty) return;
    
    // Wyliczamy maksymalną wartość bezwzględną
    final maxSample = currentPcmSamples.fold<int>(
      0,
      (prev, element) => element.abs() > prev ? element.abs() : prev,
    );
    
    // Normalizacja (zakładamy 16-bit PCM: wartość max = 32767)
    _currentAmplitude = maxSample / 32767.0;
    
    if (kDebugMode) {
      print("Aktualna amplituda (${DateTime.now()}): $_currentAmplitude");
      print("Liczba próbek użytych do obliczenia amplitudy: ${currentPcmSamples.length}");
    }
    
    notifyListeners();
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
    _amplitudeStreamSubscription?.cancel();
    _recorder.stop();
    super.dispose();
  }
}
