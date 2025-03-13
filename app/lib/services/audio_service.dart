import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import '../models/breath_classifier.dart';

class AudioService extends ChangeNotifier {
  static const int sampleRate = 44100;
  static const double refreshTime = 0.3;
  static final int chunkSize = (sampleRate * refreshTime).round(); // ~13,230 samples
  
  // Audio state
  bool _isRecording = false;
  bool get isRecording => _isRecording;
  
  // Audio device selection
  List<InputDevice> _inputDevices = [];
  List<InputDevice> get inputDevices => _inputDevices;
  InputDevice? _selectedDevice;
  InputDevice? get selectedDevice => _selectedDevice;
  bool _isLoadingDevices = false;
  bool get isLoadingDevices => _isLoadingDevices;
  
  // Audio data storage
  final List<double> _audioBuffer = [];
  List<double> get audioBuffer => _audioBuffer;

  // Temporary buffer for audio data
  final List<int> _pcmBuffer = [];
  
  List<int> _first10PcmSamples = [];
  List<int> get first10PcmSamples => _first10PcmSamples;
  
  // Increased to store more points for smoother visualization
  final List<int> _microphoneBuffer = [];
  static const int maxMicrophoneBufferSize = 44100 * 5; // 5 seconds of audio at 44.1kHz
  List<int> get microphoneBuffer => _microphoneBuffer;
  
  final List<BreathPhase> _breathPhases = [];
  List<BreathPhase> get breathPhases => _breathPhases;
  
  int _inhaleCount = 0;
  int _exhaleCount = 0;
  int get inhaleCount => _inhaleCount;
  int get exhaleCount => _exhaleCount;
  
  final int _maxHistorySeconds = 5;
  int get maxBufferSize => sampleRate * _maxHistorySeconds;
  int get maxPhaseHistory => (_maxHistorySeconds / refreshTime).round();

  Timer? _simulationTimer;
  Timer? _amplitudeRefreshTimer;
  final BreathClassifier _classifier = BreathClassifier();

  // Record plugin
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _amplitudeStreamSubscription;
  double _currentAmplitude = 0.0;
  double get currentAmplitude => _currentAmplitude;

  AudioService(){
    _classifier.initialize();
    loadInputDevices();
  }

  Future<void> loadInputDevices() async {
    _isLoadingDevices = true;
    notifyListeners();
    
    try {
      final status = await Permission.microphone.request();
      if (status == PermissionStatus.granted) {
        _inputDevices = await _recorder.listInputDevices();
        
        if (_selectedDevice == null && _inputDevices.isNotEmpty) {
          selectDevice(_inputDevices[0]);
        }
      } else {
        if (kDebugMode) {
          print('Microphone permission denied');
        }
        _inputDevices = [];
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error loading input devices: $e');
      }
      _inputDevices = [];
    } finally {
      _isLoadingDevices = false;
      notifyListeners();
    }
  }
  
  void selectDevice(InputDevice device) {
    _selectedDevice = device;
    if (kDebugMode) {
      print('Selected device: ${device.label}');
    }
    notifyListeners();
  }

  Future<bool> requestMicrophonePermission() async {
    final status = await Permission.microphone.request();
    return status == PermissionStatus.granted;
  }

  void startRecording() async {
    if (_isRecording || _selectedDevice == null) return;
    
    final bool hasPermission = await requestMicrophonePermission();
    if (!hasPermission) {
      if (kDebugMode) {
        print('Microphone permission denied');
      }
      return;
    }
    
    _isRecording = true;
    notifyListeners();

    _startRecordingMicrophoneAmplitude();
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
    // Don't clear the microphone buffer here to allow viewing the last recording

    notifyListeners();
  }

  void resetCounters() {
    _inhaleCount = 0;
    _exhaleCount = 0;
    notifyListeners();
  }

  void _startSimulation() {
    _simulationTimer = Timer.periodic(
      const Duration(milliseconds: (refreshTime * 1000 ~/ 3)),
      (_) => _generateAudioData(),
    );
  }

  void _startRecordingMicrophoneAmplitude() async {
    if (_selectedDevice == null) return;
    
    final config = RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: sampleRate, 
      numChannels: 1, // Mono
      device: _selectedDevice!,
    );
    
    final audioStream = await _recorder.startStream(config);
    
    _amplitudeRefreshTimer = Timer.periodic(
      Duration(milliseconds: (refreshTime * 1000).round()), 
      (_) => _updateAmplitude()
    );
    
    _microphoneBuffer.clear();
    _breathPhases.clear(); 
    
    _amplitudeStreamSubscription = audioStream.listen((data) {
      final pcmSamples = _recorder.convertBytesToInt16(data);
      
      synchronized(() {
        _pcmBuffer.addAll(pcmSamples);
        
        _microphoneBuffer.addAll(pcmSamples);
        
        if (_microphoneBuffer.length > maxMicrophoneBufferSize) {
          _microphoneBuffer.removeRange(0, _microphoneBuffer.length - maxMicrophoneBufferSize);
        }
      });
      
      if (pcmSamples.isNotEmpty) {
        _first10PcmSamples = pcmSamples.sublist(
          0, 
          math.min(10, pcmSamples.length)
        );
        notifyListeners();
      }
      
      if (kDebugMode) {
        print('First 10 PCM samples: ${pcmSamples.take(10).toList()}');
      }
    });
  }

  void synchronized(Function() action) {
    action();
  }

  void _updateAmplitude() {
    // Copy buffer and clear original
    List<int> currentPcmSamples = [];
    synchronized(() {
      if (_pcmBuffer.isEmpty) return;
      currentPcmSamples = List.from(_pcmBuffer);
      _pcmBuffer.clear();
    });
    
    if (currentPcmSamples.isEmpty) return;
    
    // Calculate maximum absolute value
    final maxSample = currentPcmSamples.fold<int>(
      0,
      (prev, element) => element.abs() > prev ? element.abs() : prev,
    );
    
    // Normalize (assuming 16-bit PCM: max value = 32767)
    _currentAmplitude = maxSample / 32767.0;
    
    // Convert PCM samples to floating point for the classifier
    final List<double> normalizedSamples = currentPcmSamples
        .map((sample) => sample / 32767.0)
        .toList();
    
    _classifier.classify(normalizedSamples).then((phase) {
      _breathPhases.add(phase);
      if (_breathPhases.length > maxPhaseHistory) {
        _breathPhases.removeAt(0);
      }
      
      if (phase == BreathPhase.inhale) {
        _inhaleCount++;
      } else if (phase == BreathPhase.exhale) {
        _exhaleCount++;
      }
      
      notifyListeners();
    });
    
    if (kDebugMode) {
      print('Current amplitude (${DateTime.now()}): $_currentAmplitude');
      print('Number of samples used to calculate amplitude: ${currentPcmSamples.length}');
    }
    
    notifyListeners();
  }

  void _generateAudioData() async {
    final int currentTime = DateTime.now().millisecondsSinceEpoch;
    final List<double> newSamples = List.generate(
      chunkSize ~/ 3,
      (i) {
        final double t = (currentTime / 1000.0) + (i / sampleRate);
        
        double baseBreathing = math.sin(2 * math.pi * 0.3 * t) * 0.5;
        
        double noise = math.sin(2 * math.pi * 100 * t) * 0.05 + 
                     math.sin(2 * math.pi * 220 * t) * 0.03;
                     
        return baseBreathing + noise;
      },
    );
    
    _audioBuffer.addAll(newSamples);
    if (_audioBuffer.length > maxBufferSize) {
      _audioBuffer.removeRange(0, _audioBuffer.length - maxBufferSize);
    }
    
    if (_audioBuffer.length >= chunkSize) {
      final segment = _audioBuffer.sublist(_audioBuffer.length - chunkSize);
      
      final phase = await _classifier.classify(segment);
      
      _breathPhases.add(phase);
      if (_breathPhases.length > maxPhaseHistory) {
        _breathPhases.removeAt(0);
      }
      
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
