import 'dart:async';
import 'dart:math' as math;
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import '../models/breath_classifier.dart';

class AudioService extends ChangeNotifier {
  static const int sampleRate = 48000;
  static const double refreshTime = 0.3;
  static final int chunkSize = (sampleRate * refreshTime).round();
  
  bool _isRecording = false;
  bool get isRecording => _isRecording;
  
  List<InputDevice> _inputDevices = [];
  List<InputDevice> get inputDevices => _inputDevices;
  InputDevice? _selectedDevice;
  InputDevice? get selectedDevice => _selectedDevice;
  bool _isLoadingDevices = false;
  bool get isLoadingDevices => _isLoadingDevices;
  
  final List<double> _audioBuffer = [];
  List<double> get audioBuffer => _audioBuffer;
  
  final List<int> _microphoneBuffer = [];
  static const int maxMicrophoneBufferSize = sampleRate * 5; 
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
  final BreathClassifier _classifier = BreathClassifier();

  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _amplitudeStreamSubscription;

  bool _isSaving = false;
  bool get isSaving => _isSaving;
  String? _lastSavedFilePath;
  String? get lastSavedFilePath => _lastSavedFilePath;

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

    _captureStream();
    // _startSimulation(); // currently disabled
  }

  Future<void> printInputDevices() async {
    final devices = await _recorder.listInputDevices();
    for (final device in devices) {
      if (kDebugMode) {
        print('Device id: ${device.id}, label: ${device.label}');
      }
    }
  }

  void stopRecording() async {
    if (!_isRecording) return;
    
    _isRecording = false;
    _simulationTimer?.cancel();

    await _amplitudeStreamSubscription?.cancel();
    await _recorder.stop();

    notifyListeners();
  }

  void resetCounters() {
    _inhaleCount = 0;
    _exhaleCount = 0;
    notifyListeners();
  }

  void _startSimulation() {
    const simulationInterval = Duration(milliseconds: (refreshTime * 1000 ~/ 3));
    _simulationTimer = Timer.periodic(simulationInterval, (_) => _generateAudioData());
  }

  void _captureStream() async {
    if (_selectedDevice == null) return;
    
    await _startStreamCapturing();
    _prepareBuffers();
  }
  
  Future<void> _startStreamCapturing() async {
    final config = RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: sampleRate,
      numChannels: 2,
      device: _selectedDevice!,
    );
    
    final audioStream = await _recorder.startStream(config);
    _createAudioStreamSubscription(audioStream);
  }
  
  void _prepareBuffers() {
    _microphoneBuffer.clear();
    _breathPhases.clear();
  }
  
  void _createAudioStreamSubscription(Stream<Uint8List> audioStream) {
    _amplitudeStreamSubscription = audioStream.listen((data) {
      final pcmSamples = _recorder.convertBytesToInt16(data);
      synchronized(() {
        _addToMicrophoneBuffer(pcmSamples);
      });
    });
  }
  
  void _addToMicrophoneBuffer(List<int> samples) {
    _microphoneBuffer.addAll(samples);
    
    if (_microphoneBuffer.length > maxMicrophoneBufferSize) {
      _microphoneBuffer.removeRange(0, _microphoneBuffer.length - maxMicrophoneBufferSize);
    }
  }

  void synchronized(Function() action) {
    action();
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

  Future<bool> _checkStoragePermission() async {
    if (Platform.isAndroid) {
      final status = await Permission.storage.request();
      return status.isGranted;
    } else if (Platform.isIOS) {
      return true;
    } else if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
      return true;
    }
    return false;
  }

  Future<String?> saveRecording() async {
    if (_microphoneBuffer.isEmpty) {
      return null;
    }

    _isSaving = true;
    notifyListeners();

    try {
      final hasPermission = await _checkStoragePermission();
      if (!hasPermission) {
        if (kDebugMode) {
          print('Storage permission denied');
        }
        return null;
      }

      final byteData = ByteData(_microphoneBuffer.length * 2);
      for (int i = 0; i < _microphoneBuffer.length; i++) {
        byteData.setInt16(i * 2, _microphoneBuffer[i], Endian.little);
      }
      final pcmBytes = byteData.buffer.asUint8List();

      final wavHeader = _createWavHeader(pcmBytes.length, sampleRate);

      final wavFile = Uint8List(wavHeader.length + pcmBytes.length);
      wavFile.setAll(0, wavHeader);
      wavFile.setAll(wavHeader.length, pcmBytes);

      final directory = await getApplicationDocumentsDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final filePath = '${directory.path}/recording_$timestamp.wav';

      final file = File(filePath);
      await file.writeAsBytes(wavFile);

      _lastSavedFilePath = filePath;
      if (kDebugMode) {
        print('Recording saved to: $filePath');
      }

      return filePath;
    } catch (e) {
      if (kDebugMode) {
        print('Error saving recording: $e');
      }
      return null;
    } finally {
      _isSaving = false;
      notifyListeners();
    }
  }


  Uint8List _createWavHeader(int dataLength, int sampleRate) {
    final header = ByteData(44);
    // RIFF header
    header.setUint8(0, 82); // 'R'
    header.setUint8(1, 73); // 'I'
    header.setUint8(2, 70); // 'F'
    header.setUint8(3, 70); // 'F'
    
    // File size
    header.setUint32(4, dataLength + 36, Endian.little);
    
    // WAVE header
    header.setUint8(8, 87);  // 'W'
    header.setUint8(9, 65);  // 'A'
    header.setUint8(10, 86); // 'V'
    header.setUint8(11, 69); // 'E'
    
    // fmt chunk
    header.setUint8(12, 102); // 'f'
    header.setUint8(13, 109); // 'm'
    header.setUint8(14, 116); // 't'
    header.setUint8(15, 32);  // ' '
    
    // FMT chunk size (16 for PCM)
    header.setUint32(16, 16, Endian.little);
    
    // Audio format (1 = PCM)
    header.setUint16(20, 1, Endian.little);
    
    // Liczba kanałów (2 for stereo)
    header.setUint16(22, 2, Endian.little);
    
    // Sample rate
    header.setUint32(24, sampleRate, Endian.little);
    
    // Byte rate = SampleRate * NumChannels * BitsPerSample/8
    header.setUint32(28, sampleRate * 2 * 16 ~/ 8, Endian.little);
    
    // Block align = NumChannels * BitsPerSample/8
    header.setUint16(32, 2 * 16 ~/ 8, Endian.little);
    
    // Bits per sample
    header.setUint16(34, 16, Endian.little);
    
    // data chunk
    header.setUint8(36, 100); // 'd'
    header.setUint8(37, 97);  // 'a'
    header.setUint8(38, 116); // 't'
    header.setUint8(39, 97);  // 'a'
    
    // Data chunk size
    header.setUint32(40, dataLength, Endian.little);
    
    return header.buffer.asUint8List();
  }

  @override
  void dispose() {
    _simulationTimer?.cancel();
    _amplitudeStreamSubscription?.cancel();
    _recorder.stop();
    super.dispose();
  }
}
