import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';

class AudioRecordingService {
  // Audio configuration
  static const int sampleRate = 48000;
  static const double refreshTime = 0.3;
  static final int chunkSize = (sampleRate * refreshTime).round();
  static const int maxMicrophoneBufferSize = sampleRate * 5;

  // Recording state
  bool _isRecording = false;
  bool get isRecording => _isRecording;
  
  // Audio devices
  final List<InputDevice> _inputDevices = [];
  List<InputDevice> get inputDevices => _inputDevices;
  InputDevice? _selectedDevice;
  InputDevice? get selectedDevice => _selectedDevice;
  bool _isLoadingDevices = false;
  bool get isLoadingDevices => _isLoadingDevices;
  
  // Audio buffer
  final List<int> _audioBuffer = [];
  List<int> get audioBuffer => _audioBuffer;
  
  // Recorder
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _amplitudeStreamSubscription;
  
  // Events
  final _audioStreamController = StreamController<List<int>>.broadcast();
  Stream<List<int>> get audioStream => _audioStreamController.stream;

  AudioRecordingService();

  Future<void> loadInputDevices() async {
    _isLoadingDevices = true;
    
    try {
      final status = await Permission.microphone.request();
      if (status == PermissionStatus.granted) {
        _inputDevices.clear();
        _inputDevices.addAll(await _recorder.listInputDevices());
        if (_selectedDevice == null && _inputDevices.isNotEmpty) {
          selectDevice(_inputDevices[0]);
        }
      } else {
        if (kDebugMode) {
          print('Microphone permission denied');
        }
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error loading input devices: $e');
      }
    } finally {
      _isLoadingDevices = false;
    }
  }
  
  void selectDevice(InputDevice device) {
    _selectedDevice = device;
    if (kDebugMode) {
      print('Selected device: ${device.label}');
    }
  }

  Future<bool> requestMicrophonePermission() async {
    final status = await Permission.microphone.request();
    return status == PermissionStatus.granted;
  }

  Future<void> startRecording() async {
    if (_isRecording || _selectedDevice == null) return;
    
    final bool hasPermission = await requestMicrophonePermission();
    if (!hasPermission) {
      if (kDebugMode) {
        print('Microphone permission denied');
      }
      return;
    }
    
    _isRecording = true;
    _audioBuffer.clear();
    
    await _startStreamCapturing();
  }

  Future<void> stopRecording() async {
    if (!_isRecording) return;
    
    _isRecording = false;
    await _amplitudeStreamSubscription?.cancel();
    await _recorder.stop();
  }

  Future<void> _startStreamCapturing() async {
    if (_selectedDevice == null) return;
    
    final config = RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: sampleRate,
      numChannels: 2,
      device: _selectedDevice!,
    );
    
    final audioStream = await _recorder.startStream(config);
    _createAudioStreamSubscription(audioStream);
  }
  
  void _createAudioStreamSubscription(Stream<Uint8List> audioStream) {
    _amplitudeStreamSubscription = audioStream.listen((data) {
      final pcmSamples = _recorder.convertBytesToInt16(data, Endian.big);
      _addToMicrophoneBuffer(pcmSamples);
        _audioStreamController.add(pcmSamples);
      
    });
  }
  
  void _addToMicrophoneBuffer(List<int> samples) {
    _audioBuffer.addAll(samples);
    
    if (_audioBuffer.length > maxMicrophoneBufferSize) {
      _audioBuffer.removeRange(0, _audioBuffer.length - maxMicrophoneBufferSize);
    }
  }

  Future<void> printInputDevices() async {
    final devices = await _recorder.listInputDevices();
    for (final device in devices) {
      if (kDebugMode) {
        print('Device id: ${device.id}, label: ${device.label}');
      }
    }
  }

  void dispose() {
    _amplitudeStreamSubscription?.cancel();
    _recorder.stop();
    _audioStreamController.close();
  }
}
