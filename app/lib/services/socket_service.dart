import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import '../models/breath_classifier.dart';

class SocketService extends ChangeNotifier {
  // Socket connection
  Socket? _socket;
  bool _isConnected = false;
  bool _isConnecting = false;
  String _errorMessage = '';

  // Public getters
  bool get isConnected => _isConnected;
  bool get isConnecting => _isConnecting;
  String get errorMessage => _errorMessage;

  // Stream for breath phase predictions
  final StreamController<BreathPhase> _predictionController = StreamController<BreathPhase>.broadcast();
  Stream<BreathPhase> get predictionStream => _predictionController.stream;

  final List<int> _audioBuffer = [];
  static const int requiredBufferSize = 48000 ~/ 10 * 3; // equivalent to 48000 * 0.3

  // Connection management
  Future<void> connect() async {
    if (_isConnected || _isConnecting) return;

    _isConnecting = true;
    _errorMessage = '';
    notifyListeners();

    try {
      _socket = await Socket.connect('localhost', 50000, timeout: const Duration(seconds: 5));
      _isConnected = true;
      
      if (kDebugMode) {
        print('Connected to socket server');
      }

      // Listen for responses from the server
      _socket!.listen(
        _handleServerResponse,
        onError: _handleConnectionError,
        onDone: _handleConnectionClosed,
      );
    } catch (e) {
      _handleConnectionError(e);
    } finally {
      _isConnecting = false;
      notifyListeners();
    }
  }

  void disconnect() {
    _socket?.destroy();
    _socket = null;
    _isConnected = false;
    notifyListeners();
  }

  // Server communication
  Future<void> sendAudioData(List<int> audioData) async {
    if (!_isConnected || _socket == null) {
      await connect();
      if (!_isConnected) return;
    }

    try {
      // Dodaj nowe dane do bufora
      _audioBuffer.addAll(audioData);
      
      // Sprawdź czy mamy wystarczająco danych do wysłania (0.3s audio)
      if (_audioBuffer.length >= requiredBufferSize) {
        // Weź dokładnie potrzebny kawałek audio
        final chunkToSend = _audioBuffer.sublist(0, requiredBufferSize);
        
        // Wyczyść wysłane dane z bufora
        _audioBuffer.removeRange(0, requiredBufferSize);
        
        // Convert the audio data to bytes
        final byteData = Uint16List.fromList(chunkToSend);
        
        // First send the size of the data (4 bytes)
        final sizeHeader = ByteData(4)..setInt32(0, byteData.length, Endian.big);
        _socket!.add(sizeHeader.buffer.asUint8List());
        
        // Then send the actual audio data
        _socket!.add(byteData);
        await _socket!.flush();
        
        if (kDebugMode) {
          print('Sent ${byteData.length} bytes of audio data (${requiredBufferSize} samples)');
        }
      }
    } catch (e) {
      _handleConnectionError(e);
    }
  }

  // Response handling
  void _handleServerResponse(Uint8List data) {
    if (data.length == 4) {
      final prediction = data.buffer.asByteData().getInt32(0, Endian.big);
      final phase = _convertPredictionToPhase(prediction);

      if (kDebugMode) {
        print('Received prediction from server: $phase');
      }

      _predictionController.add(phase);
    }
  }

  BreathPhase _convertPredictionToPhase(int prediction) {
    switch (prediction) {
      case 0:
        return BreathPhase.inhale;
      case 1:
        return BreathPhase.exhale;
      default:
        return BreathPhase.silence;
    }
  }

  // Error handling
  void _handleConnectionError(dynamic error) {
    if (kDebugMode) {
      print('Socket error: $error');
    }
    _errorMessage = error.toString();
    _isConnected = false;
    _isConnecting = false;
    notifyListeners();
    disconnect();
  }

  void _handleConnectionClosed() {
    if (kDebugMode) {
      print('Socket connection closed');
    }
    _isConnected = false;
    notifyListeners();
  }

  @override
  void dispose() {
    disconnect();
    _predictionController.close();
    super.dispose();
  }
}
