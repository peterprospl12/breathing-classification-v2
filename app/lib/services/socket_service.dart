import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import '../models/breath_classifier.dart';

class SocketService extends ChangeNotifier {
  Socket? _socket;
  bool _isConnected = false;
  bool _isConnecting = false;
  String _errorMessage = '';

  bool get isConnected => _isConnected;
  bool get isConnecting => _isConnecting;
  String get errorMessage => _errorMessage;

  // Stream controller for breath phase predictions
  final StreamController<BreathPhase> _predictionController = StreamController<BreathPhase>.broadcast();
  Stream<BreathPhase> get predictionStream => _predictionController.stream;

  Future<void> connect() async {
    if (_isConnected || _isConnecting) return;

    _isConnecting = true;
    _errorMessage = '';
    notifyListeners();

    try {
      _socket = await Socket.connect('localhost', 50000, timeout: const Duration(seconds: 5));
      _isConnected = true;
      _isConnecting = false;
      notifyListeners();

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
    }
  }

  void _handleServerResponse(Uint8List data) {
    if (data.length == 4) {
      // The server sends a 4-byte integer representing the prediction
      final prediction = data.buffer.asByteData().getInt32(0, Endian.big);
      
      // Convert prediction integer to BreathPhase enum
      BreathPhase phase;
      switch (prediction) {
        case 0:
          phase = BreathPhase.inhale;
          break;
        case 1:
          phase = BreathPhase.exhale;
          break;
        default:
          phase = BreathPhase.silence;
      }

      if (kDebugMode) {
        print('Received prediction from server: $phase');
      }

      // Send prediction to stream
      _predictionController.add(phase);
    }
  }

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

  Future<void> sendAudioData(List<int> audioData) async {
    if (!_isConnected || _socket == null) {
      await connect();
      if (!_isConnected) return;
    }

    try {
      // Convert the audio data to bytes
      final byteData = Uint16List.fromList(audioData);
      
      // First send the size of the data (4 bytes)
      final sizeHeader = ByteData(4)..setInt32(0, byteData.length, Endian.big);
      _socket!.add(sizeHeader.buffer.asUint8List());
      
      // Then send the actual audio data
      _socket!.add(byteData);
      await _socket!.flush();
      
      if (kDebugMode) {
        print('Sent ${byteData.length} bytes of audio data');
      }
    } catch (e) {
      _handleConnectionError(e);
    }
  }

  void disconnect() {
    _socket?.destroy();
    _socket = null;
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
