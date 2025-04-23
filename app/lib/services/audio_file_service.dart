import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:logging/logging.dart';
import 'package:breathing_app/utils/logger.dart';

class AudioFileService {
  final Logger _logger = LoggerService.getLogger('AudioFileService');
  bool _isSaving = false;
  bool get isSaving => _isSaving;
  String? _lastSavedFilePath;
  String? get lastSavedFilePath => _lastSavedFilePath;

  Future<bool> _checkStoragePermission() async {
    if (Platform.isAndroid) {
      final status = await Permission.storage.request();
      return status.isGranted;
    }
    return Platform.isIOS || Platform.isWindows || Platform.isLinux || Platform.isMacOS;
  }

  Future<String?> saveRecording(List<int> audioData, int sampleRate) async {
    if (audioData.isEmpty) {
      return null;
    }

    _isSaving = true;

    try {
      final hasPermission = await _checkStoragePermission();
      if (!hasPermission) {
        _logger.warning('Storage permission denied');
        return null;
      }

      final byteData = ByteData(audioData.length * 2);
      for (int i = 0; i < audioData.length; i++) {
        byteData.setInt16(i * 2, audioData[i], Endian.little);
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
      _logger.info('Recording saved to: $filePath');

      return filePath;
    } catch (e) {
      _logger.severe('Error saving recording: $e');
      return null;
    } finally {
      _isSaving = false;
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

    // Number of channels (2 for stereo)
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
}
