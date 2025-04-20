import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:typed_data';

enum BreathPhase { exhale, inhale, silence }

class BreathClassifier {
  static const MethodChannel _channel = MethodChannel('breathing_classifier');
  bool _isInitialized = false;
  int _initAttempts = 0;
  static const int maxInitAttempts = 3;

  static const int sampleRate = 44100;

  // Inicjalizacja wykonywana jest po stronie Kotlina
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Sprawdź, czy klasyfikator jest już zainicjalizowany po stronie natywnej
      _isInitialized = await _channel.invokeMethod<bool>('isInitialized') ?? false;

      if (_isInitialized) {
        print('Klasyfikator oddechów zainicjalizowany pomyślnie (potwierdzono przez natywny kod)');
      } else {
        print('Klasyfikator nie został poprawnie zainicjalizowany po stronie natywnej.');

        // Jeśli nie przekroczyliśmy maksymalnej liczby prób, spróbujmy ponownie za moment
        if (_initAttempts < maxInitAttempts) {
          _initAttempts++;
          print('Próba ponownej inicjalizacji $_initAttempts z $maxInitAttempts...');

          // Poczekaj chwilę i spróbuj ponownie sprawdzić
          await Future.delayed(Duration(seconds: 2));
          await initialize();
        } else {
          print('Przekroczono maksymalną liczbę prób inicjalizacji klasyfikatora.');
          throw Exception('Nie udało się zainicjalizować klasyfikatora oddechów po $maxInitAttempts próbach');
        }
      }
    } catch (e) {
      print('Błąd podczas inicjalizacji klasyfikatora: $e');
      _isInitialized = false;
      throw Exception('Nie udało się zainicjalizować klasyfikatora oddechów');
    }
  }

  // Sprawdza stan inicjalizacji klasyfikatora po stronie natywnej
  Future<bool> checkInitialized() async {
    try {
      _isInitialized = await _channel.invokeMethod<bool>('isInitialized') ?? false;
      return _isInitialized;
    } catch (e) {
      print('Błąd podczas sprawdzania stanu inicjalizacji: $e');
      _isInitialized = false;
      return false;
    }
  }

  // Klasyfikacja danych audio przez natywny wrapper Kotlina
  Future<BreathPhase> classify(List<int> audioData) async {
    if (!_isInitialized) {
      // Jeśli klasyfikator nie został zainicjalizowany, spróbuj najpierw sprawdzić stan
      _isInitialized = await checkInitialized();

      if (!_isInitialized) {
        print('Klasyfikator nie jest zainicjalizowany. Ponowna próba inicjalizacji...');
        try {
          await initialize();
        } catch (e) {
          print('Ponowna inicjalizacja nie powiodła się: $e');
          return BreathPhase.silence; // Domyślnie cisza w przypadku błędu
        }
      }
    }

    try {
      // Konwersja danych audio do formatu binarnego (Int16 jako bajty)
      final Int16List audioInt16 = Int16List(audioData.length);
      for (int i = 0; i < audioData.length; i++) {
        audioInt16[i] = audioData[i];
      }

      final ByteData byteData = audioInt16.buffer.asByteData();
      final Uint8List byteList = byteData.buffer.asUint8List();

      // Wywołaj natywną metodę klasyfikacji
      final int classIndex = await _channel.invokeMethod<int>(
        'classifyAudio',
        {'audioData': byteList}
      ) ?? 2; // Domyślnie cisza (2) w przypadku null

      print('Wynik klasyfikacji natywnej: $classIndex (${_indexToBreathPhase(classIndex)})');
      return _indexToBreathPhase(classIndex);
    } catch (e) {
      print('Błąd podczas klasyfikacji natywnej: $e');

      // Jeśli błąd dotyczy inicjalizacji, zresetuj flagę _isInitialized
      if (e.toString().contains('INIT_FAILED')) {
        _isInitialized = false;
      }

      return BreathPhase.silence; // W przypadku błędu zakładamy ciszę
    }
  }

  BreathPhase _indexToBreathPhase(int index) {
    switch (index) {
      case 0: return BreathPhase.exhale;
      case 1: return BreathPhase.inhale;
      case 2: default: return BreathPhase.silence;
    }
  }

  Color getColorForPhase(BreathPhase phase) {
    switch (phase) {
      case BreathPhase.inhale: return Colors.red;
      case BreathPhase.exhale: return Colors.green;
      case BreathPhase.silence: return Colors.blue;
    }
  }

  String getLabelForPhase(BreathPhase phase) {
    switch (phase) {
      case BreathPhase.inhale: return 'Wdech';
      case BreathPhase.exhale: return 'Wydech';
      case BreathPhase.silence: return 'Cisza';
    }
  }

  void dispose() {
    // Nic nie robimy tutaj - zasoby są zarządzane po stronie Kotlina
    // i zwalniane w MainActivity.onDestroy()
    _isInitialized = false;
  }
}