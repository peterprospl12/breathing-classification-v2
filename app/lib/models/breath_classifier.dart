import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'dart:typed_data'; 

enum BreathPhase { exhale, inhale, silence }

class BreathClassifier {
  OrtSession? _session;
  bool _isInitialized = false;
  
  static const int sampleRate = 44100;

  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Inicjalizacja środowiska ONNX Runtime
      OrtEnv.instance.init();

      // Wczytanie modelu z zasobów
      final sessionOptions = OrtSessionOptions();
      final rawAssetFile = await rootBundle.load('assets/models/breath_classifier_model_audio_input.onnx');
      final bytes = rawAssetFile.buffer.asUint8List();
      _session = OrtSession.fromBuffer(bytes, sessionOptions);

      _isInitialized = true;
    } catch (e) {
      print('Błąd podczas inicjalizacji modelu: $e');
      throw Exception('Nie udało się zainicjalizować modelu klasyfikatora oddechów');
    }
  }

  Future<BreathPhase> classify(List<int> audioData) async {
    if (!_isInitialized) {
      await initialize();
    }

    try {
      print("Audio data length: ${audioData.length}");
      
      // Konwersja z int16 na znormalizowane float32 w zakresie [-1, 1]
      final Float32List normalizedAudio = Float32List(audioData.length);
      for (int i = 0; i < audioData.length; i++) {
        // Dzielimy przez 32768.0 aby znormalizować do zakresu [-1, 1]
        normalizedAudio[i] = audioData[i] / 32768.0;
      }
      
      // Przygotuj tensor dla ONNX - kształt to [1, długość_audio]
      final inputShape = [1, audioData.length];
      final inputOrt = OrtValueTensor.createTensorWithDataList(normalizedAudio, inputShape);


      // Nazwa inputu zgodna z nazwą w modelu: 'audio_signal'
      final inputs = {'audio_signal': inputOrt};
      final runOptions = OrtRunOptions();

      // Uruchomienie wnioskowania
      final outputs = await _session?.runAsync(runOptions, inputs);
      if (outputs == null) {
        throw Exception('Brak wyników wnioskowania');
      }

      final outputTensor = outputs[0];

      final outputValues = outputTensor?.value as List<dynamic>;
      print('Kształt wyjścia: ${outputValues.length} x ${outputValues[0].length} x ${outputValues[0][0].length}');

      // Przetwarzanie przewidywań dla każdego kroku czasowego
      List<int> predictions = [];

      // Dla każdego kroku czasowego znajdź klasę z najwyższą wartością
      for (int t = 0; t < outputValues[0].length; t++) {
        final List<dynamic> logits = outputValues[0][t];
        
        int maxClassIndex = 0;
        double maxLogit = logits[0] as double;
        
        for (int c = 1; c < logits.length; c++) {
          final double value = logits[c] as double;
          if (value > maxLogit) {
            maxLogit = value;
            maxClassIndex = c;
          }
        }
        
        predictions.add(maxClassIndex);
      }

      print('Predykcje dla poszczególnych kroków czasowych: $predictions');

      // Wybierz najczęściej występującą klasę
      Map<int, int> classCounts = {};
      for (final prediction in predictions) {
        classCounts[prediction] = (classCounts[prediction] ?? 0) + 1;
      }

      int maxCount = 0;
      int mostFrequentClass = 2; // Domyślnie silence

      classCounts.forEach((cls, count) {
        if (count > maxCount) {
          maxCount = count;
          mostFrequentClass = cls;
        }
      });

      print('Liczniki klas: $classCounts');
      print('Najczęstsza klasa: $mostFrequentClass (${_indexToBreathPhase(mostFrequentClass)})');

      // Zwolnij zasoby
      inputOrt.release();
      runOptions.release();
      for (var tensor in outputs) {
        tensor?.release();
      }

      return _indexToBreathPhase(mostFrequentClass);
    } catch (e) {
      print('Błąd podczas klasyfikacji: $e');
      return BreathPhase.silence;
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
    if (_session != null) {
      _session!.release();
      _session = null;
    }
    _isInitialized = false;
  }
}