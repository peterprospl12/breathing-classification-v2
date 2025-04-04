import 'dart:async';
import '../models/breath_classifier.dart';

class BreathTrackingService {
  // Breath tracking
  final List<BreathPhase> _breathPhases = [];
  final _breathPhasesController = StreamController<BreathPhase>.broadcast();
  Stream<BreathPhase> get breathPhasesStream => _breathPhasesController.stream;
  
  int _inhaleCount = 0;
  int _exhaleCount = 0;
  int get inhaleCount => _inhaleCount;
  int get exhaleCount => _exhaleCount;
  
  // History configuration
  final int _maxHistorySeconds;
  final double _refreshTime;
  int get maxPhaseHistory => (_maxHistorySeconds / _refreshTime).round();

  BreathTrackingService({
    int maxHistorySeconds = 5,
    double refreshTime = 0.3,
  }) : 
    _maxHistorySeconds = maxHistorySeconds,
    _refreshTime = refreshTime;

  void addBreathPhase(BreathPhase phase) {
    BreathPhase lastPhase = _breathPhases.isNotEmpty ? _breathPhases.last : BreathPhase.silence;
    _breathPhases.add(phase);
    if (_breathPhases.length > maxPhaseHistory) {
      _breathPhases.removeAt(0);
    }
    
    if (phase == BreathPhase.inhale && lastPhase != BreathPhase.inhale) {
      _inhaleCount++;
    } else if (phase == BreathPhase.exhale && lastPhase != BreathPhase.exhale) {
      _exhaleCount++;
    }
    
    _breathPhasesController.add(phase);
  }

  void resetCounters() {
    _inhaleCount = 0;
    _exhaleCount = 0;
  }

  void clearHistory() {
    _breathPhases.clear();
  }

  void dispose() {
    _breathPhasesController.close();
  }
}
