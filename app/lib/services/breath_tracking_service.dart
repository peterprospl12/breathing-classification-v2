import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/breath_classifier.dart';

class BreathTrackingService {
  // Breath tracking
  final List<BreathPhase> _breathPhases = [];
  List<BreathPhase> get breathPhases => _breathPhases;
  
  int _inhaleCount = 0;
  int _exhaleCount = 0;
  int get inhaleCount => _inhaleCount;
  int get exhaleCount => _exhaleCount;
  
  // History configuration
  final int _maxHistorySeconds;
  final double _refreshTime;
  int get maxPhaseHistory => (_maxHistorySeconds / _refreshTime).round();
  
  // Events
  final _onBreathPhaseChangedController = StreamController<BreathPhase>.broadcast();
  Stream<BreathPhase> get onBreathPhaseChanged => _onBreathPhaseChangedController.stream;

  BreathTrackingService({
    int maxHistorySeconds = 5,
    double refreshTime = 0.3,
  }) : 
    _maxHistorySeconds = maxHistorySeconds,
    _refreshTime = refreshTime;

  void addBreathPhase(BreathPhase phase) {
    _breathPhases.add(phase);
    if (_breathPhases.length > maxPhaseHistory) {
      _breathPhases.removeAt(0);
    }
    
    if (phase == BreathPhase.inhale) {
      _inhaleCount++;
    } else if (phase == BreathPhase.exhale) {
      _exhaleCount++;
    }
    
    _onBreathPhaseChangedController.add(phase);
  }

  void resetCounters() {
    _inhaleCount = 0;
    _exhaleCount = 0;
  }

  void clearHistory() {
    _breathPhases.clear();
  }

  void dispose() {
    _onBreathPhaseChangedController.close();
  }
}
