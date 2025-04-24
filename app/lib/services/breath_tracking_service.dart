import 'dart:async';
import '../models/breath_classifier.dart';
import 'package:logging/logging.dart';
import 'package:breathing_app/utils/logger.dart';

class BreathTrackingService {
  final Logger _logger = LoggerService.getLogger('BreathTrackingService');

  final List<BreathPhase> _breathPhases = [];
  final _breathPhasesController = StreamController<BreathPhase>.broadcast();
  Stream<BreathPhase> get breathPhasesStream => _breathPhasesController.stream;

  int _inhaleCount = 0;
  int _exhaleCount = 0;
  int get inhaleCount => _inhaleCount;
  int get exhaleCount => _exhaleCount;

  final int _maxHistorySeconds;
  final double _refreshTime;
  int get maxPhaseHistory => (_maxHistorySeconds / _refreshTime).round();

  BreathTrackingService({
    int maxHistorySeconds = 5,
    double refreshTime = 0.3,
  }) :
    _maxHistorySeconds = maxHistorySeconds,
    _refreshTime = refreshTime {
      _logger.fine('BreathTrackingService initialized with maxHistorySeconds=$maxHistorySeconds, refreshTime=$refreshTime');
    }

  BreathPhase _previousPhase = BreathPhase.silence;

  void addBreathPhase(BreathPhase phase) {
    BreathPhase tmp = phase;
    if (phase != _previousPhase) {
      phase = _previousPhase;
    }
    
    BreathPhase lastPhase = _breathPhases.isNotEmpty ? _breathPhases.last : BreathPhase.silence;
    _breathPhases.add(phase);
    if (_breathPhases.length > maxPhaseHistory) {
      _breathPhases.removeAt(0);
    }

    if (phase == BreathPhase.inhale && lastPhase != BreathPhase.inhale) {
      _inhaleCount++;
      _logger.fine('Inhale detected, count: $_inhaleCount');
    } else if (phase == BreathPhase.exhale && lastPhase != BreathPhase.exhale) {
      _exhaleCount++;
      _logger.fine('Exhale detected, count: $_exhaleCount');
    }

    _breathPhasesController.add(phase);

    _previousPhase = tmp;
  }

  void resetCounters() {
    _inhaleCount = 0;
    _exhaleCount = 0;
    _logger.info('Breath counters reset');
  }

  void clearHistory() {
    _breathPhases.clear();
    _logger.info('Breath history cleared');
  }

  void dispose() {
    _breathPhasesController.close();
    _logger.info('BreathTrackingService disposed');
  }
}
