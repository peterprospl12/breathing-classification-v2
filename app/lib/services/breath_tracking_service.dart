import 'dart:async';
import '../models/breath_classifier.dart';
import 'package:logging/logging.dart';
import 'package:breathing_app/utils/logger.dart';

class BreathTrackingService {
  final Logger _logger = LoggerService.getLogger('BreathTrackingService');

  final List<BreathPhase> _breathPhases = [BreathPhase.inhale, BreathPhase.silence];
  final _breathPhasesController = StreamController<BreathPhase>.broadcast();
  Stream<BreathPhase> get breathPhasesStream => _breathPhasesController.stream;

  int _inhaleCount = 0;
  int _exhaleCount = 0;
  int get inhaleCount => _inhaleCount;
  int get exhaleCount => _exhaleCount;

  final int _maxHistorySeconds;
  final double _refreshTime;
  int maxPhaseHistory = 3;


  BreathTrackingService({
    int maxHistorySeconds = 5,
    double refreshTime = 0.3,
  }) :
    _maxHistorySeconds = maxHistorySeconds,
    _refreshTime = refreshTime {
      _logger.fine('BreathTrackingService initialized with maxHistorySeconds=$maxHistorySeconds, refreshTime=$refreshTime');
    }

  void addBreathPhase(BreathPhase phase) {
    BreathPhase lastPhase = _breathPhases.isNotEmpty ? _breathPhases.last : BreathPhase.silence;

    _breathPhases.add(phase);
    if (_breathPhases.length > maxPhaseHistory) {
      _breathPhases.removeAt(0);
    }

    // Determine most frequent phase in history
    Map<BreathPhase, int> phaseCounts = {};
    for (var p in _breathPhases) {
      phaseCounts[p] = (phaseCounts[p] ?? 0) + 1;
    }

    int maxCount = 0;
    BreathPhase mostFrequentPhase = phase;

    phaseCounts.forEach((p, count) {
      if (count > maxCount) {
        maxCount = count;
        mostFrequentPhase = p;
      }
    });

    // Check if there's a tie
    bool equalDistribution = true;
    int firstCount = phaseCounts.values.first;
    for (int count in phaseCounts.values) {
      if (count != firstCount) {
        equalDistribution = false;
        break;
      }
    }

    // If equal distribution (33% chance for each), use the last phase
    if (equalDistribution && phaseCounts.length > 1) {
      mostFrequentPhase = _breathPhases.last;
      _logger.fine('Equal distribution detected, using last phase: $mostFrequentPhase');
    } else {
      _logger.fine('Most frequent phase detected: $mostFrequentPhase');
    }

    // Assign the determined phase
    phase = mostFrequentPhase;

    if (phase == BreathPhase.inhale && lastPhase != BreathPhase.inhale) {
      _inhaleCount++;
      _logger.fine('Inhale detected, count: $_inhaleCount');
    } else if (phase == BreathPhase.exhale && lastPhase != BreathPhase.exhale) {
      _exhaleCount++;
      _logger.fine('Exhale detected, count: $_exhaleCount');
    }

    _breathPhasesController.add(phase);
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
