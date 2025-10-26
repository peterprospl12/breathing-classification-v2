import 'dart:async';
import 'package:breathing_app/enums/enums.dart';

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

  final Stopwatch _stopwatch = Stopwatch();
  Timer? _periodicTimer;
  Duration _currentDuration = Duration.zero;

  final _durationController = StreamController<Duration>.broadcast();
  Stream<Duration> get durationStream => _durationController.stream;

  final _tempoController = StreamController<double>.broadcast();
  Stream<double> get tempoStream => _tempoController.stream;

  BreathTrackingService({int maxHistorySeconds = 5, double refreshTime = 0.3})
    : _maxHistorySeconds = maxHistorySeconds,
      _refreshTime = refreshTime {
    _logger.fine(
      'BreathTrackingService initialized with maxHistorySeconds=$maxHistorySeconds, refreshTime=$refreshTime',
    );
    _durationController.add(_currentDuration);
    _tempoController.add(0.0);
  }

  void _calculateAndNotifyTempo() {
    final seconds = _currentDuration.inSeconds;
    if (seconds > 0 && _exhaleCount > 0) {
      final tempo = (_exhaleCount / seconds) * 60.0;
      _tempoController.add(tempo);
      _logger.fine('Calculated tempo: $tempo BPM');
    } else {
      _tempoController.add(0.0);
      _logger.fine('No valid tempo to calculate, setting to 0.0');
    }
  }

  void startTimer() {
    if (!_stopwatch.isRunning) {
      _stopwatch.start();
      _periodicTimer?.cancel();
      _periodicTimer = Timer.periodic(const Duration(milliseconds: 100), (
        timer,
      ) {
        _currentDuration = _stopwatch.elapsed;
        _durationController.add(_currentDuration);
        _calculateAndNotifyTempo();
      });
      _logger.info(
        'Timer started/resumed. Current duration: $_currentDuration',
      );
    }
  }

  void pauseTimer() {
    if (_stopwatch.isRunning) {
      _stopwatch.stop();
      _periodicTimer?.cancel();
      _logger.info('Timer paused. Current duration: $_currentDuration');
    }
  }

  void resetTimer() {
    _stopwatch.stop();
    _stopwatch.reset();
    _periodicTimer?.cancel();
    _currentDuration = Duration.zero;
    _durationController.add(_currentDuration);
    resetCounters();
    _calculateAndNotifyTempo();
    _logger.info('Timer reset.');
  }

  BreathPhase _previousPhase = BreathPhase.silence;

  void addBreathPhase(BreathPhase phase) {
    BreathPhase tmp = phase;
    if (phase != _previousPhase) {
      phase = _previousPhase;
    }

    BreathPhase lastPhase =
        _breathPhases.isNotEmpty ? _breathPhases.last : BreathPhase.silence;
    _breathPhases.add(phase);
    if (_breathPhases.length > maxPhaseHistory) {
      _breathPhases.removeAt(0);
    }

    bool countChanged = false;
    if (phase == BreathPhase.inhale && lastPhase != BreathPhase.inhale) {
      _inhaleCount++;
      countChanged = true;
      _logger.fine('Inhale detected, count: $_inhaleCount');
    } else if (phase == BreathPhase.exhale && lastPhase != BreathPhase.exhale) {
      _exhaleCount++;
      _logger.fine('Exhale detected, count: $_exhaleCount');
    }

    _breathPhasesController.add(phase);
    _previousPhase = tmp;

    if (countChanged && _stopwatch.isRunning) {
      _calculateAndNotifyTempo();
    }
  }

  void resetCounters() {
    _inhaleCount = 0;
    _exhaleCount = 0;
    _logger.info('Breath counters reset');
    _calculateAndNotifyTempo();
  }

  void clearHistory() {
    _breathPhases.clear();
    _logger.info('Breath history cleared');
  }

  void dispose() {
    _breathPhasesController.close();
    _durationController.close();
    _tempoController.close();
    _periodicTimer?.cancel();
    _stopwatch.stop();
    _logger.info('BreathTrackingService disposed');
  }
}
