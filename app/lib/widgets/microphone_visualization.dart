import 'dart:math' as math;
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/audio_service.dart';
import '../models/breath_classifier.dart';

class MicrophoneVisualizationWidget extends StatefulWidget {
  const MicrophoneVisualizationWidget({super.key});

  @override
  State<MicrophoneVisualizationWidget> createState() => _MicrophoneVisualizationWidgetState();
}

class _MicrophoneVisualizationWidgetState extends State<MicrophoneVisualizationWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  StreamSubscription<List<int>>? _audioSubscription;
  StreamSubscription<BreathPhase>? _breathPhaseSubscription;
  List<int> _audioData = [];
  final List<BreathPhase> _breathPhases = [];

  // Maximum number of breath phases to store
  static const int _maxBreathPhasesToStore = 20;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300), // how often to repaint
    );
    _animationController.repeat();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      _subscribeToStreams();
    });
  }

  void _subscribeToStreams() {
    final audioService = Provider.of<AudioService>(context, listen: false);

    // Subscribe to audio stream
    _audioSubscription = audioService.subscribeToAudioStream((audioData) {
      setState(() {
        _audioData = audioData;
      });
    });

    // Subscribe to breath phases stream
    _breathPhaseSubscription = audioService.breathPhasesStream.listen((phase) {
      setState(() {
        _breathPhases.add(phase);
        if (_breathPhases.length > _maxBreathPhasesToStore) {
          _breathPhases.removeAt(0);
        }
      });
    });
  }

  @override
  void dispose() {
    _animationController.dispose();
    _audioSubscription?.cancel();
    _breathPhaseSubscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Consumer<AudioService>(
      builder: (context, audioService, child) {
        return Container(
          height: 200, // Adjusted height
          width: double.infinity,
          decoration: BoxDecoration(
            color: theme.colorScheme.surfaceVariant.withOpacity(0.5), // Use a subtle background from theme
            borderRadius: BorderRadius.circular(12), // Consistent rounding
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: AnimatedBuilder(
              animation: _animationController,
              builder: (context, child) {
                return CustomPaint(
                  painter: MicrophoneWaveformPainter(
                    audioBuffer: _audioData,
                    breathPhases: _breathPhases,
                    isRecording: audioService.isRecording,
                    theme: theme, // Pass theme data to painter
                  ),
                  size: Size.infinite,
                );
              },
            ),
          ),
        );
      },
    );
  }
}

class MicrophoneWaveformPainter extends CustomPainter {
  final List<int> audioBuffer;
  final List<BreathPhase> breathPhases;
  final bool isRecording;
  final ThemeData theme; // Receive theme data

  static List<double> _smoothedValues = [];
  static const double _smoothingFactor = 0.2; // Lower means more smoothing

  MicrophoneWaveformPainter({
    required this.audioBuffer,
    required this.breathPhases,
    required this.isRecording,
    required this.theme, // Add theme to constructor
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Use theme colors
    final backgroundPaint = Paint()..color = theme.colorScheme.surfaceVariant.withOpacity(0.5);
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), backgroundPaint);

    _drawBreathPhases(canvas, size);
    _drawGridLines(canvas, size);

    if (audioBuffer.isEmpty && !isRecording) {
      _drawIdleWaveform(canvas, size);
      return;
    }

    // Use theme primary color for the waveform
    final paint = Paint()
      ..color = theme.colorScheme.primary
      ..strokeWidth = 1.5 // Slightly thinner line
      ..style = PaintingStyle.stroke;

    final path = Path();
    final displayPoints = size.width.toInt();
    final step = math.max(1, audioBuffer.length ~/ displayPoints);
    const maxAmplitude = 1024;
    final heightScale = size.height / 1 / maxAmplitude;
    final yCenter = size.height / 2;

    if (_smoothedValues.length != displayPoints) {
      _smoothedValues = List<double>.filled(displayPoints, yCenter);
    }

    if (audioBuffer.length >= step || isRecording) { // Ensure drawing continues even if buffer temporarily empties while recording
      path.moveTo(0, _smoothedValues[0]);

      if (isRecording) {
        for (int i = 0; i < displayPoints - 1; i++) {
          _smoothedValues[i] = _smoothedValues[i + 1];
          path.lineTo(i.toDouble(), _smoothedValues[i]);
        }

        if (audioBuffer.isNotEmpty) {
          final latestSampleIndex = audioBuffer.length - 1;
          final latestSample = audioBuffer[latestSampleIndex];
          final targetY = yCenter - latestSample * heightScale;
          _smoothedValues[displayPoints - 1] = _smoothedValues[displayPoints - 2] * (1 - _smoothingFactor) +
                                              targetY * _smoothingFactor;
        } else {
           // If buffer is empty but still recording, keep the last value
           _smoothedValues[displayPoints - 1] = _smoothedValues[displayPoints - 2];
        }
      } else {
        // Freeze waveform when not recording
        for (int i = 0; i < displayPoints - 1; i++) {
          path.lineTo(i.toDouble(), _smoothedValues[i]);
        }
      }

      path.lineTo((displayPoints - 1).toDouble(), _smoothedValues[displayPoints - 1]);
      canvas.drawPath(path, paint);

    } else {
      _drawIdleWaveform(canvas, size);
    }

    // Draw center line using theme outline color
    final centerPaint = Paint()
      ..color = theme.colorScheme.outline.withOpacity(0.5)
      ..strokeWidth = 1.0;
    canvas.drawLine(
      Offset(0, size.height / 2),
      Offset(size.width, size.height / 2),
      centerPaint,
    );

    // Draw recording indicator using theme error color
    if (isRecording) {
      final indicatorPaint = Paint()
        ..color = theme.colorScheme.error
        ..style = PaintingStyle.fill;
      canvas.drawCircle(
        Offset(size.width - 12, 12), // Adjusted position
        5, // Smaller indicator
        indicatorPaint,
      );
    }
  }

  void _drawBreathPhases(Canvas canvas, Size size) {
    final int totalPhases = breathPhases.length;
    if (totalPhases <= 0) return;

    final double segmentWidth = size.width / totalPhases;

    for (int i = 0; i < totalPhases; i++) {
      final BreathPhase phase = breathPhases[i];
      // Use AppTheme colors directly with lower opacity
      final Color phaseColor = BreathClassifier.getColorForPhase(phase).withOpacity(0.15);
      final Paint phasePaint = Paint()
        ..color = phaseColor
        ..style = PaintingStyle.fill;

      canvas.drawRect(
        Rect.fromLTWH(i * segmentWidth, 0, segmentWidth, size.height),
        phasePaint,
      );
    }
  }

  void _drawGridLines(Canvas canvas, Size size) {
    // Use theme outline color for grid
    final Paint gridPaint = Paint()
      ..color = theme.colorScheme.outline.withOpacity(0.3)
      ..strokeWidth = 0.5; // Thinner grid lines

    // Horizontal grid lines
    int numLines = 4;
    for (int i = 1; i < numLines; i++) {
       double y = i * size.height / numLines;
       canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }
  }

  void _drawIdleWaveform(Canvas canvas, Size size) {
    // Use theme onSurfaceVariant color for idle wave
    final paint = Paint()
      ..color = theme.colorScheme.onSurfaceVariant.withOpacity(0.5)
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;

    final centerY = size.height / 2;
    final path = Path();

    path.moveTo(0, centerY);
    for (double x = 0; x < size.width; x += 2) { // Increase step for smoother idle wave
      final y = centerY + math.sin(x / 10) * 1.5; // Smaller amplitude
      path.lineTo(x, y);
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(MicrophoneWaveformPainter oldDelegate) {
    // Need to check theme change as well
    return oldDelegate.isRecording != isRecording ||
           (audioBuffer.isNotEmpty && oldDelegate.audioBuffer != audioBuffer) ||
           oldDelegate.breathPhases != breathPhases ||
           oldDelegate.theme != theme;
  }
}
