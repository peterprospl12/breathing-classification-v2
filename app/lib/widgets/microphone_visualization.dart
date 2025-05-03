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
    return Consumer<AudioService>(
      builder: (context, audioService, child) {
        return Card(
          margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          elevation: 3,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          child: Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(14),
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  Theme.of(context).cardColor,
                  Theme.of(context).cardColor.withValues(alpha: 0.9),
                ],
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: _buildVisualization(audioService),
            ),
          ),
        );
      },
    );
  }

  Widget _buildVisualization(AudioService audioService) {
    return Container(
      height: 250,
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(8),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: AnimatedBuilder(
          animation: _animationController,
          builder: (context, child) {
            return CustomPaint(
              painter: MicrophoneWaveformPainter(
                audioBuffer: _audioData,
                breathPhases: _breathPhases,
                isRecording: audioService.isRecording,
              ),
              size: Size.infinite,
            );
          },
        ),
      ),
    );
  }
}

class MicrophoneWaveformPainter extends CustomPainter {
  final List<int> audioBuffer;
  final List<BreathPhase> breathPhases;
  final bool isRecording;

  static List<double> _smoothedValues = [];
  static const double _smoothingFactor = 0.2; // Lower means more smoothing

  MicrophoneWaveformPainter({
    required this.audioBuffer,
    required this.breathPhases,
    required this.isRecording
  });

  @override
  void paint(Canvas canvas, Size size) {
    final backgroundPaint = Paint()
      ..color = Colors.black;
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), backgroundPaint);

    _drawBreathPhases(canvas, size);

    _drawGridLines(canvas, size);

    if (audioBuffer.isEmpty) {
      _drawIdleWaveform(canvas, size);
      return;
    }

    final paint = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 2.0
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

    if (audioBuffer.length >= step) {
      path.moveTo(0, _smoothedValues[0]);

      // Only shift values if currently recording - this freezes the waveform when stopped
      if (isRecording) {
        // Draw the waveform from left to right, shifting older data to the left
        for (int i = 0; i < displayPoints - 1; i++) {
          // Shift values left (this creates the scrolling effect)
          _smoothedValues[i] = _smoothedValues[i + 1];
          path.lineTo(i.toDouble(), _smoothedValues[i]);
        }

        // Add the newest data point at the rightmost position
        if (audioBuffer.isNotEmpty) {
          final latestSampleIndex = audioBuffer.length - 1;
          final latestSample = audioBuffer[latestSampleIndex];

          // Apply smoothing to reduce shakiness
          final targetY = yCenter - latestSample * heightScale;
          _smoothedValues[displayPoints - 1] = _smoothedValues[displayPoints - 2] * (1 - _smoothingFactor) +
                                              targetY * _smoothingFactor;
        }
      } else {
        for (int i = 0; i < displayPoints - 1; i++) {
          path.lineTo(i.toDouble(), _smoothedValues[i]);
        }
      }

      // Add the final point
      path.lineTo((displayPoints - 1).toDouble(), _smoothedValues[displayPoints - 1]);

      canvas.drawPath(path, paint);
    } else {
      _drawIdleWaveform(canvas, size);
    }

    // Draw center line
    final centerPaint = Paint()
      ..color = Colors.grey.withValues(alpha:0.5)
      ..strokeWidth = 0.5;
    canvas.drawLine(
      Offset(0, size.height / 2),
      Offset(size.width, size.height / 2),
      centerPaint,
    );

    // Draw recording indicator
    if (isRecording) {
      final indicatorPaint = Paint()
        ..color = Colors.red
        ..style = PaintingStyle.fill;
      canvas.drawCircle(
        Offset(size.width - 16, 16),
        8,
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
      final Color phaseColor = BreathClassifier.getColorForPhase(phase).withValues(alpha:0.2);
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
    final Paint gridPaint = Paint()
      ..color = Colors.grey.withValues(alpha: 0.2)
      ..strokeWidth = 1;

    // Horizontal grid lines
    for (double i = 0; i <= size.height; i += size.height / 6) {
      canvas.drawLine(
        Offset(0, i),
        Offset(size.width, i),
        gridPaint,
      );
    }
  }

  void _drawIdleWaveform(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.grey
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    final centerY = size.height / 2;
    final path = Path();

    // Draw a flat line with small sine waves to indicate idle state
    path.moveTo(0, centerY);
    for (double x = 0; x < size.width; x += 1) {
      final y = centerY + math.sin(x / 8) * 3;
      path.lineTo(x, y);
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(MicrophoneWaveformPainter oldDelegate) {
    return oldDelegate.isRecording != isRecording ||
           (audioBuffer.isNotEmpty && oldDelegate.audioBuffer != audioBuffer) ||
           oldDelegate.breathPhases != breathPhases;
  }
}
