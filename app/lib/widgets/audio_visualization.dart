import 'package:flutter/material.dart';
import '../models/breath_classifier.dart';
import '../theme/app_theme.dart';

class AudioVisualizationWidget extends StatelessWidget {
  final List<double> audioData;
  final List<BreathPhase> breathPhases;
  final int sampleRate;
  final double refreshTime;

  const AudioVisualizationWidget({
    super.key,
    required this.audioData,
    required this.breathPhases,
    this.sampleRate = 44100,
    this.refreshTime = 0.3,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 200,
      decoration: BoxDecoration(
        color: Theme.of(context).cardTheme.color,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            spreadRadius: 1,
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: CustomPaint(
          painter: WaveformPainter(
            audioData: audioData,
            breathPhases: breathPhases,
            sampleRate: sampleRate,
            refreshTime: refreshTime,
          ),
          size: Size.infinite,
        ),
      ),
    );
  }
}

class WaveformPainter extends CustomPainter {
  final List<double> audioData;
  final List<BreathPhase> breathPhases;
  final int sampleRate;
  final double refreshTime;

  WaveformPainter({
    required this.audioData,
    required this.breathPhases,
    required this.sampleRate,
    required this.refreshTime,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (audioData.isEmpty || breathPhases.isEmpty) return;

    final double width = size.width;
    final double height = size.height;
    final double centerY = height / 2;
    final double amplitudeScale = height / 3; 

    // Clear the background
    final Paint backgroundPaint = Paint()
      ..color = Colors.black;
    canvas.drawRect(Rect.fromLTWH(0, 0, width, height), backgroundPaint);

    // Calculate how many points we can display
    final int totalPoints = audioData.length;
    final double pointsPerPixel = totalPoints / width;
    
    // Draw grid lines
    final Paint gridPaint = Paint()
      ..color = Colors.grey.withOpacity(0.2)
      ..strokeWidth = 1;
    
    // Horizontal grid lines
    for (double i = 0; i <= height; i += height / 6) {
      canvas.drawLine(
        Offset(0, i),
        Offset(width, i),
        gridPaint,
      );
    }
    
    // Draw breath phase sections
    final int totalPhases = breathPhases.length;
    if (totalPhases > 0) {
      final double segmentWidth = width / totalPhases;
      
      for (int i = 0; i < totalPhases; i++) {
        final BreathPhase phase = breathPhases[i];
        final Color phaseColor = _getColorForPhase(phase).withOpacity(0.2);
        final Paint phasePaint = Paint()
          ..color = phaseColor
          ..style = PaintingStyle.fill;
          
        canvas.drawRect(
          Rect.fromLTWH(i * segmentWidth, 0, segmentWidth, height),
          phasePaint,
        );
        
        // Phase label
        final TextSpan span = TextSpan(
          text: _getLabelForPhase(phase),
          style: TextStyle(
            color: _getColorForPhase(phase),
            fontSize: 12,
            fontWeight: FontWeight.w500,
          ),
        );
        
        final TextPainter tp = TextPainter(
          text: span,
          textDirection: TextDirection.ltr,
        );
        
        tp.layout();
        tp.paint(
          canvas, 
          Offset(i * segmentWidth + 5, 5),
        );
      }
    }

    // Draw the waveform
    final Path path = Path();
    bool pathStarted = false;
    
    for (int i = 0; i < totalPoints; i += pointsPerPixel.ceil()) {
      if (i >= totalPoints) continue;
      
      final double x = (i / totalPoints) * width;
      final double y = centerY + audioData[i] * amplitudeScale;
      
      if (!pathStarted) {
        path.moveTo(x, y);
        pathStarted = true;
      } else {
        path.lineTo(x, y);
      }
    }
    
    // Draw the waveform path
    final Paint waveformPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..strokeCap = StrokeCap.round;
    
    canvas.drawPath(path, waveformPaint);
  }

  Color _getColorForPhase(BreathPhase phase) {
    switch (phase) {
      case BreathPhase.inhale:
        return AppTheme.inhaleColor;
      case BreathPhase.exhale:
        return AppTheme.exhaleColor;
      case BreathPhase.silence:
        return AppTheme.silenceColor;
    }
  }

  String _getLabelForPhase(BreathPhase phase) {
    switch (phase) {
      case BreathPhase.inhale:
        return 'Inhale';
      case BreathPhase.exhale:
        return 'Exhale';
      case BreathPhase.silence:
        return 'Silence';
    }
  }

  @override
  bool shouldRepaint(covariant WaveformPainter oldDelegate) {
    return true; // Always repaint for smooth animation
  }
}
