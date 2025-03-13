import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:async';
import 'dart:math' as math;
import '../services/audio_service.dart';
import '../theme/app_theme.dart';
import '../models/breath_classifier.dart';

class MicrophoneVisualizationWidget extends StatefulWidget {
  const MicrophoneVisualizationWidget({Key? key}) : super(key: key);

  @override
  State<MicrophoneVisualizationWidget> createState() => _MicrophoneVisualizationWidgetState();
}

class _MicrophoneVisualizationWidgetState extends State<MicrophoneVisualizationWidget> {
  // Animation controller for auto-scrolling effect
  Timer? _animationTimer;
  double _scrollPosition = 0.0;
  
  @override
  void initState() {
    super.initState();
    // Setup timer to update scroll position - match Python's refresh rate
    _animationTimer = Timer.periodic(const Duration(milliseconds: 30), (timer) {
      setState(() {
        // Use smoother scrolling with smaller increments
        _scrollPosition += 1.0;
        if (_scrollPosition > 1000) {
          _scrollPosition = 0.0; // Reset to avoid overflow
        }
      });
    });
  }
  
  @override
  void dispose() {
    _animationTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<AudioService>(
      builder: (context, audioService, child) {
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
              painter: MicrophonePainter(
                pcmSamples: audioService.microphoneBuffer,
                isRecording: audioService.isRecording,
                scrollPosition: _scrollPosition,
                breathPhases: audioService.breathPhases,
              ),
              size: Size.infinite,
            ),
          ),
        );
      },
    );
  }
}

class MicrophonePainter extends CustomPainter {
  final List<int> pcmSamples;
  final bool isRecording;
  final double scrollPosition;
  final List<BreathPhase> breathPhases;
  
  MicrophonePainter({
    required this.pcmSamples,
    required this.isRecording,
    required this.scrollPosition,
    this.breathPhases = const [],
  });

  @override
  void paint(Canvas canvas, Size size) {
    final double width = size.width;
    final double height = size.height;
    final double centerY = height / 2;

    // Clear the background
    final Paint backgroundPaint = Paint()
      ..color = Colors.black;
    canvas.drawRect(Rect.fromLTWH(0, 0, width, height), backgroundPaint);

    // Draw grid lines
    _drawGrid(canvas, size);
    
    // Draw time markers
    _drawTimeMarkers(canvas, size);
    
    if (!isRecording && pcmSamples.isEmpty) {
      _drawNotRecordingMessage(canvas, size);
      return;
    }
    
    if (pcmSamples.isEmpty) {
      _drawWaitingForDataMessage(canvas, size);
      return;
    }

    // Draw real microphone data with Python-like visualization
    _drawPythonStyleVisualization(canvas, size);
  }

  void _drawGrid(Canvas canvas, Size size) {
    final double width = size.width;
    final double height = size.height;
    
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
    
    // Vertical grid lines with scrolling effect
    final double gridSpacing = width / 10;
    final double offset = scrollPosition % gridSpacing;
    
    for (double i = -offset; i <= width; i += gridSpacing) {
      canvas.drawLine(
        Offset(i, 0),
        Offset(i, height),
        gridPaint,
      );
    }
  }
  
  void _drawTimeMarkers(Canvas canvas, Size size) {
    final double width = size.width;
    final double height = size.height;
    
    // Draw time markers at the bottom
    final textStyle = const TextStyle(
      color: Colors.grey,
      fontSize: 10,
    );
    
    final double gridSpacing = width / 10;
    final double offset = scrollPosition % gridSpacing;
    
    // Draw time markers (current time at right, older times at left)
    for (int i = 0; i < 11; i++) {
      final double x = width - (i * gridSpacing + offset);
      if (x < 0) continue;
      
      final String timeLabel = "${i * 0.5}s";
      final textSpan = TextSpan(text: timeLabel, style: textStyle);
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      
      textPainter.layout(minWidth: 0, maxWidth: gridSpacing);
      textPainter.paint(
        canvas, 
        Offset(x - textPainter.width / 2, height - 15),
      );
    }
  }

  void _drawNotRecordingMessage(Canvas canvas, Size size) {
    final textSpan = TextSpan(
      text: 'Start recording to visualize microphone data',
      style: const TextStyle(
        color: Colors.white70,
        fontSize: 16,
      ),
    );
    
    _drawCenteredText(canvas, size, textSpan);
  }

  void _drawWaitingForDataMessage(Canvas canvas, Size size) {
    final textSpan = TextSpan(
      text: 'Waiting for microphone data...',
      style: const TextStyle(
        color: Colors.white70,
        fontSize: 16,
      ),
    );
    
    _drawCenteredText(canvas, size, textSpan);
  }

  void _drawCenteredText(Canvas canvas, Size size, TextSpan textSpan) {
    final textPainter = TextPainter(
      text: textSpan,
      textDirection: TextDirection.ltr,
      textAlign: TextAlign.center,
    );
    
    textPainter.layout(maxWidth: size.width * 0.8);
    
    final offset = Offset(
      (size.width - textPainter.width) / 2,
      (size.height - textPainter.height) / 2,
    );
    
    textPainter.paint(canvas, offset);
  }

  void _drawPythonStyleVisualization(Canvas canvas, Size size) {
    if (pcmSamples.isEmpty) return;
    
    final double width = size.width;
    final double height = size.height;
    final double centerY = height / 2;
    
    // Max value for normalization
    final int maxPcmValue = 32767; // Maximum value for 16-bit PCM
    
    // Calculate display time window (5 seconds like in Python)
    final int displayTimeSeconds = 5;
    final int totalSamples = pcmSamples.length;
    
    // Calculate total time represented by our buffer
    final double totalBufferTimeSeconds = totalSamples / AudioService.sampleRate;
    
    // Calculate refresh time chunks similar to Python code
    final double refreshTime = 0.3; // Same as Python
    final int samplesPerRefreshTime = (AudioService.sampleRate * refreshTime).round();
    
    // How many refresh-time segments we can fit
    final int numSegments = (totalSamples / samplesPerRefreshTime).ceil();
    
    // Ensure we have at least one breath phase if available
    final int numPhases = math.min(numSegments, breathPhases.length);
    
    // Draw "now" marker at the right edge
    final Paint nowMarkerPaint = Paint()
      ..color = Colors.white
      ..strokeWidth = 2;
    
    canvas.drawLine(
      Offset(width - 5, 0),
      Offset(width - 5, height),
      nowMarkerPaint,
    );
    
    // Similar to Python's plot, draw each segment with its own color
    for (int segment = 0; segment < numSegments; segment++) {
      // Get the color for this segment - use the most recent phases first
      Color segmentColor;
      if (segment < numPhases) {
        int phaseIndex = breathPhases.length - segment - 1;
        if (phaseIndex >= 0 && phaseIndex < breathPhases.length) {
          segmentColor = _getColorForPhase(breathPhases[phaseIndex]);
        } else {
          segmentColor = AppTheme.primaryColor;
        }
      } else {
        segmentColor = AppTheme.primaryColor;
      }
      
      // Calculate segment boundaries in the buffer
      int endIdx = pcmSamples.length - (segment * samplesPerRefreshTime);
      int startIdx = math.max(0, endIdx - samplesPerRefreshTime);
      
      if (startIdx >= endIdx || startIdx >= pcmSamples.length) continue;
      
      // Prepare the path for this segment's waveform
      final Path segmentPath = Path();
      bool pathStarted = false;
      
      // Calculate position on screen - more recent data on the right
      final double segmentWidth = width * refreshTime / displayTimeSeconds;
      final double rightEdge = width - (segment * segmentWidth) - scrollPosition % segmentWidth;
      final double leftEdge = rightEdge - segmentWidth;
      
      // Calculate how many points to skip for efficient rendering
      int pointsToSkip = math.max(1, (endIdx - startIdx) ~/ 200);
      
      // Draw samples for this segment, using downsampling for efficiency
      for (int i = startIdx; i < endIdx; i += pointsToSkip) {
        if (i >= pcmSamples.length) break;
        
        // Calculate x position - map the sample index to screen space
        final double progress = (i - startIdx) / (endIdx - startIdx);
        final double x = leftEdge + progress * segmentWidth;
        
        // Skip points that are off-screen
        if (x < 0) continue;
        if (x > width) break;
        
        final int sample = pcmSamples[i];
        
        // Normalize to [-1, 1] range
        final double normalizedValue = sample / maxPcmValue;
        
        // Use a non-linear mapping for better visualization of small values
        // Similar to how Python's visualization appears more sensitive
        double amplification = 0.8;
        double y;
        if (normalizedValue > 0) {
          y = centerY - math.pow(normalizedValue, 0.7) * (height / 2) * amplification;
        } else {
          y = centerY - (-math.pow(-normalizedValue, 0.7)) * (height / 2) * amplification;
        }
        
        // Draw the waveform
        if (!pathStarted) {
          segmentPath.moveTo(x, y);
          pathStarted = true;
        } else {
          segmentPath.lineTo(x, y);
        }
      }
      
      // Draw the segment path with appropriate color
      final Paint segmentPaint = Paint()
        ..color = segmentColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..strokeCap = StrokeCap.round;
      
      canvas.drawPath(segmentPath, segmentPaint);
    }
    
    // Draw informational overlay
    _drawInfoOverlay(canvas, size);
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
  
  void _drawInfoOverlay(Canvas canvas, Size size) {
    // Draw amplitude information
    final textSpan = TextSpan(
      text: 'Live Microphone Data - Breath-Colored Waveform',
      style: const TextStyle(
        color: Colors.white,
        fontSize: 12,
        fontWeight: FontWeight.bold,
      ),
    );
    
    final textPainter = TextPainter(
      text: textSpan,
      textDirection: TextDirection.ltr,
    );
    
    textPainter.layout();
    textPainter.paint(canvas, const Offset(10, 10));
    
    // Add legend for "now"
    final nowLegendSpan = TextSpan(
      text: 'Now →',
      style: const TextStyle(
        color: Colors.white,
        fontSize: 10,
      ),
    );
    
    final nowTextPainter = TextPainter(
      text: nowLegendSpan,
      textDirection: TextDirection.ltr,
    );
    
    nowTextPainter.layout();
    nowTextPainter.paint(
      canvas, 
      Offset(size.width - nowTextPainter.width - 15, 10),
    );
    
    // Draw breath phase legend
    _drawBreathPhaseLegend(canvas, size);
  }
  
  void _drawBreathPhaseLegend(Canvas canvas, Size size) {
    const double legendY = 30;
    double xOffset = 10;
    
    // Draw inhale legend
    _drawLegendItem(canvas, Offset(xOffset, legendY), AppTheme.inhaleColor, 'Inhale');
    xOffset += 70;
    
    // Draw exhale legend
    _drawLegendItem(canvas, Offset(xOffset, legendY), AppTheme.exhaleColor, 'Exhale');
    xOffset += 70;
    
    // Draw silence legend
    _drawLegendItem(canvas, Offset(xOffset, legendY), AppTheme.silenceColor, 'Silence');
  }
  
  void _drawLegendItem(Canvas canvas, Offset position, Color color, String label) {
    // Draw color indicator
    final Paint circlePaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    
    canvas.drawCircle(
      Offset(position.dx + 5, position.dy + 5),
      5,
      circlePaint,
    );
    
    // Draw label
    final textSpan = TextSpan(
      text: label,
      style: const TextStyle(
        color: Colors.white,
        fontSize: 10,
      ),
    );
    
    final textPainter = TextPainter(
      text: textSpan,
      textDirection: TextDirection.ltr,
    );
    
    textPainter.layout();
    textPainter.paint(
      canvas, 
      Offset(position.dx + 15, position.dy),
    );
  }

  @override
  bool shouldRepaint(covariant MicrophonePainter oldDelegate) {
    return true; // Always repaint for smooth animation
  }
}
