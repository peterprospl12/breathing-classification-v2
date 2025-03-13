import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:async';
import '../services/audio_service.dart';
import '../theme/app_theme.dart';

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
    // Setup timer to update scroll position
    _animationTimer = Timer.periodic(const Duration(milliseconds: 50), (timer) {
      setState(() {
        // Increase scroll position for rightward movement effect
        _scrollPosition += 2.0;
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
  
  MicrophonePainter({
    required this.pcmSamples,
    required this.isRecording,
    required this.scrollPosition,
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

    // Draw real microphone data with scrolling effect
    _drawScrollingMicrophoneData(canvas, size);
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

  void _drawScrollingMicrophoneData(Canvas canvas, Size size) {
    if (pcmSamples.isEmpty) return;
    
    final double width = size.width;
    final double height = size.height;
    final double centerY = height / 2;
    
    // Find max value for normalization
    final int maxPcmValue = 32767; // Maximum value for 16-bit PCM
    
    // Prepare the path for waveform
    final Path path = Path();
    final Paint linePaint = Paint()
      ..color = AppTheme.primaryColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round;

    // Calculate sample spacing - make it narrower for smoother display
    final double sampleSpacing = width / 200; 
    
    // Draw from newest (right) to oldest (left)
    // For scrolling effect, we offset each sample based on the scroll position
    bool pathStarted = false;
    
    for (int i = 0; i < pcmSamples.length; i++) {
      // Calculate position
      final double x = width - (i * sampleSpacing + scrollPosition % sampleSpacing);
      
      // Skip points that are off-screen
      if (x < 0) break;
      if (x > width) continue;
      
      final int sample = pcmSamples[pcmSamples.length - 1 - i];
      
      // Normalize to [-1, 1] range
      final double normalizedValue = sample / maxPcmValue;
      
      // Calculate y position - scale to 80% of half height
      final double y = centerY - normalizedValue * (height / 2) * 0.8;
      
      // Draw the waveform
      if (!pathStarted) {
        path.moveTo(x, y);
        pathStarted = true;
      } else {
        path.lineTo(x, y);
      }
      
      // Draw points only for some samples (not all, to avoid clutter)
      if (i % 5 == 0) {
        canvas.drawCircle(
          Offset(x, y),
          2.0,
          Paint()..color = normalizedValue >= 0 ? Colors.blue : Colors.red,
        );
      }
    }
    
    // Draw the waveform path
    canvas.drawPath(path, linePaint);
    
    // Draw "now" marker at the right edge
    final Paint nowMarkerPaint = Paint()
      ..color = Colors.white
      ..strokeWidth = 2;
    
    canvas.drawLine(
      Offset(width - 5, 0),
      Offset(width - 5, height),
      nowMarkerPaint,
    );
    
    // Draw informational overlay
    _drawInfoOverlay(canvas, size);
  }
  
  void _drawInfoOverlay(Canvas canvas, Size size) {
    // Draw amplitude information
    final textSpan = TextSpan(
      text: 'Live Microphone Data - Scrolling Waveform',
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
  }

  @override
  bool shouldRepaint(covariant MicrophonePainter oldDelegate) {
    return true; // Always repaint for smooth animation
  }
}
