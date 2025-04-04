import 'dart:math' as math;
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/audio_service.dart';
import '../models/breath_classifier.dart';
import '../theme/app_theme.dart';

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
  List<BreathPhase> _breathPhases = [];
  
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
    
    // Subscribe to audio and breath phase streams after the first frame
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
        // Add the new phase and maintain fixed size
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
        return Column(
          children: [
            _buildControlPanel(audioService),
            const SizedBox(height: 8),
            _buildVisualization(audioService),
            const SizedBox(height: 8),
            _buildDebugSaveButton(audioService),
          ],
        );
      },
    );
  }

  Widget _buildControlPanel(AudioService audioService) {
    return Center(
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 8),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          width: 120, 
          height: 40, 
          decoration: BoxDecoration(
            color: audioService.isRecording ? Colors.red.shade600 : Colors.green.shade600,
            borderRadius: BorderRadius.circular(20),
            boxShadow: [
              BoxShadow(
                color: (audioService.isRecording ? Colors.red : Colors.green).withValues(alpha:0.3),
                spreadRadius: 1,
                blurRadius: 6,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: Material(
            color: Colors.transparent,
            child: InkWell(
              borderRadius: BorderRadius.circular(20),
              onTap: () {
                if (audioService.isRecording) {
                  audioService.stopRecording();
                } else {
                  audioService.startRecording();
                }
              },
              child: Center(
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      audioService.isRecording ? Icons.stop_rounded : Icons.mic,
                      color: Colors.white,
                      size: 22, 
                    ),
                    const SizedBox(width: 6),
                    Text(
                      audioService.isRecording ? 'Stop' : 'Start',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 16, 
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildVisualization(AudioService audioService) {
    return Container(
      height: 200,
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

  Widget _buildDebugSaveButton(AudioService audioService) {
    // Only show save button when there's data and we're not currently saving
    final bool hasData = audioService.audioBuffer.isNotEmpty;
    
    return AnimatedOpacity(
      opacity: hasData ? 1.0 : 0.3,
      duration: const Duration(milliseconds: 300),
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 8),
        child: ElevatedButton.icon(
          onPressed: hasData && !audioService.isSaving 
            ? () => _saveRecording(audioService) 
            : null,
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.blue.shade700,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(20),
            ),
          ),
          icon: audioService.isSaving 
            ? const SizedBox(
                width: 20, 
                height: 20, 
                child: CircularProgressIndicator(
                  strokeWidth: 2, 
                  color: Colors.white,
                )
              ) 
            : const Icon(Icons.save),
          label: Text(
            audioService.isSaving ? 'Saving...' : 'Save Debug Recording',
            style: const TextStyle(
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ),
    );
  }
  
  Future<void> _saveRecording(AudioService audioService) async {
    final filePath = await audioService.saveRecording();
    
    if (!mounted) return;
    
    if (filePath != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Recording saved to: $filePath'),
          backgroundColor: Colors.green.shade700,
          behavior: SnackBarBehavior.floating,
        ),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('Failed to save recording'),
          backgroundColor: Colors.red.shade700,
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
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
    // Draw background
    final backgroundPaint = Paint()
      ..color = Colors.black;
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), backgroundPaint);
    
    // Draw breath phase sections
    _drawBreathPhases(canvas, size);
    
    // Draw grid lines
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
    
    const maxAmplitude = 1024; // Reduced from 16384 to increase visibility
    final heightScale = size.height / 1 / maxAmplitude; // Increased from /6 to /3 for better visibility
    
    final yCenter = size.height / 2;
    
    // Initialize smoothed values if needed
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
        // When not recording, just draw the current buffer without shifting
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
      final Color phaseColor = _getColorForPhase(phase).withValues(alpha:0.2);
      final Paint phasePaint = Paint()
        ..color = phaseColor
        ..style = PaintingStyle.fill;
        
      canvas.drawRect(
        Rect.fromLTWH(i * segmentWidth, 0, segmentWidth, size.height),
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
      final y = centerY + math.sin(x / 8) * 3; // More gentle waves
      path.lineTo(x, y);
    }
    
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(MicrophoneWaveformPainter oldDelegate) {
    // Only repaint if recording state changed or buffer was updated
    return oldDelegate.isRecording != isRecording || 
           (audioBuffer.isNotEmpty && oldDelegate.audioBuffer != audioBuffer) ||
           oldDelegate.breathPhases != breathPhases;
  }
}
