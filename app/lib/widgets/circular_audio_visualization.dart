import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/audio_service.dart';
import '../models/breath_classifier.dart';

class CircularVisualizationWidget extends StatefulWidget {
  const CircularVisualizationWidget({super.key});

  @override
  State<CircularVisualizationWidget> createState() => _CircularVisualizationWidgetState();
}

class _CircularVisualizationWidgetState extends State<CircularVisualizationWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  StreamSubscription<List<int>>? _audioSubscription;
  StreamSubscription<BreathPhase>? _breathPhaseSubscription;
  List<int> _audioData = [];
  BreathPhase _currentPhase = BreathPhase.silence;
  double _currentAmplitude = 0.0;

  // Smoothing factor for animations
  static const double _smoothingFactor = 0.2;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
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
      if (audioData.isNotEmpty) {
        setState(() {
          _audioData = audioData;
          // Calculate amplitude using RMS (Root Mean Square) for better audio visualization
          double sumSquares = 0.0;
          for (int sample in _audioData) {
            sumSquares += (sample * sample);
          }
          double rms = math.sqrt(sumSquares / _audioData.length);

          // Apply smoothing to the amplitude transition
          _currentAmplitude = _currentAmplitude * (1 - _smoothingFactor) +
                              (rms / 1024) * _smoothingFactor; // Scaled down for better visualization
        });
      }
    });

    // Subscribe to breath phases stream
    _breathPhaseSubscription = audioService.breathPhasesStream.listen((phase) {
      setState(() {
        _currentPhase = phase;
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
            const SizedBox(height: 16),
            _buildCircularVisualization(audioService),
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
                color: (audioService.isRecording ? Colors.red : Colors.green).withOpacity(0.3),
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

  Widget _buildCircularVisualization(AudioService audioService) {
    // Calculate the color based on current breath phase
    final Color phaseColor = BreathClassifier.getColorForPhase(_currentPhase);

    // Use sine wave effect for pulsating circle when audio is active
    final double pulseEffect = audioService.isRecording && _audioData.isNotEmpty ?
                              math.sin(_animationController.value * 2 * math.pi) * 0.05 + 1.0 : 1.0;

    // Calculate dynamic circle size based on audio amplitude with minimum size
    final double circleSize = 60 * math.max(0.3, (_currentAmplitude.clamp(0.2, 1.0) * pulseEffect));

    return Container(
      height: 150,
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
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Breath phase label
                  Text(
                    BreathClassifier.getLabelForPhase(_currentPhase),
                    style: TextStyle(
                      color: phaseColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 10),
                  // Animated circle
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 150),
                    width: circleSize,
                    height: circleSize,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: phaseColor.withOpacity(0.7),
                      boxShadow: [
                        BoxShadow(
                          color: phaseColor.withOpacity(0.5),
                          blurRadius: 10 * _currentAmplitude,
                          spreadRadius: 2 * _currentAmplitude,
                        ),
                      ],
                    ),
                    child: Center(
                      child: audioService.isRecording
                          ? _buildPulsatingDot(phaseColor)
                          : const Icon(
                              Icons.pause,
                              color: Colors.white,
                              size: 30,
                            ),
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildPulsatingDot(Color baseColor) {
    return TweenAnimationBuilder<double>(
      tween: Tween<double>(begin: 0.5, end: 1.0),
      duration: const Duration(milliseconds: 1000),
      builder: (context, value, child) {
        return Container(
          width: 12.0,
          height: 12.0,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: Colors.white.withOpacity(value),
          ),
        );
      },
      onEnd: () => setState(() {}), // Trigger rebuild for continuous animation
    );
  }
}