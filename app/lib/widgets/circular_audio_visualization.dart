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

  // Constants for multi-layered circle
  static const int _maxLayers = 4;
  static const double _minCircleSize = 40.0; // Increased from original size
  static const double _maxCircleSize = 80.0;

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
                  Theme.of(context).cardColor.withOpacity(0.9),
                ],
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  _buildControlPanel(audioService),
                  const SizedBox(height: 16),
                  _buildCircularVisualization(audioService),
                ],
              ),
            ),
          ),
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

    // Calculate amplitude percentage (0.0 to 1.0)
    final double amplitudePercent = _currentAmplitude.clamp(0.2, 1.0) * pulseEffect;

    // Calculate number of visible layers based on amplitude
    final int visibleLayers = (amplitudePercent * _maxLayers).ceil().clamp(1, _maxLayers);

    // Calculate dynamic circle size
    final double baseCircleSize = _minCircleSize + (_maxCircleSize - _minCircleSize) * amplitudePercent;

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
                  // Multi-layered animated circles
                  Stack(
                    alignment: Alignment.center,
                    children: List.generate(visibleLayers, (index) {
                      // Reverse index for drawing from outside to inside
                      int layerIndex = visibleLayers - 1 - index;

                      // Calculate size for this layer
                      double layerSizePercent = 1.0 - (layerIndex * 0.18); // Each layer is 18% smaller
                      double layerSize = baseCircleSize * layerSizePercent;

                      // Calculate color shade - darker for outer circles
                      Color layerColor = _getLayerColor(phaseColor, layerIndex, visibleLayers);

                      return AnimatedContainer(
                        duration: const Duration(milliseconds: 150),
                        width: layerSize,
                        height: layerSize,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: layerColor,
                          border: Border.all(
                            color: layerColor.withOpacity(0.9),
                            width: 3.0,
                          ),
                        ),
                      );
                    }),
                  ),
                  // Center indicator
                  if (audioService.isRecording)
                    Padding(
                      padding: const EdgeInsets.only(top: 20),
                      child: _buildPulsatingDot(phaseColor),
                    )
                  else
                    Padding(
                      padding: const EdgeInsets.only(top: 20),
                      child: Icon(
                        Icons.pause,
                        color: phaseColor.withOpacity(0.8),
                        size: 24,
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

  // Calculate color for layer based on its index
  // Outer layers (lower index) are darker, inner layers (higher index) are lighter
  Color _getLayerColor(Color baseColor, int layerIndex, int totalLayers) {
    // For single layer, use the base color with medium opacity
    if (totalLayers == 1) return baseColor.withOpacity(0.6);

    // Normalize layer index to 0.0-1.0 scale
    // 0.0 = darkest (outer layer), 1.0 = lightest (inner layer)
    double brightnessPercent = layerIndex / (totalLayers - 1);

    // Convert to HSL for better brightness control
    HSLColor hslColor = HSLColor.fromColor(baseColor);

    // Adjust lightness - outer circles are darker
    // Keep lightness between 0.25-0.65 for better visibility
    double adjustedLightness = 0.25 + brightnessPercent * 0.4;

    // Adjust saturation - outer circles are more saturated
    double adjustedSaturation = hslColor.saturation * (1.0 + (1.0 - brightnessPercent) * 0.3);
    adjustedSaturation = adjustedSaturation.clamp(0.0, 1.0);

    return hslColor
        .withLightness(adjustedLightness)
        .withSaturation(adjustedSaturation)
        .toColor()
        .withOpacity(0.7 + brightnessPercent * 0.2); // Outer circles slightly more transparent
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