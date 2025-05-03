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

  static const int _maxLayers = 4;
  static const double _minCircleSize = 60.0;
  static const double _maxCircleSize = 150.0;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
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
      if (audioData.isNotEmpty) {
        setState(() {
          _audioData = audioData;
          double sumSquares = 0.0;
          for (int sample in _audioData) {
            sumSquares += (sample * sample);
          }
          double rms = math.sqrt(sumSquares / _audioData.length);

          _currentAmplitude = _currentAmplitude * (1 - _smoothingFactor) +
                            (rms / 1024) * _smoothingFactor;
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
                  Theme.of(context).cardColor.withValues(alpha: 0.9),
                ],
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: _buildCircularVisualization(audioService),
            ),
          ),
        );
      },
    );
  }

  Widget _buildCircularVisualization(AudioService audioService) {
    final Color phaseColor = BreathClassifier.getColorForPhase(_currentPhase);

    final double pulseEffect = audioService.isRecording && _audioData.isNotEmpty ?
                              math.sin(_animationController.value * 2 * math.pi) * 0.05 + 1.0 : 1.0;

    final double amplitudePercent = _currentAmplitude.clamp(0.2, 1.0) * pulseEffect;

    final int visibleLayers = (amplitudePercent * _maxLayers).ceil().clamp(1, _maxLayers);

    final double baseCircleSize = _minCircleSize + (_maxCircleSize - _minCircleSize) * amplitudePercent;

    return Container(
      height: 250,
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.transparent,
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
                  Text(
                    BreathClassifier.getLabelForPhase(_currentPhase),
                    style: TextStyle(
                      color: phaseColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Stack(
                    alignment: Alignment.center,
                    children: List.generate(visibleLayers, (index) {
                      int layerIndex = visibleLayers - 1 - index;

                      double layerSizePercent = 1.0 - (layerIndex * 0.18);
                      double layerSize = baseCircleSize * layerSizePercent;

                      Color layerColor = _getLayerColor(phaseColor, layerIndex, visibleLayers);

                      return AnimatedContainer(
                        duration: const Duration(milliseconds: 150),
                        width: layerSize,
                        height: layerSize,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: layerColor,
                          border: Border.all(
                            color: layerColor.withValues(alpha: 0.9),
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
                        color: phaseColor.withValues(alpha: 0.8),
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

  Color _getLayerColor(Color baseColor, int layerIndex, int totalLayers) {
    if (totalLayers == 1) return baseColor.withValues(alpha: 0.6);

    double brightnessPercent = layerIndex / (totalLayers - 1);

    HSLColor hslColor = HSLColor.fromColor(baseColor);

    double adjustedLightness = 0.25 + brightnessPercent * 0.4;

    double adjustedSaturation = hslColor.saturation * (1.0 + (1.0 - brightnessPercent) * 0.3);
    adjustedSaturation = adjustedSaturation.clamp(0.0, 1.0);

    return hslColor
        .withLightness(adjustedLightness)
        .withSaturation(adjustedSaturation)
        .toColor()
        .withValues(alpha: 0.7 + brightnessPercent * 0.2);
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
            color: Colors.white.withValues(alpha: value),
          ),
        );
      },
      onEnd: () => setState(() {}),
    );
  }
}