import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/audio_service.dart';
import '../models/breath_classifier.dart';
import '../theme/app_theme.dart';

class CircularVisualizationWidget extends StatefulWidget {
  final bool showSeparateCounters;

  const CircularVisualizationWidget({
    super.key,
    required this.showSeparateCounters,
  });

  @override
  State<CircularVisualizationWidget> createState() =>
      _CircularVisualizationWidgetState();
}

class _CircularVisualizationWidgetState
    extends State<CircularVisualizationWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  StreamSubscription<List<int>>? _audioSubscription;
  StreamSubscription<BreathPhase>? _breathPhaseSubscription;
  List<int> _audioData = [];
  BreathPhase _currentPhase = BreathPhase.silence;
  double _currentAmplitude = 0.0;

  // Smoothing factor for animations
  static const double _smoothingFactor = 0.2;

  static const double _minCircleSize = 50.0;
  static const double _maxCircleSize = 120.0;

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

    _audioSubscription = audioService.subscribeToAudioStream((audioData) {
      if (audioData.isNotEmpty) {
        setState(() {
          _audioData = audioData;
          double sumSquares = 0.0;
          for (int sample in _audioData) {
            sumSquares += (sample * sample);
          }
          double rms = math.sqrt(sumSquares / _audioData.length);

          _currentAmplitude =
              _currentAmplitude * (1 - _smoothingFactor) +
              (rms / 1024) * _smoothingFactor;
        });
      }
    });

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
    final theme = Theme.of(context);
    return Consumer<AudioService>(
      builder: (context, audioService, child) {
        return Container(
          height: 200,
          width: double.infinity,
          decoration: BoxDecoration(borderRadius: BorderRadius.circular(12)),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: AnimatedBuilder(
              animation: _animationController,
              builder: (context, child) {
                return _buildCircularVisualization(
                  audioService,
                  theme,
                  widget.showSeparateCounters,
                );
              },
            ),
          ),
        );
      },
    );
  }

  Widget _buildCircularVisualization(
    AudioService audioService,
    ThemeData theme,
    bool showSeparateCounters,
  ) {
    Color phaseColor;
    String phaseLabel;
    Color neutralColor = theme.colorScheme.onSurfaceVariant.withValues(
      alpha: 0.6,
    );

    if (!showSeparateCounters) {
      if (_currentPhase == BreathPhase.exhale) {
        phaseColor = AppTheme.exhaleColor;
        phaseLabel = BreathClassifier.getLabelForPhase(_currentPhase);
      } else {
        phaseColor = neutralColor;
        phaseLabel = 'BREATHING';
      }
    } else {
      phaseColor = BreathClassifier.getColorForPhase(_currentPhase);
      phaseLabel = BreathClassifier.getLabelForPhase(_currentPhase);
    }

    final double continuousPulse =
        audioService.isRecording
            ? (math.sin(_animationController.value * 4 * math.pi) * 0.02 + 1.0)
            : 1.0;
    final double amplitudeEffect = _currentAmplitude.clamp(0.1, 1.0);

    final double baseCircleSize =
        _minCircleSize +
        (_maxCircleSize - _minCircleSize) * amplitudeEffect * continuousPulse;

    const int maxLayers = 3;
    final int visibleLayers = (amplitudeEffect * maxLayers).ceil().clamp(
      1,
      maxLayers,
    );

    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            phaseLabel.toUpperCase(),
            style: theme.textTheme.titleMedium?.copyWith(
              color: phaseColor,
              fontWeight: FontWeight.w600,
              letterSpacing: 1.0,
            ),
          ),
          const SizedBox(height: 15),
          // Circular Layers
          Stack(
            alignment: Alignment.center,
            children: List.generate(visibleLayers, (index) {
              int layerIndex = visibleLayers - 1 - index; // 0 is the innermost

              // Adjust size calculation for fewer layers
              double layerSizePercent =
                  1.0 - (layerIndex * 0.25); // Increase spacing between layers
              double layerSize = baseCircleSize * layerSizePercent;

              // Pass determined phaseColor and neutralColor to _getLayerColor
              Color layerColor = _getLayerColor(
                phaseColor,
                neutralColor,
                layerIndex,
                visibleLayers,
                theme,
                showSeparateCounters,
              );

              return AnimatedContainer(
                duration: const Duration(milliseconds: 150), // Faster animation
                width: layerSize.clamp(
                  _minCircleSize * 0.5,
                  _maxCircleSize * 1.1,
                ), // Clamp size
                height: layerSize.clamp(
                  _minCircleSize * 0.5,
                  _maxCircleSize * 1.1,
                ),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  // Use gradient for smoother appearance
                  gradient: RadialGradient(
                    colors: [
                      layerColor.withValues(alpha: 0.6),
                      layerColor.withValues(alpha: 0.1),
                    ],
                    stops: const [0.3, 1.0],
                  ),
                ),
              );
            }),
          ),

          // Recording/Pause Indicator
          const SizedBox(height: 20),
          if (audioService.isRecording)
            _buildPulsatingDot(theme)
          else
            Icon(
              Icons.pause_circle_outline,
              color: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.6),
              size: 18,
            ),
        ],
      ),
    );
  }

  Color _getLayerColor(
    Color baseColor,
    Color neutralColor,
    int layerIndex,
    int totalLayers,
    ThemeData theme,
    bool showSeparateCounters,
  ) {
    if (totalLayers <= 1) {
      return baseColor.withValues(alpha: 0.5);
    }

    Color blendTargetColor;
    if (!showSeparateCounters && baseColor == neutralColor) {
      blendTargetColor = theme.colorScheme.onSurface.withValues(alpha: 0.3);
    } else {
      blendTargetColor = theme.colorScheme.primary.withValues(alpha: 0.7);
    }

    double blendFactor =
        layerIndex / (totalLayers - 1); // 0 = innermost, 1 = outermost
    return Color.lerp(
      baseColor, // Innermost is the determined phase color (or neutral)
      blendTargetColor, // Outermost blends towards target
      blendFactor * 0.6, // Control blend intensity
    )!;
  }

  Widget _buildPulsatingDot(ThemeData theme) {
    final double opacity =
        (math.sin(_animationController.value * 4 * math.pi) * 0.3 + 0.7).clamp(
          0.4,
          1.0,
        );
    return Container(
      width: 8.0,
      height: 8.0,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: theme.colorScheme.primary.withValues(alpha: opacity),
        boxShadow: [
          BoxShadow(
            color: theme.colorScheme.primary.withValues(alpha: opacity * 0.5),
            blurRadius: 4.0,
            spreadRadius: 1.0,
          ),
        ],
      ),
    );
  }
}
