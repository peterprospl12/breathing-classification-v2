import 'package:flutter/material.dart';
import '../models/breath_classifier.dart';
import 'audio_visualization.dart';
import 'audio_metrics_widget.dart';

class AudioDisplayToggle extends StatefulWidget {
  final List<double> audioData;
  final List<BreathPhase> breathPhases;
  final int sampleRate;
  final double refreshTime;

  const AudioDisplayToggle({
    Key? key,
    required this.audioData,
    required this.breathPhases,
    this.sampleRate = 44100,
    this.refreshTime = 0.3,
  }) : super(key: key);

  @override
  State<AudioDisplayToggle> createState() => _AudioDisplayToggleState();
}

class _AudioDisplayToggleState extends State<AudioDisplayToggle> {
  // Changed default to false to show metrics first
  bool showVisualization = false;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Toggle button
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 8.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton.icon(
                icon: Icon(
                  showVisualization ? Icons.analytics : Icons.graphic_eq,
                  color: Colors.white,
                ),
                label: Text(
                  showVisualization ? 'Show Audio Metrics' : 'Show Visualization',
                  style: const TextStyle(color: Colors.white),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue.shade700,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20),
                  ),
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                ),
                onPressed: () {
                  setState(() {
                    showVisualization = !showVisualization;
                  });
                },
              ),
            ],
          ),
        ),

        // Display the appropriate widget based on the toggle state
        showVisualization
            ? AudioVisualizationWidget(
                audioData: widget.audioData,
                breathPhases: widget.breathPhases,
                sampleRate: widget.sampleRate,
                refreshTime: widget.refreshTime,
              )
            : const AudioMetricsWidget(),
      ],
    );
  }
}
