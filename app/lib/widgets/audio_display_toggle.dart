import 'package:flutter/material.dart';
import '../models/breath_classifier.dart';
import '../models/display_mode.dart';
import 'audio_visualization.dart';
import 'microphone_visualization.dart';

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
  // Default to simulation mode
  DisplayMode selectedMode = DisplayMode.simulation;

  void _selectDisplayMode(DisplayMode mode) {
    setState(() {
      selectedMode = mode;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Navigation bar with visualization options
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 8.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: DisplayMode.values.map((mode) {
              final isSelected = mode == selectedMode;
              return Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8.0),
                child: ElevatedButton.icon(
                  icon: Icon(
                    mode.icon,
                    color: Colors.white,
                  ),
                  label: Text(
                    mode.label,
                    style: const TextStyle(color: Colors.white),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: isSelected 
                        ? Colors.blue.shade700 
                        : Colors.blue.shade400,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                  ),
                  onPressed: () => _selectDisplayMode(mode),
                ),
              );
            }).toList(),
          ),
        ),

        // Display only the selected visualization
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Container(
            padding: const EdgeInsets.all(8.0),
            decoration: BoxDecoration(
              border: Border.all(
                color: Colors.blue.shade700,
                width: 2,
              ),
              borderRadius: BorderRadius.circular(16),
            ),
            child: _buildWidget(selectedMode),
          ),
        ),
      ],
    );
  }

  Widget _buildWidget(DisplayMode mode) {
    switch (mode) {
      case DisplayMode.simulation:
        return AudioVisualizationWidget(
          audioData: widget.audioData,
          breathPhases: widget.breathPhases,
          sampleRate: widget.sampleRate,
          refreshTime: widget.refreshTime,
        );
      case DisplayMode.microphone:
        return const MicrophoneVisualizationWidget();
    }
  }
}
