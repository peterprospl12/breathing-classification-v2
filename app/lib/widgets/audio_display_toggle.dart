import 'package:flutter/material.dart';
import '../models/breath_classifier.dart';
import '../models/display_mode.dart';
import 'microphone_visualization.dart';
import 'circular_audio_visualization.dart';

class AudioDisplayToggle extends StatefulWidget {
  final List<double> audioData;
  final List<BreathPhase> breathPhases;
  final int sampleRate;
  final double refreshTime;

  const AudioDisplayToggle({
    super.key,
    required this.audioData,
    required this.breathPhases,
    this.sampleRate = 44100,
    this.refreshTime = 0.3,
  });

  @override
  State<AudioDisplayToggle> createState() => _AudioDisplayToggleState();
}

class _AudioDisplayToggleState extends State<AudioDisplayToggle> {
  DisplayMode selectedMode = DisplayMode.microphone;

  void _selectDisplayMode(DisplayMode mode) {
    setState(() {
      selectedMode = mode;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Card dla przycisków wyboru trybu
        Card(
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
              padding: const EdgeInsets.symmetric(vertical: 16.0, horizontal: 16.0),
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final bool isNarrow = constraints.maxWidth < 400;
                  
                  return Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: DisplayMode.values.map((mode) {
                      final isSelected = mode == selectedMode;
                      return Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 8.0),
                        child: _buildModeButton(mode, isSelected, isNarrow),
                      );
                    }).toList(),
                  );
                },
              ),
            ),
          ),
        ),

        // Wizualizacja
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 0),
          child: Container(
            child: _buildWidget(selectedMode),
          ),
        ),
      ],
    );
  }

  Widget _buildModeButton(DisplayMode mode, bool isSelected, bool isNarrow) {
    return ElevatedButton(
      style: ElevatedButton.styleFrom(
        backgroundColor: isSelected
            ? Colors.blue.shade700
            : Colors.blue.shade400,
        foregroundColor: Colors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
        ),
        padding: isNarrow
            ? const EdgeInsets.symmetric(horizontal: 20, vertical: 8)
            : const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      ),
      onPressed: () => _selectDisplayMode(mode),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            mode.icon,
            color: Colors.white,
            size: isNarrow ? 18 : 20,
          ),
          SizedBox(width: 5),
          Text(
            isNarrow ? _getShortLabel(mode) : mode.label,
            style: TextStyle(
              color: Colors.white,
              fontSize: isNarrow ? 14 : 16,
            ),
          ),
        ],
      ),
    );
  }

  String _getShortLabel(DisplayMode mode) {
    switch (mode) {
      case DisplayMode.microphone:
        return 'Plot';
      case DisplayMode.circular:
        return 'Circle';
    }
  }

  Widget _buildWidget(DisplayMode mode) {
    switch (mode) {
      case DisplayMode.microphone:
        return const MicrophoneVisualizationWidget();
      case DisplayMode.circular:
        return const CircularVisualizationWidget();
    }
  }
}
