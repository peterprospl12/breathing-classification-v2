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
        // Navigation bar with visualization options
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 8.0),
          child: LayoutBuilder(
            builder: (context, constraints) {
              // Dostosuj układ przycisków w zależności od dostępnej szerokości
              final bool isNarrow = constraints.maxWidth < 400;

              return Wrap(
                alignment: WrapAlignment.center,
                spacing: 8.0, // przestrzeń pozioma między przyciskami
                runSpacing: 8.0, // przestrzeń pionowa gdy przyciski przechodzą do nowej linii
                children: DisplayMode.values.map((mode) {
                  final isSelected = mode == selectedMode;
                  return _buildModeButton(mode, isSelected, isNarrow);
                }).toList(),
              );
            },
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

  Widget _buildModeButton(DisplayMode mode, bool isSelected, bool isNarrow) {
    return ElevatedButton.icon(
      icon: Icon(
        mode.icon,
        color: Colors.white,
        // Mniejsze ikony na wąskich ekranach
        size: isNarrow ? 18 : 22,
      ),
      label: Text(
        // Na wąskich ekranach używaj krótszych etykiet
        isNarrow ? _getShortLabel(mode) : mode.label,
        style: TextStyle(
          color: Colors.white,
          fontSize: isNarrow ? 14 : 16,
        ),
      ),
      style: ElevatedButton.styleFrom(
        backgroundColor: isSelected
            ? Colors.blue.shade700
            : Colors.blue.shade400,
        foregroundColor: Colors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
        ),
        // Mniejsze odstępy na wąskich ekranach
        padding: isNarrow
            ? const EdgeInsets.symmetric(horizontal: 10, vertical: 8)
            : const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      ),
      onPressed: () => _selectDisplayMode(mode),
    );
  }

  // Zwraca krótszą etykietę dla wąskich ekranów
  String _getShortLabel(DisplayMode mode) {
    switch (mode) {
      case DisplayMode.microphone:
        return 'Mic';
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
