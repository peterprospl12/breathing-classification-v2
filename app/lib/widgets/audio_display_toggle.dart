import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/display_mode.dart'; // Keep for potential future use, but not strictly needed now
import '../services/audio_service.dart';
// Removed visualization imports as they are no longer built here

class AudioDisplayToggle extends StatelessWidget { // Changed to StatelessWidget
  // Removed constructor parameters: audioData, breathPhases, sampleRate, refreshTime
  const AudioDisplayToggle({super.key});

  // Removed State class (_AudioDisplayToggleState) and its methods:
  // - selectedMode variable
  // - build method logic related to visualization and settings button
  // - _showAdvancedOptionsDialog method
  // - _buildWidget method

  @override
  Widget build(BuildContext context) {
    // Use Consumer only for the button's state
    return Consumer<AudioService>(
      builder: (context, audioService, child) {
        // Build only the Card containing the control panel (Start/Stop button)
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
            // Reduced padding as it only contains the button now
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 12.0, horizontal: 16.0),
              child: _buildControlPanel(audioService), // Directly build the control panel
            ),
          ),
        );
      },
    );
  }

  // _buildControlPanel remains mostly the same, but now part of StatelessWidget
  Widget _buildControlPanel(AudioService audioService) {
    const double buttonWidth = 120;
    const double buttonHeight = 40;
    const double iconSize = 22;
    const double fontSize = 16;

    // Removed the outer Center and margin, the Card's padding handles spacing
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      width: buttonWidth,
      height: buttonHeight,
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
            // No need for setState here as Consumer handles rebuild
          },
          child: Center(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  audioService.isRecording ? Icons.stop_rounded : Icons.mic,
                  color: Colors.white,
                  size: iconSize,
                ),
                const SizedBox(width: 6),
                Text(
                  audioService.isRecording ? 'Stop' : 'Start',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: fontSize,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
