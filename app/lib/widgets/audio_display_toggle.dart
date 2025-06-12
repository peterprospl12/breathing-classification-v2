import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/audio_service.dart';

class AudioDisplayToggle extends StatelessWidget {
  const AudioDisplayToggle({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<AudioService>(
      builder: (context, audioService, child) {
        return Card(
          margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          elevation: 3,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
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
              padding: const EdgeInsets.symmetric(
                vertical: 12.0,
                horizontal: 16.0,
              ),
              child: _buildControlPanel(audioService),
            ),
          ),
        );
      },
    );
  }

  Widget _buildControlPanel(AudioService audioService) {
    const double buttonWidth = 120;
    const double buttonHeight = 40;
    const double iconSize = 22;
    const double fontSize = 16;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      width: buttonWidth,
      height: buttonHeight,
      decoration: BoxDecoration(
        color:
            audioService.isRecording
                ? Colors.red.shade600
                : Colors.green.shade600,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: (audioService.isRecording ? Colors.red : Colors.green)
                .withValues(alpha: 0.3),
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
