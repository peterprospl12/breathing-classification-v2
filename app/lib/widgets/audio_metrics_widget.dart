import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/audio_service.dart';

class AudioMetricsWidget extends StatelessWidget {
  const AudioMetricsWidget({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Consumer<AudioService>(
      builder: (context, audioService, child) {
        return Card(
          margin: const EdgeInsets.all(8.0),
          child: Padding(
            padding: const EdgeInsets.all(12.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Audio Metrics',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                
                // Amplitude display
                Text('Amplitude: ${audioService.currentAmplitude.toStringAsFixed(4)}'),
                const SizedBox(height: 4),
                LinearProgressIndicator(
                  value: audioService.currentAmplitude,
                  minHeight: 8,
                ),
                
                const SizedBox(height: 8),
                
                // PCM Samples display - more compact
                const Text('PCM Samples:'),
                const SizedBox(height: 4),
                
                // Display PCM values and visual bars with better scaling
                SizedBox(
                  height: 70, // Reduced height
                  child: audioService.first10PcmSamples.isEmpty
                      ? const Center(child: Text('No samples yet'))
                      : Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: audioService.first10PcmSamples.map((sample) {
                            // Normalize the sample value for display
                            double normalizedValue = sample / 32767.0; // PCM 16-bit max value
                            double absValue = normalizedValue.abs();
                            
                            // Zastosuj nieliniowe mapowanie dla lepszej widoczności małych wartości
                            // Używamy pierwiastka kwadratowego, który uwypukla małe wartości
                            // Dodatkowo dodajemy minimalną wysokość 2px
                            double barHeight = math.max(2.0, 40.0 * math.sqrt(absValue));
                            
                            return Expanded(
                              child: Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 1.0),
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.end,
                                  children: [
                                    Text(
                                      sample.toString(),
                                      style: const TextStyle(fontSize: 8),
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                    const SizedBox(height: 2),
                                    Container(
                                      height: barHeight,
                                      width: double.infinity,
                                      color: normalizedValue >= 0 
                                          ? Colors.blue 
                                          : Colors.red,
                                    ),
                                  ],
                                ),
                              ),
                            );
                          }).toList(),
                        ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}