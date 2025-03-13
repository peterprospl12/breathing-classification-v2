import 'package:flutter/material.dart';
import '../widgets/audio_metrics_widget.dart';

class MainScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Breath Classification'),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Add the audio metrics widget
            const AudioMetricsWidget(),
          ],
        ),
      ),
    );
  }
}