import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/breath_classifier.dart';
import '../services/audio_service.dart';
import '../widgets/audio_visualization.dart';
import '../widgets/breath_counter.dart';
import '../theme/app_theme.dart';
import '../widgets/audio_metrics_widget.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  BreathPhase _currentPhase = BreathPhase.silence;
  bool _showInfo = false;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final audioService = Provider.of<AudioService>(context);
    final classifier = Provider.of<BreathClassifier>(context, listen: false);
    
    // Update current phase if there are breath phases
    if (audioService.breathPhases.isNotEmpty) {
      _currentPhase = audioService.breathPhases.last;
    }
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Breathing Monitor'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: Icon(_showInfo ? Icons.info_outline : Icons.info),
            onPressed: () => setState(() => _showInfo = !_showInfo),
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const AudioMetricsWidget(),

              if (_showInfo)
                _buildInfoPanel(),

              // Breath counter widget
              BreathCounter(
                inhaleCount: audioService.inhaleCount,
                exhaleCount: audioService.exhaleCount,
                onReset: () => audioService.resetCounters(),
              ),
              
              // Current breath status
              _buildStatusCard(context, _currentPhase, classifier),
              
              // Visualization - setting a fixed height instead of Expanded
              SizedBox(
                height: 200, // Fixed height instead of using Expanded
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: AudioVisualizationWidget(
                    audioData: audioService.audioBuffer,
                    breathPhases: audioService.breathPhases,
                  ),
                ),
              ),

              // Legend
              _buildLegend(),
              
              // Add padding at the bottom for the floating action button
              const SizedBox(height: 80),
            ],
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          if (audioService.isRecording) {
            audioService.stopRecording();
          } else {
            audioService.startRecording();
          }
        },
        backgroundColor: audioService.isRecording ? Colors.red : AppTheme.primaryColor,
        child: Icon(
          audioService.isRecording ? Icons.stop : Icons.mic,
          color: Colors.white,
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  Widget _buildInfoPanel() {
    return Card(
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'About Breathing Monitor',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'This app monitors your breathing patterns in real-time. '
              'It detects inhales, exhales, and silence periods and '
              'visualizes them with different colors.',
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                const Icon(Icons.mic, size: 18),
                const SizedBox(width: 8),
                const Text(
                  'Tap the mic button to start/stop monitoring',
                  style: TextStyle(fontSize: 14),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.refresh, size: 18),
                const SizedBox(width: 8),
                const Text(
                  'Reset counters with the refresh button',
                  style: TextStyle(fontSize: 14),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusCard(BuildContext context, BreathPhase phase, BreathClassifier classifier) {
    final color = classifier.getColorForPhase(phase);
    final label = classifier.getLabelForPhase(phase);
    
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          children: [
            AnimatedBuilder(
              animation: _animationController,
              builder: (context, child) {
                return Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.2 + 0.6 * _animationController.value),
                    shape: BoxShape.circle,
                  ),
                  child: Center(
                    child: Container(
                      width: 16,
                      height: 16,
                      decoration: BoxDecoration(
                        color: color,
                        shape: BoxShape.circle,
                      ),
                    ),
                  ),
                );
              },
            ),
            const SizedBox(width: 16),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Current Status',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.grey,
                  ),
                ),
                Text(
                  label,
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: color,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegend() {
    return Card(
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Legend',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildLegendItem(AppTheme.inhaleColor, 'Inhale'),
                _buildLegendItem(AppTheme.exhaleColor, 'Exhale'),
                _buildLegendItem(AppTheme.silenceColor, 'Silence'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegendItem(Color color, String label) {
    return Row(
      children: [
        Container(
          width: 16,
          height: 16,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: 8),
        Text(
          label,
          style: TextStyle(
            fontSize: 14,
            color: Colors.grey[300],
          ),
        ),
      ],
    );
  }
}
