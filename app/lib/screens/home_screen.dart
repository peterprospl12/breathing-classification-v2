import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/breath_classifier.dart';
import '../services/audio_service.dart';
import '../widgets/breath_counter.dart';
import '../theme/app_theme.dart';
import '../widgets/audio_display_toggle.dart'; 

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
    
    if (audioService.breathPhases.isNotEmpty) {
      _currentPhase = audioService.breathPhases.last;
    }
    
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text('Breathing Monitor'),
        leadingWidth: 230, 
        leading: Row(
          children: [
            Material(
              color: Colors.transparent,
              child: InkWell(
                borderRadius: BorderRadius.circular(24),
                child: Container(
                  width: 48,
                  height: 48,
                  padding: const EdgeInsets.only(left: 16, right: 4),
                  child: const Icon(
                    Icons.mic,
                    color: Colors.white,
                  ),
                ),
              ),
            ),
            if (audioService.selectedDevice != null)
              Expanded(
                child: InkWell(
                  onTap: _showDeviceSelectionDialog,
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          audioService.selectedDevice!.label,
                          style: const TextStyle(fontSize: 11),
                          overflow: TextOverflow.ellipsis,
                          maxLines: 1,
                        ),
                        const Text(
                          'Tap to change',
                          style: TextStyle(fontSize: 9, color: Colors.grey),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
          ],
        ),
        actions: [
          // Show loading indicator when loading devices
          if (audioService.isLoadingDevices)
            const Center(
              child: SizedBox(
                width: 20, 
                height: 20, 
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                ),
              ),
            ),
          
          // Info button
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
              // Replace separate widgets with combined widget
              _buildCombinedCounterAndStatus(
                context,
                audioService,
                _currentPhase, 
                classifier
              ),
              
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: AudioDisplayToggle(
                  audioData: audioService.audioBuffer,
                  breathPhases: audioService.breathPhases,
                  refreshTime: 0.3,
                ),
              ),

              if (_showInfo)
                _buildInfoPanel(),
                            
              _buildLegend(),
            ],
          ),
        ),
      ),
    );
  }

  // New method to combine breath counter and status
  Widget _buildCombinedCounterAndStatus(
    BuildContext context,
    AudioService audioService,
    BreathPhase phase,
    BreathClassifier classifier
  ) {
    final color = classifier.getColorForPhase(phase);
    final label = classifier.getLabelForPhase(phase);
    
    return Card(
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Top section: Breath Counter
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildCounterItem(
                  context, 
                  'INHALE', 
                  audioService.inhaleCount, 
                  AppTheme.inhaleColor
                ),
                Container(
                  height: 50, 
                  width: 1, 
                  color: Colors.grey.withOpacity(0.3)
                ),
                _buildCounterItem(
                  context, 
                  'EXHALE', 
                  audioService.exhaleCount, 
                  AppTheme.exhaleColor
                ),
              ],
            ),
            
            // Reset button
            TextButton.icon(
              onPressed: audioService.resetCounters,
              icon: const Icon(Icons.refresh, size: 16),
              label: const Text('RESET'),
              style: TextButton.styleFrom(
                foregroundColor: Colors.grey,
                padding: EdgeInsets.zero,
              ),
            ),
            
            const Divider(height: 24),
            
            // Bottom section: Current Status - centered
            Row(
              mainAxisAlignment: MainAxisAlignment.center, // Add this to center horizontally
              children: [
                // Animated breathing indicator
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
                  crossAxisAlignment: CrossAxisAlignment.center, // Changed from start to center
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
          ],
        ),
      ),
    );
  }
  
  // Helper method for counter items
  Widget _buildCounterItem(
    BuildContext context, 
    String label, 
    int count, 
    Color color
  ) {
    return Expanded(
      child: Column(
        children: [
          Text(
            label,
            style: TextStyle(
              color: Colors.grey[400],
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            count.toString(),
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  void _showDeviceSelectionDialog() {
    final audioService = Provider.of<AudioService>(context, listen: false);
    
    // Always show the dialog now, even if no devices, to give refresh option
    showDialog(
      context: context,
      builder: (context) => StatefulBuilder( 
        builder: (context, setState) => AlertDialog(
          title: Row(
            children: [
              const Text('Select Audio Input Device'),
              const Spacer(),
              if (audioService.isLoadingDevices)
                const SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              else
                IconButton(
                  icon: const Icon(Icons.refresh),
                  tooltip: 'Refresh audio devices',
                  onPressed: () {
                    audioService.loadInputDevices();
                    setState(() {});
                  },
                ),
            ],
          ),
          content: SizedBox(
            width: double.maxFinite,
            child: audioService.inputDevices.isEmpty
                ? const Center(
                    child: Padding(
                      padding: EdgeInsets.all(16.0),
                      child: Text(
                        'No audio devices available.\nTry refreshing or check your microphone connections.',
                        textAlign: TextAlign.center,
                      ),
                    ),
                  )
                : ListView.builder(
                    shrinkWrap: true,
                    itemCount: audioService.inputDevices.length,
                    itemBuilder: (context, index) {
                      final device = audioService.inputDevices[index];
                      final bool isSelected = audioService.selectedDevice?.id == device.id;
                      
                      return ListTile(
                        title: Text(device.label),
                        leading: const Icon(Icons.mic),
                        selected: isSelected,
                        trailing: isSelected ? const Icon(Icons.check) : null,
                        onTap: () {
                          audioService.selectDevice(device);
                          Navigator.pop(context);
                        },
                      );
                    },
                  ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoPanel() {
    return const Card(
      margin: EdgeInsets.all(16),
      child: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'About Breathing Monitor',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 8),
            Text(
              'This app monitors your breathing patterns in real-time. '
              'It detects inhales, exhales, and silence periods and '
              'visualizes them with different colors.',
            ),
            SizedBox(height: 12),
            Row(
              children: [
                Icon(Icons.mic, size: 18),
                SizedBox(width: 8),
                Text(
                  'Tap the mic button to start/stop monitoring',
                  style: TextStyle(fontSize: 14),
                ),
              ],
            ),
            SizedBox(height: 8),
            Row(
              children: [
                Icon(Icons.refresh, size: 18),
                SizedBox(width: 8),
                Text(
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
