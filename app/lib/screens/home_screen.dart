import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/breath_classifier.dart';
import '../services/audio_service.dart';
import '../theme/app_theme.dart';
import '../models/display_mode.dart';
import '../widgets/microphone_visualization.dart';
import '../widgets/circular_audio_visualization.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  BreathPhase _currentPhase = BreathPhase.silence;
  StreamSubscription<BreathPhase>? _breathPhaseSubscription;
  final List<BreathPhase> _breathPhases = [];
  StreamSubscription<List<int>>? _audioSubscription;
  DisplayMode _selectedMode = DisplayMode.circular;

  static const int _maxBreathPhasesToStore = 20;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat(reverse: true);

    WidgetsBinding.instance.addPostFrameCallback((_) {
      _subscribeToStreams();
    });
  }

  void _subscribeToStreams() {
    final audioService = Provider.of<AudioService>(context, listen: false);

    _breathPhaseSubscription = audioService.breathPhasesStream.listen((phase) {
      setState(() {
        _currentPhase = phase;

        _breathPhases.add(phase);
        if (_breathPhases.length > _maxBreathPhasesToStore) {
          _breathPhases.removeAt(0);
        }
      });
    });

    _audioSubscription = audioService.subscribeToAudioStream((audioData) {
      setState(() {
      });
    });
  }

  @override
  void dispose() {
    _animationController.dispose();
    _breathPhaseSubscription?.cancel();
    _audioSubscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final audioService = Provider.of<AudioService>(context);
    final classifier = Provider.of<BreathClassifier>(context, listen: false);

    return Scaffold(
      appBar: AppBar(
        elevation: 4,
        flexibleSpace: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [Theme.of(context).primaryColor.withAlpha(90), Theme.of(context).primaryColor.withBlue(150)],
            ),
          ),
        ),
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: Colors.white.withAlpha(20),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.air, size: 18),
            ),
            const SizedBox(width: 10),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Breathing Monitor',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 0.5,
                  ),
                ),
                Text(
                  'Real-time analysis',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w300,
                    color: Colors.white.withAlpha(200),
                  ),
                ),
              ],
            ),
          ],
        ),
        centerTitle: false,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(40.0),
          child: Padding(
            padding: const EdgeInsets.only(bottom: 8.0),
            child: Row(
              children: [
                Material(
                  color: Colors.transparent,
                  child: InkWell(
                    borderRadius: BorderRadius.circular(24),
                    child: Container(
                      width: 40,
                      height: 40,
                      padding: const EdgeInsets.only(left: 16),
                      child: const Icon(
                        Icons.mic,
                        color: Colors.white,
                        size: 20,
                      ),
                    ),
                  ),
                ),
                Expanded(
                  child: audioService.selectedDevice != null
                      ? InkWell(
                          onTap: _showDeviceSelectionDialog,
                          child: Padding(
                            padding: const EdgeInsets.symmetric(vertical: 4.0, horizontal: 4.0),
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
                        )
                      : const SizedBox(),
                ),
                if (audioService.isLoadingDevices)
                  const Padding(
                    padding: EdgeInsets.only(right: 16.0),
                    child: SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            tooltip: 'Display Options',
            onPressed: () => _showAdvancedOptionsDialog(context),
          ),
          IconButton(
            icon: const Icon(Icons.info),
            tooltip: 'Show information',
            onPressed: () => _showInfoDialog(),
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildCombinedCounterAndStatus(
                context,
                audioService,
                _currentPhase,
                classifier,
                _animationController,
                _selectedMode
              ),
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 0),
                child: Container(
                  child: _buildWidget(_selectedMode),
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 24.0, horizontal: 16.0),
                child: _buildControlPanel(audioService),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCombinedCounterAndStatus(
    BuildContext context,
    AudioService audioService,
    BreathPhase phase,
    BreathClassifier classifier,
    AnimationController animationController,
    DisplayMode selectedMode
  ) {
    final color = BreathClassifier.getColorForPhase(phase);
    final label = BreathClassifier.getLabelForPhase(phase);

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
              Theme.of(context).cardColor.withAlpha(230),
            ],
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(vertical: 8),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(10),
                  color: Theme.of(context).cardColor.withAlpha(128),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    _buildCounterItem(
                      context,
                      'INHALE',
                      audioService.inhaleCount,
                      AppTheme.inhaleColor
                    ),
                    _buildVerticalResetButton(audioService.resetCounters),
                    _buildCounterItem(
                      context,
                      'EXHALE',
                      audioService.exhaleCount,
                      AppTheme.exhaleColor
                    ),
                  ],
                ),
              ),
              if (selectedMode == DisplayMode.microphone) ...[
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(10),
                    color: color.withAlpha(13),
                    border: Border.all(color: color.withAlpha(51), width: 1),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      AnimatedBuilder(
                        animation: animationController,
                        builder: (context, child) {
                          return Container(
                            width: 28,
                            height: 28,
                            decoration: BoxDecoration(
                              color: color.withAlpha((26 + 77 * animationController.value).toInt()),
                              shape: BoxShape.circle,
                              boxShadow: [
                                BoxShadow(
                                  color: color.withAlpha((77 * animationController.value).toInt()),
                                  blurRadius: 4,
                                  spreadRadius: 1 * animationController.value,
                                ),
                              ],
                            ),
                            child: Center(
                              child: Container(
                                width: 12,
                                height: 12,
                                decoration: BoxDecoration(
                                  color: color,
                                  shape: BoxShape.circle,
                                ),
                              ),
                            ),
                          );
                        },
                      ),
                      const SizedBox(width: 12),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            'STATUS',
                            style: TextStyle(
                              fontSize: 9,
                              fontWeight: FontWeight.w500,
                              letterSpacing: 0.5,
                              color: Colors.grey[600],
                            ),
                          ),
                          const SizedBox(height: 1),
                          Text(
                            label,
                            style: Theme.of(context).textTheme.titleSmall?.copyWith(
                              fontWeight: FontWeight.bold,
                              color: color,
                              letterSpacing: 0.5,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
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
        color: audioService.isRecording ? Colors.red.shade600 : Colors.green.shade600,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: (audioService.isRecording ? Colors.red : Colors.green).withValues(alpha: 0.3),
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

  Widget _buildVerticalResetButton(VoidCallback onPressed) {
    return InkWell(
      onTap: onPressed,
      borderRadius: BorderRadius.circular(8),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.refresh, size: 18, color: Colors.grey[600]),
            const SizedBox(height: 4),
            Text(
              'RESET',
              style: TextStyle(
                fontSize: 8,
                fontWeight: FontWeight.w500,
                color: Colors.grey[600],
                letterSpacing: 0.5,
              ),
            ),
          ],
        ),
      ),
    );
  }

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
              color: Colors.grey[500],
              fontSize: 10,
              fontWeight: FontWeight.bold,
              letterSpacing: 0.8,
            ),
          ),
          const SizedBox(height: 6),
          Container(
            padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 12),
            decoration: BoxDecoration(
              color: color.withAlpha(26),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text(
              count.toString(),
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _showDeviceSelectionDialog() {
    final audioService = Provider.of<AudioService>(context, listen: false);

    showDialog(
      context: context,
      builder: (context) => ChangeNotifierProvider.value(
        value: audioService,
        child: Consumer<AudioService>(
          builder: (context, audioService, _) {
            return AlertDialog(
              title: Row(
                children: [
                  const Expanded(
                    child: Text(
                      'Select Audio Input Device',
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(fontSize: 14),
                    ),
                  ),
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
                      onPressed: () => audioService.loadInputDevices(),
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
            );
          },
        ),
      ),
    );
  }

  void _showInfoDialog() {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Theme.of(context).primaryColor,
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(16),
                    topRight: Radius.circular(16),
                  ),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.info_outline, color: Colors.white, size: 24),
                    const SizedBox(width: 12),
                    const Text(
                      'App Information',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    const Spacer(),
                    IconButton(
                      icon: const Icon(Icons.close, color: Colors.white),
                      onPressed: () => Navigator.pop(context),
                      padding: EdgeInsets.zero,
                      constraints: const BoxConstraints(),
                    ),
                  ],
                ),
              ),
              const Padding(
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
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.mic, size: 18),
                        SizedBox(width: 8),
                        Flexible(
                          child: Text(
                            'Tap the mic button to toggle monitoring',
                            style: TextStyle(fontSize: 14),
                            softWrap: true,
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: 8),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.refresh, size: 18),
                        SizedBox(width: 8),
                        Flexible(
                          child: Text(
                            'Reset counters with the refresh button',
                            style: TextStyle(fontSize: 14),
                            softWrap: true,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const Divider(height: 1),
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Legend',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        _buildLegendItem(AppTheme.inhaleColor, 'Inhale'),
                        _buildLegendItem(AppTheme.exhaleColor, 'Exhale'),
                        _buildLegendItem(AppTheme.silenceColor, 'Silence'),
                      ],
                    ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                child: TextButton(
                  onPressed: () => Navigator.pop(context),
                  style: TextButton.styleFrom(
                    backgroundColor: Theme.of(context).primaryColor,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  child: const Text('CLOSE'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLegendItem(Color color, String label) {
    return Column(
      children: [
        Container(
          width: 30,
          height: 30,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
            border: Border.all(
              color: Colors.white.withAlpha(30),
              width: 2,
            ),
            boxShadow: [
              BoxShadow(
                color: color.withAlpha(128),
                blurRadius: 8,
                spreadRadius: 1,
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w500,
            color: Colors.white,
          ),
        ),
      ],
    );
  }

  Future<void> _showAdvancedOptionsDialog(BuildContext context) async {
    DisplayMode? tempSelectedMode = _selectedMode;
    final result = await showDialog<DisplayMode>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Display Options'),
          content: StatefulBuilder(
            builder: (BuildContext context, StateSetter setStateDialog) {
              return Column(
                mainAxisSize: MainAxisSize.min,
                children: DisplayMode.values.map((mode) {
                  return RadioListTile<DisplayMode>(
                    title: Text(mode.label),
                    value: mode,
                    groupValue: tempSelectedMode,
                    onChanged: (DisplayMode? value) {
                      if (value != null) {
                        setStateDialog(() {
                          tempSelectedMode = value;
                        });
                      }
                    },
                    secondary: Icon(mode.icon),
                    activeColor: Colors.blue.shade700,
                  );
                }).toList(),
              );
            },
          ),
          actions: <Widget>[
            TextButton(
              child: const Text('Cancel'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
            TextButton(
              child: const Text('Apply'),
              onPressed: () {
                Navigator.of(context).pop(tempSelectedMode);
              },
            ),
          ],
        );
      },
    );

    if (result != null && result != _selectedMode) {
      setState(() {
        _selectedMode = result;
      });
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
