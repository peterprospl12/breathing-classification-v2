import 'dart:async';
import 'package:breathing_app/enums/enums.dart';
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

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  BreathPhase _currentPhase = BreathPhase.silence;
  StreamSubscription<BreathPhase>? _breathPhaseSubscription;
  final List<BreathPhase> _breathPhases = [];
  StreamSubscription<List<int>>? _audioSubscription;
  DisplayMode _selectedMode = DisplayMode.circular;
  bool _showSeparateCounters = true;

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
      setState(() {});
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
    final theme = Theme.of(context);
    const double appBarBottomRadius = 28.0;

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      appBar: AppBar(
        elevation: 1,
        backgroundColor: theme.colorScheme.surface,
        foregroundColor: theme.colorScheme.onSurface,
        shape: const ContinuousRectangleBorder(
          borderRadius: BorderRadius.vertical(
            bottom: Radius.circular(appBarBottomRadius * 1.5),
          ),
        ),
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.air_outlined,
              size: 24,
              color: theme.colorScheme.primary,
            ),
            const SizedBox(width: 12),
            Text(
              'Breathing Monitor',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w500,
                color: theme.colorScheme.onSurface,
              ),
            ),
          ],
        ),
        centerTitle: false,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(40.0),
          child: Container(
            decoration: BoxDecoration(
              color: ElevationOverlay.applySurfaceTint(
                theme.colorScheme.surface,
                theme.colorScheme.surfaceTint,
                0.5,
              ),
              borderRadius: const BorderRadius.vertical(
                bottom: Radius.circular(appBarBottomRadius * 1.5),
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16.0, 0, 16.0, 8.0),
              child: Row(
                children: [
                  Icon(
                    Icons.mic_none_outlined,
                    color: theme.colorScheme.onSurfaceVariant,
                    size: 20,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child:
                        audioService.selectedDevice != null
                            ? InkWell(
                              onTap: _showDeviceSelectionDialog,
                              borderRadius: BorderRadius.circular(4),
                              child: Padding(
                                padding: const EdgeInsets.symmetric(
                                  vertical: 4.0,
                                ),
                                child: Text(
                                  audioService.selectedDevice!.label,
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: theme.colorScheme.onSurfaceVariant,
                                    fontWeight: FontWeight.w400,
                                  ),
                                  overflow: TextOverflow.ellipsis,
                                  maxLines: 1,
                                ),
                              ),
                            )
                            : Text(
                              'No device selected',
                              style: TextStyle(
                                fontSize: 12,
                                color: theme.colorScheme.onSurfaceVariant
                                    .withValues(alpha: 0.7),
                                fontStyle: FontStyle.italic,
                              ),
                            ),
                  ),
                  if (audioService.isLoadingDevices)
                    Padding(
                      padding: const EdgeInsets.only(left: 8.0),
                      child: SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 1.5,
                          valueColor: AlwaysStoppedAnimation<Color>(
                            theme.colorScheme.primary,
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.tune_outlined),
            tooltip: 'Display Options',
            onPressed: () => _showAdvancedOptionsDialog(context),
            color: theme.colorScheme.onSurfaceVariant,
          ),
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: 'Show information',
            onPressed: () => _showInfoDialog(),
            color: theme.colorScheme.onSurfaceVariant,
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(vertical: 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildCombinedCounterAndStatus(
                context,
                audioService,
                _currentPhase,
                classifier,
                _animationController,
                _selectedMode,
                _showSeparateCounters,
              ),
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16.0,
                  vertical: 8.0,
                ),
                child: _buildWidget(_selectedMode),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(
                  vertical: 16.0,
                  horizontal: 16.0,
                ),
                child: Center(child: _buildControlPanel(audioService)),
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
    DisplayMode selectedMode,
    bool showSeparateCounters,
  ) {
    final color = BreathClassifier.getColorForPhase(phase);
    final label = BreathClassifier.getLabelForPhase(phase);
    final theme = Theme.of(context);

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(
          color: theme.dividerColor.withValues(alpha: 0.5),
          width: 1,
        ),
      ),
      color: theme.colorScheme.surface,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              mainAxisAlignment:
                  showSeparateCounters
                      ? MainAxisAlignment.spaceEvenly
                      : MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                if (showSeparateCounters) ...[
                  _buildCounterItem(
                    context,
                    'INHALE',
                    audioService.inhaleCount,
                    AppTheme.inhaleColor,
                  ),
                  _buildVerticalResetButton(audioService.resetCounters),
                  _buildCounterItem(
                    context,
                    'EXHALE',
                    audioService.exhaleCount,
                    AppTheme.exhaleColor,
                  ),
                ] else ...[
                  _buildCounterItem(
                    context,
                    'BREATHS',
                    audioService.exhaleCount,
                    theme.colorScheme.primary,
                  ),
                  Padding(
                    padding: const EdgeInsets.only(left: 16.0),
                    child: _buildVerticalResetButton(
                      audioService.resetCounters,
                    ),
                  ),
                ],
              ],
            ),
            // Timer and Tempo Section
            const SizedBox(height: 16),
            Divider(
              height: 1,
              thickness: 1,
              color: theme.dividerColor.withValues(alpha: 0.5),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                StreamBuilder<Duration>(
                  stream: audioService.durationStream,
                  initialData: Duration.zero,
                  builder: (context, snapshot) {
                    final duration = snapshot.data ?? Duration.zero;
                    return _buildInfoItem(
                      context,
                      'TIMER',
                      _formatDuration(duration),
                      theme.colorScheme.secondary,
                    );
                  },
                ),
                StreamBuilder<double>(
                  stream: audioService.tempoStream,
                  initialData: 0.0,
                  builder: (context, snapshot) {
                    final tempo = snapshot.data ?? 0.0;
                    return _buildInfoItem(
                      context,
                      'TEMPO',
                      '${tempo.toStringAsFixed(1)} BPM',
                      theme.colorScheme.secondary,
                    );
                  },
                ),
              ],
            ),
            if (selectedMode == DisplayMode.microphone) ...[
              const SizedBox(height: 16),
              Divider(
                height: 1,
                thickness: 1,
                color: theme.dividerColor.withValues(alpha: 0.5),
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 10,
                    height: 10,
                    decoration: BoxDecoration(
                      color: color,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    label.toUpperCase(),
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.w600,
                      color: color,
                      letterSpacing: 0.8,
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildControlPanel(AudioService audioService) {
    final theme = Theme.of(context);
    final isRecording = audioService.isRecording;
    final bgColor =
        isRecording ? Colors.red.shade400 : theme.colorScheme.primary;
    const fgColor = Colors.white;
    final icon = isRecording ? Icons.stop_rounded : Icons.mic_rounded;
    final text = isRecording ? 'Stop' : 'Start';

    return SizedBox(
      width: 140,
      height: 48,
      child: ElevatedButton.icon(
        icon: Icon(icon, size: 22, color: fgColor),
        label: Text(
          text,
          style: const TextStyle(
            color: fgColor,
            fontWeight: FontWeight.bold,
            fontSize: 16,
          ),
        ),
        onPressed: () {
          if (isRecording) {
            audioService.stopRecording();
          } else {
            audioService.startRecording();
          }
        },
        style: ElevatedButton.styleFrom(
          backgroundColor: bgColor,
          foregroundColor: fgColor,
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 20),
        ),
      ),
    );
  }

  Widget _buildVerticalResetButton(VoidCallback onPressed) {
    final theme = Theme.of(context);
    return IconButton(
      icon: Icon(
        Icons.refresh,
        size: 20,
        color: theme.colorScheme.onSurfaceVariant,
      ),
      tooltip: 'Reset Counters',
      onPressed: onPressed,
      splashRadius: 20,
      padding: const EdgeInsets.all(8),
      constraints: const BoxConstraints(),
    );
  }

  Widget _buildCounterItem(
    BuildContext context,
    String label,
    int count,
    Color color,
  ) {
    final theme = Theme.of(context);
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label,
          style: theme.textTheme.labelSmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
            fontWeight: FontWeight.w500,
            letterSpacing: 0.8,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          count.toString(),
          style: theme.textTheme.headlineSmall?.copyWith(
            fontWeight: FontWeight.w600,
            color: color,
          ),
        ),
      ],
    );
  }

  Widget _buildInfoItem(
    BuildContext context,
    String label,
    String value,
    Color color,
  ) {
    final theme = Theme.of(context);
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label,
          style: theme.textTheme.labelSmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
            fontWeight: FontWeight.w500,
            letterSpacing: 0.8,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          value,
          style: theme.textTheme.headlineSmall?.copyWith(
            fontWeight: FontWeight.w600,
            color: color,
          ),
        ),
      ],
    );
  }

  String _formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    final minutes = twoDigits(duration.inMinutes.remainder(60));
    final seconds = twoDigits(duration.inSeconds.remainder(60));
    return '$minutes:$seconds';
  }

  void _showDeviceSelectionDialog() {
    final audioService = Provider.of<AudioService>(context, listen: false);
    final theme = Theme.of(context);

    showDialog(
      context: context,
      builder:
          (context) => ChangeNotifierProvider.value(
            value: audioService,
            child: Consumer<AudioService>(
              builder: (context, audioService, _) {
                return AlertDialog(
                  backgroundColor: theme.dialogTheme.backgroundColor,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  titlePadding: const EdgeInsets.fromLTRB(20, 20, 20, 10),
                  contentPadding: const EdgeInsets.fromLTRB(0, 0, 0, 10),
                  actionsPadding: const EdgeInsets.fromLTRB(10, 0, 10, 10),
                  title: Row(
                    children: [
                      Expanded(
                        child: Text(
                          'Select Input Device',
                          style: theme.textTheme.titleMedium,
                        ),
                      ),
                      if (audioService.isLoadingDevices)
                        Padding(
                          padding: const EdgeInsets.only(left: 8.0),
                          child: SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(
                              strokeWidth: 1.5,
                              valueColor: AlwaysStoppedAnimation<Color>(
                                theme.colorScheme.primary,
                              ),
                            ),
                          ),
                        )
                      else
                        IconButton(
                          icon: const Icon(Icons.refresh),
                          iconSize: 20,
                          tooltip: 'Refresh audio devices',
                          onPressed: () => audioService.loadInputDevices(),
                          color: theme.colorScheme.onSurfaceVariant,
                          splashRadius: 20,
                          constraints: const BoxConstraints(),
                          padding: const EdgeInsets.all(4),
                        ),
                    ],
                  ),
                  content: SizedBox(
                    width: double.maxFinite,
                    child:
                        audioService.inputDevices.isEmpty
                            ? Padding(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 20,
                                vertical: 30,
                              ),
                              child: Text(
                                'No audio devices found.',
                                textAlign: TextAlign.center,
                                style: theme.textTheme.bodyMedium?.copyWith(
                                  color: theme.colorScheme.onSurfaceVariant,
                                ),
                              ),
                            )
                            : ListView.builder(
                              shrinkWrap: true,
                              itemCount: audioService.inputDevices.length,
                              itemBuilder: (context, index) {
                                final device = audioService.inputDevices[index];
                                final bool isSelected =
                                    audioService.selectedDevice?.id ==
                                    device.id;

                                return ListTile(
                                  leading: Icon(
                                    Icons.mic_none_outlined,
                                    size: 20,
                                    color:
                                        isSelected
                                            ? theme.colorScheme.primary
                                            : theme
                                                .colorScheme
                                                .onSurfaceVariant,
                                  ),
                                  title: Text(
                                    device.label,
                                    style: theme.textTheme.bodyMedium?.copyWith(
                                      color:
                                          isSelected
                                              ? theme.colorScheme.primary
                                              : theme.colorScheme.onSurface,
                                      fontWeight:
                                          isSelected
                                              ? FontWeight.w500
                                              : FontWeight.normal,
                                    ),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                  trailing:
                                      isSelected
                                          ? Icon(
                                            Icons.check,
                                            size: 18,
                                            color: theme.colorScheme.primary,
                                          )
                                          : null,
                                  selected: isSelected,
                                  selectedTileColor: theme.colorScheme.primary
                                      .withValues(alpha: 0.1),
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  contentPadding: const EdgeInsets.symmetric(
                                    horizontal: 20,
                                    vertical: 0,
                                  ),
                                  dense: true,
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
                      child: Text(
                        'Cancel',
                        style: TextStyle(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),
          ),
    );
  }

  void _showInfoDialog() {
    final theme = Theme.of(context);
    showDialog(
      context: context,
      builder:
          (context) => Dialog(
            backgroundColor: theme.colorScheme.surface,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 16,
                    ),
                    decoration: BoxDecoration(
                      color: theme.colorScheme.primary.withValues(alpha: 0.1),
                      borderRadius: const BorderRadius.only(
                        topLeft: Radius.circular(16),
                        topRight: Radius.circular(16),
                      ),
                    ),
                    child: Row(
                      children: [
                        Icon(
                          Icons.info_outline,
                          color: theme.colorScheme.primary,
                          size: 24,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            'App Information',
                            style: theme.textTheme.titleLarge?.copyWith(
                              color: theme.colorScheme.primary,
                            ),
                          ),
                        ),
                        IconButton(
                          icon: Icon(
                            Icons.close,
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                          iconSize: 20,
                          onPressed: () => Navigator.pop(context),
                          padding: EdgeInsets.zero,
                          constraints: const BoxConstraints(),
                          splashRadius: 18,
                        ),
                      ],
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'About Breathing Monitor',
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Text(
                          'This app monitors your breathing patterns in real-time, detecting inhales, exhales, and silence periods.',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                        const SizedBox(height: 16),
                        _buildInfoRow(
                          theme,
                          Icons.mic_none_outlined,
                          'Tap the Start/Stop button to toggle monitoring.',
                        ),
                        const SizedBox(height: 10),
                        _buildInfoRow(
                          theme,
                          Icons.refresh_outlined,
                          'Tap the refresh icon to reset breath counts.',
                        ),
                        const SizedBox(height: 10),
                        _buildInfoRow(
                          theme,
                          Icons.tune_outlined,
                          'Use the options icon to change visualization style.',
                        ),
                      ],
                    ),
                  ),
                  Divider(
                    height: 1,
                    thickness: 1,
                    color: theme.dividerColor.withValues(alpha: 0.5),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Legend',
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceAround,
                          children: [
                            _buildLegendItem(
                              theme,
                              AppTheme.inhaleColor,
                              'Inhale',
                            ),
                            _buildLegendItem(
                              theme,
                              AppTheme.exhaleColor,
                              'Exhale',
                            ),
                            _buildLegendItem(
                              theme,
                              AppTheme.silenceColor,
                              'Silence',
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(20, 10, 20, 20),
                    child: TextButton(
                      onPressed: () => Navigator.pop(context),
                      style: TextButton.styleFrom(
                        backgroundColor: theme.colorScheme.secondaryContainer,
                        foregroundColor: theme.colorScheme.onSecondaryContainer,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
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

  Widget _buildInfoRow(ThemeData theme, IconData icon, String text) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 18, color: theme.colorScheme.onSurfaceVariant),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            text,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildLegendItem(ThemeData theme, Color color, String label) {
    return Column(
      children: [
        Container(
          width: 24,
          height: 24,
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.8),
            shape: BoxShape.circle,
            border: Border.all(color: color, width: 1.5),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: theme.textTheme.labelMedium?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }

  Future<void> _showAdvancedOptionsDialog(BuildContext context) async {
    DisplayMode tempSelectedMode = _selectedMode;
    bool tempShowSeparateCounters = _showSeparateCounters;
    final theme = Theme.of(context);

    final result = await showDialog<Map<String, dynamic>>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: theme.dialogTheme.backgroundColor,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          titlePadding: const EdgeInsets.fromLTRB(20, 20, 20, 10),
          contentPadding: const EdgeInsets.fromLTRB(0, 10, 0, 10),
          actionsPadding: const EdgeInsets.fromLTRB(10, 0, 10, 10),
          title: Text('Display Options', style: theme.textTheme.titleMedium),
          content: StatefulBuilder(
            builder: (BuildContext context, StateSetter setStateDialog) {
              return Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20.0,
                      vertical: 8.0,
                    ),
                    child: Text(
                      'Visualization Style',
                      style: theme.textTheme.labelLarge?.copyWith(
                        color: theme.colorScheme.primary,
                      ),
                    ),
                  ),
                  ...DisplayMode.values.map((mode) {
                    final bool isSelected = tempSelectedMode == mode;
                    return RadioListTile<DisplayMode>(
                      title: Text(
                        mode.label,
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color:
                              isSelected
                                  ? theme.colorScheme.primary
                                  : theme.colorScheme.onSurface,
                        ),
                      ),
                      value: mode,
                      groupValue: tempSelectedMode,
                      onChanged: (DisplayMode? value) {
                        if (value != null) {
                          setStateDialog(() {
                            tempSelectedMode = value;
                          });
                        }
                      },
                      secondary: Icon(
                        mode.icon,
                        color:
                            isSelected
                                ? theme.colorScheme.primary
                                : theme.colorScheme.onSurfaceVariant,
                      ),
                      activeColor: theme.colorScheme.primary,
                      controlAffinity: ListTileControlAffinity.trailing,
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 20,
                      ),
                      dense: true,
                    );
                  }),
                  const Divider(height: 20, thickness: 1),
                  Padding(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20.0,
                      vertical: 8.0,
                    ),
                    child: Text(
                      'Modes',
                      style: theme.textTheme.labelLarge?.copyWith(
                        color: theme.colorScheme.primary,
                      ),
                    ),
                  ),
                  SwitchListTile(
                    title: Text(
                      'Inhale and exhale',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: theme.colorScheme.onSurface,
                      ),
                    ),
                    value: tempShowSeparateCounters,
                    onChanged: (bool value) async {
                      setStateDialog(() {
                        tempShowSeparateCounters = value;
                      });

                      final audioService = Provider.of<AudioService>(
                        context,
                        listen: false,
                      );
                      await audioService.setOnlyExhaleMode(value);
                    },
                    secondary: Icon(
                      Icons.compare_arrows_outlined,
                      color:
                          tempShowSeparateCounters
                              ? theme.colorScheme.primary
                              : theme.colorScheme.onSurfaceVariant,
                    ),
                    activeColor: theme.colorScheme.primary,
                    contentPadding: const EdgeInsets.symmetric(horizontal: 20),
                    dense: true,
                  ),
                ],
              );
            },
          ),
          actions: <Widget>[
            TextButton(
              child: Text(
                'Cancel',
                style: TextStyle(color: theme.colorScheme.onSurfaceVariant),
              ),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
            TextButton(
              child: Text(
                'Apply',
                style: TextStyle(
                  color: theme.colorScheme.primary,
                  fontWeight: FontWeight.bold,
                ),
              ),
              onPressed: () {
                Navigator.of(context).pop({
                  'mode': tempSelectedMode,
                  'showSeparate': tempShowSeparateCounters,
                });
              },
            ),
          ],
        );
      },
    );

    if (result != null) {
      bool needsSetState = false;
      if (result['mode'] != null && result['mode'] != _selectedMode) {
        _selectedMode = result['mode'];
        needsSetState = true;
      }
      if (result['showSeparate'] != null &&
          result['showSeparate'] != _showSeparateCounters) {
        _showSeparateCounters = result['showSeparate'];
        needsSetState = true;
      }
      if (needsSetState) {
        setState(() {});
      }
    }
  }

  Widget _buildWidget(DisplayMode mode) {
    switch (mode) {
      case DisplayMode.microphone:
        return const MicrophoneVisualizationWidget();
      case DisplayMode.circular:
        return CircularVisualizationWidget(
          showSeparateCounters: _showSeparateCounters,
        );
    }
  }
}
