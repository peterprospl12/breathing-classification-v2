import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../theme/app_theme.dart';

class BreathCounter extends StatelessWidget {
  final int inhaleCount;
  final int exhaleCount;
  final VoidCallback onReset;

  const BreathCounter({
    super.key,
    required this.inhaleCount,
    required this.exhaleCount,
    required this.onReset,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Breath Counter',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  onPressed: onReset,
                  tooltip: 'Reset Counters',
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildCounter(context, 'INHALES', inhaleCount, AppTheme.inhaleColor),
                _buildCounter(context, 'EXHALES', exhaleCount, AppTheme.exhaleColor),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCounter(BuildContext context, String label, int count, Color color) {
    return Column(
      children: [
        Text(
          label,
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
            fontWeight: FontWeight.bold,
            color: Colors.grey,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          count.toString(),
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
            fontWeight: FontWeight.bold,
            color: color,
          ),
        )
        .animate(key: ValueKey(count))
        .scale(duration: 200.ms, curve: Curves.easeOutBack)
        .fadeIn(),
      ],
    );
  }
}
