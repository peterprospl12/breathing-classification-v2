import 'package:flutter/material.dart';

enum DisplayMode {
  microphone,
  circular
}

extension DisplayModeExtension on DisplayMode {
  String get label {
    switch (this) {
      case DisplayMode.microphone:
        return 'Microphone Data';
      case DisplayMode.circular:
        return 'Circular Data';
    }
  }

  IconData get icon {
    switch (this) {
      case DisplayMode.microphone:
        return Icons.mic;
      case DisplayMode.circular:
        return Icons.circle;
    }
  }
}
