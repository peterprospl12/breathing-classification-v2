import 'package:flutter/material.dart';

enum DisplayMode {
  simulation,
  microphone
}

extension DisplayModeExtension on DisplayMode {
  String get label {
    switch (this) {
      case DisplayMode.simulation:
        return 'Simulation';
      case DisplayMode.microphone:
        return 'Microphone Data';
    }
  }

  IconData get icon {
    switch (this) {
      case DisplayMode.simulation:
        return Icons.graphic_eq;
      case DisplayMode.microphone:
        return Icons.mic;
    }
  }
}
