import 'package:flutter/material.dart';

enum DisplayMode {
  microphone
}

extension DisplayModeExtension on DisplayMode {
  String get label {
    switch (this) {
      case DisplayMode.microphone:
        return 'Microphone Data';
    }
  }

  IconData get icon {
    switch (this) {
      case DisplayMode.microphone:
        return Icons.mic;
    }
  }
}
