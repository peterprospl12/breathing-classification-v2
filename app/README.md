# Breathing Monitor App

A Flutter application for real-time breathing pattern analysis and visualization.

## Getting Started

### Prerequisites

- Flutter SDK (version 3.7.0 or higher)
- Dart SDK
- Android Studio / VS Code with Flutter/Dart plugins

### Supported Platforms
- Android
- iOS 
- Windows
- Web

### Installation

1. Clone the repository
2. Navigate to the app directory
3. Run `flutter pub get` to install dependencies
4. Run `flutter create .` to generate platform-specific projects
5. Run `flutter run -d <platform>` to start the application
   - Example: `flutter run -d windows` for Windows
   - Example: `flutter run -d chrome` for Web

## Features

- Real-time audio visualization
- Breathing phase classification (Inhale, Exhale, Silence)
- Breath counting
- User-friendly interface with animations

## Future Enhancements

- Integration with PyTorch model for more accurate breath classification
- User-specific model training
- Breathing exercises and guidance
- Historical data tracking and analysis
