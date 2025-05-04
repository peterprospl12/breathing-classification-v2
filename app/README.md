# Breathing Monitor App

A Flutter application for real-time breathing pattern analysis and visualization.

## Getting Started

### Prerequisites

- Flutter SDK (version 3.7.0 or higher)
- Dart SDK
- VS Code with Flutter/Dart plugins

### Supported Platforms
- Android -> NOT YET
- iOS -> NOT YET
- Windows
- Web

### SDK Installation

After installing Flutter extension from the marketplace.
1. ctrl + shift + p
2. Type `doctor`
3. Choose `Flutter: Run flutter doctor`
4. Select installation folder for SDK, e.g. `C:\`
5. Add SDK to the PATH (VS Code notification)
6. Restart VS Code

### Installation

1. Clone the repository
2. Navigate to the app directory
3. Run `flutter pub get` to install dependencies
4. Run `flutter create .` to generate platform-specific projects
5. Run `flutter run` to start the application and choose from all existing sources
5. Run `flutter run -d <platform>` to start the application
   - Example: `flutter run -d windows` for Windows
   - Example: `flutter run -d chrome` for Web

### Android Logging

1. View filtered logs using:
   ```
   adb logcat -s BreathClassifierWrapper:* MainActivity:* *:S
   ```

2. If ADB command is not found:

   - Find the platform-tools folder:
     - Typically located in your Android SDK installation directory
     - Default location is often: `C:\Users\<YourUsername>\AppData\Local\Android\Sdk\platform-tools`
     - If using Android Studio, you can check SDK location in File > Settings > Appearance & Behavior > System Settings > Android SDK > Android SDK Location

   - Add the path to your PATH environment variable:
     - In Windows search, type "environment variables" and select "Edit the system environment variables"
     - Click "Environment Variables" button
     - In "System variables" (or "User variables"), find the "Path" variable, select it and click "Edit"
     - Click "New" and paste the full path to the platform-tools folder
     - Click "OK" on all dialog windows to save changes

   - Restart your terminal or VS Code for the PATH changes to take effect

   - Verify by running `adb version` in a new terminal window

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
