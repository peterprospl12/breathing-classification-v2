# audio_device_info_windows

A Flutter plugin for Windows to retrieve audio device information such as sample rate and channel count.

## Features

- Get a list of available audio input devices
- Retrieve sample rate and channel count for a specific audio device
- Identify the default audio input device

## Installation

Add this to your package's pubspec.yaml file:

```yaml
dependencies:
  audio_device_info_windows: ^0.0.1
```

## Usage

```dart
import 'package:audio_device_info_windows/audio_device_info_windows.dart';

// Create an instance of the plugin
final audioDeviceInfoPlugin = AudioDeviceInfoWindows();

// Get all audio input devices
final devices = await audioDeviceInfoPlugin.getAudioInputDevices();
// Example output: 
// [
//   {
//     'id': '{0.0.1.00000000}.{7c5f8499-6f71-422a-9793-85d52c32dc78}',
//     'name': 'Microphone (Audio Device)',
//     'isDefault': true
//   },
//   ...
// ]

// Get information for a specific device
final deviceInfo = await audioDeviceInfoPlugin.getAudioDeviceInfo(
  '{0.0.1.00000000}.{7c5f8499-6f71-422a-9793-85d52c32dc78}'
);
// Example output:
// {
//   'sampleRate': 48000, 
//   'channelCount': 2
// }
```

## Requirements

- Windows 10 or later
- Flutter for Windows
