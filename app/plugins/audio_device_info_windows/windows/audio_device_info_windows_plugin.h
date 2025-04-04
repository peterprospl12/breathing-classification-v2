#ifndef FLUTTER_PLUGIN_AUDIO_DEVICE_INFO_WINDOWS_PLUGIN_H_
#define FLUTTER_PLUGIN_AUDIO_DEVICE_INFO_WINDOWS_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace audio_device_info_windows {

class AudioDeviceInfoWindowsPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar);

  AudioDeviceInfoWindowsPlugin(flutter::PluginRegistrarWindows* registrar);

  virtual ~AudioDeviceInfoWindowsPlugin();

  // Disallow copy and assign.
  AudioDeviceInfoWindowsPlugin(const AudioDeviceInfoWindowsPlugin&) = delete;
  AudioDeviceInfoWindowsPlugin& operator=(const AudioDeviceInfoWindowsPlugin&) = delete;

 private:
  // Called when a method is called on this plugin's channel from Dart.
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  // Get audio device information using Media Foundation
  void GetAudioDeviceInfo(const std::string& device_guid,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
      
  // Get all audio input devices
  void GetAudioInputDevices(
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  // The registrar for this plugin.
  flutter::PluginRegistrarWindows* registrar_;
  
  // The method channel used to communicate with the Dart side.
  std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> channel_;
};

}  // namespace audio_device_info_windows

#endif  // FLUTTER_PLUGIN_AUDIO_DEVICE_INFO_WINDOWS_PLUGIN_H_
