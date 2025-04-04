#include "include/audio_device_info_windows/audio_device_info_windows_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "audio_device_info_windows_plugin.h"

void AudioDeviceInfoWindowsPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  audio_device_info_windows::AudioDeviceInfoWindowsPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
