#include "audio_device_info_windows_plugin.h"

// This must be included before many other Windows headers.
#include <windows.h>

// Media Foundation headers
#include <mfapi.h>
#include <mfidl.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>
#include <Audioclient.h>

// COM and WMF
#include <combaseapi.h>
#include <wrl/client.h>

// Standard libraries
#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")

using Microsoft::WRL::ComPtr;

namespace audio_device_info_windows {

// Static registration function.
void AudioDeviceInfoWindowsPlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows* registrar) {
  auto plugin = std::make_unique<AudioDeviceInfoWindowsPlugin>(registrar);
  registrar->AddPlugin(std::move(plugin));
}

AudioDeviceInfoWindowsPlugin::AudioDeviceInfoWindowsPlugin(
    flutter::PluginRegistrarWindows* registrar)
    : registrar_(registrar) {
  channel_ =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          registrar->messenger(), "audio_device_info_windows",
          &flutter::StandardMethodCodec::GetInstance());
  channel_->SetMethodCallHandler(
      [this](const auto& call, auto result) {
        HandleMethodCall(call, std::move(result));
      });
      
  // Initialize Media Foundation
  HRESULT hr = MFStartup(MF_VERSION);
  if (FAILED(hr)) {
    OutputDebugString(L"Failed to initialize Media Foundation\n");
  }
}

AudioDeviceInfoWindowsPlugin::~AudioDeviceInfoWindowsPlugin() {
  // Shutdown Media Foundation
  MFShutdown();
}

void AudioDeviceInfoWindowsPlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue>& method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  if (method_call.method_name().compare("getAudioDeviceInfo") == 0) {
    const auto* arguments = std::get_if<flutter::EncodableMap>(method_call.arguments());
    if (arguments) {
      auto guid_it = arguments->find(flutter::EncodableValue("deviceGuid"));
      if (guid_it != arguments->end()) {
        const auto* device_guid = std::get_if<std::string>(&(guid_it->second));
        if (device_guid) {
          GetAudioDeviceInfo(*device_guid, std::move(result));
          return;
        }
      }
    }
    result->Error("INVALID_ARGUMENT", "Expected deviceGuid argument");
  } else if (method_call.method_name().compare("getAudioInputDevices") == 0) {
    GetAudioInputDevices(std::move(result));
  } else {
    result->NotImplemented();
  }
}

void AudioDeviceInfoWindowsPlugin::GetAudioDeviceInfo(
    const std::string& device_guid,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  
  // Initialize COM if not already initialized
  HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  bool need_uninit = SUCCEEDED(hr);
  
  // Create device enumerator
  ComPtr<IMMDeviceEnumerator> device_enumerator;
  hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, 
                       IID_PPV_ARGS(&device_enumerator));
  if (FAILED(hr)) {
    result->Error("DEVICE_ENUM_ERROR", "Failed to create device enumerator");
    if (need_uninit) CoUninitialize();
    return;
  }
  
  // Get the device with the specified ID
  ComPtr<IMMDevice> device;
  hr = device_enumerator->GetDevice(std::wstring(device_guid.begin(), device_guid.end()).c_str(), &device);
  if (FAILED(hr)) {
    result->Error("DEVICE_NOT_FOUND", "Device with specified GUID not found");
    if (need_uninit) CoUninitialize();
    return;
  }
  
  // Get the audio client interface
  ComPtr<IAudioClient> audio_client;
  hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, 
                       reinterpret_cast<void**>(audio_client.GetAddressOf()));
  if (FAILED(hr)) {
    result->Error("AUDIO_CLIENT_ERROR", "Failed to activate audio client");
    if (need_uninit) CoUninitialize();
    return;
  }
  
  // Get the device mix format (contains sample rate, channel count, etc.)
  WAVEFORMATEX* device_format;
  hr = audio_client->GetMixFormat(&device_format);
  if (FAILED(hr)) {
    result->Error("FORMAT_ERROR", "Failed to get device format");
    if (need_uninit) CoUninitialize();
    return;
  }
  
  // Extract needed information
  int sample_rate = device_format->nSamplesPerSec;
  int channel_count = device_format->nChannels;
  
  // Free the WAVEFORMATEX structure
  CoTaskMemFree(device_format);
  
  // Create a result map to return to Dart
  flutter::EncodableMap result_map;
  result_map[flutter::EncodableValue("sampleRate")] = flutter::EncodableValue(sample_rate);
  result_map[flutter::EncodableValue("channelCount")] = flutter::EncodableValue(channel_count);
  
  // Uninitialize COM if we initialized it
  if (need_uninit) {
    CoUninitialize();
  }
  
  // Return the result to Dart
  result->Success(flutter::EncodableValue(result_map));
}

// Lepszy sposób konwersji z wstring do string
std::string WideToUtf8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(), 
                                         nullptr, 0, nullptr, nullptr);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(), 
                        &strTo[0], size_needed, nullptr, nullptr);
    return strTo;
}

void AudioDeviceInfoWindowsPlugin::GetAudioInputDevices(
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  
  // Initialize COM if not already initialized
  HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  bool need_uninit = SUCCEEDED(hr);
  
  // Create device enumerator
  ComPtr<IMMDeviceEnumerator> device_enumerator;
  hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, 
                       IID_PPV_ARGS(&device_enumerator));
  if (FAILED(hr)) {
    result->Error("DEVICE_ENUM_ERROR", "Failed to create device enumerator");
    if (need_uninit) CoUninitialize();
    return;
  }

  // Get the default capture device
  ComPtr<IMMDevice> default_device;
  hr = device_enumerator->GetDefaultAudioEndpoint(eCapture, eConsole, &default_device);
  std::wstring default_device_id;
  if (SUCCEEDED(hr)) {
    LPWSTR temp_id = nullptr;
    default_device->GetId(&temp_id);
    if (temp_id) {
      default_device_id = std::wstring(temp_id);
      CoTaskMemFree(temp_id);
    }
  }
  
  // Enumerate capture devices (microphones)
  ComPtr<IMMDeviceCollection> device_collection;
  hr = device_enumerator->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &device_collection);
  if (FAILED(hr)) {
    result->Error("ENUM_ERROR", "Failed to enumerate audio endpoints");
    if (need_uninit) CoUninitialize();
    return;
  }
  
  // Get the count of devices
  UINT device_count = 0;
  hr = device_collection->GetCount(&device_count);
  if (FAILED(hr)) {
    result->Error("COUNT_ERROR", "Failed to get device count");
    if (need_uninit) CoUninitialize();
    return;
  }
  
  // Create a list to hold the devices
  flutter::EncodableList devices_list;
  
  // Iterate through the devices
  for (UINT i = 0; i < device_count; i++) {
    ComPtr<IMMDevice> device;
    hr = device_collection->Item(i, &device);
    if (FAILED(hr)) {
      continue;
    }
    
    // Get the device ID
    LPWSTR device_id_wstr = nullptr;
    hr = device->GetId(&device_id_wstr);
    if (FAILED(hr) || device_id_wstr == nullptr) {
      continue;
    }
    
    // Convert wide string to regular string
    std::wstring ws(device_id_wstr);
    std::string device_id = WideToUtf8(ws);
    
    // Get the device properties
    ComPtr<IPropertyStore> property_store;
    hr = device->OpenPropertyStore(STGM_READ, &property_store);
    if (FAILED(hr)) {
      CoTaskMemFree(device_id_wstr);
      continue;
    }
    
    // Get the friendly name property
    PROPVARIANT friendly_name_prop;
    PropVariantInit(&friendly_name_prop);
    hr = property_store->GetValue(PKEY_Device_FriendlyName, &friendly_name_prop);
    
    std::string friendly_name = "Unknown Device";
    if (SUCCEEDED(hr) && friendly_name_prop.vt == VT_LPWSTR) {
        std::wstring ws_name(friendly_name_prop.pwszVal);
        friendly_name = WideToUtf8(ws_name);
    }
    PropVariantClear(&friendly_name_prop);
    
    // Create a map for this device's information
    flutter::EncodableMap device_map;
    device_map[flutter::EncodableValue("id")] = flutter::EncodableValue(device_id);
    device_map[flutter::EncodableValue("name")] = flutter::EncodableValue(friendly_name);
    device_map[flutter::EncodableValue("isDefault")] = 
        flutter::EncodableValue(wcscmp(default_device_id, device_id_wstr) == 0);
    
    // Add to the list
    devices_list.push_back(flutter::EncodableValue(device_map));
    
    // Free the device ID string
    CoTaskMemFree(device_id_wstr);
  }
  
  // Uninitialize COM if we initialized it
  if (need_uninit) {
    CoUninitialize();
  }
  
  // Return the list of devices
  result->Success(flutter::EncodableValue(devices_list));
}

}  // namespace audio_device_info_windows
