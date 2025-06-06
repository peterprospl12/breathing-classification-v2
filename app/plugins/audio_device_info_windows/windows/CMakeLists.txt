cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME "audio_device_info_windows")
project(${PROJECT_NAME} LANGUAGES CXX)

# This value is used when generating builds using this plugin, so it must
# not be changed
set(PLUGIN_NAME "audio_device_info_windows_plugin")

add_library(${PLUGIN_NAME} SHARED
  "audio_device_info_windows_plugin.cpp"
  "audio_device_info_windows_plugin.h"
  "audio_device_info_windows_plugin_c_api.cpp"
  "audio_device_info_windows_plugin_c_api.h"
)

apply_standard_settings(${PLUGIN_NAME})
set_target_properties(${PLUGIN_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(${PLUGIN_NAME} PRIVATE 
  flutter
  flutter_wrapper_plugin
  mfplat.lib
  mf.lib
  mfuuid.lib
  ole32.lib
)

# List of absolute paths to libraries that should be bundled with the plugin
set(audio_device_info_windows_bundled_libraries
  ""
  PARENT_SCOPE
)
