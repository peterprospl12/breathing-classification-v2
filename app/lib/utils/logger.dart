import 'package:logging/logging.dart';
import 'package:flutter/foundation.dart';

class LoggerService {
  static bool _initialized = false;

  static void init() {
    if (_initialized) return;

    Logger.root.level = Level.ALL;
    Logger.root.onRecord.listen((record) {
      if (kDebugMode) {
        print('${record.time}: ${record.loggerName}: ${record.level.name}: ${record.message}');
      }
    });

    _initialized = true;
  }

  static Logger getLogger(String className) {
    if (!_initialized) init();
    return Logger(className);
  }
}