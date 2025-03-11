import 'package:flutter/material.dart';

class AppTheme {
  // Primary colors
  static const Color primaryColor = Color(0xFF3498DB);
  static const Color accentColor = Color(0xFF2ECC71);
  
  // Status colors
  static const Color inhaleColor = Color(0xFFE74C3C);     // Red
  static const Color exhaleColor = Color(0xFF2ECC71);     // Green
  static const Color silenceColor = Color(0xFF3498DB);    // Blue
  
  // Background colors
  static const Color darkBackgroundColor = Color(0xFF121212);
  static const Color lightBackgroundColor = Color(0xFFF5F5F5);
  
  // Card colors
  static const Color darkCardColor = Color(0xFF1E1E1E);
  static const Color lightCardColor = Color(0xFFFFFFFF);

  // Typography
  static const fontFamily = 'Poppins';

  static ThemeData get lightTheme {
    return ThemeData(
      colorScheme: ColorScheme.light(
        primary: primaryColor,
        secondary: accentColor,
        background: lightBackgroundColor,
      ),
      primaryColor: primaryColor,
      fontFamily: fontFamily,
      scaffoldBackgroundColor: lightBackgroundColor,
      appBarTheme: const AppBarTheme(
        elevation: 0,
        backgroundColor: primaryColor,
        foregroundColor: Colors.white,
      ),
      cardTheme: CardTheme(
        color: lightCardColor,
        elevation: 4,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryColor,
          foregroundColor: Colors.white,
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          padding: const EdgeInsets.symmetric(vertical: 15, horizontal: 25),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
        ),
      ),
    );
  }

  static ThemeData get darkTheme {
    return ThemeData(
      colorScheme: ColorScheme.dark(
        primary: primaryColor,
        secondary: accentColor,
        background: darkBackgroundColor,
      ),
      primaryColor: primaryColor,
      fontFamily: fontFamily,
      scaffoldBackgroundColor: darkBackgroundColor,
      appBarTheme: const AppBarTheme(
        elevation: 0,
        backgroundColor: darkCardColor,
        foregroundColor: Colors.white,
      ),
      cardTheme: CardTheme(
        color: darkCardColor,
        elevation: 4,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryColor,
          foregroundColor: Colors.white,
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          padding: const EdgeInsets.symmetric(vertical: 15, horizontal: 25),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
        ),
      ),
    );
  }
}
