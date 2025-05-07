import 'package:flutter/material.dart';

class AppTheme {
  // Define base colors for reuse
  static const Color _primaryBlue = Color(0xFF3498DB); // Slightly softer blue
  static const Color _accentGreen = Color(0xFF2ECC71);
  static const Color _errorRed = Color(0xFFE74C3C);

  // Specific phase colors (can remain specific if needed)
  static const Color inhaleColor = _errorRed; // Red for inhale
  static const Color exhaleColor = _accentGreen; // Green for exhale
  static const Color silenceColor = _primaryBlue; // Blue for silence

  // Light Theme Colors
  static const Color _lightBackground = Color(0xFFF8F9FA); // Off-white
  static const Color _lightSurface = Color(0xFFFFFFFF); // White
  static const Color _lightOnPrimary = Colors.white;
  static const Color _lightOnSecondary = Colors.white;
  static const Color _lightOnBackground = Color(0xFF212529); // Dark grey
  static const Color _lightOnSurface = Color(0xFF212529);
  static const Color _lightOnError = Colors.white;
  static const Color _lightSurfaceVariant = Color(0xFFE9ECEF); // Light grey
  static const Color _lightOnSurfaceVariant = Color(0xFF495057); // Medium grey
  static const Color _lightOutline = Color(0xFFDEE2E6); // Lighter grey border

  // Dark Theme Colors - Updated with blueish tones
  static const Color _darkBackground = Color(0xFF1A1D24); // Very dark desaturated blue
  static const Color _darkSurface = Color(0xFF232730); // Slightly lighter dark blue/grey
  static const Color _darkOnPrimary = Colors.white;
  static const Color _darkOnSecondary = Colors.white;
  static const Color _darkOnBackground = Color(0xFFE1E3E8); // Light grey/blue text
  static const Color _darkOnSurface = Color(0xFFE1E3E8);
  static const Color _darkOnError = Colors.black;
  static const Color _darkSurfaceVariant = Color(0xFF303540); // Medium dark blue/grey
  static const Color _darkOnSurfaceVariant = Color(0xFFB0B3B8); // Lighter grey/blue text
  static const Color _darkOutline = Color(0xFF404550); // Darker grey/blue border

  // Typography
  static const fontFamily = 'Poppins'; // Consider importing a modern font

  static ThemeData get lightTheme {
    final baseTheme = ThemeData.light();
    return baseTheme.copyWith(
      useMaterial3: true,
      colorScheme: const ColorScheme.light(
        brightness: Brightness.light,
        primary: _primaryBlue,
        onPrimary: _lightOnPrimary,
        secondary: _accentGreen,
        onSecondary: _lightOnSecondary,
        error: _errorRed,
        onError: _lightOnError,
        background: _lightBackground,
        onBackground: _lightOnBackground,
        surface: _lightSurface,
        onSurface: _lightOnSurface,
        surfaceVariant: _lightSurfaceVariant,
        onSurfaceVariant: _lightOnSurfaceVariant,
        outline: _lightOutline,
        // Add other colors if needed: shadow, inverseSurface, etc.
      ),
      primaryColor: _primaryBlue,
      scaffoldBackgroundColor: _lightBackground,
      appBarTheme: const AppBarTheme(
        elevation: 0,
        backgroundColor: _lightSurface, // Use surface color for flat look
        foregroundColor: _lightOnSurface, // Text/icons based on surface
        iconTheme: IconThemeData(color: _lightOnSurfaceVariant),
        titleTextStyle: TextStyle(
          color: _lightOnSurface,
          fontSize: 18,
          fontWeight: FontWeight.w500,
          fontFamily: fontFamily, // Applied here
        ),
      ),
      cardTheme: CardTheme(
        elevation: 0, // Minimalist - no shadow
        color: _lightSurface,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: BorderSide(color: _lightOutline, width: 1), // Subtle border
        ),
        margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: _primaryBlue,
          foregroundColor: _lightOnPrimary,
          elevation: 1, // Very subtle elevation
          textStyle: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
            fontFamily: fontFamily, // Applied here
          ),
          padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 20),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)), // Pill shape
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: _primaryBlue,
          textStyle: const TextStyle(
            fontWeight: FontWeight.w600,
            fontFamily: fontFamily, // Applied here
          ),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
      ),
      dialogTheme: DialogTheme(
        backgroundColor: _lightSurface,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        titleTextStyle: baseTheme.textTheme.titleLarge?.copyWith(color: _lightOnSurface, fontWeight: FontWeight.w600),
        contentTextStyle: baseTheme.textTheme.bodyMedium?.copyWith(color: _lightOnSurfaceVariant),
      ),
      listTileTheme: ListTileThemeData(
        iconColor: _lightOnSurfaceVariant,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        selectedTileColor: _primaryBlue.withOpacity(0.1),
        dense: true,
      ),
      dividerTheme: DividerThemeData(
        color: _lightOutline,
        thickness: 1,
        space: 1,
      ),
      // Define TextTheme for consistency
      textTheme: _buildTextTheme(baseTheme.textTheme, _lightOnBackground, _lightOnSurfaceVariant),
    );
  }

  static ThemeData get darkTheme {
    final baseTheme = ThemeData.dark();
    return baseTheme.copyWith(
      useMaterial3: true,
      colorScheme: const ColorScheme.dark(
        brightness: Brightness.dark,
        primary: _primaryBlue,
        onPrimary: _darkOnPrimary,
        secondary: _accentGreen,
        onSecondary: _darkOnSecondary,
        error: _errorRed,
        onError: _darkOnError,
        background: _darkBackground, // Updated
        onBackground: _darkOnBackground, // Updated
        surface: _darkSurface, // Updated
        onSurface: _darkOnSurface, // Updated
        surfaceVariant: _darkSurfaceVariant, // Updated
        onSurfaceVariant: _darkOnSurfaceVariant, // Updated
        outline: _darkOutline, // Updated
        // Add other colors if needed
      ),
      primaryColor: _primaryBlue,
      scaffoldBackgroundColor: _darkBackground, // Updated
      appBarTheme: const AppBarTheme(
        elevation: 0,
        backgroundColor: _darkSurface, // Use surface color - Updated
        foregroundColor: _darkOnSurface, // Updated
        iconTheme: IconThemeData(color: _darkOnSurfaceVariant), // Updated
        titleTextStyle: TextStyle(
          color: _darkOnSurface, // Updated
          fontSize: 18,
          fontWeight: FontWeight.w500,
          fontFamily: fontFamily,
        ),
      ),
      cardTheme: CardTheme(
        elevation: 0,
        color: _darkSurface, // Updated
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: BorderSide(color: _darkOutline, width: 1), // Updated
        ),
        margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: _primaryBlue,
          foregroundColor: _darkOnPrimary,
          elevation: 1,
          textStyle: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
            fontFamily: fontFamily, // Applied here
          ),
          padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 20),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: _primaryBlue, // Keep primary for accent
          textStyle: const TextStyle(
            fontWeight: FontWeight.w600,
            fontFamily: fontFamily,
          ),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
      ),
      dialogTheme: DialogTheme(
        backgroundColor: _darkSurface, // Updated
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        titleTextStyle: baseTheme.textTheme.titleLarge?.copyWith(color: _darkOnSurface, fontWeight: FontWeight.w600), // Updated
        contentTextStyle: baseTheme.textTheme.bodyMedium?.copyWith(color: _darkOnSurfaceVariant), // Updated
      ),
      listTileTheme: ListTileThemeData(
        iconColor: _darkOnSurfaceVariant, // Updated
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        selectedTileColor: _primaryBlue.withOpacity(0.2),
        dense: true,
      ),
      dividerTheme: DividerThemeData(
        color: _darkOutline, // Updated
        thickness: 1,
        space: 1,
      ),
      textTheme: _buildTextTheme(baseTheme.textTheme, _darkOnBackground, _darkOnSurfaceVariant), // Updated text colors passed
    );
  }

  // Helper to build TextTheme with consistent font and colors
  static TextTheme _buildTextTheme(TextTheme base, Color displayColor, Color bodyColor) {
    return base.copyWith(
      displayLarge: base.displayLarge?.copyWith(fontFamily: fontFamily, color: displayColor),
      displayMedium: base.displayMedium?.copyWith(fontFamily: fontFamily, color: displayColor),
      displaySmall: base.displaySmall?.copyWith(fontFamily: fontFamily, color: displayColor),
      headlineLarge: base.headlineLarge?.copyWith(fontFamily: fontFamily, color: displayColor),
      headlineMedium: base.headlineMedium?.copyWith(fontFamily: fontFamily, color: displayColor),
      headlineSmall: base.headlineSmall?.copyWith(fontFamily: fontFamily, color: displayColor, fontWeight: FontWeight.w600),
      titleLarge: base.titleLarge?.copyWith(fontFamily: fontFamily, color: displayColor, fontWeight: FontWeight.w600),
      titleMedium: base.titleMedium?.copyWith(fontFamily: fontFamily, color: displayColor, fontWeight: FontWeight.w500),
      titleSmall: base.titleSmall?.copyWith(fontFamily: fontFamily, color: displayColor, fontWeight: FontWeight.w500),
      bodyLarge: base.bodyLarge?.copyWith(fontFamily: fontFamily, color: bodyColor),
      bodyMedium: base.bodyMedium?.copyWith(fontFamily: fontFamily, color: bodyColor),
      bodySmall: base.bodySmall?.copyWith(fontFamily: fontFamily, color: bodyColor.withOpacity(0.8)),
      labelLarge: base.labelLarge?.copyWith(fontFamily: fontFamily, color: displayColor, fontWeight: FontWeight.w600),
      labelMedium: base.labelMedium?.copyWith(fontFamily: fontFamily, color: bodyColor),
      labelSmall: base.labelSmall?.copyWith(fontFamily: fontFamily, color: bodyColor, letterSpacing: 0.5),
    ).apply(
      bodyColor: bodyColor,
      displayColor: displayColor,
    );
  }
}
