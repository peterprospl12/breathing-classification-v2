package com.example.breathing_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import androidx.annotation.NonNull
import android.util.Log
import java.io.File

class MainActivity: FlutterActivity() {
    private val CHANNEL = "breathing_classifier"
    private val TAG = "MainActivity"
    private lateinit var breathClassifierWrapper: BreathClassifierWrapper
    private var isClassifierInitialized = false

    // Funkcje do logowania, które będą widoczne w terminalu Flutter
    private fun logInfo(message: String) {
        Log.i(TAG, "BREATHING_APP_INFO: $message")
        println("BREATHING_APP_INFO: $message")
    }

    private fun logError(message: String, e: Exception? = null) {
        Log.e(TAG, "BREATHING_APP_ERROR: $message")
        println("BREATHING_APP_ERROR: $message")
        e?.let {
            Log.e(TAG, "BREATHING_APP_ERROR: ${e.message}")
            println("BREATHING_APP_ERROR: ${e.message}")
            println("BREATHING_APP_ERROR_STACK: ${e.stackTraceToString()}")
        }
    }

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // Inicjalizacja wrappera klasyfikatora
        logInfo("🚀 Rozpoczęcie konfiguracji klasyfikatora oddechów")

        // Sprawdzenie istnienia pliku modelu w assetach
        try {
            val assetManager = context.assets
            val flutterAssetsPath = "flutter_assets/assets/models"
            val files = assetManager.list(flutterAssetsPath)
            logInfo("📂 Pliki w $flutterAssetsPath: ${files?.joinToString(", ") ?: "brak plików"}")
        } catch (e: Exception) {
            logError("❌ Błąd podczas listowania assetów", e)
        }

        breathClassifierWrapper = BreathClassifierWrapper(applicationContext)
        isClassifierInitialized = breathClassifierWrapper.initialize()

        if (!isClassifierInitialized) {
            logError("❌ Inicjalizacja klasyfikatora nie powiodła się!")
        } else {
            logInfo("✅ Klasyfikator zainicjalizowany pomyślnie")
        }

        // Konfiguracja kanału metodowego dla komunikacji Flutter-Native
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            logInfo("📡 Otrzymano wywołanie metody: ${call.method}")

            when (call.method) {
                "classifyAudio" -> {
                    // Sprawdź czy inicjalizacja się powiodła - używamy metody z wrappera
                    val initialized = breathClassifierWrapper.isInitialized()
                    logInfo("🔍 Status inicjalizacji klasyfikatora: $initialized")

                    if (!initialized) {
                        logError("❌ Próba klasyfikacji z niezainicjalizowanym klasyfikatorem")
                        result.error("INIT_FAILED", "Classifier not initialized", null)
                        return@setMethodCallHandler
                    }

                    try {
                        val audioData = call.argument<ByteArray>("audioData")
                        if (audioData == null) {
                            logError("❌ Brak danych audio do klasyfikacji")
                            result.error("INVALID_ARG", "audioData argument is missing or null", null)
                            return@setMethodCallHandler
                        }

                        logInfo("🔊 Klasyfikacja danych audio o rozmiarze: ${audioData.size} bajtów")
                        val floatData = convertInt16ByteArrayToFloatArray(audioData)
                        val classificationResult = breathClassifierWrapper.classifyAudio(floatData)
                        logInfo("🏷️ Wynik klasyfikacji: $classificationResult")
                        result.success(classificationResult)
                    } catch (e: Exception) {
                        logError("❌ Błąd podczas klasyfikacji", e)
                        result.error("CLASSIFICATION_ERROR", e.message, e.stackTraceToString())
                    }
                }
                "isInitialized" -> {
                    val initialized = breathClassifierWrapper.isInitialized()
                    logInfo("🔍 Zapytanie o status inicjalizacji: $initialized")
                    result.success(initialized)
                }
                else -> {
                    logError("❓ Nieznana metoda: ${call.method}")
                    result.notImplemented()
                }
            }
        }

        logInfo("✨ Konfiguracja klasyfikatora oddechów zakończona")
    }

    /**
     * Konwertuje tablicę bajtów reprezentującą dźwięk w formacie Int16 na znormalizowaną tablicę Float32
     * @param byteArray dane wejściowe w formacie surowych bajtów Int16 (Little Endian)
     * @return FloatArray znormalizowane dane w zakresie [-1.0, 1.0]
     */
    private fun convertInt16ByteArrayToFloatArray(byteArray: ByteArray): FloatArray {
        if (byteArray.size % 2 != 0) {
            throw IllegalArgumentException("Byte array length must be even for Int16 conversion")
        }

        val floatArray = FloatArray(byteArray.size / 2)
        val buffer = java.nio.ByteBuffer.wrap(byteArray)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()

        for (i in 0 until floatArray.size) {
            // Normalizacja Int16 do Float32 [-1.0, 1.0]
            floatArray[i] = buffer.get(i) / 32768.0f
        }

        return floatArray
    }

    override fun onDestroy() {
        // Zwolnij zasoby klasyfikatora
        logInfo("🧹 Zwalnianie zasobów klasyfikatora")
        if (::breathClassifierWrapper.isInitialized) {
            breathClassifierWrapper.close()
            logInfo("✅ Zasoby klasyfikatora zwolnione")
        }
        super.onDestroy()
    }
}