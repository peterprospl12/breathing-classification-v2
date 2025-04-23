package com.example.breathing_app

import android.content.Context
import android.os.Build
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.locks.ReentrantLock

/**
 * Wrapper dla modelu ONNX klasyfikacji oddechów.
 * Obsługuje inicjalizację modelu, klasyfikację audio i zarządzanie zasobami.
 */
class BreathClassifierWrapper(private val context: Context) {
    private val TAG = "BreathClassifierWrapper"
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private val sessionLock = ReentrantLock() // Do bezpiecznego dostępu wielowątkowego

    companion object {
        const val MODEL_NAME = "breath_classifier_model_audio_input.onnx"
        const val MODEL_DATA_NAME = "breath_classifier_model_audio_input.onnx.data"
        // Różne możliwe ścieżki do modelu w zasobach
        private val POSSIBLE_ASSET_PATHS = arrayOf(
            "flutter_assets/assets/models/$MODEL_NAME",
            "assets/models/$MODEL_NAME",
            "models/$MODEL_NAME"
        )
        // Różne możliwe ścieżki do pliku danych modelu w zasobach
        private val POSSIBLE_DATA_ASSET_PATHS = arrayOf(
            "flutter_assets/assets/models/$MODEL_DATA_NAME",
            "assets/models/$MODEL_DATA_NAME",
            "models/$MODEL_DATA_NAME"
        )
    }

    // Funkcja do wyświetlania logów, które będą lepiej widoczne w terminalu Flutter
    private fun logInfo(message: String) {
        Log.i(TAG, "BREATH_CLASSIFIER_INFO: $message")
        println("BREATH_CLASSIFIER_INFO: $message")
    }

    private fun logError(message: String, e: Exception? = null) {
        Log.e(TAG, "BREATH_CLASSIFIER_ERROR: $message")
        println("BREATH_CLASSIFIER_ERROR: $message")
        e?.let {
            Log.e(TAG, "BREATH_CLASSIFIER_ERROR: ${e.message}")
            println("BREATH_CLASSIFIER_ERROR: ${e.message}")
            println("BREATH_CLASSIFIER_ERROR_STACK: ${e.stackTraceToString()}")
        }
    }

    /**
     * Kopiuje model z assets do lokalnego pliku i inicjalizuje sesję ONNX Runtime
     * @return Boolean - czy inicjalizacja zakończyła się powodzeniem
     */
    fun initialize(): Boolean {
        logInfo("➡️ Rozpoczęcie inicjalizacji klasyfikatora")
        return try {
            logDeviceInfo() // Logowanie informacji o urządzeniu do debugowania
            
            val modelFile = File(context.filesDir, MODEL_NAME)
            val modelDataFile = File(context.filesDir, MODEL_DATA_NAME)
            
            // Wypisz zawartość katalogu assets do debugowania
            logInfo("📁 Zawartość głównego katalogu assets:")
            listAvailableAssets("")
            logInfo("📁 Zawartość katalogu flutter_assets:")
            listAvailableAssets("flutter_assets")
            logInfo("📁 Zawartość katalogu flutter_assets/assets:")
            listAvailableAssets("flutter_assets/assets")
            logInfo("📁 Zawartość katalogu flutter_assets/assets/models:")
            listAvailableAssets("flutter_assets/assets/models")
            
            // Kopiowanie pliku modelu
            if (!modelFile.exists()) {
                logInfo("📦 Model nie istnieje lokalnie, kopiowanie z assets...")
                copyModelFromAssets(modelFile)
                
                // Upewnij się, że plik faktycznie został utworzony
                if (!modelFile.exists() || modelFile.length() == 0L) {
                    throw Exception("Nie udało się prawidłowo skopiować pliku modelu")
                }
            }
            
            // Kopiowanie pliku danych modelu
            if (!modelDataFile.exists()) {
                logInfo("📦 Plik danych modelu nie istnieje lokalnie, kopiowanie z assets...")
                copyModelDataFromAssets(modelDataFile)
            }
            
            // Sprawdź i wypisz stan pliku modelu
            logInfo("📄 Ścieżka do modelu: ${modelFile.absolutePath}, Rozmiar: ${modelFile.length()} bajtów")
            logInfo("📄 Model istnieje: ${modelFile.exists()}, Można czytać: ${modelFile.canRead()}")
            
            if (modelDataFile.exists()) {
                logInfo("📄 Ścieżka do danych modelu: ${modelDataFile.absolutePath}, Rozmiar: ${modelDataFile.length()} bajtów")
                logInfo("📄 Plik danych modelu istnieje: ${modelDataFile.exists()}, Można czytać: ${modelDataFile.canRead()}")
            } else {
                logInfo("⚠️ Plik danych modelu (.data) nie istnieje, kontynuujemy bez niego")
            }
            
            // Wypisz zawartość katalogu z modelem
            val filesDir = context.filesDir
            logInfo("📁 Zawartość katalogu filesDir (${filesDir.absolutePath}):")
            filesDir.listFiles()?.forEach { file ->
                logInfo("   - ${file.name}: ${file.length()} bajtów")
            }
            
            // Próbujemy utworzyć sesję z pliku
            val success = createSessionFromFile(modelFile)
            
            // Jeśli nie udało się z pliku, próbujemy utworzyć sesję bezpośrednio z bytów modelu
            if (!success) {
                logInfo("🔄 Próba utworzenia sesji bezpośrednio z bajtów modelu...")
                createSessionFromBytes()
            }
            
            // Sprawdź czy sesja została utworzona
            val initialized = session != null
            if (initialized) {
                logInfo("✅ Sesja utworzona pomyślnie")
            } else {
                logError("❌ Nie udało się utworzyć sesji ONNX")
            }
            
            initialized
        } catch (e: Exception) {
            logError("❌ Błąd inicjalizacji modelu", e)
            false
        }
    }

    /**
     * Próbuje utworzyć sesję ONNX z pliku
     */
    private fun createSessionFromFile(modelFile: File): Boolean {
        return try {
            // Konfiguracja opcji sesji
            val opts = OrtSession.SessionOptions()
            opts.setIntraOpNumThreads(2) // Wykorzystaj maksymalnie 2 wątki do obliczeń
            
            // Utwórz sesję
            sessionLock.lock()
            try {
                logInfo("🔄 Tworzenie sesji ONNX Runtime z pliku...")
                session = env.createSession(modelFile.absolutePath, opts)
                logInfo("✅ Model ONNX załadowany pomyślnie z pliku")
                
                // Wypisz informacje o wejściach i wyjściach modelu
                session?.let { sess ->
                    logInfo("📊 Informacje o modelu:")
                    logInfo("📥 Wejścia modelu: ${sess.inputNames.joinToString(", ")}")
                    logInfo("📤 Wyjścia modelu: ${sess.outputNames.joinToString(", ")}")
                }
                true
            } finally {
                sessionLock.unlock()
            }
        } catch (e: Exception) {
            logError("⚠️ Nie udało się utworzyć sesji z pliku: ${e.message}", e)
            false
        }
    }

    /**
     * Próbuje utworzyć sesję ONNX bezpośrednio z bajtów modelu
     */
    private fun createSessionFromBytes(): Boolean {
        return try {
            var modelBytes: ByteArray? = null
            
            // Próbujemy każdą możliwą ścieżkę, dopóki odczyt się nie powiedzie
            for (assetPath in POSSIBLE_ASSET_PATHS) {
                try {
                    logInfo("🔄 Próba odczytu modelu z: $assetPath")
                    context.assets.open(assetPath).use { input ->
                        modelBytes = input.readBytes()
                    }
                    logInfo("✅ Odczytano model z $assetPath, rozmiar: ${modelBytes?.size ?: 0} bajtów")
                    break  // Jeśli odczyt się powiódł, kończymy pętlę
                } catch (e: Exception) {
                    logError("❌ Nie można odczytać modelu z ścieżki: $assetPath - ${e.message}")
                }
            }
            
            if (modelBytes == null || modelBytes!!.isEmpty()) {
                logError("❌ Nie udało się odczytać modelu z żadnej ścieżki")
                return false
            }
            
            // Konfiguracja opcji sesji
            val opts = OrtSession.SessionOptions()
            opts.setIntraOpNumThreads(2)
            
            sessionLock.lock()
            try {
                logInfo("🔄 Tworzenie sesji ONNX Runtime z bajtów...")
                session = env.createSession(modelBytes!!, opts)
                logInfo("✅ Model ONNX załadowany pomyślnie z bajtów")
                
                // Wypisz informacje o wejściach i wyjściach modelu
                session?.let { sess ->
                    logInfo("📊 Informacje o modelu:")
                    logInfo("📥 Wejścia modelu: ${sess.inputNames.joinToString(", ")}")
                    logInfo("📤 Wyjścia modelu: ${sess.outputNames.joinToString(", ")}")
                }
                true
            } finally {
                sessionLock.unlock()
            }
        } catch (e: Exception) {
            logError("❌ Nie udało się utworzyć sesji z bajtów modelu", e)
            false
        }
    }

    /**
     * Kopiuje plik modelu z zasobów do lokalnego pliku
     */
    private fun copyModelFromAssets(modelFile: File) {
        var copied = false

        // Próbujemy każdą możliwą ścieżkę, dopóki kopiowanie się nie powiedzie
        for (assetPath in POSSIBLE_ASSET_PATHS) {
            try {
                logInfo("🔄 Próba kopiowania modelu z: $assetPath")
                
                context.assets.open(assetPath).use { input ->
                    FileOutputStream(modelFile).use { output ->
                        val bytes = input.readBytes()
                        output.write(bytes)
                        output.flush()
                        logInfo("✅ Skopiowano ${bytes.size} bajtów")
                    }
                }
                
                logInfo("✅ Model skopiowany z $assetPath do: ${modelFile.absolutePath}, rozmiar: ${modelFile.length()}")
                copied = true
                break  // Jeśli kopiowanie się powiodło, kończymy pętlę
                
            } catch (e: Exception) {
                logError("❌ Nie można skopiować modelu z ścieżki: $assetPath - ${e.message}")
                // Kontynuuj do następnej ścieżki
            }
        }
        
        if (!copied) {
            // Jeśli żadna ścieżka nie zadziałała, log błąd
            logError("❌ Nie udało się skopiować modelu z żadnej ścieżki.")
        }
    }

    /**
     * Kopiuje plik danych modelu (.data) z zasobów do lokalnego pliku
     */
    private fun copyModelDataFromAssets(dataFile: File) {
        var copied = false

        // Próbujemy każdą możliwą ścieżkę, dopóki kopiowanie się nie powiedzie
        for (assetPath in POSSIBLE_DATA_ASSET_PATHS) {
            try {
                logInfo("🔄 Próba kopiowania danych modelu z: $assetPath")
                
                context.assets.open(assetPath).use { input ->
                    FileOutputStream(dataFile).use { output ->
                        val bytes = input.readBytes()
                        output.write(bytes)
                        output.flush()
                        logInfo("✅ Skopiowano ${bytes.size} bajtów danych modelu")
                    }
                }
                
                logInfo("✅ Dane modelu skopiowane z $assetPath do: ${dataFile.absolutePath}, rozmiar: ${dataFile.length()}")
                copied = true
                break  // Jeśli kopiowanie się powiodło, kończymy pętlę
                
            } catch (e: Exception) {
                logError("❌ Nie można skopiować danych modelu z ścieżki: $assetPath - ${e.message}")
                // Kontynuuj do następnej ścieżki
            }
        }
        
        if (!copied) {
            // Jeśli żadna ścieżka nie zadziałała, log błąd
            logError("❌ Nie udało się skopiować danych modelu z żadnej ścieżki. Kontynuujemy bez nich.")
        }
    }

    /**
     * Wypisuje informacje o urządzeniu pomocne przy debugowaniu
     */
    private fun logDeviceInfo() {
        logInfo("📱 Informacje o urządzeniu:")
        logInfo("   - Producent: ${Build.MANUFACTURER}")
        logInfo("   - Model: ${Build.MODEL}")
        logInfo("   - Wersja SDK: ${Build.VERSION.SDK_INT}")
        logInfo("   - Wersja systemu: ${Build.VERSION.RELEASE}")
        logInfo("   - Dostępna pamięć: ${Runtime.getRuntime().maxMemory() / (1024 * 1024)} MB")
    }

    /**
     * Pomocnicza metoda do listowania plików w zasobach
     */
    private fun listAvailableAssets(path: String) {
        try {
            val files = context.assets.list(path)
            if (files.isNullOrEmpty()) {
                logInfo("   📁 Ścieżka '$path' jest pusta lub nie istnieje")
            } else {
                logInfo("   📁 Pliki w '$path': ${files.joinToString(", ")}")
            }
        } catch (e: Exception) {
            logError("❌ Nie można wylistować plików w '$path': ${e.message}")
        }
    }

    /**
     * Klasyfikuje dane audio i zwraca indeks klasy (0-wydech, 1-wdech, 2-cisza)
     * Metoda jest thread-safe
     * @param audioData FloatArray surowych znormalizowanych danych audio [-1,1]
     * @return Int indeks klasy (0, 1, lub 2)
     */
    fun classifyAudio(audioData: FloatArray): Int {
        sessionLock.lock()
        try {
            val sess = session ?: throw IllegalStateException("Sesja nie zainicjalizowana. Wywołaj initialize() najpierw.")
            val inputName = sess.inputNames.first()
            val shape = longArrayOf(1, audioData.size.toLong())

            // Utwórz tensor wejściowy
            val inputBuffer = FloatBuffer.wrap(audioData)
            val tensor = OnnxTensor.createTensor(env, inputBuffer, shape)
            
            // Uruchom model i pobierz wyniki
            return tensor.use { input ->
                sess.run(mapOf(inputName to input)).use { output ->
                    val outputTensor = output.get(0) as OnnxTensor
                    val tensorInfo = outputTensor.info
                    
                    logInfo("Wymiary tensora wyjściowego: ${tensorInfo.shape.contentToString()}")
                    
                    // Obsługa tensora wielowymiarowego [1, time_steps, num_classes]
                    val result = when (val value = outputTensor.value) {
                        // Obsługa tensora 3D jako tablicy 2D tablic 1D
                        is Array<*> -> {
                            if (value.isNotEmpty() && value[0] is Array<*>) {
                                // Tensor ma kształt [1, time_steps, num_classes]
                                val timeSteps = value[0] as Array<*>
                                
                                // Utworzenie tablicy wyników dla każdego kroku czasowego
                                val predictions = mutableListOf<Int>()
                                
                                for (step in timeSteps) {
                                    when (step) {
                                        is FloatArray -> {
                                            // Znajdź indeks z maksymalną wartością
                                            var maxIndex = 0
                                            var maxValue = step[0]
                                            
                                            for (i in 1 until step.size) {
                                                if (step[i] > maxValue) {
                                                    maxValue = step[i]
                                                    maxIndex = i
                                                }
                                            }
                                            
                                            predictions.add(maxIndex)
                                        }
                                        is DoubleArray -> {
                                            // Znajdź indeks z maksymalną wartością
                                            var maxIndex = 0
                                            var maxValue = step[0]
                                            
                                            for (i in 1 until step.size) {
                                                if (step[i] > maxValue) {
                                                    maxValue = step[i]
                                                    maxIndex = i
                                                }
                                            }
                                            
                                            predictions.add(maxIndex)
                                        }
                                        else -> {
                                            logError("Nieznany typ kroku czasowego: ${step?.javaClass}")
                                        }
                                    }
                                }
                                
                                // Znajdź najczęściej występującą klasę
                                val counts = predictions.groupingBy { it }.eachCount()
                                counts.maxByOrNull { it.value }?.key ?: 2
                            } else {
                                logError("Tensor ma nieoczekiwaną strukturę: $value")
                                2 // Domyślnie cisza
                            }
                        }
                        else -> {
                            logError("Nieznany format tensora: ${value?.javaClass}")
                            2 // Domyślnie cisza
                        }
                    }
                    
                    logInfo("Wynik klasyfikacji: $result")
                    result
                }
            }
        } catch (e: Exception) {
            logError("Błąd w czasie klasyfikacji", e)
            return 2  // Domyślnie zwróć klasę 'silence' w przypadku błędu
        } finally {
            sessionLock.unlock()
        }
    }

    /**
     * Sprawdza, czy klasyfikator został poprawnie zainicjalizowany
     */
    fun isInitialized(): Boolean {
        return session != null
    }

    /**
     * Zwalnia zasoby ONNX Runtime
     */
    fun close() {
        try {
            sessionLock.lock()
            session?.close()
            sessionLock.unlock()
            
            env.close()
            logInfo("Zasoby zwolnione")
        } catch (e: Exception) {
            logError("Błąd podczas zwalniania zasobów", e)
        }
    }
}