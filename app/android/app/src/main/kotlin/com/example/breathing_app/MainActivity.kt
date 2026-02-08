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

    private fun logInfo(message: String) {
        Log.i(TAG, "🔍 [INFO] 🔍: $message")
    }

    private fun logError(message: String, e: Exception? = null) {
        Log.e(TAG, "❌ [ERROR] ❌: $message")
        e?.let {
            Log.e(TAG, "❌ [ERROR] ❌: ${e.message}")
        }
    }

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        logInfo("Starting breath classifier configuration")

        try {
            val assetManager = context.assets
            val flutterAssetsPath = "flutter_assets/assets/models"
            val files = assetManager.list(flutterAssetsPath)
            logInfo("📂 Files in $flutterAssetsPath: ${files?.joinToString(", ") ?: "no files"}")
        } catch (e: Exception) {
            logError("Error listing assets", e)
        }

        breathClassifierWrapper = BreathClassifierWrapper(applicationContext)
        isClassifierInitialized = breathClassifierWrapper.initialize()

        if (!isClassifierInitialized) {
            logError("Classifier initialization failed!")
        } else {
            logInfo("✅ Classifier initialized successfully")
        }

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            logInfo("📡 Received method call: ${call.method}")

            when (call.method) {
                "initializeModel" -> {
                    try {
                        val modelType = call.argument<String>("modelType") ?: "standard"
                        logInfo("Initializing model: $modelType")
                        val success = breathClassifierWrapper.initializeModel(modelType)
                        result.success(success)
                    } catch (e: Exception) {
                        logError("Error initializing model", e)
                        result.error("INIT_ERROR", e.message, null)
                    }
                }
                "classifyAudio" -> {
                    val initialized = breathClassifierWrapper.isInitialized()
                    logInfo("🔍 Classifier initialization status: $initialized")

                    if (!initialized) {
                        logError("Attempt to classify with uninitialized classifier")
                        result.error("INIT_FAILED", "Classifier not initialized", null)
                        return@setMethodCallHandler
                    }

                    try {
                        val audioData = call.argument<ByteArray>("audioData")
                        if (audioData == null) {
                            logError("No audio data for classification")
                            result.error("INVALID_ARG", "audioData argument is missing or null", null)
                            return@setMethodCallHandler
                        }

                        val floatData = convertInt16ByteArrayToFloatArray(audioData)
                        logInfo("🔊 Classifying audio data of size: ${floatData.size} floats")
                        val classificationResult = breathClassifierWrapper.classifyAudio(floatData)
                        logInfo("🏷️ Classification result: $classificationResult")
                        result.success(classificationResult)
                    } catch (e: Exception) {
                        logError("Error during classification", e)
                        result.error("CLASSIFICATION_ERROR", e.message, e.stackTraceToString())
                    }
                }
                "isInitialized" -> {
                    val initialized = breathClassifierWrapper.isInitialized()
                    logInfo("🔍 Initialization status query: $initialized")
                    result.success(initialized)
                }
                else -> {
                    logError("❓ Unknown method: ${call.method}")
                    result.notImplemented()
                }
            }
        }

        logInfo("✨ Breath classifier configuration completed")
    }

    private fun convertInt16ByteArrayToFloatArray(byteArray: ByteArray): FloatArray {
        if (byteArray.size % 2 != 0) {
            throw IllegalArgumentException("Byte array length must be even for Int16 conversion")
        }

        val floatArray = FloatArray(byteArray.size / 2)
        val buffer = java.nio.ByteBuffer.wrap(byteArray)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()

        for (i in 0 until floatArray.size) {
            // Normalize Int16 to Float32 [-1.0, 1.0]
            floatArray[i] = buffer.get(i) / 32768.0f
        }

        return floatArray
    }

    override fun onDestroy() {
        logInfo("🧹 Releasing classifier resources")
        if (::breathClassifierWrapper.isInitialized) {
            breathClassifierWrapper.close()
            logInfo("✅ Classifier resources released")
        }
        super.onDestroy()
    }
}