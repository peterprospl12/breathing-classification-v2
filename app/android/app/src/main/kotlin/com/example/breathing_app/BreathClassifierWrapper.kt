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


class BreathClassifierWrapper(private val context: Context) {
    private val TAG = "BreathClassifierWrapper"
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private val sessionLock = ReentrantLock()

    companion object {
        const val MODEL_NAME = "breath_classifier_model_audio_input.onnx"
        const val MODEL_DATA_NAME = "breath_classifier_model_audio_input.onnx.data"
        private val POSSIBLE_ASSET_PATHS = arrayOf(
            "flutter_assets/assets/models/$MODEL_NAME",
            "assets/models/$MODEL_NAME",
            "models/$MODEL_NAME"
        )
        private val POSSIBLE_DATA_ASSET_PATHS = arrayOf(
            "flutter_assets/assets/models/$MODEL_DATA_NAME",
            "assets/models/$MODEL_DATA_NAME",
            "models/$MODEL_DATA_NAME"
        )
    }

    private fun logInfo(message: String) {
        Log.i(TAG, "🔍 [INFO] 🔍: $message")
    }

    private fun logError(message: String, e: Exception? = null) {
        Log.e(TAG, "❌ [ERROR] ❌: $message")
        e?.let {
            Log.e(TAG, "❌ [ERROR] ❌: ${e.message}")
            Log.e(TAG, "❌ [ERROR_STACK] ❌: ${e.stackTraceToString()}")
        }
    }

    fun initialize(): Boolean {
        logInfo("➡️ Starting classifier initialization")
        return try {
            logDeviceInfo()

            val modelFile = File(context.filesDir, MODEL_NAME)
            val modelDataFile = File(context.filesDir, MODEL_DATA_NAME)

            logInfo("📁 Main assets directory content:")
            listAvailableAssets("")
            logInfo("📁 Flutter_assets directory content:")
            listAvailableAssets("flutter_assets")
            logInfo("📁 Flutter_assets/assets directory content:")
            listAvailableAssets("flutter_assets/assets")
            logInfo("📁 Flutter_assets/assets/models directory content:")
            listAvailableAssets("flutter_assets/assets/models")

            if (!modelFile.exists()) {
                logInfo("📦 Model doesn't exist locally, copying from assets...")
                copyModelFromAssets(modelFile)

                if (!modelFile.exists() || modelFile.length() == 0L) {
                    throw Exception("Failed to properly copy the model file")
                }
            }

            if (!modelDataFile.exists()) {
                logInfo("📦 Model data file doesn't exist locally, copying from assets...")
                copyModelDataFromAssets(modelDataFile)
            }

            logInfo("📄 Model path: ${modelFile.absolutePath}, Size: ${modelFile.length()} bytes")
            logInfo("📄 Model exists: ${modelFile.exists()}, Can read: ${modelFile.canRead()}")

            if (modelDataFile.exists()) {
                logInfo("📄 Model data path: ${modelDataFile.absolutePath}, Size: ${modelDataFile.length()} bytes")
                logInfo("📄 Model data file exists: ${modelDataFile.exists()}, Can read: ${modelDataFile.canRead()}")
            } else {
                logInfo("⚠️ Model data file (.data) doesn't exist, continuing without it")
            }

            val filesDir = context.filesDir
            logInfo("📁 Content of filesDir directory (${filesDir.absolutePath}):")
            filesDir.listFiles()?.forEach { file ->
                logInfo("   - ${file.name}: ${file.length()} bytes")
            }

            val success = createSessionFromFile(modelFile)

            if (!success) {
                logInfo("🔄 Attempting to create session directly from model bytes...")
                createSessionFromBytes()
            }

            val initialized = session != null
            if (initialized) {
                logInfo("✅ Session created successfully")
            } else {
                logError("Failed to create ONNX session")
            }

            initialized
        } catch (e: Exception) {
            logError("Error initializing model", e)
            false
        }
    }

    private fun createSessionFromFile(modelFile: File): Boolean {
        return try {
            val opts = OrtSession.SessionOptions()
            opts.setIntraOpNumThreads(2)

            // Create session
            sessionLock.lock()
            try {
                logInfo("🔄 Creating ONNX Runtime session from file...")
                session = env.createSession(modelFile.absolutePath, opts)
                logInfo("✅ ONNX model loaded successfully from file")

                session?.let { sess ->
                    logInfo("📊 Model information:")
                    logInfo("📥 Model inputs: ${sess.inputNames.joinToString(", ")}")
                    logInfo("📤 Model outputs: ${sess.outputNames.joinToString(", ")}")
                }
                true
            } finally {
                sessionLock.unlock()
            }
        } catch (e: Exception) {
            logError("⚠️ Failed to create session from file: ${e.message}", e)
            false
        }
    }

    private fun createSessionFromBytes(): Boolean {
        return try {
            var modelBytes: ByteArray? = null

            for (assetPath in POSSIBLE_ASSET_PATHS) {
                try {
                    logInfo("🔄 Attempting to read model from: $assetPath")
                    context.assets.open(assetPath).use { input ->
                        modelBytes = input.readBytes()
                    }
                    logInfo("✅ Read model from $assetPath, size: ${modelBytes?.size ?: 0} bytes")
                    break
                } catch (e: Exception) {
                    logError("Cannot read model from path: $assetPath - ${e.message}")
                }
            }

            if (modelBytes == null || modelBytes!!.isEmpty()) {
                logError("Failed to read model from any path")
                return false
            }

            val opts = OrtSession.SessionOptions()
            opts.setIntraOpNumThreads(2)

            sessionLock.lock()
            try {
                logInfo("🔄 Creating ONNX Runtime session from bytes...")
                session = env.createSession(modelBytes!!, opts)
                logInfo("✅ ONNX model loaded successfully from bytes")

                session?.let { sess ->
                    logInfo("📊 Model information:")
                    logInfo("📥 Model inputs: ${sess.inputNames.joinToString(", ")}")
                    logInfo("📤 Model outputs: ${sess.outputNames.joinToString(", ")}")
                }
                true
            } finally {
                sessionLock.unlock()
            }
        } catch (e: Exception) {
            logError("Failed to create session from model bytes", e)
            false
        }
    }

    private fun copyModelFromAssets(modelFile: File) {
        var copied = false

        for (assetPath in POSSIBLE_ASSET_PATHS) {
            try {
                logInfo("🔄 Attempting to copy model from: $assetPath")

                context.assets.open(assetPath).use { input ->
                    FileOutputStream(modelFile).use { output ->
                        val bytes = input.readBytes()
                        output.write(bytes)
                        output.flush()
                        logInfo("✅ Copied ${bytes.size} bytes")
                    }
                }

                logInfo("✅ Model copied from $assetPath to: ${modelFile.absolutePath}, size: ${modelFile.length()}")
                copied = true
                break

            } catch (e: Exception) {
                logError("Cannot copy model from path: $assetPath - ${e.message}")
            }
        }

        if (!copied) {
            logError("Failed to copy model from any path.")
        }
    }

    private fun copyModelDataFromAssets(dataFile: File) {
        var copied = false

        for (assetPath in POSSIBLE_DATA_ASSET_PATHS) {
            try {
                logInfo("🔄 Attempting to copy model data from: $assetPath")

                context.assets.open(assetPath).use { input ->
                    FileOutputStream(dataFile).use { output ->
                        val bytes = input.readBytes()
                        output.write(bytes)
                        output.flush()
                        logInfo("✅ Copied ${bytes.size} bytes of model data")
                    }
                }

                logInfo("✅ Model data copied from $assetPath to: ${dataFile.absolutePath}, size: ${dataFile.length()}")
                copied = true
                break

            } catch (e: Exception) {
                logError(" Cannot copy model data from path: $assetPath - ${e.message}")
            }
        }

        if (!copied) {
            logError("Failed to copy model data from any path. Continuing without it.")
        }
    }

    private fun logDeviceInfo() {
        logInfo("📱 Device information:")
        logInfo("   - Manufacturer: ${Build.MANUFACTURER}")
        logInfo("   - Model: ${Build.MODEL}")
        logInfo("   - SDK Version: ${Build.VERSION.SDK_INT}")
        logInfo("   - System Version: ${Build.VERSION.RELEASE}")
        logInfo("   - Available memory: ${Runtime.getRuntime().maxMemory() / (1024 * 1024)} MB")
    }

    private fun listAvailableAssets(path: String) {
        try {
            val files = context.assets.list(path)
            if (files.isNullOrEmpty()) {
                logInfo("   📁 Path '$path' is empty or doesn't exist")
            } else {
                logInfo("   📁 Files in '$path': ${files.joinToString(", ")}")
            }
        } catch (e: Exception) {
            logError("Cannot list files in '$path': ${e.message}")
        }
    }

    fun classifyAudio(audioData: FloatArray): Int {
        sessionLock.lock()
        try {
            val sess = session ?: throw IllegalStateException("Session not initialized. Call initialize() first.")
            val inputName = sess.inputNames.first()
            val shape = longArrayOf(1, audioData.size.toLong())

            val inputBuffer = FloatBuffer.wrap(audioData)
            val tensor = OnnxTensor.createTensor(env, inputBuffer, shape)

            return processModelOutput(sess, inputName, tensor)
        } catch (e: Exception) {
            logError("Error during classification", e)
            return 2  // Default to 'silence' class in case of error
        } finally {
            sessionLock.unlock()
        }
    }

    private fun processModelOutput(sess: OrtSession, inputName: String, tensor: OnnxTensor): Int {
        return tensor.use { input ->
            sess.run(mapOf(inputName to input)).use { output ->
                val outputTensor = output.get(0) as OnnxTensor
                val tensorInfo = outputTensor.info
                logInfo("Output tensor dimensions: ${tensorInfo.shape.contentToString()}")

                val result = processOutputTensor(outputTensor)
                logInfo("Classification result: $result")
                result
            }
        }
    }

    private fun processOutputTensor(outputTensor: OnnxTensor): Int {
        val value = outputTensor.value

        if (value !is Array<*>) {
            logError("Unknown tensor format: ${value?.javaClass}")
            return 2 // Default to silence
        }

        if (value.isEmpty() || value[0] !is Array<*>) {
            logError("Tensor has unexpected structure: $value")
            return 2 // Default to silence
        }

        // Process tensor with shape [1, time_steps, num_classes]
        return processTimeSteps(value[0] as Array<*>)
    }

    private fun processTimeSteps(timeSteps: Array<*>): Int {
        val predictions = mutableListOf<Int>()

        for (step in timeSteps) {
            when (step) {
                is FloatArray -> predictions.add(findMaxIndex(step))
                is DoubleArray -> predictions.add(findMaxIndex(step))
                else -> logError("Unknown time step type: ${step?.javaClass}")
            }
        }

        if (predictions.isEmpty()) return 2 // Default to silence

        // Find the most common class
        val counts = predictions.groupingBy { it }.eachCount()
        return counts.maxByOrNull { it.value }?.key ?: 2
    }

    private fun findMaxIndex(array: FloatArray): Int {
        var maxIndex = 0
        var maxValue = array[0]

        for (i in 1 until array.size) {
            if (array[i] > maxValue) {
                maxValue = array[i]
                maxIndex = i
            }
        }

        return maxIndex
    }

    private fun findMaxIndex(array: DoubleArray): Int {
        var maxIndex = 0
        var maxValue = array[0]

        for (i in 1 until array.size) {
            if (array[i] > maxValue) {
                maxValue = array[i]
                maxIndex = i
            }
        }

        return maxIndex
    }

    fun isInitialized(): Boolean {
        return session != null
    }

    fun close() {
        try {
            sessionLock.lock()
            session?.close()
            sessionLock.unlock()

            env.close()
            logInfo("Resources released")
        } catch (e: Exception) {
            logError("Error while releasing resources", e)
        }
    }
}