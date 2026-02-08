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
    private var sessionExhaleOnly: OrtSession? = null
    private var currentModelType = "standard"
    private val TARGET_LENGTH: Int = 154_350 // ~3.5s at 44.1kHz
    private val LAST_WINDOW_FRACTION: Float = 0.2f / 3.5f // fraction of frames (~last 0.2s)

    private val sessionLock = ReentrantLock()

    companion object {
        const val MODEL_NAME = "best_model_epoch_31.onnx"
        const val MODEL_NAME_EXHALE = "best_model_epoch_21.onnx"

        const val MODEL_DATA_NAME = "best_model_epoch_31.onnx.data"
        const val MODEL_DATA_NAME_EXHALE = "best_model_epoch_21.onnx.data"

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
        return initializeModel("standard")
    }

    fun initializeModel(modelType: String): Boolean {
        return try {
            logInfo("➡️ Initializing model: $modelType")
            currentModelType = modelType

            when (modelType) {
                "exhale_only" -> {
                    sessionExhaleOnly = createSessionFromBytesForModel(MODEL_NAME_EXHALE, MODEL_DATA_NAME_EXHALE)
                    sessionExhaleOnly != null
                }
                else -> {
                    session = createSessionFromBytesForModel(MODEL_NAME, MODEL_DATA_NAME)
                    session != null
                }
            }
        } catch (e: Exception) {
            logError("Error initializing model $modelType", e)
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

    private fun createSessionFromBytesForModel(modelName: String, dataName: String): OrtSession? {
        return try {
            val modelFile = File(context.cacheDir, modelName)
            val dataFile = File(context.cacheDir, dataName)

            if (!modelFile.exists()) {
                copyAssetToFile(modelName, modelFile)
            }

            if (!dataFile.exists()) {
                copyAssetToFile(dataName, dataFile)
            }

            val opts = OrtSession.SessionOptions()
            opts.setIntraOpNumThreads(2)

            sessionLock.lock()
            try {
                logInfo("🔄 Creating ONNX Runtime session from: ${modelFile.absolutePath}")

                val sess = env.createSession(modelFile.absolutePath, opts)
                logInfo("✅ ONNX model loaded successfully")

                sess.let { s ->
                    logInfo("📊 Model information:")
                    logInfo("📥 Model inputs: ${s.inputNames.joinToString(", ")}")
                    logInfo("📤 Model outputs: ${s.outputNames.joinToString(", ")}")
                }
                sess
            } finally {
                sessionLock.unlock()
            }
        } catch (e: Exception) {
            logError("Failed to create session from model bytes", e)
            null
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

    private fun copyAssetToFile(assetName: String, outputFile: File) {
        val possiblePaths = arrayOf(
            "flutter_assets/assets/models/$assetName",
            "assets/models/$assetName",
            "models/$assetName"
        )

        for (assetPath in possiblePaths) {
            try {
                logInfo("🔄 Copying asset from: $assetPath to ${outputFile.absolutePath}")
                context.assets.open(assetPath).use { input ->
                    outputFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                logInfo("✅ Successfully copied $assetName (${outputFile.length()} bytes)")
                return
            } catch (e: Exception) {
                logError("Cannot copy from $assetPath: ${e.message}")
            }
        }

        logError("Failed to copy $assetName from any path")
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
                logError("Cannot copy model data from path: $assetPath - ${e.message}")
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
            val sess = when (currentModelType) {
                "exhale_only" -> sessionExhaleOnly
                else -> session
            } ?: throw IllegalStateException("Session not initialized")
            val inputName = sess.inputNames.first()
            val processedAudio = padOrTrimAudio(audioData, TARGET_LENGTH)
            val shape = longArrayOf(processedAudio.size.toLong())

            val inputBuffer = FloatBuffer.wrap(processedAudio)
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

        // Expected tensor shape: [1, time_steps, num_classes]
        val timeSteps = value[0] as Array<*>
        val totalFrames = timeSteps.size
        if (totalFrames == 0) return 2

        val lastFrames = (totalFrames * LAST_WINDOW_FRACTION).toInt().coerceAtLeast(1)
        val startIdx = (totalFrames - lastFrames).coerceAtLeast(0)

        var numClasses = -1
        var count = 0
        var classSums: FloatArray? = null

        for (i in startIdx until totalFrames) {
            val step = timeSteps[i]
            val logits: FloatArray = when (step) {
                is FloatArray -> step
                is DoubleArray -> step.map { it.toFloat() }.toFloatArray()
                is Array<*> -> {
                    // In some cases step might be an Array<Float> or similar
                    try {
                        (step as Array<Number>).map { it.toFloat() }.toFloatArray()
                    } catch (e: Exception) {
                        logError("Unknown time step type: ${step?.javaClass}")
                        continue
                    }
                }
                else -> {
                    logError("Unknown time step type: ${step?.javaClass}")
                    continue
                }
            }

            if (numClasses < 0) {
                numClasses = logits.size
                classSums = FloatArray(numClasses) { 0f }
            } else if (logits.size != numClasses) {
                logError("Inconsistent class dimension across time steps")
                continue
            }

            val probs = softmax(logits)
            for (c in 0 until numClasses) {
                classSums!![c] += probs[c]
            }
            count++
        }

        if (count == 0 || classSums == null || numClasses <= 0) return 2

        // Mean over the last window
        for (c in 0 until numClasses) {
            classSums[c] /= count.toFloat()
        }

        val pred = findMaxIndex(classSums)
        logInfo("Classification (softmax-mean last ${lastFrames}f): $pred")
        return pred
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

    private fun softmax(logits: FloatArray): FloatArray {
        var max = Float.NEGATIVE_INFINITY
        for (x in logits) if (x > max) max = x
        val exps = FloatArray(logits.size)
        var sum = 0.0
        for (i in logits.indices) {
            val v = kotlin.math.exp((logits[i] - max).toDouble())
            exps[i] = v.toFloat()
            sum += v
        }
        if (sum == 0.0) {
            val uniform = 1f / logits.size
            return FloatArray(logits.size) { uniform }
        }
        for (i in exps.indices) {
            exps[i] = (exps[i] / sum).toFloat()
        }
        return exps
    }

    private fun padOrTrimAudio(audio: FloatArray, targetLength: Int): FloatArray {
        return when {
            audio.size == targetLength -> audio
            audio.size < targetLength -> {
                val out = FloatArray(targetLength)
                System.arraycopy(audio, 0, out, 0, audio.size)
                out // zero-padded tail
            }
            else -> {
                // take last targetLength samples
                val out = FloatArray(targetLength)
                val start = audio.size - targetLength
                System.arraycopy(audio, start, out, 0, targetLength)
                out
            }
        }
    }

    fun isInitialized(): Boolean {
        return when (currentModelType) {
            "exhale_only" -> sessionExhaleOnly != null
            else -> session != null
        }
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
