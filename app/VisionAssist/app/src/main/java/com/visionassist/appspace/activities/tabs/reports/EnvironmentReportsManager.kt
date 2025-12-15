package com.visionassist.appspace.activities.tabs.reports

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.FileUtils
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.locks.ReentrantLock

object EnvironmentReportsManagerKt {
    private const val TAG = "EnvReportsManager"

    // Fine-grained locks with tryLock support
    private val scenesLock = ReentrantLock()
    private val objectsLock = ReentrantLock()
    private val objectsBySceneLock = ReentrantLock()
    private val avgNoThreadsLock = ReentrantLock()
    private val avgDetectorLatencyLock = ReentrantLock()
    private val avgClassifierLatencyLock = ReentrantLock()
    private val avgBatteryUsageLock = ReentrantLock()

    @SuppressLint("DefaultLocale")
    fun writeDetectionReport(
        context: Context,
        sceneClassId: Int,
        detectedObjects: List<Int>,
        threadCount: Int,
        avgDetectorLatency: Long,
        avgClassifierLatency: Long,
        batteryUsageIncrease: Float
    ) {
        // Only write if environment reports are enabled
        if (!AppConfig.env_reports) {
            Log.d(TAG, "Environment reports disabled, skipping write")
            return
        }

        try {
            // Get report file
            val reportFile = File(
                FileUtils.getProfileDirectory(context),
                Constants.ENV_REPORTS_FILE_NAME
            )

            // Append to file
            BufferedWriter(FileWriter(reportFile, true)).use { writer ->
                // Format timestamp
                val timestamp = SimpleDateFormat(
                    "yyyy-MM-dd HH:mm:ss",
                    Locale.getDefault()
                ).format(Date())

                // Format detected objects as comma-separated list
                val objectsList = if (detectedObjects.isEmpty()) {
                    "none"
                } else {
                    detectedObjects.joinToString(", ")
                }

                // Format battery usage increase as percentage
                val batteryPercent = String.format("%.5f%%", batteryUsageIncrease * 100)

                // Write comprehensive log entry
                val logEntry = String.format(
                    "[%s] SceneID: %d | ObjectsID: [%s] | Threads used: %d | " +
                            "DetectorAvg: %dms | ClassifierAvg: %dms | BatteryUsageIncrease: %s%n",
                    timestamp,
                    sceneClassId,
                    objectsList,
                    threadCount,
                    avgDetectorLatency,
                    avgClassifierLatency,
                    batteryPercent
                )

                writer.write(logEntry)
                writer.flush()

                Log.d(TAG, "Detection report written: ${logEntry.trim()}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in writeDetectionReport", e)
        }
    }

    @SuppressLint("DefaultLocale")
    fun writeCaptionReport(
        context: Context,
        sceneClassId: Int,
        classifierLatency: Long
    ) {
        if (!AppConfig.env_reports) {
            Log.d(TAG, "Environment reports disabled, skipping write")
            return
        }

        try {
            val reportFile = File(
                FileUtils.getProfileDirectory(context),
                Constants.ENV_REPORTS_FILE_NAME
            )

            BufferedWriter(FileWriter(reportFile, true)).use { writer ->
                val timestamp = SimpleDateFormat(
                    "yyyy-MM-dd HH:mm:ss",
                    Locale.getDefault()
                ).format(Date())

                val logEntry = String.format(
                    "[%s] CAPTION_GENERATED | SceneID: %d | ClassifierLatency: %dms%n",
                    timestamp,
                    sceneClassId,
                    classifierLatency
                )

                writer.write(logEntry)
                writer.flush()

                Log.d(TAG, "Caption report written: ${logEntry.trim()}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in writeCaptionReport", e)
        }
    }

    fun processReportsAsync(
        context: Context,
        onSuccess: (EnvironmentStatistics) -> Unit,
        onError: (Exception) -> Unit
    ) {
        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                try {
                    // Read all lines
                    val lines = readReports(context)

                    if (lines.isEmpty()) {
                        Log.d(TAG, "No reports to process")
                        return@executeAsync EnvironmentStatistics()
                    }

                    Log.d(TAG, "Processing ${lines.size} report lines")

                    // Calculate thread count
                    val threadCount = if (lines.size <= 10) 1 else 5

                    // Create shared statistics database
                    val stats = EnvironmentStatistics()

                    if (threadCount == 1) {
                        // Single-threaded processing
                        processLines(lines, stats)
                    } else {
                        // Multi-threaded processing
                        val linesPerThread = lines.size / threadCount
                        val threads = mutableListOf<Thread>()

                        for (i in 0 until threadCount) {
                            val startIdx = i * linesPerThread
                            val endIdx =
                                if (i == threadCount - 1) lines.size else (i + 1) * linesPerThread
                            val threadLines = lines.subList(startIdx, endIdx)

                            val thread = Thread {
                                processLines(threadLines, stats)
                            }
                            threads.add(thread)
                            thread.start()

                            Log.d(TAG, "Thread $i: processing lines $startIdx-${endIdx - 1}")
                        }

                        // Wait for all threads to complete
                        threads.forEach { it.join() }
                    }

                    Log.d(TAG, "All threads completed, sorting results")

                    // Sort all lists by occurrence count
                    stats.sortAllByOccurrence()

                    Log.d(
                        TAG,
                        "Statistics: ${stats.scenesList.size} scenes, ${stats.objectsList.size} objects"
                    )

                    stats
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing reports", e)
                    throw e
                }
            },
            object : BackgroundTaskExecutor.TaskCallback<EnvironmentStatistics> {
                override fun onSuccess(reports: EnvironmentStatistics) {
                    Log.d(TAG, "Reports processed successfully")
                    onSuccess(reports)
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error classifying scene", e)
                    onError(e)
                }
            }
        )
    }

    private fun processLines(
        lines: List<String>,
        stats: EnvironmentStatistics
    ) {
        // Queue for deferred operations
        val deferredOperations = ConcurrentLinkedQueue<() -> Unit>()

        for (line in lines) {
            try {
                if (line.contains("CAPTION_GENERATED")) {
                    processCaptionLine(line, stats, deferredOperations)
                } else if (line.contains("SceneID:")) {
                    processDetectionLine(line, stats, deferredOperations)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing line: $line", e)
            }
        }

        // Process queued operations with skip-and-retry
        processQueuedOperations(deferredOperations)
    }

    /**
     * Process queued operations with non-blocking skip-and-retry
     * - Try each operation
     * - If locked (throws exception), skip and re-queue
     * - Continue until queue is empty
     */
    private fun processQueuedOperations(queue: ConcurrentLinkedQueue<() -> Unit>) {
        while (queue.isNotEmpty()) {
            val operation = queue.poll()

            try {
                operation!!()
            } catch (_: IllegalStateException) {
                // Lock was busy, re-queue for next pass
                queue.add(operation)
            }
        }
    }

    private fun processCaptionLine(
        line: String,
        stats: EnvironmentStatistics,
        deferredQueue: ConcurrentLinkedQueue<() -> Unit>
    ) {
        try {
            // Extract scene ID
            val sceneIdMatch = Regex("SceneID:\\s*(\\d+)").find(line)
            val sceneId = sceneIdMatch?.groupValues?.get(1)?.toIntOrNull() ?: return

            // Extract classifier latency
            val latencyMatch = Regex("ClassifierLatency:\\s*(\\d+)ms").find(line)
            val latency = latencyMatch?.groupValues?.get(1)?.toLongOrNull() ?: return

            // Convert scene ID to name
            val sceneName =
                PhoneStatusMonitor.getInstance().modelManager.classifier.getClassName(sceneId)

            // Try to execute immediately or queue
            tryOrQueue(deferredQueue) { addToSceneList(stats, sceneName) }
            tryOrQueue(deferredQueue) { updateClassifierLatency(stats, latency) }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing caption line", e)
        }
    }

    // Helper to try immediate execution or queue if locked
    private inline fun tryOrQueue(
        queue: ConcurrentLinkedQueue<() -> Unit>,
        noinline operation: () -> Unit
    ) {
        try {
            operation()
        } catch (_: Exception) {
            // If failed due to lock contention, queue it
            queue.add(operation)
        }
    }

    private fun processDetectionLine(
        line: String,
        stats: EnvironmentStatistics,
        deferredQueue: ConcurrentLinkedQueue<() -> Unit>
    ) {
        try {
            // Extract scene ID
            val sceneIdMatch = Regex("SceneID:\\s*(\\d+)").find(line)
            val sceneId = sceneIdMatch?.groupValues?.get(1)?.toIntOrNull() ?: return
            val sceneName =
                PhoneStatusMonitor.getInstance().modelManager.classifier.getClassName(sceneId)

            // Extract objects IDs (FIXED REGEX)
            val objectsMatch = Regex("""ObjectsID:\s*\[([^]]*)]""").find(line)
            val objectsStr = objectsMatch?.groupValues?.get(1) ?: ""
            val objectIds = if (objectsStr == "none" || objectsStr.isBlank()) {
                emptyList()
            } else {
                objectsStr.split(",").mapNotNull { it.trim().toIntOrNull() }
            }

            // Extract threads used
            val threadsMatch = Regex("Threads used:\\s*(\\d+)").find(line)
            val threads = threadsMatch?.groupValues?.get(1)?.toIntOrNull() ?: return

            // Extract detector latency
            val detectorMatch = Regex("DetectorAvg:\\s*(\\d+)ms").find(line)
            val detectorLatency = detectorMatch?.groupValues?.get(1)?.toLongOrNull() ?: return

            // Extract classifier latency
            val classifierMatch = Regex("ClassifierAvg:\\s*(\\d+)ms").find(line)
            val classifierLatency = classifierMatch?.groupValues?.get(1)?.toLongOrNull() ?: return

            // Extract battery usage
            val batteryMatch = Regex("BatteryUsageIncrease:\\s*([\\d.]+)%").find(line)
            val batteryUsage = batteryMatch?.groupValues?.get(1)?.toFloatOrNull() ?: return

            // Try to execute immediately or queue
            tryOrQueue(deferredQueue) { addToSceneList(stats, sceneName) }

            for (objectId in objectIds) {
                val detectorModel = PhoneStatusMonitor.getInstance().modelManager.detector

                val objectName = detectorModel.getClassName(objectId)
                tryOrQueue(deferredQueue) { addToObjectList(stats, objectName) }
                tryOrQueue(deferredQueue) { addToObjectsBySceneList(stats, sceneName, objectName) }
            }

            tryOrQueue(deferredQueue) { updateThreadCount(stats, threads) }
            tryOrQueue(deferredQueue) { updateDetectorLatency(stats, detectorLatency) }
            tryOrQueue(deferredQueue) { updateClassifierLatency(stats, classifierLatency) }
            tryOrQueue(deferredQueue) { updateBatteryUsage(stats, batteryUsage) }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing detection line", e)
        }
    }

    private fun addToSceneList(stats: EnvironmentStatistics, sceneName: String) {
        if (!scenesLock.tryLock()) {
            throw IllegalStateException("Lock busy")  // Signal to queue
        }
        try {
            val existing = stats.scenesList.find { it.first == sceneName }
            if (existing != null) {
                val idx = stats.scenesList.indexOf(existing)
                stats.scenesList[idx] = existing.copy(second = existing.second + 1)
            } else {
                stats.scenesList.add(Pair(sceneName, 1))
            }
        } finally {
            scenesLock.unlock()
        }
    }

    private fun addToObjectList(stats: EnvironmentStatistics, objectName: String) {
        if (!objectsLock.tryLock()) {
            throw IllegalStateException("Lock busy")
        }
        try {
            val existing = stats.objectsList.find { it.first == objectName }
            if (existing != null) {
                val idx = stats.objectsList.indexOf(existing)
                stats.objectsList[idx] = existing.copy(second = existing.second + 1)
            } else {
                stats.objectsList.add(Pair(objectName, 1))
            }
        } finally {
            objectsLock.unlock()
        }
    }

    private fun addToObjectsBySceneList(
        stats: EnvironmentStatistics,
        sceneName: String,
        objectName: String
    ) {
        if (!objectsBySceneLock.tryLock()) {
            throw IllegalStateException("Lock busy")
        }
        try {
            val sceneEntry = stats.objectsBySceneList.find { it.first == sceneName }

            if (sceneEntry != null) {
                val objectList = sceneEntry.second
                val existingObject = objectList.find { it.first == objectName }

                if (existingObject != null) {
                    val idx = objectList.indexOf(existingObject)
                    objectList[idx] = existingObject.copy(second = existingObject.second + 1)
                } else {
                    objectList.add(Pair(objectName, 1))
                }
            } else {
                val objectList = mutableListOf(Pair(objectName, 1))
                stats.objectsBySceneList.add(Pair(sceneName, objectList))
            }
        } finally {
            objectsBySceneLock.unlock()
        }
    }

    private fun updateThreadCount(stats: EnvironmentStatistics, threads: Int) {
        if (!avgNoThreadsLock.tryLock()) {
            throw IllegalStateException("Lock busy")
        }
        try {
            stats.detectionRecordCount++
            val n = stats.detectionRecordCount
            stats.avgNoThreads = ((stats.avgNoThreads * (n - 1)) + threads) / n
        } finally {
            avgNoThreadsLock.unlock()
        }
    }

    private fun updateDetectorLatency(stats: EnvironmentStatistics, latency: Long) {
        if (!avgDetectorLatencyLock.tryLock()) {
            throw IllegalStateException("Lock busy")
        }
        try {
            val n = stats.detectionRecordCount
            if (n > 0) {
                stats.avgDetectorLatency = ((stats.avgDetectorLatency * (n - 1)) + latency) / n
            }
        } finally {
            avgDetectorLatencyLock.unlock()
        }
    }

    private fun updateClassifierLatency(stats: EnvironmentStatistics, latency: Long) {
        if (!avgClassifierLatencyLock.tryLock()) {
            throw IllegalStateException("Lock busy")
        }
        try {
            val totalRecords = stats.detectionRecordCount + stats.captionRecordCount + 1
            stats.captionRecordCount++
            stats.avgClassifierLatency =
                ((stats.avgClassifierLatency * (totalRecords - 1)) + latency) / totalRecords
        } finally {
            avgClassifierLatencyLock.unlock()
        }
    }

    private fun updateBatteryUsage(stats: EnvironmentStatistics, usage: Float) {
        if (!avgBatteryUsageLock.tryLock()) {
            throw IllegalStateException("Lock busy")
        }
        try {
            val n = stats.detectionRecordCount
            if (n > 0) {
                stats.avgPercMoreBatteryUsed =
                    ((stats.avgPercMoreBatteryUsed * (n - 1)) + usage) / n
            }
        } finally {
            avgBatteryUsageLock.unlock()
        }
    }

    /**
     * Read all environment reports
     */
    fun readReports(context: Context): List<String> {
        return try {
            val reportFile = File(
                FileUtils.getProfileDirectory(context),
                Constants.ENV_REPORTS_FILE_NAME
            )

            if (reportFile.exists()) {
                reportFile.readLines()
            } else {
                emptyList()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error reading reports", e)
            emptyList()
        }
    }
}