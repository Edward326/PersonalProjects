package com.visionassist.appspace.activities.tabs.reports

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.FileUtils
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

object EnvironmentReportsManagerKt {
    private const val TAG = "EnvReportsManager"

    @SuppressLint("DefaultLocale")
    fun writeDetectionReport(
        context: Context,
        sceneClassId: Int,
        detectedObjects: List<String>,
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
                    "[%s] SceneID: %d | Objects: [%s] | Threads used: %d | " +
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

    /**
     * Clear all environment reports
     */
    fun clearReports(context: Context) {
        try {
            FileUtils.createProfileDirFile(Constants.ENV_REPORTS_FILE_NAME)
            Log.d(TAG, "Reports cleared")
        } catch (e: Exception) {
            Log.e(TAG, "Error clearing reports", e)
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

    /**
     * Get report file path
     */
    fun getReportFilePath(context: Context): String {
        val reportFile = File(
            FileUtils.getProfileDirectory(context),
            Constants.ENV_REPORTS_FILE_NAME
        )
        return reportFile.absolutePath
    }

    /**
     * Check if reports exist
     */
    fun hasReports(context: Context): Boolean {
        return try {
            val reportFile = File(
                FileUtils.getProfileDirectory(context),
                Constants.ENV_REPORTS_FILE_NAME
            )
            reportFile.exists() && reportFile.length() > 0
        } catch (_: Exception) {
            false
        }
    }
}