package com.visionassist.appspace.activities.tabs.reports;

import android.annotation.SuppressLint;
import android.content.Context;
import android.util.Log;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.FileUtils;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class EnvironmentReportsManager {
    private static final String TAG = "EnvironmentReportsManager";

    /**
     * Write detection results to environment report file
     * Format: [timestamp] Scene: <scene_name> | Objects: <obj1_id>, <obj2_id>, ...
     *
     * @param context       Application context
     * @param sceneName     Detected scene classification (e.g., "kitchen", "bedroom")
     * @param detectedObjects List of synonym strings for detected objects
     */
    public static void writeDetectionReport(Context context, int sceneName, List<String> detectedObjects) {
        // Only write if environment reports are enabled
        if (!AppConfig.env_reports) {
            Log.d(TAG, "Environment reports disabled, skipping write");
            return;
        }

        try {
            // Get report file
            File reportFile = new File(FileUtils.getProfileDirectory(context), Constants.ENV_REPORTS_FILE_NAME);

            // Append to file
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(reportFile, true))) {
                // Format timestamp
                String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
                        .format(new Date());

                // Format detected objects as comma-separated list
                String objectsList = detectedObjects.isEmpty()
                        ? "none"
                        : String.join(", ", detectedObjects);

                // Write log entry
                @SuppressLint("DefaultLocale") String logEntry = String.format("[%s] %d | Objects: %s%n",
                        timestamp,
                        sceneName,
                        objectsList);

                writer.write(logEntry);
                writer.flush();

                Log.d(TAG, "Detection report written: " + logEntry.trim());
            } catch (IOException e) {
                Log.e(TAG, "Error writing to report file", e);
            }

        } catch (Exception e) {
            Log.e(TAG, "Error in writeDetectionReport", e);
        }
    }

    /**
     * Clear all environment reports
     */
    public static void clearReports(Context context) {
        try {
            FileUtils.createProfileDirFile(Constants.ENV_REPORTS_FILE_NAME);
        } catch (Exception e) {
            Log.e(TAG, "Error clearing reports", e);
        }
    }
}