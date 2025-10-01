//file manipulation methods(verify for existing files, copy asset files to internal storage, read asset files as string, load class names from file)
package com.visionassist.appspace.utils;

import android.content.Context;
import android.util.Log;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

public class FileUtils {
    private static final String TAG = "FileUtils";

    /**
     * Get the absolute file path for an asset file (copy to internal storage if needed)
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);

        if (file.exists() && file.length() > 0) {
            Log.d(TAG, "Asset file already exists: " + file.getAbsolutePath());
            return file.getAbsolutePath();
        }

        Log.d(TAG, "Copying asset to internal storage: " + assetName);

        try (InputStream inputStream = context.getAssets().open(assetName);
             FileOutputStream outputStream = new FileOutputStream(file)) {

            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            outputStream.flush();
        }

        Log.d(TAG, "Asset copied successfully: " + file.getAbsolutePath() + " (size: " + file.length() + " bytes)");
        return file.getAbsolutePath();
    }

    /**
     * Check if an asset file exists
     */
    public static boolean assetExists(Context context, String assetName) {
        try {
            InputStream inputStream = context.getAssets().open(assetName);
            inputStream.close();
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Load an asset file as a string
     */
    public static String loadAssetAsString(Context context, String assetName) throws IOException {
        StringBuilder sb = new StringBuilder();

        try (InputStream inputStream = context.getAssets().open(assetName);
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
        }
        return sb.toString();
    }

    /**
     * Load an InputStream as a string
     */
    public static String loadFileAsString(InputStream inputStream) throws IOException {
        StringBuilder sb = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
        }

        return sb.toString();
    }

    /**
     * Load class names from file - FIXED to handle both formats
     */
    public static Map<Integer, String> loadClassNames(Context context, String fileName) throws IOException {
        Map<Integer, String> classNames = new HashMap<>();

        Log.d(TAG, "Loading class names from: " + fileName);

        try (InputStream inputStream = context.getAssets().open(fileName);
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {

            String line;
            int lineNumber = 0;

            while ((line = reader.readLine()) != null) {
                line = line.trim();

                if (line.isEmpty() || line.startsWith("#")) {
                    continue; // Skip empty lines and comments
                }

                try {
                    if (line.contains(",")) {
                        // Format: "0,person" or "0, person"
                        String[] parts = line.split(",", 2);
                        if (parts.length == 2) {
                            int classId = Integer.parseInt(parts[0].trim());
                            String className = parts[1].trim();
                            classNames.put(classId, className);

                            if (Constants.DEBUG_MODE && classId < 5) {
                                Log.d(TAG, String.format("Loaded class: %d -> '%s'", classId, className));
                            }
                        }
                    } else {
                        // Format: just class names, one per line (use line number as index)
                        classNames.put(lineNumber, line);

                        if (Constants.DEBUG_MODE && lineNumber < 5) {
                            Log.d(TAG, String.format("Loaded class: %d -> '%s'", lineNumber, line));
                        }
                    }

                } catch (NumberFormatException e) {
                    Log.w(TAG, "Invalid class format at line: " + line);
                }

                lineNumber++;
            }
        }

        Log.d(TAG, String.format("Loaded %d class names from %s", classNames.size(), fileName));

        // Verify some key classes are loaded
        if (classNames.containsKey(0)) {
            Log.d(TAG, "Class 0: " + classNames.get(0));
        }
        if (classNames.containsKey(1)) {
            Log.d(TAG, "Class 1: " + classNames.get(1));
        }

        return classNames;
    }
}