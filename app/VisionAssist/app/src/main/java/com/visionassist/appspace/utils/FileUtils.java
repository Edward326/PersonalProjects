//file manipulation methods
package com.visionassist.appspace.utils;

import android.content.Context;
import android.util.Log;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;

public class FileUtils {
    private static final String TAG = "FileUtils";

    /**
     * Get the profile directory in internal storage
     */
    public static File getProfileDirectory(Context context) {
        File dir = new File(context.getFilesDir(), Constants.PROFILE_FOLDER_NAME);
        if (!dir.exists()) {
            Log.d(TAG, "Profile directory does not exist: " + dir.getAbsolutePath());
        }
        return dir;
    }

    /**
     * Get the profile file in internal storage
     */
    public static File getProfileFile(Context context) {
        File dir = getProfileDirectory(context);
        return new File(dir, Constants.PROFILE_FILE_NAME);
    }

    /**
     * Check if profile directory exists
     */
    public static boolean profileDirectoryExists(Context context) {
        File dir = getProfileDirectory(context);
        return dir.exists() && dir.isDirectory();
    }

    /**
     * Check if profile file exists in internal storage
     */
    public static boolean profileFileExists(Context context) {
        File profileFile = getProfileFile(context);
        boolean exists = profileFile.exists() && profileFile.isFile();
        if (exists) {
            Log.d(TAG, "Profile file exists: " + profileFile.getAbsolutePath() + " (size: " + profileFile.length() + " bytes)");
        }
        return exists;
    }

    /**
     * Get InputStream for profile file from internal storage
     */
    public static InputStream getProfileInputStream(Context context) throws IOException {
        File profileFile = getProfileFile(context);
        if (!profileFile.exists()) {
            throw new IOException("Profile file does not exist: " + profileFile.getAbsolutePath());
        }
        return new FileInputStream(profileFile);
    }

    /**
     * Create profile directory and file
     * @return true if created successfully, false otherwise
     */
    public static boolean createProfileStructure(Context context) {
        try {
            // Create directory
            File dir = getProfileDirectory(context);
            if (!dir.exists()) {
                if (!dir.mkdirs()) {
                    Log.e(TAG, "Failed to create profile directory: " + dir.getAbsolutePath());
                    return false;
                }
                Log.d(TAG, "Created profile directory: " + dir.getAbsolutePath());
            }

            // Create empty profile file
            File profileFile = getProfileFile(context);
            if (!profileFile.exists()) {
                if (!profileFile.createNewFile()) {
                    Log.e(TAG, "Failed to create profile file: " + profileFile.getAbsolutePath());
                    return false;
                }
                Log.d(TAG, "Created profile file: " + profileFile.getAbsolutePath());
            }

            return true;
        } catch (IOException e) {
            Log.e(TAG, "Error creating profile structure", e);
            return false;
        }
    }

    /**
     * Delete profile directory and all its contents
     * @return true if deleted successfully, false otherwise
     */
    public static boolean deleteProfileDirectory(Context context) {
        File dir = getProfileDirectory(context);
        if (!dir.exists()) {
            Log.d(TAG, "Profile directory does not exist, nothing to delete");
            return true;
        }

        try {
            return deleteRecursive(dir);
        } catch (Exception e) {
            Log.e(TAG, "Error deleting profile directory", e);
            return false;
        }
    }

    /**
     * Recursively delete a directory and its contents
     */
    private static boolean deleteRecursive(File fileOrDirectory) {
        if (fileOrDirectory.isDirectory()) {
            File[] children = fileOrDirectory.listFiles();
            if (children != null) {
                for (File child : children) {
                    if (!deleteRecursive(child)) {
                        return false;
                    }
                }
            }
        }
        boolean deleted = fileOrDirectory.delete();
        if (deleted) {
            Log.d(TAG, "Deleted: " + fileOrDirectory.getAbsolutePath());
        } else {
            Log.e(TAG, "Failed to delete: " + fileOrDirectory.getAbsolutePath());
        }
        return deleted;
    }

    /**
     * Write JSON string to profile file
     */
    public static boolean writeProfileFile(Context context, String jsonContent) {
        try {
            File profileFile = getProfileFile(context);
            try (FileOutputStream fos = new FileOutputStream(profileFile);
                 OutputStreamWriter osw = new OutputStreamWriter(fos)) {
                osw.write(jsonContent);
                osw.flush();
            }
            Log.d(TAG, "Profile file written successfully: " + profileFile.getAbsolutePath());
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Error writing profile file", e);
            return false;
        }
    }

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
     * Load class names from file
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
                    continue;
                }

                try {
                    if (line.contains(",")) {
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

        if (classNames.containsKey(0)) {
            Log.d(TAG, "Class 0: " + classNames.get(0));
        }
        if (classNames.containsKey(1)) {
            Log.d(TAG, "Class 1: " + classNames.get(1));
        }

        return classNames;
    }
}