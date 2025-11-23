//file manipulation methods
package com.visionassist.appspace.utils;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.util.Log;
import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager;
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

    @SuppressLint("StaticFieldLeak")
    static final PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
    @SuppressLint("StaticFieldLeak")
    static Activity activity=phoneMonitor.getCurrentActivity();
    @SuppressLint("StaticFieldLeak")
    static Context context=phoneMonitor.getCurrentContext();

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

    public static File getHashCacheFile(Context context) {
        File dir = getProfileDirectory(context);
        return new File(dir, Constants.HASH_CACHE_FILE_NAME);
    }

    public static File getEnvReportsFile(Context context) {
        File dir = getProfileDirectory(context);
        return new File(dir, Constants.ENV_REPORTS_FILE_NAME);
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
     * Get InputStream for hashcache file from internal storage
     */
    public static InputStream getHashCacheInputStream(Context context) throws IOException {
        File hcFile = getHashCacheFile(context);
        if (!hcFile.exists()) {
            throw new IOException("Profile file does not exist: " + hcFile.getAbsolutePath());
        }
        return new FileInputStream(hcFile);
    }

    /**
     * Get InputStream for env file from internal storage
     */
    public static InputStream getEnvReportsInputStream(Context context) throws IOException {
        File envFile = getEnvReportsFile(context);
        if (!envFile.exists()) {
            throw new IOException("Profile file does not exist: " + envFile.getAbsolutePath());
        }
        return new FileInputStream(envFile);
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
    public static boolean createProfileDirFile(String fileName) {
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
            File file = new File(dir,fileName);
            if (file.exists()) {
                deleteProfileDirFile(fileName);
            }
            if (!file.createNewFile()) {
                Log.e(TAG, "Failed to create profile file: " + file.getAbsolutePath());
                return false;
            }
            Log.d(TAG, "Created profile file: " + file.getAbsolutePath());
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Error creating profile structure", e);
            return false;
        }
    }
    public static boolean deleteProfileDirFile(String fileName) {
        try {
            // Create directory
            File dir = getProfileDirectory(context);
            if (!dir.exists()) {
                Log.d(TAG, "Profile directory does not exist");
                return false;
            }

            // Create empty profile file
            File file = new File(dir, fileName);
            if (!file.exists()) {
                Log.d(TAG, "Profile file: ~" + fileName + " does not exist");
                return false;
            }
            if (!file.delete()) {
                Log.e(TAG, "Failed to delete profile file: " + file.getAbsolutePath());
                return false;
            }
            return true;
        }
        catch (Exception e) {
            Log.e(TAG, "Error deleting profile file", e);
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
            ErrorDialogManager errorDialog = new ErrorDialogManager(phoneMonitor.getCurrentActivity());
            errorDialog.setupDialog(Constants.DIR_DELETE_ERROR);
            phoneMonitor.shutdownApp(errorDialog, phoneMonitor.getCurrentContext());
        }
        return false;
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
    public static boolean writeProfileFile(String jsonContent,String fileName) {
        try {
            File dir=getProfileDirectory(PhoneStatusMonitor.getInstance().getCurrentContext());
            if (!dir.exists())return false;

            File profileFile = new File(dir,fileName);
            if(!profileFile.exists())return false;

            try (FileOutputStream fos = new FileOutputStream(profileFile);
                 OutputStreamWriter osw = new OutputStreamWriter(fos)) {
                osw.write(jsonContent);
                osw.flush();
            }
            Log.d(TAG, "Profile file written successfully: " + profileFile.getAbsolutePath());
            return true;
        } catch (IOException e) {
                PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                errorDialog.setupDialog(Constants.FILE_WRITE_ERROR);
                phoneMonitor.shutdownApp(errorDialog,context);
        }
        return false;
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

                            if (classId < 5) {
                                Log.d(TAG, String.format("Loaded class: %d -> '%s'", classId, className));
                            }
                        }
                    } else {
                        classNames.put(lineNumber, line);

                        if (lineNumber < 5) {
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

    public static String readProfileFileAsString(Context context, String fileName) {
        try {
            File file = new File(getProfileDirectory(context), fileName);

            if (!file.exists()) {
                Log.w(TAG, "File does not exist: " + file.getAbsolutePath());
                return null;
            }

            StringBuilder sb = new StringBuilder();

            try (FileInputStream fis = new FileInputStream(file);
                 InputStreamReader isr = new InputStreamReader(fis);
                 BufferedReader reader = new BufferedReader(isr)) {

                String line;
                while ((line = reader.readLine()) != null) {
                    sb.append(line).append('\n');
                }
            }

            Log.d(TAG, "Read " + sb.length() + " characters from: " + fileName);
            return sb.toString();
        } catch (IOException e) {
            Log.e(TAG, "Error reading file: " + fileName, e);
            return null;
        }
    }
}