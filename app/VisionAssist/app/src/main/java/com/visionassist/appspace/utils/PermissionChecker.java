package com.visionassist.appspace.utils;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import androidx.core.content.ContextCompat;
import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.activities.newprofile.PermissionsActivity;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import com.visionassist.appspace.models.ttsengine.TTSManager;

public class PermissionChecker {
    private static final String TAG = "PermissionChecker";

    public static void checkAndRequestPermissions(Activity activity, Class<?> nextActivityClass, LoadingManager loadingManager, boolean blindProfile) {
        boolean cameraGranted = checkCameraPermission(activity);
        boolean storageGranted = checkStoragePermissions(activity);

        // Determine permission status option
        int permissionOption;
        if (!cameraGranted && !storageGranted) {
            permissionOption = 0; // All permissions not granted
        } else if (!cameraGranted) {
            permissionOption = 1; // Camera permission not granted
        } else if (!storageGranted) {
            permissionOption = 2; // File permissions not granted
        } else {
            // All permissions granted
            return;
        }

        if (blindProfile) {
            PhoneStatusMonitor monitor = PhoneStatusMonitor.getInstance();
            TTSManager ttsManager = monitor.getTTSManager();
            Handler speakHandler = new Handler(Looper.getMainLooper());
            Runnable ttsRetryRunnableReady = new Runnable() {
                @Override
                public void run() {
                    if (ttsManager.isReady()) {
                        ttsManager.speak(UtilsKt.load_permissionActivityWarning(activity), AppConfig.tts_pitch, AppConfig.tts_speech_rate, true,null);
                        Handler speakHandler2 = new Handler(Looper.getMainLooper());
                        Runnable ttsRetryRunnableSpeak = new Runnable() {
                            @Override
                            public void run() {
                                if(ttsManager.isDoneSpeaking()) {
                                    Intent intent = new Intent(activity, PermissionsActivity.class);
                                    intent.putExtra(Constants.EXTRA_PERMISSION_OPTION, permissionOption);
                                    intent.putExtra(Constants.EXTRA_NEXT_ACTIVITY, nextActivityClass.getName());
                                    if (loadingManager != null) {
                                        loadingManager.hideLoading();
                                        new Handler(Looper.getMainLooper()).postDelayed(() -> activity.startActivity(intent), Constants.ANIMATION_DELAY);
                                    } else
                                        activity.startActivity(intent);
                                }
                                else {
                                    Log.e(TAG, "TTS not done speaking on attempt. Retrying...");
                                    speakHandler2.postDelayed(this, Constants.RETRY_TTS_DELAY_MS);
                                }
                            }
                        };
                        speakHandler2.post(ttsRetryRunnableSpeak);
                    } else {
                        Log.e(TAG, "TTS not ready on attempt. Retrying...");
                        speakHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS);
                    }
                }
            };
            speakHandler.post(ttsRetryRunnableReady);
        } else {
            Intent intent = new Intent(activity, PermissionsActivity.class);
            intent.putExtra(Constants.EXTRA_PERMISSION_OPTION, permissionOption);
            intent.putExtra(Constants.EXTRA_NEXT_ACTIVITY, nextActivityClass.getName());
            if (loadingManager != null) {
                loadingManager.hideLoading();
                new Handler(Looper.getMainLooper()).postDelayed(() -> activity.startActivity(intent), Constants.ANIMATION_DELAY);
            } else
                activity.startActivity(intent);
        }
    }

    /**
     * Check camera permission - works across all API levels
     */
    private static boolean checkCameraPermission(Activity activity) {
        return ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Check storage permissions with backward compatibility
     * API 33+ (Android 13+): READ_MEDIA_IMAGES
     * API 24-32: READ_EXTERNAL_STORAGE and WRITE_EXTERNAL_STORAGE
     */
    private static boolean checkStoragePermissions(Activity activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) { // API 33+
            // Android 13+ uses READ_MEDIA_IMAGES
            return ContextCompat.checkSelfPermission(activity, Manifest.permission.READ_MEDIA_IMAGES)
                    == PackageManager.PERMISSION_GRANTED;
        } else {
            // Android 12 and below (API 24-32)
            boolean readGranted = ContextCompat.checkSelfPermission(activity, Manifest.permission.READ_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_GRANTED;
            boolean writeGranted = ContextCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_GRANTED;
            return readGranted && writeGranted;
        }
    }

    /**
     * Get the appropriate storage permissions array for the current API level
     */
    public static String[] getStoragePermissionsArray() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) { // API 33+
            return new String[]{Manifest.permission.READ_MEDIA_IMAGES};
        } else {
            return new String[]{
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            };
        }
    }
}