//permission checker class to check if the app has the required permissions
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

    public static void checkAndRequestPermissions(Activity activity, Class<?> nextActivityClass, LoadingManager loadingManager, boolean blindProfile) {
        boolean cameraGranted = false;
        boolean storageGranted = false;

        // Check camera permission
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            cameraGranted = true;
        }

        // Check storage permissions based on Android version
        if (Constants.API_LEVEL >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ uses READ_MEDIA_IMAGES instead of READ_EXTERNAL_STORAGE
            if (ContextCompat.checkSelfPermission(activity, Manifest.permission.READ_MEDIA_IMAGES)
                    == PackageManager.PERMISSION_GRANTED) {
                storageGranted = true;
            }
        } else {
            // Android 12 and below
            if (ContextCompat.checkSelfPermission(activity, Manifest.permission.READ_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_GRANTED &&
                    ContextCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                            == PackageManager.PERMISSION_GRANTED) {
                storageGranted = true;
            }
        }

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

        if(blindProfile) {
            PhoneStatusMonitor monitor = PhoneStatusMonitor.getInstance();
            TTSManager ttsManager = monitor.getTTSManager();
            Handler speakHandler = new Handler(Looper.getMainLooper());
            Runnable ttsRetryRunnable = new Runnable() {
                @Override
                public void run() {
                    if (ttsManager.isReady()) {
                        // SUCCESS: TTS is ready. Get the localized message based on the active language.
                        ttsManager.speak(UtilsKt.load_PermissionActivityWarning(activity),AppConfig.TTS_PITCH,AppConfig.TTS_SPEECH_RATE);
                        Handler speakHandler2 = new Handler(Looper.getMainLooper());
                        speakHandler2.postDelayed(() -> {
                        Intent intent = new Intent(activity, PermissionsActivity.class);
                        intent.putExtra(Constants.EXTRA_PERMISSION_OPTION, permissionOption);
                        intent.putExtra(Constants.EXTRA_NEXT_ACTIVITY, nextActivityClass.getName());
                        if (loadingManager != null) loadingManager.hideLoading();
                        activity.startActivity(intent);
                        },Constants.BLINDNESS_SHUTDOWN_DELAY_MS);
                    } else{
                        Log.e("MainActivity", "TTS not ready on attempt. Retrying...");
                        speakHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS); // Retry after delay
                    }
                }
            };
            speakHandler.post(ttsRetryRunnable);
        }
        else {
            // Navigate to PermissionsActivity with the option and next activity
            Intent intent = new Intent(activity, PermissionsActivity.class);
            intent.putExtra(Constants.EXTRA_PERMISSION_OPTION, permissionOption);
            intent.putExtra(Constants.EXTRA_NEXT_ACTIVITY, nextActivityClass.getName());
            if (loadingManager != null) loadingManager.hideLoading();
            activity.startActivity(intent);
        }
    }
}