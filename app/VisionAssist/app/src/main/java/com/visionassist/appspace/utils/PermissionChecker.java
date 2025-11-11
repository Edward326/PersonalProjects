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
import com.visionassist.appspace.R;
import com.visionassist.appspace.activities.newprofile.PermissionsActivity;
import com.visionassist.appspace.models.ttsengine.TTSManager;

public class PermissionChecker {
    private static final String TAG = "PermissionChecker";

    public static void checkAndRequestPermissions(Activity activity, boolean blindProfile,Runnable onPermissionsGranted) {
        PhoneStatusMonitor phoneMonitor=PhoneStatusMonitor.getInstance();
        phoneMonitor.isReturningFromPermissions=false;
        boolean cameraGranted = checkCameraPermission(activity);
        boolean microphoneGranted = checkMicrophonePermission(activity);
        boolean storageGranted = checkStoragePermissions(activity);

        // Determine permission status option
        int permissionOption;
        if (!cameraGranted && !microphoneGranted && !storageGranted) {
            permissionOption = 0; // All permissions not granted
        } else if (!cameraGranted) {
            permissionOption = 1; // Camera permission not granted
        } else if (!microphoneGranted) {
            permissionOption = 2; // Microphone permission not granted
        } else if (!storageGranted) {
            permissionOption = 3; // File permissions not granted
        } else {
            // All permissions granted
            if (onPermissionsGranted != null) {
                onPermissionsGranted.run();
            }
            return;
        }

        PhoneStatusMonitor.getInstance().isReturningFromPermissions=true;
        if (blindProfile) {
            PhoneStatusMonitor monitor = PhoneStatusMonitor.getInstance();
            TTSManager ttsManager = monitor.getTTSManager();
            Handler speakHandler = new Handler(Looper.getMainLooper());
            Runnable ttsRetryRunnableReady = new Runnable() {
                @Override
                public void run() {
                    if (ttsManager.isReady()) {
                        String speech;
                        float pitch, rate;
                        if (PhoneStatusMonitor.getInstance().profileLoaded) {
                            speech = UtilsKt.load_permissionActivityWarning(activity);
                            pitch = AppConfig.tts_pitch;
                            rate = AppConfig.tts_speech_rate;
                        } else {
                            speech = activity.getString(R.string.audio_permission_warning_en);
                            pitch = Constants.TTS_PITCH;
                            rate = Constants.TTS_SPEECH_RATE;
                        }
                        ttsManager.speak(speech, pitch, rate, false, null);
                        Handler speakHandler2 = new Handler(Looper.getMainLooper());
                        Runnable ttsRetryRunnableSpeak = new Runnable() {
                            @Override
                            public void run() {
                                if (ttsManager.isDoneSpeaking()) {
                                    Intent intent = new Intent(activity, PermissionsActivity.class);
                                    intent.putExtra(Constants.EXTRA_PERMISSION_OPTION, permissionOption);
                                    activity.startActivity(intent);
                                } else {
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
            activity.startActivity(intent);
        }
    }

    private static boolean checkCameraPermission(Activity activity) {
        return ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    private static boolean checkMicrophonePermission(Activity activity) {
        return ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
    }

    private static boolean checkStoragePermissions(Activity activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // API 33+ (Android 13+): Granular media permissions
            // For reading images

            // Return true if at least images permission is granted
            return ContextCompat.checkSelfPermission(activity,
                    Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED;

        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // API 30-32 (Android 11-12): Scoped storage, but still uses READ_EXTERNAL_STORAGE
            return ContextCompat.checkSelfPermission(activity,
                    Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;

        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // API 29 (Android 10): Scoped storage introduced
            // READ_EXTERNAL_STORAGE still works, WRITE is limited
            return ContextCompat.checkSelfPermission(activity,
                    Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;

        } else {
            // API 24-28 (Android 7-9): Traditional storage permissions
            boolean readGranted = ContextCompat.checkSelfPermission(activity,
                    Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
            boolean writeGranted = ContextCompat.checkSelfPermission(activity,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;

            return readGranted && writeGranted;
        }
    }

    public static String[] getStoragePermissionsArray() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // API 33+ (Android 13+)
            return new String[]{
                    Manifest.permission.READ_MEDIA_IMAGES,
            };

        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // API 29-32 (Android 10-12)
            return new String[]{
                    Manifest.permission.READ_EXTERNAL_STORAGE
            };

        } else {
            // API 24-28 (Android 7-9)
            return new String[]{
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            };
        }
    }
}