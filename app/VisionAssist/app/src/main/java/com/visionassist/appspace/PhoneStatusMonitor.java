package com.visionassist.appspace;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Application;
import android.content.Context;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Pair;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.visionassist.appspace.jetpack.managers.ErrorDialogManager;
import com.visionassist.appspace.models.ttsengine.TTSManager;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.Utils;
import com.visionassist.appspace.utils.UtilsKt;

public class PhoneStatusMonitor implements Application.ActivityLifecycleCallbacks {
    private static final String TAG = "PhoneStatusMonitor";

    private Context appContext;
    private Handler handler;
    private Runnable monitoringRunnable;
    private boolean isMonitoring = false;
    private int activeActivityCount = 0;
    private Activity currentActivity;
    private boolean errorShown = false;
    private boolean isPaused = false;
    private TTSManager ttsManager;
    @SuppressLint("StaticFieldLeak")
    private static PhoneStatusMonitor instance;

    private PhoneStatusMonitor(Context context) {
        this.appContext = context.getApplicationContext();
        this.handler = new Handler(Looper.getMainLooper());
        this.ttsManager = new TTSManager(this.appContext);
        setupMonitoringRunnable();
    }

    public static void initialize(Application application) {
        if (instance == null) {
            instance = new PhoneStatusMonitor(application);
            application.registerActivityLifecycleCallbacks(instance);
            Log.d(TAG, "PhoneStatusMonitor initialized");
        }
    }

    public static PhoneStatusMonitor getInstance() {
        return instance;
    }

    private void setupMonitoringRunnable() {
        monitoringRunnable = new Runnable() {
            @Override
            public void run() {
                // Check if monitoring is active AND not paused
                if (isMonitoring && !errorShown && !isPaused) {
                    Log.d(TAG, "Monitoring phone status...");
                    checkPhoneStatus();
                    handler.postDelayed(this, Constants.WAIT_CHECK);
                } else if (isMonitoring && isPaused) {
                    // Still schedule next check, just skip this one
                    handler.postDelayed(this, Constants.WAIT_CHECK);
                }
            }
        };
    }

    private void checkPhoneStatus() {
        try {
            // Util.checkPhoneStatus returns Pair<BatteryStatus, TemperatureStatus>
            Pair<Integer, Integer> status = Utils.checkPhoneStatus(appContext);
            int batteryStatus = status.first;
            int temperatureStatus = status.second;

            Log.d(TAG, "Battery: " + batteryStatus + ", Temperature: " + temperatureStatus);
            batteryStatus(batteryStatus, temperatureStatus);

        } catch (Exception e) {
            Log.e(TAG, "Error checking phone status", e);
        }
    }

    private void batteryStatus(int batteryStatus, int temperatureStatus) {
        if (Constants.APPLY_BATTERY_CHECK && batteryStatus == 1) {
            errorShown = true;
            // Only use TTS/retry logic if the app is in blindness mode.
                Handler writeHandler = new Handler(Looper.getMainLooper());

                // We use a final reference because it's captured by the Runnable
                final String[] finalErrorMessage = {null};

                // Recursive Runnable for TTS retry and language check
                Runnable ttsRetryRunnable = new Runnable() {
                    @Override
                    public void run() {
                        if (ttsManager.isReady()) {
                            // SUCCESS: TTS is ready. Get the localized message based on the active language.
                            String currentLangCode = ttsManager.tts.getVoice().getLocale().getLanguage();
                            finalErrorMessage[0] = UtilsKt.load_batteryLowText(appContext, currentLangCode);
                            showErrorAndShutdown(finalErrorMessage[0]);
                        } else{
                            Log.w(TAG, "TTS not ready on attempt. Retrying...");
                            writeHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS); // Retry after delay
                        }
                    }
                };
                writeHandler.post(ttsRetryRunnable);
            }
        else
            temperatureStatus(temperatureStatus);
    }

    private void temperatureStatus(int temperatureStatus) {
        if (Constants.APPLY_TEMPERATURE_CHECK && temperatureStatus == 1) {
            errorShown = true;
                // Only use TTS/retry logic if the app is in blindness mode.
                Handler writeHandler = new Handler(Looper.getMainLooper());

                // We use a final reference because it's captured by the Runnable
                final String[] finalErrorMessage = {null};

                // Recursive Runnable for TTS retry and language check
                Runnable ttsRetryRunnable = new Runnable() {
                    @Override
                    public void run() {
                        if (ttsManager.isReady()) {
                            // SUCCESS: TTS is ready. Get the localized message based on the active language.
                            String currentLangCode = ttsManager.tts.getVoice().getLocale().getLanguage();
                            finalErrorMessage[0] = UtilsKt.load_tempErrorText(appContext, currentLangCode);
                            showErrorAndShutdown(finalErrorMessage[0]);
                        } else{
                            Log.w(TAG, "TTS not ready on attempt. Retrying...");
                            writeHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS); // Retry after delay
                        }
                    }
                };
                writeHandler.post(ttsRetryRunnable);
            }
    }

    private void showAlertDialogAndScheduleShutdown(String message) {
        if (currentActivity == null || currentActivity.isFinishing()) {
            // Cannot show alert, but still schedule a shutdown for safety
            handler.postDelayed(this::exitApp, 100);
            return;
        }

        // 1. Show the AlertDialog on the main thread (No Exit button)
        currentActivity.runOnUiThread(() -> new AlertDialog.Builder(currentActivity)
                .setTitle(UtilsKt.load_criticalWarning(appContext,ttsManager.tts.getVoice().getLocale().getLanguage()))
                .setMessage(message)
                .setCancelable(false) // User cannot dismiss this
                .show());

        // 2. Schedule automatic shutdown after the delay
        handler.postDelayed(this::exitApp, Constants.SHUTDOWN_DELAY_MS);
    }

    private void showErrorAndShutdown(String message) {
        if (currentActivity != null && !currentActivity.isFinishing()) {

            // 1. Conditional Speaking Logic (with loop/retry)
            if (AppConfig.blindness) {
                Log.d(TAG, "App is in blindness mode, attempting TTS.");

                Handler speakHandler = new Handler(Looper.getMainLooper());

                // Recursive Runnable for TTS retry
                Runnable ttsRetryRunnable = new Runnable() {
                    @Override
                    public void run() {
                        // Assuming TTSManager has an isReady() check
                        if (ttsManager.isReady()) {
                            // SUCCESS: Speak the message and immediately proceed.
                            ttsManager.speak(message, AppConfig.tts_pitch, AppConfig.tts_speech_rate);
                            // Proceed to shutdown *after* speaking

                            handler.postDelayed(PhoneStatusMonitor.this::exitApp, Constants.BLINDNESS_SHUTDOWN_DELAY_MS);
                        } else{
                            Log.w(TAG, "TTS not ready on attempt ");
                            speakHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS); // Retry after delay
                        }
                    }
                };
                // Start the attempt loop
                speakHandler.post(ttsRetryRunnable);
            } else {
                // Not in blindness mode: proceed directly to show alert and schedule shutdown
                showAlertDialogAndScheduleShutdown(message);
            }
        }
    }

    public void shutdownApp(ErrorDialogManager errorManager) {
        if(errorManager!=null) {
            errorManager.showDialog();
            new Handler(Looper.getMainLooper()).postDelayed(this::exitApp, Constants.ERROR_READ_DELAY);
        }
       else
           exitApp();
    }

    private void exitApp(){
        stopMonitoring();

        // Crucial: Shut down the TTS engine first
        if (ttsManager != null) {
            ttsManager.shutdown();
        }

        if (currentActivity != null) {
            // Close all activities associated with this application
            currentActivity.finishAffinity();
        }
        // Force exit after a short delay to allow the OS to clean up
        new Handler(Looper.getMainLooper()).postDelayed(() -> System.exit(0), 500);
    }

    private void startMonitoring() {
        if (!isMonitoring) {
            isMonitoring = true;
            errorShown = false;
            handler.post(monitoringRunnable);
            Log.d(TAG, "Monitoring started");
        }
    }

    private void stopMonitoring() {
        if (isMonitoring) {
            isMonitoring = false;
            handler.removeCallbacks(monitoringRunnable);
            Log.d(TAG, "Monitoring stopped");
        }
    }

    public void pauseMonitoring() {
        isPaused = true;
        Log.d(TAG, "Monitoring paused");
    }

    public void resumeMonitoring() {
        isPaused = false;
        Log.d(TAG, "Monitoring resumed");
    }

    public TTSManager getTTSManager() {
        return ttsManager;
    }

    // --- ActivityLifecycleCallbacks Implementation (omitted for brevity, assume they are present) ---
    // (Ensure all lifecycle callback methods are fully present in your final file)
    @Override
    public void onActivityCreated(@NonNull Activity activity, @Nullable Bundle savedInstanceState) {
        currentActivity = activity;
    }

    @Override
    public void onActivityStarted(@NonNull Activity activity) {
        activeActivityCount++;
        currentActivity = activity;
        if (activeActivityCount == 1) {
            startMonitoring(); // Start monitoring when the first activity starts
        }
    }

    @Override
    public void onActivityResumed(@NonNull Activity activity) {
        currentActivity = activity;
    }

    @Override
    public void onActivityPaused(@NonNull Activity activity) {
    }

    @Override
    public void onActivityStopped(@NonNull Activity activity) {
        activeActivityCount--;
        if (activeActivityCount == 0) {
            stopMonitoring(); // Stop monitoring when the last activity stops
        }
    }

    @Override
    public void onActivitySaveInstanceState(@NonNull Activity activity, @NonNull Bundle outState) {
    }

    @Override
    public void onActivityDestroyed(@NonNull Activity activity) {
        if (currentActivity == activity) {
            currentActivity = null;
        }
    }

    public void shutdown() {
        stopMonitoring();
        if (appContext instanceof Application) {
            ((Application) appContext).unregisterActivityLifecycleCallbacks(this);
        }
        // Crucial: Shut down the TTS engine when the monitor is shut down
        if (ttsManager != null) {
            ttsManager.shutdown();
        }
    }
}