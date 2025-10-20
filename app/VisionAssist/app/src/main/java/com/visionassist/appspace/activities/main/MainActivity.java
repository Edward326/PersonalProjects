package com.visionassist.appspace.activities.main;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.compose.ui.platform.ComposeView;
import com.visionassist.appspace.ExceptionVisionAssist;
import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.R;
import com.visionassist.appspace.database.DBManager;
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import com.visionassist.appspace.models.ttsengine.TTSManager;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.BackgroundTaskExecutor;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.PermissionChecker;
import com.visionassist.appspace.utils.Utils;
import org.json.JSONObject;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private PhoneStatusMonitor monitor = PhoneStatusMonitor.getInstance();
    private TTSManager ttsManager = monitor.getTTSManager();
    private DBManager dbManager = monitor.getDBManager();
    private LoadingManager loadingManager;
    private BackgroundTaskExecutor backgroundExecutor = BackgroundTaskExecutor.getInstance();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 1. Initialize views
        ImageView logoImage = findViewById(R.id.logo_image);
        ComposeView loadingBox = findViewById(R.id.loading_box);
        loadingManager = new LoadingManager(loadingBox, true, this);
        loadingManager.setupLoadingBox();

        logoImage.setVisibility(View.VISIBLE);
        loadingManager.showLoading("Verifying profile, please wait");

        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            // Check permissions first (quick, can be done on main thread)
            PermissionChecker.checkAndRequestPermissions(this, MainActivity.class, loadingManager, false);

            // Execute profile check in background
            backgroundExecutor.executeAsync(
                    () -> {
                        Log.d(TAG, "Starting profile check in background");
                        return Utils.checkProfile(this);
                    },
                    // Callback: Handle result
                    new BackgroundTaskExecutor.TaskCallback<Pair<Integer, JSONObject>>() {
                        @Override
                        public void onSuccess(Pair<Integer, JSONObject> profileStatusDecider) throws Exception {
                            if (profileStatusDecider.first != 0) {
                                // Profile needs selection/configuration
                                Utils.profileSelector(profileStatusDecider, loadingManager);
                            } else {
                                // Profile is valid, proceed with upload and TTS initialization
                                handleValidProfile(profileStatusDecider.second);
                            }
                        }

                        public void onError(Exception e) {
                            if (e instanceof ExceptionVisionAssist) {
                                LoadingManager ref = ((ExceptionVisionAssist) e).getLoadingManager();
                                int errorCode = ((ExceptionVisionAssist) e).getErrorCode();

                                Log.e(TAG, "Thrown special exception, error code: " + errorCode);

                                ErrorDialogManager errorDialog = new ErrorDialogManager(monitor.getCurrentActivity());
                                errorDialog.setupDialog(errorCode, String.valueOf(R.string.exit_error_en));
                                if (ref != null) ref.hideLoading();
                                monitor.shutdownApp(errorDialog, monitor.getCurrentContext());
                            } else {
                                Log.e(TAG, "Thrown exception, explanation: ", e);
                                ErrorDialogManager errorDialog = new ErrorDialogManager(monitor.getCurrentActivity());
                                errorDialog.setupDialog(Constants.EXCEPTION_CLASS_ERROR, String.valueOf(R.string.exit_error_en));
                                monitor.shutdownApp(errorDialog, monitor.getCurrentContext());
                            }
                        }
                    }
            );
        }, Constants.ANIMATION_DELAY + 1000);
    }

    private void handleValidProfile(JSONObject profileData) {
        // Execute profile upload in background
        backgroundExecutor.executeAsync(
                // Background task: Upload profile
                () -> {
                    Log.d(TAG, "Uploading profile in background");
                    Utils.uploadProfile(profileData, loadingManager);
                },
                // On complete: Wait for TTS and navigate
                this::waitForTTSAndNavigate,
                // On error: Handle error
                () -> {
                }
        );
    }

    private void waitForTTSAndNavigate() {
        Handler writeHandler = new Handler(Looper.getMainLooper());

        // Recursive Runnable for TTS retry and language check
        Runnable ttsRetryRunnable = new Runnable() {
            @Override
            public void run() {
                if (ttsManager.isReady()) {
                    navigateToHome();
                } else {
                    Log.w(TAG, "TTS not ready on attempt. Retrying...");
                    writeHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS);
                }
            }
        };
        writeHandler.post(ttsRetryRunnable);
    }


    // onResume method is needed in case of missing TTS language data
    @Override
    protected void onResume() {
        super.onResume();

        if (monitor.getTTSManager() != null) {
            monitor.getTTSManager().recheckPendingLanguage();
        }

        // Execute TTS check in background to avoid blocking UI
        backgroundExecutor.executeAsync(
                // Background task: Check if TTS is ready
                () -> {
                    Log.d(TAG, "Checking TTS status in onResume");
                    return ttsManager.isReady();
                },
                // Callback: Handle TTS status
                new BackgroundTaskExecutor.TaskCallback<Boolean>() {
                    @Override
                    public void onSuccess(Boolean isReady) {
                        if (isReady) {
                            navigateToHome();
                        } else {
                            waitForTTSInResume();
                        }
                    }

                    @Override
                    public void onError(Exception e) {
                        Log.e(TAG, "Error checking TTS in onResume", e);
                        waitForTTSInResume(); // Fallback to retry mechanism
                    }
                }
        );
    }

    private void waitForTTSInResume() {
        Handler writeHandler = new Handler(Looper.getMainLooper());

        Runnable ttsRetryRunnable = new Runnable() {
            @Override
            public void run() {
                if (ttsManager.isReady()) {
                    navigateToHome();
                } else {
                    Log.w(TAG, "TTS not ready in onResume. Retrying...");
                    writeHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS);
                }
            }
        };
        writeHandler.post(ttsRetryRunnable);
    }

    private void navigateToHome() {
        monitor.isProfileLoaded(true);
        //make sync here
        Class<?> nextActivityClass = (AppConfig.blindness)
                ? BlindHomeActivity.class
                : HomeActivity.class;
        Intent intent = new Intent(MainActivity.this, nextActivityClass);
        loadingManager.hideLoading();
        startActivity(intent);
    }
}