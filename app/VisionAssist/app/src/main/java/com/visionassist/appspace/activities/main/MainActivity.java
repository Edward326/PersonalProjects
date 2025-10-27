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
    private BackgroundTaskExecutor backgroundExecutor = BackgroundTaskExecutor.getInstance();
    private LoadingManager loadingManager;
    private JSONObject profileData;

    private boolean firstResume = true;
    private boolean waitingForTTSLanguage = false;
    private boolean profileAlreadyChecked = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ImageView logoImage = findViewById(R.id.logo_image);
        ComposeView loadingBox = findViewById(R.id.loading_box);
        logoImage.setVisibility(View.VISIBLE);
        loadingManager = new LoadingManager(loadingBox, true, this);
        loadingManager.setupLoadingBox();
    }

    @Override
    protected void onStart() {
        super.onStart();

        if (firstResume) {
            firstResume = false;
            loadingManager.showLoading("Verifying permissions, please wait");
            new Handler(Looper.getMainLooper()).postDelayed(() -> PermissionChecker.checkAndRequestPermissions(this, true), Constants.ANIMATION_DELAY + 1000);
        } else if (waitingForTTSLanguage) {
            // Returning from TTS settings, don't check permissions again
            Log.d(TAG, "Returning from TTS settings, skipping permission check");
        } else {
            // Returning from another activity, check permissions quickly
            new Handler(Looper.getMainLooper()).postDelayed(() -> PermissionChecker.checkAndRequestPermissions(this, true), 1000);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        // Case 1: Returning from TTS settings
        if (waitingForTTSLanguage) {
            Log.d(TAG, "Returned from TTS settings, rechecking language");
            ttsManager.recheckPendingLanguage();

            // Wait for TTS and navigate
            waitForTTSAndNavigate();
            return; // Don't do profile check again!
        }

        // Case 2: Profile already checked, just verify TTS is ready
        if (profileAlreadyChecked) {
            Log.d(TAG, "Profile already checked, waiting for TTS");
            waitForTTSAndNavigate();
            return;
        }

        loadingManager.changeText("Verifying profile, please wait");
        backgroundExecutor.executeAsync(
                () -> {
                    Log.d(TAG, "Starting profile check in background");
                    return Utils.checkProfile(this);
                },
                new BackgroundTaskExecutor.TaskCallback<Pair<Integer, JSONObject>>() {
                    @Override
                    public void onSuccess(Pair<Integer, JSONObject> profileStatusDecider) throws Exception {
                        if (profileStatusDecider.first != 0) {
                            Utils.profileSelector(profileStatusDecider, loadingManager);
                        } else {
                            profileData= profileStatusDecider.second;
                            handleValidProfile();
                        }
                    }

                    @Override
                    public void onError(Exception e) {
                        handleProfileError(e);
                    }
                }
        );
    }

    private void handleValidProfile() {
        profileAlreadyChecked = true;
        loadingManager.changeText("Loading profile, please wait");

        // Profile loaded, now setup TTS
        backgroundExecutor.executeAsync(
                () -> {
                    Log.d(TAG, "Loading profile in background");
                    Utils.uploadProfile(profileData, loadingManager);
                },
                    this::setupTTSAndNavigate,
                () -> {
                    Log.e(TAG, "Error loading profile");
                    handleProfileError(new Exception("Error loading profile"));
                }
        );
    }

    private void setupTTSAndNavigate() {
            waitingForTTSLanguage = true;
            ttsManager.changeLanguage(AppConfig.mainLanguage, this);
            waitForTTSAndNavigate();
    }

    private void waitForTTSAndNavigate() {
        Handler handler = new Handler(Looper.getMainLooper());
        Runnable checkTTS = new Runnable() {
            @Override
            public void run() {
                if (ttsManager.isReady()) {
                    Log.d(TAG, "TTS is ready, navigating to home");
                    waitingForTTSLanguage = false;
                    navigateToHome();
                } else {
                    Log.w(TAG, "TTS not ready, retrying...");
                    handler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS);
                }
            }
        };
        handler.post(checkTTS);
    }

    private void navigateToHome() {
        monitor.isProfileLoaded(true);
        //make the sync with firebase
        //loadingManager.changeText("Loading profile, please wait");
        //dbManager.autoSyncProfile(profileData,loadingManager);

        Class<?> nextActivityClass = AppConfig.blindness
                ? BlindHomeActivity.class
                : HomeActivity.class;
        Intent intent = new Intent(this, nextActivityClass);
        loadingManager.hideLoading();
        startActivity(intent);
        finish();
    }

    private void handleProfileError(Exception e) {
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