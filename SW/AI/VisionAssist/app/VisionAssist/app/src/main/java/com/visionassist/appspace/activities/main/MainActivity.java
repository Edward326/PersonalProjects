package com.visionassist.appspace.activities.main;

import android.content.Context;
import android.content.Intent;
import android.media.AudioManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Pair;
import android.view.KeyEvent;
import android.view.View;
import android.view.accessibility.AccessibilityManager;
import android.widget.ImageView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.compose.ui.platform.ComposeView;
import androidx.core.content.FileProvider;
import androidx.core.view.WindowCompat;

import com.visionassist.appspace.ExceptionVisionAssist;
import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.R;
import com.visionassist.appspace.activities.tabs.home.caption.BlindCaptionActivity;
import com.visionassist.appspace.activities.tabs.home.caption.CaptionActivity;
import com.visionassist.appspace.activities.tabs.home.detection.BlindDetectionActivity;
import com.visionassist.appspace.activities.tabs.home.detection.LiveDetectionActivity;
import com.visionassist.appspace.activities.tabs.home.detection.StaticDetectionActivity;
import com.visionassist.appspace.database.DBManager;
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import com.visionassist.appspace.models.ttsengine.TTSManager;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.BackgroundTaskExecutor;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.PermissionChecker;
import com.visionassist.appspace.utils.Utils;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private PhoneStatusMonitor monitor = PhoneStatusMonitor.getInstance();
    private TTSManager ttsManager = monitor.getTTSManager();
    private BackgroundTaskExecutor backgroundExecutor = BackgroundTaskExecutor.getInstance();
    private Handler handler = new Handler(Looper.getMainLooper());
    private Handler ttsHandler = new Handler(Looper.getMainLooper());
    private LoadingManager loadingManager;
    private JSONObject profileData;
    private boolean waitingForTTSLanguage = false;
    private boolean profileAlreadyChecked = false;
    private boolean isNavigateToHome = false;
    private boolean ready = false;
    private int quickActionIndex;
    private ActivityResultLauncher<Uri> takePictureLauncher;
    private Uri currentPhotoUri;
    private String currentPhotoPath;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);

        registerCameraLauncher();

        setContentView(R.layout.activity_main);
        disableTalkBackForActivity();
        AudioManager audioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
        int maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC);
        audioManager.setStreamVolume(
                AudioManager.STREAM_MUSIC,  // Stream type
                maxVolume,                   // Volume level (max)
                0                            // Flags (0 = no UI)
        );

        ImageView logoImage = findViewById(R.id.logo_image);
        ComposeView loadingBox = findViewById(R.id.loading_box);
        logoImage.setVisibility(View.VISIBLE);
        monitor.checkPhoneStatus();

        loadingManager = new LoadingManager(loadingBox, true, this);
        loadingManager.setupLoadingBox();

        Intent intent = getIntent();
        quickActionIndex = intent.getIntExtra("QUICK_ACTION_INDEX", 0);

        loadingManager.showLoading("Verifying permissions, please wait");
        handler.postDelayed(() -> PermissionChecker.checkAndRequestPermissions(this, true, this::checkProfileTask), Constants.ANIMATION_DELAY);
    }

    private void registerCameraLauncher() {
        takePictureLauncher = registerForActivityResult(
                new ActivityResultContracts.TakePicture(),
                isSuccess -> {
                    if (isSuccess) {
                        navigateToTargetActivityWithBitmap();
                    } else {
                        Log.e(TAG, "Image capture failed or cancelled");
                        showCameraError();
                    }
                }
        );
        Log.d(TAG, "Camera launcher registered");
    }

    private void navigateToTargetActivityWithBitmap() {
        try {
            // Determine target activity class based on quick action
            Class<?> targetActivityClass =
                    AppConfig.blindness ? BlindCaptionActivity.class :
                            switch (quickActionIndex) {
                                case 1 -> // StaticDetection
                                        StaticDetectionActivity.class;
                                case 3 -> // Caption
                                        CaptionActivity.class;
                                default -> StaticDetectionActivity.class;
                            };

            // Create intent with image URI
            Intent intent = new Intent(this, targetActivityClass);
            intent.putExtra(Constants.EXTRA_IMAGE_URI, currentPhotoUri.toString());
            intent.putExtra("ABSOLUTE_PATH", currentPhotoPath);
            intent.putExtra("QUICK_ACTION_INDEX", quickActionIndex);

            loadingManager.hideLoading();
            startActivity(intent);
            finish();
        } catch (Exception e) {
            Log.e(TAG, "Error navigating to target activity", e);
            showCameraError();
        }
    }

    private void launchCamera() {
        try {
            File photoFile = File.createTempFile(
                    "temp_visionassist",
                    ".jpg",
                    getCacheDir()
            );

            currentPhotoPath = photoFile.getAbsolutePath();

            currentPhotoUri = FileProvider.getUriForFile(
                    this,
                    getPackageName() + ".fileprovider",
                    photoFile
            );

            Log.d(TAG, "Launching camera with URI: " + currentPhotoUri);
            takePictureLauncher.launch(currentPhotoUri);
        } catch (IOException e) {
            Log.e(TAG, "Error creating temp file", e);
            showCameraError();
        }
    }

    private void showCameraError() {
        PhoneStatusMonitor monitor = PhoneStatusMonitor.getInstance();
        ErrorDialogManager errorDialog = new ErrorDialogManager(monitor.getCurrentActivity());
        errorDialog.setupDialog(Constants.CAMERA_MAKE_PHOTO);
        monitor.shutdownApp(errorDialog, monitor.getCurrentContext());
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (isNavigateToHome) {
            navigateToHome();
        } else if (waitingForTTSLanguage) {
            ttsHandler.removeCallbacksAndMessages(null);
            Log.d(TAG, "Returned from TTS settings, rechecking language");
            ttsManager.recheckPendingLanguage();
            waitForTTSAndNavigate();
        } else if (profileAlreadyChecked) {
            uploadProfileTask();
        } else if (PhoneStatusMonitor.getInstance().isReturningFromPermissions) {
            handler.postDelayed(() -> PermissionChecker.checkAndRequestPermissions(this, true, this::checkProfileTask), 1000);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();

        if (!ttsManager.isDoneSpeaking())
            ttsManager.stopSpeaking();
    }

    private void checkProfileTask() {
        loadingManager.changeText("Verifying profile, please wait");
        handler.postDelayed(() -> {
            try {
                Pair<Integer, JSONObject> profileStatusDecider = Utils.checkProfile(this);
                if (profileStatusDecider.first != 0) {
                    Utils.profileSelector(profileStatusDecider, loadingManager);
                } else {
                    profileData = profileStatusDecider.second;
                    uploadProfileTask();
                }
            } catch (Exception e) {
                handleProfileError(e);
            }
        }, 1500);
    }

    private void uploadProfileTask() {
        profileAlreadyChecked = true;
        loadingManager.changeText("Uploading profile, please wait");
        handler.postDelayed(() ->
                        Utils.uploadProfile(profileData,
                                () -> PhoneStatusMonitor.getInstance().getModelManager().loadAssets(this::setTTSLanguage)),
                1500);
    }

    private void setTTSLanguage() {
        if (!AppConfig.mainLanguage.getCode().equals(ttsManager.getCurrentLocale().getLanguage())) {
            Log.d(TAG, "TTS is not init on the lang selected");
            waitingForTTSLanguage = true;
            ttsManager.changeLanguage(AppConfig.mainLanguage, this);
            waitForTTSAndNavigate();
        } else {
            Log.d(TAG, "TTS is already init, continue the thread");
            waitForTTSAndNavigate();
        }
    }

    private void waitForTTSAndNavigate() {
        Runnable checkTTS = new Runnable() {
            @Override
            public void run() {
                if (ttsManager.isReady()) {
                    Log.d(TAG, "TTS is ready, navigating to home");
                    waitingForTTSLanguage = false;
                    navigateToHome();
                } else {
                    Log.w(TAG, "TTS not ready, retrying...");
                    ttsHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS);
                }
            }
        };
        ttsHandler.post(checkTTS);
    }

    private void synchronizeProfile(DBManager dbManager) {
        backgroundExecutor.executeAsync(
                () -> {
                    Log.d(TAG, "Synchronizing profile with remote database");
                    dbManager.syncProfile(profileData);
                    return 0;
                },
                new BackgroundTaskExecutor.TaskCallback<>() {
                    @Override
                    public void onSuccess(Integer result) {
                        ready = true;
                    }

                    @Override
                    public void onError(Exception e) {
                        if (e instanceof JSONException) {
                            handleProfileError(new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager));
                        }
                        handleProfileError(e);
                    }
                }
        );
        Runnable checkSyncDone = new Runnable() {
            @Override
            public void run() {
                if (ready) {
                    navigateToHomeEnd();
                } else {
                    Log.e(TAG, "Sync not ready. Retrying...");
                    handler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS);
                }
            }
        };
        handler.post(checkSyncDone);
    }

    private void navigateToHome() {
        isNavigateToHome = true;
        try {
            if (quickActionIndex != 0)
                navigateToHomeEnd();
            else {
                DBManager dbManager = monitor.getDBManager();
                if (dbManager.isRemoteProfile(profileData, loadingManager))
                    synchronizeProfile(dbManager);
                else
                    navigateToHomeEnd();
            }
        } catch (ExceptionVisionAssist e) {
            handleProfileError(e);
        }
    }

    private void navigateToHomeEnd() {
        monitor.isProfileLoaded(true);

        if (quickActionIndex == 0) {
            Class<?> nextActivityClass = AppConfig.blindness
                    ? BlindHomeActivity.class
                    : HomeActivity.class;
            Intent intent = new Intent(this, nextActivityClass);
            loadingManager.hideLoading();
            startActivity(intent);
            finish();
        } else {
            loadingManager.hideLoading();
            if (!AppConfig.blindness)
                switch (quickActionIndex) {
                    case 1, 3:
                        launchCamera();
                        break;
                    case 2:
                        Intent intent = new Intent(this, LiveDetectionActivity.class);
                        intent.putExtra("QUICK_ACTION_INDEX", quickActionIndex);
                        startActivity(intent);
                        finish();
                }
            else switch (quickActionIndex) {
                case 1:
                    Intent intent = new Intent(this, BlindDetectionActivity.class);
                    intent.putExtra("QUICK_ACTION_INDEX", quickActionIndex);
                    startActivity(intent);
                    finish();
                    break;
                case 2:
                    launchCamera();
            }
        }
    }

    private void handleProfileError(Exception e) {
        if (e instanceof ExceptionVisionAssist) {
            LoadingManager ref = ((ExceptionVisionAssist) e).getLoadingManager();
            int errorCode = ((ExceptionVisionAssist) e).getErrorCode();

            Log.e(TAG, "Thrown special exception, error code: " + errorCode);

            ErrorDialogManager errorDialog = new ErrorDialogManager(monitor.getCurrentActivity());
            errorDialog.setupDialog(errorCode);
            if (ref != null) ref.hideLoading();
            monitor.shutdownApp(errorDialog, monitor.getCurrentContext());
        } else {
            Log.e(TAG, "Thrown exception, explanation: ", e);
            ErrorDialogManager errorDialog = new ErrorDialogManager(monitor.getCurrentActivity());
            errorDialog.setupDialog(Constants.EXCEPTION_CLASS_ERROR);
            monitor.shutdownApp(errorDialog, monitor.getCurrentContext());
        }
    }

    private void disableTalkBackForActivity() {
        try {
            // Check if TalkBack is currently enabled
            AccessibilityManager accessibilityManager =
                    (AccessibilityManager) getSystemService(ACCESSIBILITY_SERVICE);

            if (accessibilityManager == null || !accessibilityManager.isEnabled()) {
                Log.d(TAG, "TalkBack is not active system-wide, no need to disable it for activity");
                return;
            }

            // Get the root content view of the activity (android.R.id.content is the root container)
            View rootView = findViewById(android.R.id.content);
            if (rootView != null) {
                // Set the view as not important for accessibility
                // This prevents TalkBack from announcing content in this activity
                rootView.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS);
                Log.d(TAG, "TalkBack is active system-wide, TalkBack disabled for MainActivity");
            }
        } catch (Exception e) {
            Log.e(TAG, "Error disabling TalkBack for activity", e);
        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        // Use a switch statement for key code checks
        return switch (keyCode) {
            case KeyEvent.KEYCODE_VOLUME_DOWN -> {
                Log.d(TAG, "Volume button down pressed");
                yield true;
            }
            case KeyEvent.KEYCODE_VOLUME_UP -> {
                Log.d(TAG, "Volume button up pressed");
                yield true;
            }
            default ->

                // For all other keys, call the super implementation
                    super.onKeyDown(keyCode, event);
        };
    }
}