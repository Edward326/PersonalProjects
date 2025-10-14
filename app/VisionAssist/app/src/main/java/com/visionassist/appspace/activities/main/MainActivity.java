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
import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.R;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import com.visionassist.appspace.models.ttsengine.TTSManager;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.PermissionChecker;
import com.visionassist.appspace.utils.Utils;
import org.json.JSONObject;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private PhoneStatusMonitor monitor = PhoneStatusMonitor.getInstance();
    private TTSManager ttsManager=monitor.getTTSManager();
    private LoadingManager loadingManager;

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
        new Handler(Looper.getMainLooper()).postDelayed(() ->
        {
            PermissionChecker.checkAndRequestPermissions(this, MainActivity.class, loadingManager, false);
            Pair<Integer, JSONObject> profileStatusDecider = Utils.checkProfile(this);
            if (profileStatusDecider.first != 0)
                Utils.profileSelector(this, this, profileStatusDecider, loadingManager);
            else {
                Utils.uploadProfile(this, this, profileStatusDecider.second);
                Handler writeHandler = new Handler(Looper.getMainLooper());

                // Recursive Runnable for TTS retry and language check
                Runnable ttsRetryRunnable = new Runnable() {
                    @Override
                    public void run() {
                        if (ttsManager.isReady()) {
                            monitor.isProfileLoaded(true);
                            Class<?> nextActivityClass = (AppConfig.blindness) ? BlindHomeActivity.class : HomeActivity.class;
                            Intent intent = new Intent(MainActivity.this, nextActivityClass);
                            loadingManager.hideLoading();
                            new Handler(Looper.getMainLooper()).postDelayed(() -> MainActivity.this.startActivity(intent), Constants.ANIMATION_DELAY);  // 100ms delay
                        } else {
                            Log.w(TAG, "TTS not ready on attempt. Retrying...");
                            writeHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS); // Retry after delay
                        }
                    }
                };
                writeHandler.post(ttsRetryRunnable);
            }
        }, Constants.ANIMATION_DELAY + 1500);
    }

    //onResume method is needed in case of missing TTS language data
    @Override
    protected void onResume() {
        super.onResume();
        if (monitor.getTTSManager() != null) {
            monitor.getTTSManager().recheckPendingLanguage();
        }
        Handler writeHandler = new Handler(Looper.getMainLooper());
        // Recursive Runnable for TTS retry and language check
        Runnable ttsRetryRunnable = new Runnable() {
            @Override
            public void run() {
                if (ttsManager.isReady()) {
                    monitor.isProfileLoaded(true);
                    Class<?> nextActivityClass = (AppConfig.blindness) ? BlindHomeActivity.class : HomeActivity.class;
                    Intent intent = new Intent(MainActivity.this, nextActivityClass);
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> MainActivity.this.startActivity(intent), Constants.ANIMATION_DELAY);  // 100ms delay
                } else {
                    Log.w(TAG, "TTS not ready on attempt. Retrying...");
                    writeHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS); // Retry after delay
                }
            }
        };
        writeHandler.post(ttsRetryRunnable);
    }
}