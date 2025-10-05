package com.visionassist.appspace.activities.main;

import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.compose.ui.platform.ComposeView;
import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.R;
import com.visionassist.appspace.jetpack.managers.LoadingManager;

public class MainActivity extends AppCompatActivity {

    private LoadingManager loadingManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 1. Initialize views
        ImageView logoImage = findViewById(R.id.logo_image);
        ComposeView loadingBox = findViewById(R.id.loading_box);
        loadingManager = new LoadingManager(loadingBox,true,this);
        loadingManager.setupLoadingBox();

        logoImage.setVisibility(View.VISIBLE);

        //loadingManager.showLoading("Verifying profile, please wait");

        //Utils.checkProfile(this);
        /*
         PhoneStatusMonitor monitor = PhoneStatusMonitor.getInstance();
        TTSManager ttsManager = monitor.getTTSManager();

        // Use it to change language
        findViewById(R.id.romanian_button).setOnClickListener(v -> {
            Language romanian = new Language("ro", "Romanian", "RO");
            ttsManager.changeLanguage(romanian, this);
        });
         */

        //loadingManager.hideLoading();
    }
    private void simulateLoadingTask() {
        new Thread(() -> {
            try {
                // ⏳ Simulate some 5-second task
                runOnUiThread(() -> loadingManager.showLoading("Please wait"));
                Thread.sleep(5000);
                runOnUiThread(() -> loadingManager.hideLoading());
                Thread.sleep(2000);
                runOnUiThread(() -> loadingManager.showLoading("Va rugam asteptati"));
                Thread.sleep(5000);
                runOnUiThread(() -> loadingManager.hideLoading());
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }).start();
    }
    /*
    @Override
    protected void onResume() {
        super.onResume();
        // Check if language was installed
        PhoneStatusMonitor monitor = PhoneStatusMonitor.getInstance();
        if (monitor != null && monitor.getTTSManager() != null) {
            monitor.getTTSManager().recheckPendingLanguage();
        }
    }
    */
}