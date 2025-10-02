package com.visionassist.appspace.activities.main;

import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.compose.ui.platform.ComposeView;
import com.visionassist.appspace.R;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import com.visionassist.appspace.utils.JSONValidation;
import com.visionassist.appspace.utils.Utils;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private LoadingManager loadingManager;
    private ImageView logoImage;
    private View loadingOverlay;
    private ComposeView loadingBox;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 1. Initialize views
        logoImage = findViewById(R.id.logo_image);
        loadingBox = findViewById(R.id.loading_box);
        loadingManager = new LoadingManager(loadingBox,true);
        loadingManager.setupLoadingBox();

        logoImage.setVisibility(View.VISIBLE);
        loadingManager.showLoading("Verifying profile, please wait");

        //Utils.checkProfile(this);


        loadingManager.hideLoading();
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
}