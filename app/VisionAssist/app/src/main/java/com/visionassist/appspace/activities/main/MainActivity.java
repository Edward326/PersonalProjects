package com.visionassist.appspace.activities.main;

import android.content.Intent;
import android.os.Bundle;
import android.util.Pair;
import android.view.View;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.compose.ui.platform.ComposeView;
import com.visionassist.appspace.R;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.PermissionChecker;
import com.visionassist.appspace.utils.Utils;
import org.json.JSONObject;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 1. Initialize views
        ImageView logoImage = findViewById(R.id.logo_image);
        ComposeView loadingBox = findViewById(R.id.loading_box);
        LoadingManager loadingManager = new LoadingManager(loadingBox,true,this);
        loadingManager.setupLoadingBox();

        logoImage.setVisibility(View.VISIBLE);
        loadingManager.showLoading("Verifying profile, please wait");
        Pair<Integer, JSONObject> profileStatusDecider=Utils.checkProfile(this);
        if(profileStatusDecider.first!=0)
            Utils.profileSelector(profileStatusDecider,loadingManager);
        else {
            Utils.uploadProfile(profileStatusDecider.second);
            Class<?> nextActivityClass = (AppConfig.blindness) ? BlindHomeActivity.class : HomeActivity.class;
            PermissionChecker.checkAndRequestPermissions(this, nextActivityClass, loadingManager,AppConfig.blindness);
            Intent intent = new Intent(this, nextActivityClass);
            loadingManager.hideLoading();
            this.startActivity(intent);
        }

        /*
        //in case you want to change the TTS language(global object)
        PhoneStatusMonitor monitor = PhoneStatusMonitor.getInstance();
        TTSManager ttsManager = monitor.getTTSManager();

        // Use it to change language
        findViewById(R.id.romanian_button).setOnClickListener(v -> {
            Language romanian = new Language("ro", "Romanian", "RO");
            ttsManager.changeLanguage(romanian, this);
        });

        //onResume method is needed in case of missing data, when the user returns from TTS settings to return back to thr current activity
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
}