package com.visionassist.appspace.activities.main;

import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import com.visionassist.appspace.ExceptionVisionAssist;
import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.R;
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.FileUtils;
import java.io.IOException;

public class HomeActivity extends AppCompatActivity {
    private static final String TAG = "HomeActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        ImageView logoImage = findViewById(R.id.logo_image);
        logoImage.setVisibility(View.VISIBLE);

        try {
            String content1 = "Data of the hash_cache.json:\n"+FileUtils.loadFileAsString(FileUtils.getHashCacheInputStream(this));
            //String content2 = "\n\nData of the env_reports.json:\n"+FileUtils.loadFileAsString(FileUtils.getEnvReportsInputStream(this));
            String content3 = "\n\nData of the AppConfig:\n"+ AppConfig.listAppConfig();
            String content4="\n\nIsInitProfileLoaded:\t"+ PhoneStatusMonitor.getInstance().profileLoaded;
            String content5 = "\n\nData of the profile.json:\n"+FileUtils.loadFileAsString(FileUtils.getProfileInputStream(this));

            String concat=content1+content3+content4+content5;
            Log.d(TAG, "HomeActivity created\n\n---STATUS---\n"+concat);
        } catch (IOException e) {
            handleProfileError(e);
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

    private void handleProfileError(Exception e) {
        PhoneStatusMonitor monitor=PhoneStatusMonitor.getInstance();
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
}