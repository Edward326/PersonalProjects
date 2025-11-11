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
            String content = "Data of the User:\n"+FileUtils.loadFileAsString(FileUtils.getProfileInputStream(this));
            Log.d(TAG, "HomeActivity created\n"+content);
        } catch (IOException e) {
            handleProfileError(e);
        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        // Use a switch statement for key code checks
        switch (keyCode) {
            case KeyEvent.KEYCODE_VOLUME_DOWN:
                Log.d(TAG, "Volume button down pressed");
                return true;
            case KeyEvent.KEYCODE_VOLUME_UP:
                Log.d(TAG, "Volume button up pressed");
                return true;
        }

        // For all other keys, call the super implementation
        return super.onKeyDown(keyCode, event);
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