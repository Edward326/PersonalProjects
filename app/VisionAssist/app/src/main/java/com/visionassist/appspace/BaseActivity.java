package com.visionassist.appspace;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;
import com.visionassist.appspace.activities.main.MainActivity;

/**
 * Base activity that handles process death recovery.
 * All activities should extend this instead of AppCompatActivity.
 */
public abstract class BaseActivity extends AppCompatActivity {
    private static final String TAG = "BaseActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Check if app was killed and restored
        if (PhoneStatusMonitor.getInstance().profileLoaded) {
            Log.w(TAG, "App state lost due to process death, restarting from MainActivity");
            restartFromMainActivity();
        }
    }

    /**
     * Restart the app from MainActivity to properly initialize everything.
     */
    protected void restartFromMainActivity() {
        Intent intent = new Intent(this, MainActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        startActivity(intent);
        finish();
    }
}