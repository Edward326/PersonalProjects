package com.visionassist.appspace.activities.newprofile;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import com.visionassist.appspace.R;
import com.visionassist.appspace.utils.Constants;

public class PermissionsActivity extends AppCompatActivity {
    private static final String TAG = "PermissionsActivity";
    private static final int PERMISSION_REQUEST_CODE = 100;

    // Instance variables to store intent data
    private int permissionOption;
    private String nextActivityClassName;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_permissions);

        // Retrieve data from intent
        Intent intent = getIntent();
        permissionOption = intent.getIntExtra(Constants.EXTRA_PERMISSION_OPTION, 0);
        nextActivityClassName = intent.getStringExtra(Constants.EXTRA_NEXT_ACTIVITY);

        // Initialize UI and request permissions
        //initializePermissionRequest();
    }

    private void initializePermissionRequest() {
        // Determine which permissions to request based on option
        String[] permissionsToRequest = getRequiredPermissions();

        if (permissionsToRequest.length > 0) {
            ActivityCompat.requestPermissions(
                    this,
                    permissionsToRequest,
                    PERMISSION_REQUEST_CODE
            );
        } else {
            // All permissions already granted
            navigateToNextActivity();
        }
    }

    private String[] getRequiredPermissions() {
        switch (permissionOption) {
            case 0: // All permissions not granted
                return getAllPermissions();

            case 1: // Camera permission not granted
                return new String[]{Manifest.permission.CAMERA};

            case 2: // Storage permissions not granted
                return getStoragePermissions();

            default:
                return new String[0];
        }
    }

    private String[] getAllPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            return new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.READ_MEDIA_IMAGES
            };
        } else {
            return new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            };
        }
    }

    private String[] getStoragePermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            return new String[]{Manifest.permission.READ_MEDIA_IMAGES};
        } else {
            return new String[]{
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            };
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == PERMISSION_REQUEST_CODE) {
            boolean allGranted = true;

            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }

            if (allGranted) {
                Log.d(TAG, "All permissions granted");
                navigateToNextActivity();
            } else {
                Log.d(TAG, "Some permissions denied");
                // Handle permission denial
                handlePermissionDenial();
            }
        }
    }

    private void navigateToNextActivity() {
        if (nextActivityClassName != null && !nextActivityClassName.isEmpty()) {
            try {
                Class<?> nextActivityClass = Class.forName(nextActivityClassName);
                Intent intent = new Intent(this, nextActivityClass);
                startActivity(intent);
                finish();
            } catch (ClassNotFoundException e) {
                Log.e(TAG, "Next activity class not found: " + nextActivityClassName, e);
                finish();
            }
        } else {
            Log.e(TAG, "Next activity class name is null or empty");
            finish();
        }
    }

    private void handlePermissionDenial() {
        // You can show a dialog explaining why permissions are needed
        // Or allow user to try again
        // For now, just finish the activity
        Log.w(TAG, "Permissions denied, finishing activity");
        finish();
    }

    // Getter methods to access instance variables if needed elsewhere
    public int getPermissionOption() {
        return permissionOption;
    }

    public String getNextActivityClassName() {
        return nextActivityClassName;
    }
}