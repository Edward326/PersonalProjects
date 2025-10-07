package com.visionassist.appspace.activities.newprofile;

import static com.visionassist.appspace.utils.Constants.SETTINGS_REQUEST;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.compose.ui.platform.ComposeView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.R;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import com.visionassist.appspace.jetpack.managers.PermissionDialogManager;
import com.visionassist.appspace.utils.Constants;

public class PermissionsActivity extends AppCompatActivity {
    private static final String TAG = "PermissionsActivity";

    private int permissionOption;
    private String nextActivityClassName;
    private PermissionDialogManager dialogManager;
    private PermissionDialogManager dialogManagerSettings;

    private boolean cameraGranted = false;
    private boolean storageGranted = false;
    private boolean waitingForSettingsReturn = false;
    private String currentPermissionType = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
        if (phoneMonitor != null) {
            phoneMonitor.pauseMonitoring();
        }

        setContentView(R.layout.activity_permissions);

        // Initialize dialog box
        TextView titleView = findViewById(R.id.permissions_text);
        ComposeView dialogBox = findViewById(R.id.permission_dialog_box);
        ComposeView loadingBox = findViewById(R.id.loading_box);
        LoadingManager loadingManager = new LoadingManager(loadingBox, false, this);
        loadingManager.setupLoadingBox();
        dialogManager = new PermissionDialogManager(dialogBox,false,false,this);
        dialogManagerSettings = new PermissionDialogManager(dialogBox,false,true,this);
        Intent intent = getIntent();
        permissionOption = intent.getIntExtra(Constants.EXTRA_PERMISSION_OPTION, 0);
        nextActivityClassName = intent.getStringExtra(Constants.EXTRA_NEXT_ACTIVITY);

        titleView.setVisibility(View.VISIBLE);
        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            Log.d(TAG, "Permission option: " + permissionOption);
            handlePermissions();
            //if (phoneMonitor != null) {
            //    phoneMonitor.resumeMonitoring();
            //}
        }, Constants.PERMISSION_SLEEP);
    }

    private void handlePermissions() {
        switch (permissionOption) {
            case 0: // All permissions missing
                handleAllPermissions();
                break;
            case 1: // Camera permission missing
                handleCameraPermission();
                break;
            case 2: // Storage permissions missing
                handleStoragePermissions();
                break;
            default:
                navigateToNextActivity();
        }
    }

    // Option 0: Handle all permissions
    private void handleAllPermissions() {
        // First check and request camera, then storage
        handleCameraPermission();
    }

    // Option 1: Handle camera permission only
    private void handleCameraPermission() {
        currentPermissionType = "camera";

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            cameraGranted = true;

            // If we're handling all permissions, move to storage
            if (permissionOption == 0) {
                handleStoragePermissions();
            } else {
                navigateToNextActivity();
            }
            return;
        }

        // Request camera permission
        ActivityCompat.requestPermissions(
                this,
                new String[]{Manifest.permission.CAMERA},
                Constants.CAMERA_PERMISSION_REQUEST
        );
    }

    // Option 2: Handle storage permissions only
    private void handleStoragePermissions() {
        currentPermissionType = "storage";

        String[] storagePerms = getStoragePermissions();
        boolean allGranted = true;

        for (String perm : storagePerms) {
            if (ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                break;
            }
        }

        if (allGranted) {
            storageGranted = true;
            navigateToNextActivity();
            return;
        }

        // Request storage permissions
        ActivityCompat.requestPermissions(
                this,
                storagePerms,
                Constants.STORAGE_PERMISSION_REQUEST
        );
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        boolean allGranted = true;
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                break;
            }
        }

        if (requestCode == Constants.CAMERA_PERMISSION_REQUEST) {
            if (allGranted) {
                cameraGranted = true;
                Log.d(TAG, "Camera permission granted");

                // If handling all permissions, move to storage
                if (permissionOption == 0) {
                    handleStoragePermissions();
                } else {
                    navigateToNextActivity();
                }
            } else {
                // Check if we should show rationale or if user selected "Never ask again"
                if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
                    showPermissionDeniedDialog("camera");
                } else {
                    showGoToSettingsDialog("camera");
                }
            }
        } else if (requestCode == Constants.STORAGE_PERMISSION_REQUEST) {
            if (allGranted) {
                storageGranted = true;
                Log.d(TAG, "Storage permissions granted");
                navigateToNextActivity();
            } else {
                String[] storagePerms = getStoragePermissions();
                if (shouldShowRequestPermissionRationale(storagePerms[0])) {
                    showPermissionDeniedDialog("storage");
                } else {
                    showGoToSettingsDialog("storage");
                }
            }
        }
    }

    private void showPermissionDeniedDialog(String permType) {
        dialogManager.setupDialog(() -> {
            dialogManager.hideDialog();

            // Retry permission request
            new Handler(Looper.getMainLooper()).postDelayed(() -> {
                if (permType.equals("camera")) {
                    handleCameraPermission();
                } else {
                    handleStoragePermissions();
                }
            }, 500);
            return null;
        });

        dialogManager.showDialog();
    }

    private void showGoToSettingsDialog(String permType) {
        waitingForSettingsReturn = true;
        currentPermissionType = permType;

        dialogManager.setupDialog(() -> {
            dialogManager.hideDialog();

            // Open app settings
            new Handler(Looper.getMainLooper()).postDelayed(() -> {
                Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                intent.setData(android.net.Uri.parse("package:" + getPackageName()));
                startActivityForResult(intent, SETTINGS_REQUEST);
            }, 500);
            return null;
        });

        dialogManager.showDialog();
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (waitingForSettingsReturn) {
            waitingForSettingsReturn = false;

            // Check if permission was granted in settings
            new Handler(Looper.getMainLooper()).postDelayed(() -> {
                checkPermissionAfterSettings();
            }, 500);
        }
    }

    private void checkPermissionAfterSettings() {
        if (currentPermissionType.equals("camera")) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED) {
                cameraGranted = true;

                if (permissionOption == 0) {
                    handleStoragePermissions();
                } else {
                    navigateToNextActivity();
                }
            } else {
                // Still not granted, show dialog again
                showGoToSettingsDialog("camera");
            }
        } else if (currentPermissionType.equals("storage")) {
            String[] storagePerms = getStoragePermissions();
            boolean allGranted = true;

            for (String perm : storagePerms) {
                if (ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }

            if (allGranted) {
                storageGranted = true;
                navigateToNextActivity();
            } else {
                showGoToSettingsDialog("storage");
            }
        }
    }

    private String[] getStoragePermissions() {
        if (Constants.API_LEVEL >= Build.VERSION_CODES.TIRAMISU) {
            return new String[]{Manifest.permission.READ_MEDIA_IMAGES};
        } else {
            return new String[]{
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            };
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
}