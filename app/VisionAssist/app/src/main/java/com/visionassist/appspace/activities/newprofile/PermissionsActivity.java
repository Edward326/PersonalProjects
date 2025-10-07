package com.visionassist.appspace.activities.newprofile;

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
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
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

    private PhoneStatusMonitor phoneMonitor;
    private int permissionOption;
    private String nextActivityClassName;
    private PermissionDialogManager dialogManager;
    private PermissionDialogManager dialogManagerSettings;

    // Launcher for opening Settings
    private ActivityResultLauncher<Intent> settingsLauncher;

    private boolean waitingForSettingsReturn = false;
    private String currentPermissionType = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 1. Register the Launcher in onCreate
        settingsLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    // This callback runs when the user returns from the Settings app
                    if (waitingForSettingsReturn) {
                        waitingForSettingsReturn = false;

                        // 2. Post a delayed check to read the updated permission status reliably
                        new Handler(Looper.getMainLooper()).postDelayed(this::checkPermissionAfterSettings, 500);
                    }
                }
        );

        phoneMonitor=PhoneStatusMonitor.getInstance();
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
            new Handler(Looper.getMainLooper()).post(() -> {
                if (permType.equals("camera")) {
                    handleCameraPermission();
                } else {
                    handleStoragePermissions();
                }
            });
            return null;
        });

        // Add delay to allow Compose to finish setup
        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            dialogManager.showDialog();
        }, 100);  // 100ms delay
    }

    private void showGoToSettingsDialog(String permType) {
        waitingForSettingsReturn = true;
        currentPermissionType = permType;

        dialogManagerSettings.setupDialog(() -> {
            dialogManagerSettings.hideDialog();

            // Open app settings
            new Handler(Looper.getMainLooper()).post(() -> {
                Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                intent.setData(android.net.Uri.parse("package:" + getPackageName()));

                // Use the Launcher to start the Intent
                settingsLauncher.launch(intent);
            });
            return null;
        });

        // Add delay to allow Compose to finish setup
        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            dialogManagerSettings.showDialog();
        }, 100);  // 100ms delay
    }

    // Removed the old onResume logic entirely, as the settingsLauncher handles the return.

    // 3. The actual permission check function, called after the 500ms delay.
    private void checkPermissionAfterSettings() {
        if (currentPermissionType.equals("camera")) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED) {
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
                if (phoneMonitor != null) {
                    phoneMonitor.resumeMonitoring();
                }
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