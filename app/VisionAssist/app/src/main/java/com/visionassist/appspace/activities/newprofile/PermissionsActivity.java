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

    private ActivityResultLauncher<Intent> settingsLauncher;

    private boolean waitingForSettingsReturn = false;
    private String currentPermissionType = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        settingsLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (waitingForSettingsReturn) {
                        waitingForSettingsReturn = false;
                        new Handler(Looper.getMainLooper()).postDelayed(this::checkPermissionAfterSettings, 500);
                    }
                }
        );

        phoneMonitor = PhoneStatusMonitor.getInstance();
        if (phoneMonitor != null) {
            phoneMonitor.pauseMonitoring();
        }

        setContentView(R.layout.activity_permissions);

        TextView titleView = findViewById(R.id.permissions_text);
        ComposeView dialogBox = findViewById(R.id.permission_dialog_box);
        ComposeView loadingBox = findViewById(R.id.loading_box);
        LoadingManager loadingManager = new LoadingManager(loadingBox, false, this);
        loadingManager.setupLoadingBox();
        dialogManager = new PermissionDialogManager(dialogBox, false, false, this);
        dialogManagerSettings = new PermissionDialogManager(dialogBox, false, true, this);

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

    private void handleAllPermissions() {
        handleCameraPermission();
    }

    private void handleCameraPermission() {
        currentPermissionType = "camera";

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {

            if (permissionOption == 0) {
                handleStoragePermissions();
            } else {
                navigateToNextActivity();
            }
            return;
        }

        ActivityCompat.requestPermissions(
                this,
                new String[]{Manifest.permission.CAMERA},
                Constants.CAMERA_PERMISSION_REQUEST
        );
    }

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

                if (permissionOption == 0) {
                    handleStoragePermissions();
                } else {
                    navigateToNextActivity();
                }
            } else {
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

            new Handler(Looper.getMainLooper()).post(() -> {
                if (permType.equals("camera")) {
                    handleCameraPermission();
                } else {
                    handleStoragePermissions();
                }
            });
            return null;
        });

        new Handler(Looper.getMainLooper()).postDelayed(() -> dialogManager.showDialog(), 100);
    }

    private void showGoToSettingsDialog(String permType) {
        waitingForSettingsReturn = true;
        currentPermissionType = permType;

        dialogManagerSettings.setupDialog(() -> {
            dialogManagerSettings.hideDialog();

            new Handler(Looper.getMainLooper()).post(() -> {
                Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                intent.setData(android.net.Uri.parse("package:" + getPackageName()));
                settingsLauncher.launch(intent);
            });
            return null;
        });

        new Handler(Looper.getMainLooper()).postDelayed(() -> dialogManagerSettings.showDialog(), 100);
    }

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

    /**
     * Get storage permissions array with backward compatibility
     * API 33+ (Android 13+): READ_MEDIA_IMAGES
     * API 24-32: READ_EXTERNAL_STORAGE and WRITE_EXTERNAL_STORAGE
     */
    private String[] getStoragePermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) { // API 33+
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