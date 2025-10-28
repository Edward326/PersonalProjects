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
import com.visionassist.appspace.R;
import com.visionassist.appspace.jetpack.managers.PermissionDialogManager;
import com.visionassist.appspace.utils.Constants;

public class PermissionsActivity extends AppCompatActivity {
    private static final String TAG = "PermissionsActivity";

    private int permissionOption;
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

        setContentView(R.layout.activity_permissions);
        TextView titleView = findViewById(R.id.permissions_text);
        ComposeView dialogBox = findViewById(R.id.permission_dialog_box);
        titleView.setVisibility(View.VISIBLE);
        dialogManager = new PermissionDialogManager(dialogBox, false, this);
        dialogManagerSettings = new PermissionDialogManager(dialogBox, true, this);

        Intent intent = getIntent();
        permissionOption = intent.getIntExtra(Constants.EXTRA_PERMISSION_OPTION, 0);
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
            case 2: // Microphone permission missing
                handleMicrophonePermission();
                break;
            case 3: // Storage permissions missing
                handleStoragePermissions();
                break;
            default:
                finish();
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
                handleMicrophonePermission();
            } else {
                finish();
            }
            return;
        }

        ActivityCompat.requestPermissions(
                this,
                new String[]{Manifest.permission.CAMERA},
                Constants.CAMERA_PERMISSION_REQUEST
        );
    }

    private void handleMicrophonePermission() {
        currentPermissionType = "microphone";

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED) {

            if (permissionOption == 0) {
                handleStoragePermissions();
            } else {
                finish();
            }
            return;
        }

        ActivityCompat.requestPermissions(
                this,
                new String[]{Manifest.permission.RECORD_AUDIO},
                Constants.MICROPHONE_PERMISSION_REQUEST
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
            finish();
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
                    handleMicrophonePermission();
                } else {
                    finish();
                }
            } else {
                if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
                    showPermissionDeniedDialog("camera");
                } else {
                    showGoToSettingsDialog("camera");
                }
            }
        } else if (requestCode == Constants.MICROPHONE_PERMISSION_REQUEST) {
            if (allGranted) {
                Log.d(TAG, "Microphone permission granted");

                if (permissionOption == 0) {
                    handleStoragePermissions();
                } else {
                    finish();
                }
            } else {
                if (shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO)) {
                    showPermissionDeniedDialog("microphone");
                } else {
                    showGoToSettingsDialog("microphone");
                }
            }
        } else if (requestCode == Constants.STORAGE_PERMISSION_REQUEST) {
            if (allGranted) {
                Log.d(TAG, "Storage permissions granted");
                finish();
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
                } else if (permType.equals("microphone")) {
                    handleMicrophonePermission();
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
        switch (currentPermissionType) {
            case "camera":
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        == PackageManager.PERMISSION_GRANTED) {
                    if (permissionOption == 0) {
                        handleMicrophonePermission();
                    } else {
                        finish();
                    }
                } else {
                    showGoToSettingsDialog("camera");
                }
                break;
            case "microphone":
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                        == PackageManager.PERMISSION_GRANTED) {
                    if (permissionOption == 0) {
                        handleStoragePermissions();
                    } else {
                        finish();
                    }
                } else {
                    showGoToSettingsDialog("microphone");
                }
                break;
            case "storage":
                String[] storagePerms = getStoragePermissions();
                boolean allGranted = true;

                for (String perm : storagePerms) {
                    if (ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) {
                        allGranted = false;
                        break;
                    }
                }

                if (allGranted) {
                    finish();
                } else {
                    showGoToSettingsDialog("storage");
                }
                break;
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
}