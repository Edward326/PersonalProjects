//permission checker class to check if the app has the required permissions
package com.visionassist.appspace.utils;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import androidx.core.content.ContextCompat;
import com.visionassist.appspace.activities.newprofile.PermissionsActivity;

public class PermissionChecker {

    public static void checkAndRequestPermissions(Activity activity, Class<?> nextActivityClass) {
        boolean cameraGranted = false;
        boolean storageGranted = false;

        // Check camera permission
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            cameraGranted = true;
        }

        // Check storage permissions based on Android version
        if (Constants.API_LEVEL >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ uses READ_MEDIA_IMAGES instead of READ_EXTERNAL_STORAGE
            if (ContextCompat.checkSelfPermission(activity, Manifest.permission.READ_MEDIA_IMAGES)
                    == PackageManager.PERMISSION_GRANTED) {
                storageGranted = true;
            }
        } else {
            // Android 12 and below
            if (ContextCompat.checkSelfPermission(activity, Manifest.permission.READ_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_GRANTED &&
                    ContextCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                            == PackageManager.PERMISSION_GRANTED) {
                storageGranted = true;
            }
        }

        // Determine permission status option
        int permissionOption;
        if (!cameraGranted && !storageGranted) {
            permissionOption = 0; // All permissions not granted
        } else if (!cameraGranted) {
            permissionOption = 1; // Camera permission not granted
        } else if (!storageGranted) {
            permissionOption = 2; // File permissions not granted
        } else {
            // All permissions granted
            return;
        }

        // Navigate to PermissionsActivity with the option and next activity
        Intent intent = new Intent(activity, PermissionsActivity.class);
        intent.putExtra(Constants.EXTRA_PERMISSION_OPTION, permissionOption);
        intent.putExtra(Constants.EXTRA_NEXT_ACTIVITY, nextActivityClass.getName());
        activity.startActivity(intent);
    }
}