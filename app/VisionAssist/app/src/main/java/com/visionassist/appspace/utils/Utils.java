package com.visionassist.appspace.utils;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Pair;
import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.R;
import com.visionassist.appspace.activities.newprofile.ConfigurationActivity;
import com.visionassist.appspace.activities.newprofile.EnvironmentReportsIActivity;
import com.visionassist.appspace.activities.newprofile.UserAccesibility1Activity;
import com.visionassist.appspace.activities.newprofile.UserAccesibility2Activity;
import com.visionassist.appspace.activities.newprofile.UserHashCachingActivity;
import com.visionassist.appspace.activities.newprofile.UserInfoActivity;
import com.visionassist.appspace.activities.newprofile.UserInfoE1Activity;
import com.visionassist.appspace.activities.newprofile.UserInfoE2Activity;
import com.visionassist.appspace.activities.newprofile.UserInfoE3Activity;
import com.visionassist.appspace.activities.newprofile.UserInfoE4Activity;
import com.visionassist.appspace.activities.newprofile.WelcomeActivity;
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.IOException;
import java.io.InputStream;

public class Utils {
    private static final String TAG = "Utils";

    public static Pair<Integer, JSONObject> checkProfile(Context context) {
        // Check if profile directory exists
        if (!FileUtils.profileDirectoryExists(context)) {
            Log.d(TAG, "Profile directory does not exist");
            return new Pair<>(1, null);
        }

        // Check if profile file exists
        if (!FileUtils.profileFileExists(context)) {
            Log.d(TAG, "Profile file does not exist");
            return new Pair<>(1, null);
        }

        // Read and validate profile file
        try {
            InputStream inputStream = FileUtils.getProfileInputStream(context);
            return JSONValidation.validateProfile(inputStream);
        } catch (IOException e) {
            Log.e(TAG, "Error reading profile file", e);
            return new Pair<>(1, null);
        }
    }

    public static void profileSelector(Activity activity, Context context, Pair<Integer, JSONObject> profileStatusDecider, LoadingManager loadingManager) {
        Intent intent;

        switch (profileStatusDecider.first) {

            case 1:
                // Delete existing profile directory if it exists
                if (FileUtils.profileDirectoryExists(context)) {
                    boolean deleted = FileUtils.deleteProfileDirectory(context);
                    Log.d(TAG, "Profile directory deletion: " + (deleted ? "success" : "failed"));
                }

                // Create fresh profile structure
                boolean created = FileUtils.createProfileStructure(context);
                if (!created) {
                    Log.e(TAG, "Failed to create profile structure");
                    // Still continue to ConfigurationActivity, let it handle the error
                }

                intent = new Intent(context, ConfigurationActivity.class);
                loadingManager.hideLoading();
                Intent finalIntent = intent;
                new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent), Constants.ANIMATION_DELAY);
                break;

            case -3:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    intent = new Intent(context, WelcomeActivity.class);
                    Intent finalIntent1 = intent;
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent1), Constants.ANIMATION_DELAY);
                    break;
                } catch (JSONException e) {
                    PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                    ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                    errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
                    phoneMonitor.shutdownApp(errorDialog);
                }

            case -2:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    // REMOVED: PermissionChecker.checkAndRequestPermissions call
                    intent = new Intent(context, WelcomeActivity.class);
                    Intent finalIntent2 = intent;
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent2), Constants.ANIMATION_DELAY);
                    break;
                } catch (JSONException e) {
                    PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                    ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                    errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
                    phoneMonitor.shutdownApp(errorDialog);
                }

            case -1:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, UserInfoActivity.class);
                    Intent finalIntent3 = intent;
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent3), Constants.ANIMATION_DELAY);
                    break;
                } catch (JSONException e) {
                    PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                    ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                    errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
                    phoneMonitor.shutdownApp(errorDialog);
                }

            case 2:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, UserInfoE1Activity.class);
                    Intent finalIntent4 = intent;
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent4), Constants.ANIMATION_DELAY);
                    break;
                } catch (JSONException e) {
                    PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                    ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                    errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
                    phoneMonitor.shutdownApp(errorDialog);
                }

            case 3:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, UserInfoE2Activity.class);
                    Intent finalIntent5 = intent;
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent5), Constants.ANIMATION_DELAY);
                    break;
                } catch (JSONException e) {
                    PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                    ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                    errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
                    phoneMonitor.shutdownApp(errorDialog);
                }

            case 4:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, UserInfoE3Activity.class);
                    Intent finalIntent6 = intent;
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent6), Constants.ANIMATION_DELAY);
                    break;
                } catch (JSONException e) {
                    PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                    ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                    errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
                    phoneMonitor.shutdownApp(errorDialog);
                }

            case 5:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, UserInfoE4Activity.class);
                    Intent finalIntent7 = intent;
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent7), Constants.ANIMATION_DELAY);
                    break;
                } catch (JSONException e) {
                    PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                    ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                    errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
                    phoneMonitor.shutdownApp(errorDialog);
                }

            case 6:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, UserAccesibility1Activity.class);
                    Intent finalIntent8 = intent;
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent8), Constants.ANIMATION_DELAY);
                    break;
                } catch (JSONException e) {
                    PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                    ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                    errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
                    phoneMonitor.shutdownApp(errorDialog);
                }

            case 7:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, UserAccesibility2Activity.class);
                    Intent finalIntent9 = intent;
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent9), Constants.ANIMATION_DELAY);
                    break;
                } catch (JSONException e) {
                    PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                    ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                    errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
                    phoneMonitor.shutdownApp(errorDialog);
                }

            case 8:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, UserHashCachingActivity.class);
                    Intent finalIntent10 = intent;
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent10), Constants.ANIMATION_DELAY);
                    break;
                } catch (JSONException e) {
                    PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                    ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                    errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
                    phoneMonitor.shutdownApp(errorDialog);
                }

            case 9:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, EnvironmentReportsIActivity.class);
                    Intent finalIntent11 = intent;
                    loadingManager.hideLoading();
                    new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent11), Constants.ANIMATION_DELAY);
                    break;
                } catch (JSONException e) {
                    PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
                    ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
                    errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
                    phoneMonitor.shutdownApp(errorDialog);
                }

            default:
                intent = new Intent(context, ConfigurationActivity.class);
                loadingManager.hideLoading();
                Intent finalIntent12 = intent;
                new Handler(Looper.getMainLooper()).postDelayed(() -> context.startActivity(finalIntent12), Constants.ANIMATION_DELAY);
                break;
        }
    }

    public static void uploadProfile(Activity activity, JSONObject profileSource) {
        try {
            AppConfig.blindness = profileSource.getBoolean("blindness");
            AppConfig.mainLanguage = languageExtractor(profileSource);
            AppConfig.user_name = profileSource.getString("user_name");
            AppConfig.isContributor = profileSource.getBoolean("contributor");
            if (AppConfig.isContributor) {
                AppConfig.age = profileSource.getInt("age");
                if (!AppConfig.blindness)
                    AppConfig.visual_condition = profileSource.getString("visual_condition");
                else {
                    AppConfig.tts_pitch = (float) profileSource.getDouble("tts_pitch");
                    AppConfig.tts_speech_rate = (float) profileSource.getDouble("tts_speed");
                }
                AppConfig.email = profileSource.getString("email");
            }
            if (!AppConfig.blindness) {
                AppConfig.bbox_color = profileSource.getString("bbox_color");
                AppConfig.label_color = profileSource.getString("label_color");
                AppConfig.label_bck_color = profileSource.getString("label_bck_color");
                AppConfig.isBold = profileSource.getBoolean("bold");
                AppConfig.show_confidence = profileSource.getBoolean("show_confidence");
                AppConfig.caption_color = profileSource.getString("caption_color");
                AppConfig.caption_bck_color = profileSource.getString("caption_bck_color");
                AppConfig.haptics = profileSource.getBoolean("haptics");
            }
            AppConfig.hash_caching = profileSource.getString("hash_caching");
            AppConfig.env_reports = profileSource.getBoolean("env_reports");
        } catch (JSONException e) {
            PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
            ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
            errorDialog.setupDialog(Constants.JSON_PARSE_ERROR, String.valueOf(R.string.exit_error_en));
            phoneMonitor.shutdownApp(errorDialog);
        }
    }

    public static Language languageExtractor(JSONObject profileStatusDecider) throws JSONException {
        return new Language(
                profileStatusDecider.getString("language_code"),
                profileStatusDecider.getString("language_desc"),
                profileStatusDecider.getString("language_country"));
    }

    public static Pair<Integer, Integer> checkPhoneStatus(Context context) {
        int battery = -1, temperature = -1;
        if (Constants.APPLY_BATTERY_CHECK)
            battery = checkIfBatteryIsLow(context);

        if (Constants.APPLY_TEMPERATURE_CHECK)
            temperature = checkPhoneTemperature(context);

        return new Pair<>(battery, temperature);
    }

    public static int checkPhoneTemperature(Context context) {
        try {
            IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
            Intent batteryStatus = context.registerReceiver(null, ifilter);

            if (batteryStatus != null) {
                int temperature = batteryStatus.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1);

                if (temperature != -1) {
                    float tempCelsius = temperature / 10.0f;
                    Log.d(TAG, "Phone temperature: " + tempCelsius + "°C");
                    return (tempCelsius >= Constants.MAX_PHONE_TEMPERATURE) ? 1 : 0;
                }
            }

            Log.e(TAG, "Could not retrieve phone temperature.");
            return 0;

        } catch (Exception e) {
            Log.e(TAG, "Error checking phone temperature", e);
            return 0;
        }
    }

    public static int checkIfBatteryIsLow(Context context) {
        try {
            IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
            Intent batteryStatus = context.registerReceiver(null, ifilter);

            if (batteryStatus != null) {
                int batteryPct = returnBatteryLevel(context);
                if (batteryPct == -1)
                    return 0;
                Log.d(TAG, "Battery level: " + batteryPct);
                boolean isLow = batteryPct <= Constants.MIN_BATTERY_LEVEL;
                return (isLow) ? 1 : 0;
            }

            Log.e(TAG, "Battery status intent is null.");
            return 0;

        } catch (Exception e) {
            Log.e(TAG, "Error checking battery low status", e);
            return 0;
        }
    }

    public static int returnBatteryLevel(Context context) {
        try {
            IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
            Intent batteryStatus = context.registerReceiver(null, ifilter);

            if (batteryStatus != null) {
                int level = batteryStatus.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
                int scale = batteryStatus.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
                return (int) ((level / (float) scale) * 100);
            }

            Log.e(TAG, "Battery status intent is null.");
            return -1;

        } catch (Exception e) {
            Log.e(TAG, "Error retrieving battery level", e);
            return -1;
        }
    }
}