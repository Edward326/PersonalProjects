package com.visionassist.appspace.utils;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Pair;

import com.visionassist.appspace.ExceptionVisionAssist;
import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.activities.newprofile.ConfigurationActivity;
import com.visionassist.appspace.activities.newprofile.LoadProfileActivity;
import com.visionassist.appspace.activities.newprofile.NewProfileActivity;
import com.visionassist.appspace.activities.newprofile.UserAccessibility1Activity;
import com.visionassist.appspace.activities.newprofile.UserHashCachingActivity;
import com.visionassist.appspace.activities.newprofile.UserInfoActivity;
import com.visionassist.appspace.activities.newprofile.UserInfoE3Activity;
import com.visionassist.appspace.activities.newprofile.WelcomeActivity;
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection;
import com.visionassist.appspace.database.DBConstants;
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager;
import com.visionassist.appspace.jetpack.managers.LoadingManager;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.InputStream;

public class Utils {
    private static final String TAG = "Utils";

    @SuppressLint("StaticFieldLeak")
    static final PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
    @SuppressLint("StaticFieldLeak")
    static Activity activity = phoneMonitor.getCurrentActivity();
    @SuppressLint("StaticFieldLeak")
    static Context context = phoneMonitor.getCurrentContext();

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
        } catch (Exception e) {
            Log.e(TAG, "Error reading profile file", e);
            return new Pair<>(1, null);
        }
    }

    public static  void waitForTTSAndNavigate(Pair<Integer, JSONObject> profileStatusDecider, LoadingManager loadingManager) throws Exception{
        Handler ttsHandler = new Handler(Looper.getMainLooper());
        Runnable checkTTS = new Runnable() {
            @Override
            public void run() {
                if (PhoneStatusMonitor.getInstance().getTTSManager().isReady()) {
                    Log.d(TAG, "TTS was set to: "+PhoneStatusMonitor.getInstance().getTTSManager().getCurrentLocale().getLanguage());
                    try {
                        profileSelectorMain(profileStatusDecider, loadingManager);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                } else {
                    Log.w(TAG, "TTS not ready, retrying...");
                    ttsHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS);
                }
            }
        };
        ttsHandler.post(checkTTS);
    }

    public static void profileSelector(Pair<Integer, JSONObject> profileStatusDecider, LoadingManager loadingManager) throws Exception {
        if (profileStatusDecider.second!=null &&
                profileStatusDecider.second.has("blindness")
                && profileStatusDecider.second.has("language_code")
                && profileStatusDecider.second.has("language_desc")
                && profileStatusDecider.second.has("language_country"))
        {
                PhoneStatusMonitor.getInstance().getTTSManager().changeLanguage(languageExtractor(profileStatusDecider.second), PhoneStatusMonitor.getInstance().getCurrentActivity());
                waitForTTSAndNavigate(profileStatusDecider,loadingManager);
        } else
            profileSelectorMain(profileStatusDecider, loadingManager);
    }

    public static void profileSelectorMain(Pair<Integer, JSONObject> profileStatusDecider, LoadingManager loadingManager) throws Exception {
        Intent intent;
        Context context = phoneMonitor.getCurrentContext();

        switch (profileStatusDecider.first) {
            case 1:
                if(FileUtils.getProfileFile(context).exists()){
                    String content5 = "Last data of the profile.json:\n"+FileUtils.loadFileAsString(FileUtils.getProfileInputStream(context));
                    Log.i(TAG, content5);
                }

                // Delete existing profile directory if it exists
                if (FileUtils.profileDirectoryExists(context)) {
                    boolean deleted = FileUtils.deleteProfileDirectory(context);
                    Log.d(TAG, "Profile directory deletion: " + (deleted ? "success" : "failed"));
                }

                // Create fresh profile structure
                boolean created = FileUtils.createProfileDirFile(Constants.PROFILE_FILE_NAME);
                if (!created) {
                    Log.e(TAG, "Failed to create profile structure");
                    // Still continue to ConfigurationActivity, let it handle the error
                }
                intent = new Intent(context, ConfigurationActivity.class);
                Intent finalIntent = intent;
                loadingManager.hideLoading();
                context.startActivity(finalIntent);
                activity.finish();
                break;

            case -4:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    FileUtils.deleteProfileDirFile(DBConstants.PROFILE_COPY_FILE);
                    FileUtils.deleteProfileDirFile(Constants.HASH_CACHE_FILE_NAME);
                    FileUtils.deleteProfileDirFile(Constants.ENV_REPORTS_FILE_NAME);
                    intent = new Intent(context, LoadProfileActivity.class);
                    Intent finalIntent1 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent1);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            case -3:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, NewProfileActivity.class);
                    Intent finalIntent1 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent1);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            case -2:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, WelcomeActivity.class);
                    intent.putExtra(Constants.EXTRA_WELCOME_OPTION, true);
                    Intent finalIntent2 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent2);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            case -1:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    intent = new Intent(context, WelcomeActivity.class);
                    intent.putExtra(Constants.EXTRA_WELCOME_OPTION, false);
                    Intent finalIntent3 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent3);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            case 2:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    intent = new Intent(context, UserInfoActivity.class);
                    intent.putExtra(Constants.EXTRA_USERINFO_OPTION, 1);
                    Intent finalIntent4 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent4);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            case 3:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    AppConfig.user_name = profileStatusDecider.second.getString("user_name");
                    AppConfig.isContributor = profileStatusDecider.second.getBoolean("contributor");
                    intent = new Intent(context, UserInfoActivity.class);
                    intent.putExtra(Constants.EXTRA_USERINFO_OPTION, 2);
                    Intent finalIntent5 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent5);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            case 4:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    AppConfig.user_name = profileStatusDecider.second.getString("user_name");
                    AppConfig.isContributor = profileStatusDecider.second.getBoolean("contributor");
                    AppConfig.age = profileStatusDecider.second.getInt("age");
                    intent = new Intent(context, UserInfoActivity.class);
                    intent.putExtra(Constants.EXTRA_USERINFO_OPTION, 3);
                    Intent finalIntent6 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent6);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            case 5:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    AppConfig.user_name = profileStatusDecider.second.getString("user_name");
                    AppConfig.isContributor = profileStatusDecider.second.getBoolean("contributor");
                    if (AppConfig.isContributor)
                        AppConfig.age = profileStatusDecider.second.getInt("age");

                    intent = new Intent(context, UserInfoE3Activity.class);
                    Intent finalIntent7 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent7);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            case 6:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    AppConfig.user_name = profileStatusDecider.second.getString("user_name");
                    AppConfig.isContributor = profileStatusDecider.second.getBoolean("contributor");
                    if (AppConfig.isContributor) {
                        AppConfig.age = profileStatusDecider.second.getInt("age");
                        AppConfig.visual_condition = profileStatusDecider.second.getString("visual_condition");
                    }

                    intent = new Intent(context, UserAccessibility1Activity.class);
                    intent.putExtra(Constants.EXTRA_USERACC_OPTION, 1);
                    Intent finalIntent8 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent8);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            case 7:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    AppConfig.user_name = profileStatusDecider.second.getString("user_name");
                    AppConfig.isContributor = profileStatusDecider.second.getBoolean("contributor");
                    if (AppConfig.isContributor) {
                        AppConfig.age = profileStatusDecider.second.getInt("age");
                        AppConfig.visual_condition = profileStatusDecider.second.getString("visual_condition");
                    }
                    AppConfig.bbox_color = profileStatusDecider.second.getString("bbox_color");
                    AppConfig.label_color = profileStatusDecider.second.getString("label_color");
                    AppConfig.label_bck_color = profileStatusDecider.second.getString("label_bck_color");
                    AppConfig.isBold = profileStatusDecider.second.getBoolean("bold");
                    AppConfig.show_confidence = profileStatusDecider.second.getBoolean("show_confidence");

                    intent = new Intent(context, UserAccessibility1Activity.class);
                    intent.putExtra(Constants.EXTRA_USERACC_OPTION, 2);
                    Intent finalIntent9 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent9);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            case 8:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    AppConfig.user_name = profileStatusDecider.second.getString("user_name");
                    AppConfig.isContributor = profileStatusDecider.second.getBoolean("contributor");
                    if (AppConfig.isContributor) {
                        AppConfig.age = profileStatusDecider.second.getInt("age");
                        if (!AppConfig.blindness)
                            AppConfig.visual_condition = profileStatusDecider.second.getString("visual_condition");
                    }
                    if (AppConfig.blindness) {
                        AppConfig.tts_speech_rate = (float) profileStatusDecider.second.getDouble("tts_speech_rate");
                        AppConfig.tts_pitch = (float) profileStatusDecider.second.getDouble("tts_pitch");
                    } else {
                        AppConfig.bbox_color = profileStatusDecider.second.getString("bbox_color");
                        AppConfig.label_color = profileStatusDecider.second.getString("label_color");
                        AppConfig.label_bck_color = profileStatusDecider.second.getString("label_bck_color");
                        AppConfig.isBold = profileStatusDecider.second.getBoolean("bold");
                        AppConfig.show_confidence = profileStatusDecider.second.getBoolean("show_confidence");
                        AppConfig.caption_color = profileStatusDecider.second.getString("caption_color");
                        AppConfig.caption_bck_color = profileStatusDecider.second.getString("caption_bck_color");
                        AppConfig.haptics = profileStatusDecider.second.getBoolean("haptics");
                    }

                    intent = new Intent(context, UserHashCachingActivity.class);
                    intent.putExtra(Constants.EXTRA_HCACHING_OPTION, 1);
                    Intent finalIntent10 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent10);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            case 9:
                try {
                    AppConfig.blindness = profileStatusDecider.second.getBoolean("blindness");
                    AppConfig.mainLanguage = languageExtractor(profileStatusDecider.second);
                    AppConfig.user_name = profileStatusDecider.second.getString("user_name");
                    AppConfig.isContributor = profileStatusDecider.second.getBoolean("contributor");
                    if (AppConfig.isContributor) {
                        AppConfig.age = profileStatusDecider.second.getInt("age");
                        AppConfig.visual_condition = profileStatusDecider.second.getString("visual_condition");
                    }
                    AppConfig.bbox_color = profileStatusDecider.second.getString("bbox_color");
                    AppConfig.label_color = profileStatusDecider.second.getString("label_color");
                    AppConfig.label_bck_color = profileStatusDecider.second.getString("label_bck_color");
                    AppConfig.isBold = profileStatusDecider.second.getBoolean("bold");
                    AppConfig.show_confidence = profileStatusDecider.second.getBoolean("show_confidence");
                    AppConfig.caption_color = profileStatusDecider.second.getString("caption_color");
                    AppConfig.caption_bck_color = profileStatusDecider.second.getString("caption_bck_color");
                    AppConfig.haptics = profileStatusDecider.second.getBoolean("haptics");
                    AppConfig.hash_caching = profileStatusDecider.second.getString("hash_caching");

                    intent = new Intent(context, UserHashCachingActivity.class);
                    intent.putExtra(Constants.EXTRA_HCACHING_OPTION, 2);
                    Intent finalIntent11 = intent;
                    loadingManager.hideLoading();
                    context.startActivity(finalIntent11);
                    activity.finish();
                    break;
                } catch (JSONException e) {
                    throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
                }

            default:
                intent = new Intent(context, ConfigurationActivity.class);
                loadingManager.hideLoading();
                Intent finalIntent12 = intent;
                context.startActivity(finalIntent12);
                activity.finish();
                break;
        }
    }

    public static void uploadProfile(JSONObject profileSource, Runnable afterUploadProfile) {
        try {
            AppConfig.blindness = profileSource.getBoolean("blindness");
            AppConfig.mainLanguage = languageExtractor(profileSource);
            AppConfig.user_name = profileSource.getString("user_name");

            if (!AppConfig.blindness) {
                AppConfig.bbox_color = profileSource.getString("bbox_color");
                AppConfig.label_color = profileSource.getString("label_color");
                AppConfig.label_bck_color = profileSource.getString("label_bck_color");
                AppConfig.isBold = profileSource.getBoolean("bold");
                AppConfig.show_confidence = profileSource.getBoolean("show_confidence");
                AppConfig.caption_color = profileSource.getString("caption_color");
                AppConfig.caption_bck_color = profileSource.getString("caption_bck_color");
                AppConfig.haptics = profileSource.getBoolean("haptics");
                AppConfig.SoA=profileSource.getBoolean("soa");
            } else {
                AppConfig.tts_speech_rate = (float) profileSource.getDouble("tts_speech_rate");
                AppConfig.tts_pitch = (float) profileSource.getDouble("tts_pitch");
            }
            AppConfig.hash_caching = profileSource.getString("hash_caching");
            if (!AppConfig.blindness)
                AppConfig.env_reports = profileSource.getBoolean("env_reports");

            int firstTimeCounts=profileSource.getInt("init");
            AppConfig.showTutorial=firstTimeCounts<=2;
            if(AppConfig.showTutorial) {
                profileSource.put("init",++firstTimeCounts);
                ProfileFileCollection.writeProfile(profileSource);
            }

            if (afterUploadProfile != null)
                afterUploadProfile.run();
        } catch (Exception e) {
            Log.e(TAG, "Thrown exception, explanation: ", e);
            ErrorDialogManager errorDialog = new ErrorDialogManager(activity);
            errorDialog.setupDialog(Constants.JSON_PARSE_ERROR);
            phoneMonitor.shutdownApp(errorDialog, context);
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