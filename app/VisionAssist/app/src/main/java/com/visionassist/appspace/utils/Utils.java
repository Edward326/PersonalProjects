package com.visionassist.appspace.utils;

import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.util.Log;
import android.util.Pair;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import org.json.JSONObject;
import java.io.IOException;
import java.io.InputStream;

public class Utils {
    private static final String TAG = "Utils";

    public static void profileSelector(Pair<Integer, JSONObject> profileStatusDecider, LoadingManager loadingManager){
        //select profile based on opt
    }

    public static void uploadProfile(JSONObject profileSource){
        //upload profile to AppConfig
    }

    public static Pair<Integer, JSONObject> checkProfile(Context context) {
        if(!FileUtils.assetExists(context, Constants.PROFILE_FILE))
            return new Pair<>(1, null);
        try {
            InputStream inputStream = context.getAssets().open(Constants.PROFILE_FILE);
            return JSONValidation.validateProfile(inputStream);
        }catch (IOException e) {
            return new Pair<>(1, null);
        }
    }

    public static Pair<Integer, Integer> checkPhoneStatus(Context context) {
        // Check battery level
        int battery=-1, temperature=-1;
        if(Constants.APPLY_BATTERY_CHECK)
            battery=checkIfBatteryIsLow(context);

        // Check phone temperature
        if(Constants.APPLY_TEMPERATURE_CHECK)
            temperature= checkPhoneTemperature(context);

        return new Pair<>(battery, temperature);
    }

    public static int checkPhoneTemperature(Context context) {
        try {
            IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
            Intent batteryStatus = context.registerReceiver(null, ifilter);

            if (batteryStatus != null) {
                // Temperature is in tenths of degrees Celsius
                int temperature = batteryStatus.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1);

                if (temperature != -1) {
                    // Convert to degrees Celsius
                    float tempCelsius = temperature / 10.0f;
                    Log.d(TAG, "Phone temperature: " + tempCelsius + "°C");
                    // Check against threshold from Constants
                    return (tempCelsius >= Constants.MAX_PHONE_TEMPERATURE) ? 1 : 0;
                }
            }

            // If we can't read temperature, assume it's OK
            Log.e(TAG, "Could not retrieve phone temperature.");
            return 0;

        } catch (Exception e) {
            Log.e(TAG, "Error checking phone temperature", e);
            return 0; // Default to OK if error occurs
        }
    }

    public static int checkIfBatteryIsLow(Context context) {
        try {
            // ACTION_BATTERY_CHANGED is a sticky broadcast, we get the last one immediately.
            IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
            Intent batteryStatus = context.registerReceiver(null, ifilter);

            if (batteryStatus != null) {
                // EXTRA_BATTERY_LOW is a boolean: true if the system considers the battery to be low.
                boolean isLow;
                //if(Constants.API_LEVEL>= 28) {
                //    isLow = batteryStatus.getBooleanExtra(BatteryManager.EXTRA_BATTERY_LOW, false);
                //}
                //else {
                    int batteryPct = returnBatteryLevel(context);
                    if(batteryPct==-1)
                        return 0; // Unable to retrieve battery level, assume not low)
                    Log.d(TAG, "Battery level: " + batteryPct);
                    isLow = batteryPct <= Constants.MIN_BATTERY_LEVEL;
                //}

                return (isLow)? 1 : 0;
            }

            Log.e(TAG, "Battery status intent is null.");
            // Default to not low if status cannot be retrieved
            return 0;

        } catch (Exception e) {
            Log.e(TAG, "Error checking battery low status", e);
            return 0; // Default to OK if error occurs
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
            return -1; // Unable to retrieve battery level

        } catch (Exception e) {
            Log.e(TAG, "Error retrieving battery level", e);
            return -1; // Error occurred
        }
    }
}