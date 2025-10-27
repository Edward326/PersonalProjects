//JSON file validation methods, verify if the file is json, its format(if the app format is respected) and for a full valid profile.
package com.visionassist.appspace.utils;

import android.util.Log;
import android.util.Pair;

import com.visionassist.appspace.PhoneStatusMonitor;

import org.json.JSONException;
import org.json.JSONObject;
import java.io.IOException;
import java.io.InputStream;

public class JSONValidation {
    private static final String TAG = "JSONValidation";

    public static Pair<Integer, JSONObject> validateProfile(InputStream inputStream) throws IOException, JSONException {
        String profileContent;

        // Read the InputStream into a String
        try {
            profileContent = FileUtils.loadFileAsString(inputStream);

        } catch (Exception e) {
            Log.d(TAG, "Could not read contents of inputStream");
            return new Pair<>(1, null); // Error reading file
        }

        // Check if content is empty
        if (profileContent == null || profileContent.trim().isEmpty()) {
            Log.d(TAG, "File is empty or string[] is null");
            return new Pair<>(1, null); // Empty file
        }

        // Parse JSON
        JSONObject jsonObject;
        try {
            jsonObject = new JSONObject(profileContent);
        } catch (JSONException e) {
            Log.d(TAG, "Invalid JSON format of file");
            return new Pair<>(1, null); // Invalid JSON format
        }

        boolean isBlindProfile;
        if (!jsonObject.has("blindness"))
            return new Pair<>(1, null); // ConfigurationActivity
        else {
            try {
                isBlindProfile = jsonObject.getBoolean("blindness");
            } catch (JSONException e) {
                Log.d(TAG, "Could not parse blindness value");
                return new Pair<>(1, null); // Error parsing blindness value
            }
        }

        if (!jsonObject.has("language_code")) {
            if (!jsonObject.has("language_desc")
                    && !jsonObject.has("language_country"))
                return new Pair<>(-3, jsonObject);   // WelcomeActivity
            else if (jsonObject.has("language_desc")
                || jsonObject.has("language_country"))
                return new Pair<>(1, jsonObject);   // ConfigurationActivity

        }
        else{
            if (!jsonObject.has("language_desc")
                    || !jsonObject.has("language_country"))
                return new Pair<>(1, jsonObject);   // ConfigurationActivity
        }

        if (!jsonObject.has("new_profile"))
            return new Pair<>(-2, jsonObject);   // pmerissions+LoadingActivity

        if (!jsonObject.has("user_name"))
            return new Pair<>(-1, jsonObject);   // UserInfoActivity

        boolean isContributor;
        if (!jsonObject.has("contributor"))
            return new Pair<>(1, jsonObject);   // ConfigurationActivity
        else {
            try {
                isContributor = jsonObject.getBoolean("contributor");
            } catch (JSONException e) {
                Log.d(TAG, "Could not parse contributor value");
                return new Pair<>(1, null); // Error parsing contributor value
            }
        }

        if (isContributor) {
            if (!jsonObject.has("age")) {
                return new Pair<>(2, jsonObject);   //UserInfoE1Activity
            }
            if (!jsonObject.has("visual_condition") && !isBlindProfile) {
                return new Pair<>(3, jsonObject);   //UserInfoE2Activity
            }
            if (!jsonObject.has("email")) {
                return new Pair<>(4, jsonObject);   //UserInfoE3Activity
            }
            if (isBlindProfile) {
                if (!jsonObject.has("tts_pitch"))
                    if (jsonObject.has("tts_speed"))
                        return new Pair<>(1, jsonObject);   // ConfigurationActivity
                    else
                        return new Pair<>(5, jsonObject);   //UserInfoE4Activity
                if (!jsonObject.has("tts_speed"))
                    return new Pair<>(1, jsonObject);   // ConfigurationActivity

                if (!jsonObject.has("hash_caching"))
                    return new Pair<>(8, jsonObject);   // UserHashCachingActivity

                if (!jsonObject.has("env_reports"))
                    return new Pair<>(9, jsonObject);   // EnvironmentReportsIActivity

                if(jsonObject.getBoolean("hash_caching"))
                    if(!FileUtils.getHashCacheFile(PhoneStatusMonitor.getInstance().getCurrentContext()).exists())
                        return new Pair<>(0, jsonObject);
                if(jsonObject.getBoolean("env_reports"))
                    if(!FileUtils.getEnvReportsFile(PhoneStatusMonitor.getInstance().getCurrentContext()).exists())
                        return new Pair<>(0, jsonObject);

                inputStream.close();
                // All validations passed
                return new Pair<>(0, jsonObject);
            }
        } else {
            if (isBlindProfile) {
                if (!jsonObject.has("tts_pitch"))
                    if (jsonObject.has("tts_speed"))
                        return new Pair<>(1, jsonObject);   // ConfigurationActivity
                    else
                        return new Pair<>(5, jsonObject);   //UserInfoE4Activity
                if (!jsonObject.has("tts_speed"))
                    return new Pair<>(1, jsonObject);   // ConfigurationActivity

                if (!jsonObject.has("hash_caching"))
                    return new Pair<>(8, jsonObject);   // UserHashCachingActivity

                if (!jsonObject.has("env_reports"))
                    return new Pair<>(9, jsonObject);   // EnvironmentReportsIActivity

                if(jsonObject.getBoolean("hash_caching"))
                    if(!FileUtils.getHashCacheFile(PhoneStatusMonitor.getInstance().getCurrentContext()).exists())
                        return new Pair<>(0, jsonObject);
                if(jsonObject.getBoolean("env_reports"))
                    if(!FileUtils.getEnvReportsFile(PhoneStatusMonitor.getInstance().getCurrentContext()).exists())
                        return new Pair<>(0, jsonObject);

                inputStream.close();
                // All validations passed
                return new Pair<>(0, jsonObject);
            }
        }

        if (!jsonObject.has("bbox_color")) {
            if (!jsonObject.has("label_color")
                    && !jsonObject.has("label_bck_color"))
                return new Pair<>(6, jsonObject);    // UserAccesibility1Activity
            else if (jsonObject.has("label_color")
                    || jsonObject.has("label_bck_color"))
                return new Pair<>(1, jsonObject);   // ConfigurationActivity
        }
        else{
            if (!jsonObject.has("label_color")
                    || !jsonObject.has("label_bck_color"))
                return new Pair<>(1, jsonObject);   // ConfigurationActivity
        }

        if (!jsonObject.has("caption_color"))
            if (jsonObject.has("caption_bck_color"))
                return new Pair<>(1, jsonObject);   // ConfigurationActivity
            else
                return new Pair<>(7, jsonObject);   // UserAccesibility2Activity
        if (!jsonObject.has("caption_bck_color"))
            return new Pair<>(1, jsonObject);   // ConfigurationActivity

        if (!jsonObject.has("hash_caching"))
            return new Pair<>(8, jsonObject);   // UserHashCachingActivity

        if (!jsonObject.has("env_reports"))
            return new Pair<>(9, jsonObject);   // EnvironmentReportsIActivity

        if(jsonObject.getBoolean("hash_caching"))
            if(!FileUtils.getHashCacheFile(PhoneStatusMonitor.getInstance().getCurrentContext()).exists())
                return new Pair<>(0, jsonObject);
        if(jsonObject.getBoolean("env_reports"))
            if(!FileUtils.getEnvReportsFile(PhoneStatusMonitor.getInstance().getCurrentContext()).exists())
                return new Pair<>(0, jsonObject);

        inputStream.close();
        // All validations passed
        return new Pair<>(0, jsonObject);
    }
}