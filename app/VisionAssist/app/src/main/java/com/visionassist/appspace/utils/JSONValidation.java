//JSON file validation methods, verify if the file is json, its format(if the app format is respected) and for a full valid profile.
package com.visionassist.appspace.utils;

import android.util.Log;
import android.util.Pair;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.IOException;
import java.io.InputStream;

public class JSONValidation {
    private static final String TAG = "JSONValidation";

    public static Pair<Integer, JSONObject> validateProfile(InputStream inputStream) throws IOException {
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

        if(jsonObject.has("blindness") && jsonObject.length()==1)
            return new Pair<>(-3,jsonObject);//go to WelcomeActivity
        else {
            if (!jsonObject.has("blindness"))
                return new Pair<>(1, null); // Missing blindness key
        }

        boolean isBlindProfile;
        try {
            isBlindProfile = jsonObject.getBoolean("blindness");
        } catch (JSONException e) {
            Log.d(TAG, "Could not parse blindness value");
            return new Pair<>(1, null); // Error parsing blindness value
        }

        if(jsonObject.has("user_language") && jsonObject.length()<=2){
            if(jsonObject.has("new_profile"))
                return new Pair<>(-1,jsonObject);//go to UserInfo1Activity
            else
                return new Pair<>(-2,jsonObject);//go to LoadingActivity
        }

        if(jsonObject.has("user_name")) {
            try {
                if (jsonObject.getBoolean("contributor")) {
                    if (!jsonObject.has("age")) {
                        return new Pair<>(2, jsonObject);
                    }
                    if (!jsonObject.has("visual_condition") && !isBlindProfile) {
                        return new Pair<>(3, jsonObject);
                    }
                    if (!jsonObject.has("email")) {
                        return new Pair<>(4, jsonObject);
                    }
                    if (isBlindProfile) {
                        if (!jsonObject.has("tts_pitch"))
                            return new Pair<>(5, jsonObject);
                    }
                }
            }catch (JSONException e) {
                return new Pair<>(1, null); // Error parsing contributor value
            }
        }
        else
            return new Pair<>(1, null); // Invalid profile

        if(!isBlindProfile) {
            if (!jsonObject.has("bbox_color"))
                return new Pair<>(6, jsonObject);
            if(!jsonObject.has("caption_color"))
                return new Pair<>(7, jsonObject);
        }

        if (!jsonObject.has("hash_caching"))
            return new Pair<>(8, jsonObject);

        if (!jsonObject.has("env_reports") && !isBlindProfile) {
            return new Pair<>(9, jsonObject);
        }

        inputStream.close();

        // All validations passed
        return new Pair<>(0, jsonObject);
    }
}