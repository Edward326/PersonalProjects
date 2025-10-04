//JSON file validation methods, verify if the file is json, its format(if the app format is respected) and for a full valid profile.
package com.visionassist.appspace.utils;

import android.util.Log;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.IOException;
import java.io.InputStream;

public class JSONValidation {
    private static final String TAG = "JSONValidation";

    public static int validateProfile(InputStream inputStream,boolean profileExistance) throws IOException {
        String profileContent;

        // Read the InputStream into a String
        try {
            profileContent = FileUtils.loadFileAsString(inputStream);

        } catch (Exception e) {
            Log.d(TAG, "Could not read contents of inputStream");
            return 1; // Error reading file
        }

        // Check if content is empty
        if (profileContent == null || profileContent.trim().isEmpty()) {
            Log.d(TAG, "File is empty or string[] is null");
            return 1; // Empty file
        }

        // Parse JSON
        JSONObject jsonObject;
        try {
            jsonObject = new JSONObject(profileContent);
        } catch (JSONException e) {
            Log.d(TAG, "Invalid JSON format of file");
            return 1; // Invalid JSON format
        }

        if(jsonObject.has("blindness") && jsonObject.length()==1)
            return -3;//go to WelcomeActivity
        else {
            if (!jsonObject.has("blindness"))
                return 1;
        }

        boolean isBlindProfile = false;
        try {
            isBlindProfile = jsonObject.getBoolean("blindness");
        } catch (JSONException e) {
            Log.d(TAG, "Could not parse blindness value");
            return 1;
        }

        if(jsonObject.has("user_language") && jsonObject.length()<=2){
            if(jsonObject.has("new_profile"))
                return -1;//go to UserInfo1Activity
            else
                return -2;//go to LoadingActivity
        }

        if(jsonObject.has("user_name")) {
            try {
                if (jsonObject.getBoolean("contributor") == true) {
                    if (!jsonObject.has("age")) {
                        return 2;
                    }
                    if (!jsonObject.has("visual_condition") && !isBlindProfile) {
                        return 3;
                    }
                    if (!jsonObject.has("email")) {
                        return 4;
                    }
                    if (isBlindProfile) {
                        if (!jsonObject.has("tts_pitch"))
                            return 5;
                    }
                }
            }catch (JSONException e) {
                return 1;
            }
        }
        else
            return 1;

        if(!isBlindProfile) {
            if (!jsonObject.has("bbox_color"))
                return 6;
            if(!jsonObject.has("caption_color"))
                return 7;
        }

        if (!jsonObject.has("hash_caching"))
            return 8;

        if (!jsonObject.has("env_reports") && !isBlindProfile) {
            return 9;
        }

        inputStream.close();

        // All validations passed
        return 0;
    }
}