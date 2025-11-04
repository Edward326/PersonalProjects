package com.visionassist.appspace.activities.newprofile.jsonCollection;

import android.content.Context;
import android.util.Log;

import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.FileUtils;
import com.visionassist.appspace.utils.Language;

import org.json.JSONObject;

import java.io.File;

public class ProfileFileCollection {
    private static final String TAG = "ProfileJsonManager";

    public static boolean configurationActivityWrite(boolean blindness) {
        try {
            Context context = PhoneStatusMonitor.getInstance().getCurrentContext();
            File profileFile = FileUtils.getProfileFile(context);

            JSONObject jsonObject;
            if (profileFile.exists() && profileFile.length() > 0) {
                String content = FileUtils.loadFileAsString(FileUtils.getProfileInputStream(context));
                jsonObject = new JSONObject(content);
            } else {
                jsonObject = new JSONObject();
            }

            jsonObject.put("blindness", blindness);
            boolean success = FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
            if (success)
                Log.d(TAG, "ConfigurationActivity: Fields written successfully");
            return success;
        } catch (Exception e) {
            Log.e(TAG, "ConfigurationActivity: Error writing fields", e);
            return false;
        }
    }

    public static boolean configurationActivityDelete() {
        try {
            Context context = PhoneStatusMonitor.getInstance().getCurrentContext();
            File profileFile = FileUtils.getProfileFile(context);

            if (!profileFile.exists() || profileFile.length() == 0) {
                Log.d(TAG, "ConfigurationActivity: Profile file doesn't exist, nothing to delete");
                return true;
            }

            String content = FileUtils.loadFileAsString(FileUtils.getProfileInputStream(context));
            JSONObject jsonObject = new JSONObject(content);

            if (jsonObject.has("blindness")) {
                jsonObject.remove("blindness");
                boolean success = FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
                if (success) {
                    Log.d(TAG, "ConfigurationActivity: Fields deleted successfully");
                }
            }
            return true;
        } catch (Exception e) {
            Log.e(TAG, "ConfigurationActivity: Error deleting fields", e);
            return false;
        }
    }

    public static boolean welcomeActivityWrite(boolean writeNewProfile, Language language, Boolean newProfileValue) {
        try {
            Context context = PhoneStatusMonitor.getInstance().getCurrentContext();
            File profileFile = FileUtils.getProfileFile(context);

            JSONObject jsonObject;
            if (profileFile.exists() && profileFile.length() > 0) {
                String content = FileUtils.loadFileAsString(FileUtils.getProfileInputStream(context));
                jsonObject = new JSONObject(content);
            } else {
                jsonObject = new JSONObject();
            }

            if (!writeNewProfile) {
                // Write language data
                if (language != null) {
                    jsonObject.put("language_code", language.getCode());
                    jsonObject.put("language_desc", language.getName());
                    jsonObject.put("language_country", language.getCountry());
                    Log.d(TAG, "WelcomeActivity: Fields written successfully");
                }
            } else {
                // Write new_profile data
                if (newProfileValue != null) {
                    jsonObject.put("new_profile", newProfileValue);
                    Log.d(TAG, "WelcomeActivity: Fields written successfully");
                }
            }
            return FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
        } catch (Exception e) {
            Log.e(TAG, "WelcomeActivity: Error writing fields", e);
            return false;
        }
    }

    public static boolean welcomeActivityDelete(boolean deleteAll) {
        try {
            Context context = PhoneStatusMonitor.getInstance().getCurrentContext();
            File profileFile = FileUtils.getProfileFile(context);

            if (!profileFile.exists() || profileFile.length() == 0) {
                Log.d(TAG, "Profile file doesn't exist, nothing to delete");
                return true;
            }

            String content = FileUtils.loadFileAsString(FileUtils.getProfileInputStream(context));
            JSONObject jsonObject = new JSONObject(content);

            if (!deleteAll) {
                // Delete only new_profile field
                // Delete language data and new_profile
                if (jsonObject.has("language_code")) {
                    jsonObject.remove("language_code");
                }
                if (jsonObject.has("language_desc")) {
                    jsonObject.remove("language_desc");
                }
                if (jsonObject.has("language_country")) {
                    jsonObject.remove("language_country");
                }
                Log.d(TAG, "WelcomeActivity: Fields deleted successfully");
            } else {
                // Delete language data and new_profile
                if (jsonObject.has("language_code")) {
                    jsonObject.remove("language_code");
                }
                if (jsonObject.has("language_desc")) {
                    jsonObject.remove("language_desc");
                }
                if (jsonObject.has("language_country")) {
                    jsonObject.remove("language_country");
                }
                if (jsonObject.has("new_profile")) {
                    jsonObject.remove("new_profile");
                }
                Log.d(TAG, "WelcomeActivity: Fields deleted successfully");
            }
            return FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
        } catch (Exception e) {
            Log.e(TAG, "WelcomeActivity: Error deleting fields", e);
            return false;
        }
    }

    public static boolean writeProfile(JSONObject jsonObject) {
        try {
            if (jsonObject == null) {
                Log.e(TAG, "Cannot write null JSONObject");
                return false;
            }

            return FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);

        } catch (Exception e) {
            Log.e(TAG, "Error writing profile", e);
            return false;
        }
    }

    public static boolean clearProfile() {
        try {
            JSONObject emptyJson = new JSONObject();
            return FileUtils.writeProfileFile(emptyJson.toString(), Constants.PROFILE_FILE_NAME);
        } catch (Exception e) {
            Log.e(TAG, "Error clearing profile", e);
            return false;
        }
    }
}