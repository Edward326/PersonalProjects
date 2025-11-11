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

    public static boolean writeConfigurationActivity(boolean blindness) {
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

    public static boolean deleteConfigurationActivity() {
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

    public static boolean writeWelcomeActivity(boolean writeNewProfile, Language language, Boolean newProfileValue) {
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

    public static boolean deleteWelcomeActivity(boolean deleteNewProfile) {
        try {
            Context context = PhoneStatusMonitor.getInstance().getCurrentContext();
            File profileFile = FileUtils.getProfileFile(context);

            if (!profileFile.exists() || profileFile.length() == 0) {
                Log.d(TAG, "Profile file doesn't exist, nothing to delete");
                return true;
            }

            String content = FileUtils.loadFileAsString(FileUtils.getProfileInputStream(context));
            JSONObject jsonObject = new JSONObject(content);

            if (!deleteNewProfile) {
                // Delete only language data
                if (jsonObject.has("language_code")) {
                    jsonObject.remove("language_code");
                }
                if (jsonObject.has("language_desc")) {
                    jsonObject.remove("language_desc");
                }
                if (jsonObject.has("language_country")) {
                    jsonObject.remove("language_country");
                }
                Log.d(TAG, "WelcomeActivity: Language fields deleted successfully");
            } else {
                // Delete new_profile field
                if (jsonObject.has("new_profile")) {
                    jsonObject.remove("new_profile");
                }
                Log.d(TAG, "WelcomeActivity: new_profile field deleted successfully");
            }
            return FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
        } catch (Exception e) {
            Log.e(TAG, "WelcomeActivity: Error deleting fields", e);
            return false;
        }
    }

    public static boolean writeNewProfileActivity(boolean isRemote, String email, String passwordHash) {
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

            // Write remote field
            jsonObject.put("remote", isRemote);

            if (isRemote) {
                // Write email and password_hash for remote profiles
                if (email != null && passwordHash != null) {
                    jsonObject.put("email", email);
                    jsonObject.put("password_hash", passwordHash);
                    Log.d(TAG, "NewProfileActivity: Remote profile fields written successfully");
                }
            } else {
                Log.d(TAG, "NewProfileActivity: Local profile field written successfully");
            }

            return FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
        } catch (Exception e) {
            Log.e(TAG, "NewProfileActivity: Error writing fields", e);
            return false;
        }
    }

    public static boolean writeUserInfoActivity(int opt, boolean isContributor, String name, int age, String problem) {
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

            switch (opt) {
                case 0: // Write name and contributor
                    jsonObject.put("user_name", name);
                    jsonObject.put("contributor", isContributor);
                    Log.d(TAG, "UserInfoActivity: Name and contributor fields written successfully");
                    break;
                case 1: // Write age
                    jsonObject.put("age", age);
                    Log.d(TAG, "UserInfoActivity: Age field written successfully");
                    break;
                case 2: // Write visual condition
                    jsonObject.put("visual_condition", problem);
                    Log.d(TAG, "UserInfoActivity: Visual condition field written successfully");
                    break;
                default:
                    Log.e(TAG, "UserInfoActivity: Invalid option: " + opt);
                    return false;
            }

            return FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
        } catch (Exception e) {
            Log.e(TAG, "UserInfoActivity: Error writing fields", e);
            return false;
        }
    }

    public static boolean deleteUserInfoActivity(int opt) {
        try {
            Context context = PhoneStatusMonitor.getInstance().getCurrentContext();
            File profileFile = FileUtils.getProfileFile(context);

            if (!profileFile.exists() || profileFile.length() == 0) {
                Log.d(TAG, "UserInfoActivity: Profile file doesn't exist, nothing to delete");
                return true;
            }

            String content = FileUtils.loadFileAsString(FileUtils.getProfileInputStream(context));
            JSONObject jsonObject = new JSONObject(content);

            switch (opt) {
                case 0: // Delete name and contributor
                    if (jsonObject.has("user_name")) {
                        jsonObject.remove("user_name");
                    }
                    if (jsonObject.has("contributor")) {
                        jsonObject.remove("contributor");
                    }
                    Log.d(TAG, "UserInfoActivity: Name and contributor fields deleted successfully");
                    break;
                case 1: // Delete age
                    if (jsonObject.has("age")) {
                        jsonObject.remove("age");
                    }
                    Log.d(TAG, "UserInfoActivity: Age field deleted successfully");
                    break;
                case 2: // Delete visual condition
                    if (jsonObject.has("visual_condition")) {
                        jsonObject.remove("visual_condition");
                    }
                    Log.d(TAG, "UserInfoActivity: Visual condition field deleted successfully");
                    break;
                default:
                    Log.e(TAG, "UserInfoActivity: Invalid option: " + opt);
                    return false;
            }

            return FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
        } catch (Exception e) {
            Log.e(TAG, "UserInfoActivity: Error deleting fields", e);
            return false;
        }
    }

    public static boolean writeUserInfoE3Activity(float pitch, float speed) {
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

            jsonObject.put("tts_pitch", pitch);
            jsonObject.put("tts_speech_rate", speed);
            boolean success = FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
            if (success)
                Log.d(TAG, "ConfigurationActivity: Fields written successfully");
            return success;
        } catch (Exception e) {
            Log.e(TAG, "ConfigurationActivity: Error writing fields", e);
            return false;
        }
    }

    public static boolean deleteUserInfoE3Activity() {
        try {
            Context context = PhoneStatusMonitor.getInstance().getCurrentContext();
            File profileFile = FileUtils.getProfileFile(context);

            if (!profileFile.exists() || profileFile.length() == 0) {
                Log.d(TAG, "ConfigurationActivity: Profile file doesn't exist, nothing to delete");
                return true;
            }

            String content = FileUtils.loadFileAsString(FileUtils.getProfileInputStream(context));
            JSONObject jsonObject = new JSONObject(content);

            if (jsonObject.has("tts_pitch") && jsonObject.has("tts_speech_rate")) {
                jsonObject.remove("tts_pitch");
                jsonObject.remove("tts_speech_rate");
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

    public static boolean writeUserAccessibility1BoundingBox(
            String bboxColor,
            String labelColor,
            String labelBckColor,
            boolean bold,
            boolean showConfidence
    ) {
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

            jsonObject.put("bbox_color", bboxColor);
            jsonObject.put("label_color", labelColor);
            jsonObject.put("label_bck_color", labelBckColor);
            jsonObject.put("bold", bold);
            jsonObject.put("show_confidence", showConfidence);

            boolean success = FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
            if (success)
                Log.d(TAG, "UserAccessibility1BoundingBox: Fields written successfully");
            return success;
        } catch (Exception e) {
            Log.e(TAG, "UserAccessibility1BoundingBox: Error writing fields", e);
            return false;
        }
    }

    public static boolean writeUserAccessibility1Caption(
            String captionColor,
            String captionBckColor,
            boolean haptics
    ) {
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

            jsonObject.put("caption_color", captionColor);
            jsonObject.put("caption_bck_color", captionBckColor);
            jsonObject.put("haptics", haptics);

            boolean success = FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
            if (success)
                Log.d(TAG, "UserAccessibility1Caption: Fields written successfully");
            return success;
        } catch (Exception e) {
            Log.e(TAG, "UserAccessibility1Caption: Error writing fields", e);
            return false;
        }
    }

    public static boolean deleteUserAccessibility1Activity(boolean deleteCaption) {
        try {
            Context context = PhoneStatusMonitor.getInstance().getCurrentContext();
            File profileFile = FileUtils.getProfileFile(context);

            if (!profileFile.exists() || profileFile.length() == 0) {
                Log.d(TAG, "UserAccessibility1: Profile file doesn't exist, nothing to delete");
                return true;
            }

            String content = FileUtils.loadFileAsString(FileUtils.getProfileInputStream(context));
            JSONObject jsonObject = new JSONObject(content);

            if (!deleteCaption) {
                // Delete BoundingBox fields
                if (jsonObject.has("bbox_color")) {
                    jsonObject.remove("bbox_color");
                }
                if (jsonObject.has("label_color")) {
                    jsonObject.remove("label_color");
                }
                if (jsonObject.has("label_bck_color")) {
                    jsonObject.remove("label_bck_color");
                }
                if (jsonObject.has("bold")) {
                    jsonObject.remove("bold");
                }
                if (jsonObject.has("show_confidence")) {
                    jsonObject.remove("show_confidence");
                }
                Log.d(TAG, "UserAccessibility1: BoundingBox fields deleted successfully");
            } else {
                // Delete Caption fields
                if (jsonObject.has("caption_color")) {
                    jsonObject.remove("caption_color");
                }
                if (jsonObject.has("caption_bck_color")) {
                    jsonObject.remove("caption_bck_color");
                }
                if (jsonObject.has("haptics")) {
                    jsonObject.remove("haptics");
                }
                Log.d(TAG, "UserAccessibility1: Caption fields deleted successfully");
            }

            return FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
        } catch (Exception e) {
            Log.e(TAG, "UserAccessibility1: Error deleting fields", e);
            return false;
        }
    }

    public static boolean writeUserHashCachingActivity(String value, boolean isEnvReports) {
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

            // Add the appropriate field
            if (isEnvReports) {
                jsonObject.put("env_reports", Boolean.parseBoolean(value));
            } else {
                jsonObject.put("hash_caching", value);
            }

            // Write back to file
            boolean success = FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
            if (success)
                Log.d(TAG, "UserHashCachingActivity: Fields written successfully");
            return success;
        } catch (Exception e) {
            Log.e(TAG, "UserHashCachingActivity: Error writing fields", e);
            return false;
        }
    }

    public static boolean deleteUserHashCachingActivity() {
        try {
            Context context = PhoneStatusMonitor.getInstance().getCurrentContext();
            File profileFile = FileUtils.getProfileFile(context);

            if (!profileFile.exists() || profileFile.length() == 0) {
                Log.d(TAG, "UserHashCachingActivity: Profile file doesn't exist, nothing to delete");
                return true;
            }

            String content = FileUtils.loadFileAsString(FileUtils.getProfileInputStream(context));
            JSONObject jsonObject = new JSONObject(content);

            if (jsonObject.has("hash_caching")) {
                jsonObject.remove("hash_caching");
                boolean success = FileUtils.writeProfileFile(jsonObject.toString(), Constants.PROFILE_FILE_NAME);
                if (success) {
                    Log.d(TAG, "UserHashCachingActivity: Fields deleted successfully");
                }
            }
            return true;
        } catch (Exception e) {
            Log.e(TAG, "UserHashCachingActivity: Error deleting fields", e);
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