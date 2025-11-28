//user profile uploaded from profiles to be further used in app
package com.visionassist.appspace.utils;

public class AppConfig {

    public static boolean blindness;
    public static Language mainLanguage=new Language("en", "English","US");
    public static String user_name;
    public static boolean isContributor;
    public static int age;
    public static String visual_condition;
    public static float tts_speech_rate=Constants.TTS_SPEECH_RATE;
    public static float tts_pitch=Constants.TTS_PITCH;
    public static String bbox_color="#00FF00";
    public static String label_color="#FFFFFF";
    public static String label_bck_color="#000000";
    public static boolean isBold=true;
    public static boolean show_confidence=true;
    public static String caption_color;
    public static String caption_bck_color;
    public static boolean haptics;
    public static String hash_caching;
    public static boolean env_reports;
    public static boolean showTutorial=true;

    public static String listAppConfig() {
        return "blindness: " + blindness + "\n" +
                "mainLanguage: " + (mainLanguage != null ? mainLanguage.getName() + " (" + mainLanguage.getCode() + ")" : "null") + "\n" +
                "user_name: " + (user_name != null ? user_name : "null") + "\n" +
                "isContributor: " + isContributor + "\n" +
                "tts_speech_rate: " + tts_speech_rate + "\n" +
                "tts_pitch: " + tts_pitch + "\n" +
                "bbox_color: " + (bbox_color != null ? bbox_color : "null") + "\n" +
                "label_color: " + (label_color != null ? label_color : "null") + "\n" +
                "label_bck_color: " + (label_bck_color != null ? label_bck_color : "null") + "\n" +
                "isBold: " + isBold + "\n" +
                "show_confidence: " + show_confidence + "\n" +
                "caption_color: " + (caption_color != null ? caption_color : "null") + "\n" +
                "caption_bck_color: " + (caption_bck_color != null ? caption_bck_color : "null") + "\n" +
                "haptics: " + haptics + "\n" +
                "hash_caching: " + (hash_caching != null ? hash_caching : "null") + "\n" +
                "env_reports: " + env_reports + "\n" +
                "================";
    }
}