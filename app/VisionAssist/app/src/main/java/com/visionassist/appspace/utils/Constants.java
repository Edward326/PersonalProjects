package com.visionassist.appspace.utils;

import android.graphics.Color;
import android.os.Build;

public class Constants {
    // Model Configuration
    public static final String YOLO_MODEL_DETECTOR_FILE = "yolov8s_mobile.onnx";
    public static final String DETECTOR_CLASSES_FILE_EN = "detector_class_names_en.txt";
    public static final String DETECTOR_CLASSES_FILE_RO = "detector_class_names_ro.txt";
    public static final String YOLO_MODEL_CLASSIFIER_FILE = "yolov8m_cls_mobile.onnx";
    public static final String CLASSIFIER_CLASSES_FILE_EN = "classifier_class_names_en.txt";
    public static final String CLASSIFIER_CLASSES_FILE_RO = "classifier_class_names_ro.txt";
    public static final String BLIP_MODEL_FILE = "blip_captioner_quantized.onnx";
    public static final String VOSK_MODEL_DIR = "vosk-model-en-us-0.22-lgraph";
    public static final String USELESS_WORDS_FILE = "useless_words.txt";
    public static final String OBJECT_SYNONYMS_FILE = "object_synonyms.json";

    // Image Processing
    public static final int DETECTOR_INPUT_SIZE = 640;
    public static final int CLASSIFIER_INPUT_SIZE = 224;

    // Detection Thresholds
    public static final float CONFIDENCE_THRESHOLD = 0.25f;
    public static final float NMS_THRESHOLD = 0.45f;

    // Drawing Configuration
    public static final float BBOX_STROKE_WIDTH_SCREEN = 0.02f;
    public static final float TEXT_SIZE_WIDTH_SCREEN = 0.06f;

    // Text-to-Speech Configuration
    public static final float TTS_SPEECH_RATE = 1.0f;
    public static final float TTS_PITCH = 1.0f;

    // Profile Storage Configuration - NEW
    public static final String PROFILE_FOLDER_NAME = "visionassist_data";
    public static final String PROFILE_FILE_NAME = "profile.json";
    public static final String HASH_CACHE_FILE_NAME = "hash_cache.txt";
    public static final String ENV_REPORTS_FILE_NAME = "env_reports.txt";

    // Intent Extras
    public static final String EXTRA_PERMISSION_OPTION = "permission_option";
    public static final String EXTRA_WELCOME_OPTION = "welcome_option";
    public static final String EXTRA_USERINFO_OPTION = "userinfo_option";
    public static final String EXTRA_USERACC_OPTION = "useracc_option";
    public static final String EXTRA_HCACHING_OPTION = "hcaching_option";
    public static final String EXTRA_IMAGE_URI = "image_uri";
    public static final String EXTRA_MATCHED_INDICES = "class_indices";
    public static final String EXTRA_SYNONYMS_WORDS = "synonyms_words";

    // Battery and Temperature Checks
    public static final boolean APPLY_BATTERY_CHECK = true;
    public static final int MIN_BATTERY_LEVEL = 15;
    public static final boolean APPLY_TEMPERATURE_CHECK = true;
    public static final int MAX_PHONE_TEMPERATURE = 45;

    // Monitoring settings
    public static final long WAIT_CHECK = 5000;

    // API Level of the device
    public static final int API_LEVEL = Build.VERSION.SDK_INT;

    // Shutdown delay after showing error (in milliseconds)
    public static final int SHUTDOWN_DELAY_MS = 10000;
    public static final int RETRY_TTS_DELAY_MS = 1000;
    public static final int LOAD_CHECK_DELAY_MS = 1000;
    public static final int MAX_LANGUAGE_CHECK_ATTEMPTS = 3;
    public static final int VOLUME_DOWN_DELAY_MS = 1000;

    // TTS Repeat Configuration
    public static final long REPEAT_DELAY = 2000; // 2 seconds wait for volume button
    public static final float LOW_SPEECH_RATE = 0.5f; // Lower pitch for repeat
    public static final float HIGH_SPEECH_RATE = 1.5f; // Lower pitch for repeat

    // Permission Request Codes
    public static final int CAMERA_PERMISSION_REQUEST = 101;
    public static final int STORAGE_PERMISSION_REQUEST = 102;
    public static final int MICROPHONE_PERMISSION_REQUEST = 101;

    // Time to wait before popping the permission intent
    public static final int PERMISSION_SLEEP = 1000;


    // UI overlay for notification
    public static final int ANIMATION_DELAY = 500;
    public static final float BACKGROUND_OPACITY = 0.5f;

    // Error codes, followed by shutdown
    public static final int REGISTER_CAMERA_LAUNCHER=-3;
    public static final int CAMERA_MAKE_PHOTO=-4;
    public static final int ERROR_READ_DELAY = 8000;
    public static final int EXCEPTION_CLASS_ERROR = -1;
    public static final int DIR_DELETE_ERROR = 0;
    public static final int JSON_PARSE_ERROR = 1;
    public static final int FILE_WRITE_ERROR = 2;
    public static final int DETECTOR_LOAD_ERROR = 3;
    public static final int CAPTIONER_LOAD_ERROR = 4;

    // Error codes, used in load_profile_activity
    public static final int LOAD_PROFILE_EXCEPTION = -2;
    public static final int LOAD_PROFILE_SUCCESS = 9;
    public static final int LOAD_PROFILE_FILE_MISSING = 10;
    public static final int LOAD_PROFILE_FILE_STREAMOPEN = 11;
    public static final int LOAD_PROFILE_FILE_INVALID = 12;
    public static final int LOAD_PROFILE_FILE_UPLOAD = 13;
    public static final int LOAD_PROFILE_FILE_HC_UPLOAD_ERROR = 14;
    public static final int LOAD_PROFILE_FILE_ENVR_UPLOAD_ERROR = 15;

    // Success notification display time
    public static final int SUCCESS_NOTIFICATION_DELAY = 5000;

    // Number of Models to load in main activity
    public static final int MODELS_COUNT = 5;
    public static final int MODELS_OWN_ASSETS_COUNT = 5;

    // Standard font size and button size
    public static final int LOGO_SIZE = 200;
    public static final int STD_TITLE_SIZE = 40;
    public static final int STD_SUBTITLE_SIZE = 32;

    public static final int STD_FONT_SIZE = 14;
    public static final int STD_FONT_SIZE_LW = STD_FONT_SIZE - 2;
    public static final int STD_BUTTON_FONT_SIZE = STD_FONT_SIZE + 2;
    public static final int STD_ERROR_FONT_SIZE = STD_FONT_SIZE + 4;
    public static final int STD_SLIDER_INFO_SIZE = 23;
    public static final int STD_BUTTON_HEIGHT = 58;
    public static final int STD_BUTTON_PAGE_HEIGHT = 86;
    public static final int STD_INFO_BUTTON_SIZE = 35;
    public static final int STD_LOGINCARD_WIDTH = 265;
    public static final int NAV_BUTTONS_WIDTH = 110;
    public static final int NAV_BUTTONS_HEIGHT = 70;

    // Standard screen weighters
    public static final float STD_TITLE_MARGIN_TOP=0.372f;
    public static final float STD_SUBTITLE_MARGIN_TOP=0.51f;
    public static final float STD_SUBTITLE2_MARGIN_TOP=0.48f;
    public static final float STD_TITLE_SUBTITLE_MARGIN_TOP=0.045f;
    public static final float STD_SUBTITLE_BODY_MARGIN_TOP=0.04f;
    public static final float STD_NAV_MARGIN_BOTTOM=0.10f;

    // CameraX activities parameters
    public static final int DEFAULT_FPS = 10;
    public static final float BATTERY_USAGE_THRESHOLD = 0.7f;
    public static final long BATTERY_WARNING_DISPLAY_MS = 10000;

    // UserAccessibility1Activity params
    public static final int PREVIEW_UPDATE_DELAY = 1000;
}