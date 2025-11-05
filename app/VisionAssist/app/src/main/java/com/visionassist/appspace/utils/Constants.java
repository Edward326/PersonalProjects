package com.visionassist.appspace.utils;

import android.graphics.Color;
import android.os.Build;

public class Constants {
    // Model Configuration
    public static final String YOLO_MODEL_FILE = "yolov8n_mobile.ptl";
    public static final String BLIP_MODEL_FILE = "blip_captioner_quantized.onnx";
    public static final String BLIP_MODEL_REGULAR = "blip_captioner.onnx";
    public static final String COCO_CLASSES_FILE = "detector_class_names.txt";

    // Add flag to prefer quantized models
    public static final boolean USE_QUANTIZED_MODELS = true;

    // Image Processing
    public static final int INPUT_WIDTH = 640;
    public static final int INPUT_HEIGHT = 640;

    // Detection Thresholds
    public static final float CONFIDENCE_THRESHOLD = 0.25f;
    public static final float NMS_THRESHOLD = 0.45f;

    // Drawing Configuration
    public static final int BBOX_COLOR = Color.GREEN;
    public static final float BBOX_STROKE_WIDTH = 16.0f;
    public static final int TEXT_COLOR = Color.WHITE;
    public static final int TEXT_BACKGROUND_COLOR = Color.BLACK;
    public static final float TEXT_SIZE = 24.0f;

    // Text-to-Speech Configuration
    public static final float TTS_SPEECH_RATE = 1.0f;
    public static final float TTS_PITCH = 1.0f;
    public static final float CAPTION_TEXT_SIZE = 25.0f;

    // Profile Storage Configuration - NEW
    public static final String PROFILE_FOLDER_NAME = "visionassist_data";
    public static final String PROFILE_FILE_NAME = "profile.json";
    public static final String HASH_CACHE_FILE_NAME = "hash_cache.txt";
    public static final String ENV_REPORTS_FILE_NAME = "env_reports.txt";

    // Intent Extras
    public static final String EXTRA_IMAGE_PATH = "extra_image_path";
    public static final String EXTRA_CAPTION_TEXT = "extra_caption_text";
    public static final String EXTRA_PERMISSION_OPTION = "permission_option";
    public static final String EXTRA_WELCOME_OPTION = "welcome_option";

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
    public static final int MAX_LANGUAGE_CHECK_ATTEMPTS = 3;

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

    // Error codes, followed by shutdown
    public static final int ERROR_READ_DELAY = 10000;
    public static final int EXCEPTION_CLASS_ERROR = -1;
    public static final int DIR_DELETE_ERROR = 0;
    public static final int JSON_PARSE_ERROR = 1;
    public static final int FILE_WRITE_ERROR = 2;
    public static final int DETECTOR_LOAD_ERROR = 3;
    public static final int CAPTIONER_LOAD_ERROR = 4;
    public static final int TRANSLATER_LOAD_ERROR = 5;
    public static final int CLASSIFIER_LOAD_ERROR = 6;
    public static final int STT_LOAD_ERROR = 7;
    public static final int ASSETS_ERROR = 8;

    // Error codes, used in load_profile_activity
    public static final int LOAD_PROFILE_EXCEPTION = -2;
    public static final int LOAD_PROFILE_SUCCESS = 9;
    public static final int LOAD_PROFILE_FILE_MISSING = 10;
    public static final int LOAD_PROFILE_FILE_STREAMOPEN = 11;
    public static final int LOAD_PROFILE_FILE_INVALID = 12;
    public static final int LOAD_PROFILE_FILE_UPLOAD = 13;
    public static final int LOAD_PROFILE_FILE_HC_UPLOAD_ERROR = 14;
    public static final int LOAD_PROFILE_FILE_ENVR_UPLOAD_ERROR = 15;

    // Error codes, used in new_profile_activity
    public static final int CREATE_PROFILE_EXCEPTION = -3;
    public static final int CREATE_PROFILE_SUCCESS = 16;

    // Success notification display time
    public static final int SUCCESS_NOTIFICATION_DELAY = 5000;

    // Intent extras for LoadProfileActivity
    public static final String EXTRA_LOAD_OPTION = "load_option";

    // Number of Models to load in main activity
    public static final int MODELS_COUNT = 5;
    public static final int MODELS_OWN_ASSETS_COUNT = 5;

    //Standard font size and button size
    public static final int STD_FONT_SIZE = 16;
    public static final int STD_FONT_SIZE_LW = STD_FONT_SIZE-2;
    public static final int STD_BUTTON_FONT_SIZE = STD_FONT_SIZE+5;
    public static final int STD_ERROR_FONT_SIZE = STD_FONT_SIZE+2;
    public static final int STD_BUTTON_HEIGHT = 75;

    // Debug Configuration
    public static final boolean DEBUG_MODE = true;
}