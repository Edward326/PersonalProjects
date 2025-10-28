//app constants
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

    // Battery and Temperature Checks
    public static final boolean APPLY_BATTERY_CHECK = true;
    public static final int MIN_BATTERY_LEVEL = 15;
    public static final boolean APPLY_TEMPERATURE_CHECK = true;
    public static final int MAX_PHONE_TEMPERATURE = 40;

    // Monitoring settings
    public static final long WAIT_CHECK = 5000;

    // API Level of the device
    public static final int API_LEVEL = Build.VERSION.SDK_INT;

    // Shutdown delay after showing error (in milliseconds)
    public static final int SHUTDOWN_DELAY_MS = 10000;
    public static final int RETRY_TTS_DELAY_MS = 1000;
    public static final int MAX_LANGUAGE_CHECK_ATTEMPTS = 3;

    // TTS Repeat Configuration
    public static final long REPEAT_DELAY = 3000; // 5 seconds wait for volume button
    public static final float LOW_SPEECH_RATE = 0.5f; // Lower pitch for repeat
    public static final float HIGH_SPEECH_RATE = 1.5f; // Lower pitch for repeat

    // Permission Request Codes
    public static final int CAMERA_PERMISSION_REQUEST = 101;
    public static final int STORAGE_PERMISSION_REQUEST = 102;
    public static final int MICROPHONE_PERMISSION_REQUEST = 101;

    // Time to wait before popping the permission intent
    public static final int PERMISSION_SLEEP = 1000;


    // UI overlay for notification
    public static final int ANIMATION_DELAY = 1500;

    // Error read delay
    public static final int ERROR_READ_DELAY = 10000;

    // Error codes
    public static final int EXCEPTION_CLASS_ERROR = 0;
    public static final int DIR_DELETE_ERROR = 1;
    public static final int JSON_PARSE_ERROR = 1;
    public static final int FILE_WRITE_ERROR = 2;
    public static final int DETECTOR_LOAD_ERROR = 3;
    public static final int CAPTIONER_LOAD_ERROR = 4;
    public static final int TRANSLATER_LOAD_ERROR = 5;
    public static final int CLASSIFIER_LOAD_ERROR = 6;
    public static final int STT_LOAD_ERROR = 7;
    public static final int ASSETS_ERROR = 8;

    // Number of Models to load in main activity
    public static final int MODELS_COUNT=5;
    public static final int MODELS_OWN_ASSETS_COUNT=5;

    // Debug Configuration
    public static final boolean DEBUG_MODE = true;
}