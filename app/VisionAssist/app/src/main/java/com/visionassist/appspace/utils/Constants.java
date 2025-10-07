//app constants
package com.visionassist.appspace.utils;

import android.graphics.Color;
import android.os.Build;

public class Constants {
    // Model Configuration
    public static final String YOLO_MODEL_FILE = "yolov8n_mobile.ptl";
    public static final String BLIP_MODEL_FILE = "blip_captioner_quantized.onnx"; // CHANGED THIS
    public static final String BLIP_MODEL_REGULAR = "blip_captioner.onnx"; // Fallback
    public static final String COCO_CLASSES_FILE = "detector_class_names.txt";

    // Add flag to prefer quantized models
    public static final boolean USE_QUANTIZED_MODELS = true;

    // Image Processing
    public static final int INPUT_WIDTH = 640;
    public static final int INPUT_HEIGHT = 640;

    // Detection Thresholds - ADJUSTED for better detection
    public static final float CONFIDENCE_THRESHOLD = 0.25f; // Lowered from typical 0.5 for more detections
    public static final float NMS_THRESHOLD = 0.45f; // Standard NMS threshold

    // Drawing Configuration - IMPROVED for better visibility
    public static final int BBOX_COLOR = Color.GREEN;
    public static final float BBOX_STROKE_WIDTH = 16.0f; // Increased thickness
    public static final int TEXT_COLOR = Color.WHITE;
    public static final int TEXT_BACKGROUND_COLOR = Color.BLACK;
    public static final float TEXT_SIZE = 24.0f; // Increased text size

    // Text-to-Speech Configuration
    public static final float TTS_SPEECH_RATE = 1.0f;
    public static final float TTS_PITCH = 1.0f;
    public static final float CAPTION_TEXT_SIZE = 25.0f;

    // File Handling
    public static final String PROFILE_FILE = "profiles/profile.json";

    // Intent Extras
    public static final String EXTRA_IMAGE_PATH = "extra_image_path";
    public static final String EXTRA_CAPTION_TEXT = "extra_caption_text";
    public static final String EXTRA_PERMISSION_OPTION = "permission_option";
    public static final String EXTRA_NEXT_ACTIVITY = "next_activity";

    // Battery and Temperature Checks
    public static final boolean APPLY_BATTERY_CHECK = true;
    public static final int MIN_BATTERY_LEVEL = 15; // Minimum battery level percentage
    public static final boolean APPLY_TEMPERATURE_CHECK = true;
    public static final int MAX_PHONE_TEMPERATURE = 40; // Maximum temperature in Celsius

    // Monitoring settings
    public static final long WAIT_CHECK = 5000; // Check every 5 seconds (in milliseconds)

    // API Level of the device
    public static final int API_LEVEL = Build.VERSION.SDK_INT;

    // Shutdown delay after showing error (in milliseconds)
    public static final int SHUTDOWN_DELAY_MS = 10000;
    public static final int RETRY_TTS_DELAY_MS = 1000;
    public static final long BLINDNESS_SHUTDOWN_DELAY_MS =10000;

    // Permission Request Codes
    public static final int CAMERA_PERMISSION_REQUEST = 101;
    public static final int STORAGE_PERMISSION_REQUEST = 102;

    // Time to wait before popping the permission intent
    public static final int PERMISSION_SLEEP=2000;

    // Language Check Attempts
    public static final int MAX_LANGUAGE_CHECK_ATTEMPTS = 3;

    // Debug Configuration
    public static final boolean DEBUG_MODE = true; // Enable for detailed logging
}