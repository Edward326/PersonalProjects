//app constants
package com.visionassist.appspace.utils;

import android.graphics.Color;

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
    public static final float TTS_SPEECH_RATE = 0.8f;
    public static final float TTS_PITCH = 1.0f;
    public static final float CAPTION_TEXT_SIZE = 25.0f;

    // File Handling
    public static final String PROFILE_FILE = "profiles/profile.json";
    public static final String TEMP_IMAGE_NAME = "temp_capture";

    // Intent Extras
    public static final String EXTRA_IMAGE_PATH = "extra_image_path";
    public static final String EXTRA_CAPTION_TEXT = "extra_caption_text";

    // Debug Configuration
    public static final boolean DEBUG_MODE = true; // Enable for detailed logging
}