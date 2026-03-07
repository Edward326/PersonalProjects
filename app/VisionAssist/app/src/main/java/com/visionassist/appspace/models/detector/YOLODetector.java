package com.visionassist.appspace.models.detector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.DisplayMetrics;
import android.util.Log;
import com.visionassist.appspace.utils.*;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class YOLODetector {
    private static final String TAG = "YOLODetector";

    private OrtEnvironment ortEnvironment;
    private OrtSession ortSession;
    private Map<Integer, String> classNames;
    private Context context;
    public String type;

    // Model parameters
    private static final int NUM_DETECTIONS = 8400;
    private static final int NUM_FEATURES = 111; // 4 bbox + 107 classes

    // Preprocessing parameters (same as PyTorch version)
    private float scaleX = 1.0f;
    private float scaleY = 1.0f;
    private int offsetX = 0;
    private int offsetY = 0;

    public YOLODetector(Context context) {
        this.context = context;
    }

    public int loadModel(String model_filepath,String type) {
        try {
            Log.d(TAG, "Loading Detector Model YOLO ...");

            // Create ONNX Runtime environment
            ortEnvironment = OrtEnvironment.getEnvironment();

            // Load model from assets
            String modelPath = FileUtils.assetFilePath(context, model_filepath);

            // Create session options
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

            // Load session
            assert ortEnvironment != null;
            ortSession = ortEnvironment.createSession(modelPath, options);

            Log.d(TAG, "Model loaded successfully, model selected:"+model_filepath);
            this.type=type;

            loadClassNames();
            Log.d(TAG, "Detector class names loaded: " + classNames.size() + " classes");

            return 0;
        } catch (Exception e) {
            Log.e(TAG, "Failed to load Detector Model YOLO, model selected:"+model_filepath, e);
            return -1;
        }
    }

    public void loadClassNames() throws IOException {
        try {
            String classNameFile = AppConfig.mainLanguage.getCode().equals("en") ? Constants.DETECTOR_CLASSES_FILE_EN : Constants.DETECTOR_CLASSES_FILE_RO;
            classNames = FileUtils.loadClassNames(context, classNameFile);
        } catch (Exception e) {
            Log.e(TAG, "Failed to load class names", e);
            throw new IOException("Failed to load class names: " + e.getMessage());
        }
    }

    public DetectionResult detectObjects(Bitmap bitmap,String threadInfo) {
        try {
            Log.d(TAG,"["+threadInfo+"]"+"DETECTION STARTED, MODEL:"+type);
            Log.d(TAG, "["+threadInfo+"]"+"Starting detection on image: " + bitmap.getWidth() + "x" + bitmap.getHeight());

            // Step 1: Preprocess image
            long startPreprocess = System.currentTimeMillis();
            float[] inputArray = preprocessImage(bitmap,threadInfo);
            long preprocessTime = System.currentTimeMillis() - startPreprocess;

            if (inputArray == null) {
                Log.e(TAG, "["+threadInfo+"]"+"Failed to preprocess image");
                return new DetectionResult(new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),new ArrayList<>());
            }

            // Step 2: Run inference
            long startInference = System.currentTimeMillis();
            float[] output = runInference(inputArray,threadInfo);
            long inferenceTime = System.currentTimeMillis() - startInference;


            if (output == null) {
                Log.e(TAG, "["+threadInfo+"]"+"Inference failed");
                return new DetectionResult(new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),new ArrayList<>());
            }

            // Step 3: Post-process results
            long startPostprocess = System.currentTimeMillis();
            DetectionResult result = postprocessOutput(output, bitmap.getWidth(), bitmap.getHeight(),threadInfo);
            long postprocessTime = System.currentTimeMillis() - startPostprocess;

            Log.d(TAG, "["+threadInfo+"]"+"Preprocessing completed in " + preprocessTime + "ms");
            Log.d(TAG, "["+threadInfo+"]"+"Inference completed in " + inferenceTime + "ms");
            Log.d(TAG, "["+threadInfo+"]"+"Post-processing completed in " + postprocessTime + "ms");
            Log.d(TAG, "["+threadInfo+"]"+"Total time for inference: " + (preprocessTime + inferenceTime + postprocessTime) + "ms");
            Log.d(TAG, "["+threadInfo+"]"+"Detection completed:\nObjects found(" + result.getDetectionCount() + ")\n" + result.listBoundingBoxes());

            return result;
        } catch (Exception e) {
            Log.e(TAG, "["+threadInfo+"]"+"Error during object detection", e);
            return new DetectionResult(new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),new ArrayList<>());
        }
    }

    private float[] preprocessImage(Bitmap bitmap,String threadInfo) {
        try {
            // Resize bitmap with padding (same as PyTorch version)
            Bitmap resizedBitmap = resizeBitmapWithPadding(bitmap, Constants.DETECTOR_INPUT_SIZE, Constants.DETECTOR_INPUT_SIZE);

            //Log.d(TAG, "Original size: " + bitmap.getWidth() + "x" + bitmap.getHeight());
            //Log.d(TAG, "Resized size: " + resizedBitmap.getWidth() + "x" + resizedBitmap.getHeight());

            // Convert to float array (CHW format, normalized to [0, 1])
            float[] inputArray = new float[3 * Constants.DETECTOR_INPUT_SIZE * Constants.DETECTOR_INPUT_SIZE];
            int[] pixels = new int[Constants.DETECTOR_INPUT_SIZE * Constants.DETECTOR_INPUT_SIZE];
            resizedBitmap.getPixels(pixels, 0, Constants.DETECTOR_INPUT_SIZE, 0, 0, Constants.DETECTOR_INPUT_SIZE, Constants.DETECTOR_INPUT_SIZE);

            for (int i = 0; i < pixels.length; i++) {
                int pixel = pixels[i];

                // Extract RGB and normalize to [0, 1]
                float r = ((pixel >> 16) & 0xFF) / 255.0f;
                float g = ((pixel >> 8) & 0xFF) / 255.0f;
                float b = (pixel & 0xFF) / 255.0f;

                // Store in CHW format (channels first)
                inputArray[i] = r;                              // R channel
                inputArray[Constants.DETECTOR_INPUT_SIZE * Constants.DETECTOR_INPUT_SIZE + i] = g;    // G channel
                inputArray[2 * Constants.DETECTOR_INPUT_SIZE * Constants.DETECTOR_INPUT_SIZE + i] = b; // B channel
            }

            return inputArray;

        } catch (Exception e) {
            Log.e(TAG, "["+threadInfo+"]"+"Error preprocessing image", e);
            return null;
        }
    }

    private Bitmap resizeBitmapWithPadding(Bitmap originalBitmap, int targetWidth, int targetHeight) {
        int originalWidth = originalBitmap.getWidth();
        int originalHeight = originalBitmap.getHeight();

        // Calculate scale factor to maintain aspect ratio
        float scale = Math.min(
                (float) targetWidth / originalWidth,
                (float) targetHeight / originalHeight
        );

        int scaledWidth = Math.round(originalWidth * scale);
        int scaledHeight = Math.round(originalHeight * scale);

        // Calculate padding
        int padX = (targetWidth - scaledWidth) / 2;
        int padY = (targetHeight - scaledHeight) / 2;


        Log.d(TAG, String.format("Preprocessing: %dx%d -> %dx%d (scale=%.3f, pad=%d,%d)",
                originalWidth, originalHeight, targetWidth, targetHeight, scale, padX, padY));


        // Create target bitmap with gray padding
        Bitmap targetBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(targetBitmap);

        // Fill with gray color (RGB: 114, 114, 114)
        canvas.drawColor(Color.rgb(114, 114, 114));

        // Scale and center the original image
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(originalBitmap, scaledWidth, scaledHeight, true);
        canvas.drawBitmap(scaledBitmap, padX, padY, null);

        // Store scale and offset for coordinate conversion
        this.scaleX = scale;
        this.scaleY = scale;
        this.offsetX = padX;
        this.offsetY = padY;

        return targetBitmap;
    }

    private float[] runInference(float[] inputArray,String threadInfo) {
        try {
            // Create input tensor: shape [1, 3, 640, 640]
            long[] shape = {1, 3, Constants.DETECTOR_INPUT_SIZE, Constants.DETECTOR_INPUT_SIZE};

            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    ortEnvironment,
                    FloatBuffer.wrap(inputArray),
                    shape
            );

            // Run inference
            Map<String, OnnxTensor> inputs = Collections.singletonMap("images", inputTensor);
            OrtSession.Result results = ortSession.run(inputs);

            // Get output tensor
            // ONNX output shape: [1, 111, 8400] (same as PyTorch Mobile)
            OnnxTensor outputTensor = (OnnxTensor) results.get(0);
            float[][][] outputArray = (float[][][]) outputTensor.getValue();

            // Flatten to 1D array: [111 * 8400]
            float[] flatOutput = new float[NUM_FEATURES * NUM_DETECTIONS];
            int idx = 0;
            for (int i = 0; i < NUM_FEATURES; i++) {
                for (int j = 0; j < NUM_DETECTIONS; j++) {
                    flatOutput[idx++] = outputArray[0][i][j];
                }
            }

            // Cleanup
            inputTensor.close();
            results.close();

            return flatOutput;

        } catch (Exception e) {
            Log.e(TAG, "["+threadInfo+"]"+"Error during inference", e);
            return null;
        }
    }

    private DetectionResult postprocessOutput(float[] output, int originalWidth, int originalHeight,String threadInfo) {
        // THIS IS EXACTLY THE SAME AS YOUR PYTORCH VERSION
        // The output format is identical: [134, 8400]

        List<RectF> boundingBoxes = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        List<Integer> class_indices = new ArrayList<>();

        final float DEBUG_CONFIDENCE_THRESHOLD = 0.001f;
        int highConfCount = 0;
        int validBoxCount = 0;

        // Process each detection (same logic as PyTorch version)
        for (int i = 0; i < NUM_DETECTIONS; i++) {
            float centerX = output[i];
            float centerY = output[NUM_DETECTIONS + i];
            float width = output[2 * NUM_DETECTIONS + i];
            float height = output[3 * NUM_DETECTIONS + i];

            // Find the class with highest confidence
            float maxClassConf = 0;
            int bestClass = -1;

            for (int classIdx = 0; classIdx < NUM_FEATURES-4; classIdx++) {
                float classConf = output[(4 + classIdx) * NUM_DETECTIONS + i];
                if (classConf > maxClassConf) {
                    maxClassConf = classConf;
                    bestClass = classIdx;
                }
            }

            if (maxClassConf > DEBUG_CONFIDENCE_THRESHOLD) {
                highConfCount++;

                // Check if bounding box is reasonable
                if (width > 0 && height > 0 && centerX >= 0 && centerY >= 0 &&
                        centerX <= Constants.DETECTOR_INPUT_SIZE && centerY <= Constants.DETECTOR_INPUT_SIZE) {

                    validBoxCount++;

                    // Apply actual confidence threshold
                    if (maxClassConf >= Constants.CONFIDENCE_THRESHOLD) {
                        // Convert center format to corner format
                        float left = centerX - width / 2;
                        float top = centerY - height / 2;
                        float right = centerX + width / 2;
                        float bottom = centerY + height / 2;

                        // Convert coordinates from model space to original image space
                        RectF bbox = convertCoordinates(left, top, right, bottom, originalWidth, originalHeight);

                        String className = classNames.getOrDefault(bestClass, "unknown");
                        if (className != null && !className.equals("unknown")) {
                            boundingBoxes.add(bbox);
                            confidences.add(maxClassConf);
                            labels.add(className);
                            class_indices.add(bestClass);
                            //Log.d(TAG, String.format("Valid detection: %s (%.3f) at [%.1f,%.1f,%.1f,%.1f]",
                            //        className, maxClassConf, bbox.left, bbox.top, bbox.right, bbox.bottom));
                        }
                    }
                }
            }
        }

        Log.d(TAG, String.format("["+threadInfo+"]"+"Detection summary: %d high-conf (>%.3f), %d valid boxes, %d passed threshold",
                highConfCount, DEBUG_CONFIDENCE_THRESHOLD, validBoxCount, boundingBoxes.size()));

        // Apply Non-Maximum Suppression (same as PyTorch version)
        List<Integer> keepIndices = applyNMS(boundingBoxes, confidences);

        // Filter results based on NMS
        List<RectF> finalBoxes = new ArrayList<>();
        List<Float> finalConfidences = new ArrayList<>();
        List<String> finalLabels = new ArrayList<>();
        List<Integer> finalClassIndices = new ArrayList<>();

        for (int idx : keepIndices) {
            finalBoxes.add(boundingBoxes.get(idx));
            finalConfidences.add(confidences.get(idx));
            finalLabels.add(labels.get(idx));
            finalClassIndices.add(class_indices.get(idx));
        }


        Log.d(TAG, String.format("["+threadInfo+"]"+"NMS: %d -> %d detections", boundingBoxes.size(), finalBoxes.size()));

        return new DetectionResult(finalBoxes, finalConfidences, finalLabels,finalClassIndices);
    }

    private RectF convertCoordinates(float left, float top, float right, float bottom,
                                     int originalWidth, int originalHeight) {
        // Same as PyTorch version
        left -= offsetX;
        top -= offsetY;
        right -= offsetX;
        bottom -= offsetY;

        left /= scaleX;
        top /= scaleY;
        right /= scaleX;
        bottom /= scaleY;

        left = Math.max(0, Math.min(originalWidth, left));
        top = Math.max(0, Math.min(originalHeight, top));
        right = Math.max(0, Math.min(originalWidth, right));
        bottom = Math.max(0, Math.min(originalHeight, bottom));

        return new RectF(left, top, right, bottom);
    }

    private List<Integer> applyNMS(List<RectF> boxes, List<Float> confidences) {
        // Same as PyTorch version
        if (boxes.isEmpty()) {
            return new ArrayList<>();
        }

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            indices.add(i);
        }

        indices.sort((i1, i2) -> Float.compare(confidences.get(i2), confidences.get(i1)));

        List<Integer> keep = new ArrayList<>();
        boolean[] suppressed = new boolean[boxes.size()];

        for (int idx : indices) {
            if (suppressed[idx]) continue;

            keep.add(idx);
            RectF boxA = boxes.get(idx);

            for (int otherIdx : indices) {
                if (otherIdx == idx || suppressed[otherIdx]) continue;

                RectF boxB = boxes.get(otherIdx);
                float iou = calculateIoU(boxA, boxB);

                if (iou > Constants.NMS_THRESHOLD) {
                    suppressed[otherIdx] = true;
                }
            }
        }

        return keep;
    }

    private float calculateIoU(RectF boxA, RectF boxB) {
        // Same as PyTorch version
        float intersectionLeft = Math.max(boxA.left, boxB.left);
        float intersectionTop = Math.max(boxA.top, boxB.top);
        float intersectionRight = Math.min(boxA.right, boxB.right);
        float intersectionBottom = Math.min(boxA.bottom, boxB.bottom);

        if (intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom) {
            return 0.0f;
        }

        float intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop);

        float areaA = (boxA.right - boxA.left) * (boxA.bottom - boxA.top);
        float areaB = (boxB.right - boxB.left) * (boxB.bottom - boxB.top);

        float unionArea = areaA + areaB - intersectionArea;

        return unionArea > 0 ? intersectionArea / unionArea : 0.0f;
    }

    public static Bitmap drawDetections(Bitmap originalBitmap, DetectionResult detectionResult) {
        if (originalBitmap == null || detectionResult == null) {
            Log.e(TAG, "Cannot draw detections: null inputs");
            return originalBitmap;
        }

        // Create mutable copy
        Bitmap mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);

        int screenWidth = mutableBitmap.getWidth();

        // Parse colors from AppConfig (format: "#RRGGBB")
        int bboxColor = parseColor(AppConfig.bbox_color);
        int labelColor = parseColor(AppConfig.label_color);
        int labelBckColor = parseColor(AppConfig.label_bck_color);

        // Get detection data
        List<RectF> boundingBoxes = detectionResult.getBoundingBoxes();
        List<String> labels = detectionResult.getLabels();
        List<Float> confidences = detectionResult.getConfidences();

        // Draw each detection
        for (int i = 0; i < boundingBoxes.size(); i++) {
            RectF bbox = boundingBoxes.get(i);
            String label = labels.get(i);
            int confidence = Math.round(confidences.get(i) * 100);

            drawBoundingBox(
                    canvas,
                    bbox,
                    label,
                    confidence,
                    bboxColor,
                    labelColor,
                    labelBckColor,
                    screenWidth,
                    AppConfig.isBold,
                    AppConfig.show_confidence
            );
        }

        return mutableBitmap;
    }

    public static Bitmap drawDetectionsWithSmartResize(
            Bitmap originalBitmap,
            DetectionResult detectionResult,
            float offsetDp,
            Float textSizeRatio,
            DisplayMetrics displayMetrics
    ) {
        if (originalBitmap == null || detectionResult == null) {
            Log.e(TAG, "Cannot draw detections: null inputs");
            return originalBitmap;
        }

        // Create mutable copy
        Bitmap mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);

        int screenWidth = mutableBitmap.getWidth();
        int screenHeight = mutableBitmap.getHeight();

        // Convert DP to PX
        float offsetPx = offsetDp * displayMetrics.density;
        float minDistancePx = Constants.BBOX_MIN_DISTANCE * displayMetrics.density;

        // Use default text size if not provided
        float textRatio = (textSizeRatio != null) ? textSizeRatio : Constants.TEXT_SIZE_WIDTH_SCREEN;

        // Parse colors from AppConfig
        int bboxColor = parseColor(AppConfig.bbox_color);
        int labelColor = parseColor(AppConfig.label_color);
        int labelBckColor = parseColor(AppConfig.label_bck_color);

        // Get detection data
        List<RectF> originalBBoxes = detectionResult.getBoundingBoxes();
        List<String> labels = detectionResult.getLabels();
        List<Float> confidences = detectionResult.getConfidences();

        // Track processed bboxes and their text areas for collision detection
        List<RectF> processedBBoxes = new ArrayList<>();
        List<RectF> processedTextAreas = new ArrayList<>();

        // Process each detection
        for (int i = 0; i < originalBBoxes.size(); i++) {
            RectF originalBox = originalBBoxes.get(i);
            String label = labels.get(i);
            int confidence = Math.round(confidences.get(i) * 100);

            // Step 1: Try to expand bbox
            RectF expandedBox = tryExpandBBox(
                    originalBox,
                    offsetPx,
                    screenWidth,
                    screenHeight,
                    processedBBoxes,
                    minDistancePx
            );

            // Step 2: Calculate text size and area
            String labelText = AppConfig.show_confidence ? label + " " + confidence + "%" : label;
            float textSize = screenWidth * textRatio;

            // Create Paint to measure text
            Paint textPaint = new Paint();
            textPaint.setTextSize(textSize);
            textPaint.setFakeBoldText(AppConfig.isBold);
            textPaint.setAntiAlias(true);

            Rect textBounds = new Rect();
            textPaint.getTextBounds(labelText, 0, labelText.length(), textBounds);

            float paddingHorizontal = 16f;
            float paddingVertical = 8f;

            // Text area positioned above bbox
            RectF textArea = new RectF(
                    expandedBox.left,
                    expandedBox.top - textBounds.height() - paddingVertical * 2,
                    expandedBox.left + textBounds.width() + paddingHorizontal * 2,
                    expandedBox.top
            );

            // Step 3: Check if text area is valid (within screen and no collisions)
            RectF finalTextArea = adjustTextArea(
                    textArea,
                    expandedBox,
                    screenWidth,
                    processedBBoxes,
                    processedTextAreas,
                    minDistancePx,
                    textPaint,
                    labelText,
                    paddingHorizontal,
                    paddingVertical
            );

            // Step 4: Draw bbox and text
            drawBoundingBoxWithCustomText(
                    canvas,
                    expandedBox,
                    labelText,
                    finalTextArea,
                    bboxColor,
                    labelColor,
                    labelBckColor,
                    screenWidth,
                    textPaint,
                    paddingHorizontal,
                    paddingVertical
            );

            // Step 5: Add to processed lists
            processedBBoxes.add(expandedBox);
            processedTextAreas.add(finalTextArea);
        }

        Log.d(TAG, String.format("Drew %d detections with offsetDp=%.1f, textRatio=%.3f",
                originalBBoxes.size(), offsetDp, textRatio));

        return mutableBitmap;
    }

    private static RectF tryExpandBBox(
            RectF originalBox,
            float offsetPx,
            int screenWidth,
            int screenHeight,
            List<RectF> processedBBoxes,
            float minDistancePx
    ) {
        // Try to expand
        float testLeft = Math.max(0, originalBox.left - offsetPx);
        float testTop = Math.max(0, originalBox.top - offsetPx);
        float testRight = Math.min(screenWidth, originalBox.right + offsetPx);
        float testBottom = Math.min(screenHeight, originalBox.bottom + offsetPx);

        RectF expandedBox = new RectF(testLeft, testTop, testRight, testBottom);

        // Check against already processed bboxes
        for (RectF processedBox : processedBBoxes) {
            if (isBoxTooClose(expandedBox, processedBox, minDistancePx)) {
                // Can't expand, return original
                return new RectF(originalBox);
            }
        }

        // Expansion is valid
        return expandedBox;
    }

    private static boolean isBoxTooClose(RectF box1, RectF box2, float minDistance) {
        // Calculate distances between boxes
        float horizontalGap;
        float verticalGap;

        // Horizontal gap
        if (box1.right < box2.left) {
            horizontalGap = box2.left - box1.right;
        } else if (box2.right < box1.left) {
            horizontalGap = box1.left - box2.right;
        } else {
            // Overlapping horizontally - definitely too close
            return true;
        }

        // Vertical gap
        if (box1.bottom < box2.top) {
            verticalGap = box2.top - box1.bottom;
        } else if (box2.bottom < box1.top) {
            verticalGap = box1.top - box2.bottom;
        } else {
            // Overlapping vertically - definitely too close
            return true;
        }

        // If either gap is less than minimum distance, boxes are too close
        // (We only care about the closest distance)
        float minGap = Math.min(horizontalGap, verticalGap);
        return minGap < minDistance;
    }

    private static RectF adjustTextArea(
            RectF proposedTextArea,
            RectF bbox,
            int screenWidth,
            List<RectF> processedBBoxes,
            List<RectF> processedTextAreas,
            float minDistancePx,
            Paint textPaint,
            String labelText,
            float paddingHorizontal,
            float paddingVertical
    ) {
        RectF textArea = new RectF(proposedTextArea);

        // Check 1: Text area goes off top of screen
        if (textArea.top < 0) {
            // Move text below bbox instead
            textArea.top = bbox.bottom;
            textArea.bottom = bbox.bottom + (proposedTextArea.bottom - proposedTextArea.top);
        }

        // Check 2: Text area goes off right of screen
        if (textArea.right > screenWidth) {
            float shift = textArea.right - screenWidth;
            textArea.left -= shift;
            textArea.right -= shift;
            // Make sure it doesn't go off left
            if (textArea.left < 0) {
                textArea.left = 0;
                textArea.right = Math.min(screenWidth, textArea.width());
            }
        }

        // Check 3: Text area overlaps with processed bboxes
        boolean hasCollision = false;
        for (RectF processedBox : processedBBoxes) {
            if (RectF.intersects(textArea, processedBox) ||
                    isBoxTooClose(textArea, processedBox, minDistancePx)) {
                hasCollision = true;
                break;
            }
        }

        // Check 4: Text area overlaps with other text areas
        if (!hasCollision) {
            for (RectF processedText : processedTextAreas) {
                if (RectF.intersects(textArea, processedText) ||
                        isBoxTooClose(textArea, processedText, minDistancePx)) {
                    hasCollision = true;
                    break;
                }
            }
        }

        // If collision detected, try to reduce text size
        if (hasCollision) {
            // Reduce text size by 20%
            float newTextSize = textPaint.getTextSize() * 0.8f;
            textPaint.setTextSize(newTextSize);

            Rect textBounds = new Rect();
            textPaint.getTextBounds(labelText, 0, labelText.length(), textBounds);

            // Recalculate text area with smaller size
            textArea = new RectF(
                    bbox.left,
                    bbox.top - textBounds.height() - paddingVertical * 2,
                    bbox.left + textBounds.width() + paddingHorizontal * 2,
                    bbox.top
            );

            // Recheck screen bounds
            if (textArea.top < 0) {
                textArea.top = bbox.bottom;
                textArea.bottom = bbox.bottom + (textBounds.height() + paddingVertical * 2);
            }
            if (textArea.right > screenWidth) {
                float shift = textArea.right - screenWidth;
                textArea.left = Math.max(0, textArea.left - shift);
                textArea.right = screenWidth;
            }
        }

        return textArea;
    }

    private static void drawBoundingBoxWithCustomText(
            Canvas canvas,
            RectF bbox,
            String labelText,
            RectF textArea,
            int bboxColor,
            int labelColor,
            int labelBckColor,
            int screenWidth,
            Paint textPaint,
            float paddingHorizontal,
            float paddingVertical
    ) {
        // Calculate stroke width
        float strokeWidth = screenWidth * Constants.BBOX_STROKE_WIDTH_SCREEN;

        // Draw bounding box
        Paint boxPaint = new Paint();
        boxPaint.setColor(bboxColor);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(strokeWidth);
        canvas.drawRect(bbox, boxPaint);

        // Setup text paint colors
        textPaint.setColor(labelColor);

        // Draw text background
        Paint bgPaint = new Paint();
        bgPaint.setColor(labelBckColor);
        bgPaint.setStyle(Paint.Style.FILL);
        canvas.drawRect(textArea, bgPaint);

        // Draw text
        canvas.drawText(
                labelText,
                textArea.left + paddingHorizontal,
                textArea.bottom - paddingVertical,
                textPaint
        );
    }

    /**
     * Parse color string from AppConfig format (#RRGGBB) to int
     */
    private static int parseColor(String colorString) {
        try {
            if (colorString == null || colorString.isEmpty()) {
                return 0xFF00FF00; // Default green
            }
            // Android Color.parseColor equivalent
            return android.graphics.Color.parseColor(colorString);
        } catch (Exception e) {
            Log.e(TAG, "Error parsing color: " + colorString, e);
            return 0xFF00FF00; // Default green
        }
    }

    /**
     * Draw single bounding box with label
     * Based on UserAccessibility1Activity.kt implementation
     */
    private static void drawBoundingBox(
            Canvas canvas,
            RectF rect,
            String label,
            int confidence,
            int bboxColor,
            int textColor,
            int bgColor,
            int screenWidth,
            boolean isBold,
            boolean showConfidence
    ) {
        // Calculate stroke width (2% of screen width)
        float strokeWidth = screenWidth * Constants.BBOX_STROKE_WIDTH_SCREEN;

        // Draw bounding box
        Paint boxPaint = new Paint();
        boxPaint.setColor(bboxColor);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(strokeWidth);
        canvas.drawRect(rect, boxPaint);

        // Calculate text size (6% of screen width)
        float textSize = screenWidth * Constants.TEXT_SIZE_WIDTH_SCREEN;

        // Prepare label text
        String labelText = showConfidence ? label + " " + confidence + "%" : label;

        // Setup text paint
        Paint textPaint = new Paint();
        textPaint.setColor(textColor);
        textPaint.setTextSize(textSize);
        textPaint.setFakeBoldText(isBold);
        textPaint.setAntiAlias(true);

        // Measure text bounds
        Rect textBounds = new Rect();
        textPaint.getTextBounds(labelText, 0, labelText.length(), textBounds);

        // Setup background paint
        Paint bgPaint = new Paint();
        bgPaint.setColor(bgColor);
        bgPaint.setStyle(Paint.Style.FILL);

        // Calculate background rectangle with padding
        float paddingHorizontal = 16f;
        float paddingVertical = 8f;

        RectF textBgRect = new RectF(
                rect.left,
                rect.top - textBounds.height() - paddingVertical * 2,
                rect.left + textBounds.width() + paddingHorizontal * 2,
                rect.top
        );

        // Draw background
        canvas.drawRect(textBgRect, bgPaint);

        // Draw text
        canvas.drawText(
                labelText,
                rect.left + paddingHorizontal,
                rect.top - paddingVertical,
                textPaint
        );
    }

    public String getClassName(int i){
        return classNames.getOrDefault(i,"unknown");
    }

    public void close() {
        try {
            if (ortSession != null) {
                ortSession.close();
            }
            if (ortEnvironment != null) {
                ortEnvironment.close();
            }
        } catch (Exception e) {
            Log.e(TAG, "Error closing ONNX session", e);
        }
    }
}