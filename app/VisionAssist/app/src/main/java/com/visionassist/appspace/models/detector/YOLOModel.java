package com.visionassist.appspace.models.detector;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;
import com.visionassist.appspace.models.detector.DetectionResult;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.FileUtils;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public class YOLOModel {
    private static final String TAG = "YOLODetector";

    private Module model;
    private Map<Integer, String> classNames;
    private Context context;

    // Debug flags
    private static final boolean ENABLE_DETAILED_LOGGING = Constants.DEBUG_MODE;
    private static final boolean LOG_TENSOR_STATS = Constants.DEBUG_MODE;

    public YOLOModel(Context context) throws IOException {
        this.context = context;
        loadModel();
        loadClassNames();
    }

    private void loadModel() throws IOException {
        try {
            String modelPath = FileUtils.assetFilePath(context, Constants.YOLO_MODEL_FILE);
            model = Module.load(modelPath);
            Log.d(TAG, "YOLO model loaded successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to load YOLO model", e);
            throw new IOException("Failed to load YOLO model: " + e.getMessage());
        }
    }

    private void loadClassNames() throws IOException {
        try {
            classNames = FileUtils.loadClassNames(context, Constants.COCO_CLASSES_FILE);
            Log.d(TAG, "Class names loaded: " + classNames.size() + " classes");

            // Debug: Print first few classes
            for (int i = 0; i < Math.min(5, classNames.size()); i++) {
                Log.d(TAG, "Class " + i + ": " + classNames.get(i));
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to load class names", e);
            throw new IOException("Failed to load class names: " + e.getMessage());
        }
    }

    public DetectionResult detectObjects(Bitmap bitmap) {
        try {
            Log.d(TAG, "Preprocessing stage:\ninit_res: "+bitmap.getWidth() + "x" + bitmap.getHeight());
            // Step 1: Preprocess image
            Tensor inputTensor = preprocessImage(bitmap);
            if (inputTensor == null) {
                Log.e(TAG, "Failed to preprocess image");
                return new DetectionResult(new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
            }

            // Step 2: Run inference
            long startTime = System.currentTimeMillis();
            Tensor outputTensor = runInference(inputTensor);
            long inferenceTime = System.currentTimeMillis() - startTime;
            Log.d(TAG, "Inference completed in " + inferenceTime + "ms");

            if (outputTensor == null) {
                Log.e(TAG, "Inference failed");
                return new DetectionResult(new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
            }

            // Step 3: Post-process results
            DetectionResult result = postprocessOutput(outputTensor, bitmap.getWidth(), bitmap.getHeight());

            Log.d(TAG, "Detection completed: " + result.getDetectionCount() + " objects found");
            return result;

        } catch (Exception e) {
            Log.e(TAG, "Error during object detection", e);
            return new DetectionResult(new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        }
    }

    private Tensor preprocessImage(Bitmap bitmap) {
        try {
            // Resize bitmap to model input size while maintaining aspect ratio
            Bitmap resizedBitmap = resizeBitmapWithPadding(bitmap, Constants.INPUT_WIDTH, Constants.INPUT_HEIGHT);

            if (ENABLE_DETAILED_LOGGING) {
                Log.d(TAG, "Original size: " + bitmap.getWidth() + "x" + bitmap.getHeight());
                Log.d(TAG, "Resized size: " + resizedBitmap.getWidth() + "x" + resizedBitmap.getHeight());
            }

            // Convert to tensor with proper normalization for YOLOv8
            // YOLOv8 expects input normalized to [0, 1] range, not ImageNet normalization
            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                    resizedBitmap,
                    new float[]{0.0f, 0.0f, 0.0f},  // No mean subtraction
                    new float[]{1.0f, 1.0f, 1.0f}   // No std normalization (values stay in [0,1])
            );

            if (LOG_TENSOR_STATS) {
                long[] shape = inputTensor.shape(); // [1,3,H,W]
                Log.d(TAG, "Input tensor shape: " + Arrays.toString(shape));
                float[] data = inputTensor.getDataAsFloatArray();

                int channels = (int) shape[1];
                int height   = (int) shape[2];
                int width    = (int) shape[3];

                float[] minC = new float[channels];
                float[] maxC = new float[channels];
                float[] sumC = new float[channels];

                Arrays.fill(minC, Float.MAX_VALUE);
                Arrays.fill(maxC, Float.MIN_VALUE);

                int idx = 0;
                for (int c = 0; c < channels; c++) {
                    for (int h = 0; h < height; h++) {
                        for (int w = 0; w < width; w++) {
                            float val = data[idx++];
                            minC[c] = Math.min(minC[c], val);
                            maxC[c] = Math.max(maxC[c], val);
                            sumC[c] += val;
                        }
                    }
                }

                for (int c = 0; c < channels; c++) {
                    float meanC = sumC[c] / (height * width);
                    Log.d(TAG, String.format(
                            "Channel %d stats → Min: %.3f, Max: %.3f, Mean: %.3f",
                            c, minC[c], maxC[c], meanC
                    ));
                }
            }

            return inputTensor;

        } catch (Exception e) {
            Log.e(TAG, "Error preprocessing image", e);
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

        if (ENABLE_DETAILED_LOGGING) {
            Log.d(TAG, String.format("Preprocessing: %dx%d -> %dx%d (scale=%.3f, pad=%d,%d)",
                    originalWidth, originalHeight, targetWidth, targetHeight, scale, padX, padY));
        }

        // Create target bitmap with gray padding
        Bitmap targetBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(targetBitmap);

        // Fill with gray color (RGB: 114, 114, 114) - common in YOLO preprocessing
        canvas.drawColor(Color.rgb(114, 114, 114));

        // Scale and center the original image
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(originalBitmap, scaledWidth, scaledHeight, true);
        canvas.drawBitmap(scaledBitmap, padX, padY, null);

        // Store scale and offset for later coordinate conversion
        this.scaleX = scale;
        this.scaleY = scale;
        this.offsetX = padX;
        this.offsetY = padY;

        return targetBitmap;
    }

    // Store preprocessing parameters for coordinate conversion
    private float scaleX = 1.0f;
    private float scaleY = 1.0f;
    private int offsetX = 0;
    private int offsetY = 0;

    private Tensor runInference(Tensor inputTensor) {
        try {
            IValue inputIValue = IValue.from(inputTensor);
            IValue outputIValue = model.forward(inputIValue);

            Tensor outputTensor = outputIValue.toTensor();

            if (LOG_TENSOR_STATS) {
                long[] shape = outputTensor.shape();
                Log.d(TAG, "Output tensor shape: " + java.util.Arrays.toString(shape));

                // Log output statistics
                float[] data = outputTensor.getDataAsFloatArray();
                float min = Float.MAX_VALUE, max = Float.MIN_VALUE, sum = 0;
                int sampleSize = Math.min(1000, data.length);

                for (int i = 0; i < sampleSize; i++) {
                    float val = data[i];
                    min = Math.min(min, val);
                    max = Math.max(max, val);
                    sum += val;
                }

                float mean = sum / sampleSize;
                Log.d(TAG, String.format("Output tensor stats - Min: %.3f, Max: %.3f, Mean: %.3f", min, max, mean));
            }

            return outputTensor;

        } catch (Exception e) {
            Log.e(TAG, "Error during inference", e);
            return null;
        }
    }

    private DetectionResult postprocessOutput(Tensor outputTensor, int originalWidth, int originalHeight) {
        long[] shape = outputTensor.shape();
        Log.d(TAG, "Processing YOLO output with shape: " + java.util.Arrays.toString(shape));

        // YOLOv8 output shape: [1, 84, 8400]
        // where 84 = 4 (bbox coordinates) + 80 (class scores)
        // and 8400 = number of anchor points (80x80 + 40x40 + 20x20 feature maps)

        if (shape.length != 3) {
            Log.e(TAG, "Unexpected output tensor shape");
            return new DetectionResult(new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        }

        int batchSize = (int) shape[0];      // Should be 1
        int numFeatures = (int) shape[1];    // Should be 84 (4 bbox + 80 classes)
        int numDetections = (int) shape[2];  // Should be 8400

        Log.d(TAG, String.format("Processing %d detections with %d features each", numDetections, numFeatures));

        //if (numFeatures != 84) {
        //    Log.w(TAG, "Expected 84 features (4 bbox + 80 classes), got " + numFeatures);
        //}

        float[] output = outputTensor.getDataAsFloatArray();

        List<RectF> boundingBoxes = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        final float DEBUG_CONFIDENCE_THRESHOLD = 0.001f;
        int highConfCount = 0;
        int validBoxCount = 0;

        // Process each detection
        for (int i = 0; i < numDetections; i++) {
            // YOLOv8 format: data is stored as [feature][detection]
            // So for detection i: bbox coords are at indices [0:4][i], classes at [4:84][i]

            float centerX = output[0 * numDetections + i];  // First row
            float centerY = output[1 * numDetections + i];  // Second row
            float width = output[2 * numDetections + i];    // Third row
            float height = output[3 * numDetections + i];   // Fourth row

            // Find the class with highest confidence
            float maxClassConf = 0;
            int bestClass = -1;

            for (int classIdx = 0; classIdx < 80; classIdx++) {
                float classConf = output[(4 + classIdx) * numDetections + i];
                if (classConf > maxClassConf) {
                    maxClassConf = classConf;
                    bestClass = classIdx;
                }
            }

            // YOLOv8 outputs are already sigmoid-activated, no need to apply sigmoid again

            // Log first few detections for debugging
            if (i < 10 || maxClassConf > DEBUG_CONFIDENCE_THRESHOLD) {
                Log.d(TAG, String.format("Detection %d: bbox=[%.2f,%.2f,%.2f,%.2f], maxConf=%.4f, class=%d",
                        i, centerX, centerY, width, height, maxClassConf, bestClass));
            }

            if (maxClassConf > DEBUG_CONFIDENCE_THRESHOLD) {
                highConfCount++;

                // Check if bounding box is reasonable
                if (width > 0 && height > 0 && centerX >= 0 && centerY >= 0 &&
                        centerX <= Constants.INPUT_WIDTH && centerY <= Constants.INPUT_HEIGHT) {

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

                        boundingBoxes.add(bbox);
                        confidences.add(maxClassConf);

                        String className = classNames.getOrDefault(bestClass, "unknown");
                        labels.add(className);

                        if (ENABLE_DETAILED_LOGGING) {
                            Log.d(TAG, String.format("Valid detection: %s (%.3f) at [%.1f,%.1f,%.1f,%.1f]",
                                    className, maxClassConf, bbox.left, bbox.top, bbox.right, bbox.bottom));
                        }
                    }
                }
            }
        }

        Log.d(TAG, String.format("Detection summary: %d high-conf (>%.3f), %d valid boxes, %d passed threshold",
                highConfCount, DEBUG_CONFIDENCE_THRESHOLD, validBoxCount, boundingBoxes.size()));

        // Apply Non-Maximum Suppression
        List<Integer> keepIndices = applyNMS(boundingBoxes, confidences, Constants.NMS_THRESHOLD);

        // Filter results based on NMS
        List<RectF> finalBoxes = new ArrayList<>();
        List<Float> finalConfidences = new ArrayList<>();
        List<String> finalLabels = new ArrayList<>();

        for (int idx : keepIndices) {
            finalBoxes.add(boundingBoxes.get(idx));
            finalConfidences.add(confidences.get(idx));
            finalLabels.add(labels.get(idx));
        }

        Log.d(TAG, String.format("NMS: %d -> %d detections", boundingBoxes.size(), finalBoxes.size()));

        return new DetectionResult(finalBoxes, finalConfidences, finalLabels);
    }

    private RectF convertCoordinates(float left, float top, float right, float bottom,
                                     int originalWidth, int originalHeight) {
        // Convert from model coordinates (640x640) to original image coordinates

        // First, remove padding offset
        left -= offsetX;
        top -= offsetY;
        right -= offsetX;
        bottom -= offsetY;

        // Then scale back to original dimensions
        left /= scaleX;
        top /= scaleY;
        right /= scaleX;
        bottom /= scaleY;

        // Clamp to image bounds
        left = Math.max(0, Math.min(originalWidth, left));
        top = Math.max(0, Math.min(originalHeight, top));
        right = Math.max(0, Math.min(originalWidth, right));
        bottom = Math.max(0, Math.min(originalHeight, bottom));

        return new RectF(left, top, right, bottom);
    }

    private List<Integer> applyNMS(List<RectF> boxes, List<Float> confidences, float nmsThreshold) {
        if (boxes.isEmpty()) {
            return new ArrayList<>();
        }

        // Create list of indices sorted by confidence (descending)
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            indices.add(i);
        }

        indices.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer i1, Integer i2) {
                return Float.compare(confidences.get(i2), confidences.get(i1));
            }
        });

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

                if (iou > nmsThreshold) {
                    suppressed[otherIdx] = true;
                }
            }
        }

        return keep;
    }

    private float calculateIoU(RectF boxA, RectF boxB) {
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

    public Bitmap drawDetections(Bitmap originalBitmap, DetectionResult detectionResult) {
        Bitmap mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);

        Paint boxPaint = new Paint();
        boxPaint.setColor(Constants.BBOX_COLOR);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(Constants.BBOX_STROKE_WIDTH);

        Paint textPaint = new Paint();
        textPaint.setColor(Constants.TEXT_COLOR);
        textPaint.setTextSize(Constants.TEXT_SIZE);
        textPaint.setAntiAlias(true);

        Paint backgroundPaint = new Paint();
        backgroundPaint.setColor(Constants.TEXT_BACKGROUND_COLOR);
        backgroundPaint.setStyle(Paint.Style.FILL);

        List<RectF> boxes = detectionResult.getBoundingBoxes();
        List<Float> confidences = detectionResult.getConfidences();
        List<String> labels = detectionResult.getLabels();

        for (int i = 0; i < boxes.size(); i++) {
            RectF box = boxes.get(i);
            float confidence = confidences.get(i);
            String label = labels.get(i);

            // Draw bounding box
            canvas.drawRect(box, boxPaint);

            // Draw label with confidence
            String text = label + " " + String.format("%.2f", confidence);
            float textWidth = textPaint.measureText(text);
            float textHeight = textPaint.getTextSize();

            // Draw text background
            canvas.drawRect(box.left, box.top - textHeight - 4,
                    box.left + textWidth + 8, box.top, backgroundPaint);

            // Draw text
            canvas.drawText(text, box.left + 4, box.top - 4, textPaint);
        }

        return mutableBitmap;
    }
}