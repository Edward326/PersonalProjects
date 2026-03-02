package com.visionassist.appspace.models.classifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.util.Log;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.FileUtils;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class YOLOClassifier {
    private static final String TAG = "YOLOClassifier";

    private OrtEnvironment ortEnvironment;
    private OrtSession ortSession;
    private Map<Integer, String> classNames;
    private Context context;

    public YOLOClassifier(Context context) {
        this.context = context;
    }

    public int loadModel() {
        try {
            Log.d(TAG, "Loading Classifier Model YOLO ...");

            // Create ONNX Runtime environment
            ortEnvironment = OrtEnvironment.getEnvironment();

            // Load model from assets
            String modelPath = FileUtils.assetFilePath(context, Constants.YOLO_MODEL_CLASSIFIER_FILE);

            // Create session options
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

            // Load session
            ortSession = ortEnvironment.createSession(modelPath, options);

            Log.d(TAG, "Classifier Model loaded successfully");

            loadClassNames();
            Log.d(TAG, "Classifier class names loaded: " + classNames.size() + " classes");

            return 0;
        } catch (Exception e) {
            Log.e(TAG, "Failed to load Classifier Model YOLO", e);
            return -1;
        }
    }

    private void loadClassNames() throws IOException {
        try {
            // Load class names based on current app language
            String classNameFile = AppConfig.mainLanguage.getCode().equals("en")
                    ? Constants.CLASSIFIER_CLASSES_FILE_EN
                    : Constants.CLASSIFIER_CLASSES_FILE_RO;

            classNames = FileUtils.loadClassNames(context, classNameFile);
        } catch (Exception e) {
            Log.e(TAG, "Failed to load classifier class names", e);
            throw new IOException("Failed to load class names: " + e.getMessage());
        }
    }

    public int detectScene(Bitmap bitmap,String threadInfo) {
        try {
            Log.d(TAG, "["+threadInfo+"]"+"scene classification on image: " + bitmap.getWidth() + "x" + bitmap.getHeight());

            // Step 1: Preprocess image to 224x224
            long startPreprocess = System.currentTimeMillis();
            float[] inputArray = preprocessImage(bitmap,threadInfo);
            long preprocessTime = System.currentTimeMillis() - startPreprocess;

            if (inputArray == null) {
                Log.e(TAG, "["+threadInfo+"]"+"Failed to preprocess image");
                return -1;
            }

            // Step 2: Run inference
            long startInference = System.currentTimeMillis();
            float[] output = runInference(inputArray,threadInfo);
            long inferenceTime = System.currentTimeMillis() - startInference;

            if (output == null) {
                Log.e(TAG, "["+threadInfo+"]"+"Inference failed");
                return -1;
            }

            // Step 3: Get class with maximum confidence
            long startPostprocess = System.currentTimeMillis();
            int detectedScene = getTopClass(output,threadInfo);
            long postprocessTime = System.currentTimeMillis() - startPostprocess;

            Log.d(TAG, "["+threadInfo+"]"+"Preprocessing completed in " + preprocessTime + "ms");
            Log.d(TAG, "["+threadInfo+"]"+"Inference completed in " + inferenceTime + "ms");
            Log.d(TAG, "["+threadInfo+"]"+"Post-processing completed in " + postprocessTime + "ms");
            Log.d(TAG, "["+threadInfo+"]"+"Total time for inference: " + (preprocessTime + inferenceTime + postprocessTime) + "ms");
            Log.d(TAG, "["+threadInfo+"]"+"Scene classification completed:\nScene id found: " + detectedScene);

            return detectedScene;

        } catch (Exception e) {
            Log.e(TAG, "["+threadInfo+"]"+"Error during scene classification", e);
            return -1;
        }
    }

    private float[] preprocessImage(Bitmap bitmap, String threadInfo) {
        try {
            // Resize bitmap to 224x224 with padding (maintaining aspect ratio)
            Bitmap resizedBitmap = resizeBitmapWithPadding(bitmap, Constants.CLASSIFIER_INPUT_SIZE, Constants.CLASSIFIER_INPUT_SIZE);

            //Log.d(TAG, "Original size: " + bitmap.getWidth() + "x" + bitmap.getHeight());
            //Log.d(TAG, "Resized size: " + resizedBitmap.getWidth() + "x" + resizedBitmap.getHeight());

            // Convert to float array (CHW format, normalized to [0, 1])
            float[] inputArray = new float[3 * Constants.CLASSIFIER_INPUT_SIZE * Constants.CLASSIFIER_INPUT_SIZE];
            int[] pixels = new int[Constants.CLASSIFIER_INPUT_SIZE * Constants.CLASSIFIER_INPUT_SIZE];
            resizedBitmap.getPixels(pixels, 0, Constants.CLASSIFIER_INPUT_SIZE, 0, 0, Constants.CLASSIFIER_INPUT_SIZE, Constants.CLASSIFIER_INPUT_SIZE);

            for (int i = 0; i < pixels.length; i++) {
                int pixel = pixels[i];

                // Extract RGB and normalize to [0, 1]
                float r = ((pixel >> 16) & 0xFF) / 255.0f;
                float g = ((pixel >> 8) & 0xFF) / 255.0f;
                float b = (pixel & 0xFF) / 255.0f;

                // Store in CHW format (channels first)
                inputArray[i] = r;                              // R channel
                inputArray[Constants.CLASSIFIER_INPUT_SIZE * Constants.CLASSIFIER_INPUT_SIZE + i] = g;    // G channel
                inputArray[2 * Constants.CLASSIFIER_INPUT_SIZE * Constants.CLASSIFIER_INPUT_SIZE + i] = b; // B channel
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

        return targetBitmap;
    }

    private float[] runInference(float[] inputArray,String threadInfo) {
        try {
            // Create input tensor: shape [1, 3, 224, 224]
            long[] shape = {1, 3, Constants.CLASSIFIER_INPUT_SIZE, Constants.CLASSIFIER_INPUT_SIZE};

            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    ortEnvironment,
                    FloatBuffer.wrap(inputArray),
                    shape
            );

            // Run inference
            Map<String, OnnxTensor> inputs = Collections.singletonMap("images", inputTensor);
            OrtSession.Result results = ortSession.run(inputs);

            // Get output tensor
            // ONNX classifier output shape: [1, 32] (batch_size, num_classes)
            OnnxTensor outputTensor = (OnnxTensor) results.get(0);
            float[][] outputArray = (float[][]) outputTensor.getValue();

            // Extract the class probabilities (first batch element)
            float[] classScores = outputArray[0];

            // Cleanup
            inputTensor.close();
            results.close();

            return classScores;

        } catch (Exception e) {
            Log.e(TAG, "["+threadInfo+"]"+"Error during inference", e);
            return null;
        }
    }

    private int getTopClass(float[] classScores,String threadInfo) {
        if (classScores == null || classScores.length == 0) {
            Log.e(TAG, "["+threadInfo+"]"+"Invalid class scores");
            return -1;
        }

        // Find the index with maximum confidence
        int maxIndex = 0;
        float maxConfidence = classScores[0];

        for (int i = 1; i < classScores.length; i++) {
            if (classScores[i] > maxConfidence) {
                maxConfidence = classScores[i];
                maxIndex = i;
            }
        }

        // Get class name from map
        String className = classNames.getOrDefault(maxIndex, "unknown");
        Log.d(TAG, String.format("["+threadInfo+"]"+"Top prediction: class_id=%d, name=%s, confidence=%.3f",
                maxIndex, className, maxConfidence));

        return maxIndex;
    }

    public String getClassName(int classId) {
        return classNames.getOrDefault(classId, "unknown");
    }

    /**
     * Close and cleanup resources
     */
    public void close() {
        try {
            if (ortSession != null) {
                ortSession.close();
            }
            if (ortEnvironment != null) {
                ortEnvironment.close();
            }
            Log.d(TAG, "Classifier closed successfully");
        } catch (Exception e) {
            Log.e(TAG, "Error closing classifier", e);
        }
    }
}