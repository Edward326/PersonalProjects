package com.visionassist.appspace.models.captioner;
import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.FileUtils;
import java.io.File;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class BLIPModel {
    private static final String TAG = "BlipCaptioner";

    private OrtEnvironment ortEnvironment;
    private OrtSession ortSession;
    private final Context context;
    private Tokenizer tokenizer;
    private Random random;

    // BLIP preprocessing parameters
    private float[] imageMean = {0.48145466f, 0.4578275f, 0.40821073f};
    private float[] imageStd = {0.26862954f, 0.26130258f, 0.27577711f};
    private int imageSize = 384;
    private int maxLength = 50; // Maximum caption length

    public BLIPModel(Context context) {
        this.context = context;
        this.random = new Random();
    }

    public int initModel(){
        try {
            this.ortEnvironment = OrtEnvironment.getEnvironment();
            this.tokenizer = new Tokenizer(context);
            loadPreprocessingInfo();
            return loadModel();
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize BlipCaptioner", e);
            return -1;
        }
    }

    private void loadPreprocessingInfo() {
        JsonObject preprocessingInfo;
        try {
            if (FileUtils.assetExists(context, "model_info.json")) {
                String jsonContent = FileUtils.loadAssetAsString(context, "model_info.json");
                preprocessingInfo = new Gson().fromJson(jsonContent, JsonObject.class);

                if (preprocessingInfo.has("preprocessing")) {
                    JsonObject preprocessing = preprocessingInfo.getAsJsonObject("preprocessing");

                    if (preprocessing.has("image_size")) {
                        if (preprocessing.get("image_size").isJsonArray()) {
                            int[] size = new Gson().fromJson(preprocessing.get("image_size"), int[].class);
                            imageSize = size[0];
                        } else {
                            imageSize = preprocessing.get("image_size").getAsInt();
                        }
                    }

                    if (preprocessing.has("normalization")) {
                        JsonObject norm = preprocessing.getAsJsonObject("normalization");
                        if (norm.has("mean")) {
                            float[] mean = new Gson().fromJson(norm.get("mean"), float[].class);
                            if (mean != null && mean.length == 3) {
                                imageMean = mean;
                            }
                        }
                        if (norm.has("std")) {
                            float[] std = new Gson().fromJson(norm.get("std"), float[].class);
                            if (std != null && std.length == 3) {
                                imageStd = std;
                            }
                        }
                    }
                }

                if (preprocessingInfo.has("generation_config")) {
                    JsonObject genConfig = preprocessingInfo.getAsJsonObject("generation_config");
                    if (genConfig.has("max_length")) {
                        maxLength = genConfig.get("max_length").getAsInt();
                    }
                }

                Log.d(TAG, "Preprocessing info loaded from JSON");
            }
        } catch (Exception e) {
            Log.w(TAG, "Could not load preprocessing info, using defaults", e);
        }
    }

    public int loadModel() {
        String modelPath;

        try {
                Log.d(TAG, "Attempting to load quantized BLIP model...");

                if (FileUtils.assetExists(context, Constants.BLIP_MODEL_FILE)) {
                    modelPath = FileUtils.assetFilePath(context, Constants.BLIP_MODEL_FILE);

                    File modelFile = new File(modelPath);
                    Log.d(TAG, "Quantized model size: " + (modelFile.length() / 1024 / 1024) + " MB");

                    OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
                    sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

                    ortSession = ortEnvironment.createSession(modelPath, sessionOptions);

                    Log.d(TAG, "Quantized BLIP model loaded successfully!");
                } else {
                    Log.w(TAG, "Quantized model not found: " + Constants.BLIP_MODEL_FILE);
                    throw new Exception("Quantized model not available");
                }

        } catch (Exception e) {
            Log.w(TAG, "Failed to load quantized model, captioner not loaded", e);
            return -1;
        }

        if (ortSession != null) {
            Log.d(TAG, "Model inputs: " + ortSession.getInputNames());
            Log.d(TAG, "Model outputs: " + ortSession.getOutputNames());
            return 0;
        }
        else return -1;
    }

    public int[] generateCaption(Bitmap bitmap) {
        long startTime = System.currentTimeMillis();

        try {
            int[] result = generateRealCaption(bitmap);

            long endTime = System.currentTimeMillis();
            Log.d(TAG, String.format("Caption generated in %dms",endTime - startTime));

            return result;
        } catch (Exception e) {
            Log.e(TAG, "Real caption generation failed, using fallback", e);
            return null;
        }
    }

    private int[] generateRealCaption(Bitmap bitmap) throws Exception {
        // Step 1: Preprocess image to [1, 3, 384, 384] tensor
        float[][][] imageArray = preprocessImage(bitmap);

        // Step 2: Implement autoregressive generation loop
        return autoregressiveGenerate(imageArray);

        // Step 3: Decode tokens to text using vocabulary

        //return tokenizer.decode(generatedTokens);
    }

    private float[][][] preprocessImage(Bitmap bitmap) {
        // Resize bitmap to model input size
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true);

        int width = resizedBitmap.getWidth();
        int height = resizedBitmap.getHeight();

        float[][][] imageArray = new float[3][height][width];

        // Extract pixels and normalize
        int[] pixels = new int[width * height];
        resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int pixel = pixels[i * width + j];

                // Extract RGB values and normalize to [0,1]
                float r = ((pixel >> 16) & 0xFF) / 255.0f;
                float g = ((pixel >> 8) & 0xFF) / 255.0f;
                float b = (pixel & 0xFF) / 255.0f;

                // Apply BLIP normalization: (pixel - mean) / std
                imageArray[0][i][j] = (r - imageMean[0]) / imageStd[0]; // R channel
                imageArray[1][i][j] = (g - imageMean[1]) / imageStd[1]; // G channel
                imageArray[2][i][j] = (b - imageMean[2]) / imageStd[2]; // B channel
            }
        }

        return imageArray;
    }

    private int[] autoregressiveGenerate(float[][][] imageArray) throws OrtException {
        // Initialize sequence with BOS token
        int[] currentSequence = new int[maxLength];
        currentSequence[0] = (int)tokenizer.getBosToken(); // Start with <BOS>
        int currentLength = 1;

        Log.d(TAG, "Starting autoregressive generation with BOS token: " + tokenizer.getBosToken());

        // Generate tokens one by one
        for (int step = 0; step < maxLength - 1; step++) {
            Log.d(TAG, "Generation step " + step + ", current length: " + currentLength);

            // Create input tensors for current state
            Map<String, OnnxTensor> inputs = createInputTensors(imageArray, currentSequence, currentLength);

            try {
                // Run inference
                OrtSession.Result results = ortSession.run(inputs);

                // Get logits: [batch_size, sequence_length, vocab_size]
                OnnxTensor outputTensor = (OnnxTensor) results.get(0);
                float[][][] logits = (float[][][]) outputTensor.getValue();

                // Get next token from the last position
                long nextToken = getNextToken(logits[0][currentLength - 1]); // Use last position

                Log.d(TAG, "Generated token: " + nextToken + " (" + tokenizer.getTokenString(nextToken) + ")");

                // Clean up
                for (OnnxTensor tensor : inputs.values()) {
                    tensor.close();
                }
                results.close();

                // Check for EOS token
                if (nextToken == tokenizer.getEosToken()) {
                    Log.d(TAG, "EOS token encountered, stopping generation");
                    break;
                }

                // Add next token to sequence
                currentSequence[currentLength] = (int)nextToken;
                currentLength++;

            } catch (Exception e) {
                Log.e(TAG, "Error during generation step " + step, e);
                // Clean up inputs on error
                for (OnnxTensor tensor : inputs.values()) {
                    tensor.close();
                }
                throw e;
            }
        }

        Log.d(TAG, "Generation completed, final length: " + currentLength);

        // Return generated tokens (excluding BOS)
        return Arrays.copyOfRange(currentSequence, 1, currentLength);
    }

    private Map<String, OnnxTensor> createInputTensors(float[][][] imageArray, int[] sequence, int seqLength) throws OrtException {
        Map<String, OnnxTensor> inputs = new HashMap<>();

        // Create image tensor [1, 3, H, W]
        long[] imageShape = {1, 3, imageSize, imageSize};
        FloatBuffer imageBuffer = FloatBuffer.allocate(3 * imageSize * imageSize);

        // Fill buffer in CHW format
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < imageSize; h++) {
                for (int w = 0; w < imageSize; w++) {
                    imageBuffer.put(imageArray[c][h][w]);
                }
            }
        }
        imageBuffer.rewind();

        OnnxTensor imageTensor = OnnxTensor.createTensor(ortEnvironment, imageBuffer, imageShape);
        inputs.put("pixel_values", imageTensor);

        // Create decoder input IDs tensor [1, sequence_length]
        long[] textShape = {1, seqLength};
        LongBuffer inputIdsBuffer = LongBuffer.allocate(seqLength);

        // Fill with current sequence
        for (int i = 0; i < seqLength; i++) {
            inputIdsBuffer.put(sequence[i]);
        }
        inputIdsBuffer.rewind();

        OnnxTensor inputIdsTensor = OnnxTensor.createTensor(ortEnvironment, inputIdsBuffer, textShape);
        inputs.put("decoder_input_ids", inputIdsTensor);

        return inputs;
    }

    private long getNextToken(float[] logits) {
        // Simple greedy decoding - choose token with highest probability
        int bestToken = 0;
        float bestScore = logits[0];

        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > bestScore) {
                bestScore = logits[i];
                bestToken = i;
            }
        }

        // Add some temperature sampling for variety (optional)
        if (random.nextFloat() < 0.1f) { // 10% chance for sampling
            return sampleFromLogits(logits, 0.8f);
        }

        return bestToken;
    }

    private long sampleFromLogits(float[] logits, float temperature) {
        // Apply temperature scaling
        float[] scaledLogits = new float[logits.length];
        float maxLogit = Float.NEGATIVE_INFINITY;

        // Find max for numerical stability
        for (float logit : logits) {
            maxLogit = Math.max(maxLogit, logit);
        }

        // Scale and compute softmax
        float sumExp = 0.0f;
        for (int i = 0; i < logits.length; i++) {
            scaledLogits[i] = (float) Math.exp((logits[i] - maxLogit) / temperature);
            sumExp += scaledLogits[i];
        }

        // Normalize to probabilities
        for (int i = 0; i < scaledLogits.length; i++) {
            scaledLogits[i] /= sumExp;
        }

        // Sample from distribution
        float randomValue = random.nextFloat();
        float cumulativeProb = 0.0f;

        for (int i = 0; i < scaledLogits.length; i++) {
            cumulativeProb += scaledLogits[i];
            if (randomValue <= cumulativeProb) {
                return i;
            }
        }

        // Fallback to greedy
        return getNextToken(logits);
    }

    private String generateFallbackCaption(List<String> labels) {
        if (labels == null || labels.isEmpty()) {
            String[] fallbacks = {
                    "I can see an image but cannot identify specific objects clearly.",
                    "This appears to be a scene with some objects that I cannot identify precisely.",
                    "I'm looking at an image but cannot determine the specific contents."
            };
            return fallbacks[random.nextInt(fallbacks.length)];
        }

        StringBuilder caption = new StringBuilder();

        String[] starters = {
                "I can identify ",
                "This image shows ",
                "I can see ",
                "The image contains "
        };

        caption.append(starters[random.nextInt(starters.length)]);

        if (labels.size() == 1) {
            caption.append(addArticle(labels.get(0)));
        } else if (labels.size() == 2) {
            caption.append(addArticle(labels.get(0)))
                    .append(" and ")
                    .append(addArticle(labels.get(1)));
        } else {
            caption.append("several objects including ");
            for (int i = 0; i < Math.min(3, labels.size()); i++) {
                if (i > 0) caption.append(i == labels.size() - 1 ? " and " : ", ");
                caption.append(addArticle(labels.get(i)));
            }
            if (labels.size() > 3) {
                caption.append(" among others");
            }
        }

        caption.append(".");
        return caption.toString();
    }

    private String addArticle(String noun) {
        if (noun.matches("^[aeiouAEIOU].*")) {
            return "an " + noun;
        } else {
            return "a " + noun;
        }
    }

    public Tokenizer getTokenizer(){
        return tokenizer;
    }

    public void close() {
        try {
            if (ortSession != null) {
                ortSession.close();
                ortSession = null;
            }
            Log.d(TAG, "BlipCaptioner resources cleaned up");
        } catch (OrtException e) {
            Log.e(TAG, "Error closing ONNX session", e);
        }
    }
}