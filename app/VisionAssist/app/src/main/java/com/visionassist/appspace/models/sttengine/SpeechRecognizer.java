package com.visionassist.appspace.models.sttengine;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;
import com.visionassist.appspace.utils.Constants;
import org.json.JSONArray;
import org.json.JSONObject;
import org.vosk.Model;
import org.vosk.Recognizer;
import org.vosk.android.RecognitionListener;
import org.vosk.android.SpeechService;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class SpeechRecognizer {
    private static final String TAG = "SpeechRecognizer";

    private Context context;
    private Model model;
    private SpeechService speechService;

    // Useless words to filter out
    private Set<String> uselessWords;

    // Map of class index -> list of synonyms
    private Map<Integer, List<String>> objectSynonyms;

    // Recognition listener callback
    private RecognitionCallback recognitionCallback;

    private static final float SAMPLE_RATE = 16000.0f;
    public boolean isReady=false;

    public interface RecognitionCallback {
        void onResult(String recognizedText);

        void onError(String error);
    }

    public SpeechRecognizer(Context context) {
        this.context = context;
    }

    public int loadModel() {
        try {
            Log.d(TAG, "Loading speech recognition model...");

            // Load useless words
            if (loadUselessWords() != 0) {
                Log.e(TAG, "Failed to load useless words");
                return -1;
            }
            Log.d(TAG, "Loaded " + uselessWords.size() + " useless words");

            // Load object synonyms
            if (loadObjectSynonyms() != 0) {
                Log.e(TAG, "Failed to load object synonyms");
                return -1;
            }
            Log.d(TAG, "Loaded synonyms for " + objectSynonyms.size() + " object classes");

            loadVoskModel();

            return 0;
        } catch (Exception e) {
            Log.e(TAG, "Failed to load speech recognition model", e);
            return -1;
        }
    }

    private int loadUselessWords() {
        try {
            uselessWords = new HashSet<>();

            InputStream inputStream = context.getAssets().open(Constants.USELESS_WORDS_FILE);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim().toLowerCase();
                if (!line.isEmpty()) {
                    uselessWords.add(line);
                }
            }

            reader.close();
            inputStream.close();
            return 0;

        } catch (IOException e) {
            Log.e(TAG, "Failed to load useless words", e);
            return -1;
        }
    }

    private int loadObjectSynonyms() {
        try {
            objectSynonyms = new HashMap<>();

            // Read JSON file
            InputStream inputStream = context.getAssets().open(Constants.OBJECT_SYNONYMS_FILE);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            StringBuilder jsonBuilder = new StringBuilder();

            String line;
            while ((line = reader.readLine()) != null) {
                jsonBuilder.append(line);
            }

            reader.close();
            inputStream.close();

            // Parse JSON
            JSONObject jsonObject = new JSONObject(jsonBuilder.toString());

            // Iterate through all keys
            for (Iterator<String> it = jsonObject.keys(); it.hasNext(); ) {
                String key = it.next();
                int classIndex = Integer.parseInt(key);

                JSONArray synonymsArray = jsonObject.getJSONArray(key);
                List<String> synonymsList = new ArrayList<>();

                for (int i = 0; i < synonymsArray.length(); i++) {
                    String synonym = synonymsArray.getString(i).toLowerCase();
                    synonymsList.add(synonym);
                }

                objectSynonyms.put(classIndex, synonymsList);
            }
            return 0;

        } catch (Exception e) {
            Log.e(TAG, "Failed to load object synonyms", e);
            return -1;
        }
    }

    private void loadVoskModel() {
        new Thread(() -> {
            try {
                File modelDir = new File(context.getFilesDir(), Constants.VOSK_MODEL_DIR);

                // Check if model already exists in internal storage
                if (modelDir.exists() && new File(modelDir, "uuid").exists()) {
                    Log.d(TAG, "Loading existing model from: " + modelDir.getAbsolutePath());
                    model = new Model(modelDir.getAbsolutePath());
                    isReady = true;
                    Log.d(TAG, "✅ Model loaded successfully");
                    return;
                }

                // Model doesn't exist, copy from assets
                Log.d(TAG, "Copying model from assets to internal storage...");

                if (!modelDir.exists()) {
                    modelDir.mkdirs();
                }

                // Copy all files from assets/vosk-model-small-en-us-0.15 to internal storage
                copyAssetFolder(Constants.VOSK_MODEL_DIR, modelDir.getAbsolutePath());

                // Verify uuid exists
                File uuidFile = new File(modelDir, "uuid");
                if (!uuidFile.exists()) {
                    Log.e(TAG, "uuid file missing after copy!");
                    isReady = false;
                    return;
                }

                Log.d(TAG, "Model copied successfully, loading...");

                // Load the model
                model = new Model(modelDir.getAbsolutePath());
                isReady = true;
                Log.d(TAG, "✅ Model loaded successfully");

            } catch (Exception e) {
                Log.e(TAG, "❌ Failed to load model", e);
                isReady = false;
            }
        }).start();
    }

    private void copyAssetFolder(String assetFolder, String targetPath) throws IOException {
        AssetManager assetManager = context.getAssets();
        String[] files = assetManager.list(assetFolder);

        if (files == null || files.length == 0) {
            // It's a file, not a folder
            copyAssetFile(assetFolder, targetPath);
            return;
        }

        // It's a folder
        File targetDir = new File(targetPath);
        if (!targetDir.exists()) {
            targetDir.mkdirs();
        }

        for (String filename : files) {
            String assetPath = assetFolder + "/" + filename;
            String targetFilePath = targetPath + "/" + filename;

            String[] subFiles = assetManager.list(assetPath);
            if (subFiles != null && subFiles.length > 0) {
                // It's a subfolder
                copyAssetFolder(assetPath, targetFilePath);
            } else {
                // It's a file
                copyAssetFile(assetPath, targetFilePath);
            }
        }
    }

    private void copyAssetFile(String assetPath, String targetPath) throws IOException {
        InputStream in = context.getAssets().open(assetPath);
        FileOutputStream out = new FileOutputStream(targetPath);

        byte[] buffer = new byte[8192];
        int read;
        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }

        in.close();
        out.close();
    }

    public void startListening(RecognitionCallback callback) {
        if (model == null) {
            Log.e(TAG, "Model not loaded");
            if (callback != null) {
                callback.onError("Model not loaded");
            }
            return;
        }

        this.recognitionCallback = callback;

        try {
            Recognizer recognizer = new Recognizer(model, SAMPLE_RATE);

            speechService = new SpeechService(recognizer, SAMPLE_RATE);

            speechService.startListening(new RecognitionListener() {
                @Override
                public void onResult(String hypothesis) {
                    try {
                        JSONObject jsonResult = new JSONObject(hypothesis);
                        String text = jsonResult.getString("text");

                        if (!text.isEmpty()) {
                            stopListening();
                            Log.d(TAG, "Recognized: " + text);
                            if (recognitionCallback != null) {
                                recognitionCallback.onResult(text);
                            }
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "Error parsing result", e);
                        if (callback != null) {
                            callback.onError(e.getMessage());
                        }
                    }
                }

                @Override
                public void onFinalResult(String hypothesis) {
                    // Called when recognition is complete
                    try {
                        JSONObject jsonResult = new JSONObject(hypothesis);
                        String text = jsonResult.getString("text");

                        if (!text.isEmpty()) {
                            stopListening();
                            Log.d(TAG, "Final result: " + text);
                            if (recognitionCallback != null) {
                                recognitionCallback.onResult(text);
                            }
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "Error parsing final result", e);
                        if (callback != null) {
                            callback.onError(e.getMessage());
                        }
                    }
                }

                @Override
                public void onPartialResult(String hypothesis) {
                    // Optional: handle partial results
                    Log.d(TAG, "Partial: " + hypothesis);
                }

                @Override
                public void onError(Exception e) {
                    Log.e(TAG, "Recognition error", e);
                    if (recognitionCallback != null) {
                        recognitionCallback.onError(e.getMessage());
                    }
                }

                @Override
                public void onTimeout() {
                    Log.d(TAG, "Recognition timeout");
                    if (recognitionCallback != null) {
                        recognitionCallback.onError("Timeout");
                    }
                }
            });

            Log.d(TAG, "🎤 Started listening for speech");

        } catch (Exception e) {
            Log.e(TAG, "Failed to start listening", e);
            if (callback != null) {
                callback.onError(e.getMessage());
            }
        }
    }

    public void stopListening() {
        if (speechService != null) {
            speechService.stop();
            speechService.shutdown();
            speechService = null;
            Log.d(TAG, "🛑 Stopped listening");
        }
    }

    public List<Integer> processRecognizedText(String recognizedText) {
        if (recognizedText == null || recognizedText.trim().isEmpty()) {
            Log.w(TAG, "Empty recognized text");
            return new ArrayList<>();
        }

        Log.d(TAG, "Processing: \"" + recognizedText + "\"");

        // Step 1: Split into words and convert to lowercase
        String[] words = recognizedText.toLowerCase().trim().split("\\s+");

        // Step 2: Remove useless words
        List<String> meaningfulWords = new ArrayList<>();
        for (String word : words) {
            // Remove punctuation
            word = word.replaceAll("[^a-z0-9]", "");

            if (!word.isEmpty() && !uselessWords.contains(word)) {
                meaningfulWords.add(word);
            }
        }

        Log.d(TAG, "Meaningful words after filtering: " + meaningfulWords);

        // Step 3: Match words with object synonyms
        List<Integer> matchedIndices = new ArrayList<>();
        Set<Integer> uniqueIndices = new HashSet<>(); // Avoid duplicates

        for (String word : meaningfulWords) {
            // Search through all object synonyms
            for (Map.Entry<Integer, List<String>> entry : objectSynonyms.entrySet()) {
                int classIndex = entry.getKey();
                List<String> synonyms = entry.getValue();

                // Check if word matches any synonym
                if (synonyms.contains(word)) {
                    if (!uniqueIndices.contains(classIndex)) {
                        matchedIndices.add(classIndex);
                        uniqueIndices.add(classIndex);

                        Log.d(TAG, "Matched \"" + word + "\" -> class " + classIndex);
                    }
                    break; // Move to next word
                }
            }
        }

        // Step 4: Also try to match multi-word phrases
        String fullText = String.join(" ", meaningfulWords);
        for (Map.Entry<Integer, List<String>> entry : objectSynonyms.entrySet()) {
            int classIndex = entry.getKey();
            List<String> synonyms = entry.getValue();

            for (String synonym : synonyms) {
                if (synonym.contains(" ") && fullText.contains(synonym)) {
                    if (!uniqueIndices.contains(classIndex)) {
                        matchedIndices.add(classIndex);
                        uniqueIndices.add(classIndex);

                        Log.d(TAG, "Matched phrase \"" + synonym + "\" -> class " + classIndex);
                    }
                }
            }
        }

        Log.d(TAG, "Final matched indices: " + matchedIndices);
        return matchedIndices;
    }

    public Model getModel() {
        return model;
    }

    /**
     * Cleanup resources
     */
    public void close() {
        stopListening();

        if (model != null) {
            model.close();
            model = null;
        }

        Log.d(TAG, "Speech recognizer closed");
    }
}