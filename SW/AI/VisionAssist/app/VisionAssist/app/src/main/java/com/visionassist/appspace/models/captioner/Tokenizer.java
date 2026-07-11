package com.visionassist.appspace.models.captioner;

import android.content.Context;
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.visionassist.appspace.utils.FileUtils;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * BLIP tokenizer that loads BERT vocabulary from vocab.txt file
 * This handles the actual vocabulary used by your exported BLIP model
 */
public class Tokenizer {
    private static final String TAG = "BLIPTokenizer";

    // BERT/BLIP special token IDs (standard values)
    public static final int PAD_TOKEN = 0;      // [PAD]
    public static final int UNK_TOKEN = 100;    // [UNK]
    public static final int CLS_TOKEN = 101;    // [CLS] - BOS for BLIP
    public static final int SEP_TOKEN = 102;    // [SEP] - EOS for BLIP
    public static final int MASK_TOKEN = 103;   // [MASK]

    private Map<String, Integer> tokenToId;
    private Map<Integer, String> idToToken;
    private Context context;
    private boolean isLoaded = false;
    private int vocabSize = 0;

    public Tokenizer(Context context) {
        this.context = context;
        this.tokenToId = new HashMap<>();
        this.idToToken = new HashMap<>();
        loadVocabulary();
    }

    private void loadVocabulary() {
        try {
            Log.d(TAG, "Attempting to load BLIP vocabulary...");

            // Try different vocabulary file names that might be in assets
            String[] vocabFiles = {
                    "vocab.txt",           // Standard BERT vocab
                    "blip_vocab.txt",      // BLIP specific vocab
                    "tokenizer_vocab.txt", // Alternative name
                    "blip_vocab.json"      // JSON format
            };

            boolean loaded = false;
            for (String vocabFile : vocabFiles) {
                if (FileUtils.assetExists(context, vocabFile)) {
                    Log.d(TAG, "Found vocabulary file: " + vocabFile);

                    if (vocabFile.endsWith(".json")) {
                        loadFromJson(vocabFile);
                    } else {
                        loadFromVocabTxt(vocabFile);
                    }
                    loaded = true;
                    break;
                }
            }

            if (!loaded) {
                Log.w(TAG, "No vocabulary file found, creating minimal vocabulary");
                createMinimalVocabulary();
            }

            isLoaded = true;
            Log.d(TAG, "Vocabulary loaded successfully: " + vocabSize + " tokens");

            // Log some sample tokens for verification
            logSampleTokens();

        } catch (Exception e) {
            Log.e(TAG, "Failed to load vocabulary", e);
            createMinimalVocabulary();
            isLoaded = true;
        }
    }

    private void loadFromVocabTxt(String filename) throws Exception {
        Log.d(TAG, "Loading vocabulary from " + filename);

        InputStream inputStream = context.getAssets().open(filename);
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

        String line;
        int tokenId = 0;

        while ((line = reader.readLine()) != null) {
            String token = line.trim();
            if (!token.isEmpty()) {
                tokenToId.put(token, tokenId);
                idToToken.put(tokenId, token);
                tokenId++;
            }
        }

        reader.close();
        vocabSize = tokenId;

        Log.d(TAG, "Loaded " + vocabSize + " tokens from " + filename);
    }

    private void loadFromJson(String filename) throws Exception {
        Log.d(TAG, "Loading vocabulary from JSON: " + filename);

        String jsonContent = FileUtils.loadAssetAsString(context, filename);
        Type type = new TypeToken<Map<String, Integer>>() {
        }.getType();
        Map<String, Integer> vocab = new Gson().fromJson(jsonContent, type);

        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            int tokenId = entry.getValue();
            String token = entry.getKey();
            tokenToId.put(token, tokenId);
            idToToken.put(tokenId, token);
        }

        vocabSize = vocab.size();
        Log.d(TAG, "Loaded " + vocabSize + " tokens from JSON");
    }

    private void createMinimalVocabulary() {
        Log.d(TAG, "Creating minimal vocabulary for BLIP");

        tokenToId.clear();
        idToToken.clear();

        // Add essential special tokens
        addToken("[PAD]", PAD_TOKEN);
        addToken("[UNK]", UNK_TOKEN);
        addToken("[CLS]", CLS_TOKEN);
        addToken("[SEP]", SEP_TOKEN);
        addToken("[MASK]", MASK_TOKEN);

        // Add common punctuation
        addToken(".", 1012);
        addToken(",", 1010);
        addToken("!", 999);
        addToken("?", 1029);

        // Add essential words for image captions
        String[] captionWords = {
                // Articles
                "a", "an", "the",

                // Common nouns for images
                "person", "people", "man", "woman", "child", "boy", "girl",
                "dog", "cat", "car", "truck", "bus", "bicycle", "motorcycle",
                "house", "building", "tree", "flower", "water", "sky", "road",
                "table", "chair", "book", "phone", "computer", "camera",
                "food", "apple", "pizza", "cake", "bottle", "cup",
                "ball", "toy", "game", "music", "movie",

                // Verbs
                "is", "are", "has", "have", "sit", "stand", "walk", "run",
                "eat", "drink", "read", "write", "play", "work", "drive",
                "hold", "carry", "wear", "look", "watch", "listen",

                // Adjectives
                "big", "small", "large", "little", "old", "new", "young",
                "red", "blue", "green", "yellow", "black", "white", "brown",
                "good", "nice", "beautiful", "happy", "sad", "fast", "slow",

                // Prepositions and conjunctions
                "in", "on", "at", "with", "and", "or", "but", "to", "for", "of",

                // Numbers
                "one", "two", "three", "many", "some", "few"
        };

        // Add essential words starting from ID 2000
        for (int i = 0; i < captionWords.length; i++) {
            addToken(captionWords[i], 2000 + i);
        }

        vocabSize = tokenToId.size();
        Log.d(TAG, "Created minimal vocabulary with " + vocabSize + " tokens");
    }

    private void addToken(String token, int id) {
        tokenToId.put(token, id);
        idToToken.put(id, token);
    }

    private void logSampleTokens() {
        // Verify special tokens
        Log.d(TAG, "Special tokens verification:");
        Log.d(TAG, "PAD (0): " + idToToken.get(PAD_TOKEN));
        Log.d(TAG, "UNK (100): " + idToToken.get(UNK_TOKEN));
        Log.d(TAG, "CLS/BOS (101): " + idToToken.get(CLS_TOKEN));
        Log.d(TAG, "SEP/EOS (102): " + idToToken.get(SEP_TOKEN));

        // Log some common words if they exist
        String[] checkWords = {"a", "the", "person", "image", "photo", "picture"};
        for (String word : checkWords) {
            Integer id = tokenToId.get(word);
            if (id != null) {
                Log.d(TAG, "Found word '" + word + "' with ID: " + id);
            }
        }
    }

    // Public API methods
    public long getBosToken() {
        return CLS_TOKEN;
    }

    public long getEosToken() {
        return SEP_TOKEN;
    }

    public long getPadToken() {
        return PAD_TOKEN;
    }

    public long getUnkToken() {
        return UNK_TOKEN;
    }

    public boolean isLoaded() {
        return isLoaded;
    }

    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * Basic tokenization - splits text into words and maps to IDs
     */
    public int[] encode(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new int[]{CLS_TOKEN, SEP_TOKEN};
        }

        // Simple tokenization for captions
        String cleanText = text.toLowerCase()
                .replaceAll("[^a-zA-Z0-9\\s.,!?]", " ")
                .replaceAll("\\s+", " ")
                .trim();

        List<Integer> tokens = new ArrayList<>();
        tokens.add(CLS_TOKEN); // Start token

        // Tokenize words
        String[] words = cleanText.split("\\s+");
        for (String word : words) {
            if (word.isEmpty()) continue;

            // Handle punctuation attached to words
            String[] parts = word.split("(?=[.,!?])|(?<=[.,!?])");
            for (String part : parts) {
                if (!part.isEmpty()) {
                    Integer tokenId = tokenToId.getOrDefault(part, UNK_TOKEN);
                    tokens.add(tokenId);
                }
            }
        }

        tokens.add(SEP_TOKEN); // End token

        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Decode token IDs back to readable text
     */
    public String decode(int[] tokenIds) {
        if (tokenIds == null || tokenIds.length == 0) {
            return "";
        }

        StringBuilder text = new StringBuilder();
        boolean first = true;

        for (int tokenId : tokenIds) {
            String token = idToToken.get(tokenId);
            if (token == null) {
                Log.w(TAG, "Unknown token ID: " + tokenId);
                continue;
            }

            // Skip special tokens in output
            if (tokenId == CLS_TOKEN || tokenId == SEP_TOKEN || tokenId == PAD_TOKEN) {
                continue;
            }

            // Handle unknown tokens
            if (tokenId == UNK_TOKEN) {
                continue; // Skip unknown tokens
            }

            // Handle punctuation
            if (token.matches("[.,!?:;]")) {
                text.append(token);
            } else {
                if (!first) {
                    text.append(" ");
                }

                // Handle BERT subword tokens (starting with ##)
                if (token.startsWith("##")) {
                    // Remove ## and append directly (subword)
                    text.append(token.substring(2));
                } else {
                    text.append(token);
                }
                first = false;
            }
        }

        String result = text.toString().trim();

        // Fallback if decoding fails
        if (result.isEmpty()) {
            return "A scene with various objects.";
        } else
            result = Character.toUpperCase(result.charAt(0)) + result.substring(1);

        // Ensure it ends with punctuation
        if (!result.matches(".*[.!?]$")) {
            result += ".";
        }

        return result;
    }

    /**
     * Get token string by ID
     */
    public String getTokenString(int tokenId) {
        return idToToken.getOrDefault(tokenId, "[UNK]");
    }

    /**
     * Get token ID by string
     */
    public long getTokenId(String token) {
        return tokenToId.getOrDefault(token, UNK_TOKEN);
    }

    /**
     * Check if a token ID is a special token
     */
    public boolean isSpecialToken(long tokenId) {
        return tokenId == PAD_TOKEN || tokenId == UNK_TOKEN ||
                tokenId == CLS_TOKEN || tokenId == SEP_TOKEN || tokenId == MASK_TOKEN;
    }
}