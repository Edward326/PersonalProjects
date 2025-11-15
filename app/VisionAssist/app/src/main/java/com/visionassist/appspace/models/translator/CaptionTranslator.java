package com.visionassist.appspace.models.translator;

import android.content.Context;
import android.util.Log;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.common.model.DownloadConditions;
import com.google.mlkit.common.model.RemoteModelManager;
import com.google.mlkit.nl.translate.TranslateLanguage;
import com.google.mlkit.nl.translate.TranslateRemoteModel;
import com.google.mlkit.nl.translate.Translation;
import com.google.mlkit.nl.translate.TranslatorOptions;

import java.util.concurrent.ExecutionException;

import com.google.mlkit.nl.translate.Translator;

public class CaptionTranslator {
    private static final String TAG = "Translator";

    private Translator translator;
    private Context context;
    private boolean isModelDownloaded = false;

    public CaptionTranslator(Context context) {
        this.context = context;
    }

    public int initializeTranslator() {
        try {
            Log.d(TAG, "Initializing English → Romanian translator...");

            // Create translator options (English to Romanian)
            TranslatorOptions options = new TranslatorOptions.Builder()
                    .setSourceLanguage(TranslateLanguage.ENGLISH)
                    .setTargetLanguage(TranslateLanguage.ROMANIAN)
                    .build();

            // Create translator instance
            translator = Translation.getClient(options);

            Log.d(TAG, "Translator instance created");

            // Check if model is already downloaded
            RemoteModelManager modelManager = RemoteModelManager.getInstance();
            TranslateRemoteModel roModel = new TranslateRemoteModel.Builder(TranslateLanguage.ROMANIAN).build();

            Task<Boolean> isModelDownloadedTask = modelManager.isModelDownloaded(roModel);

            try {
                // Wait for the check to complete (synchronous)
                Boolean modelExists = Tasks.await(isModelDownloadedTask);

                if (modelExists != null && modelExists) {
                    Log.d(TAG, "Romanian translation model already downloaded");
                    isModelDownloaded = true;
                    return 0;
                } else {
                    Log.d(TAG, "Romanian translation model not found, downloading...");
                    return downloadModel();
                }

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error checking if model is downloaded", e);
                // If check fails, try to download anyway
                return downloadModel();
            }

        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize translator", e);
            return -1;
        }
    }

    private int downloadModel() {
        try {
            // Set download conditions (require WiFi to avoid data charges)
            DownloadConditions conditions = new DownloadConditions.Builder()
                    //.requireWifi()
                    .build();

            // Download the model
            Task<Void> downloadTask = translator.downloadModelIfNeeded(conditions);

            try {
                // Wait for download to complete (synchronous)
                Tasks.await(downloadTask);

                if (downloadTask.isSuccessful()) {
                    Log.d(TAG, "Translation model downloaded successfully");
                    isModelDownloaded = true;
                    return 0;
                } else {
                    Log.e(TAG, "Failed to download translation model");
                    Exception exception = downloadTask.getException();
                    if (exception != null) {
                        Log.e(TAG, "Download error: " + exception.getMessage(), exception);
                    }
                    isModelDownloaded = false;
                    return -1;
                }

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error during model download", e);
                isModelDownloaded = false;
                return -1;
            }

        } catch (Exception e) {
            Log.e(TAG, "Failed to download model", e);
            isModelDownloaded = false;
            return -1;
        }
    }

    public String translate(String englishText) {
        if (translator == null) {
            Log.e(TAG, "Translator not initialized");
            return null;
        }

        if (!isModelDownloaded) {
            Log.e(TAG, "Translation model not downloaded");
            return null;
        }

        if (englishText == null || englishText.trim().isEmpty()) {
            Log.w(TAG, "Empty input text");
            return null;
        }

        try {
            Log.d(TAG, "Translating:\n\"" + englishText + "\"");

            // Translate (synchronous)
            Task<String> translationTask = translator.translate(englishText);

            try {
                // Wait for translation to complete
                String romanianText = Tasks.await(translationTask);

                if (translationTask.isSuccessful() && romanianText != null) {
                    Log.d(TAG, "Translation successful\n" +
                            "EN: " + englishText +
                            "\nRO: " + romanianText);
                    return romanianText;
                } else {
                    Log.e(TAG, "❌ Translation failed");
                    Exception exception = translationTask.getException();
                    if (exception != null) {
                        Log.e(TAG, "Translation error: " + exception.getMessage(), exception);
                    }
                    return null;
                }

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error during translation", e);
                return null;
            }

        } catch (Exception e) {
            Log.e(TAG, "Unexpected error during translation", e);
            return null;
        }
    }

    /**
     * Close the translator and free resources
     */
    public void close() {
        if (translator != null) {
            translator.close();
            translator = null;
            Log.d(TAG, "Translator closed");
        }
    }
}