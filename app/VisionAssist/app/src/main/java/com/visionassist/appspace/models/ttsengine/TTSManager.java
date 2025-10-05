package com.visionassist.appspace.models.ttsengine;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.speech.tts.TextToSpeech;
import android.util.Log;

import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.Language;
import java.util.Locale;

public class TTSManager implements TextToSpeech.OnInitListener {
    private static final String TAG = "TTSManager";

    public TextToSpeech tts;
    private boolean isInitialized = false;
    private Context context;
    private Activity currentActivity;
    private int languageCheckAttempts = 0;
    private Language pendingLanguage = null;
    private boolean waitingForSettings = false;

    public TTSManager(Context context) {
        this.context = context.getApplicationContext();
        tts = new TextToSpeech(context, this);
    }

    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            tts.setLanguage(Locale.getDefault());
            isInitialized = true;
            Log.d(TAG, "TTS initialized. Language: " + tts.getVoice().getLocale().getLanguage());
        } else {
            Log.e(TAG, "TTS Initialization Failed! Status: " + status);
            isInitialized = false;
        }
    }

    /**
     * Changes the TTS language. Shows dialog if language data is not installed.
     * Call recheckPendingLanguage() in your Activity's onResume() after returning from settings.
     * @param language Language object with code and country
     * @param activity Current activity (needed to show dialogs and open settings)
     */
    public void changeLanguage(Language language, Activity activity) {
        this.currentActivity = activity;
        this.languageCheckAttempts = 0;
        this.pendingLanguage = language;

        // Disable TTS during language change
        isInitialized = false;

        // Attempt to change language
        attemptLanguageChange(language);
    }

    private void attemptLanguageChange(Language language) {
        if (tts == null) {
            Log.e(TAG, "TTS is null, cannot change language");
            return;
        }

        // Create locale from language data
        Locale locale = new Locale(language.getCode(), language.getCountry());

        // Check if language is available
        int result = tts.isLanguageAvailable(locale);

        if (result == TextToSpeech.LANG_AVAILABLE ||
                result == TextToSpeech.LANG_COUNTRY_AVAILABLE ||
                result == TextToSpeech.LANG_COUNTRY_VAR_AVAILABLE) {

            // Language is available, set it
            tts.setLanguage(locale);
            isInitialized = true;
            pendingLanguage = null;
            waitingForSettings = false;
            Log.d(TAG, "Language changed successfully to: " + language.getName());

        } else if (result == TextToSpeech.LANG_MISSING_DATA ||
                result == TextToSpeech.LANG_NOT_SUPPORTED) {

            // Language data is missing
            Log.w(TAG, "Language data missing for: " + language.getName());
            showLanguageInstallDialog(language);
        }
    }

    private void showLanguageInstallDialog(Language language) {
        if (currentActivity == null || currentActivity.isFinishing()) {
            Log.e(TAG, "Cannot show dialog, activity is null or finishing");
            // Fallback to default language
            tts.setLanguage(Locale.getDefault());
            isInitialized = true;
            pendingLanguage = null;
            return;
        }

        currentActivity.runOnUiThread(() -> {
            new AlertDialog.Builder(currentActivity)
                    .setTitle("Language Data Required")
                    .setMessage("The " + language.getName() + " language is not installed on your device. " +
                            "Please download it from TTS settings.\n\n" +
                            "After downloading, return to this app and the language will be automatically checked.")
                    .setCancelable(false)
                    .setPositiveButton("Open Settings", (dialog, which) -> {
                        waitingForSettings = true;
                        openTTSSettings();
                        // Don't check immediately - wait for onResume() callback
                    })
                    .setNegativeButton("Use Default", (dialog, which) -> {
                        // User chose to use default language
                        tts.setLanguage(Locale.getDefault());
                        isInitialized = true;
                        pendingLanguage = null;
                        waitingForSettings = false;
                        Log.d(TAG, "User chose default language");
                    })
                    .show();
        });
    }

    /**
     * Call this method in your Activity's onResume() to check if language was installed.
     * This allows any amount of time for the user to download language data.
     *
     * Example in your Activity:
     * @Override
     * protected void onResume() {
     *     super.onResume();
     *     if (ttsManager != null) {
     *         ttsManager.recheckPendingLanguage();
     *     }
     * }
     */
    public void recheckPendingLanguage() {
        if (!waitingForSettings || pendingLanguage == null) {
            return;
        }

        waitingForSettings = false;
        Log.d(TAG, "Rechecking language after returning from settings...");

        Locale locale = new Locale(pendingLanguage.getCode(), pendingLanguage.getCountry());
        int result = tts.isLanguageAvailable(locale);

        if (result == TextToSpeech.LANG_AVAILABLE ||
                result == TextToSpeech.LANG_COUNTRY_AVAILABLE ||
                result == TextToSpeech.LANG_COUNTRY_VAR_AVAILABLE) {

            // Success! Language is now available
            tts.setLanguage(locale);
            isInitialized = true;
            Log.d(TAG, "Language successfully installed and set: " + pendingLanguage.getName());

            // Show success message
            if (currentActivity != null && !currentActivity.isFinishing()) {
                currentActivity.runOnUiThread(() -> {
                    new AlertDialog.Builder(currentActivity)
                            .setTitle("Success")
                            .setMessage(pendingLanguage.getName() + " language has been set successfully.")
                            .setPositiveButton("OK", null)
                            .show();
                });
            }

            pendingLanguage = null;
            languageCheckAttempts = 0;

        } else {
            // Still not available
            languageCheckAttempts++;
            if (languageCheckAttempts < Constants.MAX_LANGUAGE_CHECK_ATTEMPTS) {
                Log.w(TAG, "Language still not available. Attempt " + languageCheckAttempts);
                showLanguageInstallDialog(pendingLanguage);
            } else {
                // Max attempts reached, use default
                Log.e(TAG, "Max attempts reached. Using default language.");
                if (currentActivity != null && !currentActivity.isFinishing()) {
                    currentActivity.runOnUiThread(() -> {
                        new AlertDialog.Builder(currentActivity)
                                .setTitle("Language Unavailable")
                                .setMessage("Could not install " + pendingLanguage.getName() +
                                        ". Using default language instead.")
                                .setPositiveButton("OK", null)
                                .show();
                    });
                }
                tts.setLanguage(Locale.getDefault());
                isInitialized = true;
                pendingLanguage = null;
                languageCheckAttempts = 0;
            }
        }
    }

    /**
     * Opens the TTS settings page
     */
    private void openTTSSettings() {
        try {
            Intent intent = new Intent();
            intent.setAction("com.android.settings.TTS_SETTINGS");
            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            context.startActivity(intent);
        } catch (Exception e) {
            Log.e(TAG, "Could not open TTS settings", e);
        }
    }

    public void speak(String text, float pitch, float speed) {
        if (!isInitialized || tts == null) {
            Log.e(TAG, "TTS not initialized");
            return;
        }

        if (text == null || text.isEmpty()) {
            Log.e(TAG, "Text is empty");
            return;
        }

        Log.d(TAG, "Attempting to speak: " + text);
        Log.d(TAG, "Current TTS language: " + tts.getVoice().getLocale().getLanguage());
        Log.d(TAG, "Pitch: " + pitch + ", Speed: " + speed);

        tts.setPitch(pitch);
        tts.setSpeechRate(speed);

        int result = tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "TTS_VISION_ASSIST");

        if (result == TextToSpeech.ERROR) {
            Log.e(TAG, "TTS speak() returned ERROR");
        } else {
            Log.d(TAG, "TTS speak() called successfully");
        }
    }

    public void shutdown() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
            isInitialized = false;
        }
        pendingLanguage = null;
        waitingForSettings = false;
    }

    public boolean isReady() {
        return isInitialized;
    }

    public Locale getCurrentLocale() {
        if (tts != null && tts.getVoice() != null) {
            return tts.getVoice().getLocale();
        }
        return Locale.getDefault();
    }
}