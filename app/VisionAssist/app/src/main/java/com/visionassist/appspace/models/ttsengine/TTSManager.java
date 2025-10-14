package com.visionassist.appspace.models.ttsengine;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.media.AudioAttributes;
import android.media.SoundPool;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;

import com.visionassist.appspace.R;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.Language;

import java.util.Locale;

public class TTSManager implements TextToSpeech.OnInitListener {
    private static final String TAG = "TTSManager";
    private static final String UTTERANCE_ID = "TTS_VISION_ASSIST";
    private static final String UTTERANCE_ID_REPEAT = "TTS_VISION_ASSIST_REPEAT";


    public TextToSpeech tts;
    private Context context;
    private Activity currentActivity;
    private int languageCheckAttempts = 0;
    private Language pendingLanguage = null;
    private boolean waitingForSettings = false;
    private boolean isInitialized = false;
    private boolean isDoneSpeaking = false;
    private Vibrator vibrator;
    private Handler repeatHandler = new Handler(Looper.getMainLooper());
    private Runnable repeatTimeoutRunnable;

    private String currentText = "";
    private float currentPitch = 1.0f;
    private long[] currentVibrationPattern = null;
    private boolean changeSpeed = false;
    private boolean isRepeating = false;

    private SoundPool soundPool;
    private int repeatCueSoundId = 0;
    private static final int REPEAT_CUE_RES_ID = R.raw.repeat_alert;
    private Handler ttsDelayHandler = new Handler(Looper.getMainLooper());

    public TTSManager(Context context) {
        this.context = context;
        this.vibrator = (Vibrator) context.getSystemService(Context.VIBRATOR_SERVICE);
        initSoundPool(context);
        tts = new TextToSpeech(context, this);
    }

    private void initSoundPool(Context context) {
        AudioAttributes audioAttributes = new AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_NOTIFICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build();
        soundPool = new SoundPool.Builder()
                .setMaxStreams(1)
                .setAudioAttributes(audioAttributes)
                .build();
        try {
            repeatCueSoundId = soundPool.load(context, REPEAT_CUE_RES_ID, 1);
            Log.d(TAG, "SoundPool loaded sound with ID: " + repeatCueSoundId);
        } catch (Exception e) {
            Log.e(TAG, "Error loading the sound resoruce");
            repeatCueSoundId = 0;
        }
    }

    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            tts.setLanguage(Locale.getDefault());
            isInitialized = true;
            setupUtteranceListener();
            Log.d(TAG, "TTS initialized. Language: " + tts.getVoice().getLocale().getLanguage());
        } else {
            Log.e(TAG, "TTS Initialization Failed! Status: " + status);
            isInitialized = false;
        }
    }

    /**
     * Changes the TTS language. Shows dialog if language data is not installed.
     * Call recheckPendingLanguage() in your Activity's onResume() after returning from settings.
     *
     * @param language Language object with code and country
     * @param activity Current activity (needed to show dialogs and open settings)
     */
    public void changeLanguage(Language language, Activity activity) {
        // Disable TTS during language change
        isInitialized = false;

        this.currentActivity = activity;
        this.languageCheckAttempts = 0;
        this.pendingLanguage = language;

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
                    .setMessage("The " + language.getName() + " language is not installed on your device." +
                            "Please download it from TTS settings." +
                            "After downloading, return to this app and the language will be automatically checked.")
                    .setCancelable(false)
                    .setPositiveButton("Settings", (dialog, which) -> {
                        waitingForSettings = true;
                        openTTSSettings();
                        // Don't check immediately - wait for onResume() callback
                    })
                    .setNegativeButton("Use default("+Locale.getDefault().getLanguage()+")", (dialog, which) -> {
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
     * <p>
     * Example in your Activity:
     *
     * @Override protected void onResume() {
     * super.onResume();
     * if (ttsManager != null) {
     * ttsManager.recheckPendingLanguage();
     * }
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
                                        ".Using default("+Locale.getDefault().getLanguage()+") language instead.")
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

    private void setupUtteranceListener() {
        tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
            @Override
            public void onStart(String utteranceId) {
                Log.d(TAG, "TTS started speaking: " + utteranceId);
            }

            @Override
            public void onDone(String utteranceId) {
                Log.d(TAG, "TTS finished speaking: " + utteranceId);

                if (utteranceId.equals(UTTERANCE_ID_REPEAT)) {
                    handleRepeatComplete();
                } else {
                    handleInitialSpeakingComplete();
                }
            }

            @Override
            public void onError(String utteranceId) {
                Log.e(TAG, "TTS error for utterance: " + utteranceId);
                handleSpeakingError();
            }
        });
    }

    private void handleInitialSpeakingComplete() {
        if (!isRepeating) {
            Log.d(TAG, "Initial speaking complete, no repeat requested");
            isDoneSpeaking = true;
        } else {
            Log.d(TAG, "Initial speaking complete, starting repeat delay timer");
            startRepeatDelay();
        }
    }

    private void handleRepeatComplete() {
        Log.d(TAG, "Repeat speaking complete, marking as done");
        cancelRepeatDelay();
        isDoneSpeaking = true;
        isRepeating = false;
    }

    private void handleSpeakingError() {
        Log.e(TAG, "TTS error occurred");
        cancelRepeatDelay();
        isDoneSpeaking = true;
        isRepeating = false;
    }

    private void startRepeatDelay() {
        Log.d(TAG, "Starting repeat delay for " + Constants.REPEAT_DELAY + "ms");

        repeatTimeoutRunnable = () -> {
            Log.d(TAG, "Repeat delay timeout reached, no volume button pressed");
            isDoneSpeaking = true;
            isRepeating = false;
        };

        repeatHandler.postDelayed(repeatTimeoutRunnable, Constants.REPEAT_DELAY);
    }

    private void cancelRepeatDelay() {
        if (repeatTimeoutRunnable != null) {
            repeatHandler.removeCallbacks(repeatTimeoutRunnable);
            repeatTimeoutRunnable = null;
        }
        ttsDelayHandler.removeCallbacksAndMessages(null);
    }

    public void onVolumeDownPressed() {
        if (isRepeating) {
            changeSpeed = !changeSpeed;
            Log.d(TAG, "Volume down pressed");

            cancelRepeatDelay();
            ttsDelayHandler.removeCallbacksAndMessages(null);

            if (tts.isSpeaking()) {
                Log.d(TAG, "Interrupting initial speech, speaking slower immediately");
                tts.stop();
                speakWithCustomSpeed(changeSpeed ? Constants.LOW_SPEECH_RATE : Constants.HIGH_SPEECH_RATE);
            } else {
                Log.d(TAG, "In delay period, speaking slower immediately");
                speakWithCustomSpeed(changeSpeed ? Constants.LOW_SPEECH_RATE : Constants.HIGH_SPEECH_RATE);
            }
        }
    }

    private void speakWithCustomSpeed(float speech_rate) {
        startSpeakingAction(currentText, currentPitch, speech_rate, currentVibrationPattern, UTTERANCE_ID_REPEAT);
    }

    private void startSpeakingAction(String text, float pitch, float speed, long[] vibrationPattern, String utteranceId) {
        if (vibrationPattern != null && vibrator != null) {
            if (Constants.API_LEVEL >= android.os.Build.VERSION_CODES.O) {
                VibrationEffect effect = VibrationEffect.createWaveform(vibrationPattern, -1);
                vibrator.vibrate(effect);
            } else {
                vibrator.vibrate(vibrationPattern, -1);
            }
            Log.d(TAG, "Vibrating phone");
        }

        tts.setPitch(pitch);
        tts.setSpeechRate(speed);

        Bundle params = new Bundle();
        int result = tts.speak(text, TextToSpeech.QUEUE_FLUSH, params, utteranceId);

        if (result == TextToSpeech.ERROR) {
            Log.e(TAG, "TTS speak() returned ERROR for " + utteranceId);
            isDoneSpeaking = true;
            isRepeating = false;
        } else {
            Log.d(TAG, "TTS speak() called successfully for " + utteranceId);
        }
    }

    /**
     * Main speak method with optional repeat functionality and vibration
     *
     * @param text             The text to speak
     * @param pitch            The pitch (1.0 = normal)
     * @param speed            The speed (1.0 = normal)
     * @param repeat           If true, allows user to press volume down for slower repeat
     * @param vibrationPattern Pattern for vibration (null for no vibration)
     */
    public void speak(String text, float pitch, float speed, boolean repeat, long[] vibrationPattern) {
        isDoneSpeaking = false;
        cancelRepeatDelay();

        if (!isInitialized || tts == null) {
            Log.e(TAG, "TTS not initialized");
            isDoneSpeaking = true;
            return;
        }

        if (text == null || text.isEmpty()) {
            Log.e(TAG, "Text is empty");
            isDoneSpeaking = true;
            return;
        }

        currentText = text;
        currentPitch = pitch;
        currentVibrationPattern = vibrationPattern;
        isRepeating = repeat;

        Log.d(TAG, "Speaking initiated. Repeat enabled: " + repeat);
        Log.d(TAG, "Text: " + text);

        if (repeat && soundPool != null && repeatCueSoundId != 0) {
            soundPool.play(repeatCueSoundId, 0.7f, 0.7f, 1, 0, 1.0f);
            ttsDelayHandler.postDelayed(() -> startSpeakingAction(text, pitch, speed, vibrationPattern, UTTERANCE_ID), 2000);
        } else {
            startSpeakingAction(text, pitch, speed, vibrationPattern, UTTERANCE_ID);
        }
    }

    public void shutdown() {
        cancelRepeatDelay();
        ttsDelayHandler.removeCallbacksAndMessages(null);
        if (tts != null) {
            tts.stop();
            tts.shutdown();
            isInitialized = false;
        }
        if (soundPool != null) {
            soundPool.release();
        }
        isDoneSpeaking = true;
        isRepeating = false;
    }

    public void stopSpeaking() {
        if (tts.isSpeaking()) {
            tts.stop();
            isDoneSpeaking = true;
            Log.e(TAG, "TTS was suddenly stopped by an external source");
        }
    }

    public boolean isReady() {
        return isInitialized;
    }

    public boolean isDoneSpeaking() {
        return isDoneSpeaking;
    }

    public Locale getCurrentLocale() {
        if (tts != null && tts.getVoice() != null) {
            return tts.getVoice().getLocale();
        }
        return Locale.getDefault();
    }
}