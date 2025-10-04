package com.visionassist.appspace.models.ttsengine;

import android.content.Context;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.Language;
import java.util.Locale;

public class TTSManager implements TextToSpeech.OnInitListener {
    private static final String TAG = "TTSManager";

    private TextToSpeech tts;
    private boolean isInitialized = false;

    public TTSManager(Context context) {
        // The second parameter 'this' ensures onInit is called upon completion
        tts = new TextToSpeech(context, this);
    }

    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {

            // Get the language code from the Kotlin configuration
            Language mainLanguage = AppConfig.mainLanguage;
            Locale locale = new Locale(mainLanguage.getCode());

            int result = tts.setLanguage(locale);

            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e(TAG, "Language " + mainLanguage.getCode() + " not supported by TTS engine.");
                // Fallback to default locale if main language is not supported
                tts.setLanguage(Locale.getDefault());
            }
            isInitialized = true;
            Log.d(TAG, "TTS Initialization successful. Language set to: " + locale.getDisplayLanguage());

        } else {
            Log.e(TAG, "TTS Initialization Failed! Status: " + status);
            isInitialized = false;
        }
    }

    public void speak(String text, float pitch, float speed) {
        if (!isInitialized || tts == null || text == null || text.isEmpty()) {
            Log.e(TAG, "TTS not initialized or text is empty.");
            return;
        }

        // 1. Set Pitch and Speed
        // Note: setPitch and setSpeechRate return failure codes if values are outside 0-2.0 range.
        tts.setPitch(pitch);
        tts.setSpeechRate(speed);

        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "TTS_VISON_ASSIST_UTTERANCE");
    }

    /**
     * Releases the resources used by the TTS engine. Call this in onDestroy() of your Activity.
     */
    public void shutdown() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
            isInitialized = false;
        }
    }
}