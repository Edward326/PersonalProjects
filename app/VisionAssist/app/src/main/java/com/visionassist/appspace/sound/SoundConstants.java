package com.visionassist.appspace.sound;

import com.visionassist.appspace.R;

public class SoundConstants {

    // ==================== SOUND RESOURCE IDs ====================
    // TTS sounds
    public static final int TTS_REPEAT_ID = R.raw.repeat_alert;

    // HomeActivity sounds
    public static final int STT_ERROR_ID = R.raw.stt_error;
    public static final int STT_SPEAK_OPEN_ID = R.raw.stt_speak_opened;
    public static final int STT_SPEAK_CLOSE_ID = R.raw.stt_speak_closed;
    public static final int OPEN_UP_ID = R.raw.home_startup;

    // ==================== SOUND DURATIONS (ms) ====================
    // Navigation durations
    public static final int DEFAULT_MS = 2000;
    public static final int TTS_REPEAT_MS = 2000;
    public static final int STT_ERROR_MS = 1000;
    public static final int STT_SPEAK_OPEN_MS = 1000;
    public static final int STT_SPEAK_CLOSE_MS = 1000;
    public static final int OPEN_UP_MS = 3000;


    // ==================== SOUND TO DURATION MAPPING ====================
    public static int getDuration(int soundResId) {
        if (soundResId == TTS_REPEAT_ID) {
            return TTS_REPEAT_MS;
        }
        if (soundResId == STT_ERROR_ID) {
            return STT_ERROR_MS;
        }
        if (soundResId == STT_SPEAK_OPEN_ID) {
            return STT_SPEAK_OPEN_MS;
        }
        if (soundResId == STT_SPEAK_CLOSE_ID) {
            return STT_SPEAK_CLOSE_MS;
        }

        // Default duration if sound not found
        return DEFAULT_MS;
    }
}