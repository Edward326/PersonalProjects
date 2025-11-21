package com.visionassist.appspace.sound;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.SoundPool;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.util.HashMap;
import java.util.Map;

public class SoundManager {
    private static final String TAG = "SoundManager";

    private Context context;
    private SoundPool soundPool;
    private Handler handler;
    // Map of resource IDs to loaded sound IDs
    private Map<Integer, Integer> soundMap;

    // Playback settings
    private static final float DEFAULT_RATE = 1.0f;
    private static final int DEFAULT_PRIORITY = 1;
    private static final int NO_LOOP = 0;
    private static final int MAX_STREAMS = 1;

    public SoundManager(Context context) {
        this.context = context.getApplicationContext();
        this.handler = new Handler(Looper.getMainLooper());
        this.soundMap = new HashMap<>();

        Log.d(TAG, "Initializing SoundManager...");
        initializeSoundPool();
        loadSounds();
    }

    private void initializeSoundPool() {
        try {
            AudioAttributes audioAttributes = new AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_NOTIFICATION)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build();

            soundPool = new SoundPool.Builder()
                    .setMaxStreams(MAX_STREAMS)
                    .setAudioAttributes(audioAttributes)
                    .build();

            Log.d(TAG, "SoundPool initialized successfully");
        } catch (Exception e) {
            soundPool = null;
            Log.e(TAG, "Failed to initialize SoundPool", e);
        }
    }

    private void loadSounds() {
        if (soundPool == null) {
            Log.e(TAG, "Cannot load sounds - SoundPool is null");
            return;
        }
        // Navigation sounds
        loadSound(SoundConstants.TTS_REPEAT_ID);
        loadSound(SoundConstants.STT_ERROR_ID);
        loadSound(SoundConstants.STT_SPEAK_OPEN_ID);
        loadSound(SoundConstants.STT_SPEAK_CLOSE_ID);
        loadSound(SoundConstants.OPEN_UP_ID);
        loadSound(SoundConstants.FIND_MY_OBJECT_DONE_ID);

        Log.d(TAG, "All sounds loaded successfully (" + soundMap.size() + " sounds)");
    }

    private void loadSound(int resourceId) {
        try {
            int soundId = soundPool.load(context, resourceId, DEFAULT_PRIORITY);
            soundMap.put(resourceId, soundId);
        } catch (Exception e) {
            soundMap = null;
        }
    }

    public void play(int soundResourceId, float leftVolume, float rightVolume, Runnable onComplete) {
        // If SoundPool is null, execute callback immediately
        try {
            if (soundPool == null || soundMap == null) {
                Log.w(TAG, "SoundPool not initialized, executing callback immediately");
                if (onComplete != null) {
                    handler.post(onComplete);
                }
                return;
            }

            // Get the loaded sound ID
            int soundId = soundMap.get(soundResourceId);

            // Play the sound
            int streamId = soundPool.play(
                    soundId,           // Sound ID
                    leftVolume,    // Left volume
                    rightVolume,    // Right volume
                    DEFAULT_PRIORITY,  // Priority
                    NO_LOOP,          // Loop (0 = no loop)
                    DEFAULT_RATE      // Playback rate
            );

            if (streamId == 0) {
                Log.e(TAG, "Failed to play sound (ID: " + soundResourceId + ")");
                if (onComplete != null) {
                    handler.post(onComplete);
                }
                return;
            }

            Log.d(TAG, "Playing sound: " + soundResourceId + " (stream: " + streamId + ")");
            // Get sound duration
            int duration = SoundConstants.getDuration(soundResourceId);

            // Post callback after sound duration
            if (onComplete != null) {
                handler.postDelayed(onComplete, duration);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error playing sound: " + soundResourceId, e);
            if (onComplete != null) {
                handler.post(onComplete);
            }
        }
    }

    public void release() {
        Log.d(TAG, "Releasing SoundManager resources...");

        // Cancel all pending callbacks
        if (handler != null) {
            handler.removeCallbacksAndMessages(null);
        }

        // Release SoundPool
        if (soundPool != null) {
            try {
                soundPool.release();
                soundPool = null;
                Log.d(TAG, "SoundPool released");
            } catch (Exception e) {
                Log.e(TAG, "Error releasing SoundPool", e);
            }
        }

        // Clear sound map
        if (soundMap != null) {
            soundMap.clear();
        }
        Log.d(TAG, "SoundManager released successfully");
    }

    public void releaseCallback() {
        if (soundPool != null)
            soundPool.autoPause();
        if (handler != null) {
            handler.removeCallbacksAndMessages(null);
        }
    }
}
