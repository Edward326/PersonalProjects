package com.visionassist.appspace.activities.tabs;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.util.Log;

import java.util.Arrays;

public class LightManager implements SensorEventListener {
    private static final String TAG = "LightManager";

    private SensorManager sensorManager;
    private Sensor lightSensor;
    private boolean isMonitoring = false;

    // For ambient light smoothing (prevent rapid changes)
    private static final int LIGHT_WINDOW_SIZE = 5;  // Average over 5 readings (~1 second at 5Hz)
    private float[] recentLuxValues = new float[LIGHT_WINDOW_SIZE];
    private int luxIndex = 0;
    private boolean windowFilled = false;

    // Thresholds with hysteresis to prevent flickering
    private static final float LUX_THRESHOLD_DARK = 10f;    // Turn flashlight ON below this

    // Current state
    private float currentLux = 0f;
    private boolean isDarkState = false;  // Track current dark/bright state

    // Callback
    private Handler callbackHandler;
    private Runnable callbackRunnable;

    public LightManager(Context context, Handler handler, Runnable runnable) {
        this.sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        this.lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT);
        this.callbackHandler = handler;
        this.callbackRunnable = runnable;

        if (lightSensor == null) {
            Log.w(TAG, "Ambient light sensor not available on this device!");
        }
    }

    public void startMonitoring() {
        if (isMonitoring) {
            Log.w(TAG, "Already monitoring");
            return;
        }

        if (lightSensor == null) {
            Log.e(TAG, "Cannot start monitoring - light sensor not available");
            return;
        }

        isMonitoring = true;

        // Reset arrays
        Arrays.fill(recentLuxValues, 0f);
        luxIndex = 0;
        windowFilled = false;
        currentLux = 0f;

        // Register sensor (SENSOR_DELAY_NORMAL = ~200ms = 5Hz)
        sensorManager.registerListener(this, lightSensor, 500_000);
    }

    public void stopMonitoring() {
        if (!isMonitoring) {
            return;
        }

        sensorManager.unregisterListener(this);
        isMonitoring = false;

        Log.d(TAG, "🌡️ Light monitoring stopped");
    }

    public boolean isDark() {
        // Hysteresis: Different thresholds for turning ON vs OFF
        return isDarkState;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_LIGHT) {
            handleLightSensor(event);
        }
    }

    private void handleLightSensor(SensorEvent event) {
        // Get raw lux value
        float rawLux = event.values[0];

        // Store in rolling window for smoothing
        recentLuxValues[luxIndex] = rawLux;
        luxIndex = (luxIndex + 1) % LIGHT_WINDOW_SIZE;

        // Mark window as filled after first complete cycle
        if (luxIndex == 0 && !windowFilled) {
            windowFilled = true;
        }

        // Calculate smoothed average
        float sum = 0f;
        int count = windowFilled ? LIGHT_WINDOW_SIZE : luxIndex;
        for (int i = 0; i < count; i++) {
            sum += recentLuxValues[i];
        }
        currentLux = sum / count;

        isDarkState = currentLux <= LUX_THRESHOLD_DARK;

        notifyActivity();
    }

    private void notifyActivity() {
        callbackHandler.post(() -> callbackRunnable.run());
    }


    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Light sensor accuracy changes are rare and don't affect functionality
        Log.d(TAG, "Sensor accuracy changed: " + accuracy);
    }
}