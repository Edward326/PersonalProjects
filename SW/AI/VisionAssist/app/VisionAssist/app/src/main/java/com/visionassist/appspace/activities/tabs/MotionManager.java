package com.visionassist.appspace.activities.tabs;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.util.Log;

import java.util.Arrays;

public class MotionManager implements SensorEventListener {
    private static final String TAG = "MotionManager";

    private SensorManager sensorManager;
    private Sensor accelerometer;
    private Sensor gyroscope;
    private boolean isMonitoring = false;

    // For linear motion detection
    private static final int ACC_WINDOW_SIZE = 10;  // 0.5 seconds at 20Hz
    private float[] recentAccelerations = new float[ACC_WINDOW_SIZE];
    private int accIndex = 0;

    // For rotation detection
    private static final int GYRO_WINDOW_SIZE = 10;
    private float[] recentRotations = new float[GYRO_WINDOW_SIZE];
    private int gyroIndex = 0;

    // Gravity filter for better accuracy
    private float[] gravity = new float[3];
    private static final float ALPHA = 0.8f;  // Low-pass filter constant

    private Handler callbackHandler;
    private Runnable callbackRunnable;

    public MotionManager(Context context, Handler handler, Runnable runnable) {
        this.sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        this.accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        this.gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        this.callbackHandler = handler;
        this.callbackRunnable = runnable;

        // Initialize gravity
        gravity[0] = 0;
        gravity[1] = 0;
        gravity[2] = SensorManager.GRAVITY_EARTH;
    }

    public void startMonitoring() {
        if (isMonitoring) return;

        isMonitoring = true;

        // Reset arrays
        Arrays.fill(recentAccelerations, 0f);
        Arrays.fill(recentRotations, 0f);
        accIndex = 0;
        gyroIndex = 0;

        // Register both sensors (200ms = 5Hz)
        sensorManager.registerListener(this, accelerometer, 200_000);
        if (gyroscope != null) {
            sensorManager.registerListener(this, gyroscope, 200_000);
        } else {
            Log.w(TAG, "Gyroscope not available - rotation detection disabled");
        }
    }

    public void stopMonitoring() {
        if (!isMonitoring) return;
        sensorManager.unregisterListener(this);
        isMonitoring = false;
    }

    /**
     * Returns combined motion speed:
     * - Linear motion (walking/moving phone through space)
     * - Rotational motion (rotating/scanning with phone)
     */
    public float getSpeed() {
        float linearSpeed = getLinearSpeed();
        float rotationSpeed = getRotationSpeed();

        // Combine both: use the maximum of the two
        // This way, either walking OR rotating will trigger fast model

        return Math.max(linearSpeed, rotationSpeed * 2.0f);
    }

    /**
     * Get linear motion speed (walking/moving through space)
     */
    public float getLinearSpeed() {
        float sum = 0f;
        for (float acc : recentAccelerations) {
            sum += acc;
        }
        return sum / ACC_WINDOW_SIZE;
    }

    /**
     * Get rotation speed (how fast user is rotating/scanning the phone)
     */
    public float getRotationSpeed() {
        float sum = 0f;
        for (float rot : recentRotations) {
            sum += rot;
        }
        return sum / GYRO_WINDOW_SIZE;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            handleAccelerometer(event);
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            handleGyroscope(event);
        }
    }

    private void handleAccelerometer(SensorEvent event) {
        // Apply low-pass filter to isolate gravity
        gravity[0] = ALPHA * gravity[0] + (1 - ALPHA) * event.values[0];
        gravity[1] = ALPHA * gravity[1] + (1 - ALPHA) * event.values[1];
        gravity[2] = ALPHA * gravity[2] + (1 - ALPHA) * event.values[2];

        // Remove gravity to get linear acceleration
        float linearX = event.values[0] - gravity[0];
        float linearY = event.values[1] - gravity[1];
        float linearZ = event.values[2] - gravity[2];

        // Calculate magnitude of linear acceleration
        float linearAcc = (float) Math.sqrt(
                linearX * linearX +
                        linearY * linearY +
                        linearZ * linearZ
        );

        // Store in rolling window
        recentAccelerations[accIndex] = linearAcc;
        accIndex = (accIndex + 1) % ACC_WINDOW_SIZE;

        // Notify activity
        notifyActivity();
    }

    private void handleGyroscope(SensorEvent event) {
        // Gyroscope gives rotation rate in rad/s around x, y, z axes
        float rotX = event.values[0];  // Rotation around X (pitch)
        float rotY = event.values[1];  // Rotation around Y (roll)
        float rotZ = event.values[2];  // Rotation around Z (yaw)

        // Calculate magnitude of rotation (how fast rotating in any direction)
        float rotationRate = (float) Math.sqrt(
                rotX * rotX +
                        rotY * rotY +
                        rotZ * rotZ
        );

        // Store in rolling window
        recentRotations[gyroIndex] = rotationRate;
        gyroIndex = (gyroIndex + 1) % GYRO_WINDOW_SIZE;

        // Notify activity
        notifyActivity();
    }

    private void notifyActivity() {
        callbackHandler.post(() -> callbackRunnable.run());
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}
}