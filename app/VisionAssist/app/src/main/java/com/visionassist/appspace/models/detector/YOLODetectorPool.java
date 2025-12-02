package com.visionassist.appspace.models.detector;

import android.content.Context;
import android.util.Log;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.Constants;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class YOLODetectorPool {
    private static final String TAG = "YOLODetectorPool";

    private final Context context;

    // Model instance pools
    private final ConcurrentLinkedQueue<YOLODetector> nanoModelsAvailable;
    private final ConcurrentLinkedQueue<YOLODetector> accModelsAvailable;

    // All instances (for cleanup)
    private final ConcurrentLinkedQueue<YOLODetector> allNanoInstances;
    private final ConcurrentLinkedQueue<YOLODetector> allAccInstances;

    // Semaphores for controlling access
    private Semaphore nanoSemaphore;
    private Semaphore accSemaphore;

    // Pool configuration
    private int totalNanoInstances = 0;
    private int totalAccInstances = 0;

    // Statistics
    private final AtomicInteger activeNanoInferences = new AtomicInteger(0);
    private final AtomicInteger activeAccInferences = new AtomicInteger(0);
    private final AtomicInteger totalInferences = new AtomicInteger(0);
    private final AtomicInteger poolHits = new AtomicInteger(0);
    private final AtomicInteger poolMisses = new AtomicInteger(0);

    // Initialization state
    private volatile boolean isInitialized = false;
    private volatile boolean initializationFailed = false;
    private String failureReason = null;

    // Memory constants (in bytes)
    public static final long MB = 1_048_576L;
    private static final long THREAD_MEM_USED = 14 * MB;
    public long ACTUAL_THREAD_MEM_USED_MB;
    private static final int STD_NO_THREADS=2;

    public YOLODetectorPool(Context context) {
        this.context = context;
        this.nanoModelsAvailable = new ConcurrentLinkedQueue<>();
        this.accModelsAvailable = new ConcurrentLinkedQueue<>();
        this.allNanoInstances = new ConcurrentLinkedQueue<>();
        this.allAccInstances = new ConcurrentLinkedQueue<>();
        if (AppConfig.env_reports)
            this.ACTUAL_THREAD_MEM_USED_MB = THREAD_MEM_USED + 2 * MB;// + classifier mem needed
        else
            this.ACTUAL_THREAD_MEM_USED_MB = THREAD_MEM_USED;
    }

    public boolean initialize() {
        if (isInitialized) {
            Log.w(TAG, "Pool already initialized");
            return true;
        }

        if (initializationFailed) {
            Log.e(TAG, "Pool initialization previously failed: " + failureReason);
            return false;
        }

        long startTime = System.currentTimeMillis();

        try {
            // Step 1: Calculate available memory
            MemoryInfo memoryInfo = calculateAvailableMemory();
            logMemoryInfo(memoryInfo);

            // Step 3: Calculate optimal pool sizes
            PoolConfiguration config = calculateOptimalPoolSizes(
                    memoryInfo
            );
            logPoolConfiguration(config);

            // Step 4: Validate configuration
            if (!validateConfiguration(config)) {
                initializationFailed = true;
                failureReason = "Insufficient memory for even 1 detector instance";
                Log.e(TAG, "INITIALIZATION FAILED: " + failureReason);
                return false;
            }

            // Step 5: Create model instances
            boolean success = createModelInstances(config);

            if (!success) {
                initializationFailed = true;
                failureReason = "Failed to create model instances";
                cleanup();
                Log.e(TAG, "INITIALIZATION FAILED: " + failureReason);
                return false;
            }

            // Step 6: Initialize semaphores
            nanoSemaphore = new Semaphore(totalNanoInstances);
            accSemaphore = new Semaphore(totalAccInstances);

            isInitialized = true;
            long duration = System.currentTimeMillis() - startTime;

            Log.d(TAG, "POOL INITIALIZED SUCCESSFULLY in " + duration + " ms");

            return true;

        } catch (Exception e) {
            Log.e(TAG, "Critical error during pool initialization", e);
            initializationFailed = true;
            failureReason = "Exception: " + e.getMessage();
            cleanup();
            return false;
        }
    }

    private boolean createModelInstances(PoolConfiguration config) {
        Log.d(TAG, "Creating model instances...");

        // Create nano instances
        for (int i = 0; i < config.nanoInstances; i++) {
            YOLODetector detector = new YOLODetector(context);
            int result = detector.loadModel(Constants.YOLO_MODEL_DETECTOR_SPEED_FILE, "N_" + i);

            if (result != 0) {
                Log.e(TAG, "Failed to load nano instance " + i);
                return false;
            }

            nanoModelsAvailable.offer(detector);
            allNanoInstances.add(detector);
            totalNanoInstances++;

            Log.d(TAG, "✓ Nano instance " + i + " loaded");
        }

        // Create accuracy instances
        for (int i = 0; i < config.accInstances; i++) {
            YOLODetector detector = new YOLODetector(context);
            int result = detector.loadModel(Constants.YOLO_MODEL_DETECTOR_ACC_FILE, "S_" + i);

            if (result != 0) {
                Log.e(TAG, "Failed to load accuracy instance " + i);
                return false;
            }

            accModelsAvailable.offer(detector);
            allAccInstances.add(detector);
            totalAccInstances++;

            Log.d(TAG, "✓ Accuracy instance " + i + " loaded");
        }

        return true;
    }

    private MemoryInfo calculateAvailableMemory() {
        Runtime runtime = Runtime.getRuntime();

        long maxMemory = runtime.maxMemory();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;
        long availableMemory = maxMemory - usedMemory;

        // Subtract reserved memory
        long usableMemory = (long) (availableMemory - (availableMemory * 0.4f));

        return new MemoryInfo(
                maxMemory,
                totalMemory,
                freeMemory,
                usedMemory,
                availableMemory,
                usableMemory
        );
    }

    private PoolConfiguration calculateOptimalPoolSizes(
            MemoryInfo memory
    ) {
        // Calculate max instances that fit in memory
        long modelInstances = memory.usableMemory / ACTUAL_THREAD_MEM_USED_MB;

        if(modelInstances>=STD_NO_THREADS+2)
            return new PoolConfiguration(2, STD_NO_THREADS);
        else
        {
            if(modelInstances<2)
                return new PoolConfiguration(0,0);
            else {
                if(modelInstances<3)
                return new PoolConfiguration(2,1);
                else
                    return new PoolConfiguration(2,((int)modelInstances)-2);
            }
        }
    }

    private boolean validateConfiguration(PoolConfiguration config) {
        if (config.nanoInstances < 1 && config.accInstances < 1) {
            Log.e(TAG, "Cannot create any model instances!");
            return false;
        }

        return true;
    }

    public YOLODetector acquireNanoModel(int timeoutSeconds) {
        if (!isInitialized) {
            Log.e(TAG, "Pool not initialized! Cannot acquire nano model.");
            poolMisses.incrementAndGet();
            return null;
        }

        try {
            boolean acquired = nanoSemaphore.tryAcquire(timeoutSeconds, TimeUnit.SECONDS);

            if (!acquired) {
                Log.w(TAG, "Timeout waiting for nano model (waited " + timeoutSeconds + "s)");
                poolMisses.incrementAndGet();
                return null;
            }

            YOLODetector detector = nanoModelsAvailable.poll();

            if (detector == null) {
                Log.e(TAG, "Semaphore acquired but no nano model available!");
                nanoSemaphore.release();
                poolMisses.incrementAndGet();
                return null;
            }

            activeNanoInferences.incrementAndGet();
            totalInferences.incrementAndGet();
            poolHits.incrementAndGet();

            Log.d(TAG, "✓ Nano model acquired (active: " + activeNanoInferences.get() +
                    "/" + totalNanoInstances + ")");

            return detector;

        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted while acquiring nano model", e);
            Thread.currentThread().interrupt();
            poolMisses.incrementAndGet();
            return null;
        }
    }

    public YOLODetector acquireAccModel(int timeoutSeconds) {
        if (!isInitialized) {
            Log.e(TAG, "Pool not initialized! Cannot acquire accuracy model.");
            poolMisses.incrementAndGet();
            return null;
        }

        try {
            boolean acquired = accSemaphore.tryAcquire(timeoutSeconds, TimeUnit.SECONDS);

            if (!acquired) {
                Log.w(TAG, "Timeout waiting for accuracy model (waited " + timeoutSeconds + "s)");
                poolMisses.incrementAndGet();
                return null;
            }

            YOLODetector detector = accModelsAvailable.poll();

            if (detector == null) {
                Log.e(TAG, "Semaphore acquired but no accuracy model available!");
                accSemaphore.release();
                poolMisses.incrementAndGet();
                return null;
            }

            activeAccInferences.incrementAndGet();
            totalInferences.incrementAndGet();
            poolHits.incrementAndGet();

            Log.d(TAG, "✓ Accuracy model acquired (active: " + activeAccInferences.get() +
                    "/" + totalAccInstances + ")");

            return detector;

        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted while acquiring accuracy model", e);
            Thread.currentThread().interrupt();
            poolMisses.incrementAndGet();
            return null;
        }
    }

    public void releaseNanoModel(YOLODetector detector) {
        if (detector == null) {
            Log.w(TAG, "Attempted to release null nano model");
            return;
        }

        nanoModelsAvailable.offer(detector);
        activeNanoInferences.decrementAndGet();
        nanoSemaphore.release();

        Log.d(TAG, "✓ Nano model released (active: " + activeNanoInferences.get() +
                "/" + totalNanoInstances + ")");
    }

    public void releaseAccModel(YOLODetector detector) {
        if (detector == null) {
            Log.w(TAG, "Attempted to release null accuracy model");
            return;
        }

        accModelsAvailable.offer(detector);
        activeAccInferences.decrementAndGet();
        accSemaphore.release();

        Log.d(TAG, "✓ Accuracy model released (active: " + activeAccInferences.get() +
                "/" + totalAccInstances + ")");
    }

    public DetectorWrapper acquireDetector(boolean preferNano, int timeoutSeconds) {
        YOLODetector detector;
        DetectorType type;

        if (preferNano) {
            detector = acquireNanoModel(timeoutSeconds);
            type = DetectorType.NANO;

            if (detector == null) {
                Log.d(TAG, "Nano unavailable, falling back to accuracy");
                detector = acquireAccModel(timeoutSeconds);
                type = DetectorType.ACCURACY;
            }
        } else {
            detector = acquireAccModel(timeoutSeconds);
            type = DetectorType.ACCURACY;

            if (detector == null) {
                Log.d(TAG, "Accuracy unavailable, falling back to nano");
                detector = acquireNanoModel(timeoutSeconds);
                type = DetectorType.NANO;
            }
        }

        return detector != null ? new DetectorWrapper(detector, type) : null;
    }

    public void releaseDetector(DetectorWrapper wrapper) {
        if (wrapper == null) return;

        if (wrapper.type == DetectorType.NANO) {
            releaseNanoModel(wrapper.detector);
        } else {
            releaseAccModel(wrapper.detector);
        }
    }

    public void cleanup() {
        Log.d(TAG, "Cleaning up detector pool...");

        try {
            // Wait for all instances to be returned (up to 5 seconds)
            int waitCount = 0;
            int expectedTotal = totalNanoInstances + totalAccInstances;

            while ((nanoModelsAvailable.size() + accModelsAvailable.size()) < expectedTotal &&
                    waitCount < 50) {
                Thread.sleep(100);
                waitCount++;
            }

            int returned = nanoModelsAvailable.size() + accModelsAvailable.size();
            if (returned < expectedTotal) {
                Log.w(TAG, "Not all detectors returned before cleanup (" +
                        returned + "/" + expectedTotal + ")");
            }

            // Close all nano instances
            while (!allNanoInstances.isEmpty()) {
                YOLODetector detector = allNanoInstances.poll();
                if (detector != null) {
                    detector.close();
                }
            }

            // Close all accuracy instances
            while (!allAccInstances.isEmpty()) {
                YOLODetector detector = allAccInstances.poll();
                if (detector != null) {
                    detector.close();
                }
            }

            // Clear queues
            nanoModelsAvailable.clear();
            accModelsAvailable.clear();

            isInitialized = false;

            Log.d(TAG, "Pool cleaned up successfully");

        } catch (Exception e) {
            Log.e(TAG, "Error during cleanup", e);
        }
    }

    private void logMemoryInfo(MemoryInfo info) {
        Log.d(TAG, "Memory Analysis:");
        Log.d(TAG, "  Max memory:       " + (info.maxMemory / MB) + " MB");
        Log.d(TAG, "  Total allocated:  " + (info.totalMemory / MB) + " MB");
        Log.d(TAG, "  Currently used:   " + (info.usedMemory / MB) + " MB");
        Log.d(TAG, "  Available:        " + (info.availableMemory / MB) + " MB");
        Log.d(TAG, "  Usable for pool:  " + (info.usableMemory / MB) + " MB");
    }

    private void logPoolConfiguration(PoolConfiguration config) {
        Log.d(TAG, "Pool Configuration:");
        Log.d(TAG, "  Nano instances:     " + config.nanoInstances);
        Log.d(TAG, "  Accuracy instances: " + config.accInstances);
    }

    private record MemoryInfo(long maxMemory, long totalMemory, long freeMemory, long usedMemory,
                              long availableMemory, long usableMemory) {
    }

    private record PoolConfiguration(int nanoInstances, int accInstances) {
    }

    public enum DetectorType {
        NANO,
        ACCURACY
    }

    public static class DetectorWrapper {
        public final YOLODetector detector;
        public final DetectorType type;

        public DetectorWrapper(YOLODetector detector, DetectorType type) {
            this.detector = detector;
            this.type = type;
        }
    }
}