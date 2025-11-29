package com.visionassist.appspace.models.detector;

import android.content.Context;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.util.Log;
import android.util.Size;
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
    private static final long MB = 1_048_576L;
    private static final long MODEL_NANO_SIZE = 13 * MB;
    private static final long MODEL_ACC_SIZE = 43 * MB;
    // Reserve for app operations
    // Classifier, Captioner, etc.

    public YOLODetectorPool(Context context) {
        this.context = context;
        this.nanoModelsAvailable = new ConcurrentLinkedQueue<>();
        this.accModelsAvailable = new ConcurrentLinkedQueue<>();
        this.allNanoInstances = new ConcurrentLinkedQueue<>();
        this.allAccInstances = new ConcurrentLinkedQueue<>();
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

            // Step 2: Calculate per-thread memory requirements
            long perThreadMemory = calculatePerThreadMemory();
            Log.d(TAG, "Per-thread memory requirement: " + (perThreadMemory / MB) + " MB");

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

    private long calculatePerThreadMemory() {
        Size cameraResolution = getBackCameraResolution();

        int cameraWidth = cameraResolution.getWidth();
        int cameraHeight = cameraResolution.getHeight();

        Log.d(TAG, "Camera resolution: " + cameraWidth + "x" + cameraHeight);

        // Bitmap size (ARGB_8888 = 4 bytes per pixel)
        long bitmapSize = (long) cameraWidth * cameraHeight * 4;

        // Input tensor (640x640x3 float32)
        long inputTensorSize = 640L * 640L * 3L * 4L;

        // Output tensor (8400x85 float32)
        long outputTensorSize = 8400L * 85L * 4L;

        // Preprocessing overhead (resize buffer + normalization)
        long preprocessOverhead = 640L * 640L * 4L + 500_000L;

        // Postprocessing overhead (NMS arrays)
        long postprocessOverhead = 1_000_000L;

        // Thread stack
        long threadStack = MB;

        long total = bitmapSize + inputTensorSize + outputTensorSize +
                preprocessOverhead + postprocessOverhead + threadStack;

        Log.d(TAG, "Per-thread breakdown:");
        Log.d(TAG, "  Bitmap (" + cameraWidth + "x" + cameraHeight + "): " + (bitmapSize / MB) + " MB");
        Log.d(TAG, "  Input tensor: " + (inputTensorSize / MB) + " MB");
        Log.d(TAG, "  Output tensor: " + (outputTensorSize / MB) + " MB");
        Log.d(TAG, "  Overhead: " + ((preprocessOverhead + postprocessOverhead + threadStack) / MB) + " MB");
        Log.d(TAG, "  TOTAL per thread: " + (total / MB) + " MB");

        return total;
    }

    private Size getBackCameraResolution() {
        try {
            CameraManager cameraManager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);

            if (cameraManager == null) {
                Log.w(TAG, "CameraManager not available, using default resolution");
                return new Size(1920, 1080);  // Default fallback
            }

            // Get all camera IDs
            String[] cameraIds = cameraManager.getCameraIdList();

            // Find back camera
            for (String cameraId : cameraIds) {
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);

                // Check if this is back camera
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) {

                    // Get stream configuration map
                    StreamConfigurationMap map = characteristics.get(
                            CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
                    );

                    if (map == null) {
                        Log.w(TAG, "Stream configuration map not available");
                        continue;
                    }

                    // Get available output sizes for ImageCapture
                    // We want JPEG sizes since that's what ImageCapture uses
                    Size[] outputSizes = map.getOutputSizes(android.graphics.ImageFormat.JPEG);

                    if (outputSizes == null || outputSizes.length == 0) {
                        Log.w(TAG, "No output sizes available");
                        continue;
                    }

                    // Find the resolution that matches what your app requests
                    // In BlindFindMyObjectActivity, you use setTargetResolution(Size(screenWidth, screenHeight))
                    // CameraX will pick the closest available resolution

                    // Strategy: Get the largest resolution that's reasonable (not 4K which is overkill)
                    Size bestSize = findBestCameraResolution(outputSizes);

                    Log.d(TAG, "Back camera found: " + cameraId);
                    Log.d(TAG, "Selected resolution: " + bestSize.getWidth() + "x" + bestSize.getHeight());

                    return bestSize;
                }
            }

            Log.w(TAG, "Back camera not found, using default resolution");
            return new Size(1920, 1080);  // Default fallback

        } catch (CameraAccessException e) {
            Log.e(TAG, "Error accessing camera", e);
            return new Size(1920, 1080);  // Default fallback
        } catch (Exception e) {
            Log.e(TAG, "Unexpected error getting camera resolution", e);
            return new Size(1920, 1080);  // Default fallback
        }
    }

    private Size findBestCameraResolution(Size[] availableSizes) {
        // Sort by total pixels (descending)
        java.util.Arrays.sort(availableSizes, (s1, s2) -> {
            long pixels1 = (long) s1.getWidth() * s1.getHeight();
            long pixels2 = (long) s2.getWidth() * s2.getHeight();
            return Long.compare(pixels2, pixels1);
        });

        Log.d(TAG, "Available camera resolutions:");
        for (Size size : availableSizes) {
            Log.d(TAG, "  " + size.getWidth() + "x" + size.getHeight() +
                    " (" + String.format("%.1f", (size.getWidth() * size.getHeight() / 1_000_000.0)) + " MP)");
        }

        // Target Full HD (1920x1080 = 2.07 MP)
        final long TARGET_PIXELS = 1920L * 1080L;

        // Find closest to Full HD
        Size bestSize = availableSizes[0];
        long bestDiff = Long.MAX_VALUE;

        for (Size size : availableSizes) {
            long pixels = (long) size.getWidth() * size.getHeight();

            // Skip if too small (less than 0.5 MP)
            if (pixels < 500_000) {
                continue;
            }

            // Skip if too large (more than 8 MP - that's 4K territory)
            if (pixels > 8_000_000) {
                continue;
            }

            long diff = Math.abs(pixels - TARGET_PIXELS);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestSize = size;
            }

            // If we found exact Full HD, use it
            if (size.getWidth() == 1920 && size.getHeight() == 1080) {
                return size;
            }
        }

        return bestSize;
    }

    private PoolConfiguration calculateOptimalPoolSizes(
            MemoryInfo memory
    ) {
        // Calculate max instances that fit in memory
        long maxNanoInstances = memory.usableMemory / MODEL_NANO_SIZE;

        if (2 * MODEL_NANO_SIZE + MODEL_ACC_SIZE >= memory.usableMemory)
            return new PoolConfiguration(1, 1);

        // Adjust based on motion state
        int nanoInstances;
        int accInstances;

        // Conservative allocation
        nanoInstances = (int) Math.min(maxNanoInstances, 2);
        accInstances = (int) ((memory.usableMemory - nanoInstances * MODEL_NANO_SIZE) / MODEL_ACC_SIZE);

        return new PoolConfiguration(nanoInstances, accInstances);
    }

    private boolean validateConfiguration(PoolConfiguration config) {
        if (config.nanoInstances < 1 && config.accInstances < 1) {
            Log.e(TAG, "Cannot create any model instances!");
            return false;
        }

        long totalMemoryNeeded = (config.nanoInstances * MODEL_NANO_SIZE) +
                (config.accInstances * MODEL_ACC_SIZE);

        MemoryInfo memory = calculateAvailableMemory();

        if (totalMemoryNeeded > memory.usableMemory) {
            Log.w(TAG, "Configuration requires " + (totalMemoryNeeded / MB) + " MB but only " +
                    (memory.usableMemory / MB) + " MB available");
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
        Log.d(TAG, "  Total memory needed: " +
                ((config.nanoInstances * MODEL_NANO_SIZE + config.accInstances * MODEL_ACC_SIZE) / MB) + " MB");
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