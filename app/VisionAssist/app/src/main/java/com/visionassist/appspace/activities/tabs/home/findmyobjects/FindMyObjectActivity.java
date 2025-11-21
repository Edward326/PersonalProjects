package com.visionassist.appspace.activities.tabs.home.findmyobjects;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.compose.ui.platform.ComposeView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.R;
import com.visionassist.appspace.jetpack.design.FindMyObjectUIKt;
import com.visionassist.appspace.activities.main.HomeActivity;
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsManager;
import com.visionassist.appspace.models.classifier.YOLOClassifier;
import com.visionassist.appspace.models.detector.DetectionResult;
import com.visionassist.appspace.models.detector.YOLODetector;
import com.visionassist.appspace.sound.SoundConstants;
import com.visionassist.appspace.sound.SoundManager;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.BackgroundTaskExecutor;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.UtilsKt;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import kotlin.Unit;
import kotlin.jvm.functions.Function0;

public class FindMyObjectActivity extends AppCompatActivity {
    private static final String TAG = "FindMyObjectActivity";
    private static final int CAMERA_PERMISSION_REQUEST = 100;

    // UI Components
    private PreviewView cameraPreview;
    private ImageView resultImageView;
    private ComposeView sliderCompose;
    private ComposeView navigationCompose;

    // Models
    private YOLODetector detector;
    private YOLOClassifier classifier;

    // Managers
    private PhoneStatusMonitor monitor = PhoneStatusMonitor.getInstance();
    private SoundManager soundManager = monitor.getSoundManager();
    private BackgroundTaskExecutor backgroundExecutor = BackgroundTaskExecutor.getInstance();

    // Detection data
    private Map<Integer, String> objectsToFind; // classIndex -> synonym
    private List<Integer> remainingClassIndices;

    // Camera
    private ProcessCameraProvider cameraProvider;
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;

    // Detection state
    private AtomicBoolean detectionInProgress = new AtomicBoolean(false);
    private AtomicBoolean stopDetection = new AtomicBoolean(false);
    private AtomicBoolean resultsReady = new AtomicBoolean(false);
    private AtomicBoolean writeComplete = new AtomicBoolean(false);
    private AtomicInteger frameDelayMs = new AtomicInteger(1000 / Constants.DEFAULT_FPS);

    // Results
    private volatile Bitmap resultBitmap;

    // Handlers
    private Handler mainHandler = new Handler(Looper.getMainLooper());
    private Handler detectionHandler = new Handler(Looper.getMainLooper());

    // Thread synchronization
    private AtomicBoolean classifierDone = new AtomicBoolean(false);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_find_my_object);

        initializeViews();
        extractIntentData();
        initializeModels();

        if (checkCameraPermission()) {
            startDetectionProcess();
        } else {
            requestCameraPermission();
        }
    }

    private void initializeViews() {
        cameraPreview = findViewById(R.id.camera_preview);
        resultImageView = findViewById(R.id.result_image);
        sliderCompose = findViewById(R.id.slider_compose);
        navigationCompose = findViewById(R.id.navigation_compose);

        // Hide result initially
        resultImageView.setVisibility(View.GONE);
        navigationCompose.setVisibility(View.GONE);

        cameraExecutor = Executors.newSingleThreadExecutor();
    }

    private void extractIntentData() {
        Intent intent = getIntent();
        int[] originalClassIndices = intent.getIntArrayExtra(Constants.EXTRA_MATCHED_INDICES);
        String[] originalMatchedWords = intent.getStringArrayExtra(Constants.EXTRA_SYNONYMS_WORDS);

        if (originalClassIndices == null || originalMatchedWords == null) {
            Log.e(TAG, "Missing intent data");
            finish();
            return;
        }

        // Build objectsToFind map
        objectsToFind = new HashMap<>();
        remainingClassIndices = new ArrayList<>();

        for (int i = 0; i < originalClassIndices.length; i++) {
            int classIdx = originalClassIndices[i];
            String synonym = originalMatchedWords[i];
            objectsToFind.put(classIdx, synonym);
            remainingClassIndices.add(classIdx);
        }

        Log.d(TAG, "Objects to find: " + objectsToFind);
        Log.d(TAG, "Remaining classes: " + remainingClassIndices);
    }

    private void initializeModels() {
        detector = monitor.getModelManager().getDetector();
        classifier = monitor.getModelManager().getClassifier();

        if (detector == null) {
            Log.e(TAG, "Detector not loaded");
            finish();
        }
    }

    private boolean checkCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA},
                CAMERA_PERMISSION_REQUEST);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startDetectionProcess();
            } else {
                Log.e(TAG, "Camera permission denied");
                finish();
            }
        }
    }

    private void startDetectionProcess() {
        startCameraX();
        setupSliderUI();
        monitor.startMonitoring();
    }

    @SuppressLint("SetTextI18n")
    private void setupSliderUI() {
        FindMyObjectUIKt.setupFPSSlider(
                sliderCompose,
                Constants.DEFAULT_FPS,
                fps -> {
                    int delayMs = 1000 / fps;
                    frameDelayMs.set(delayMs);
                    Log.d(TAG, "FPS updated to " + fps + " (delay: " + delayMs + "ms)");
                    return null;
                }
        );
    }

    private void startCameraX() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
                startDetectionLoop();
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(cameraPreview.getSurfaceProvider());

        imageCapture = new ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build();

        try {
            cameraProvider.unbindAll();
            Log.d(TAG, "Camera bound successfully");
        } catch (Exception e) {
            Log.e(TAG, "Error binding camera", e);
        }
    }

    private void startDetectionLoop() {
        stopDetection.set(false);
        detectionHandler.post(new Runnable() {
            @Override
            public void run() {
                if (stopDetection.get() || remainingClassIndices.isEmpty()) {
                    Log.d(TAG, "Detection loop stopped");
                    return;
                }

                if (!detectionInProgress.get()) {
                    captureAndDetect();
                }

                // Schedule next detection based on FPS slider
                detectionHandler.postDelayed(this, frameDelayMs.get());
            }
        });
    }

    private void captureAndDetect() {
        if (imageCapture == null) return;

        detectionInProgress.set(true);

        imageCapture.takePicture(cameraExecutor, new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                Bitmap bitmap = imageProxyToBitmap(image);
                image.close();

                if (bitmap != null) {
                    processDetection(bitmap);
                } else {
                    detectionInProgress.set(false);
                }
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e(TAG, "Image capture failed", exception);
                detectionInProgress.set(false);
            }
        });
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
    }

    private void processDetection(Bitmap bitmap) {
        // Run detection in background
        backgroundExecutor.executeAsync(
                () -> detector.detectObjects(bitmap),
                new BackgroundTaskExecutor.TaskCallback<>() {
                    @Override
                    public void onSuccess(DetectionResult detectionResult) {
                        handleDetectionResult(bitmap, detectionResult);
                    }

                    @Override
                    public void onError(Exception e) {
                        Log.e(TAG, "Detection error", e);
                        detectionInProgress.set(false);
                    }
                }
        );
    }

    private void handleDetectionResult(Bitmap bitmap, DetectionResult detectionResult) {
        if (resultsReady.get()) {
            // Another thread already found objects
            detectionInProgress.set(false);
            return;
        }

        List<Integer> detectedClasses = detectionResult.getClassIndices();
        List<Integer> foundClasses = new ArrayList<>();

        // Check if any target classes were detected
        for (Integer classIdx : remainingClassIndices) {
            if (detectedClasses.contains(classIdx)) {
                foundClasses.add(classIdx);
            }
        }

        if (foundClasses.isEmpty()) {
            // No target objects found
            detectionInProgress.set(false);
            return;
        }

        List<RectF> boundingBoxes = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        List<Integer> class_indices = new ArrayList<>();

        for (int i = 0; i < detectionResult.getDetectionCount(); i++) {
            if (foundClasses.contains(detectionResult.getClassIndices().get(i))){
                boundingBoxes.add(detectionResult.getBoundingBoxes().get(i));
                confidences.add(detectionResult.getConfidences().get(i));
                labels.add(detectionResult.getLabels().get(i));
                class_indices.add(detectionResult.getClassIndices().get(i));
            }
        }

        DetectionResult detectionResultFiltered=new DetectionResult(boundingBoxes,confidences,labels,class_indices);

        // Target objects found! Signal stop and process
        if (resultsReady.compareAndSet(false, true)) {
            stopDetection.set(true);

            if (AppConfig.env_reports && classifier != null) {
                // Run classifier
                classifyScene(bitmap, detectionResultFiltered, foundClasses);
            } else {
                // Skip classifier
                processFoundObjects(bitmap, detectionResultFiltered, foundClasses, -1);
            }
        }
    }

    private void classifyScene(Bitmap bitmap, DetectionResult detectionResult, List<Integer> foundClasses) {
        backgroundExecutor.executeAsync(
                () -> classifier.detectScene(bitmap),
                new BackgroundTaskExecutor.TaskCallback<>() {
                    @Override
                    public void onSuccess(Integer sceneId) {
                        // Classifier done, continue processing
                        processFoundObjects(bitmap, detectionResult, foundClasses, sceneId);
                    }

                    @Override
                    public void onError(Exception e) {
                        Log.e(TAG, "Classification error", e);
                        // Continue without scene name
                        try {
                            processFoundObjects(bitmap, detectionResult, foundClasses, -1);
                        } catch (Exception ex) {
                            Log.e(TAG, "Error in processFoundObjects", ex);
                        }
                    }
                }
        );
    }

    private void processFoundObjects(Bitmap originalBitmap, DetectionResult detectionResult, List<Integer> foundClasses, int sceneName) {
        if (writeComplete.get()) return;
        writeComplete.set(true);
        monitor.stopMonitoring();

        // Build list of found synonyms
        List<String> foundSynonyms = new ArrayList<>();
        for (Integer classIdx : foundClasses) {
            foundSynonyms.add(objectsToFind.get(classIdx));
            remainingClassIndices.remove(classIdx);
        }

        // Draw bounding boxes on original bitmap
        // YOLODetector.drawDetections needs the ORIGINAL bitmap and DetectionResult
        resultBitmap = detector.drawDetections(originalBitmap, detectionResult);


        // Write environment report
        EnvironmentReportsManager.writeDetectionReport(this, sceneName, foundSynonyms);

        // Signal complete
        writeComplete.set(true);
        // Show results on UI thread
        mainHandler.post(this::showResults);
    }

    private void showResults() {
        cameraPreview.setVisibility(View.GONE);
        sliderCompose.setVisibility(View.GONE);

        resultImageView.setVisibility(View.VISIBLE);
        resultImageView.setImageBitmap(resultBitmap);

        setupNavigationUI();

        if (AppConfig.haptics) {
            UtilsKt.vibrate(UtilsKt.haptic_model0());
        }

        soundManager.play(SoundConstants.FIND_MY_OBJECT_DONE_ID, 0.7f, 0.7f, null);
    }

    private void setupNavigationUI() {
        boolean hasMore = !remainingClassIndices.isEmpty();

        navigationCompose.setVisibility(View.VISIBLE);
        FindMyObjectUIKt.setupNavigationUI(
                navigationCompose,
                hasMore,
                handleBackClick(),
                hasMore ? handleNextClick() : null
        );
    }

    private @NotNull Function0<@NotNull Unit> handleBackClick() {
        finish();
        Intent intent = new Intent(this, HomeActivity.class);
        startActivity(intent);
        return null;
    }

    private @Nullable Function0<@NotNull Unit> handleNextClick() {
        if (!remainingClassIndices.isEmpty()) {
            // Reset and restart
            resetForNext();
            startDetectionProcess();
        }
        return null;
    }

    private void resetForNext() {
        resultImageView.setVisibility(View.GONE);
        navigationCompose.setVisibility(View.GONE);
        cameraPreview.setVisibility(View.VISIBLE);
        sliderCompose.setVisibility(View.VISIBLE);

        resultsReady.set(false);
        writeComplete.set(false);
        detectionInProgress.set(false);
        classifierDone.set(false);
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        return switch (keyCode) {
            case KeyEvent.KEYCODE_VOLUME_DOWN -> {
                handleBackClick();
                yield true;
            }
            case KeyEvent.KEYCODE_VOLUME_UP -> {
                if (!remainingClassIndices.isEmpty()) {
                    handleNextClick();
                }
                yield true;
            }
            default -> super.onKeyDown(keyCode, event);
        };
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopDetection.set(true);
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        cameraExecutor.shutdown();
    }
}