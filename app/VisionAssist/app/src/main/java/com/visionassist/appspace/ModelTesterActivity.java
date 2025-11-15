package com.visionassist.appspace;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import com.visionassist.appspace.models.classifier.YOLOClassifier;
import com.visionassist.appspace.models.sttengine.SpeechRecognizer;
import com.visionassist.appspace.models.translator.CaptionTranslator;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class ModelTesterActivity extends AppCompatActivity {

    private static final String TAG = "ModelTester";
    private static final int PERMISSION_REQUEST_CODE = 100;

    // Models
    private YOLOClassifier classifier;
    private CaptionTranslator translator;
    private SpeechRecognizer speechRecognizer;

    // UI Elements
    private TextView tvStatus;
    private Button btnCapture;
    private Button btnListen;
    private Button btnStopListen;

    // Handler for checking model readiness
    private Handler mainHandler;

    // Camera
    private ActivityResultLauncher<Uri> takePictureLauncher;
    private Uri currentPhotoUri;

    // Flags
    private boolean allModelsReady = false;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_model_tester);

        // Initialize UI
        tvStatus = findViewById(R.id.tv_status);
        btnCapture = findViewById(R.id.btn_capture);
        btnListen = findViewById(R.id.btn_listen);
        btnStopListen = findViewById(R.id.btn_stop_listen);

        // Disable buttons initially
        btnCapture.setEnabled(false);
        btnListen.setEnabled(false);
        btnStopListen.setEnabled(false);

        // Initialize handler
        mainHandler = new Handler(Looper.getMainLooper());

        // Setup camera launcher
        setupCameraLauncher();

        // Setup button listeners
        btnCapture.setOnClickListener(v -> captureImage());
        btnListen.setOnClickListener(v -> startListening());
        btnStopListen.setOnClickListener(v -> stopListening());

        // Check permissions and initialize
        checkPermissionsAndInitialize();
    }

    private void setupCameraLauncher() {
        takePictureLauncher = registerForActivityResult(
                new ActivityResultContracts.TakePicture(),
                isSuccess -> {
                    if (isSuccess && currentPhotoUri != null) {
                        try {
                            Bitmap bitmap = MediaStore.Images.Media.getBitmap(
                                    getContentResolver(), currentPhotoUri);
                            testClassifier(bitmap);
                        } catch (IOException e) {
                            Log.e(TAG, "Error loading captured image", e);
                            updateStatus("❌ Failed to load image");
                        }
                    } else {
                        updateStatus("❌ Image capture failed");
                    }
                });
    }

    private void checkPermissionsAndInitialize() {
        // Check required permissions
        String[] permissions = {
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
        };

        boolean allGranted = true;
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission)
                    != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                break;
            }
        }

        if (allGranted) {
            initializeModels();
        } else {
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == PERMISSION_REQUEST_CODE) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }

            if (allGranted) {
                initializeModels();
            } else {
                updateStatus("❌ Permissions denied. Cannot proceed.");
                Toast.makeText(this, "All permissions required for testing",
                        Toast.LENGTH_LONG).show();
            }
        }
    }

    private void initializeModels() {
        updateStatus("🔄 Initializing models...");
        Log.d(TAG, "Starting model initialization");

        new Thread(() -> {
            // Step 1: Initialize YOLOClassifier
            updateStatus("🔄 Loading YOLO Classifier...");
            classifier = new YOLOClassifier(this);
            int classifierStatus = classifier.loadModel();

            if (classifierStatus != 0) {
                Log.e(TAG, "❌ Classifier failed to load");
                updateStatus("❌ Classifier initialization failed");
                return;
            }

            Log.d(TAG, "✅ Classifier loaded successfully");
            updateStatus("✅ Classifier loaded\n🔄 Loading Translator...");

            // Step 2: Initialize CaptionTranslator
            translator = new CaptionTranslator(this);
            int translatorStatus = translator.initializeTranslator();

            if (translatorStatus != 0) {
                Log.e(TAG, "❌ Translator failed to load");
                updateStatus("❌ Translator initialization failed");
                return;
            }

            Log.d(TAG, "✅ Translator loaded successfully");
            updateStatus("✅ Translator loaded\n🔄 Loading Speech Recognizer...");

            // Step 3: Initialize SpeechRecognizer
            speechRecognizer = new SpeechRecognizer(this);
            int speechStatus = speechRecognizer.loadModel();

            if (speechStatus != 0) {
                Log.e(TAG, "❌ Speech recognizer failed to load");
                updateStatus("❌ Speech Recognizer initialization failed");
                return;
            }

            Log.d(TAG, "✅ Speech recognizer started loading");
            updateStatus("✅ Speech files loaded\n🔄 Waiting for Vosk model...");

            // Step 4: Wait for Vosk model to be ready (async unpacking)
            checkVoskModelReady();

        }).start();
    }

    private void checkVoskModelReady() {
        mainHandler.post(new Runnable() {
            @Override
            public void run() {
                if (speechRecognizer.isReady) {
                    // Check if model is actually loaded
                    if (speechRecognizer.getModel() != null) {
                        Log.d(TAG, "✅ Vosk model ready!");
                        onAllModelsReady();
                    } else {
                        Log.e(TAG, "❌ Vosk model is null despite isReady flag");
                        updateStatus("❌ Speech model failed to load");
                    }
                } else {
                    // Not ready yet, check again in 1 second
                    Log.d(TAG, "⏳ Waiting for Vosk model...");
                    mainHandler.postDelayed(this, 1000);
                }
            }
        });
    }

    private void onAllModelsReady() {
        allModelsReady = true;

        Log.d(TAG, "🎉 All models ready!");
        updateStatus("✅ All models ready!\n\nTap 'Capture' to test classifier\nTap 'Listen' to test speech");

        // Enable testing buttons
        runOnUiThread(() -> {
            btnCapture.setEnabled(true);
            btnListen.setEnabled(true);
            Toast.makeText(this, "All models ready! You can start testing.",
                    Toast.LENGTH_LONG).show();
        });
    }

    private void captureImage() {
        if (!allModelsReady) {
            Toast.makeText(this, "Models not ready yet", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            // Create temporary file
            File photoFile = File.createTempFile("test_image", ".jpg", getCacheDir());
            currentPhotoUri = FileProvider.getUriForFile(
                    this,
                    getPackageName() + ".provider",
                    photoFile
            );

            // Launch camera
            takePictureLauncher.launch(currentPhotoUri);
            updateStatus("📸 Capturing image...");

        } catch (IOException e) {
            Log.e(TAG, "Error creating image file", e);
            updateStatus("❌ Failed to create image file");
        }
    }

    private void testClassifier(Bitmap bitmap) {
        updateStatus("🔄 Testing Classifier...");
        Log.d(TAG, "=== CLASSIFIER TEST START ===");

        new Thread(() -> {
            // Test classifier
            String scene = classifier.detectScene(bitmap);

            Log.d(TAG, "🎯 Classifier Result: " + scene);
            Log.d(TAG, "=== CLASSIFIER TEST END ===");

            updateStatus("✅ Classifier Test Complete!\n\nDetected Scene: " + scene);

        }).start();
    }

    private void startListening() {
        if (!allModelsReady) {
            Toast.makeText(this, "Models not ready yet", Toast.LENGTH_SHORT).show();
            return;
        }

        updateStatus("🎤 Listening... Speak now!");
        btnListen.setEnabled(false);
        btnStopListen.setEnabled(true);

        Log.d(TAG, "=== SPEECH RECOGNITION TEST START ===");

        speechRecognizer.startListening(new SpeechRecognizer.RecognitionCallback() {
            @Override
            public void onResult(String recognizedText) {
                Log.d(TAG, "🎤 Recognized Text: \"" + recognizedText + "\"");

                // Process recognized text to get object indices
                List<Integer> objectIndices = speechRecognizer.processRecognizedText(recognizedText);

                Log.d(TAG, "📋 Matched Object Indices: " + objectIndices);

                // Test translator with recognized text
                testTranslator(recognizedText);

                runOnUiThread(() -> {
                    btnListen.setEnabled(true);
                    btnStopListen.setEnabled(false);
                });
            }

            @Override
            public void onError(String error) {
                Log.e(TAG, "❌ Speech Recognition Error: " + error);
                Log.d(TAG, "=== SPEECH RECOGNITION TEST END (ERROR) ===");

                updateStatus("❌ Speech Recognition Error:\n" + error);

                runOnUiThread(() -> {
                    btnListen.setEnabled(true);
                    btnStopListen.setEnabled(false);
                });
            }
        });
    }

    private void stopListening() {
        speechRecognizer.stopListening();
        updateStatus("🛑 Stopped listening");
        btnListen.setEnabled(true);
        btnStopListen.setEnabled(false);

        Log.d(TAG, "=== SPEECH RECOGNITION TEST END (MANUAL STOP) ===");
    }

    private void testTranslator(String englishText) {
        Log.d(TAG, "=== TRANSLATOR TEST START ===");
        Log.d(TAG, "📝 Input Text: \"" + englishText + "\"");

        new Thread(() -> {
            String romanianText = translator.translate(englishText);

            if (romanianText != null) {
                Log.d(TAG, "✅ Translation Successful!");

                updateStatus("Complete Test Results:\n\n" +
                        "Recognized: \"" + englishText + "\"\n\n" +
                        "Translated: \"" + romanianText + "\"");
            } else {
                Log.e(TAG, "❌ Translation Failed");
                updateStatus("✅ Speech recognized but translation failed");
            }

            Log.d(TAG, "=== TRANSLATOR TEST END ===");
        }).start();
    }

    private void updateStatus(String message) {
        runOnUiThread(() -> {
            tvStatus.setText(message);
            Log.d(TAG, "Status: " + message.replace("\n", " | "));
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        // Cleanup all models
        if (classifier != null) {
            classifier.close();
        }
        if (translator != null) {
            translator.close();
        }
        if (speechRecognizer != null) {
            speechRecognizer.close();
        }

        Log.d(TAG, "All models cleaned up");
    }
}