package com.visionassist.appspace.activities.tabs.home;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import com.visionassist.appspace.R;
import com.visionassist.appspace.models.captioner.BLIPModel;
import com.visionassist.appspace.models.detector.YOLODetector;
import com.visionassist.appspace.models.detector.DetectionResult;
import com.visionassist.appspace.utils.Constants;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DetectionActivity extends AppCompatActivity {

    private static final String TAG = "DetectionActivity";

    private ImageView resultImageView;
    private ProgressBar progressBar;
    private Button generateCaptionButton;
    private Button viewCaptionButton;
    private TextView statusTextView;

    private YOLODetector yoloDetector;
    private BLIPModel blipCaptioner;

    private Bitmap originalBitmap;
    private DetectionResult lastDetectionResult;
    private String generatedCaption;

    private final ExecutorService executorService = Executors.newSingleThreadExecutor();
    private final Handler handler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detection);

        resultImageView = findViewById(R.id.result_image_view);
        progressBar = findViewById(R.id.progress_bar);
        statusTextView = findViewById(R.id.status_text_view);
        generateCaptionButton = findViewById(R.id.btn_generate_caption);
        viewCaptionButton = findViewById(R.id.btn_view_caption);

        // Hide buttons until processing is done
        generateCaptionButton.setVisibility(View.GONE);
        viewCaptionButton.setVisibility(View.GONE);

        String imagePath = getIntent().getStringExtra(Constants.EXTRA_IMAGE_PATH);
        if (imagePath == null) {
            Toast.makeText(this, "Error: No image path provided.", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        Uri imageUri = Uri.parse(imagePath);
        loadAndProcessImage(imageUri);

        generateCaptionButton.setOnClickListener(v -> runCaptioning());
        viewCaptionButton.setOnClickListener(v -> {
            Intent intent = new Intent(DetectionActivity.this, CaptionActivity.class);
            intent.putExtra(Constants.EXTRA_CAPTION_TEXT, generatedCaption);
            startActivity(intent);
        });
    }

    private void loadAndProcessImage(Uri imageUri) {
        showLoading("Loading models...");
        Log.d(TAG, "Loading models");

        executorService.execute(() -> {
            try {
                // Load models in the background
                yoloDetector = new YOLODetector(this);
                blipCaptioner = new BLIPModel(this);
                Log.d(TAG, "Models loaded successfully.");

                Log.d(TAG, "Bitmap conversion started.");
                // Load image
                InputStream inputStream = getContentResolver().openInputStream(imageUri);
                originalBitmap = BitmapFactory.decodeStream(inputStream);
                Log.d(TAG, "Bitmap conversion completed.");

                handler.post(() -> {
                    resultImageView.setImageBitmap(originalBitmap);
                    runDetection();
                });
            } catch (Exception e) {
                Log.e(TAG, "Error loading image or models", e);
                handler.post(() -> showError("Failed to load image or models."));
            }
        });
    }

    private void runDetection() {
        showLoading("Detecting objects...");
        executorService.execute(() -> {
            lastDetectionResult = yoloDetector.detectObjects(originalBitmap);
            Bitmap bitmapWithDetections = yoloDetector.drawDetections(originalBitmap, lastDetectionResult);

            handler.post(() -> {
                hideLoading();
                resultImageView.setImageBitmap(bitmapWithDetections);
                if (lastDetectionResult.hasDetections()) {
                    generateCaptionButton.setVisibility(View.VISIBLE);
                } else {
                    statusTextView.setText("No objects detected.");
                    statusTextView.setVisibility(View.VISIBLE);
                }
            });
        });
    }

    private void runCaptioning() {
        if (lastDetectionResult == null || !lastDetectionResult.hasDetections()) {
            Toast.makeText(this, "No objects were detected to generate a caption for.", Toast.LENGTH_SHORT).show();
            return;
        }

        showLoading("Generating caption...");
        generateCaptionButton.setVisibility(View.GONE);

        executorService.execute(() -> {
            generatedCaption = blipCaptioner.generateCaption(originalBitmap, lastDetectionResult.getLabels());

            handler.post(() -> {
                hideLoading();
                if (generatedCaption != null && !generatedCaption.startsWith("Error:")) {
                    viewCaptionButton.setVisibility(View.VISIBLE);
                } else {
                    showError(generatedCaption);
                }
            });
        });
    }

    private void showLoading(String message) {
        progressBar.setVisibility(View.VISIBLE);
        statusTextView.setText(message);
        statusTextView.setVisibility(View.VISIBLE);
    }

    private void hideLoading() {
        progressBar.setVisibility(View.GONE);
        statusTextView.setVisibility(View.GONE);
    }

    private void showError(String message) {
        hideLoading();
        statusTextView.setText(message);
        statusTextView.setVisibility(View.VISIBLE);
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executorService.shutdown();
        if (blipCaptioner != null) {
            blipCaptioner.close();
        }
    }
}