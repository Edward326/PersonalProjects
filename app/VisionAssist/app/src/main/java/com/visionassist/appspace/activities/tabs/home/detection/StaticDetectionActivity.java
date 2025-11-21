package com.visionassist.appspace.activities.tabs.home.detection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import com.visionassist.appspace.R;
import com.visionassist.appspace.utils.Constants;

public class StaticDetectionActivity extends AppCompatActivity {
    private static final String TAG = "StaticDetectionActivity";

    private ImageView imageView;
    private Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Find ImageView
        imageView = findViewById(R.id.static_detection_image);

        // Load image from intent
        loadImageFromUri();
    }

    /**
     * Load image from Intent extras
     */
    private void loadImageFromUri() {
        try {
            String uriString = getIntent().getStringExtra(Constants.EXTRA_IMAGE_URI);

            if (uriString != null) {
                Uri imageUri = Uri.parse(uriString);

                // Load bitmap from URI
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);

                if (bitmap != null) {
                    // Display in ImageView if you have one
                    if (imageView != null) {
                        imageView.setImageBitmap(bitmap);
                    }

                    Log.d(TAG, "Image loaded from URI: " + bitmap.getWidth() + "x" + bitmap.getHeight());
                } else {
                    Log.e(TAG, "Failed to load bitmap from URI");
                }
            } else {
                Log.e(TAG, "No image URI in intent");
            }
        } catch (Exception e) {
            Log.e(TAG, "Error loading image from URI", e);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        // Recycle bitmap to free memory
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
            bitmap = null;
        }
    }
}