package com.visionassist.appspace.activities.tabs.home.caption;

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

public class CaptionActivity extends AppCompatActivity {
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

/*
// OLD VERSION
package com.visionassist.appspace.activities.tabs.home.caption;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import com.visionassist.appspace.R;
import com.visionassist.appspace.utils.Constants;
import java.util.Locale;

public class CaptionActivity extends AppCompatActivity implements TextToSpeech.OnInitListener {

    private static final String TAG = "CaptionActivity";

    private TextToSpeech tts;
    private TextView captionTextView;
    private Button speakButton;

    private String captionToSpeak;
    private boolean isTtsInitialized = false;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_caption);

        captionTextView = findViewById(R.id.caption_text_view);
        speakButton = findViewById(R.id.btn_speak);
        speakButton.setEnabled(false); // Disabled until TTS is ready

        captionToSpeak = getIntent().getStringExtra(Constants.EXTRA_CAPTION_TEXT);
        if (captionToSpeak == null || captionToSpeak.isEmpty()) {
            captionToSpeak = "No caption was generated.";
        }

        captionTextView.setText(captionToSpeak);
        captionTextView.setTextSize(Constants.CAPTION_TEXT_SIZE);

        // Initialize TextToSpeech engine
        tts = new TextToSpeech(this, this);

        speakButton.setOnClickListener(v -> speakCaption());
    }

    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            int result = tts.setLanguage(Locale.US);
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e(TAG, "TTS language is not supported.");
                Toast.makeText(this, "TTS language not supported.", Toast.LENGTH_SHORT).show();
            } else {
                isTtsInitialized = true;
                speakButton.setEnabled(true);
                tts.setSpeechRate(Constants.TTS_SPEECH_RATE);
                tts.setPitch(Constants.TTS_PITCH);
                // Automatically speak when the screen loads
                speakCaption();
            }
        } else {
            Log.e(TAG, "TTS initialization failed.");
            Toast.makeText(this, "TTS could not be initialized.", Toast.LENGTH_SHORT).show();
        }
    }

    private void speakCaption() {
        if (!isTtsInitialized) {
            Toast.makeText(this, "Text-to-Speech is not ready.", Toast.LENGTH_SHORT).show();
            return;
        }
        tts.speak(captionToSpeak, TextToSpeech.QUEUE_FLUSH, null, null);
    }


    @Override
    protected void onDestroy() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        super.onDestroy();
    }
}
 */