package com.visionassist.appspace.activities.tabs.home.detection;

import android.os.Bundle;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;
import com.visionassist.appspace.R;

public class LiveDetectionActivity extends AppCompatActivity {

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        Log.i("LiveDetectionActivity","Activity started, onCreate finished");
    }
}