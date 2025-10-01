package com.visionassist.appspace;
import androidx.multidex.MultiDexApplication;
import android.util.Log;

public class VisionAssistApplication extends MultiDexApplication {
    private static final String TAG = "VisionAssistApp";

    @Override
    public void onCreate() {
        super.onCreate();
        Log.d(TAG, "VisionAssist Application started");
    }
}